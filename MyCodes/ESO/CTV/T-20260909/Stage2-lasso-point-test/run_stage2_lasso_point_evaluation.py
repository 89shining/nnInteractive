#!/usr/bin/env python
"""Deterministic external Stage-2 lasso-plus-point evaluation for nnInteractive."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from scipy import ndimage

STAGE2_DIR = Path(__file__).resolve().parents[1] / "Stage2-point"
if str(STAGE2_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE2_DIR))

from dataset import load_case_with_spacing, patient_dirs, patient_id
from interaction_builder import joint_axial_lasso
from losses import unprompted_hard_dice, whole_volume_hard_dice
from point_clicker import next_error_click
from prediction_wrapper import JointLassoSession

PLAN = Path("/home/wusi/nnInteractive/MyResults/Eso/20260909_CTV/Stage1-lasso/TestResults/external_test_prompt_plans/stage1_prompt_plan.json")
MODEL = Path("/home/wusi/nnInteractive/nnInteractive_v1.0")
DATA = Path("/home/wusi/SAM2/MyTrain/SAM2data/Eso/20260909_CTV/PreprocessDataNii/test")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data-root", type=Path, default=DATA)
    p.add_argument("--plan", type=Path, default=PLAN)
    p.add_argument("--model-dir", type=Path, default=MODEL)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--save-predictions", action="store_true")
    return p.parse_args()


def load_plan(path: Path):
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != 2 or document.get("plan_type") != "deterministic":
        raise ValueError("Stage2 main test requires a schema-v2 deterministic frozen mask-prompt plan")
    return document["patients"]


def describe(values):
    values = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    if not len(values):
        return {"n_valid": 0, "mean": None, "std": None, "median": None, "q25": None, "q75": None}
    return {"n_valid": int(len(values)), "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "median": float(np.median(values)), "q25": float(np.quantile(values, .25)),
            "q75": float(np.quantile(values, .75))}


def hd95_endpoints(pred, gt, reference):
    if not pred.any(): return None, None, None
    pred_surface = pred ^ ndimage.binary_erosion(pred)
    gt_surface = gt ^ ndimage.binary_erosion(gt)
    spacing = tuple(reversed(reference.GetSpacing()))
    distances = np.concatenate((ndimage.distance_transform_edt(~gt_surface, sampling=spacing)[pred_surface],
                                ndimage.distance_transform_edt(~pred_surface, sampling=spacing)[gt_surface]))
    origin, step = reference.GetOrigin()[2], reference.GetDirection()[8] * reference.GetSpacing()[2]
    def ends(mask):
        coordinates = origin + step * np.where(mask)[0]
        return float(coordinates.max()), float(coordinates.min())
    pred_sup, pred_inf = ends(pred); gt_sup, gt_inf = ends(gt)
    return float(np.percentile(distances, 95)), abs(pred_sup - gt_sup), abs(pred_inf - gt_inf)


def save_prediction(prediction, reference, path):
    image = sitk.GetImageFromArray(prediction.astype(np.uint8)); image.CopyInformation(reference)
    path.parent.mkdir(parents=True, exist_ok=True); sitk.WriteImage(image, str(path), True)


def main():
    args = parse_args(); device = torch.device(args.device)
    if args.fold != 0: raise ValueError("Only the available fold0 checkpoint may be evaluated")
    if not args.checkpoint.is_file(): raise FileNotFoundError(args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if "model_state_dict" not in checkpoint: raise ValueError("Checkpoint is not an nnInteractive Stage2 checkpoint")
    config = checkpoint.get("config", {})
    if isinstance(config, dict) and "fold" in config and config["fold"] is not None and int(config["fold"]) != args.fold:
        raise RuntimeError("Checkpoint fold mismatch")
    plans, cases = load_plan(args.plan), {patient_id(case): case for case in patient_dirs(args.data_root)}
    if set(cases) != set(map(int, plans)): raise RuntimeError("External-test data and frozen prompt-plan patient IDs differ")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    session = JointLassoSession(device=device, do_autozoom=True, verbose=False, use_pinned_memory=True)
    session.initialize_from_trained_model_folder(str(args.model_dir), use_fold=0)
    session.network.load_state_dict(checkpoint["model_state_dict"], strict=True); session.network.eval()
    (args.output_dir / "run_manifest.json").write_text(json.dumps({
        "evaluation_scope": "external_test_stage2_lasso_point", "fold": args.fold, "patients": len(cases),
        "checkpoint": str(args.checkpoint), "stage1_checkpoint": checkpoint.get("stage1_checkpoint"), "plan": str(args.plan),
        "lasso_prompt": "frozen deterministic SI-stratified K1..K5",
        "point_prompt": "native residual correction; largest 26-connected FN/FP; deterministic spacing-aware EDT maximum",
        "point_budgets": [0, 1, 2, 3, 4, 5], "initial_lasso_slices_excluded_from_points": True,
    }, indent=2), encoding="utf-8")
    rows, click_rows = [], []
    with torch.inference_mode():
        for pid, case in sorted(cases.items()):
            image, target, spacing = load_case_with_spacing(case); target = target.astype(bool)
            reference = sitk.ReadImage(str(case / "CTV.nii.gz")); target_tensor = torch.from_numpy(target).to(device)
            for key, record in sorted(plans[str(pid)]["plans"].items(), key=lambda item: int(item[0][1:])):
                prompts = [int(z) for z in record["slices"]]
                if not record["feasible"] or any(not target[z].any() for z in prompts) or any(b - a < 2 for a, b in zip(prompts, prompts[1:])):
                    raise RuntimeError(f"Invalid frozen mask prompts p_{pid} {key}")
                session.set_joint_lassos(image, joint_axial_lasso(target, prompts)); session.begin_native_trajectory()
                prediction = session.native_predict_queued(); effective_t = 0
                for requested_t in range(6):
                    pred = prediction.detach().cpu().numpy().astype(bool); empty = not pred.any()
                    hd95, sup, inf = hd95_endpoints(pred, target, reference) if not empty else (None, None, None)
                    rows.append({"patient_id": pid, "fold": args.fold, "plan_key": key, "K": len(prompts),
                                 "requested_T": requested_t, "effective_T": effective_t, "prompt_slices": json.dumps(prompts),
                                 "whole_volume_dice": whole_volume_hard_dice(prediction.to(device), target_tensor),
                                 # Native queued inference keeps its hard prediction on
                                 # CPU; align it with the GPU target before boolean
                                 # indexing inside the shared metric helper.
                                 "unprompted_dice": unprompted_hard_dice(prediction.to(device), target_tensor, prompts),
                                 "hd95_3d_mm": hd95, "prediction_empty": int(empty), "sup_endpoint_mae_mm": sup, "inf_endpoint_mae_mm": inf})
                    if args.save_predictions: save_prediction(pred, reference, args.output_dir / "predictions" / key / f"T{requested_t}" / f"p_{pid:03d}.nii.gz")
                    print(f"[{key} T{requested_t}] p_{pid:03d} Dice={rows[-1]['whole_volume_dice']:.4f}", flush=True)
                    if requested_t == 5: continue
                    click = next_error_click(pred, target, prompts, spacing, rng=None)
                    if click is None:
                        click_rows.append({"patient_id": pid, "plan_key": key, "K": len(prompts), "round": requested_t + 1, "applied": 0, "reason": "exact_prediction"})
                        continue
                    session.add_point_interaction(click.xyz, include_interaction=click.positive, run_prediction=False)
                    prediction = session.native_predict_queued(); effective_t += 1
                    click_rows.append({"patient_id": pid, "plan_key": key, "K": len(prompts), "round": requested_t + 1, "applied": 1,
                                       "z": click.zyx[0], "y": click.zyx[1], "x": click.zyx[2], "positive": int(click.positive)})
    with (args.output_dir / "per_case_metrics.csv").open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    click_fields = sorted({field for row in click_rows for field in row})
    with (args.output_dir / "point_trace.csv").open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=click_fields); w.writeheader(); w.writerows(click_rows)
    metrics = ["whole_volume_dice", "hd95_3d_mm", "sup_endpoint_mae_mm", "inf_endpoint_mae_mm", "unprompted_dice"]
    summary = {"evaluation_scope": "external_test_stage2_lasso_point", "checkpoint": str(args.checkpoint),
               "stage1_checkpoint": checkpoint.get("stage1_checkpoint"), "plan": str(args.plan), "n": len(rows), "by_K_T": {}}
    for k in range(1, 6):
        for t in range(6):
            chosen = [row for row in rows if row["K"] == k and row["requested_T"] == t]
            summary["by_K_T"][f"K{k}_T{t}"] = {"n": len(chosen), "n_empty_prediction": sum(row["prediction_empty"] for row in chosen),
                **{metric: describe([row[metric] for row in chosen]) for metric in metrics}}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    session.executor.shutdown(wait=False, cancel_futures=True)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
