#!/usr/bin/env python
"""Immutable-plan Stage-1 nnInteractive test inference.

This driver deliberately contains no Stage-2 correction or validation-plan
sampling. Each K is evaluated with a newly initialized native inference
session, and every prompted slice comes solely from the supplied JSON plan.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from scipy import ndimage

STAGE1_DIR = Path(__file__).resolve().parents[1] / 'Stage1-lasso'
if str(STAGE1_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE1_DIR))

from dataset import load_case, patient_dirs, patient_id  # noqa: E402
from interaction_builder import joint_axial_lasso  # noqa: E402
from losses import unprompted_hard_dice, whole_volume_hard_dice  # noqa: E402
from prediction_wrapper import JointLassoSession  # noqa: E402

DATA = Path('/home/wusi/SAM2/MyTrain/SAM2data/Eso/20260909_CTV/PreprocessDataNii/train')
SPLITS = Path('/home/wusi/SAM2/MyTrain/MyCodes/ESO/CTV/T-20260901/shared_splits.json')
MODEL = Path('/home/wusi/nnInteractive/nnInteractive_v1.0')
TRAIN_RESULTS = Path('/home/wusi/nnInteractive/MyResults/Eso/20260909_CTV/Stage1-lasso/TrainResults')
TEST_RESULTS = TRAIN_RESULTS.parent / 'TestResults'
PROMPT_PLANS = TEST_RESULTS / 'prompt_plans' / 'stage1_prompt_plan.json'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, default=DATA)
    parser.add_argument('--split-path', type=Path, default=SPLITS)
    parser.add_argument('--model-dir', type=Path, default=MODEL)
    parser.add_argument('--train-results', type=Path, default=TRAIN_RESULTS)
    parser.add_argument('--plan', type=Path, default=PROMPT_PLANS)
    parser.add_argument('--output-dir', type=Path, default=None)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--external-test', action='store_true',
                        help='Evaluate every case in --data-root with the requested checkpoint; do not apply a training split.')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--save-predictions', action='store_true')
    return parser.parse_args()


def load_fold_by_patient(split_path: Path) -> dict[int, int]:
    folds = json.loads(split_path.read_text(encoding='utf-8'))['folds']
    result: dict[int, int] = {}
    for item in folds:
        fold = int(item['fold'])
        for name in item['val']:
            pid = patient_id(Path(name))
            if pid in result:
                raise RuntimeError(f'Patient {pid} appears in validation sets of multiple folds')
            result[pid] = fold
    return result


def load_plan(path: Path) -> tuple[dict[str, dict], bool]:
    document = json.loads(path.read_text(encoding='utf-8'))
    if document.get('schema_version') != 2 or document.get('plan_type') not in {'deterministic', 'random_robustness'}:
        raise ValueError('Plan must be a schema-v2 immutable plan document with explicit plan_type')
    plans = document.get('patients')
    if not isinstance(plans, dict) or not plans:
        raise ValueError('Plan document must contain a nonempty patients mapping')
    keys = {key for patient in plans.values() for key in patient.get('plans', {})}
    deterministic = document['plan_type'] == 'deterministic'
    if not keys or any(not (key.startswith('K') or key.startswith('seed')) for key in keys):
        raise ValueError('Unrecognized immutable prompt-plan key format')
    return plans, deterministic


def plan_items(patient: dict, deterministic: bool):
    items = patient['plans'].items()
    if deterministic:
        return sorted(items, key=lambda item: int(item[0][1:]))
    return sorted(items, key=lambda item: (int(item[0].split('_K')[-1]), item[0]))


def session_for_k(model_dir: Path, checkpoint: Path, device: torch.device) -> JointLassoSession:
    session = JointLassoSession(device=device, do_autozoom=True, verbose=False, use_pinned_memory=True)
    session.initialize_from_trained_model_folder(str(model_dir), use_fold=0)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    session.network.load_state_dict(state['model_state_dict'], strict=True)
    session.network.eval()
    return session


def write_prediction(prediction_zyx: torch.Tensor, reference_path: Path, destination: Path) -> None:
    """Save onto the exact target grid consumed by ``load_case()``."""
    reference = sitk.ReadImage(str(reference_path))
    output = sitk.GetImageFromArray(prediction_zyx.detach().cpu().numpy().astype(np.uint8))
    output.CopyInformation(reference)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(output, str(destination), useCompression=True)


def hd95_and_endpoints(pred: np.ndarray, gt: np.ndarray, reference: sitk.Image):
    if not gt.any():
        raise RuntimeError('Ground-truth CTV is empty')
    if not pred.any():
        return None, None, None
    pred_surface = pred ^ ndimage.binary_erosion(pred)
    gt_surface = gt ^ ndimage.binary_erosion(gt)
    spacing = tuple(reversed(reference.GetSpacing()))
    distances = np.concatenate((
        ndimage.distance_transform_edt(~gt_surface, sampling=spacing)[pred_surface],
        ndimage.distance_transform_edt(~pred_surface, sampling=spacing)[gt_surface],
    ))
    origin_z = reference.GetOrigin()[2]
    step_z = reference.GetDirection()[8] * reference.GetSpacing()[2]
    def ends(mask):
        values = origin_z + step_z * np.where(mask)[0]
        return float(values.max()), float(values.min())
    pred_sup, pred_inf = ends(pred); gt_sup, gt_inf = ends(gt)
    return float(np.percentile(distances, 95)), abs(pred_sup - gt_sup), abs(pred_inf - gt_inf)


def describe(values):
    values = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    if not len(values):
        return {'n_valid': 0, 'mean': None, 'std': None, 'median': None, 'q25': None, 'q75': None}
    return {'n_valid': int(len(values)), 'mean': float(values.mean()), 'std': float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            'median': float(np.median(values)), 'q25': float(np.quantile(values, .25)), 'q75': float(np.quantile(values, .75))}


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA device requested but unavailable')
    plans, deterministic = load_plan(args.plan)
    mode = 'main_deterministic' if deterministic else 'random_robustness'
    output_dir = args.output_dir or (TEST_RESULTS / 'fold0_checkpoint' if args.external_test
                                     else TEST_RESULTS / mode / f'fold_{args.fold}_only')
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = {patient_id(case): case for case in patient_dirs(args.data_root)}
    if args.external_test:
        if args.fold != 0:
            raise ValueError('External test mode is restricted to the available fold0 checkpoint')
        fold_by_patient = {pid: args.fold for pid in cases}
        heldout = set(cases)
        if not heldout or not heldout.issubset(set(map(int, plans))):
            raise RuntimeError('Plan/data coverage mismatch for external test')
    else:
        fold_by_patient = load_fold_by_patient(args.split_path)
        missing_cases = sorted(set(map(int, plans)) - set(cases))
        missing_folds = sorted(set(map(int, plans)) - set(fold_by_patient))
        if missing_cases or missing_folds:
            raise RuntimeError(f'Plan cannot be evaluated: missing_cases={missing_cases}, missing_folds={missing_folds}')
        if args.fold not in range(5):
            raise ValueError(f'--fold must be 0..4, got {args.fold}')
        heldout = {pid for pid, fold in fold_by_patient.items() if fold == args.fold}
        if not heldout or not heldout.issubset(set(map(int, plans))):
            raise RuntimeError('Plan/data mismatch for requested held-out fold')

    grouped: dict[tuple[int, str], list[tuple[int, dict]]] = defaultdict(list)
    for patient_key, patient in plans.items():
        pid = int(patient_key)
        if pid not in heldout:
            continue
        for plan_key, record in plan_items(patient, deterministic):
            if not record['feasible']:
                raise RuntimeError(f'Infeasible immutable plan: patient={pid}, key={plan_key}, reason={record["reason"]}')
            prompts = [int(z) for z in record['slices']]
            if not prompts:
                raise RuntimeError(f'Empty prompt plan: patient={pid}, key={plan_key}')
            grouped[(fold_by_patient[pid], plan_key)].append((pid, record))

    rows: list[dict] = []
    for (fold, plan_key), tasks in sorted(grouped.items()):
        if fold != args.fold:
            continue
        checkpoint = args.train_results / 'Mixed_K1_5' / f'fold_{fold}' / 'checkpoints' / 'best.pth'
        if not checkpoint.is_file():
            raise FileNotFoundError(f'Missing held-out fold checkpoint: {checkpoint}')
        # New session for every (fold, K/seed-K) prevents cross-K interaction state.
        session = session_for_k(args.model_dir, checkpoint, device)
        with torch.inference_mode():
            for index, (pid, record) in enumerate(sorted(tasks), 1):
                case_dir = cases[pid]
                # ``load_case`` uses this preprocessed GT target as well. Keep
                # prediction geometry tied to that same source, rather than to
                # a separately named prompt-generator mask.
                target_reference = case_dir / 'CTV.nii.gz'
                if not target_reference.is_file():
                    raise FileNotFoundError(f'Missing load_case target reference: {target_reference}')
                image, target = load_case(case_dir)
                prompts = [int(z) for z in record['slices']]
                if any(z < 0 or z >= image.shape[0] or not target[z].any() for z in prompts):
                    raise RuntimeError(f'Plan prompt is not GT-positive/in-grid: patient={pid}, key={plan_key}')
                started = time.monotonic()
                session.set_joint_lassos(image, joint_axial_lasso(target, prompts))
                prediction = session.native_predict_joint().to(device)
                pred_np = prediction.detach().cpu().numpy().astype(bool)
                reference = sitk.ReadImage(str(target_reference))
                hd95, sup_mae, inf_mae = hd95_and_endpoints(pred_np, target.astype(bool), reference)
                row = {
                    'patient_id': pid, 'fold': fold, 'plan_key': plan_key,
                    'K': len(prompts), 'prompt_slices': json.dumps(prompts),
                    'unprompted_dice': unprompted_hard_dice(prediction, torch.from_numpy(target).to(device), prompts),
                    'whole_volume_dice': whole_volume_hard_dice(prediction, torch.from_numpy(target).to(device)),
                    'hd95_3d_mm': hd95, 'prediction_empty': int(not pred_np.any()),
                    'sup_endpoint_mae_mm': sup_mae, 'inf_endpoint_mae_mm': inf_mae,
                    'seconds': time.monotonic() - started,
                }
                rows.append(row)
                if args.save_predictions:
                    write_prediction(prediction, target_reference, output_dir / 'predictions' / plan_key / f'p_{pid:03d}.nii.gz')
                print(f'[{mode} fold{fold} {plan_key} {index}/{len(tasks)}] p_{pid:03d} W={row["whole_volume_dice"]:.4f} U={row["unprompted_dice"]:.4f}')
        del session
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    with (output_dir / 'per_case_metrics.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    metrics = ['whole_volume_dice', 'hd95_3d_mm', 'sup_endpoint_mae_mm', 'inf_endpoint_mae_mm', 'unprompted_dice']
    summary = {'mode': mode, 'plan': str(args.plan), 'data_root': str(args.data_root), 'model_dir': str(args.model_dir),
               'evaluation_scope': 'external_test_with_fold0_checkpoint' if args.external_test else f'fold_{args.fold}_only', 'num_evaluations': len(rows),
               'primary_metrics': metrics[:4], 'auxiliary_metrics': ['unprompted_dice'], 'by_plan': {}, 'by_K': {}}
    for key in sorted({row['plan_key'] for row in rows}):
        selected = [row for row in rows if row['plan_key'] == key]
        summary['by_plan'][key] = {'n': len(selected), 'n_empty_prediction': sum(row['prediction_empty'] for row in selected),
                                   **{metric: describe([row[metric] for row in selected]) for metric in metrics}}
    for k in range(1, 6):
        selected = [row for row in rows if row['K'] == k]
        summary['by_K'][f'K{k}'] = {'n': len(selected), 'n_empty_prediction': sum(row['prediction_empty'] for row in selected),
                                    **{metric: describe([row[metric] for row in selected]) for metric in metrics}}
    (output_dir / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
