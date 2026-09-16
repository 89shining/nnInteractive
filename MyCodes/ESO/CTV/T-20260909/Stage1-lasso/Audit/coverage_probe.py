#!/usr/bin/env python
"""Single-case training-mode feasibility probe; it never calls optimizer.step()."""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from dataset import load_case, patient_id
from interaction_builder import joint_axial_lasso
from losses import unprompted_dice_ce
from prediction_wrapper import JointLassoSession, differentiable_predict_joint

DATA = Path('/home/wusi/SAM2/MyTrain/SAM2data/Eso/20260909_CTV/PreprocessDataNii/train')
PLAN = Path('/home/wusi/SAM2/MyTrain/SAM2data/Eso/20260909_CTV/Stage1-mask/TrainResults/validation_prompt_plan.json')
MODEL = Path('/home/wusi/nnInteractive/nnInteractive_v1.0')

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--patient', default='p_9')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--data-root', type=Path, default=DATA)
    parser.add_argument('--plan', type=Path, default=PLAN)
    parser.add_argument('--model-dir', type=Path, default=MODEL)
    parser.add_argument('--out', type=Path, default=Path(__file__).resolve().parent / 'coverage_probe_report.json')
    args = parser.parse_args()

    plan = json.loads(args.plan.read_text())
    case_dir = args.data_root / args.patient
    record = plan['folds'][str(args.fold)][str(patient_id(case_dir))]
    image, gt = load_case(case_dir)
    device = torch.device('cuda')
    session = JointLassoSession(device=device, do_autozoom=True, verbose=False)
    session.initialize_from_trained_model_folder(str(args.model_dir), use_fold=0)
    session.network.train()
    for parameter in session.network.parameters():
        parameter.requires_grad_(True)

    report = {'patient': args.patient, 'fold': args.fold, 'patch_size_xyz': list(session.configuration_manager.patch_size), 'runs': []}
    for k in (1, 5):
        prompts = list(map(int, record['placements'][str(k)][0]['prompt_frame_ids']))
        session.network.zero_grad(set_to_none=True)
        session.set_joint_lassos(image, joint_axial_lasso(gt, prompts))
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        start_forward = time.perf_counter()
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits, coverage, _ = differentiable_predict_joint(
                session,
                complete_coverage=True,
                activation_checkpointing=True,
                prompt_slices=prompts,
            )
            loss_mask = torch.ones(gt.shape, dtype=torch.bool, device=device)
            loss_mask[prompts] = False
            if not bool(coverage[loss_mask].all()):
                raise RuntimeError(f'K={k}: completion did not cover all unprompted voxels')
            loss = unprompted_dice_ce(logits, torch.from_numpy(gt).to(device), prompts)
        forward_seconds = time.perf_counter() - start_forward
        start_backward = time.perf_counter()
        loss.backward()
        backward_seconds = time.perf_counter() - start_backward
        report['runs'].append({
            'K': k,
            'prompt_slices': prompts,
            'loss': float(loss.detach()),
            'coverage_before_completion': session.last_prediction_stats['native_covered_voxels'],
            'coverage_after_completion': session.last_prediction_stats['final_covered_voxels'],
            'unprompted_voxels': int(loss_mask.sum().item()),
            'stats': session.last_prediction_stats,
            'peak_allocated_bytes': int(torch.cuda.max_memory_allocated(device)),
            'peak_reserved_bytes': int(torch.cuda.max_memory_reserved(device)),
            'forward_seconds': forward_seconds,
            'backward_seconds': backward_seconds,
        })
        print(json.dumps(report['runs'][-1], indent=2), flush=True)
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    print('PROBE_PASS', json.dumps(report, indent=2), flush=True)

if __name__ == '__main__':
    main()
