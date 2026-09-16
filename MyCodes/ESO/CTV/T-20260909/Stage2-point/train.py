#!/usr/bin/env python
"""Five-fold nnInteractive Stage-2 point-correction fine-tuning."""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset import load_case_with_spacing, patient_dirs, patient_id
from prompt_sampler import positive_slices, sample_train
from prediction_wrapper import JointLassoSession
from stage2_loops import train_episode, validate_fold


DATA = Path('/home/wusi/SAM2/MyTrain/SAM2data/Eso/20260909_CTV/PreprocessDataNii/train')
SPLITS = Path('/home/wusi/SAM2/MyTrain/MyCodes/ESO/CTV/T-20260901/shared_splits.json')
PLAN = Path('/home/wusi/SAM2/MyTrain/SAM2data/Eso/20260909_CTV/Stage1-mask/TrainResults/validation_prompt_plan.json')
MODEL = Path('/home/wusi/nnInteractive/nnInteractive_v1.0')
STAGE1 = Path('/home/wusi/nnInteractive/MyResults/Eso/20260909_CTV/Stage1-lasso/TrainResults')
OUT = Path('/home/wusi/nnInteractive/MyResults/Eso/20260909_CTV/Stage2-point/TrainResults')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, default=DATA)
    parser.add_argument('--split-path', type=Path, default=SPLITS)
    parser.add_argument('--validation-plan', type=Path, default=PLAN)
    parser.add_argument('--model-dir', type=Path, default=MODEL)
    parser.add_argument('--stage1-root', type=Path, default=STAGE1)
    parser.add_argument('--stage1-ckpt', type=Path, default=None,
                        help='Optional explicit Stage-1 best; allowed only with one matching --fold.')
    parser.add_argument('--output-root', type=Path, default=OUT)
    parser.add_argument('--fold', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=20260909)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--no-resume', action='store_true')
    return parser.parse_args()


def load_folds(path: Path, root: Path):
    by_name = {case.name: case for case in patient_dirs(root)}
    output = []
    for item in json.loads(path.read_text())['folds']:
        output.append({
            'fold': int(item['fold']),
            'train': [by_name[name] for name in item['train']],
            'val': [by_name[name] for name in item['val']],
        })
    return output


def stage1_checkpoint(args, fold: int) -> Path:
    if args.stage1_ckpt is None:
        path = args.stage1_root / 'Mixed_K1_5' / f'fold_{fold}' / 'checkpoints' / 'best.pth'
    else:
        if args.fold != fold:
            raise RuntimeError('--stage1-ckpt requires exactly one explicit matching --fold')
        path = args.stage1_ckpt
        if f'fold_{fold}' not in {parent.name for parent in path.parents}:
            raise RuntimeError(f'Refusing cross-fold initialization: {path} is not under fold_{fold}')
    if not path.is_file():
        raise FileNotFoundError(f'Missing matching Stage-1 best checkpoint for fold {fold}: {path}')
    return path


def save_checkpoint(path: Path, *, epoch, model, optimizer, scheduler, best, best_epoch, patience, args, stage1_ckpt):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_metric': best,
        'best_epoch': best_epoch,
        'patience_counter': patience,
        'stage1_checkpoint': str(stage1_ckpt),
        'config': vars(args),
    }, path)


def gradient_audit(model):
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    with_grad = [(name, parameter) for name, parameter in trainable if parameter.grad is not None]
    nonzero = [(name, parameter) for name, parameter in with_grad if bool((parameter.grad != 0).any())]
    if not nonzero:
        raise RuntimeError('Gradient audit failed: no nonzero trainable gradient')
    bad = [name for name, parameter in with_grad if not torch.isfinite(parameter.grad).all()]
    if bad:
        raise RuntimeError(f'Gradient audit failed: non-finite gradients: {bad[:10]}')
    print('[nnInteractive Stage2 gradient audit] PASS', {
        'trainable': len(trainable), 'with_grad': len(with_grad), 'nonzero': len(nonzero),
    })
    return nonzero[0]


def run_fold(args, spec, plan, device: torch.device):
    fold = spec['fold']
    stage1_ckpt = stage1_checkpoint(args, fold)
    run_dir = args.output_root / 'Workflow_K3_T0_5' / f'fold_{fold}'
    checkpoint_dir = run_dir / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / 'completed.flag').exists():
        print(f'fold {fold}: already completed; skip')
        return

    session = JointLassoSession(device=device, do_autozoom=True, verbose=False, use_pinned_memory=True)
    session.initialize_from_trained_model_folder(str(args.model_dir), use_fold=0)
    # Native trajectory uses an independent inference-only copy. The trainable
    # terminal network never enters inference_mode, preventing runtime caches
    # from becoming inference tensors before backward.
    native_session = JointLassoSession(device=device, do_autozoom=True, verbose=False, use_pinned_memory=True)
    native_session.initialize_from_trained_model_folder(str(args.model_dir), use_fold=0)
    model = session.network
    stage1_state = torch.load(stage1_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(stage1_state['model_state_dict'], strict=True)
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if total != trainable:
        raise RuntimeError('Stage-2 requires full-network fine-tuning')
    print(f'[fold {fold}] Stage-1 initialization: {stage1_ckpt}')
    print(f'[full-network FT] total={total:,} trainable={trainable:,} frozen={total-trainable:,}')

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    latest = checkpoint_dir / 'latest.pth'
    start, best, best_epoch, patience = 1, -float('inf'), -1, 0
    if latest.exists() and not args.no_resume:
        state = torch.load(latest, map_location=device, weights_only=False)
        if Path(state.get('stage1_checkpoint', stage1_ckpt)) != stage1_ckpt:
            raise RuntimeError('Resume checkpoint was initialized from a different Stage-1 checkpoint')
        model.load_state_dict(state['model_state_dict'])
        optimizer.load_state_dict(state['optimizer_state_dict'])
        scheduler.load_state_dict(state['scheduler_state_dict'])
        start, best = int(state['epoch']) + 1, float(state['best_metric'])
        best_epoch, patience = int(state['best_epoch']), int(state['patience_counter'])

    log_path = run_dir / 'training_log.csv'
    write_header = not log_path.exists() or start == 1
    with log_path.open('a', newline='') as log_file:
        writer = csv.DictWriter(log_file, fieldnames=[
            'epoch', 'loss_total', 'mean_budget', 'mean_realized_clicks', 'coverage_fraction',
            'D0', 'D1', 'D2', 'D3', 'D4', 'D5', 'S_workflow', 'delta_D5', 'seconds', 'native_forwards', 'completion_forwards', 'grid_tiles', 'skipped_tiles', 'filled_voxels',
        ])
        if write_header:
            writer.writeheader()
        audit_done = False
        for epoch in range(start, args.epochs + 1):
            model.train()
            started = time.time()
            losses, budgets, realized, coverage = [], [], [], []
            native_forwards, completion_forwards, grid_tiles, skipped_tiles, filled_voxels = [], [], [], [], []
            epoch_cases = list(spec['train'])
            # Match Stage-1's shuffled patient exposure while keeping resume
            # behaviour deterministic for a fixed epoch/fold/seed.
            random.Random(
                args.seed + fold * 10_000_019 + epoch * 1_000_003
            ).shuffle(epoch_cases)
            for case in epoch_cases:
                image, target, case_spacing = load_case_with_spacing(case)
                rng = random.Random(args.seed + fold * 10_000_019 + epoch * 1_000_003 + patient_id(case))
                prompts = sample_train(positive_slices(target), rng, gap=2)
                budget = rng.randrange(6)  # Uniform T in {0,1,2,3,4,5}.
                # Synchronize native control weights after the preceding optimizer step.
                native_session.network.load_state_dict(model.state_dict(), strict=True)
                native_session.network.eval()
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                    result = train_episode(session, image, target, case_spacing, prompts, budget, rng, device, native_session=native_session)
                if not torch.isfinite(result.loss):
                    raise RuntimeError(f'{case.name}: non-finite terminal loss')
                result.loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                if not audit_done:
                    audit_name, audit_parameter = gradient_audit(model)
                    before = audit_parameter.detach().clone()
                optimizer.step()
                if not audit_done:
                    if not bool((audit_parameter.detach() - before).abs().max() > 0): raise RuntimeError(f'Optimizer update audit failed for {audit_name}')
                    audit_done = True
                losses.append(float(result.loss.detach())); budgets.append(result.budget); realized.append(result.realized_clicks); coverage.append(result.coverage_fraction)
                stats = result.prediction_stats
                native_forwards.append(stats['native_forwards']); completion_forwards.append(stats['completion_forwards']); grid_tiles.append(stats['grid_tiles']); skipped_tiles.append(stats['skipped_tiles']); filled_voxels.append(stats['filled_voxels'])
            scheduler.step()

            metrics = None
            if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
                model.eval()
                metrics = validate_fold(session, spec['val'], plan, fold, device)
                value = metrics['S_workflow']
                if value > best:
                    best, best_epoch, patience = value, epoch, 0
                    save_checkpoint(checkpoint_dir / 'best.pth', epoch=epoch, model=model, optimizer=optimizer,
                                    scheduler=scheduler, best=best, best_epoch=best_epoch, patience=patience,
                                    args=args, stage1_ckpt=stage1_ckpt)
                else:
                    patience += 1
            save_checkpoint(latest, epoch=epoch, model=model, optimizer=optimizer, scheduler=scheduler,
                            best=best, best_epoch=best_epoch, patience=patience, args=args, stage1_ckpt=stage1_ckpt)
            row = {
                'epoch': epoch, 'loss_total': float(np.mean(losses)), 'mean_budget': float(np.mean(budgets)),
                'mean_realized_clicks': float(np.mean(realized)), 'coverage_fraction': float(np.mean(coverage)),
                'seconds': time.time() - started, 'native_forwards': float(np.mean(native_forwards)), 'completion_forwards': float(np.mean(completion_forwards)), 'grid_tiles': float(np.mean(grid_tiles)), 'skipped_tiles': float(np.mean(skipped_tiles)), 'filled_voxels': float(np.mean(filled_voxels)),
            }
            if metrics:
                row.update(metrics)
            writer.writerow(row)
            log_file.flush()
            print(f'[fold {fold}] epoch {epoch}/{args.epochs} loss={row["loss_total"]:.5f} '
                  f'budget={row["mean_budget"]:.2f} clicks={row["mean_realized_clicks"]:.2f} '
                  f'val={metrics or "SKIP"} best={best:.5f}@{best_epoch} time={row["seconds"]:.1f}s')
            if patience >= 2:
                break
    (run_dir / 'completed.flag').write_text('completed\n')
    session.executor.shutdown(wait=False, cancel_futures=True)
    native_session.executor.shutdown(wait=False, cancel_futures=True)


def main():
    args = parse_args()
    device = torch.device(args.device)
    if args.stage1_ckpt is not None and args.fold is None:
        raise RuntimeError('--stage1-ckpt is only safe with an explicit --fold')
    required = [
        args.model_dir / 'dataset.json', args.model_dir / 'plans.json',
        args.model_dir / 'inference_session_class.json', args.model_dir / 'fold_0' / 'checkpoint_final.pth',
    ]
    if not all(path.is_file() for path in required):
        raise FileNotFoundError('Incomplete official nnInteractive_v1.0 model bundle')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    plan = json.loads(args.validation_plan.read_text())
    folds = load_folds(args.split_path, args.data_root)
    print({
        'data_root': str(args.data_root), 'model_dir': str(args.model_dir),
        'stage1_root': str(args.stage1_root), 'output_root': str(args.output_root),
        'K_train': 'dynamic feasible 1..5', 'gap': '2 slices (10 mm)', 'T_train': 'Uniform{0..5}',
        'interaction': 'joint component-wise axial lasso + native accumulated points',
        'loss': '0.5 soft Dice + 0.5 CE, terminal, non-initial-lasso slices only',
        'validation': 'K=3, two fixed placements, D0..D5 whole-volume Dice',
        'best_metric': 'mean(D0..D5)', 'lr': args.lr, 'weight_decay': args.weight_decay,
        'bf16': device.type == 'cuda', 'gradient_clip': 1.0,
    })
    for spec in folds:
        if args.fold is None or spec['fold'] == args.fold:
            run_fold(args, spec, plan, device)


if __name__ == '__main__':
    main()
