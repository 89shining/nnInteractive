#!/usr/bin/env python
"""Create immutable, model-agnostic Stage-1 axial prompt plans and QC reports."""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from functools import lru_cache
from pathlib import Path

import SimpleITK as sitk
import numpy as np

KS = (1, 2, 3, 4, 5)
DEFAULT_DATA = Path('/home/wusi/SAM2/MyTrain/SAM2data/Eso/20260909_CTV/PreprocessDataNii/train')
DEFAULT_OUTPUT = Path('/home/wusi/nnInteractive/MyResults/Eso/20260909_CTV/Stage1-lasso/TestResults/prompt_plans')


def positive_slices(patient_dir: Path) -> list[int]:
    # Match all three Stage-1 datasets and evaluation drivers exactly.
    mask = sitk.GetArrayFromImage(sitk.ReadImage(str(patient_dir / 'CTV.nii.gz')))
    return np.flatnonzero(mask.reshape(mask.shape[0], -1).any(axis=1)).astype(int).tolist()


def nearest_rank(q: float, n: int) -> int:
    # Half ties resolve toward the smaller rank, deterministically.
    return int(np.floor(q * (n - 1) + 0.5 - 1e-12))


def deterministic_plan(pos: list[int], k: int, min_gap: int) -> dict:
    if len(pos) < k:
        return {'feasible': False, 'reason': 'fewer positive slices than K', 'slices': [],
                'strata_rank_bounds': [], 'stratum_center_ranks': [], 'normalized_stratum_centers': []}
    bounds = stratum_bounds(len(pos), k)
    # Integer stratum centers, not continuous global quantiles: this is the
    # exact deterministic counterpart to one-random-slice-per-stratum training.
    targets = [(lo + hi - 1) / 2 for lo, hi in bounds]
    qs = [target / (len(pos) - 1) if len(pos) > 1 else 0.5 for target in targets]

    @lru_cache(None)
    def solve(stratum: int, previous_z: int):
        if stratum == k:
            return 0.0, ()
        best = None
        lo, hi = bounds[stratum]
        for i in range(lo, hi):
            if previous_z >= 0 and pos[i] - previous_z < min_gap:
                continue
            tail = solve(stratum + 1, pos[i])
            if tail is None:
                continue
            candidate = (abs(i - targets[stratum]) + tail[0], (pos[i],) + tail[1])
            if best is None or candidate < best:
                best = candidate
        return best

    answer = solve(0, -1)
    if answer is None:
        return {'feasible': False, 'reason': f'cannot satisfy min_gap={min_gap}', 'slices': [],
                'strata_rank_bounds': bounds, 'stratum_center_ranks': targets,
                'normalized_stratum_centers': qs}
    naive = [pos[nearest_rank(q, len(pos))] for q in qs]
    selected = list(answer[1])
    return {
        'feasible': True, 'reason': '', 'slices': selected,
        'strata_rank_bounds': bounds, 'stratum_center_ranks': targets,
        'normalized_stratum_centers': qs,
        'naive_slices': naive,
        'duplicate_adjusted': len(set(naive)) != len(naive) and selected != naive,
        'gap_adjusted': any(b - a < min_gap for a, b in zip(naive, naive[1:])) and selected != naive,
    }


def stratum_bounds(n: int, k: int) -> list[tuple[int, int]]:
    """Half-open, contiguous strata over ordered positive-slice ranks."""
    return [(i * n // k, (i + 1) * n // k) for i in range(k)]


def random_plan(pos: list[int], k: int, min_gap: int, global_seed: int, patient_id: int) -> dict:
    """Uniformly sample feasible choices with exactly one positive slice per stratum."""
    if len(pos) < k:
        return {'feasible': False, 'reason': 'fewer positive slices than K', 'slices': []}
    bounds = stratum_bounds(len(pos), k)

    @lru_cache(None)
    def count(stratum: int, previous_z: int) -> int:
        if stratum == k:
            return 1
        lo, hi = bounds[stratum]
        return sum(count(stratum + 1, pos[i]) for i in range(lo, hi)
                   if previous_z < 0 or pos[i] - previous_z >= min_gap)

    total = count(0, -1)
    if not total:
        return {'feasible': False, 'reason': f'cannot satisfy min_gap={min_gap}', 'slices': []}
    case_seed = global_seed + patient_id * 10007 + k * 101
    rng, chosen, previous_z = random.Random(case_seed), [], -1
    for stratum, (lo, hi) in enumerate(bounds):
        candidates = [(i, count(stratum + 1, pos[i])) for i in range(lo, hi)
                      if (previous_z < 0 or pos[i] - previous_z >= min_gap)]
        draw = rng.randrange(sum(weight for _, weight in candidates))
        for i, weight in candidates:
            if draw < weight:
                chosen.append(pos[i]); previous_z = pos[i]; break
            draw -= weight
    return {'feasible': True, 'reason': '', 'slices': chosen, 'feasible_combinations': total,
            'case_seed': case_seed, 'strata_rank_bounds': bounds}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--placement', choices=('deterministic', 'random'), default='deterministic')
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument('--min-gap', type=int, default=2)
    args = parser.parse_args()
    patients = sorted(p for p in args.data_root.iterdir() if p.is_dir() and p.name.startswith('p_'))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_rows, plans, positive_sets, summary = [], {}, {}, Counter()
    seed_list = [None] if args.placement == 'deterministic' else args.seeds
    for patient_dir in patients:
        patient_id = int(patient_dir.name.rsplit('_', 1)[1]); pos = positive_slices(patient_dir)
        positive_sets[patient_id] = set(pos)
        plans[str(patient_id)] = {'patient_id': patient_id, 'positive_slice_min': pos[0] if pos else None,
                                  'positive_slice_max': pos[-1] if pos else None, 'num_positive_slices': len(pos), 'plans': {}}
        for seed in seed_list:
            for k in KS:
                record = deterministic_plan(pos, k, args.min_gap) if seed is None else random_plan(pos, k, args.min_gap, seed, patient_id)
                key = f'K{k}' if seed is None else f'seed{seed}_K{k}'
                plans[str(patient_id)]['plans'][key] = record
                summary[f'K{k}_feasible' if record['feasible'] else f'K{k}_infeasible'] += 1
                summary['duplicate_adjustments'] += int(record.get('duplicate_adjusted', False))
                summary['gap_adjustments'] += int(record.get('gap_adjusted', False))
                all_rows.append({'patient_id': patient_id, 'K': k, 'seed': '' if seed is None else seed,
                                 'positive_slice_min': pos[0] if pos else '', 'positive_slice_max': pos[-1] if pos else '',
                                 'num_positive_slices': len(pos),
                                 'strata_rank_bounds': json.dumps(record.get('strata_rank_bounds', [])),
                                 'stratum_center_ranks': json.dumps(record.get('stratum_center_ranks', [])),
                                 'normalized_stratum_centers': json.dumps(record.get('normalized_stratum_centers', [])),
                                 'selected_slices': json.dumps(record['slices']), 'feasible': record['feasible'], 'reason': record['reason']})
    stem = 'stage1_prompt_plan' if args.placement == 'deterministic' else 'stage1_random_prompt_plan'
    suffix = '' if args.placement == 'deterministic' else '_seeds_' + '-'.join(map(str, args.seeds))
    plan_document = {
        'schema_version': 2,
        'plan_type': 'deterministic' if args.placement == 'deterministic' else 'random_robustness',
        'patients': plans,
    }
    (args.output_dir / f'{stem}{suffix}.json').write_text(json.dumps(plan_document, indent=2), encoding='utf-8')
    with (args.output_dir / f'{stem}{suffix}.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=all_rows[0].keys()); writer.writeheader(); writer.writerows(all_rows)
    invalid_slice = any(any(z not in positive_sets[int(patient_id)] for z in patient['plans'][key]['slices'])
                        for patient_id, patient in plans.items() for key in patient['plans'])
    invalid_gap = any(any(b - a < args.min_gap for a, b in zip(record['slices'], record['slices'][1:]))
                      for patient in plans.values() for record in patient['plans'].values())
    examples = {pid: plans[pid]['plans'] for pid in list(plans)[:10]}
    qc = {'schema_version': 2, 'plan_type': plan_document['plan_type'], 'placement': args.placement, 'patients': len(patients), 'positive_slice_counts': [plans[str(int(p.name.rsplit('_', 1)[1]))]['num_positive_slices'] for p in patients],
          'selected_non_positive_slice_exists': invalid_slice, 'gap_violation_exists': invalid_gap,
          'first_10_patients': examples, **summary}
    (args.output_dir / f'{stem}{suffix}_qc.json').write_text(json.dumps(qc, indent=2), encoding='utf-8')
    print(json.dumps(qc, indent=2))


if __name__ == '__main__':
    main()
