from __future__ import annotations
import random
from typing import Sequence
import numpy as np

def positive_slices(mask_zyx: np.ndarray) -> list[int]:
    return np.flatnonzero(mask_zyx.reshape(mask_zyx.shape[0], -1).any(1)).astype(int).tolist()

def stratified_spaced(positive: Sequence[int], k: int, gap: int, rng: random.Random) -> list[int]:
    values = sorted(set(map(int, positive)))
    bins = [list(x.astype(int)) for x in np.array_split(np.asarray(values), k)]
    selected: list[int] = []
    def visit(i: int) -> bool:
        if i == len(bins): return True
        choices = bins[i][:]; rng.shuffle(choices)
        for z in choices:
            if not selected or z - selected[-1] >= gap:
                selected.append(z)
                if visit(i + 1): return True
                selected.pop()
        return False
    if k < 1 or k > len(values) or not visit(0):
        raise ValueError(f"No feasible placement for K={k}, gap={gap}")
    return selected

def sample_train(positive: Sequence[int], rng: random.Random, gap: int = 2) -> list[int]:
    feasible = []
    for k in range(1, 6):
        try: stratified_spaced(positive, k, gap, random.Random(0)); feasible.append(k)
        except ValueError: pass
    if not feasible: raise ValueError("No feasible K in 1..5")
    return stratified_spaced(positive, rng.choice(feasible), gap, rng)

