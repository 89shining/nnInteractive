"""Deterministic/seeded 3-D residual click oracle for nnInteractive Stage-2."""
from __future__ import annotations

from dataclasses import dataclass
import math
import random
import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class CorrectionClick:
    """A click in dataset [Z,Y,X] coordinates; ``positive`` denotes FN."""
    zyx: tuple[int, int, int]
    positive: bool

    @property
    def xyz(self) -> tuple[int, int, int]:
        z, y, x = self.zyx
        return x, y, z


def _largest_component(mask: np.ndarray, error_type: str):
    labels, count = ndimage.label(mask, structure=np.ones((3, 3, 3), dtype=np.uint8))
    if count == 0: return None
    sizes = np.bincount(labels.ravel(), minlength=int(count) + 1); sizes[0] = 0
    largest = int(sizes.max()); best = None
    for index in np.flatnonzero(sizes == largest):
        first = tuple(int(v) for v in np.argwhere(labels == index)[0])
        candidate = (-largest, first, error_type, int(index))
        if best is None or candidate < best: best = candidate
    return best, labels

def next_error_click(
    prediction_zyx: np.ndarray,
    target_zyx: np.ndarray,
    initial_lasso_slices: list[int],
    spacing_zyx: tuple[float, float, float],
    *,
    rng: random.Random | None,
) -> CorrectionClick | None:
    """Choose largest 26-connected FN/FP component, excluding lasso slices.

    Training supplies ``rng`` and samples uniformly from the top EDT quartile
    (the component's spacing-aware inner 25%). Validation supplies ``None``
    and uses the deterministic EDT maximum.
    """
    pred = np.asarray(prediction_zyx, dtype=bool)
    truth = np.asarray(target_zyx, dtype=bool)
    if pred.shape != truth.shape or pred.ndim != 3:
        raise ValueError("Prediction/target must share [Z,Y,X] grid")
    excluded = np.zeros(pred.shape[0], dtype=bool)
    for z in initial_lasso_slices:
        if not 0 <= int(z) < pred.shape[0]:
            raise IndexError(f"Initial lasso slice {z} is outside Z={pred.shape[0]}")
        excluded[int(z)] = True
    # Identical to SAM2's oracle: evaluate *every* 26-connected FN/FP
    # component, then break ties by its lexicographically first [Z,Y,X] voxel
    # and finally FN before FP.
    winner = None; winning_labels = None
    for error_type, residual in (("FN", truth & ~pred), ("FP", pred & ~truth)):
        residual = residual.copy(); residual[excluded] = False
        item = _largest_component(residual, error_type)
        if item is None: continue
        candidate, labels = item
        if winner is None or candidate < winner:
            winner, winning_labels = candidate, labels
    if winner is None: return None
    _, _, error_type, label = winner
    component = winning_labels == label
    positive = error_type == "FN"
    distance = ndimage.distance_transform_edt(component, sampling=spacing_zyx)
    points = np.argwhere(component)
    scores = distance[component]
    if rng is None:
        maximum = float(scores.max())
        chosen = points[np.isclose(scores, maximum)][0]
    else:
        keep = max(1, int(math.ceil(0.25 * len(points))))
        # Stable order gives exactly the SAM2 top-ceil(25%) set even when EDT
        # values tie, then the supplied episode RNG selects uniformly within it.
        inner = points[np.argsort(-scores, kind="stable")[:keep]]
        chosen = inner[rng.randrange(len(inner))]
    z, y, x = (int(v) for v in chosen)
    if z in set(initial_lasso_slices):
        raise RuntimeError("Correction click landed on an initial lasso slice")
    return CorrectionClick((z, y, x), positive)
