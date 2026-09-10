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


def _components(mask: np.ndarray) -> list[np.ndarray]:
    labels, count = ndimage.label(mask, structure=np.ones((3, 3, 3), dtype=np.uint8))
    return [labels == index for index in range(1, int(count) + 1)]


def _lexicographic_first(mask: np.ndarray) -> tuple[int, int, int]:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        raise ValueError("Expected nonempty connected component")
    return tuple(int(value) for value in coords[0])


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
    candidates: list[tuple[int, tuple[int, int, int], str, np.ndarray]] = []
    for error_type, residual in (("FN", truth & ~pred), ("FP", pred & ~truth)):
        residual = residual.copy()
        residual[excluded] = False
        for component in _components(residual):
            candidates.append((int(component.sum()), _lexicographic_first(component), error_type, component))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    _, _, error_type, component = candidates[0]
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
