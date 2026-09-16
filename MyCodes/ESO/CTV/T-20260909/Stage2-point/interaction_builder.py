"""Component-wise axial closed-contour lasso construction.

K always counts prompted axial slices.  A prompted slice may contain more
than one disconnected CTV region; each region contributes its own closed
contour, and all contours are OR-merged into the single joint interaction
volume at unit strength.
"""
from __future__ import annotations
import numpy as np

def connected_components_4(mask_yx: np.ndarray) -> int:
    """Small dependency-free 4-connected component count for prompt QC."""
    mask=np.asarray(mask_yx,dtype=bool); seen=np.zeros_like(mask); count=0
    height,width=mask.shape
    for y,x in zip(*np.nonzero(mask)):
        if seen[y,x]: continue
        count+=1; stack=[(int(y),int(x))]; seen[y,x]=True
        while stack:
            cy,cx=stack.pop()
            for ny,nx in ((cy-1,cx),(cy+1,cx),(cy,cx-1),(cy,cx+1)):
                if 0<=ny<height and 0<=nx<width and mask[ny,nx] and not seen[ny,nx]:
                    seen[ny,nx]=True; stack.append((ny,nx))
    return count

def component_masks_4(mask_yx: np.ndarray) -> list[np.ndarray]:
    """Return one binary mask per 4-connected foreground component."""
    mask = np.asarray(mask_yx, dtype=bool)
    seen = np.zeros_like(mask, dtype=bool)
    height, width = mask.shape
    components: list[np.ndarray] = []
    for y, x in zip(*np.nonzero(mask)):
        if seen[y, x]:
            continue
        component = np.zeros_like(mask, dtype=bool)
        stack = [(int(y), int(x))]
        seen[y, x] = True
        while stack:
            cy, cx = stack.pop()
            component[cy, cx] = True
            for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        components.append(component)
    return components

def closed_contour(mask_yx: np.ndarray) -> np.ndarray:
    """One-voxel closed boundary; GT filling is deliberately not used as lasso."""
    mask = np.asarray(mask_yx, dtype=bool)
    if not mask.any(): raise ValueError("Cannot make a lasso from an empty mask")
    interior = mask.copy()
    interior[0] = interior[-1] = False; interior[:, 0] = interior[:, -1] = False
    interior[1:-1, 1:-1] &= mask[:-2, 1:-1] & mask[2:, 1:-1] & mask[1:-1, :-2] & mask[1:-1, 2:]
    return (mask & ~interior).astype(np.float32)

def joint_axial_lasso(mask_zyx: np.ndarray, prompt_slices: list[int]) -> np.ndarray:
    """Joint unit-strength 4-neighbour contours for K prompted slices."""
    result = np.zeros_like(mask_zyx, dtype=np.float32)
    for z in sorted(set(map(int, prompt_slices))):
        mask = np.asarray(mask_zyx[z], dtype=bool)
        if not mask.any(): raise ValueError(f"Prompt slice {z} is empty")
        result[z] = closed_contour(mask)
    if float(result.max()) != 1.0: raise RuntimeError("Lasso must have unit strength")
    return result
