from __future__ import annotations
import torch
import torch.nn.functional as F

class DifferentiableLogitBuffer:
    """[2,X,Y,Z] last-write-wins logits plus an explicit coverage mask."""
    def __init__(self, spatial_xyz: tuple[int, int, int], device: torch.device):
        self.logits = torch.zeros((2, *spatial_xyz), device=device, dtype=torch.float32)
        self.coverage = torch.zeros(spatial_xyz, device=device, dtype=torch.bool)

    def overwrite(self, patch_logits: torch.Tensor, bbox: list[list[int]]) -> None:
        if patch_logits.shape[0] != 2: raise ValueError("Expected two-class logits")
        starts = [max(0, b[0]) for b in bbox]; ends = [min(self.logits.shape[i+1], b[1]) for i,b in enumerate(bbox)]
        if any(a >= b for a,b in zip(starts, ends)): return
        src = [slice(a - box[0], b - box[0]) for a,b,box in zip(starts, ends, bbox)]
        dst = tuple(slice(a,b) for a,b in zip(starts, ends))
        patch = patch_logits[(slice(None), *src)].float()
        # CopySlices is graph-safe; sanity tests verify last-write gradient semantics.
        updated = self.logits.clone()
        updated[(slice(None), *dst)] = patch
        self.logits = updated
        self.coverage[dst] = True

    def overwrite_zoomed(self, logits: torch.Tensor, scaled_size: list[int], bbox: list[list[int]]) -> None:
        if tuple(logits.shape[1:]) != tuple(scaled_size):
            logits = F.interpolate(logits[None].float(), size=scaled_size, mode="trilinear", align_corners=False)[0]
        self.overwrite(logits, bbox)

    def fill_only_uncovered(self, patch_logits: torch.Tensor, bbox: list[list[int]]) -> int:
        """Write raw logits only where this buffer has no earlier native/sweep logits.

        The valid-volume slice deliberately excludes any padded part of an edge
        patch. Existing native logits are retained voxel-for-voxel.
        """
        if patch_logits.shape[0] != 2:
            raise ValueError("Expected two-class logits")
        starts = [max(0, box[0]) for box in bbox]
        ends = [min(self.logits.shape[i + 1], box[1]) for i, box in enumerate(bbox)]
        if any(start >= end for start, end in zip(starts, ends)):
            return 0
        src = tuple(slice(start - box[0], end - box[0]) for start, end, box in zip(starts, ends, bbox))
        dst = tuple(slice(start, end) for start, end in zip(starts, ends))
        fill = ~self.coverage[dst]
        count = int(fill.sum().item())
        if count == 0:
            return 0
        patch = patch_logits[(slice(None), *src)].float()
        current = self.logits[(slice(None), *dst)]
        updated = self.logits.clone()
        updated[(slice(None), *dst)] = torch.where(fill.unsqueeze(0), patch, current)
        self.logits = updated
        self.coverage[dst] |= fill
        return count
