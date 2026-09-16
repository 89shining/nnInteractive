from __future__ import annotations
import torch
import torch.nn.functional as F

class DifferentiableLogitBuffer:
    """[2,X,Y,Z] last-write-wins logits with lazy graph-safe materialization."""
    def __init__(self, spatial_xyz: tuple[int, int, int], device: torch.device):
        self.spatial_xyz = tuple(map(int, spatial_xyz))
        self.device = device
        self.coverage = torch.zeros(self.spatial_xyz, device=device, dtype=torch.bool)
        self._writes: list[tuple[torch.Tensor, tuple[slice, slice, slice], torch.Tensor | None]] = []

    @property
    def logits(self) -> torch.Tensor:
        out = torch.zeros((2, *self.spatial_xyz), device=self.device, dtype=torch.float32)
        for patch, dst, fill_mask in self._writes:
            target = (slice(None), *dst)
            if fill_mask is None:
                out[target] = patch
            else:
                # The recorded mask is the coverage state at write time, so this
                # reproduces fill-only-uncovered while preserving gradients to patch.
                out[target] = torch.where(fill_mask.unsqueeze(0), patch, out[target])
        return out

    def overwrite(self, patch_logits: torch.Tensor, bbox: list[list[int]]) -> None:
        if patch_logits.shape[0] != 2: raise ValueError("Expected two-class logits")
        starts=[max(0,b[0]) for b in bbox]; ends=[min(self.spatial_xyz[i],b[1]) for i,b in enumerate(bbox)]
        if any(a>=b for a,b in zip(starts,ends)): return
        src=tuple(slice(a-box[0],b-box[0]) for a,b,box in zip(starts,ends,bbox)); dst=tuple(slice(a,b) for a,b in zip(starts,ends))
        self._writes.append((patch_logits[(slice(None),*src)].float(),dst,None)); self.coverage[dst]=True

    def overwrite_zoomed(self, logits: torch.Tensor, scaled_size: list[int], bbox: list[list[int]]) -> None:
        if tuple(logits.shape[1:]) != tuple(scaled_size): logits=F.interpolate(logits[None].float(),size=scaled_size,mode="trilinear",align_corners=False)[0]
        self.overwrite(logits,bbox)

    def fill_only_uncovered(self, patch_logits: torch.Tensor, bbox: list[list[int]]) -> int:
        if patch_logits.shape[0] != 2: raise ValueError("Expected two-class logits")
        starts=[max(0,b[0]) for b in bbox]; ends=[min(self.spatial_xyz[i],b[1]) for i,b in enumerate(bbox)]
        if any(a>=b for a,b in zip(starts,ends)): return 0
        src=tuple(slice(a-box[0],b-box[0]) for a,b,box in zip(starts,ends,bbox)); dst=tuple(slice(a,b) for a,b in zip(starts,ends))
        fill=~self.coverage[dst]; count=int(fill.sum().item())
        if count==0: return 0
        self._writes.append((patch_logits[(slice(None),*src)].float(),dst,fill)); self.coverage[dst] |= fill
        return count
