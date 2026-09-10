from __future__ import annotations
import torch
import torch.nn.functional as F

def unprompted_dice_ce(logits_2zyx: torch.Tensor, target_zyx: torch.Tensor, prompt_slices: list[int]) -> torch.Tensor:
    if logits_2zyx.ndim != 4 or logits_2zyx.shape[0] != 2: raise ValueError("Expected [2,Z,Y,X]")
    target = target_zyx.to(logits_2zyx.device, dtype=torch.long)
    mask = torch.ones(target.shape, dtype=torch.bool, device=target.device)
    mask[torch.as_tensor(prompt_slices, device=target.device, dtype=torch.long)] = False
    if not bool(mask.any()): raise ValueError("No unprompted voxels")
    ce = F.cross_entropy(logits_2zyx[None], target[None], reduction="none")[0]
    ce = ce[mask].mean()
    fg = logits_2zyx.softmax(0)[1]
    truth = target.float()
    eps = 1e-5
    dice = (2 * (fg[mask] * truth[mask]).sum() + eps) / (fg[mask].sum() + truth[mask].sum() + eps)
    return 0.5 * (1 - dice) + 0.5 * ce

def unprompted_hard_dice(pred_zyx: torch.Tensor, target_zyx: torch.Tensor, prompt_slices: list[int]) -> float:
    keep = torch.ones(target_zyx.shape, dtype=torch.bool, device=target_zyx.device)
    keep[torch.as_tensor(prompt_slices, device=target_zyx.device)] = False
    pred, target = pred_zyx.bool()[keep], target_zyx.bool()[keep]
    return float((2 * (pred & target).sum().float() + 1e-5) / (pred.sum() + target.sum() + 1e-5))

def whole_volume_hard_dice(pred_zyx: torch.Tensor, target_zyx: torch.Tensor) -> float:
    pred, target = pred_zyx.bool(), target_zyx.bool()
    return float((2 * (pred & target).sum().float() + 1e-5) / (pred.sum() + target.sum() + 1e-5))
