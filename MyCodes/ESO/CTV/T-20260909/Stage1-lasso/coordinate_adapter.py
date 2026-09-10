"""The sole [Z,Y,X] <-> nnInteractive [X,Y,Z] conversion boundary.

The target convention is source-confirmed: the official session documents its
input as [C,X,Y,Z], and its crop/bbox helpers document [X,Y,Z].
"""
from __future__ import annotations
import numpy as np
import torch

def zyx_to_xyz(array: np.ndarray) -> np.ndarray:
    if array.ndim != 3:
        raise ValueError(f"Expected [Z,Y,X], got {array.shape}")
    return np.ascontiguousarray(array.transpose(2, 1, 0))

def xyz_to_zyx(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 3:
        return tensor.permute(2, 1, 0).contiguous()
    if tensor.ndim == 4:  # [C,X,Y,Z]
        return tensor.permute(0, 3, 2, 1).contiguous()
    raise ValueError(f"Expected [X,Y,Z] or [C,X,Y,Z], got {tuple(tensor.shape)}")
