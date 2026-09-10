from __future__ import annotations
import re
from pathlib import Path
import numpy as np
import SimpleITK as sitk

def patient_id(path: Path) -> int:
    match = re.search(r"(\d+)$", path.name)
    if not match: raise ValueError(f"Cannot parse patient id: {path}")
    return int(match.group(1))

def patient_dirs(root: Path) -> list[Path]:
    return sorted((p for p in root.glob("p_*") if p.is_dir()), key=patient_id)

def load_case(path: Path) -> tuple[np.ndarray, np.ndarray]:
    image_obj = sitk.ReadImage(str(path / "image.nii.gz"))
    ctv_obj = sitk.ReadImage(str(path / "CTV.nii.gz"))
    image = sitk.GetArrayFromImage(image_obj).astype(np.float32)
    ctv = (sitk.GetArrayFromImage(ctv_obj) > 0).astype(np.uint8)
    if image.shape != ctv.shape or image.ndim != 3: raise ValueError(f"Grid mismatch: {path}")
    if image_obj.GetSize() != ctv_obj.GetSize(): raise ValueError(f"NIfTI size mismatch: {path}")
    for label, left, right in (
        ("spacing", image_obj.GetSpacing(), ctv_obj.GetSpacing()),
        ("origin", image_obj.GetOrigin(), ctv_obj.GetOrigin()),
        ("direction", image_obj.GetDirection(), ctv_obj.GetDirection()),
    ):
        if not np.allclose(left, right, rtol=0.0, atol=1e-6):
            raise ValueError(f"NIfTI {label} mismatch: {path}")
    if image.shape[1:] != (512, 512): raise ValueError(f"Expected 512x512: {path}, {image.shape}")
    if not np.isfinite(image).all() or image.min() < -1e-6 or image.max() > 1.0 + 1e-6:
        raise ValueError(f"Expected offline [0,1] CT: {path}")
    if tuple(round(v, 6) for v in image_obj.GetSpacing())[2] != 5.0:
        raise ValueError(f"Expected Z=5 mm: {path}, {image_obj.GetSpacing()}")
    return image, ctv
