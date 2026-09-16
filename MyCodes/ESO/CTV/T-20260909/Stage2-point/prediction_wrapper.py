"""Joint-lasso nnInteractive episodes: native hard control plus trainable logits."""
from __future__ import annotations
import itertools
import os, sys
import copy
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from coordinate_adapter import zyx_to_xyz, xyz_to_zyx
from differentiable_buffer import DifferentiableLogitBuffer

def _add_project_root() -> None:
    candidates = []
    configured = os.environ.get("NNINTERACTIVE_PROJECT_ROOT")
    if configured: candidates.append(Path(configured))
    candidates += [Path("/home/wusi/nnInteractive"), *Path(__file__).resolve().parents]
    for root in candidates:
        if (root / "nnInteractive").is_dir():
            # A pip package with the same name may already be installed. Force
            # this repository ahead of site-packages rather than merely adding it.
            root_str = str(root)
            sys.path[:] = [entry for entry in sys.path if entry != root_str]
            sys.path.insert(0, root_str)
            return
    raise FileNotFoundError("Set NNINTERACTIVE_PROJECT_ROOT to the nnInteractive repository root")

_add_project_root()
for _module_name in list(sys.modules):
    if _module_name == "nnInteractive" or _module_name.startswith("nnInteractive."):
        sys.modules.pop(_module_name, None)
from nnInteractive.inference.inference_session import nnInteractiveInferenceSession
from nnInteractive.utils.crop import crop_and_pad_into_buffer, paste_tensor, crop_to_valid, pad_cropped
from nnInteractive.utils.bboxes import generate_bounding_boxes
from nnInteractive.utils.erosion_dilation import iterative_3x3_same_padding_pool3d
from nnInteractive.utils.rounding import round_to_nearest_odd
from acvl_utils.cropping_and_padding.bounding_boxes import crop_and_pad_nd

class JointLassoSession(nnInteractiveInferenceSession):
    """Adds one batch registration method; native public methods remain unchanged."""
    def set_joint_lassos(self, image_zyx: np.ndarray, merged_lasso_zyx: np.ndarray) -> None:
        if image_zyx.shape != merged_lasso_zyx.shape: raise ValueError("Image/lasso grid mismatch")
        self.set_image(zyx_to_xyz(image_zyx)[None].astype(np.float32, copy=False))
        self._finish_preprocessing_and_initialize_interactions()
        self.reset_interactions()
        self.new_interaction_centers = []
        self.new_interaction_zoom_out_factors = []
        self.interaction_decay = 1.0
        lasso = torch.from_numpy(zyx_to_xyz(merged_lasso_zyx).astype(np.float32, copy=False))
        # Reuse official crop alignment, but do not enqueue one item per lasso.
        from acvl_utils.cropping_and_padding.bounding_boxes import crop_and_pad_nd
        lasso = crop_and_pad_nd(lasso, self.preprocessed_props["bbox_used_for_cropping"])
        torch.maximum(self.interactions[-6], lasso.to(self.interactions.device), out=self.interactions[-6])
        if float(self.interactions[-6].max()) != 1.0: raise RuntimeError("Joint lasso lost unit strength")
        self._generic_add_patch_from_image(lasso)
        if len(self.new_interaction_centers) != 1: raise RuntimeError("Expected exactly one union-ROI queue item")

    def native_predict_joint(self) -> torch.Tensor:
        shape_xyz = tuple(self.original_image_shape[1:])
        self.set_target_buffer(torch.zeros(shape_xyz, dtype=torch.uint8))
        self._predict()
        return xyz_to_zyx(self.target_buffer)

    def begin_native_trajectory(self) -> None:
        """Allocate a fresh full-volume native output buffer for one episode."""
        shape_xyz = tuple(self.original_image_shape[1:])
        self.set_target_buffer(torch.zeros(shape_xyz, dtype=torch.uint8))

    def native_predict_queued(self) -> torch.Tensor:
        """Consume exactly the currently queued interaction and retain native state."""
        if self.target_buffer is None:
            self.begin_native_trajectory()
        if len(self.new_interaction_centers) != 1:
            raise RuntimeError("A Stage-2 native prediction requires exactly one queued center")
        self._predict()
        return xyz_to_zyx(self.target_buffer)


@dataclass(frozen=True)
class PredictionSnapshot:
    """Deep copy of every mutable field consumed by native ``_predict``.

    The snapshot is deliberately taken after an interaction has been registered
    but before its native prediction is consumed. It therefore represents S(T-).
    """
    interactions: torch.Tensor
    centers: tuple[tuple[int, ...], ...]
    zoom_factors: tuple[float, ...]
    target_buffer: torch.Tensor | np.ndarray | None
    has_positive_bbox: bool


def snapshot_prediction_state(session: JointLassoSession) -> PredictionSnapshot:
    if len(session.new_interaction_centers) != 1:
        raise RuntimeError("Snapshot requires exactly one queued prediction center")
    target = session.target_buffer
    if isinstance(target, torch.Tensor):
        target = target.clone()
    elif isinstance(target, np.ndarray):
        target = target.copy()
    return PredictionSnapshot(
        interactions=session.interactions.clone(),
        centers=tuple(tuple(int(v) for v in center) for center in session.new_interaction_centers),
        zoom_factors=tuple(float(v) for v in session.new_interaction_zoom_out_factors),
        target_buffer=target,
        has_positive_bbox=bool(session.has_positive_bbox),
    )


def _normal_detached(value: torch.Tensor) -> torch.Tensor:
    """Copy an inference-mode tensor into normal storage safe for backward."""
    value = value.detach()
    with torch.inference_mode(False):
        result = torch.empty(value.shape, device=value.device, dtype=value.dtype)
        result.copy_(value)
    if result.is_inference(): raise RuntimeError("Could not normalize inference tensor")
    return result

def isolated_replay_session(source: JointLassoSession, snapshot: PredictionSnapshot) -> JointLassoSession:
    """Create an independent mutable session state sharing only model/read-only data.

    This avoids temporarily restoring the real clinical trajectory session. The
    network object is intentionally shared so replay logits backpropagate to the
    model being optimized, whereas all replay writes remain isolated.
    """
    replay = JointLassoSession(
        device=source.device,
        do_autozoom=source.do_autozoom,
        verbose=source.verbose,
        use_pinned_memory=False,
    )
    # Loaded model/configuration and image preprocessing are read-only during
    # prediction. Mutable prediction fields are cloned below.
    for name in (
        "network", "label_manager", "dataset_json", "trainer_name",
        "configuration_manager", "plans_manager", "pad_mode_data",
        "preferred_scribble_thickness", "point_interaction", "preprocessed_image",
        "preprocessed_props", "original_image_shape", "interaction_decay",
    ):
        value = getattr(source, name)
        # Native preprocessing may materialize this tensor inside inference_mode.
        # The terminal replay must own normal tensor storage for autograd.
        if name == "preprocessed_image" and isinstance(value, torch.Tensor): value = _normal_detached(value)
        setattr(replay, name, value)
    replay.interactions = _normal_detached(snapshot.interactions)
    replay.new_interaction_centers = [list(center) for center in snapshot.centers]
    replay.new_interaction_zoom_out_factors = list(snapshot.zoom_factors)
    replay.has_positive_bbox = snapshot.has_positive_bbox
    target = snapshot.target_buffer
    replay.target_buffer = _normal_detached(target) if isinstance(target, torch.Tensor) else (target.copy() if isinstance(target, np.ndarray) else None)
    return replay

def _tensor_info(name, x):
    if os.environ.get("NNI_TENSOR_DEBUG") != "1": return
    if isinstance(x, torch.Tensor):
        print(name, "shape=", tuple(x.shape), "device=", x.device, "dtype=", x.dtype, "requires_grad=", x.requires_grad, "is_inference=", x.is_inference(), flush=True)

def _network_forward(session: JointLassoSession, network_input: torch.Tensor, *, activation_checkpointing: bool) -> torch.Tensor:
    _tensor_info("network_input", network_input)
    _tensor_info("preprocessed_image", session.preprocessed_image)
    _tensor_info("interactions", session.interactions)
    _tensor_info("target_buffer", session.target_buffer)
    assert not network_input.is_inference(), "terminal network_input is inference"
    handles=[]
    def make_hook(name):
        def hook(module, inputs):
            for i,x in enumerate(inputs):
                if isinstance(x,torch.Tensor) and x.is_inference(): raise RuntimeError(f"Inference tensor reached Conv3D: {name}, input={i}, shape={tuple(x.shape)}")
            if module.weight.is_inference(): raise RuntimeError(f"Inference Conv3D weight: {name}")
            if module.bias is not None and module.bias.is_inference(): raise RuntimeError(f"Inference Conv3D bias: {name}")
        return hook
    for name,module in session.network.named_modules():
        if isinstance(module,torch.nn.Conv3d): handles.append(module.register_forward_pre_hook(make_hook(name)))
    try:
        batched=network_input[None]; assert not batched.is_inference()
        return checkpoint(session.network,batched,use_reentrant=False)[0] if activation_checkpointing and torch.is_grad_enabled() else session.network(batched)[0]
    finally:
        for h in handles: h.remove()

def _build_input(session: JointLassoSession, center: list[int], zoom: float):
    scaled_size = [round(v * zoom) for v in session.configuration_manager.patch_size]
    bbox = [[c - p // 2, c + p // 2 + p % 2] for c, p in zip(center, scaled_size)]
    image, image_pad = crop_to_valid(session.preprocessed_image, bbox)
    interactions, interaction_pad = crop_to_valid(session.interactions, bbox)
    if image.is_inference() or interactions.is_inference():
        raise RuntimeError(f"Replay inference provenance: image={image.is_inference()} interactions={interactions.is_inference()}")
    image = image.to(session.device, non_blocking=True)
    interactions = interactions.to(session.device, non_blocking=True)
    if tuple(scaled_size) != tuple(session.configuration_manager.patch_size):
        tmp = pad_cropped(interactions, interaction_pad) if any(x for y in interaction_pad for x in y) else interactions
        pool = round_to_nearest_odd(zoom * 2 - 1)
        if pool > 1:
            for channel in range(3, 7):
                tmp[channel:channel+1] = iterative_3x3_same_padding_pool3d(tmp[None, channel:channel+1], pool)[0]
        interactions = F.interpolate(tmp[None], session.configuration_manager.patch_size, mode="area")[0]
        image = F.interpolate((pad_cropped(image, image_pad) if any(x for y in image_pad for x in y) else image)[None], session.configuration_manager.patch_size, mode="trilinear", align_corners=False)[0]
    else:
        image = pad_cropped(image, image_pad) if any(x for y in image_pad for x in y) else image
        interactions = pad_cropped(interactions, interaction_pad) if any(x for y in interaction_pad for x in y) else interactions
    return torch.cat((image, interactions)), scaled_size, bbox

def _build_fixed_grid_input(
    image_cxyz: torch.Tensor,
    interactions_cxyz: torch.Tensor,
    bbox: list[list[int]],
    device: torch.device,
) -> torch.Tensor:
    """Build one native-size, non-resized grid patch without mutating state."""
    image, image_pad = crop_to_valid(image_cxyz, bbox)
    interactions, interaction_pad = crop_to_valid(interactions_cxyz, bbox)
    image = pad_cropped(image, image_pad) if any(x for pair in image_pad for x in pair) else image
    interactions = pad_cropped(interactions, interaction_pad) if any(x for pair in interaction_pad for x in pair) else interactions
    if image.shape[1:] != interactions.shape[1:]:
        raise RuntimeError("Completion image/interaction patch grid mismatch")
    return torch.cat((image, interactions)).to(device, non_blocking=True)

def _original_grid_workspace(session: JointLassoSession) -> tuple[torch.Tensor, torch.Tensor]:
    """Embed the native cropped workspace in original coordinates.

    nnInteractive sees constant-zero padding outside its nonzero-image crop.
    The training-only completion workspace represents those same locations by
    zero-filled tensors, while retaining native-normalized image/interactions
    in the crop. It never modifies ``session.interactions``.
    """
    shape_xyz = tuple(session.original_image_shape[1:])
    bbox = session.preprocessed_props["bbox_used_for_cropping"]
    slices = tuple(slice(interval[0], interval[1]) for interval in bbox)
    image = torch.zeros((1, *shape_xyz), dtype=session.preprocessed_image.dtype, device=session.preprocessed_image.device)
    interactions = torch.zeros((session.interactions.shape[0], *shape_xyz), dtype=session.interactions.dtype, device=session.interactions.device)
    image[(slice(None), *slices)] = session.preprocessed_image
    interactions[(slice(None), *slices)] = session.interactions
    return image, interactions

def _complete_uncovered_fixed_grid(
    session: JointLassoSession,
    full: DifferentiableLogitBuffer,
    *,
    activation_checkpointing: bool,
    prompt_slices: list[int] | None,
) -> dict[str, int]:
    """Training-only 192^3 native-grid completion; native logits always win."""
    image, interactions = _original_grid_workspace(session)
    before = session.interactions.clone()
    patch_size = tuple(int(v) for v in session.configuration_manager.patch_size)
    shape_xyz = tuple(int(v) for v in full.coverage.shape)
    unprompted = torch.ones(shape_xyz, device=full.coverage.device, dtype=torch.bool)
    if prompt_slices:
        unprompted[:, :, torch.as_tensor(prompt_slices, device=unprompted.device, dtype=torch.long)] = False
    starts = [range(0, size, patch) for size, patch in zip(shape_xyz, patch_size)]
    stats = {"grid_tiles": 0, "completion_forwards": 0, "skipped_tiles": 0, "filled_voxels": 0}
    for anchor in itertools.product(*starts):
        bbox = [[int(start), int(start + patch)] for start, patch in zip(anchor, patch_size)]
        stats["grid_tiles"] += 1
        valid = tuple(slice(start, min(end, shape_xyz[axis])) for axis, (start, end) in enumerate(bbox))
        # A prompted slice is excluded from the loss. Do not execute a sweep
        # forward solely to fill an otherwise uncovered prompted voxel.
        if not bool(((~full.coverage[valid]) & unprompted[valid]).any()):
            stats["skipped_tiles"] += 1
            continue
        network_input = _build_fixed_grid_input(image, interactions, bbox, session.device)
        logits = _network_forward(session, network_input, activation_checkpointing=activation_checkpointing)
        stats["completion_forwards"] += 1
        stats["filled_voxels"] += full.fill_only_uncovered(logits, bbox)
    if not torch.equal(before, session.interactions):
        raise RuntimeError("Completion sweep mutated native interaction state")
    return stats

def differentiable_predict_joint(
    session: JointLassoSession,
    *,
    complete_coverage: bool = False,
    activation_checkpointing: bool = False,
    prompt_slices: list[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, list[list[list[int]]]]:
    """Return [2,Z,Y,X] logits, [Z,Y,X] coverage, and native-control patch trace.

    Patch-selection decisions deliberately use detached hard masks. Gradients flow
    only from final last-write raw logits to loss, matching native overwrite order.
    """
    if len(session.new_interaction_centers) != 1: raise RuntimeError("Joint union queue required")
    center, zoom = session.new_interaction_centers[-1], min(4.0, session.new_interaction_zoom_out_factors[-1])
    cropped = DifferentiableLogitBuffer(tuple(session.preprocessed_image.shape[1:]), session.device)
    stats = {"native_forwards": 0, "completion_forwards": 0, "grid_tiles": 0, "skipped_tiles": 0, "filled_voxels": 0}
    trace: list[list[list[int]]] = []
    while True:
        network_input, scaled_size, bbox = _build_input(session, center, zoom)
        logits = _network_forward(session, network_input, activation_checkpointing=activation_checkpointing)
        stats["native_forwards"] += 1
        cropped.overwrite_zoomed(logits, scaled_size, bbox)
        trace.append({'type':'coarse','center':[int(v) for v in center],'zoom':float(zoom),'bbox':bbox})
        # Native _predict compares the raw 192^3 argmax prediction against
        # the previous interaction state resampled into that same space.
        # Resizing the new prediction first changes AutoZoom decisions.
        hard_patch = logits.detach().argmax(0)
        # Match nnInteractiveInferenceSession._predict exactly: the hard
        # interaction buffer is a bare [X,Y,Z] tensor, so it requires the
        # generic crop-and-pad helper rather than the [C,X,Y,Z] helper.
        previous = crop_and_pad_nd(session.interactions[0], bbox)
        if tuple(previous.shape) != tuple(hard_patch.shape):
            previous = F.interpolate(previous[None, None].float(), size=hard_patch.shape, mode="nearest")[0, 0]
        changed = session._detect_change_at_border(hard_patch, previous)
        if not (changed and session.do_autozoom and zoom < 4): break
        zoom = min(4.0, zoom * 1.5)
    if zoom == 1:
        paste_tensor(session.interactions[0], hard_patch, bbox)
    else:
        # Exactly match native order: resize the final 192^3 hard prediction,
        # compare it with the *old* interaction state, then paste coarse state.
        hard_scaled = F.interpolate(hard_patch[None, None].float(), size=scaled_size, mode="trilinear")[0, 0]
        hard_scaled = (hard_scaled >= 0.5).to(torch.uint8)
        diff, _ = session._compute_diff_map(hard_scaled, session.interactions[0], bbox, scaled_size)
        paste_tensor(session.interactions[0], hard_scaled, bbox)
        refine_boxes = generate_bounding_boxes(diff, session.configuration_manager.patch_size, stride="auto", margin=(10,10,10), max_depth=3)
        if not refine_boxes:  # Native _refine_coarse fallback around the queued initialization.
            patch_size = session.configuration_manager.patch_size
            refine_boxes = [[[c - p // 2, c - p // 2 + p] for c, p in zip(center, patch_size)]]
        for refine_bbox in refine_boxes:
            patch = torch.zeros((8, *session.configuration_manager.patch_size), device=session.device)
            crop_and_pad_into_buffer(patch[0], refine_bbox, session.preprocessed_image[0])
            crop_and_pad_into_buffer(patch[1], refine_bbox, session.interactions[0])
            crop_and_pad_into_buffer(patch[2:], refine_bbox, session.interactions[1:])
            logits = _network_forward(session, patch, activation_checkpointing=activation_checkpointing)
            stats["native_forwards"] += 1
            cropped.overwrite(logits, refine_bbox)
            trace.append({'type':'refine','bbox':refine_bbox})
            paste_tensor(session.interactions[0], logits.detach().argmax(0).to(torch.uint8), refine_bbox)
    full = DifferentiableLogitBuffer(tuple(session.original_image_shape[1:]), session.device)
    full.overwrite(cropped.logits, session.preprocessed_props["bbox_used_for_cropping"])
    # Full buffer coverage is copied with the same crop offset as logits.
    b = session.preprocessed_props["bbox_used_for_cropping"]
    full.coverage[tuple(slice(max(0,x[0]), min(full.coverage.shape[i],x[1])) for i,x in enumerate(b))] = cropped.coverage
    stats["native_covered_voxels"] = int(full.coverage.sum().item())
    if complete_coverage:
        stats.update(_complete_uncovered_fixed_grid(
            session,
            full,
            activation_checkpointing=activation_checkpointing,
            prompt_slices=prompt_slices,
        ))
    stats["final_covered_voxels"] = int(full.coverage.sum().item())
    session.last_prediction_stats = stats
    return xyz_to_zyx(full.logits), xyz_to_zyx(full.coverage), trace
