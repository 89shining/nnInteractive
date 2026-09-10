"""Native-trajectory / isolated-terminal-replay loops for nnInteractive Stage-2."""
from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any
import numpy as np
import torch

from interaction_builder import joint_axial_lasso
from losses import unprompted_dice_ce, whole_volume_hard_dice
from point_clicker import CorrectionClick, next_error_click
from prediction_wrapper import (
    JointLassoSession,
    PredictionSnapshot,
    differentiable_predict_joint,
    isolated_replay_session,
    snapshot_prediction_state,
)


@dataclass
class EpisodeResult:
    loss: torch.Tensor
    budget: int
    realized_clicks: int
    clicks: list[CorrectionClick]
    coverage_fraction: float
    native_prediction: torch.Tensor | None


def _terminal_logits(
    source: JointLassoSession,
    snapshot: PredictionSnapshot,
    prompts: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Replay S(T-) in a distinct mutable session and return loss-only logits."""
    replay = isolated_replay_session(source, snapshot)
    try:
        return differentiable_predict_joint(
            replay,
            complete_coverage=True,
            activation_checkpointing=True,
            prompt_slices=prompts,
        )[:2]
    finally:
        # Executor owns no source state, but shutting it down avoids accumulating
        # idle worker threads across patient episodes.
        replay.executor.shutdown(wait=False, cancel_futures=True)


def train_episode(
    session: JointLassoSession,
    image_zyx: np.ndarray,
    target_zyx: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    prompt_slices: list[int],
    budget: int,
    rng: random.Random,
    device: torch.device,
) -> EpisodeResult:
    """One patient update: native intermediate trajectory, terminal replay only."""
    if budget < 0 or budget > 5:
        raise ValueError("Stage-2 correction budget must be in 0..5")
    prompts = sorted(set(map(int, prompt_slices)))
    session.set_joint_lassos(image_zyx, joint_axial_lasso(target_zyx, prompts))
    # S(0-) is queued by set_joint_lassos. T=0 intentionally does not perform
    # a wasteful native P0: it directly supervises its isolated raw-logit replay.
    terminal_snapshot = snapshot_prediction_state(session)
    clicks: list[CorrectionClick] = []
    native_prediction: torch.Tensor | None = None

    if budget > 0:
        session.begin_native_trajectory()
        # Native states exist solely for residual/click generation. Keep this
        # explicit even though the current official _predict is inference-mode.
        with torch.inference_mode():
            native_prediction = session.native_predict_queued()  # P0 native
        for _ in range(1, budget + 1):
            click = next_error_click(
                native_prediction.detach().cpu().numpy(), target_zyx, prompts,
                spacing_zyx, rng=rng,
            )
            # Exact P(t-1) has no error. The latest snapshot still reproduces
            # that state, so it is the correct supervised terminal instead of
            # inventing a zero-information correction point.
            if click is None:
                break
            session.add_point_interaction(click.xyz, include_interaction=click.positive, run_prediction=False)
            terminal_snapshot = snapshot_prediction_state(session)  # S(t-)
            with torch.inference_mode():
                native_prediction = session.native_predict_queued()  # Pt native
            clicks.append(click)

    logits, coverage = _terminal_logits(session, terminal_snapshot, prompts)
    loss_mask = torch.ones(target_zyx.shape, dtype=torch.bool, device=device)
    loss_mask[torch.as_tensor(prompts, device=device)] = False
    if not bool(coverage[loss_mask].all()):
        missing = int((~coverage[loss_mask]).sum().item())
        raise RuntimeError(f"Terminal completion failed: {missing} unprompted voxels uncovered")
    target = torch.from_numpy(target_zyx).to(device)
    loss = unprompted_dice_ce(logits, target, prompts)
    return EpisodeResult(
        loss=loss,
        budget=budget,
        realized_clicks=len(clicks),
        clicks=clicks,
        coverage_fraction=float(coverage.float().mean().item()),
        native_prediction=native_prediction,
    )


@torch.inference_mode()
def validate_case(
    session: JointLassoSession,
    image_zyx: np.ndarray,
    target_zyx: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    prompt_slices: list[int],
    device: torch.device,
) -> tuple[list[float], list[CorrectionClick]]:
    """Native K=3 workflow D0..D5 for one fixed placement."""
    prompts = sorted(set(map(int, prompt_slices)))
    session.set_joint_lassos(image_zyx, joint_axial_lasso(target_zyx, prompts))
    session.begin_native_trajectory()
    prediction = session.native_predict_queued()
    target = torch.from_numpy(target_zyx).to(device)
    scores = [whole_volume_hard_dice(prediction.to(device), target)]
    clicks: list[CorrectionClick] = []
    for _ in range(5):
        click = next_error_click(prediction.cpu().numpy(), target_zyx, prompts, spacing_zyx, rng=None)
        if click is None:
            scores.append(scores[-1])
            continue
        session.add_point_interaction(click.xyz, include_interaction=click.positive, run_prediction=False)
        prediction = session.native_predict_queued()
        clicks.append(click)
        scores.append(whole_volume_hard_dice(prediction.to(device), target))
    if len(scores) != 6:
        raise RuntimeError("Validation must emit D0..D5")
    return scores, clicks


def validate_fold(session: JointLassoSession, val_cases, plan: dict[str, Any], fold: int, device: torch.device):
    """K=3, two fixed Stage-1 placements; patient-average then cohort-average."""
    per_patient: list[np.ndarray] = []
    for index, case in enumerate(val_cases, 1):
        from dataset import load_case, patient_id, spacing_zyx as read_spacing
        image, target = load_case(case)
        record = plan["folds"][str(fold)][str(patient_id(case))]
        placements = record["placements"]["3"]
        if len(placements) != 2:
            raise RuntimeError(f"{case.name}: validation requires exactly two K=3 placements")
        scores = []
        for placement in placements:
            prompts = list(map(int, placement["prompt_frame_ids"]))
            one, _ = validate_case(session, image, target, read_spacing(case), prompts, device)
            scores.append(one)
        per_patient.append(np.mean(np.asarray(scores, dtype=np.float64), axis=0))
        print(f"[val fold{fold} patient {index}/{len(val_cases)}] {case.name}")
    mean = np.mean(np.asarray(per_patient, dtype=np.float64), axis=0)
    result = {f"D{i}": float(mean[i]) for i in range(6)}
    result["S_workflow"] = float(mean.mean())
    result["delta_D5"] = float(mean[5] - mean[0])
    return result
