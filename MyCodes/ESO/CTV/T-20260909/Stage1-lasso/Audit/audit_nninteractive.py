#!/usr/bin/env python
"""Static feasibility audit for the nnInteractive CTV Stage-1 protocol.

This program never trains a model and never writes to the data or model folders.
It creates NNINTERACTIVE_SOURCE_AUDIT.md and audit_result.json beside itself.
Formal training code never imports this script, but it requires the resulting
audit_result.json to record PASS with matching source/checkpoint fingerprints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable


@dataclass
class Finding:
    name: str
    status: str
    evidence: str


def require_text(source: str, needle: str, name: str, evidence: str) -> Finding:
    return Finding(name, "PASS" if needle in source else "BLOCKED", evidence)


def read_text(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.read_text(encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


TRAINING_CODE_FILES = (
    "coordinate_adapter.py",
    "dataset.py",
    "differentiable_buffer.py",
    "interaction_builder.py",
    "losses.py",
    "prediction_wrapper.py",
    "prompt_sampler.py",
    "train.py",
)


def training_code_manifest(stage1_dir: Path) -> dict[str, str]:
    missing = [name for name in TRAINING_CODE_FILES if not (stage1_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing Stage-1 training files: {missing}")
    return {name: sha256(stage1_dir / name) for name in TRAINING_CODE_FILES}


def git_commit(repo: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def semantic_pattern(source: str, pattern: str, name: str, evidence: str) -> Finding:
    return Finding(name, "PASS" if re.search(pattern, source, re.DOTALL) else "BLOCKED", evidence)


def markdown(
    findings: Iterable[Finding], repo: Path, model_dir: Path, fingerprints: dict[str, object]
) -> str:
    rows = "\n".join(
        f"| {f.name} | {f.status} | {f.evidence} |" for f in findings
    )
    overall = "PASS" if all(f.status == "PASS" for f in findings) else "BLOCKED"
    passed = [f for f in findings if f.status == "PASS"]
    blocked = [f for f in findings if f.status != "PASS"]
    confirmed = "\n".join(f"- **{f.name}**: {f.evidence}" for f in passed) or "- None"
    unconfirmed = "\n".join(f"- **{f.name}**: {f.evidence}" for f in blocked) or "- None"
    fingerprint_lines = "\n".join(
        f"- `{key}`: `{value}`" for key, value in fingerprints.items()
    )
    return f"""# nnInteractive CTV Stage-1 source audit

This report audits the source and the official model bundle only. It does not
run training, alter data, or claim full-volume coverage without a runtime probe.

- Source repository: `{repo}`
- Official model bundle: `{model_dir}`
- Static feasibility result: **{overall}**

| Check | Result | Evidence |
| --- | --- | --- |
{rows}

## Confirmed by this audit

{confirmed}

## Unconfirmed or blocked assumptions

{unconfirmed}

## Audit fingerprints

{fingerprint_lines}

## Go / No-Go rule

Formal training remains **BLOCKED** until a runtime audit on the target A6000
environment proves native/wrapper trajectory equivalence and full unprompted
coverage after the approved training-only fixed-grid completion sweep.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--runtime-report", type=Path, default=None)
    args = parser.parse_args()

    repo = args.repo.resolve()
    model_dir = args.model_dir.resolve()
    source_path = repo / "nnInteractive" / "inference" / "inference_session.py"
    crop_path = repo / "nnInteractive" / "utils" / "crop.py"
    readme_path = next(
        (candidate for candidate in (repo / "README.md", repo / "readme.md") if candidate.is_file()),
        None,
    )
    if readme_path is None:
        raise FileNotFoundError("Neither README.md nor readme.md exists in repository root")
    source = read_text(source_path)
    crop_source = read_text(crop_path)
    readme = read_text(readme_path)

    required_bundle = [
        model_dir / "dataset.json",
        model_dir / "plans.json",
        model_dir / "inference_session_class.json",
        model_dir / "fold_0" / "checkpoint_final.pth",
    ]
    runtime = None
    if args.runtime_report is not None:
        runtime = json.loads(args.runtime_report.read_text(encoding="utf-8"))
    runtime_pass = bool(
        runtime is not None
        and runtime.get("result") == "PASS"
        and runtime.get("complete") is True
        and isinstance(runtime.get("audited_fold"), int)
        and int(runtime.get("eligible_patients", 0)) > 0
        and int(runtime.get("expected_episodes", 0)) > 0
        and int(runtime.get("checked_episodes", -1)) == int(runtime.get("expected_episodes", -2))
        and int(runtime.get("expected_episodes", -1)) == int(runtime.get("eligible_patients", -2)) * 5 * 2
        and not runtime.get("coverage_or_trajectory_failures", [])
    )
    findings = [
        Finding(
            "official model bundle",
            "PASS" if all(path.is_file() for path in required_bundle) else "BLOCKED",
            "requires dataset.json, plans.json, inference_session_class.json, and fold_0/checkpoint_final.pth",
        ),
        semantic_pattern(
            readme,
            r"single slice.*closed contour",
            "official lasso representation",
            "README defines a lasso as a one-slice closed contour",
        ),
        semantic_pattern(
            source,
            r"def\s+add_lasso_interaction.*?interaction_channel\s*=\s*-6.*?torch\.maximum",
            "positive-lasso interaction channel",
            "add_lasso_interaction writes positive lasso to channel -6",
        ),
        semantic_pattern(
            source,
            r"self\.interactions\s*\[\s*-6\s*:\s*-4\s*\]\s*\*=\s*self\.interaction_decay",
            "native interaction decay",
            "batch wrapper may bypass sequential decay only after unit-strength verification",
        ),
        semantic_pattern(
            source,
            r"def\s+_add_patch_for_lasso_interaction.*?_generic_add_patch_from_image\s*\(\s*lasso_image\s*\)",
            "union ROI entry point",
            "native generic image helper derives queue initialization from nonzero extent",
        ),
        semantic_pattern(
            source,
            r"new_interaction_centers\s*\[\s*-1\s*\]",
            "last-center default risk",
            "joint wrapper must enqueue one union item, not K native entries",
        ),
        semantic_pattern(
            source,
            r"@torch\.inference_mode\(\)\s*def\s+_predict",
            "native inference-only path",
            "validation may use _predict; differentiable training requires a wrapper",
        ),
        semantic_pattern(
            source,
            r"self\.network\s*\(\s*input_for_predict\s*\[\s*None\s*\]\s*\)\s*\[\s*0\s*\]\.argmax\s*\(",
            "raw-logit interception point",
            "training wrapper must retain the two-class tensor before argmax",
        ),
        semantic_pattern(
            crop_source,
            r"target_tensor\s*\[\s*tuple\s*\(\s*target_slices\s*\)\s*\]\s*=\s*sub_source",
            "overwrite write-back semantics",
            "native spatial policy is last-write-wins",
        ),
        Finding(
            "runtime full-supervision coverage",
            "PASS" if runtime_pass else "BLOCKED",
            "A6000 runtime report confirms completion coverage for every audited unprompted voxel"
            if runtime_pass else "requires a passing A6000 completion-coverage runtime report",
        ),
        Finding(
            "training/validation patch-sequence equivalence",
            "PASS" if runtime_pass else "BLOCKED",
            "A6000 runtime report found no native/wrapper trace mismatch"
            if runtime_pass else "requires a passing A6000 native/wrapper trace comparison",
        ),
    ]

    checkpoint = model_dir / "fold_0" / "checkpoint_final.pth"
    dataset_json = model_dir / "dataset.json"
    plans_json = model_dir / "plans.json"
    session_class_json = model_dir / "inference_session_class.json"
    fingerprints = {
        "repository_git_commit": git_commit(repo),
        "audit_script_sha256": sha256(Path(__file__).resolve()),
        "inference_session_py_sha256": sha256(source_path),
        "crop_py_sha256": sha256(crop_path),
        "model_checkpoint_sha256": sha256(checkpoint) if checkpoint.is_file() else None,
        "model_checkpoint_size_bytes": checkpoint.stat().st_size if checkpoint.is_file() else None,
        "model_checkpoint_mtime_ns": checkpoint.stat().st_mtime_ns if checkpoint.is_file() else None,
        "dataset_json_sha256": sha256(dataset_json) if dataset_json.is_file() else None,
        "plans_json_sha256": sha256(plans_json) if plans_json.is_file() else None,
        "inference_session_class_json_sha256": sha256(session_class_json) if session_class_json.is_file() else None,
        "runtime_report_sha256": sha256(args.runtime_report) if args.runtime_report is not None else None,
        "training_code_sha256": training_code_manifest(Path(__file__).resolve().parents[1]),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = markdown(findings, repo, model_dir, fingerprints)
    (args.output_dir / "NNINTERACTIVE_SOURCE_AUDIT.md").write_text(report, encoding="utf-8")
    overall = "PASS" if all(f.status == "PASS" for f in findings) else "BLOCKED"
    payload = {
        "result": overall,
        "repo": str(repo),
        "model_dir": str(model_dir),
        "audited_fold": runtime.get("audited_fold") if runtime is not None else None,
        "fingerprints": fingerprints,
        "findings": [asdict(f) for f in findings],
    }
    (args.output_dir / "audit_result.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Audit result: {overall}")
    print(args.output_dir / "NNINTERACTIVE_SOURCE_AUDIT.md")


if __name__ == "__main__":
    main()
