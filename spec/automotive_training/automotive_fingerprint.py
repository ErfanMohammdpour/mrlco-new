#!/usr/bin/env python3
"""Canonical training fingerprint for the automotive primary.

Changing ANY component below must change the fingerprint. The fingerprint is written
into the run directory so a later meta-test evaluation can prove which contract the
checkpoint was produced under.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC = REPO_ROOT / "spec" / "automotive_training"
DATASET = REPO_ROOT / "env" / "mec_offloaing_envs" / "data" / "automotive_mc_v1"

#: files whose bytes define the code side of the fingerprint
CODE_FILES = (
    "automotive_loader.py",
    "automotive_env.py",
    "automotive_primary.py",
    "automotive_trainer.py",
    "automotive_dag.py",
    "automotive_resources.py",
    "automotive_constraints.py",
    "mc_runtime.py",
    "heft_reference_v2.py",
    "execution_uncertainty_v1.yaml",
    "constraints_automotive_v1.yaml",
    "frozen_automotive_primary.yaml",
)
SCHEDULER_FILES = (
    "env/mec_offloaing_envs/scheduler/encoder_obs.py",
    "env/mec_offloaing_envs/scheduler/model.py",
    "env/mec_offloaing_envs/scheduler/adapter.py",
    "env/mec_offloaing_envs/scheduler/engine.py",
    "env/mec_offloaing_envs/scheduler/resources.py",
    "env/mec_offloaing_envs/scheduler/energy_model.py",
    "env/mec_offloaing_envs/scheduler/radio.py",
)
CONFIG_FILES = (
    "spec/frozen_experiment.yaml",
    "spec/automotive_mc_v1/workload_model_v2.yaml",
    "spec/automotive_mc_v1/sla_registry_v2.yaml",
    "spec/automotive_mc_v1/criticality_policy.yaml",
    "spec/automotive_mc_v1/deadline_model_v2.py",
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git_sha() -> str:
    """Resolve HEAD without requiring the git binary (containers often lack it).

    A launcher running on the host can inject `MARGO_GIT_SHA`, which takes precedence
    because it is the revision of the tree the container actually mounted.
    """
    import os
    import subprocess

    injected = os.environ.get("MARGO_GIT_SHA")
    if injected:
        return str(injected).strip()

    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                             capture_output=True, text=True, check=True).stdout.strip()
        if out:
            return out
    except Exception:
        pass
    try:  # read .git/HEAD directly
        head = (REPO_ROOT / ".git" / "HEAD").read_text().strip()
        if head.startswith("ref:"):
            ref = REPO_ROOT / ".git" / head.split(":", 1)[1].strip()
            if ref.exists():
                return ref.read_text().strip()
            packed = REPO_ROOT / ".git" / "packed-refs"
            if packed.exists():
                name = head.split(":", 1)[1].strip()
                for line in packed.read_text().splitlines():
                    parts = line.split()
                    if len(parts) == 2 and parts[1] == name:
                        return parts[0]
            return "unknown_ref_%s" % head.split("/")[-1]
        return head
    except Exception:
        return "unknown"


def _code_dirty():
    """True/False when git (or the launcher) can tell, otherwise an explicit marker."""
    import os
    import subprocess

    injected = os.environ.get("MARGO_CODE_DIRTY")
    if injected is not None and str(injected).strip() != "":
        return str(injected).strip().lower() in ("1", "true", "yes", "dirty")

    try:
        out = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT,
                             capture_output=True, text=True, check=True).stdout
        return bool(out.strip())
    except Exception:
        return "unknown_no_git"


def training_fingerprint(*, obs_version: str, scheduler_axes: Mapping[str, str],
                         checkpoint_rule_sha: str, sampler_budget: Mapping[str, Any],
                         run_kind: str = "primary", outer_iterations: int | None = None,
                         evaluation_protocol_sha: str | None = None,
                         decoding: str | None = None,
                         dataset_dir: Path | None = None) -> dict:
    d = Path(dataset_dir) if dataset_dir else DATASET
    provenance = json.loads((d / "provenance.json").read_text())
    calibration = json.loads((d / "calibration_report.json").read_text())
    parts = {
        "git_sha": _git_sha(),
        "code_dirty": _code_dirty(),
        "dataset_version": provenance.get("dataset_version"),
        "dataset_manifest_sha": provenance.get("dataset_manifest_sha"),
        "graphs_sha": provenance.get("graphs_sha"),
        "splits_sha": provenance.get("splits_sha"),
        "split_policy_sha": provenance.get("split_policy_sha"),
        "calibration_report_sha": provenance.get("calibration_report_sha"),
        "provenance_sha": sha256_file(d / "provenance.json"),
        "certification_manifest_sha": provenance.get("certification_manifest_sha"),
        "calibration_source": calibration.get("calibration_source"),
        "calibration_parameters": calibration.get("parameters"),
        "obs_version": str(obs_version),
        "scheduler_axes": dict(scheduler_axes),
        "checkpoint_rule_sha": str(checkpoint_rule_sha),
        "sampler_budget": dict(sampler_budget),
        "run_kind": str(run_kind),
        "outer_iterations": (None if outer_iterations is None else int(outer_iterations)),
        "evaluation_protocol_sha": evaluation_protocol_sha,
        "decoding": decoding,
        "energy_constraint": "not_configured",
        "deadline_mask": "off",
        "objective_mode": "latency_only",
        "code_sha256": {name: sha256_file(SPEC / name) for name in CODE_FILES},
        "scheduler_code_sha256": {name: sha256_file(REPO_ROOT / name) for name in SCHEDULER_FILES},
        "config_sha256": {name: sha256_file(REPO_ROOT / name) for name in CONFIG_FILES},
    }
    digest = hashlib.sha256(json.dumps(parts, sort_keys=True,
                                       separators=(",", ":")).encode()).hexdigest()
    return {"fingerprint": digest, "canonical_sha256": digest, "parts": parts}


def write_fingerprint(run_dir: Path, **kwargs) -> dict:
    out = training_fingerprint(**kwargs)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "training_fingerprint.json").write_text(
        json.dumps(out, indent=2, sort_keys=True) + "\n")
    return out
