#!/usr/bin/env python3
"""Phase 0 gate entrypoint.

Layers:
  1) preflight — validator / mutations / toy oracles
  2) closure — literature-derived numeric freeze + provenance + outer method

Exit 0 requires BOTH preflight and closure PASS.
Does NOT authorize Phase 1 by itself; Phase 1 starts only after tagged freeze.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

SPEC = Path(__file__).parent.resolve()
ROOT = SPEC.parent.resolve()
FROZEN = SPEC / "frozen_experiment.yaml"
FORBIDDEN_GRID_KEYS = (
    "inner_learning_rate_candidates",
    "outer_step_size_candidates",
    "k_steps_candidates",
    "hyperparameter_selection",
)
SYMBOLIC_PENDING = "selected_by_validation_protocol"


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(ROOT))
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def almost_eq(a: float, b: float, tol: float = 1e-15) -> bool:
    return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(b)))


def evaluate_closure() -> tuple[str, list[str]]:
    reasons: list[str] = []
    if not FROZEN.exists():
        return "FAIL", [f"missing {FROZEN.name}"]

    text = FROZEN.read_text()
    frozen = yaml.safe_load(text)
    learning = frozen.get("learning") or {}
    provenance = frozen.get("hyperparameter_provenance") or {}

    for key in FORBIDDEN_GRID_KEYS:
        if key in frozen:
            reasons.append(f"forbidden grid/selection key present: {key}")

    if SYMBOLIC_PENDING in text:
        reasons.append(f"forbidden symbolic value present: {SYMBOLIC_PENDING}")
    lower = text.lower()
    for bad in ("optimized hyperparameter", "best hyperparameter"):
        if bad in lower:
            reasons.append(f"forbidden claim language present: {bad}")

    for key in ("inner_learning_rate", "k_steps"):
        val = learning.get(key)
        if val == SYMBOLIC_PENDING or not is_number(val):
            reasons.append(f"learning.{key} must be numeric (got={val!r})")

    outer = learning.get("outer_optimizer") or {}
    if learning.get("outer_update_method") != "mrlco_first_order_mean_pseudogradient":
        reasons.append(
            f"outer_update_method must be mrlco_first_order_mean_pseudogradient "
            f"(got={learning.get('outer_update_method')!r})"
        )
    if outer.get("name") != "adam":
        reasons.append(f"outer_optimizer.name must be adam (got={outer.get('name')!r})")
    if not is_number(outer.get("learning_rate")):
        reasons.append(f"outer_optimizer.learning_rate must be numeric (got={outer.get('learning_rate')!r})")
    if "outer_step_size" in learning:
        reasons.append("learning.outer_step_size must not exist (Reptile alias forbidden)")

    # Exact frozen literature defaults
    if is_number(learning.get("inner_learning_rate")) and not almost_eq(learning["inner_learning_rate"], 5.0e-4):
        reasons.append(f"inner_learning_rate must be 5e-4 (got={learning.get('inner_learning_rate')})")
    if is_number(outer.get("learning_rate")) and not almost_eq(outer["learning_rate"], 5.0e-4):
        reasons.append(f"outer_optimizer.learning_rate must be 5e-4 (got={outer.get('learning_rate')})")
    if learning.get("k_steps") != 3:
        reasons.append(f"k_steps must be 3 (got={learning.get('k_steps')!r})")
    if learning.get("meta_batch_size_distributions") != 10:
        reasons.append(
            f"meta_batch_size_distributions must be 10 (got={learning.get('meta_batch_size_distributions')!r})"
        )
    if learning.get("support_trajectories_per_meta_task") != 20:
        reasons.append(
            "support_trajectories_per_meta_task must be 20 "
            f"(got={learning.get('support_trajectories_per_meta_task')!r})"
        )

    if provenance.get("policy") != "fixed_literature_derived_defaults":
        reasons.append(f"hyperparameter_provenance.policy mismatch (got={provenance.get('policy')!r})")
    if provenance.get("source_arxiv") != "2008.02033v5":
        reasons.append(f"hyperparameter_provenance.source_arxiv mismatch (got={provenance.get('source_arxiv')!r})")
    if provenance.get("optimization_claim") is not False:
        reasons.append("hyperparameter_provenance.optimization_claim must be false")
    if provenance.get("validation_tuning_performed") is not False:
        reasons.append("hyperparameter_provenance.validation_tuning_performed must be false")

    if reasons:
        return "FAIL", reasons
    return "PASS", []


def main() -> int:
    run(
        [
            sys.executable,
            str(SPEC / "manifest_validator.py"),
            "--manifest",
            str(SPEC / "dataset_manifest.jsonl"),
            "--margo-root",
            str(ROOT),
            "--mode",
            "final",
            "--split-policy",
            str(SPEC / "split_policy.json"),
            "--split-summary",
            str(SPEC / "split_summary.json"),
        ]
    )
    run([sys.executable, str(SPEC / "manifest_validator_mutation_tests.py")])
    run([sys.executable, str(SPEC / "toy_oracles" / "oracle_checker.py")])

    closure_status, closure_reasons = evaluate_closure()
    overall = "CLOSED" if closure_status == "PASS" else "IN PROGRESS"
    phase1 = "YES_AFTER_TAG" if closure_status == "PASS" else "NO"

    print(
        f"""
GATE SNAPSHOT
Phase 0 preflight checks:        PASS
Phase 0 closure gate:            {closure_status}
Data/Split gate:                 PASS
ADR-004:                         ACCEPTED
ADR-005:                         ACCEPTED (semantics; sim/encoder adoption Phase 1/2)
ADR-006:                         ACCEPTED (fixed literature-derived defaults)
Outer update method:             mrlco_first_order_mean_pseudogradient
Hyperparameter provenance:       fixed_literature_derived_defaults
Toy oracle strength:             HARDENED
Phase 0 overall:                 {overall}
Ready for Phase 1 behavior fix:  {phase1}
"""
    )
    if closure_reasons:
        print("closure blockers:")
        for r in closure_reasons:
            print(f"  - {r}")
        return 1

    print("Phase 0 closure: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
