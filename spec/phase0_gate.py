#!/usr/bin/env python3
"""Phase 0 gate entrypoint.

Two layers:
  1) preflight — validator / mutations / toy oracles (must PASS for exit 0 today)
  2) closure — selection evidence + numeric frozen learning values (BLOCKED until present)

Exit 0 on preflight PASS does NOT mean Phase 0 is closed and does NOT authorize Phase 1.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

SPEC = Path(__file__).parent.resolve()
ROOT = SPEC.parent.resolve()
FROZEN = SPEC / "frozen_experiment.yaml"
SELECTION_EVIDENCE = SPEC / "hyperparameter_selection_evidence.json"
SYMBOLIC_PENDING = "selected_by_validation_protocol"


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(ROOT))
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def is_positive_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and float(value) > 0


def evaluate_closure() -> tuple[str, list[str]]:
    """Return (status, reasons). Status is PASS or BLOCKED_PENDING_SELECTION_EVIDENCE."""
    reasons: list[str] = []
    if not FROZEN.exists():
        return "BLOCKED_PENDING_SELECTION_EVIDENCE", [f"missing {FROZEN.name}"]

    frozen = yaml.safe_load(FROZEN.read_text())
    learning = frozen.get("learning") or {}
    for key in ("inner_learning_rate", "outer_step_size", "k_steps"):
        val = learning.get(key)
        if val == SYMBOLIC_PENDING or not is_positive_number(val):
            reasons.append(f"learning.{key} not numeric selected value (got={val!r})")

    if not SELECTION_EVIDENCE.exists():
        reasons.append(f"missing evidence artifact: {SELECTION_EVIDENCE.name}")
    else:
        # Lightweight shape check without importing jsonschema.
        import json

        try:
            evidence = json.loads(SELECTION_EVIDENCE.read_text())
        except Exception as exc:  # noqa: BLE001
            reasons.append(f"evidence artifact unreadable: {exc}")
            evidence = None
        if isinstance(evidence, dict):
            for key in (
                "winning_inner_learning_rate",
                "winning_outer_step_size",
                "winning_k_steps",
                "seed_metrics",
                "checkpoint_outer_iteration",
            ):
                if key not in evidence:
                    reasons.append(f"evidence missing key: {key}")

    if reasons:
        return "BLOCKED_PENDING_SELECTION_EVIDENCE", reasons
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

    print(
        f"""
GATE SNAPSHOT
Phase 0 preflight checks:        PASS
Phase 0 closure gate:            {closure_status}
Data/Split gate:                 PASS
ADR-004:                         ACCEPTED
ADR-005:                         ACCEPTED (semantics; sim/encoder adoption Phase 1/2)
Learning protocol form:          PASS
Reward attribution:              DEFINED (post_hoc_telescoping + all_UE)
Numeric selection evidence:      {"PASS" if closure_status == "PASS" else "MISSING"}
Toy oracle existence:            PASS
Toy oracle strength:             HARDENED (intervals/routes/resources/energy)
Phase 0 overall:                 {"CLOSED" if closure_status == "PASS" else "IN PROGRESS"}
Ready for Phase 1 behavior fix:  NO
"""
    )
    if closure_reasons:
        print("closure blockers:")
        for r in closure_reasons:
            print(f"  - {r}")
    # Preflight green => exit 0; closure remains informational until evidence exists.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
