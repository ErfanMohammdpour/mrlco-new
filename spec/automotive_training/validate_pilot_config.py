#!/usr/bin/env python3
"""Enforce that the 500-iteration pilot differs from the frozen primary config in
NOTHING except the run block, the method id and the schema/status tags.

Exit != 0 on any other difference, so a schedule cannot be silently compressed by
editing the pilot file.

CLI: python3 spec/automotive_training/validate_pilot_config.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

BASE = Path(__file__).resolve().parent
PRIMARY = BASE / "frozen_automotive_primary.yaml"
PILOT = BASE / "frozen_automotive_pilot_500.yaml"

#: keys that are ALLOWED to differ between the frozen primary and the pilot
ALLOWED_DIFFS = ("schema_version", "status", "method_id", "run", "learning")


def _flatten(obj, prefix=""):
    out = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            out.update(_flatten(value, "%s.%s" % (prefix, key) if prefix else str(key)))
    elif isinstance(obj, list):
        out[prefix] = json.dumps(obj, sort_keys=True)
    else:
        out[prefix] = obj
    return out


def main() -> int:
    primary = yaml.safe_load(PRIMARY.read_text())
    pilot = yaml.safe_load(PILOT.read_text())
    a, b = _flatten(primary), _flatten(pilot)
    violations = []
    for key in sorted(set(a) | set(b)):
        if a.get(key, "<missing>") != b.get(key, "<missing>"):
            top = key.split(".")[0]
            allowed = top in ALLOWED_DIFFS
            if top == "learning" and key != "learning.outer_iterations" and not key.endswith(
                    "outer_iterations_note"):
                allowed = False
            if not allowed:
                violations.append("%s: primary=%r pilot=%r" % (key, a.get(key), b.get(key)))
    run = pilot.get("run") or {}
    checks = {
        "pilot_run_kind": run.get("kind") == "primary_500",
        "pilot_outer_iterations": int(run.get("outer_iterations", 0)) == 500,
        "pilot_seeds": list(run.get("seeds", [])) == [0, 1, 2, 3, 4],
        "pilot_validation_cadence": int(run.get("validation_cadence", 0)) == 50,
        "learning_outer_iterations": int(pilot["learning"].get("outer_iterations", 0)) == 500,
        "meta_batch_unchanged": int(pilot["learning"]["meta_batch_size"]) == 10,
        "support_unchanged": int(pilot["learning"]["support_trajectories_per_meta_task"]) == 20,
        "tokens_unchanged": int(pilot["learning"]["tokens_per_trajectory"]) == 20,
        "inner_steps_unchanged": int(pilot["learning"]["inner_ppo_apply_steps"]) == 3,
    }
    for name, ok in checks.items():
        if not ok:
            violations.append("check failed: %s" % name)
    report = {"status": "PASS" if not violations else "FAIL",
              "allowed_differences": list(ALLOWED_DIFFS),
              "unexpected_differences": violations,
              "checks": checks,
              "pilot_sha256": __import__("hashlib").sha256(PILOT.read_bytes()).hexdigest(),
              "primary_sha256": __import__("hashlib").sha256(PRIMARY.read_bytes()).hexdigest()}
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
