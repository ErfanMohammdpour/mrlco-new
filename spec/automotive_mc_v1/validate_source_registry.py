#!/usr/bin/env python3
"""Admission invariants for the MARGO-AUTOMOTIVE-MC-v1 source registry.

Markdown rules regress; this enforces them. Exit code 1 on any violation.

    allowed_for_generation == true
        iff (source_verification in ACCEPTABLE and transcription_verified)
            or (source_verification == synthetic_calibrated
                and generation_rule_verified and generation_rule_id)

plus: a required row must carry every admission field; secondary_lead_unverified
is never allowed; a communication-scope requirement or any measured_latency may
never define the graph deadline.

Step 2 passes when blocked_required_parameters == 0, not when every reference-only
row is admissible.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

ACCEPTABLE = {"official_verified", "primary_verified", "peer_reviewed_verified"}
ADMISSION_FIELDS = (
    "source_verification",
    "transcription_verified",
    "measurement_scope",
    "semantic_role",
    "allowed_for_generation",
    "required_by_generator",
)
REGISTRY = Path(__file__).resolve().parent / "SOURCE_REGISTRY.yaml"


def expected_allowed(row: dict) -> bool:
    sv = row.get("source_verification")
    if sv in ACCEPTABLE:
        return bool(row.get("transcription_verified")) is True
    if sv == "synthetic_calibrated":
        return bool(row.get("generation_rule_verified")) and bool(row.get("generation_rule_id"))
    return False


def validate(doc: dict) -> tuple[list[str], dict]:
    violations: list[str] = []
    rows = [q for s in doc.get("sources", []) for q in (s.get("parameters") or [])]
    required = blocked_required = 0
    for row in rows:
        name = row.get("parameter_name", "?")
        is_required = row.get("required_by_generator") is True
        if is_required:
            required += 1
            for field in ADMISSION_FIELDS:
                if field not in row:
                    violations.append(f"{name}: required row missing {field}")
        sv = row.get("source_verification")
        if sv == "secondary_lead_unverified" and row.get("allowed_for_generation"):
            violations.append(f"{name}: secondary_lead_unverified must not be allowed")
        allowed = row.get("allowed_for_generation") is True
        if allowed != expected_allowed(row):
            violations.append(
                f"{name}: allowed_for_generation={allowed} contradicts the gate "
                f"(source_verification={sv!r}, transcription_verified="
                f"{row.get('transcription_verified')!r})"
            )
        if row.get("defines_D_G"):
            if row.get("semantic_role") == "measured_latency":
                violations.append(f"{name}: measured_latency may not define D_G")
            if row.get("semantic_role") == "requirement" and row.get("measurement_scope") == "communication":
                violations.append(f"{name}: communication requirement may not define D_G")
        if is_required and not allowed:
            blocked_required += 1
    stats = {
        "numeric_rows": len(rows),
        "required_by_generator": required,
        "blocked_required_parameters": blocked_required,
        "allowed_rows": sum(1 for r in rows if r.get("allowed_for_generation")),
        "violations": len(violations),
    }
    return violations, stats


def main() -> int:
    doc = yaml.safe_load(REGISTRY.read_text())
    violations, stats = validate(doc)
    print(json.dumps({"registry_schema": doc.get("schema_version"), **stats,
                      "gate": "PASS" if not violations and stats["blocked_required_parameters"] == 0 else "BLOCKED",
                      "details": violations}, indent=2, sort_keys=True))
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
