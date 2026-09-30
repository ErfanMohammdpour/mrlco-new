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

import hashlib
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
MANIFEST = Path(__file__).resolve().parent / "REQUIRED_PARAMETER_MANIFEST.yaml"
PIN = Path(__file__).resolve().parent / "registry_pin.json"
EVIDENCE = Path(__file__).resolve().parent / "evidence" / "TS122186_v160200_tables_excerpt.txt"
LEDGER_FIELDS = ("document_version", "section", "table", "requirement_id",
                 "column", "value", "unit", "evidence_sha256")
CATEGORIES = {"execution", "payload", "communication_budget",
              "communication_capacity", "compute_capacity", "application_e2e", "rule"}
BINDING_POLICIES = {"single", "use_case_conditioned", "all_must_be_allowed"}


def expected_allowed(row: dict) -> bool:
    sv = row.get("source_verification")
    if sv in ACCEPTABLE:
        return bool(row.get("transcription_verified")) is True
    if sv == "synthetic_calibrated":
        return bool(row.get("generation_rule_verified")) and bool(row.get("generation_rule_id"))
    return False


def validate_manifest(manifest: dict, registry_rows: dict[str, dict]) -> tuple[list[str], dict]:
    """Missing/blocked required parameters, including ones absent from the registry."""
    violations: list[str] = []
    missing = blocked = 0
    seen: set[str] = set()
    families = set(manifest.get("application_families") or [])
    covered: set[str] = set()
    for entry in manifest.get("required_parameters") or []:
        pid = entry.get("id", "?")
        if pid in seen:
            violations.append(f"manifest: duplicate required parameter id {pid}")
        seen.add(pid)
        if entry.get("category") not in CATEGORIES:
            violations.append(f"manifest: {pid} has unknown category {entry.get('category')!r}")
        binds = entry.get("registry_binding") or []
        policy = entry.get("binding_policy") or ("single" if len(binds) <= 1 else None)
        if len(binds) > 1:
            if policy not in BINDING_POLICIES:
                violations.append(
                    f"manifest: {pid} binds {len(binds)} rows without an explicit "
                    "binding_policy (an aggregated value must not satisfy every family by accident)"
                )
            elif policy == "use_case_conditioned":
                bmap = entry.get("binding_map") or {}
                mapped = [r for rows_ in bmap.values() for r in rows_]
                if sorted(mapped) != sorted(binds):
                    violations.append(
                        f"manifest: {pid} binding_map {sorted(mapped)} does not cover "
                        f"its bindings {sorted(binds)}"
                    )
                for use_case, rows_ in bmap.items():
                    for r in rows_:
                        if r not in registry_rows:
                            violations.append(f"manifest: {pid} binding_map[{use_case}] names unknown row {r}")
        present = [b for b in binds if b in registry_rows]
        if entry.get("requires_numeric"):
            if binds and not present:
                violations.append(f"manifest: {pid} binds unknown registry rows {binds}")
            if not binds:
                missing += 1                      # not registered yet: invisible before
            elif any(registry_rows[b].get("allowed_for_generation") is not True for b in present):
                blocked += 1
        if entry.get("category") == "application_e2e":
            covered.update(entry.get("required_for_families") or [])
    for family in sorted(families - covered):
        violations.append(f"manifest: application family {family} has no application_e2e required parameter")
    stats = {
        "required_parameter_entries": len(seen),
        "missing_required_parameters": missing,
        "blocked_required_parameters": blocked,
        "rule_only_parameters": sum(1 for e in (manifest.get("required_parameters") or [])
                                    if not e.get("requires_numeric")),
    }
    return violations, stats


def check_pin(doc_hash: str) -> tuple[list[str], dict]:
    """A stale pin means the registry changed without regenerating the pin."""
    if not PIN.exists():
        return ["registry_pin.json missing"], {"registry_pin_stale": True}
    pin = json.loads(PIN.read_text())
    recorded = pin.get("source_registry_sha256")
    stale = recorded != doc_hash
    return ([f"registry_pin_stale: pin {str(recorded)[:16]} != registry {doc_hash[:16]}"] if stale else [],
            {"registry_pin_stale": stale, "pinned_sha256": recorded})


def validate(doc: dict) -> tuple[list[str], dict]:
    violations: list[str] = []
    expected_evidence = (hashlib.sha256(EVIDENCE.read_bytes()).hexdigest()
                         if EVIDENCE.exists() else "")
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
        if row.get("transcription_verified") is True:
            led = row.get("transcription_ledger")
            if not led:
                violations.append(f"{name}: transcription_verified without an explicit ledger")
            else:
                for field in LEDGER_FIELDS:
                    if not led.get(field):
                        violations.append(f"{name}: ledger missing {field}")
                recorded = led.get("evidence_sha256")
                if recorded and expected_evidence and recorded != expected_evidence:
                    violations.append(
                        f"{name}: ledger references stale evidence {str(recorded)[:16]} "
                        f"!= current {expected_evidence[:16]}")
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
        "ledger_entries": sum(1 for r in rows if r.get("transcription_ledger")),
        "evidence_sha256_current": expected_evidence,
        "violations": len(violations),
    }
    return violations, stats


def main() -> int:
    doc = yaml.safe_load(REGISTRY.read_text())
    row_violations, stats = validate(doc)
    rows = {q["parameter_name"]: q for s in doc.get("sources", [])
            for q in (s.get("parameters") or [])}
    manifest = yaml.safe_load(MANIFEST.read_text()) if MANIFEST.exists() else {}
    man_violations, man_stats = validate_manifest(manifest, rows)
    pin_violations, pin_stats = check_pin(hashlib.sha256(REGISTRY.read_bytes()).hexdigest())
    violations = row_violations + man_violations + pin_violations
    passed = (not violations and man_stats["missing_required_parameters"] == 0
              and man_stats["blocked_required_parameters"] == 0
              and pin_stats.get("registry_pin_stale") is False)
    print(json.dumps({"registry_schema": doc.get("schema_version"), **stats, **man_stats,
                      **pin_stats, "gate": "PASS" if passed else "BLOCKED",
                      "details": violations}, indent=2, sort_keys=True))
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
