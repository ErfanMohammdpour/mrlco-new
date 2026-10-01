#!/usr/bin/env python3
"""M5 validator: workload, payload and resource models. Exit != 0 on violation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

BASE = Path(__file__).resolve().parent
WORKLOAD = BASE / "workload_model.yaml"
RESOURCES = BASE / "resource_profiles.yaml"
SEMANTICS = BASE / "task_semantics.yaml"
TEMPLATES = BASE / "application_templates.yaml"
CLASSES = ["camera", "lidar", "radar", "feature_tensor", "object_list",
           "trajectory", "control", "v2x_message"]
EVIDENCE_CLASSES = {"measured_cpu_anchor", "derived_from_cpu_anchor", "source_calibrated_synthetic"}
GPU_MARKERS = ("ros2", "jetson", "gpu", "accelerator")
THREE_GPP_MBPS = {10.0, 25.0, 30.0, 700.0}


def validate_workload(doc: dict) -> tuple[list[str], dict]:
    v: list[str] = []
    ref = doc.get("reference_compute_model") or {}
    for field in ("formula", "f_ref_hz", "reference_tier", "xi_cycles_per_bit",
                  "reference_timing_statistic", "C_ref_definition", "caveats"):
        if not ref.get(field):
            v.append(f"reference_compute_model: missing {field}")
    if float(ref.get("f_ref_hz", 0)) <= 0:
        v.append("f_ref_hz must be positive")
    if float(ref.get("xi_cycles_per_bit", 0)) <= 0:
        v.append("xi must be positive")
    statistic = ref.get("reference_timing_statistic")
    rules = doc.get("task_class_rules") or []
    anchors = 0
    calibrated = 0
    for rule in rules:
        rid = rule.get("class_id", "?")
        ev = rule.get("evidence_class")
        if ev not in EVIDENCE_CLASSES:
            v.append(f"rule {rid}: unknown evidence_class {ev!r}")
        if ev == "measured_cpu_anchor":
            anchors += 1
            src = str(rule.get("source_row") or "").lower()
            if any(m in src for m in GPU_MARKERS):
                v.append(f"rule {rid}: GPU/accelerator row used as a CPU anchor")
            if not rule.get("t_ref_s"):
                v.append(f"rule {rid}: measured CPU anchor needs t_ref_s")
        else:
            calibrated += 1
            if not rule.get("t_ref_s_range"):
                v.append(f"rule {rid}: calibrated rule needs t_ref_s_range")
        if rule.get("t_ref_statistic") != statistic and not rule.get("statistic_rule_id"):
            v.append(f"rule {rid}: mixes timing statistic without an explicit statistic_rule_id")
        if not rule.get("rule_id") or not rule.get("applies_to_roles"):
            v.append(f"rule {rid}: missing rule_id or applies_to_roles")
    payloads = doc.get("payload_models") or {}
    for cls in CLASSES:
        model = payloads.get(cls)
        if not model:
            v.append(f"payload model missing for class {cls!r}")
            continue
        for field in ("formula", "units", "rule_id", "evidence_class", "formula_kind"):
            if not model.get(field):
                v.append(f"payload {cls}: missing {field}")
        if "W_i" in str(model.get("formula")) or "compute_workload" in str(model.get("variables")):
            v.append(f"payload {cls}: must not depend on compute demand W_i")
        if cls == "camera" and model.get("formula_kind") == "bits_per_pixel":
            if "* C *" in str(model.get("formula")) or "channels" in str(model.get("variables")):
                v.append("payload camera: channels double-counted with bits_per_pixel")
    if "raw_sensor" in payloads:
        v.append("a generic raw_sensor payload model is forbidden")
    return v, {"workload_rules": len(rules), "cpu_anchors": anchors,
               "calibrated_classes": calibrated, "payload_models": len(payloads)}


def validate_resources(doc: dict) -> tuple[list[str], dict]:
    v: list[str] = []
    if doc.get("scheduler_uses") != "achievable_capacity":
        v.append("scheduler_uses must be achievable_capacity")
    profiles = doc.get("profiles") or {}
    if len(profiles) < 3:
        v.append("expected at least degraded/nominal/strong profiles")
    required = ("f_ue_hz", "f_helper_hz", "f_mec_hz",
                "r_mec_ul_bps", "r_mec_dl_bps", "r_v2v_bps")
    for name, prof in profiles.items():
        if not prof.get("rule_id") or not prof.get("evidence_class"):
            v.append(f"profile {name}: missing rule_id/evidence_class")
        for field in required:
            entry = prof.get(field)
            if not entry:
                v.append(f"profile {name}: missing {field}")
                continue
            if not entry.get("units"):
                v.append(f"profile {name}/{field}: missing units")
            rng = entry.get("range")
            if not rng or len(rng) != 2 or float(rng[0]) <= 0 or float(rng[1]) < float(rng[0]):
                v.append(f"profile {name}/{field}: invalid range {rng!r}")
        for forbidden_field in ("service_required_rate", "offered_traffic"):
            if forbidden_field in prof:
                v.append(f"profile {name}: {forbidden_field} must not be a scheduler input")
        for field in ("r_mec_ul_bps", "r_mec_dl_bps", "r_v2v_bps"):
            val = ((prof.get(field) or {}).get("range") or [None])[0]
            if val is not None and float(val) / 1e6 in THREE_GPP_MBPS and not prof.get("rule_id"):
                v.append(f"profile {name}: 3GPP service rate copied into capacity without a rule")
    return v, {"profiles": len(profiles)}


def cross_check_m4(workload: dict, semantics: dict, templates: dict) -> tuple[list[str], dict]:
    v: list[str] = []
    expected_workload = (workload.get("reference_compute_model") or {}).get("schema_version", "workload_model_v1")
    default_wl = (semantics.get("defaults") or {}).get("workload_model_ref", "workload_model_v1")
    if default_wl != "workload_model_v1":
        v.append(f"task_semantics workload_model_ref {default_wl!r} does not resolve to an M5 model")
    payloads = workload.get("payload_models") or {}
    known_models = set(payloads) | {m.get("model_ref") for m in payloads.values() if m.get("model_ref")}
    for tpl in (templates.get("templates") or []):
        for edge in tpl.get("edges") or []:
            ref = edge.get("payload_model_ref")
            cls = edge.get("payload_class")
            if ref not in known_models and ref != f"payload_{cls}_v1":
                v.append(f"{tpl.get('template_id')}: payload_model_ref {ref!r} has no M5 model")
    roles = {r["semantic_role"] for r in (semantics.get("roles") or [])}
    covered = {r for rule in (workload.get("task_class_rules") or [])
               for r in (rule.get("applies_to_roles") or [])}
    for role in sorted(roles - covered):
        v.append(f"role {role!r} has no workload rule")
    return v, {"roles_covered": len(roles & covered), "roles_total": len(roles),
               "expected_workload_ref": expected_workload}


def main() -> int:
    workload = yaml.safe_load(WORKLOAD.read_text())
    resources = yaml.safe_load(RESOURCES.read_text())
    semantics = yaml.safe_load(SEMANTICS.read_text())
    templates = yaml.safe_load(TEMPLATES.read_text()) if TEMPLATES.exists() else {}
    wv, ws = validate_workload(workload)
    rv, rs = validate_resources(resources)
    cv, cs = cross_check_m4(workload, semantics, templates)
    violations = wv + rv + cv
    print(json.dumps({**ws, **rs, **cs,
                      "status": "PASS" if not violations else "FAIL",
                      "violations": violations}, indent=2, sort_keys=True))
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
