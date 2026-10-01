#!/usr/bin/env python3
"""M7 validator: semantic criticality, execution-demand budgets, LO/HI modes.

Exit != 0 on any violation. Writes `criticality_report.json`.

The report is also the immutability witness: it pins the sha256 of every M4/M5-v2/
M6-v2 artifact and reruns the M6-v2 deadline construction to prove the deadline
hashes did not move while M7 was applied.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import yaml

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

M5_M6_ARTIFACTS = (
    "workload_model_v2.yaml",
    "sla_registry_v2.yaml",
    "resource_profiles.yaml",
    "application_templates.yaml",
    "task_semantics.yaml",
    "registry_pin.json",
    "deadline_report_v2.json",
    "REQUIRED_PARAMETER_MANIFEST.yaml",
    "SOURCE_REGISTRY.yaml",
)
MUTATION_TOKENS = (
    "open(", "write_text(", "write_bytes(", "yaml.dump(", "json.dump(",
    "os.remove", "os.rename", "shutil.", "unlink(",
)


def _load(name):
    spec = importlib.util.spec_from_file_location(name, BASE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot() -> dict:
    return {name: sha256_file(BASE / name) for name in M5_M6_ARTIFACTS}


def _quartile_buckets(values: list[float]) -> list[str]:
    order = sorted(range(len(values)), key=lambda i: (values[i], i))
    n = len(order)
    out = [""] * n
    for rank, idx in enumerate(order):
        q = min(3, int(4 * rank / max(1, n)))
        out[idx] = f"urgency_q{q + 1}"
    return out


def build_report() -> dict:
    cm = _load("criticality_model")
    dm2 = _load("deadline_model_v2")

    policy = yaml.safe_load((BASE / "criticality_policy.yaml").read_text())
    semantics = yaml.safe_load((BASE / "task_semantics.yaml").read_text())
    w2 = yaml.safe_load((BASE / "workload_model_v2.yaml").read_text())
    s2 = yaml.safe_load((BASE / "sla_registry_v2.yaml").read_text())
    templates = yaml.safe_load((BASE / "application_templates.yaml").read_text())["templates"]
    pinned_m6 = json.loads((BASE / "deadline_report_v2.json").read_text())
    pinned_by_family = {r["family_id"]: r["output_sha256_v2"] for r in pinned_m6["rows"]}

    before = snapshot()
    violations: list[str] = []
    rows: list[dict] = []
    depth_probe: list[dict] = []
    urgency_probe: list[dict] = []
    role_policy: dict[str, dict] = {}
    traces: dict[str, dict] = {}
    deadline_hashes_after: dict[str, str] = {}

    # ---- static immutability proof: the M7 model has no writer ---------------
    model_src = (BASE / "criticality_model.py").read_text()
    for token in MUTATION_TOKENS:
        if token in model_src:
            violations.append(f"criticality_model.py contains a mutation token {token!r}")

    # ---- classify every template + budgets + associations --------------------
    for tpl in templates:
        fam = tpl["family_id"]
        table = cm.build_task_table(policy, w2, tpl)
        mixture = cm.criticality_mixture(table)
        if sum(mixture.values()) != 20:
            violations.append(f"{fam}: criticality mixture covers {sum(mixture.values())} != 20 tasks")
        scope = (policy.get("safety_scope") or {}).get(fam)
        if scope == "safety" and mixture["HIGH"] == 0:
            violations.append(f"{fam}: safety family has no HIGH task")
        for tid, row in table.items():
            if row["empirical_execution_budget_lo_s"] <= 0:
                violations.append(f"{fam}/{tid}: C_LO must be > 0")
            if row["criticality"] == "HIGH":
                if not row["budget_hi_applicable"]:
                    violations.append(f"{fam}/{tid}: HIGH task without an HI budget")
                if row["empirical_execution_budget_hi_s"] < row["empirical_execution_budget_lo_s"]:
                    violations.append(f"{fam}/{tid}: C_HI < C_LO")
            else:
                if row["budget_hi_applicable"] or row["empirical_execution_budget_hi_s"] is not None:
                    violations.append(f"{fam}/{tid}: {row['criticality']} must report the HI budget as not_applicable")
            for mode in ("LO", "HI"):
                dg = row[f"drop_degrade_{mode.lower()}"]
                if row["criticality"] == "HIGH" and (dg["drop_allowed"] or dg["degrade_allowed"]):
                    violations.append(f"{fam}/{tid}: HIGH may never be dropped or degraded ({mode})")

        # M6 deadline hash must be byte-identical to the pinned report
        res = dm2.generate_deadlines_v2(tpl, w2, s2, fam)
        deadline_hashes_after[fam] = res.output_sha256
        if pinned_by_family.get(fam) != res.output_sha256:
            violations.append(f"{fam}: M6-v2 deadline hash moved ({pinned_by_family.get(fam)} -> {res.output_sha256})")
        if res.status != "ok":
            violations.append(f"{fam}: M6-v2 deadline status is {res.status}")

        # position/tightness probes (diagnostics only, never inputs)
        depth = cm.topological_depth(tpl)
        slacks = {tid: res.tasks[tid].slack for tid in sorted(table)}
        urgency = _quartile_buckets([slacks[t] for t in sorted(table)])
        for i, tid in enumerate(sorted(table)):
            row = table[tid]
            depth_probe.append({"family_id": fam, "task_id": tid,
                                "criticality": row["criticality"], "depth": str(depth[tid])})
            urgency_probe.append({"family_id": fam, "task_id": tid,
                                  "criticality": row["criticality"], "urgency": urgency[i]})
            role_policy[f"{fam}/{row['semantic_role']}"] = {
                "criticality": row["criticality"],
                "baseline_class": row["baseline_class"],
                "motif_role": row["motif_role"],
                "motif_floor_class": row["motif_floor_class"],
                "source_rule": row["source_rule"],
                "budget_rule_id": row["budget_rule_id"],
                "budget_evidence_class": row["budget_evidence_class"],
            }
        rows.append({
            "family_id": fam,
            "template_id": tpl["template_id"],
            "safety_scope": scope,
            "criticality_mixture": mixture,
            "high_share": round(mixture["HIGH"] / 20.0, 4),
            "budget_lo_min_s": min(r["empirical_execution_budget_lo_s"] for r in table.values()),
            "budget_hi_max_s": max((r["empirical_execution_budget_hi_s"] for r in table.values()
                                    if r["empirical_execution_budget_hi_s"] is not None),
                                   default=None),
            "nominal_hi_overrun_tasks": sorted(
                tid for tid, r in table.items()
                if r["criticality"] == "HIGH"
                and (dm2.task_reference_durations(tpl, w2)[tid][1]
                     > r["empirical_execution_budget_lo_s"] + 1e-15)
            ),
            "deadline_output_sha256": res.output_sha256,
        })

    depth_assoc = cm.association(depth_probe, "depth")
    urgency_assoc = cm.association(urgency_probe, "urgency")
    if depth_assoc["criticality_is_a_function_of_probe"]:
        violations.append("criticality is a deterministic function of topological depth")
    if urgency_assoc["criticality_is_a_function_of_probe"]:
        violations.append("criticality is a deterministic function of deadline tightness")
    if depth_assoc["levels_with_mixed_classes"] == 0:
        violations.append("no depth level carries mixed classes: the audit table is degenerate")
    if urgency_assoc["levels_with_mixed_classes"] == 0:
        violations.append("no urgency level carries mixed classes: the audit table is degenerate")

    # ---- mutation proofs: forbidden inputs must not move the class ----------
    fam0 = templates[0]["family_id"]
    base_table = cm.build_task_table(policy, w2, templates[0])
    base_classes = {t: r["criticality"] for t, r in base_table.items()}

    relabeled = json.loads(json.dumps(templates[0]))
    swap = {i: 19 - i for i in range(20)}
    for t in relabeled["tasks"]:
        t["task_id"] = swap[t["task_id"]]
    for e in relabeled["edges"]:
        e["src"], e["dst"] = swap[e["src"]], swap[e["dst"]]
    relabeled_table = cm.build_task_table(policy, w2, relabeled)
    relabeled_classes = {swap[t]: r["criticality"] for t, r in relabeled_table.items()}
    if relabeled_classes != base_classes:
        violations.append("relabelling task ids moved the class (criticality depends on task id)")

    w_probe = json.loads(json.dumps(w2))
    for rule in w_probe["task_class_rules"]:
        if rule.get("t_ref_s_range"):
            rule["t_ref_s_range"] = [7.0 * v for v in rule["t_ref_s_range"]]
    if {t: r["criticality"] for t, r in cm.build_task_table(policy, w_probe, templates[0]).items()} != base_classes:
        violations.append("scaling the workload moved the class (criticality depends on workload)")

    policy_probe = json.loads(json.dumps(policy))
    if {t: r["criticality"] for t, r in cm.build_task_table(policy_probe, w2, templates[0]).items()} != base_classes:
        violations.append("class assignment is not reproducible")

    # ---- mode machine: adversarial probes -----------------------------------
    for tpl in templates:
        fam = tpl["family_id"]
        table = cm.build_task_table(policy, w2, tpl)
        order = sorted(table)
        high_ids = [t for t in order if table[t]["criticality"] == "HIGH"]

        quiet = {t: 0.25 * table[t]["empirical_execution_budget_lo_s"] for t in order}
        tr = cm.simulate_mode_trace(policy, table, quiet, order)
        if tr.final_mode != "LO" or tr.switches:
            violations.append(f"{fam}: a no-overrun trace switched mode")

        low_overrun = dict(quiet)
        non_high = [t for t in order if table[t]["criticality"] != "HIGH"]
        if non_high:
            low_overrun[non_high[0]] = 5.0 * table[non_high[0]]["empirical_execution_budget_lo_s"]
            tr2 = cm.simulate_mode_trace(policy, table, low_overrun, order)
            if tr2.final_mode != "LO" or tr2.switches:
                violations.append(f"{fam}: a non-HIGH overrun switched the mode")
        if high_ids:
            hot = dict(quiet)
            trigger = high_ids[1] if len(high_ids) > 1 else high_ids[0]
            hot[trigger] = 2.0 * table[trigger]["empirical_execution_budget_lo_s"]
            tr3 = cm.simulate_mode_trace(policy, table, hot, order)
            if tr3.final_mode != "HI" or len(tr3.switches) != 1:
                violations.append(f"{fam}: HIGH overrun did not produce exactly one LO->HI switch")
            else:
                sw = tr3.switches[0].as_dict()
                if sorted(sw) != sorted(cm.SWITCH_LOG_FIELDS):
                    violations.append(f"{fam}: mode-switch log fields are incomplete")
                if sw["triggering_task_id"] != trigger or sw["previous_mode"] != "LO" or sw["new_mode"] != "HI":
                    violations.append(f"{fam}: mode-switch content is wrong")
            # stickiness: a later low-demand task cannot leave HI
            tr4 = cm.simulate_mode_trace(policy, table, hot, list(reversed(order)))
            if tr4.final_mode != "HI":
                violations.append(f"{fam}: HI is not sticky until graph completion")
            traces[fam] = tr3.as_dict()

    # ---- immutability: nothing on disk moved --------------------------------
    after = snapshot()
    moved = [k for k in before if before[k] != after[k]]
    if moved:
        violations.append(f"M7 mutated frozen artifacts: {moved}")

    report = {
        "schema_version": "criticality_report_v1",
        "milestone": "M7",
        "criticality_model_version": cm.CRITICALITY_MODEL_VERSION,
        "policy_id": policy["policy_id"],
        "criticality_policy_sha256": cm.policy_sha256(policy),
        "classes": list(cm.CLASSES),
        "axes": {
            "criticality_inputs": policy["axis_separation"]["allowed_inputs"],
            "criticality_forbidden_inputs": policy["axis_separation"]["forbidden_inputs"],
            "budget_axis": policy["budgets"]["axis"],
            "budget_not_safety_level": policy["budgets"]["not_safety_level"],
            "wcet_claim": policy["budgets"]["wcet_claim"],
            "initial_mode": policy["mode_semantics"]["initial_mode"],
            "hi_exit": policy["mode_semantics"]["hi_exit"],
        },
        "criticality_is_not_deadline_tightness": {
            "depth_association": depth_assoc,
            "deadline_urgency_association": urgency_assoc,
            "independent": not (depth_assoc["criticality_is_a_function_of_probe"]
                                or urgency_assoc["criticality_is_a_function_of_probe"]),
        },
        "mode_switch": {
            "rule_id": policy["mode_semantics"]["switch_rule_id"],
            "switch_log_fields": list(cm.SWITCH_LOG_FIELDS),
            "high_never_silently_dropped": True,
            "drop_degrade_policy": policy["mode_semantics"]["drop_degrade_policy"],
            "probe_traces": traces,
        },
        "role_policy_table": role_policy,
        "family_rows": rows,
        "immutability": {
            "pinned_artifact_sha256": after,
            "pinned_artifact_sha256_before": before,
            "m6_deadline_output_sha256_after_m7": deadline_hashes_after,
            "m6_deadline_output_sha256_pinned": pinned_by_family,
            "frozen_fields_untouched": [
                "D_G", "E_i", "L_i", "d_i", "S_i", "W_i", "B_e",
                "resource profiles", "SLA config",
            ],
            "m7_writes_only": ["criticality_policy.yaml", "criticality_model.py",
                               "validate_automotive_m7.py", "criticality_report.json"],
        },
        "status": "PASS" if not violations else "FAIL",
        "violations": violations,
    }
    (BASE / "criticality_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> int:
    rep = build_report()
    print(json.dumps({
        "status": rep["status"],
        "criticality_policy_sha256": rep["criticality_policy_sha256"],
        "families": len(rep["family_rows"]),
        "mixtures": {r["family_id"]: r["criticality_mixture"] for r in rep["family_rows"]},
        "criticality_is_not_deadline_tightness": rep["criticality_is_not_deadline_tightness"]["independent"],
        "depth_nmi": rep["criticality_is_not_deadline_tightness"]["depth_association"]["normalized_mutual_information"],
        "urgency_nmi": rep["criticality_is_not_deadline_tightness"]["deadline_urgency_association"]["normalized_mutual_information"],
        "violations": rep["violations"],
    }, indent=2, sort_keys=True))
    return 1 if rep["violations"] else 0


if __name__ == "__main__":
    sys.exit(main())
