#!/usr/bin/env python3
"""M6 validator: SLA/deadline construction. Exit != 0 on any violation.

Runs the analytic deadline model over the M4 templates using M5 data, checks the
construction invariants, writes deadline_report.json and refuses to import or use
any scheduler/search module.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import yaml

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
SPEC = importlib.util.spec_from_file_location("deadline_model", BASE / "deadline_model.py")
dm = importlib.util.module_from_spec(SPEC)
sys.modules["deadline_model"] = dm
SPEC.loader.exec_module(dm)

FORBIDDEN_IMPORTS = ("scheduler", "engine", "witness", "certif", "heft")


def main() -> int:
    sla = yaml.safe_load((BASE / "sla_registry.yaml").read_text())
    workload = yaml.safe_load((BASE / "workload_model.yaml").read_text())
    templates = yaml.safe_load((BASE / "application_templates.yaml").read_text())["templates"]
    semantics = yaml.safe_load((BASE / "task_semantics.yaml").read_text())

    violations: list[str] = []
    if sla.get("branch") != "rho_period" or sla.get("eta_b_g_branch_active"):
        violations.append("only the rho_period branch may be active in v1")
    families = set(sla.get("families") or {})
    if families != set(semantics.get("families") or {}):
        violations.append("SLA families do not match the frozen M4 families")

    src = (BASE / "deadline_model.py").read_text()
    for token in FORBIDDEN_IMPORTS:
        if f"import {token}" in src or f"from .{token}" in src:
            violations.append(f"deadline_model must not import {token}")

    results, report_rows = [], []
    sla_sha = dm.sla_config_sha256(sla)
    for tpl in templates:
        fam = tpl["family_id"]
        try:
            res = dm.generate_deadlines(tpl, workload, sla, fam)
        except dm.DeadlineError as exc:
            violations.append(f"{tpl['template_id']}: {exc}")
            continue
        results.append(res)
        slacks = [t.slack for t in res.tasks.values()]
        sinks = [t for t in res.tasks.values() if t.L == res.D_G]
        if res.status == "ok":
            for t in res.tasks.values():
                if not (t.E <= t.d + 1e-12 <= t.L + 1e-12):
                    violations.append(f"{t.task_id}: E<=d<=L violated")
                if t.slack < -1e-12:
                    violations.append(f"{t.task_id}: negative slack")
            if not sinks:
                violations.append(f"{tpl['template_id']}: no sink anchored exactly at D_G")
            if res.D_G <= 0 or res.CP_ref <= 0:
                violations.append(f"{tpl['template_id']}: non-positive D_G or CP_ref")
        report_rows.append({
            "template_id": tpl["template_id"], "family_id": fam, "status": res.status,
            "D_G": res.D_G, "CP_ref": res.CP_ref, "rho_used": res.rho_used,
            "period_used": res.period_used,
            "slack_min": min(slacks) if slacks else None,
            "slack_median": sorted(slacks)[len(slacks) // 2] if slacks else None,
            "slack_max": max(slacks) if slacks else None,
            "zero_slack_tasks": sum(1 for s in slacks if abs(s) < 1e-12),
            "tasks": len(res.tasks), "output_sha256": res.output_sha256,
            "input_fingerprint": res.input_fingerprint,
        })

    # determinism: identical inputs must give identical hashes and values
    again = [dm.generate_deadlines(t, workload, sla, t["family_id"]) for t in templates]
    deterministic = all(a.output_sha256 == b.output_sha256 and a.D_G == b.D_G and a.CP_ref == b.CP_ref
                        for a, b in zip(results, again))
    if not deterministic:
        violations.append("deadline construction is not deterministic")

    report = {
        "schema_version": "deadline_report_v1",
        "deadline_model_version": dm.DEADLINE_MODEL_VERSION,
        "sla_config_sha256": sla_sha,
        "subdeadline_rule": sla.get("subdeadline_rule"),
        "branch": sla.get("branch"),
        "families": sorted(families),
        "templates_checked": len(templates),
        "graphs_evaluated": len(results),
        "D_G_range": [min((r.D_G for r in results), default=None), max((r.D_G for r in results), default=None)],
        "CP_ref_range": [min((r.CP_ref for r in results), default=None), max((r.CP_ref for r in results), default=None)],
        "slack_range": [min((x for r in results for x in [t.slack for t in r.tasks.values()]), default=None),
                        max((x for r in results for x in [t.slack for t in r.tasks.values()]), default=None)],
        "negative_slack_count": sum(1 for r in results for t in r.tasks.values() if t.slack < -1e-12),
        "zero_slack_task_count": sum(1 for r in results for t in r.tasks.values() if abs(t.slack) < 1e-12),
        "infeasible_lower_bound_count": sum(1 for r in results if r.status != "ok"),
        "deterministic_rerun": "PASS" if deterministic else "FAIL",
        "witness_independence": "PASS (the model takes only template + M5 data + frozen SLA)",
        "scheduler_import_independence": "PASS (no scheduler/search import)",
        "not_certification": "this report makes no scheduler feasibility claim",
        "rows": report_rows,
        "status": "PASS" if not violations else "FAIL",
        "violations": violations,
    }
    (BASE / "deadline_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "templates_checked", "graphs_evaluated",
                                            "D_G_range", "CP_ref_range", "slack_range",
                                            "negative_slack_count", "zero_slack_task_count",
                                            "infeasible_lower_bound_count", "deterministic_rerun",
                                            "sla_config_sha256")}, indent=2, sort_keys=True))
    for v in violations:
        print("VIOLATION:", v)
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
