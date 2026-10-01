#!/usr/bin/env python3
"""M10 validator: run certification on the frozen dataset and verify it. Exit != 0.

Proves that certification is READ-ONLY with respect to every frozen requirement:
the sha256 of graphs.jsonl, dataset_manifest.jsonl and splits.jsonl must be identical
before and after certification, and each graph's D_G/W_i/B_e/criticality/budgets must
be unchanged.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

import automotive_dataset as ad  # noqa: E402
import automotive_splits as asp  # noqa: E402
import certify_automotive_m10 as cm  # noqa: E402

ALLOWED_STATUS = ("certified_feasible", "witness_not_found", "stress_or_infeasible")
REQUIRED_PROVENANCE_KEYS = (
    "git_sha", "source_registry_sha", "required_parameter_manifest_sha", "evidence_shas",
    "workload_model_v2_sha", "resource_profile_sha", "sla_registry_v2_sha",
    "deadline_model_v2_sha", "criticality_policy_sha", "task_semantics_sha", "generator_sha",
    "generation_config_sha", "split_policy_sha", "dataset_manifest_sha",
)
FROZEN_FIELDS = ("D_G_s", "resource", "criticality_counts", "sla_id", "sla_regime",
                 "resource_profile", "workload_regime", "parent_seed", "template_lineage")


def build_report() -> dict:
    d = ad.DATASET_DIR
    graphs, _ = asp.load_graphs(d)
    before = {
        "graphs.jsonl": ad.sha256_file(d / "graphs.jsonl"),
        "dataset_manifest.jsonl": ad.sha256_file(d / "dataset_manifest.jsonl"),
        "splits.jsonl": ad.sha256_file(d / "splits.jsonl"),
    }
    frozen_before = {g["graph_id"]: {k: ad.sha256_json(g[k]) for k in FROZEN_FIELDS}
                     for g in graphs}
    res = cm.certify(d, write=True)
    after = {k: ad.sha256_file(d / k) for k in before}
    graphs_after, _ = asp.load_graphs(d)
    frozen_after = {g["graph_id"]: {k: ad.sha256_json(g[k]) for k in FROZEN_FIELDS}
                    for g in graphs_after}

    violations: list[str] = []
    for key in before:
        if before[key] != after[key]:
            violations.append(f"certification rewrote {key}")
    if frozen_before != frozen_after:
        violations.append("certification mutated a frozen graph field")
    if len(graphs_after) != len(graphs):
        violations.append("graph count changed during certification")
    for g in graphs_after:
        for t in g["tasks"]:
            if t["criticality"] == "HIGH" and (
                    t["empirical_execution_budget_hi_s"] is None
                    or float(t["empirical_execution_budget_hi_s"]) < float(t["empirical_execution_budget_lo_s"])):
                violations.append(f"{g['graph_id']}/{t['task_id']}: frozen MC budget was changed")

    audits = res["audits"]
    if audits["status"] != "PASS":
        violations.append(f"audit failures: {audits['failed']}")

    # manifest.jsonl must carry one certification record per graph
    manifest_path = d / "manifest.jsonl"
    if not manifest_path.exists():
        violations.append("manifest.jsonl is missing")
    else:
        records = [json.loads(l) for l in manifest_path.read_text().splitlines() if l.strip()]
        if len(records) != len(graphs):
            violations.append(f"manifest.jsonl has {len(records)} records for {len(graphs)} graphs")
        for r in records:
            if r.get("certification", {}).get("status") not in ALLOWED_STATUS:
                violations.append(f"{r.get('graph_id')}: invalid certification status")
            if r.get("certification", {}).get("requirements_rewritten") is not False:
                violations.append(f"{r.get('graph_id')}: certification must not rewrite requirements")
            for key in ("split", "role_in_split", "canonical_sha256", "parent_seed",
                        "template_lineage", "resource_profile", "sla_id", "criticality_mixture"):
                if key not in r:
                    violations.append(f"{r.get('graph_id')}: manifest record missing {key}")

    prov_path = d / "provenance.json"
    if not prov_path.exists():
        violations.append("provenance.json is missing")
        prov = {}
    else:
        prov = json.loads(prov_path.read_text())
        for key in REQUIRED_PROVENANCE_KEYS:
            if key not in prov:
                violations.append(f"provenance.json missing {key}")
        if "required_parameter_manifest_sha" not in prov or "dataset_manifest_sha" not in prov:
            violations.append("the two manifest hashes must keep distinct names")
        if prov.get("required_parameter_manifest_sha") == prov.get("dataset_manifest_sha"):
            violations.append("required_parameter_manifest_sha and dataset_manifest_sha collapsed")
        if prov.get("leakage_status") != "PASS":
            violations.append("provenance records a leakage failure")

    cal_path = d / "calibration_report.json"
    cal = json.loads(cal_path.read_text()) if cal_path.exists() else {}
    if cal.get("calibration_source") != "meta_train":
        violations.append("calibration source is not meta-train")
    if (cal.get("isolation") or {}).get("validation_graphs_used", 1) != 0:
        violations.append("calibration used validation graphs")
    if (cal.get("isolation") or {}).get("meta_test_graphs_used", 1) != 0:
        violations.append("calibration used meta-test graphs")

    feas = res["feasibility"]
    for key in ("certified_feasible", "witness_not_found", "stress_or_infeasible"):
        if key not in feas["status_counts"]:
            violations.append(f"feasibility report missing status {key}")
    if abs(sum(feas["status_rates"].values()) - 1.0) > 1e-9:
        violations.append("feasibility rates do not sum to 1")

    report = {
        "schema_version": "m10_validation_report_v1",
        "milestone": "M10",
        "certification_version": cm.CERTIFICATION_VERSION,
        "audit_status": audits["status"],
        "failed_audits": audits["failed"],
        "status_counts": feas["status_counts"],
        "status_rates": feas["status_rates"],
        "rho_G": feas["rho_G"],
        "mixed_placement_useful_rate": feas["mixed_placement_useful_rate"],
        "pure_plan_best_share": feas["pure_plan_best_share"],
        "method_comparison": feas["method_comparison"],
        "certification_read_only": before == after and frozen_before == frozen_after,
        "frozen_hashes_before": before,
        "frozen_hashes_after": after,
        "provenance": prov,
        "audit_detail": audits["audits"],
        "status": "PASS" if not violations else "FAIL",
        "violations": violations,
    }
    (BASE / "m10_validation_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> int:
    rep = build_report()
    print(json.dumps({"status": rep["status"], "audit_status": rep["audit_status"],
                      "failed_audits": rep["failed_audits"],
                      "status_counts": rep["status_counts"],
                      "certification_read_only": rep["certification_read_only"],
                      "violations": rep["violations"][:10]}, indent=2, sort_keys=True))
    return 1 if rep["violations"] else 0


if __name__ == "__main__":
    sys.exit(main())
