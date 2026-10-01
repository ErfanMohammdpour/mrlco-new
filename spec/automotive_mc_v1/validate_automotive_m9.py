#!/usr/bin/env python3
"""M9 validator: split assignment, support/query disjointness, zero leakage. Exit != 0.

Writes `m9_split_report.json`. The leakage checker inside `automotive_splits.py` is the
authority; this validator re-runs it, checks the on-disk split artifacts are fresh, and
asserts the calibration/meta-test isolation declarations.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

import automotive_dataset as ad  # noqa: E402
import automotive_splits as asp  # noqa: E402


def build_report() -> dict:
    graphs, d = asp.load_graphs()
    policy = json.loads((ad.SPEC_DIR / "split_policy.json").read_text())
    res = asp.run(d, check_only=True)
    assignment = res["assignment"]
    summary = res["summary"]
    violations = list(res["violations"])

    # fresh on-disk artifacts
    fresh = []
    for gid in sorted(assignment):
        a = assignment[gid]
        g = next(x for x in graphs if x["graph_id"] == gid)
        fresh.append(ad.canonical_json({"graph_id": gid, **a,
                                        "canonical_sha256": g["canonical_sha256"]}))
    fresh_text = "".join(l + "\n" for l in fresh)
    splits_path = d / "splits.jsonl"
    if not splits_path.exists():
        violations.append("splits.jsonl is missing")
    elif splits_path.read_text() != fresh_text:
        violations.append("on-disk splits.jsonl is stale")
    if (d / "split_summary.json").read_text() != json.dumps(summary, indent=2, sort_keys=True) + "\n":
        violations.append("on-disk split_summary.json is stale")
    if (d / "split_policy.json").read_text() != json.dumps(policy, indent=2, sort_keys=True) + "\n":
        violations.append("on-disk split_policy.json copy is stale")

    counts = summary["counts"]
    if sum(counts.values()) != len(graphs):
        violations.append("split counts do not cover every graph")
    role_counts = summary["role_counts"]
    for split in ("validation", "meta_test"):
        if role_counts.get(f"{split}/support") != int(policy["support_query"]["support_count"]):
            violations.append(f"{split}: wrong support count")
        if role_counts.get(f"{split}/query") != int(policy["support_query"]["query_count"]):
            violations.append(f"{split}: wrong query count")
    if counts.get("meta_train", 0) <= 0 or counts.get("validation", 0) <= 0 or counts.get("meta_test", 0) <= 0:
        violations.append("every split role must be non-empty")
    if summary["calibration_source"] != "meta_train":
        violations.append("calibration source is not meta-train")
    if (policy.get("meta_test_isolation") or {}).get("opened") is not False:
        violations.append("meta-test must stay unopened")
    # near-duplicate siblings must never cross a boundary (explicit re-check)
    groups: dict[str, set] = {}
    for g in graphs:
        groups.setdefault(str(g["parent_seed"]), set()).add(assignment[g["graph_id"]]["split"])
    for seed, splits in groups.items():
        if len(splits) > 1:
            violations.append(f"parent seed {seed} crosses splits {sorted(splits)}")

    report = {
        "schema_version": "m9_split_report_v1",
        "milestone": "M9",
        "split_rule_id": policy["split_rule_id"],
        "split_unit": policy["split_unit"],
        "split_policy_sha256": ad.sha256_file(ad.SPEC_DIR / "split_policy.json"),
        "counts": counts,
        "role_counts": role_counts,
        "factor_distributions": summary["factor_distributions"],
        "leakage": res["status"],
        "leakage_keys": policy["leakage_keys"],
        "calibration_source": summary["calibration_source"],
        "meta_test_isolation": summary["meta_test_isolation"],
        "support_query": policy["support_query"],
        "status": "PASS" if not violations else "FAIL",
        "violations": violations,
    }
    (BASE / "m9_split_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> int:
    rep = build_report()
    print(json.dumps({"status": rep["status"], "counts": rep["counts"],
                      "role_counts": rep["role_counts"], "leakage": rep["leakage"],
                      "violations": rep["violations"]}, indent=2, sort_keys=True))
    return 1 if rep["violations"] else 0


if __name__ == "__main__":
    sys.exit(main())
