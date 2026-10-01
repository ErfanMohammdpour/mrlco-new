#!/usr/bin/env python3
"""M9 splits + leakage checker for MARGO-AUTOMOTIVE-MC-v1.

The split is built from generating factors and lineage (never by shuffling files):
a graph's role follows its template lineage, its parent-seed sibling group stays
together, and the held-out roles are stratified by the regime cells. The leakage
checker refuses (exit != 0) any cross-role sharing of a lineage, a parent seed, a
semantic/topological signature, a near-duplicate signature or a canonical hash, and
any support/query overlap.

CLI:
    python3 automotive_splits.py [--dataset DIR] [--check-only]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

import automotive_dataset as ad  # noqa: E402

SPLIT_ROLES = ("meta_train", "validation", "meta_test")
ROLE_NAMES = ("train", "support", "query", "excluded")


def load_graphs(dataset_dir: Path | None = None) -> tuple[list[dict], Path]:
    d = Path(dataset_dir) if dataset_dir else ad.DATASET_DIR
    graphs_path = d / "graphs.jsonl"
    if not graphs_path.exists():
        raise ad.DatasetError(f"no materialized dataset at {graphs_path}")
    graphs = [json.loads(line) for line in graphs_path.read_text().splitlines() if line.strip()]
    return graphs, d


def near_duplicate_signature(g: dict) -> str:
    return ad.sha256_json({
        "template_lineage": g["template_lineage"],
        "topology_signature": g["topology_signature"],
        "resource_profile": g["resource_profile"],
        "workload_regime": g["workload_regime"],
        "sla_regime": g["sla_regime"],
        "resource_level": g["resource_level"],
        "criticality_mixture": g["criticality_mixture"],
    })


def stratum_of(g: dict, policy: dict) -> str:
    return "|".join(str(g[k]) for k in policy["support_query"]["strata"])


def _rank(policy: dict, g: dict) -> str:
    material = "\0".join([policy["support_query"]["rank_rule"], policy["split_seed"],
                          g["graph_id"], g["canonical_sha256"]])
    return ad.sha256_text(material)


def assign(g: dict, policy: dict, index_in_stratum: int, support_count: int) -> dict:
    family = g["application_family"]
    split = next((role for role in SPLIT_ROLES if family in policy["roles"][role]), None)
    if split is None:
        return {"split": "excluded", "role_in_split": "excluded", "stratum": None,
                "assignment_rank": None}
    if split == "meta_train":
        role = "train"
    else:
        role = "support" if index_in_stratum < support_count else "query"
    return {"split": split, "role_in_split": role, "stratum": stratum_of(g, policy),
            "assignment_rank": _rank(policy, g)}


def allocate_support(policy: dict, strata_counts: dict) -> dict:
    """Proportional largest-remainder support quota per stratum."""
    support = int(policy["support_query"]["support_count"])
    total_graphs = int(policy["support_query"]["support_count"] +
                       policy["support_query"]["query_count"])
    quota = {}
    remainders = []
    for sid in sorted(strata_counts):
        n = strata_counts[sid]
        exact = support * n / total_graphs
        base = int(exact)
        quota[sid] = base
        remainders.append((exact - base, sid))
    remaining = support - sum(quota.values())
    for _rem, sid in sorted(remainders, key=lambda t: (-t[0], t[1])):
        if remaining <= 0:
            break
        quota[sid] += 1
        remaining -= 1
    if remaining != 0:
        raise ad.DatasetError("support quota allocation did not close")
    return quota


def build_assignment(graphs: list[dict], policy: dict) -> dict:
    assignment: dict[str, dict] = {}
    for split in ("validation", "meta_test"):
        members = [g for g in graphs if g["application_family"] in policy["roles"][split]]
        strata: dict[str, list[dict]] = {}
        for g in members:
            strata.setdefault(stratum_of(g, policy), []).append(g)
        quota = allocate_support(policy, {k: len(v) for k, v in strata.items()})
        for sid in sorted(strata):
            ranked = sorted(strata[sid], key=lambda g: (_rank(policy, g), g["graph_id"]))
            for idx, g in enumerate(ranked):
                assignment[g["graph_id"]] = assign(g, policy, idx, quota[sid])
    for g in graphs:
        if g["graph_id"] not in assignment:
            assignment[g["graph_id"]] = assign(g, policy, 0, 0)
    return assignment


def leakage_checks(graphs: list[dict], assignment: dict, policy: dict) -> list[str]:
    v: list[str] = []
    by_id = {g["graph_id"]: g for g in graphs}
    if len(by_id) != len(graphs):
        v.append("duplicate graph_id in the materialized dataset")
    missing = [g["graph_id"] for g in graphs if g["graph_id"] not in assignment]
    if missing:
        v.append(f"{len(missing)} graphs have no split assignment")
    unknown = [k for k in assignment if k not in by_id]
    if unknown:
        v.append(f"{len(unknown)} split assignments reference unknown graphs")
    roles_used = {a["split"] for a in assignment.values()} - {"excluded"}
    for split in SPLIT_ROLES:
        if split not in roles_used:
            v.append(f"split role {split} is empty")
    # per-key cross-role sharing
    for key in policy["leakage_keys"]:
        buckets: dict[str, set] = {}
        for g in graphs:
            a = assignment.get(g["graph_id"])
            if not a:
                continue
            value = (near_duplicate_signature(g) if key == "near_duplicate_signature"
                     else g[key])
            buckets.setdefault(str(value), set()).add(a["split"])
        for value, splits in buckets.items():
            if len(splits) > 1:
                v.append(f"leakage: {key}={value} appears in splits {sorted(splits)}")
    # support/query disjointness + duplicate membership
    seen: dict[str, str] = {}
    for g in graphs:
        a = assignment[g["graph_id"]]
        key = (a["split"], a["role_in_split"])
        if a["role_in_split"] == "excluded":
            continue
        ident = f"{a['split']}/{a['role_in_split']}"
        if g["graph_id"] in seen:
            v.append(f"{g['graph_id']} assigned twice ({seen[g['graph_id']]} and {ident})")
        seen[g["graph_id"]] = ident
    for split in ("validation", "meta_test"):
        support = {gid for gid, a in assignment.items()
                   if a["split"] == split and a["role_in_split"] == "support"}
        query = {gid for gid, a in assignment.items()
                 if a["split"] == split and a["role_in_split"] == "query"}
        if support & query:
            v.append(f"{split}: support and query overlap on {sorted(support & query)}")
        expected_support = int(policy["support_query"]["support_count"])
        expected_query = int(policy["support_query"]["query_count"])
        if len(support) != expected_support:
            v.append(f"{split}: support count {len(support)} != {expected_support}")
        if len(query) != expected_query:
            v.append(f"{split}: query count {len(query)} != {expected_query}")
    # calibration isolation
    cal = policy.get("calibration") or {}
    if cal.get("source") != "meta_train":
        v.append("calibration source must be meta_train")
    for gid, a in assignment.items():
        if a["split"] == "meta_train" and a["role_in_split"] != "train":
            v.append(f"{gid}: a meta_train graph must carry role 'train'")
    # meta-test isolation declaration
    iso = policy.get("meta_test_isolation") or {}
    if iso.get("opened") not in (False, None):
        v.append("meta_test may not be opened during dataset materialization")
    for use in iso.get("opened_for") or []:
        if use in (iso.get("forbidden_uses") or []):
            v.append(f"meta_test opened for a forbidden use: {use}")
    return v


def factor_distributions(graphs: list[dict], assignment: dict, key: str) -> dict:
    out: dict[str, dict] = {}
    for g in graphs:
        split = assignment[g["graph_id"]]["split"]
        out.setdefault(split, {})
        value = str(g[key])
        out[split][value] = out[split].get(value, 0) + 1
    return {k: dict(sorted(v.items())) for k, v in sorted(out.items())}


def split_summary(graphs: list[dict], assignment: dict, policy: dict) -> dict:
    counts = {split: 0 for split in SPLIT_ROLES}
    role_counts: dict[str, int] = {}
    for a in assignment.values():
        counts[a["split"]] = counts.get(a["split"], 0) + 1
        role_counts[f"{a['split']}/{a['role_in_split']}"] = \
            role_counts.get(f"{a['split']}/{a['role_in_split']}", 0) + 1
    return {
        "schema_version": "automotive_split_summary_v1",
        "dataset_version": policy["dataset_version"],
        "split_rule_id": policy["split_rule_id"],
        "split_policy_sha256": ad.sha256_json(policy),
        "graph_count": len(graphs),
        "counts": {k: counts[k] for k in SPLIT_ROLES},
        "role_counts": dict(sorted(role_counts.items())),
        "factor_distributions": {
            k: factor_distributions(graphs, assignment, k) for k in
            ("application_family", "template_lineage", "topology_regime", "workload_regime",
             "resource_profile", "resource_level", "sla_regime", "criticality_mixture",
             "safety_scope")
        },
        "support_query": policy["support_query"],
        "calibration_source": policy["calibration"]["source"],
        "meta_test_isolation": policy["meta_test_isolation"],
        "leakage": "PASS",
    }


def write_splits(dataset_dir: Path, graphs: list[dict], assignment: dict,
                 policy: dict, summary: dict) -> None:
    lines = []
    for gid in sorted(assignment):
        a = assignment[gid]
        lines.append(ad.canonical_json({"graph_id": gid, **a,
                                        "canonical_sha256": next(
                                            g["canonical_sha256"] for g in graphs
                                            if g["graph_id"] == gid)}))
    (dataset_dir / "splits.jsonl").write_text("".join(l + "\n" for l in lines))
    (dataset_dir / "split_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (dataset_dir / "split_policy.json").write_text(
        json.dumps(policy, indent=2, sort_keys=True) + "\n")


def run(dataset_dir: Path | None = None, check_only: bool = False) -> dict:
    graphs, d = load_graphs(dataset_dir)
    policy = json.loads((ad.SPEC_DIR / "split_policy.json").read_text())
    assignment = build_assignment(graphs, policy)
    violations = leakage_checks(graphs, assignment, policy)
    summary = split_summary(graphs, assignment, policy)
    summary["leakage"] = "PASS" if not violations else "FAIL"
    summary["violations"] = violations
    if not check_only and not violations:
        write_splits(d, graphs, assignment, policy, summary)
    return {"status": "PASS" if not violations else "FAIL", "violations": violations,
            "summary": summary, "assignment": assignment, "dataset_dir": str(d)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(ad.DATASET_DIR))
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()
    res = run(Path(args.dataset), check_only=args.check_only)
    print(json.dumps({"status": res["status"], "counts": res["summary"]["counts"],
                      "role_counts": res["summary"]["role_counts"],
                      "leakage": res["summary"]["leakage"],
                      "violations": res["violations"]}, indent=2, sort_keys=True))
    return 1 if res["violations"] else 0


if __name__ == "__main__":
    sys.exit(main())
