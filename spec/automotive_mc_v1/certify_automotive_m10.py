#!/usr/bin/env python3
"""M10 independent certification + final scientific audit for MARGO-AUTOMOTIVE-MC-v1.

Certification runs on the FROZEN materialized dataset and may not rewrite workload,
payload, D_G, task deadlines, criticality, C_LO/C_HI or the resource profile. It only
EVALUATES placements with the canonical scheduler and records the outcome.

Statuses:
    certified_feasible    a witness with makespan <= D_G was found
    witness_not_found     no witness found, and D_G is not below the true bound
    stress_or_infeasible  D_G is below the analytic lower bound (model-level infeasible)

`witness_not_found` is NEVER reported as mathematical infeasibility. `rho_G` is a
diagnostic and never redefines D_G.

CLI:
    python3 certify_automotive_m10.py [--dataset DIR]
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import sys
import tempfile
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

import automotive_dataset as ad  # noqa: E402
import automotive_generator as ag  # noqa: E402
import automotive_splits as asp  # noqa: E402

if str(ad.REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(ad.REPO_ROOT))

from env.mec_offloaing_envs.scheduler.energy_model import (  # noqa: E402
    EnergyModelSpec, TierSpec,
)
from env.mec_offloaing_envs.scheduler.energy_scope import energy_scalar  # noqa: E402
from env.mec_offloaing_envs.scheduler.engine import schedule  # noqa: E402
from env.mec_offloaing_envs.scheduler.model import (  # noqa: E402
    CanonicalDAG, CanonicalTask, Location,
)
from env.mec_offloaing_envs.scheduler.resources import (  # noqa: E402
    TIMING_PHYSICAL, ResourceConfig,
)

CERTIFICATION_VERSION = "automotive_certification_v1"
XI = 300.0
SEARCH_BUDGET = 240
_KEEP_PLANS = ("mixed_search",)
KAPPA_PLACEHOLDER = 1.0e-27  # timing-tier spec only; energy uses the legacy axis


# --------------------------------------------------------------------------- #
# Frozen graph -> canonical scheduling objects
# --------------------------------------------------------------------------- #
def to_canonical(g: dict) -> CanonicalDAG:
    tasks = []
    for t in g["tasks"]:
        tasks.append(CanonicalTask(
            task_id=int(t["task_id"]),
            compute_workload_bytes=int(t["compute_workload_bytes"]),
            task_output_bytes=int(t["task_output_bytes"]),
            external_input_bytes=int(t["external_input_bytes"]),
            cycles_per_bit=XI,
            deadline_s=float(t["deadline_s"]),
            deadline_type=str(t["deadline_type"]),
            criticality_class=str(t["criticality"]).lower(),
            tardiness_weight=float(t["tardiness_weight"]),
        ))
    edges = [(int(e["src"]), int(e["dst"]), int(e["payload_bytes"])) for e in g["edges"]]
    return CanonicalDAG.from_records(tasks, edges)


def to_resources(g: dict) -> ResourceConfig:
    r = g["resource"]
    tiers = {
        "ue": TierSpec(f_hz=float(r["f_ue_hz"]), kappa=KAPPA_PLACEHOLDER,
                       source="dataset_resource_draw"),
        "helper": TierSpec(f_hz=float(r["f_helper_hz"]), kappa=KAPPA_PLACEHOLDER,
                           source="dataset_resource_draw"),
        "mec": TierSpec(f_hz=float(r["f_mec_hz"]), kappa=KAPPA_PLACEHOLDER,
                        source="dataset_resource_draw"),
    }
    timing = EnergyModelSpec(model="physical_v1", energy_scope="mobile",
                             cycles_per_bit=XI, include_rx_energy=False, tiers=tiers,
                             ue_tx_w=0.1, mec_tx_w=0.1, helper_tx_w=0.06)
    return ResourceConfig(
        ue_cpu_bytes_per_second=float(r["f_ue_hz"]) / (8.0 * XI),
        mec_cpu_bytes_per_second=float(r["f_mec_hz"]) / (8.0 * XI),
        helper_cpu_bytes_per_second=float(r["f_helper_hz"]) / (8.0 * XI),
        mec_uplink_bytes_per_second=float(r["r_mec_ul_bps"]) / 8.0,
        mec_downlink_bytes_per_second=float(r["r_mec_dl_bps"]) / 8.0,
        v2v_bytes_per_second=float(r["r_v2v_bps"]) / 8.0,
        rho_ue=1.0, f_l=1.0, zeta=2.0,
        ptx_mec_w=0.1, prx_mec_w=0.05, ptx_v2v_w=0.06, prx_v2v_w=0.03,
        rho_helper=0.7, f_v2v=1.0,
        energy_model=None, radio_model=None,
        timing_model=TIMING_PHYSICAL, timing_tiers=timing,
        energy_scope="",
        source_config_sha256=ad.sha256_json(r),
    )


def lower_bound_makespan(g: dict) -> float:
    """True lower bound: per-task fastest-tier compute, zero communication.

    Every task must run somewhere, so its duration is at least its minimum over the
    three tiers of THIS graph's frozen resource profile, and dependencies add >= 0.
    The reference-tier critical path is NOT used here: when a faster tier exists it
    is not a lower bound, and treating it as one would mislabel feasible graphs.
    """
    f_max = max(float(g["resource"]["f_ue_hz"]), float(g["resource"]["f_helper_hz"]),
                float(g["resource"]["f_mec_hz"]))
    finish: dict[int, float] = {}
    for t in g["tasks"]:
        tid = int(t["task_id"])
        dur = int(t["compute_workload_bytes"]) * 8.0 * XI / f_max
        start = max((finish[p] for p in t["predecessors"]), default=0.0)
        finish[tid] = start + dur
    return max(finish.values()) if finish else 0.0


# --------------------------------------------------------------------------- #
# Placement methods
# --------------------------------------------------------------------------- #
def evaluate(dag, order, actions, resources):
    return schedule(dag, order, actions, resources)


def _plan(g, locations):
    return [int(locations[int(t["task_id"])]) for t in g["tasks"]]


def pure_plans(g):
    return {
        "all_UE": _plan(g, {int(t["task_id"]): 0 for t in g["tasks"]}),
        "all_MEC": _plan(g, {int(t["task_id"]): 1 for t in g["tasks"]}),
        "all_HELPER": _plan(g, {int(t["task_id"]): 2 for t in g["tasks"]}),
    }


def greedy_plan(g, dag, order, resources):
    n = len(order)
    chosen: list[int] = []
    for k in range(n):
        best_metric, best_action = None, None
        for action in (0, 1, 2):
            plan = chosen + [action] + [0] * (n - k - 1)
            metric = evaluate(dag, order, plan, resources).makespan_seconds
            if (best_metric is None or metric + 1e-12 < best_metric
                    or (abs(metric - best_metric) <= 1e-12 and action < best_action)):
                best_metric, best_action = metric, action
        chosen.append(int(best_action))
    return chosen


def heft_plan(g, dag, order, resources):
    """HEFT-style reference placement: mean-cost upward rank, then min-finish pick."""
    r = g["resource"]
    tiers = ("f_ue_hz", "f_helper_hz", "f_mec_hz")
    mean_f = sum(float(r[k]) for k in tiers) / 3.0
    mean_rate = (float(r["r_mec_ul_bps"]) + float(r["r_mec_dl_bps"]) + float(r["r_v2v_bps"])) / 3.0
    task_by_id = {int(t["task_id"]): t for t in g["tasks"]}
    w = {tid: int(t["compute_workload_bytes"]) * 8.0 * XI / mean_f
         for tid, t in task_by_id.items()}
    edge_b = {(int(e["src"]), int(e["dst"])): int(e["payload_bytes"]) for e in g["edges"]}
    rank: dict[int, float] = {}
    for tid in reversed(order):
        succs = task_by_id[tid]["successors"]
        tail = max((edge_b[(tid, s)] * 8.0 / mean_rate + rank[s] for s in succs), default=0.0)
        rank[tid] = w[tid] + tail
    order_by_rank = sorted(order, key=lambda t: (-rank[t], t))
    assigned: dict[int, int] = {}
    for tid in order_by_rank:
        best = None
        for action in (0, 1, 2):
            f = tiers[action]
            dur = int(task_by_id[tid]["compute_workload_bytes"]) * 8.0 * XI / float(r[f])
            ready = 0.0
            for p in task_by_id[tid]["predecessors"]:
                ready = max(ready, edge_b[(p, tid)] * 8.0 / mean_rate)
            cost = ready + dur
            if best is None or cost < best[0] - 1e-12 or (abs(cost - best[0]) <= 1e-12
                                                         and action < best[1]):
                best = (cost, action)
        assigned[tid] = int(best[1])
    return _plan(g, assigned)


def mixed_search(g, dag, order, resources, starts, budget=SEARCH_BUDGET):
    """Bounded deterministic coordinate-descent over single-task relocations."""
    n = len(order)
    used = 0
    best_actions, best_metric = None, None
    for start in starts:
        actions = list(start)
        metric = evaluate(dag, order, actions, resources).makespan_seconds
        used += 1
        if best_metric is None or metric < best_metric - 1e-12:
            best_metric, best_actions = metric, list(actions)
        improved = True
        while improved and used < budget:
            improved = False
            for k in range(n):
                if used >= budget:
                    break
                current = actions[k]
                for action in (0, 1, 2):
                    if action == current or used >= budget:
                        continue
                    probe = list(actions)
                    probe[k] = action
                    m = evaluate(dag, order, probe, resources).makespan_seconds
                    used += 1
                    if m < metric - 1e-12:
                        actions, metric, improved = probe, m, True
                        current = action
                        if m < best_metric - 1e-12:
                            best_metric, best_actions = m, list(probe)
        if used >= budget:
            break
    return best_actions, best_metric, used


def certify_graph(g: dict, ref_cp: float) -> dict:
    dag = to_canonical(g)
    resources = to_resources(g)
    order = [int(t["task_id"]) for t in g["tasks"]]
    D_G = float(g["D_G_s"])
    methods: dict[str, dict] = {}

    def record(name, actions):
        res = evaluate(dag, order, actions, resources)
        methods[name] = {
            "makespan_s": res.makespan_seconds,
            "plan": [int(a) for a in actions],
            "is_pure": len(set(int(a) for a in actions)) == 1,
            "location_counts": {
                "UE": sum(1 for a in actions if int(a) == 0),
                "MEC": sum(1 for a in actions if int(a) == 1),
                "HELPER": sum(1 for a in actions if int(a) == 2),
            },
            "firm_miss_count": res.firm_miss_count,
            "hard_miss_count": res.hard_miss_count,
            "max_tardiness_s": res.max_tardiness_s,
            "total_mobile_joules_diagnostic": energy_scalar(res, scope="mobile"),
        }
        return res

    for name, plan in sorted(pure_plans(g).items()):
        record(name, plan)
    greedy = greedy_plan(g, dag, order, resources)
    record("greedy", greedy)
    heft = heft_plan(g, dag, order, resources)
    record("heft_reference", heft)

    baseline_best = min(methods, key=lambda k: (methods[k]["makespan_s"], k))
    starts = []
    for name in (baseline_best, "greedy", "heft_reference"):
        starts.append(list(methods[name]["plan"]))
    search_actions, search_metric, used = mixed_search(g, dag, order, resources, starts)
    if search_metric is None:
        search_actions, search_metric = list(methods[baseline_best]["plan"]), methods[baseline_best]["makespan_s"]
    methods["mixed_search"] = {
        "makespan_s": search_metric,
        "plan": [int(a) for a in search_actions],
        "is_pure": len(set(search_actions)) == 1,
        "location_counts": {
            "UE": sum(1 for a in search_actions if a == 0),
            "MEC": sum(1 for a in search_actions if a == 1),
            "HELPER": sum(1 for a in search_actions if a == 2),
        },
        "firm_miss_count": methods[baseline_best]["firm_miss_count"],
        "hard_miss_count": 0,
        "max_tardiness_s": 0.0,
        "total_mobile_joules_diagnostic": 0.0,
        "search_evaluations": used,
    }
    res_search = evaluate(dag, order, search_actions, resources)
    methods["mixed_search"].update({
        "firm_miss_count": res_search.firm_miss_count,
        "hard_miss_count": res_search.hard_miss_count,
        "max_tardiness_s": res_search.max_tardiness_s,
        "total_mobile_joules_diagnostic": energy_scalar(res_search, scope="mobile"),
    })

    best_method = min(methods, key=lambda k: (methods[k]["makespan_s"], k))
    best_makespan = methods[best_method]["makespan_s"]
    best_pure = min((k for k in methods if methods[k]["is_pure"] and k != "mixed_search"),
                    key=lambda k: (methods[k]["makespan_s"], k))
    pure_makespan = methods[best_pure]["makespan_s"]
    lb = lower_bound_makespan(g)
    if best_makespan <= D_G + 1e-12:
        status = "certified_feasible"
    elif D_G < lb - 1e-12:
        status = "stress_or_infeasible"
    else:
        status = "witness_not_found"
    mixed_useful = (not methods["mixed_search"]["is_pure"]
                    and methods["mixed_search"]["makespan_s"] <= pure_makespan - 1e-9)
    return {
        "status": status,
        "D_G_s": D_G,
        "best_method": best_method,
        "best_makespan_s": best_makespan,
        "best_plan_is_pure": bool(methods[best_method]["is_pure"]),
        "lower_bound_makespan_s": lb,
        "reference_tier_critical_path_s": ref_cp,
        "reference_tier_stress": bool(ref_cp > D_G + 1e-12),
        "rho_G": best_makespan / D_G,
        "heft_rho_G": methods["heft_reference"]["makespan_s"] / D_G,
        "pure_best_method": best_pure,
        "pure_best_makespan_s": pure_makespan,
        "mixed_plan_useful": bool(mixed_useful),
        "search_evaluations": used,
        "methods": methods,
        "deadline_misses": {
            "firm_miss_count_best": methods[best_method]["firm_miss_count"],
            "hard_miss_count_best": methods[best_method]["hard_miss_count"],
            "task_count": len(g["tasks"]),
        },
        "energy_diagnostic_j": methods[best_method]["total_mobile_joules_diagnostic"],
        "score_definition": "makespan_s from the canonical scheduler (sink return to UE included)",
    }


# --------------------------------------------------------------------------- #
# Aggregation + audits
# --------------------------------------------------------------------------- #
def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    idx = min(len(xs) - 1, max(0, int(round(q * (len(xs) - 1)))))
    return xs[idx]


def method_comparison(rows: list[dict]) -> dict:
    names = sorted({m for r in rows for m in r["methods"]})
    out = {}
    for name in names:
        wins = sum(1 for r in rows if r["best_method"] == name)
        vals = [r["methods"][name]["makespan_s"] for r in rows]
        out[name] = {
            "wins": wins,
            "win_rate": wins / len(rows) if rows else 0.0,
            "mean_makespan_s": statistics.fmean(vals) if vals else 0.0,
            "median_makespan_s": statistics.median(vals) if vals else 0.0,
        }
    return out


def run_audits(graphs: list[dict], assignment: dict, policy: dict, rows: list[dict],
               docs: dict, split_result: dict) -> dict:
    audits: dict[str, dict] = {}

    def audit(name: str, passed: bool, detail: dict, severity: str = "fail"):
        if passed:
            status = "PASS"
        else:
            status = {"fail": "FAIL", "warn": "WARN", "info": "INFO"}.get(severity, "FAIL")
        audits[name] = {"status": status, "detail": detail}

    makespans = sorted(r["best_makespan_s"] for r in rows)
    feasible = [r["best_makespan_s"] for r in rows if r["status"] == "certified_feasible"]
    timing_ok = (not makespans or (statistics.median(makespans) < 5.0 and max(makespans) < 60.0))
    audit("1_timing_scale", timing_ok, {
        "min_s": makespans[0] if makespans else None,
        "median_s": statistics.median(makespans) if makespans else None,
        "max_s": makespans[-1] if makespans else None,
        "feasible_median_s": statistics.median(feasible) if feasible else None,
        "historical_regression_band_s": [528.0, 1245.0],
        "note": "a normal graph must never return to the historical 528-1245 s behaviour",
    })

    task_total = sum(g["task_count"] for g in graphs)
    deadline_count = sum(1 for g in graphs for t in g["tasks"] if float(t["deadline_s"]) > 0)
    audit("2_deadlines", deadline_count == task_total and task_total > 0, {
        "task_total": task_total, "deadline_count": deadline_count,
        "deadline_rate": deadline_count / task_total if task_total else 0.0,
        "type": "firm (subdeadline target); graph-level requirement is hard (makespan <= D_G)",
    })

    counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
    for g in graphs:
        for k, v in g["criticality_counts"].items():
            counts[k] = counts.get(k, 0) + int(v)
    audit("3_criticality", all(v > 0 for v in counts.values()) and counts["HIGH"] > 0, {
        "counts": counts, "high_task_total": counts["HIGH"],
        "non_safety_family_high": "mapping_background is declared non_safety_background and carries LOW by policy",
    })

    budget_rows = [t for g in graphs for t in g["tasks"]]
    hi_rows = [t for t in budget_rows if t["criticality"] == "HIGH"]
    mc_ok = all(float(t["empirical_execution_budget_lo_s"]) > 0 for t in budget_rows) and all(
        float(t["empirical_execution_budget_hi_s"]) >= float(t["empirical_execution_budget_lo_s"])
        for t in hi_rows)
    audit("4_mc_budgets", mc_ok, {
        "high_tasks_with_hi_budget": len(hi_rows),
        "min_C_LO_s": min(float(t["empirical_execution_budget_lo_s"]) for t in budget_rows),
        "max_C_HI_s": max(float(t["empirical_execution_budget_hi_s"]) for t in hi_rows),
        "wcet_claim": False,
    })

    bands = ad.realised_payload_bands(graphs)
    classes = sorted(bands)
    disjoint = True
    for i, a in enumerate(classes):
        for b in classes[i + 1:]:
            lo_a, hi_a = bands[a]
            lo_b, hi_b = bands[b]
            if not (hi_a < lo_b or hi_b < lo_a):
                disjoint = False
    audit("5_payload_semantics", len(classes) == 8 and disjoint, {
        "classes": {k: bands[k] for k in classes},
        "class_count": len(classes),
        "pairwise_disjoint": disjoint,
        "note": "all eight payload classes are instantiated and their realised byte bands do not overlap",
    })

    pure_dominated = sum(1 for r in rows if r["best_plan_is_pure"])
    dominance = pure_dominated / len(rows) if rows else 0.0
    audit("6_placement_difficulty", dominance <= 0.9, {
        "pure_plan_best_share": dominance,
        "mixed_plan_best_share": 1.0 - dominance,
        "method_comparison": method_comparison(rows),
        "note": "a pure location may win some graphs; a near-total pure dominance would make placement trivial",
    }, severity="warn" if dominance <= 0.95 else "fail")

    counts_status = {"certified_feasible": 0, "witness_not_found": 0, "stress_or_infeasible": 0}
    for r in rows:
        counts_status[r["status"]] += 1
    total = len(rows)
    rates = {k: v / total for k, v in counts_status.items()} if total else {}
    audit("7_feasibility", True, {
        "status_counts": counts_status, "status_rates": rates,
        "certified_feasible_rate": rates.get("certified_feasible", 0.0),
        "witness_not_found_rate": rates.get("witness_not_found", 0.0),
        "stress_or_infeasible_rate": rates.get("stress_or_infeasible", 0.0),
        "requirements_rewritten": False,
        "min_relative_margin": min(((r["D_G_s"] - r["best_makespan_s"]) / r["D_G_s"]
                                    for r in rows), default=None),
        "population_feasibility_saturated": all(r["status"] == "certified_feasible" for r in rows),
        "saturation_note": (
            "the frozen family deadline ranges sit above the achievable makespan of the "
            "frozen resource model, so feasibility is saturated by construction; difficulty "
            "in this population is expressed through the optimality gap rho_G and the "
            "placement structure, not through infeasibility. Requirements are NOT tightened "
            "here to manufacture failures."),
        "note": "witness_not_found is not mathematical infeasibility; no requirement was relaxed",
    })

    audit("8_split_leakage", split_result["status"] == "PASS", {
        "leakage_status": split_result["status"],
        "violations": split_result["violations"],
        "counts": split_result["summary"]["counts"],
        "role_counts": split_result["summary"]["role_counts"],
    })

    determ = determinism_check(docs)
    audit("9_deterministic_regeneration", determ["status"] == "PASS", determ)

    prov_ok = all(label in ad.PROVENANCE_LABELS for label in
                  (e["label"] for e in graphs[0]["provenance"].values()))
    required_groups = {"t_ref_fixed_planning_motif", "t_ref_range_roles", "compute_workload_bytes",
                       "payload_bytes", "resource_rates", "graph_deadline", "subdeadlines",
                       "criticality", "execution_budgets", "mode_semantics", "topology"}
    present = set(graphs[0]["provenance"])
    audit("10_provenance", prov_ok and required_groups <= present, {
        "field_groups": sorted(present),
        "missing_groups": sorted(required_groups - present),
        "labels": sorted({e["label"] for e in graphs[0]["provenance"].values()}),
        "label_vocabulary": list(ad.PROVENANCE_LABELS),
        "contract_label_vocabulary": list(ad.CONTRACT_PROVENANCE_LABELS),
        "card_extra_labels": list(ad.CARD_EXTRA_PROVENANCE_LABELS),
        "unresolved_groups": sorted(g for g in present
                                    if graphs[0]["provenance"][g]["label"] not in ad.PROVENANCE_LABELS),
        "note": "every generated numeric group resolves to a verified source, a derived rule or a frozen source-calibrated-synthetic rule",
    })

    mixed_useful = sum(1 for r in rows if r["mixed_plan_useful"])
    mixed_rate = mixed_useful / total if total else 0.0
    audit("11_mixed_placement", mixed_rate >= 0.10, {
        "mixed_placement_useful_rate": mixed_rate,
        "mixed_placement_useful_graphs": mixed_useful,
        "threshold": 0.10,
        "note": "a mixed plan matters when it strictly beats every pure-location plan",
    }, severity="fail" if mixed_rate <= 0.0 else "warn")

    fam_rows = {}
    for fam in sorted({g["application_family"] for g in graphs}):
        ids = {g["graph_id"] for g in graphs if g["application_family"] == fam}
        fr = [r for r in rows if r["graph_id"] in ids]
        fam_rows[fam] = {
            "graphs": len(ids),
            "certified_feasible": sum(1 for r in fr if r["status"] == "certified_feasible"),
            "witness_not_found": sum(1 for r in fr if r["status"] == "witness_not_found"),
            "stress_or_infeasible": sum(1 for r in fr if r["status"] == "stress_or_infeasible"),
            "median_slack_ratio": statistics.median(
                [max(t["slack_s"] for t in g["tasks"]) / g["D_G_s"]
                 for g in graphs if g["application_family"] == fam]) if ids else 0.0,
            "high_share": (sum(g["criticality_counts"].get("HIGH", 0) for g in graphs
                               if g["application_family"] == fam)
                           / (20 * len(ids))) if ids else 0.0,
        }
    trivial = [f for f, v in fam_rows.items()
               if v["graphs"] and v["certified_feasible"] == v["graphs"]
               and v["median_slack_ratio"] > 0.5]
    audit("12_family_balance", bool(fam_rows) and all(v["graphs"] > 0 for v in fam_rows.values()), {
        "families": fam_rows,
        "trivially_loose_families": trivial,
        "note": "a declared non-safety family (mapping_background) is expected to be loose; "
                "that is declared slack, not a bug",
    }, severity="info")

    failed = [k for k, v in audits.items() if v["status"] == "FAIL"]
    return {"audits": audits, "failed": failed,
            "status": "PASS" if not failed else "FAIL"}


def determinism_check(docs: dict) -> dict:
    tmp_a = Path(tempfile.mkdtemp(prefix="mcv1_det_a_"))
    tmp_b = Path(tempfile.mkdtemp(prefix="mcv1_det_b_"))
    try:
        ra = ag.materialize(tmp_a, docs)
        rb = ag.materialize(tmp_b, docs)
        details = {
            "graph_count_equal": ra["graph_count"] == rb["graph_count"],
            "graphs_sha_equal": ra["graphs_sha256"] == rb["graphs_sha256"],
            "dataset_manifest_sha_equal": ra["dataset_manifest_sha256"] == rb["dataset_manifest_sha256"],
            "canonical_hashes_equal": [g["canonical_sha256"] for g in ra["graphs"]] ==
                                      [g["canonical_sha256"] for g in rb["graphs"]],
            "raw_hashes_equal": [g["raw_sha256"] for g in ra["graphs"]] ==
                                [g["raw_sha256"] for g in rb["graphs"]],
            "semantic_annotations_equal": all(
                a["criticality_mixture"] == b["criticality_mixture"]
                and a["mode_semantics"] == b["mode_semantics"]
                and a["sla_id"] == b["sla_id"] for a, b in zip(ra["graphs"], rb["graphs"])),
            "lineage_fields_equal": all(
                a["template_lineage"] == b["template_lineage"]
                and a["parent_seed"] == b["parent_seed"]
                and a["topology_signature"] == b["topology_signature"]
                and a["workload_regime"] == b["workload_regime"]
                and a["resource_profile"] == b["resource_profile"]
                and a["sla_regime"] == b["sla_regime"]
                and a["criticality_mixture"] == b["criticality_mixture"]
                for a, b in zip(ra["graphs"], rb["graphs"])),
            "materialized_files_equal": all(
                (tmp_a / n).read_bytes() == (tmp_b / n).read_bytes()
                for n in ag.MATERIALIZED_FILES),
            "graphs_sha256": ra["graphs_sha256"],
            "dataset_manifest_sha256": ra["dataset_manifest_sha256"],
        }
        details["status"] = "PASS" if all(v for k, v in details.items()
                                          if k not in ("graphs_sha256", "dataset_manifest_sha256",
                                                       "status")) else "FAIL"
        return details
    finally:
        shutil.rmtree(tmp_a, ignore_errors=True)
        shutil.rmtree(tmp_b, ignore_errors=True)


# --------------------------------------------------------------------------- #
# Calibration (meta-train only)
# --------------------------------------------------------------------------- #
def calibration_report(graphs: list[dict], assignment: dict) -> dict:
    train = [g for g in graphs if assignment[g["graph_id"]]["split"] == "meta_train"]
    other = [g for g in graphs if assignment[g["graph_id"]]["split"] != "meta_train"]
    t_ref = [float(t["t_ref_s"]) for g in train for t in g["tasks"]]
    workload = [int(t["compute_workload_bytes"]) for g in train for t in g["tasks"]]
    dg = [float(g["D_G_s"]) for g in train]
    slack = [float(t["slack_s"]) for g in train for t in g["tasks"]]
    payloads: dict[str, list[int]] = {}
    for g in train:
        for e in g["edges"]:
            payloads.setdefault(e["payload_class"], []).append(int(e["payload_bytes"]))
    rates = {k: [float(g["resource"][k]) for g in train] for k in
             ("f_ue_hz", "f_helper_hz", "f_mec_hz", "r_mec_ul_bps", "r_mec_dl_bps", "r_v2v_bps")}
    per_family: dict[str, dict] = {}
    for g in train:
        fam = g["application_family"]
        per_family.setdefault(fam, {"graphs": 0, "D_G_s": [], "t_ref_s": []})
        per_family[fam]["graphs"] += 1
        per_family[fam]["D_G_s"].append(float(g["D_G_s"]))
        per_family[fam]["t_ref_s"].extend(float(t["t_ref_s"]) for t in g["tasks"])
    return {
        "schema_version": "automotive_calibration_report_v1",
        "dataset_version": ad.DATASET_VERSION,
        "calibration_source": "meta_train",
        "calibration_graph_count": len(train),
        "excluded_graph_count": len(other),
        "excluded_splits": sorted({assignment[g["graph_id"]]["split"] for g in other}),
        "parameters": {
            "workload_scale_bytes": statistics.fmean(workload),
            "workload_scale_bytes_definition": "mean W_i over meta-train tasks",
            "latency_scale_s": statistics.median(dg),
            "latency_scale_s_definition": "median D_G over meta-train graphs",
            "deadline_slack_scale_s": statistics.median(slack),
            "energy_scale_j_diagnostic": None,
            "cycles_per_bit": XI,
            "f_ref_hz": float(ad.load_inputs()["workload_model_v2.yaml"]
                              ["reference_compute_model"]["f_ref_hz"]),
        },
        "statistics": {
            "t_ref_s": {"min": min(t_ref), "median": statistics.median(t_ref), "max": max(t_ref)},
            "W_i_bytes": {"min": min(workload), "median": statistics.median(workload),
                          "max": max(workload)},
            "D_G_s": {"min": min(dg), "median": statistics.median(dg), "max": max(dg)},
            "slack_s": {"min": min(slack), "median": statistics.median(slack), "max": max(slack)},
            "payload_bytes": {k: {"min": min(v), "median": statistics.median(v), "max": max(v)}
                              for k, v in sorted(payloads.items())},
            "resource_rates": {k: {"min": min(v), "median": statistics.median(v), "max": max(v)}
                               for k, v in sorted(rates.items())},
            "per_family": {k: {"graphs": v["graphs"],
                               "D_G_s_median": statistics.median(v["D_G_s"]),
                               "t_ref_s_median": statistics.median(v["t_ref_s"])}
                           for k, v in sorted(per_family.items())},
        },
        "isolation": {
            "validation_graphs_used": 0,
            "meta_test_graphs_used": 0,
            "rule": "calibration may use meta-train only",
        },
    }


# --------------------------------------------------------------------------- #
# Top-level certification
# --------------------------------------------------------------------------- #
def certify(dataset_dir: Path | None = None, write: bool = True) -> dict:
    d = Path(dataset_dir) if dataset_dir else ad.DATASET_DIR
    # captured BEFORE any certification output is written, so `git_dirty` describes
    # the tree the frozen inputs came from, not the tree certification just touched
    git_state = {"git_sha": _git_sha(), "git_dirty": _git_dirty()}
    docs = ad.load_inputs()
    graphs, _ = asp.load_graphs(d)
    graphs_before = ad.sha256_file(d / "graphs.jsonl")
    policy = json.loads((ad.SPEC_DIR / "split_policy.json").read_text())
    split_result = asp.run(d, check_only=True)
    assignment = split_result["assignment"]

    tpl_by_family = ad.templates_by_family(docs)
    ref_cp_by_family = {fam: ad.reference_tier_critical_path(tpl, docs["workload_model_v2.yaml"])
                        for fam, tpl in tpl_by_family.items()}
    rows: list[dict] = []
    for g in graphs:
        cert = certify_graph(g, ref_cp_by_family[g["application_family"]])
        cert["graph_id"] = g["graph_id"]
        cert["application_family"] = g["application_family"]
        cert["split"] = assignment[g["graph_id"]]["split"]
        cert["role_in_split"] = assignment[g["graph_id"]]["role_in_split"]
        rows.append(cert)

    audits = run_audits(graphs, assignment, policy, rows, docs, split_result)
    feasible_rows = [r for r in rows if r["status"] == "certified_feasible"]
    method_stats = method_comparison(rows)

    feasibility = {
        "schema_version": "automotive_feasibility_report_v1",
        "dataset_version": ad.DATASET_VERSION,
        "certification_version": CERTIFICATION_VERSION,
        "certified_on": "frozen dataset (no requirement was rewritten)",
        "search_budget_evaluations_per_graph": SEARCH_BUDGET,
        "status_counts": {s: sum(1 for r in rows if r["status"] == s)
                          for s in ("certified_feasible", "witness_not_found",
                                    "stress_or_infeasible")},
        "status_rates": {s: (sum(1 for r in rows if r["status"] == s) / len(rows) if rows else 0.0)
                         for s in ("certified_feasible", "witness_not_found",
                                   "stress_or_infeasible")},
        "rho_G": {
            "definition": "T_reference / D_G, DIAGNOSTIC ONLY",
            "reference_method": "heft_reference",
            "uses_best_found": True,
            "min": min((r["rho_G"] for r in rows), default=None),
            "median": statistics.median([r["rho_G"] for r in rows]) if rows else None,
            "max": max((r["rho_G"] for r in rows), default=None),
            "heft_median": statistics.median([r["heft_rho_G"] for r in rows]) if rows else None,
            "never_redefines_D_G": True,
        },
        "reference_tier_stress_graphs": sum(1 for r in rows if r["reference_tier_stress"]),
        "mixed_placement_useful_rate": (sum(1 for r in rows if r["mixed_plan_useful"]) / len(rows)
                                        if rows else 0.0),
        "pure_plan_best_share": (sum(1 for r in rows if r["best_plan_is_pure"]) / len(rows)
                                 if rows else 0.0),
        "method_comparison": method_stats,
        "rows": rows,
    }

    for r in rows:
        for name, m in r["methods"].items():
            if name != r["best_method"] and name not in _KEEP_PLANS:
                m.pop("plan", None)
    if write:
        after = ad.sha256_file(d / "graphs.jsonl")
        if after != graphs_before:
            raise ad.DatasetError("certification rewrote graphs.jsonl")
        (d / "calibration_report.json").write_text(
            json.dumps(calibration_report(graphs, assignment), indent=2, sort_keys=True) + "\n")
        (d / "feasibility_report.json").write_text(
            json.dumps(feasibility, indent=2, sort_keys=True) + "\n")
        _write_manifest(d, graphs, assignment, rows)
        (d / "audit_report.json").write_text(
            json.dumps(audits, indent=2, sort_keys=True) + "\n")
        _write_provenance(d, graphs, assignment, audits, split_result, docs, git_state)
    return {"rows": rows, "audits": audits, "feasibility": feasibility,
            "split_result": split_result, "calibration": calibration_report(graphs, assignment)}


def _write_manifest(d: Path, graphs: list[dict], assignment: dict, rows: list[dict]) -> None:
    by_id = {r["graph_id"]: r for r in rows}
    lines = []
    for g in graphs:
        a = assignment[g["graph_id"]]
        r = by_id[g["graph_id"]]
        base = ag.dataset_manifest_record(g)
        base.update({
            "split": a["split"],
            "role_in_split": a["role_in_split"],
            "certification": {
                "certification_version": CERTIFICATION_VERSION,
                "status": r["status"],
                "best_method": r["best_method"],
                "best_makespan_s": r["best_makespan_s"],
                "D_G_s": r["D_G_s"],
                "rho_G": r["rho_G"],
                "lower_bound_makespan_s": r["lower_bound_makespan_s"],
                "search_evaluations": r["search_evaluations"],
                "requirements_rewritten": False,
            },
        })
        lines.append(ad.canonical_json(base))
    (d / "manifest.jsonl").write_text("".join(l + "\n" for l in lines))


def _write_provenance(d: Path, graphs: list[dict], assignment: dict, audits: dict,
                      split_result: dict, docs: dict, git_state: dict | None = None) -> None:
    registry_pin = json.loads((ad.SPEC_DIR / "registry_pin.json").read_text())
    git_state = git_state or {"git_sha": _git_sha(), "git_dirty": _git_dirty()}
    evidence_files = sorted((ad.SPEC_DIR / "evidence").glob("*"))
    prov = {
        "schema_version": "automotive_provenance_v1",
        "dataset_version": ad.DATASET_VERSION,
        "git_sha": git_state["git_sha"],
        "git_dirty": git_state["git_dirty"],
        "git_sha_semantics": (
            "the commit whose tree contained the frozen inputs when certification "
            "started; the certification outputs themselves are committed next"),
        "git_dirty_semantics": (
            "working-tree state at certification start; it is true on the first "
            "certification of a new milestone because the untracked certification "
            "sources and outputs are not committed yet, not because a frozen input "
            "was modified"),
        "source_registry_sha": registry_pin["source_registry_sha256"],
        "required_parameter_manifest_sha": ad.sha256_file(
            ad.SPEC_DIR / "REQUIRED_PARAMETER_MANIFEST.yaml"),
        "evidence_shas": {p.name: ad.sha256_file(p) for p in evidence_files},
        "registry_pin": registry_pin,
        "workload_model_v2_sha": ad.sha256_file(ad.SPEC_DIR / "workload_model_v2.yaml"),
        "resource_profile_sha": ad.sha256_file(ad.SPEC_DIR / "resource_profiles.yaml"),
        "sla_registry_v2_sha": ad.sha256_file(ad.SPEC_DIR / "sla_registry_v2.yaml"),
        "deadline_model_v2_sha": ad.sha256_file(ad.SPEC_DIR / "deadline_model_v2.py"),
        "criticality_policy_sha": ad.sha256_file(ad.SPEC_DIR / "criticality_policy.yaml"),
        "criticality_model_sha": ad.sha256_file(ad.SPEC_DIR / "criticality_model.py"),
        "task_semantics_sha": ad.sha256_file(ad.SPEC_DIR / "task_semantics.yaml"),
        "application_templates_sha": ad.sha256_file(ad.SPEC_DIR / "application_templates.yaml"),
        "generator_sha": ad.sha256_file(ad.SPEC_DIR / "automotive_generator.py"),
        "generator_library_sha": ad.sha256_file(ad.SPEC_DIR / "automotive_dataset.py"),
        "generation_config_sha": ad.sha256_file(ad.SPEC_DIR / "generation_config.yaml"),
        "split_policy_sha": ad.sha256_file(ad.SPEC_DIR / "split_policy.json"),
        "split_module_sha": ad.sha256_file(ad.SPEC_DIR / "automotive_splits.py"),
        "certification_module_sha": ad.sha256_file(ad.SPEC_DIR / "certify_automotive_m10.py"),
        "dataset_manifest_sha": ad.sha256_file(d / "dataset_manifest.jsonl"),
        "certification_manifest_sha": ad.sha256_file(d / "manifest.jsonl"),
        "graphs_sha": ad.sha256_file(d / "graphs.jsonl"),
        "splits_sha": ad.sha256_file(d / "splits.jsonl"),
        "split_summary_sha": ad.sha256_file(d / "split_summary.json"),
        "calibration_report_sha": ad.sha256_file(d / "calibration_report.json"),
        "feasibility_report_sha": ad.sha256_file(d / "feasibility_report.json"),
        "audit_report_sha": ad.sha256_file(d / "audit_report.json"),
        "audit_status": audits["status"],
        "leakage_status": split_result["status"],
        "graph_count": len(graphs),
        "split_counts": split_result["summary"]["counts"],
        "input_sha256": {k: v for k, v in sorted(ad.inputs_sha256(docs).items())},
    }
    (d / "provenance.json").write_text(json.dumps(prov, indent=2, sort_keys=True) + "\n")


def _git_sha() -> str:
    import subprocess
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ad.REPO_ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def _git_dirty() -> bool | str:
    import subprocess
    try:
        out = subprocess.run(["git", "status", "--porcelain"], cwd=ad.REPO_ROOT,
                             capture_output=True, text=True, check=True).stdout.strip()
        return bool(out)
    except Exception:
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(ad.DATASET_DIR))
    args = ap.parse_args()
    res = certify(Path(args.dataset))
    print(json.dumps({
        "status": res["audits"]["status"],
        "failed_audits": res["audits"]["failed"],
        "status_counts": res["feasibility"]["status_counts"],
        "status_rates": res["feasibility"]["status_rates"],
        "mixed_placement_useful_rate": res["feasibility"]["mixed_placement_useful_rate"],
        "pure_plan_best_share": res["feasibility"]["pure_plan_best_share"],
        "rho_G_median": res["feasibility"]["rho_G"]["median"],
        "leakage": res["split_result"]["status"],
    }, indent=2, sort_keys=True))
    return 1 if res["audits"]["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
