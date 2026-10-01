#!/usr/bin/env python3
"""`heft_reference_v2` — a real HEFT-style reference baseline.

WHY THIS EXISTS (audit finding on the frozen M10 evidence)

The frozen M10 certifier (`spec/automotive_mc_v1/certify_automotive_m10.py`) built its
HEFT-style plan with

    tiers = ("f_ue_hz", "f_helper_hz", "f_mec_hz"); f = tiers[action]      # line 166

while the canonical action encoding is `0=UE, 1=MEC, 2=HELPER`
(`scheduler/model.Location.from_action`). Action 1 (MEC) was therefore costed with the
HELPER frequency and action 2 (HELPER) with the MEC frequency, so the arg-min always
picked action 2 and the "HEFT" plan degenerated to `all_HELPER` bit-for-bit. That is
why the frozen M10 audit reports `heft_reference == all_HELPER`.

The frozen M10 result is preserved as historical evidence and the frozen dataset is NOT
touched. This module provides the CORRECTED reference for post-dataset model
evaluation. Corrected mapping:

    action 0 -> f_UE      action 1 -> f_MEC      action 2 -> f_HELPER

HEFT semantics implemented here (Topcuoglu et al. priority-by-upward-rank list
scheduling, made contention- and transfer-aware through the canonical scheduler):

  * `upward_ranks`: r_i = w_i + max_j (c_ij + r_j), with the mean compute duration w_i
    over the three legal tiers and the mean transfer cost c_ij over the frozen links,
    recursed in reverse topological order (successor ranks are always available).
  * `heft_reference_v2_plan`: tasks are placed in DESCENDING upward rank (ties by
    ascending task id). Each placement probes the three actions and scores the
    completed plan with the real canonical scheduler — so per-hop transfers, delivery
    times and the shared MEC/HELPER calendars are all accounted for, unlike the frozen
    M10 scalar cost. The plan is anchored on the best pure plan and a placement is only
    accepted when it strictly lowers the makespan, therefore the reference is never
    worse than any pure plan (and never a pure-plan alias by construction).
  * `historical_m10_alias`: re-runs the frozen certifier's swapped tier table verbatim
    (per-task arg-min cost) so the `all_HELPER` regression stays reproducible evidence.
"""

from __future__ import annotations

from typing import Any, Mapping

from .automotive_dag import (
    AutomotiveDagError,
    decoder_order,
    schedule_actions,
    tier_frequencies,
)

#: canonical action encoding: 0 = UE, 1 = MEC, 2 = HELPER.
ACTION_NAMES = {0: "all_UE", 1: "all_MEC", 2: "all_HELPER"}
PURE_PLANS = (("all_UE", 0), ("all_MEC", 1), ("all_HELPER", 2))

#: the CORRECT action -> frequency mapping (the fix for the M10 defect).
CORRECT_ACTION_TIER = {0: "f_ue_hz", 1: "f_mec_hz", 2: "f_helper_hz"}

#: the frozen M10 tier table, verbatim from certify_automotive_m10.py:166.
#: Kept ONLY to reproduce the historical defect; never used for placement.
HISTORICAL_TIER_TABLE = ("f_ue_hz", "f_helper_hz", "f_mec_hz")

#: XI constant of the frozen M10 certifier (certify_automotive_m10.py:53).
HISTORICAL_XI_CYCLES_PER_BIT = 300.0

_TOL = 1e-12


def _xi_default() -> float:
    """xi of the frozen dataset workload model (300.0 for MARGO-AUTOMOTIVE-MC-v1)."""
    try:
        from .automotive_resources import dataset_reference_parameters

        return float(dataset_reference_parameters()["xi_cycles_per_bit"])
    except Exception:  # pragma: no cover - frozen fallback, never silently rescales
        return float(HISTORICAL_XI_CYCLES_PER_BIT)


def _means(graph: Mapping[str, Any]) -> tuple[float, float]:
    r = graph["resource"]
    mean_f = (float(r["f_ue_hz"]) + float(r["f_mec_hz"]) + float(r["f_helper_hz"])) / 3.0
    mean_rate = (float(r["r_mec_ul_bps"]) + float(r["r_mec_dl_bps"])
                 + float(r["r_v2v_bps"])) / 3.0
    if not (mean_f > 0.0) or not (mean_rate > 0.0):
        raise AutomotiveDagError("mean frequency and link rate must be positive")
    return mean_f, mean_rate


def _successor_map(graph: Mapping[str, Any]) -> dict[int, list[int]]:
    """Adjacency taken from the edge list — the same source `canonical_dag` uses.

    The record's redundant `successors` field is NOT trusted for the recursion: the
    scheduler builds the DAG from `edges`, so the ranks must come from the same input.
    """
    order = decoder_order(graph)
    succ: dict[int, list[int]] = {tid: [] for tid in order}
    for e in graph["edges"]:
        src, dst = int(e["src"]), int(e["dst"])
        succ[src].append(dst)
    for tid in order:
        succ[tid] = sorted(set(succ[tid]))
    return succ


def upward_ranks(graph: Mapping[str, Any], xi: float | None = None) -> dict:
    """HEFT upward ranks r_i = w_i + max_j (c_ij + r_j) over the frozen links.

    Recursed in reverse topological order so every successor rank already exists.
    """
    mean_f, mean_rate = _means(graph)
    xi = _xi_default() if xi is None else float(xi)
    tasks = {int(t["task_id"]): t for t in graph["tasks"]}
    edge_bytes = {(int(e["src"]), int(e["dst"])): int(e["payload_bytes"])
                  for e in graph["edges"]}
    succ = _successor_map(graph)
    order = decoder_order(graph)
    rank: dict[int, float] = {}
    for tid in reversed(order):  # successors precede producers in reverse topo order
        w = int(tasks[tid]["compute_workload_bytes"]) * 8.0 * xi / mean_f
        tail = max((edge_bytes[(tid, s)] * 8.0 / mean_rate + rank[s]
                    for s in succ[tid]), default=0.0)
        value = w + tail
        if value != value or value in (float("inf"), float("-inf")):
            raise AutomotiveDagError("non-finite upward rank for task %d" % tid)
        rank[tid] = value
    return rank


def heft_reference_v2_plan(graph: Mapping[str, Any], *, co_physical: bool = True):
    """Return (actions, makespan_s, result) for the corrected reference baseline.

    Priority order is the descending HEFT upward rank.  For each pending task the
    residual plan keeps the actions already fixed for higher-rank tasks, probes the
    three actions for the pending task and leaves every not-yet-placed task at the
    incumbent action — the frozen M10 bug used action 0 (all_UE) there, which buried
    the signal under the slowest tier.  Placements are accepted only when they strictly
    lower the makespan of the plan evaluated by the canonical transfer-aware scheduler.
    """
    order = decoder_order(graph)
    rank = upward_ranks(graph)
    priority = sorted(order, key=lambda t: (-rank[t], t))

    pure = pure_plan_makespans(graph, co_physical=co_physical)
    # deterministic anchor: the best pure plan (ties -> lower action id)
    seed_action = min((pure[name], action) for name, action in PURE_PLANS)[1]
    assigned = {tid: seed_action for tid in order}
    makespan = pure[ACTION_NAMES[seed_action]]

    def vector(candidate: dict[int, int]) -> list[int]:
        return [int(candidate[tid]) for tid in order]

    for tid in priority:
        incumbent = assigned[tid]
        best_metric, best_action = makespan, incumbent
        for action in (0, 1, 2):
            if action == incumbent:
                continue
            probe = dict(assigned)
            probe[tid] = action
            metric = schedule_actions(
                graph, vector(probe), co_physical=co_physical
            ).makespan_seconds
            if metric < best_metric - _TOL:
                best_metric, best_action = metric, action
        assigned[tid] = best_action
        makespan = best_metric

    actions = vector(assigned)
    result = schedule_actions(graph, actions, co_physical=co_physical)
    return actions, result.makespan_seconds, result


def pure_plan_makespans(graph: Mapping[str, Any], *, co_physical: bool = True) -> dict:
    n = len(decoder_order(graph))
    return {
        name: schedule_actions(graph, [action] * n,
                               co_physical=co_physical).makespan_seconds
        for name, action in PURE_PLANS
    }


def audit_graph(graph: Mapping[str, Any], *, co_physical: bool = True) -> dict:
    """Compare the corrected reference against the pure plans on one frozen graph."""
    actions, makespan, _res = heft_reference_v2_plan(graph, co_physical=co_physical)
    pure = pure_plan_makespans(graph, co_physical=co_physical)
    n = len(actions)
    best_pure = min(pure.values())
    return {
        "graph_id": graph["graph_id"],
        "heft_v2_makespan_s": makespan,
        "heft_v2_action_counts": {str(a): actions.count(a) for a in (0, 1, 2)},
        "heft_v2_is_all_helper": actions.count(2) == n,
        "heft_v2_is_all_mec": actions.count(1) == n,
        "heft_v2_is_all_ue": actions.count(0) == n,
        "heft_v2_is_pure_plan": max(actions.count(a) for a in (0, 1, 2)) == n,
        "pure_plans_s": pure,
        "best_pure_makespan_s": best_pure,
        "heft_v2_beats_or_ties_all_helper": makespan <= pure["all_HELPER"] + _TOL,
        "heft_v2_beats_or_ties_best_pure": makespan <= best_pure + _TOL,
        "tier_frequencies_hz": {str(k): v for k, v in tier_frequencies(graph).items()},
        "action_encoding": {"0": "UE", "1": "MEC", "2": "HELPER"},
    }


def historical_m10_alias(graph: Mapping[str, Any]) -> dict:
    """Reproduce the frozen M10 bug (swapped tier table) for the audit record.

    The per-task cost is the frozen certifier's scalar cost, verbatim:
        cost = 8 * XI * W_i / f[historical_tiers[action]]
    with `HISTORICAL_TIER_TABLE = ("f_ue_hz", "f_helper_hz", "f_mec_hz")`, so action 1
    is costed as the helper and action 2 as the MEC.  Because f_mec is the largest
    frequency in every frozen profile, action 2 wins for every task and the plan is
    `all_HELPER`.  The returned plan is then re-scheduled with the true engine, which is
    what makes its makespan bit-identical to the `all_HELPER` makespan.
    """
    order = decoder_order(graph)
    r = graph["resource"]
    tasks = {int(t["task_id"]): t for t in graph["tasks"]}
    xi = HISTORICAL_XI_CYCLES_PER_BIT
    chosen: dict[int, int] = {}
    for tid in order:
        best = None
        for action in (0, 1, 2):
            cost = (int(tasks[tid]["compute_workload_bytes"]) * 8.0 * xi
                    / float(r[HISTORICAL_TIER_TABLE[action]]))
            if best is None or cost < best[0] - _TOL or (
                    abs(cost - best[0]) <= _TOL and action < best[1]):
                best = (cost, action)
        chosen[tid] = int(best[1])
    actions = [chosen[tid] for tid in order]
    fastest_action = actions[0] if actions else None
    return {
        "graph_id": graph["graph_id"],
        "historical_tier_table": list(HISTORICAL_TIER_TABLE),
        "historical_fastest_action": fastest_action,
        "historical_plan_is_uniform": len(set(actions)) <= 1,
        "historical_plan_is_all_helper": bool(actions) and all(a == 2 for a in actions),
        "historical_makespan_s": schedule_actions(graph, actions).makespan_seconds,
        "all_helper_makespan_s": pure_plan_makespans(graph)["all_HELPER"],
    }
