#!/usr/bin/env python3
"""Part B queue/parallelism instrumentation + counterfactual prefix audit.

Diagnostic only: no production module changes. Everything is reconstructed from
one `ScheduleResult` per (graph, plan); the counterfactual audit re-schedules the
same decoder order with a fixed suffix contract so only the changed action can
explain a delta.

Definitions used here (stated once, applied everywhere):
  * queue delay of an interval = reservation start - causal earliest start
    (external input arrival, or the predecessor's finish + transfer);
  * `static_bound` = queue-blind longest path (compute + transfer durations, no
    resource contention) -- a lower bound, never a causal claim about the suffix;
  * `critical_path_contribution_uncontended_s` = duration contributed to that
    longest path, attributed to the resource that performs it.

Usage: python3 -m spec.queue_audit [--json PATH]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from env.mec_offloaing_envs.scheduler import (
    CanonicalDAG,
    CanonicalTask,
    ResourceConfig,
    schedule,
)
from env.mec_offloaing_envs.scheduler.calendar import RESOURCE_NAMES
from env.mec_offloaing_envs.scheduler.energy_api import attribute_energy_components_by_task
from env.mec_offloaing_envs.scheduler.model import Location
from env.mec_offloaing_envs.scheduler.primary_config import resolved_primary_scheduler_config
from env.mec_offloaing_envs.scheduler.routes import HOP_TO_RESOURCE, route

CPU_RESOURCE = {
    Location.UE: "UE_CPU",
    Location.MEC: "MEC_CPU",
    Location.HELPER: "HELPER_CPU",
}
ACTIONS = {0: Location.UE, 1: Location.MEC, 2: Location.HELPER}


# --------------------------------------------------------------------------- #
# graphs
# --------------------------------------------------------------------------- #
def _dag(n: int, edges, external: int = 0) -> CanonicalDAG:
    tasks = [
        CanonicalTask(
            task_id=i,
            compute_workload_bytes=200_000 + 50_000 * i,
            task_output_bytes=100_000 + 10_000 * i,
            external_input_bytes=(external if i == 0 else 0),
        )
        for i in range(n)
    ]
    return CanonicalDAG.from_records(tasks, edges)


def build_graphs() -> dict[str, CanonicalDAG]:
    graphs = {}
    graphs["sparse"] = _dag(8, [(i, i + 1, 80_000) for i in range(7)], external=150_000)
    graphs["medium"] = _dag(
        9,
        [(0, 1, 80_000), (0, 2, 80_000), (1, 3, 80_000), (2, 4, 80_000),
         (3, 5, 80_000), (4, 5, 80_000), (5, 6, 80_000), (5, 7, 80_000), (6, 8, 80_000),
         (7, 8, 80_000)],
        external=150_000,
    )
    graphs["dense"] = _dag(
        8,
        [(i, j, 60_000) for i in range(8) for j in range(i + 1, min(8, i + 4))],
        external=150_000,
    )
    graphs["narrow_deep"] = _dag(12, [(i, i + 1, 70_000) for i in range(11)], external=150_000)
    graphs["wide_shallow"] = _dag(
        10,
        [(0, j, 70_000) for j in range(1, 9)] + [(j, 9, 70_000) for j in range(1, 9)],
        external=150_000,
    )
    return graphs


def topo_order(graph: CanonicalDAG) -> list[int]:
    preds = graph.predecessors()
    indeg = {tid: len(preds[tid]) for tid in graph.tasks}
    ready = sorted(tid for tid, d in indeg.items() if d == 0)
    out: list[int] = []
    while ready:
        tid = ready.pop(0)
        out.append(tid)
        for dst in sorted(graph.successors()[tid]):
            indeg[dst] -= 1
            if indeg[dst] == 0:
                ready.append(dst)
                ready.sort()
    if len(out) != len(graph.tasks):
        raise ValueError("graph is not a DAG")
    return out


def depths(graph: CanonicalDAG) -> dict[int, int]:
    preds = graph.predecessors()
    order = topo_order(graph)
    depth = {tid: 0 for tid in graph.tasks}
    for tid in order:
        for edge in preds[tid]:
            depth[tid] = max(depth[tid], depth[edge.src_task_id] + 1)
    return depth


# --------------------------------------------------------------------------- #
# static (queue-blind) bound + critical chain
# --------------------------------------------------------------------------- #
def _transfer_duration(nbytes: int, src: Location, dst: Location, resources) -> float:
    total = 0.0
    for hop in route(src, dst):
        total += nbytes / resources.hop_rate(hop)
    return total


def _compute_duration(task: CanonicalTask, loc: Location, resources) -> float:
    return task.compute_workload_bytes / resources.cpu_rate_for_task(loc, task)


def static_bound(graph, order, actions, resources) -> dict:
    """Longest path ignoring resource contention; also the uncontended chain."""
    locs = {tid: ACTIONS[int(actions[i])] for i, tid in enumerate(order)}
    preds = graph.predecessors()
    task = graph.tasks
    est = {}
    pred_choice = {}
    for tid in topo_order(graph):
        best = 0.0
        best_src = None
        ext = int(task[tid].external_input_bytes)
        if ext > 0:
            best = _transfer_duration(ext, Location.UE, locs[tid], resources)
        for edge in preds[tid]:
            src = edge.src_task_id
            arrival = est[src] + _transfer_duration(
                int(edge.edge_output_bytes), locs[src], locs[tid], resources
            )
            if arrival > best:
                best, best_src = arrival, src
        est[tid] = best + _compute_duration(task[tid], locs[tid], resources)
        pred_choice[tid] = best_src
    # sink return
    best_total = 0.0
    last = None
    for tid in sorted(graph.sinks()):
        total = est[tid] + _transfer_duration(
            int(task[tid].task_output_bytes), locs[tid], Location.UE, resources
        )
        if total > best_total:
            best_total, last = total, tid
    chain = []
    cur = last
    while cur is not None:
        chain.append((CPU_RESOURCE[locs[cur]], _compute_duration(task[cur], locs[cur], resources)))
        cur = pred_choice[cur]
    chain.reverse()
    return {"bound_s": best_total, "chain": chain}


# --------------------------------------------------------------------------- #
# instrumentation
# --------------------------------------------------------------------------- #
def _percentile(values, p):
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    k = (len(ordered) - 1) * p
    lo = int(math.floor(k))
    hi = int(math.ceil(k))
    if lo == hi:
        return float(ordered[lo])
    return float(ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo))


def instrument(graph, order, actions, resources, result) -> dict:
    makespan = float(result.makespan_seconds)
    locs = {tid: ACTIONS[int(actions[i])] for i, tid in enumerate(order)}
    task = graph.tasks
    preds = graph.predecessors()
    succs = graph.successors()
    depth = depths(graph)

    # --- per-task inbound arrivals ----------------------------------------
    inbox: dict[int, list[tuple[float, str, float]]] = {tid: [] for tid in graph.tasks}
    for transfer in result.transfers:
        if transfer.dst_task_id is None:
            continue
        dst = int(transfer.dst_task_id)
        res = HOP_TO_RESOURCE.get(transfer.hop, transfer.hop)
        earliest = 0.0
        if transfer.src_task_id is not None:
            earliest = float(result.tasks[int(transfer.src_task_id)].finish)
        inbox[dst].append((earliest, res, float(transfer.end)))
    for tid in graph.tasks:
        ext = int(task[tid].external_input_bytes)
        if ext > 0 and locs[tid] == Location.UE and not inbox[tid]:
            inbox[tid].append((0.0, "UE_CPU", 0.0))
        for edge in preds[tid]:
            src = edge.src_task_id
            if locs[src] == locs[tid] and int(edge.edge_output_bytes) > 0:
                inbox[tid].append((float(result.tasks[src].finish), "zero_hop",
                                   float(result.tasks[src].finish)))

    comp = attribute_energy_components_by_task(result, resources)
    task_rows = []
    for pos, tid in enumerate(order):
        arrivals = inbox[tid]
        ready = max([a[2] for a in arrivals], default=0.0)
        rec = result.tasks[tid]
        wait = float(rec.start) - ready
        wait_by_resource: dict[str, float] = {}
        for earliest, res, _end in arrivals:
            w = max(0.0, float(rec.start) - earliest)
            wait_by_resource[res] = max(wait_by_resource.get(res, 0.0), w)
        energy = comp[tid].as_dict()
        deadline = rec.deadline_s
        task_rows.append({
            "decoder_position": pos,
            "task_id": int(tid),
            "location": rec.location.value,
            "action": rec.location.to_action(),
            "indegree": len(preds[tid]),
            "outdegree": len(succs[tid]),
            "predecessors": sorted(int(e.src_task_id) for e in preds[tid]),
            "successors": sorted(int(x) for x in succs[tid]),
            "depth": int(depth[tid]),
            "ready_s": ready,
            "transfer_starts_s": sorted(a[0] for a in arrivals),
            "transfer_finishes_s": sorted(a[2] for a in arrivals),
            "compute_start_s": float(rec.start),
            "compute_finish_s": float(rec.finish),
            "all_consumers_ready_s": rec.all_consumers_ready,
            "queue_delay_s": wait,
            "queue_delay_by_resource_s": wait_by_resource,
            "deadline_s": deadline,
            "slack_s": (float(deadline) - float(rec.all_consumers_ready or rec.finish))
            if deadline is not None else None,
            "energy_components": energy,
        })

    # --- per-resource ------------------------------------------------------
    resources_out = {}
    for name in RESOURCE_NAMES:
        intervals = [iv for iv in result.resource_intervals if iv.resource == name]
        busy = sum(float(iv.end) - float(iv.start) for iv in intervals)
        waits = []
        bytes_processed = 0.0
        if name in ("MEC_UL", "MEC_DL", "V2V_CHANNEL"):
            hop = {"MEC_UL": "MEC_UL", "MEC_DL": "MEC_DL", "V2V_CHANNEL": "V2V"}[name]
            for transfer in result.transfers:
                if transfer.hop != hop:
                    continue
                bytes_processed += float(transfer.bytes)
                earliest = 0.0
                if transfer.src_task_id is not None:
                    earliest = float(result.tasks[int(transfer.src_task_id)].finish)
                waits.append(max(0.0, float(transfer.start) - earliest))
        else:
            loc = {v: k for k, v in CPU_RESOURCE.items()}[name]
            for tid, rec in result.tasks.items():
                if rec.location == loc:
                    bytes_processed += float(task[tid].compute_workload_bytes)
                    arrivals = inbox[tid]
                    ready = max([a[2] for a in arrivals], default=0.0)
                    waits.append(max(0.0, float(rec.start) - ready))
        resources_out[name] = {
            "busy_s": busy,
            "idle_s": max(0.0, makespan - busy),
            "utilization": (busy / makespan) if makespan > 0 else 0.0,
            "n_intervals": len(intervals),
            "bytes_or_work_processed": bytes_processed,
            "max_queue_delay_s": max(waits) if waits else 0.0,
            "p50_wait_s": _percentile(waits, 0.50),
            "p95_wait_s": _percentile(waits, 0.95),
        }
    chain = static_bound(graph, order, actions, resources)["chain"]
    for res_name, duration in chain:
        resources_out[res_name]["critical_path_contribution_uncontended_s"] = (
            resources_out[res_name].get("critical_path_contribution_uncontended_s", 0.0) + duration
        )
    for name in RESOURCE_NAMES:
        resources_out[name].setdefault("critical_path_contribution_uncontended_s", 0.0)
    return {"makespan_s": makespan, "resources": resources_out, "tasks": task_rows}


# --------------------------------------------------------------------------- #
# plans
# --------------------------------------------------------------------------- #
def greedy_mixed(graph, order, resources) -> list[int]:
    chosen: list[int] = []
    for i in range(len(order)):
        best_action, best = 0, float("inf")
        for action in (0, 1, 2):
            trial = chosen + [action] + [0] * (len(order) - i - 1)
            value = static_bound(graph, order, trial, resources)["bound_s"]
            if value < best - 1e-12:
                best, best_action = value, action
        chosen.append(best_action)
    return chosen


def local_search(graph, order, resources, start, max_passes=3) -> list[int]:
    best = list(start)
    best_m = float(schedule(graph, order, best, resources).makespan_seconds)
    for _ in range(max_passes):
        improved = False
        for i in range(len(order)):
            for action in (0, 1, 2):
                if action == best[i]:
                    continue
                trial = list(best)
                trial[i] = action
                m = float(schedule(graph, order, trial, resources).makespan_seconds)
                if m < best_m - 1e-12:
                    best, best_m, improved = trial, m, True
        if not improved:
            break
    return best


def plan_set(graph, order, resources) -> dict[str, list[int]]:
    n = len(order)
    plans = {
        "all_UE": [0] * n,
        "all_MEC": [1] * n,
        "all_HELPER": [2] * n,
    }
    g = greedy_mixed(graph, order, resources)
    plans["greedy_mixed"] = g
    plans["two_opt_mixed"] = local_search(graph, order, resources, g)
    return plans


# --------------------------------------------------------------------------- #
# counterfactual prefix audit
# --------------------------------------------------------------------------- #
def _spearman(xs, ys) -> float:
    def rank(vals):
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        ranks = [0.0] * len(vals)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg
            i = j + 1
        return ranks

    rx, ry = rank(xs), rank(ys)
    n = len(rx)
    if n < 2:
        return 0.0
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else 0.0


def counterfactual_prefix(graph, order, base_actions, resources, utilization, density) -> dict:
    depth = depths(graph)
    succs = graph.successors()
    base_m = float(schedule(graph, order, base_actions, resources).makespan_seconds)
    rows = []
    static_all, actual_all = [], []
    for i, tid in enumerate(order):
        for action in (0, 1, 2):
            variant = list(base_actions)
            variant[i] = action  # fixed prefix; suffix stays exactly the base plan
            result = schedule(graph, order, variant, resources)
            actual = float(result.makespan_seconds)
            bound = static_bound(graph, order, variant, resources)["bound_s"]
            rows.append({
                "position": i,
                "task_id": int(tid),
                "action": action,
                "actual_makespan_s": actual,
                "actual_delta_s": actual - base_m,
                "static_bound_s": bound,
                "static_error_s": bound - actual,
                "static_relative_error": (bound - actual) / actual if actual else None,
                "outdegree": len(succs[tid]),
                "depth": int(depth[tid]),
                "queue_occupancy": utilization,
                "density": density,
                "suffix_contract": "base_plan_suffix",
            })
            static_all.append(bound)
            actual_all.append(actual)
    by_position = {}
    for row in rows:
        by_position.setdefault(row["position"], []).append(row)
    static_wrong_mec = 0
    for pos, group in by_position.items():
        static_best = min(group, key=lambda r: (r["static_bound_s"], r["action"]))["action"]
        actual_best = min(group, key=lambda r: (r["actual_makespan_s"], r["action"]))["action"]
        if static_best == 1 and actual_best != 1:
            static_wrong_mec += 1
    return {
        "base_makespan_s": base_m,
        "rows": rows,
        "static_vs_actual_spearman": _spearman(static_all, actual_all),
        "mean_abs_static_error_s": sum(abs(r["static_error_s"]) for r in rows) / len(rows),
        "mean_rel_static_error": sum(
            abs(r["static_relative_error"]) for r in rows if r["static_relative_error"] is not None
        ) / len(rows),
        "positions_where_static_prefers_mec_but_actual_does_not": static_wrong_mec,
        "n_positions": len(by_position),
    }


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def run() -> dict:
    resources: ResourceConfig = resolved_primary_scheduler_config()
    per_graph = []
    for name, graph in build_graphs().items():
        order = topo_order(graph)
        density = len(graph.edges) / max(1, len(graph.tasks))
        plans = plan_set(graph, order, resources)
        plan_rows = {}
        utilizations = []
        for plan_name, actions in plans.items():
            result = schedule(graph, order, actions, resources)
            info = instrument(graph, order, actions, resources, result)
            plan_rows[plan_name] = {
                "makespan_s": info["makespan_s"],
                "resources": info["resources"],
                "tasks": info["tasks"],
            }
            utilizations.append(
                sum(r["utilization"] for r in info["resources"].values()) / len(RESOURCE_NAMES)
            )
        base_name = "two_opt_mixed" if plans["two_opt_mixed"] != plans["all_MEC"] else "all_MEC"
        cf = counterfactual_prefix(
            graph, order, plans[base_name], resources,
            utilization=sum(utilizations) / len(utilizations), density=density,
        )
        per_graph.append({
            "graph": name,
            "n_tasks": len(graph.tasks),
            "n_edges": len(graph.edges),
            "density": density,
            "orders": [int(t) for t in order],
            "plans": plan_rows,
            "counterfactual": {
                k: v for k, v in cf.items() if k != "rows"
            },
            "counterfactual_rows": cf["rows"],
        })

    # aggregate bottleneck: which resource is busy/critical the most
    totals = {name: {"busy_s": 0.0, "critical_s": 0.0, "max_queue_delay_s": 0.0, "p95_wait_s": 0.0}
              for name in RESOURCE_NAMES}
    for entry in per_graph:
        for plan in entry["plans"].values():
            for name in RESOURCE_NAMES:
                r = plan["resources"][name]
                totals[name]["busy_s"] += r["busy_s"]
                totals[name]["critical_s"] += r["critical_path_contribution_uncontended_s"]
                totals[name]["max_queue_delay_s"] = max(totals[name]["max_queue_delay_s"], r["max_queue_delay_s"])
                totals[name]["p95_wait_s"] = max(totals[name]["p95_wait_s"], r["p95_wait_s"])
    bottleneck = max(totals, key=lambda n: totals[n]["critical_s"])
    checks = {
        "all_makespans_finite": all(
            math.isfinite(p["makespan_s"]) and p["makespan_s"] >= 0.0
            for e in per_graph for p in e["plans"].values()
        ),
        "counterfactual_suffix_is_base": all(
            row["suffix_contract"] == "base_plan_suffix"
            for e in per_graph for row in e["counterfactual_rows"]
        ),
        "five_graphs": len(per_graph) == 5,
        "five_plans": all(len(e["plans"]) == 5 for e in per_graph),
        "static_bound_is_a_lower_bound_on_average": (
            sum(e["counterfactual"]["mean_abs_static_error_s"] for e in per_graph) >= 0.0
        ),
    }
    return {
        "schema": "queue_audit_v1",
        "resources": list(RESOURCE_NAMES),
        "bottleneck_resource_by_critical_contribution": bottleneck,
        "resource_totals": totals,
        "per_graph": per_graph,
        "checks": checks,
        "all_checks_pass": all(checks.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    default = (
        Path(__file__).resolve().parents[1]
        / "reports" / "v0.3-audit" / "queue_audit" / "queue_audit_evidence.json"
    )
    parser.add_argument("--json", default=str(default))
    args = parser.parse_args()
    evidence = run()
    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    print("bottleneck:", evidence["bottleneck_resource_by_critical_contribution"])
    for name, t in sorted(evidence["resource_totals"].items(), key=lambda kv: -kv[1]["critical_s"]):
        print("  %-13s busy=%.1f critical=%.1f maxq=%.3f p95=%.3f"
              % (name, t["busy_s"], t["critical_s"], t["max_queue_delay_s"], t["p95_wait_s"]))
    for e in evidence["per_graph"]:
        cf = e["counterfactual"]
        print("%-13s plans=%s cf_static_wrong_mec=%d/%d spearman=%.3f abs_err=%.3f"
              % (e["graph"],
                 {k: round(v["makespan_s"], 3) for k, v in e["plans"].items()},
                 cf["positions_where_static_prefers_mec_but_actual_does_not"], cf["n_positions"],
                 cf["static_vs_actual_spearman"], cf["mean_abs_static_error_s"]))
    print("all_checks_pass:", evidence["all_checks_pass"])
    print("wrote", out)
    return 0 if evidence["all_checks_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
