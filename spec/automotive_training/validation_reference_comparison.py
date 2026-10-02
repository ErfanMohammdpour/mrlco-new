#!/usr/bin/env python3
"""Same-instance reference comparison for the automotive primary.

The training log's `Average greedy latency,` column is NOT a validation-side
comparison: for the automotive env `greedy_solution()` returns the BEST PURE PLAN
(min over all-UE / all-MEC / all-HELPER) on the NOMINAL instance (no MC realization, no
drops, no HI capping) and it is computed once at stack build over the meta-train graphs.
This script produces the comparable numbers instead: deterministic reference plans
evaluated on the SAME validation-query graphs with the SAME mixed-criticality
realizations the validation evaluator uses (base_seed=303, reset_count=1, one slot per
graph). Print the model's logged validation latency next to this table.

CLI: python3 spec/automotive_training/validation_reference_comparison.py [--json PATH]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spec.automotive_training.automotive_constraints import evaluate_constraints  # noqa: E402
from spec.automotive_training.automotive_env import AutomotiveEnv, _constraint_view  # noqa: E402
from spec.automotive_training.automotive_loader import load_dataset  # noqa: E402
from spec.automotive_training.automotive_primary import AutomotiveResourceCluster  # noqa: E402
from spec.automotive_training.heft_reference_v2 import heft_reference_v2_plan  # noqa: E402

VALIDATION_BASE_SEED = 303  # must match AutomotiveHeldOutEvaluator's query rollout


def greedy_coordinate_descent(env, index, mc, n_tasks: int = 20):
    chosen: list[int] = []
    for k in range(n_tasks):
        best = None
        for action in (0, 1, 2):
            plan = chosen + [action] + [0] * (n_tasks - k - 1)
            result, _e, _m = env._schedule(index, plan, mc)
            value = float(result.makespan_seconds)
            if best is None or value < best[0] - 1e-12 or (
                    abs(value - best[0]) <= 1e-12 and action < best[1]):
                best = (value, action)
        chosen.append(int(best[1]))
    return chosen


def run() -> dict:
    dataset = load_dataset()
    graphs = dataset.validation_query()
    env = AutomotiveEnv(graphs, AutomotiveResourceCluster(), role="validation",
                        single_dist=True, slots_per_task=len(graphs),
                        base_seed=VALIDATION_BASE_SEED)
    env.set_task({"dist_index": 0, "graph_indices": np.arange(len(graphs), dtype=np.int32)})
    env.reset()

    names = ["all_UE", "all_MEC", "all_HELPER", "greedy_coordinate_descent_MC",
             "heft_reference_v2_on_MC_instance", "heft_reference_v2_nominal"]
    makespans: dict[str, list[float]] = {n: [] for n in names}
    hard_violations: dict[str, list[bool]] = {n: [] for n in names}
    for index, graph in enumerate(graphs):
        mc = env._slot_mc[index]
        plans = {"all_UE": [0] * 20, "all_MEC": [1] * 20, "all_HELPER": [2] * 20,
                 "greedy_coordinate_descent_MC": greedy_coordinate_descent(env, index, mc)}
        heft_actions, heft_nominal, _res = heft_reference_v2_plan(graph.as_record(),
                                                                 co_physical=True)
        plans["heft_reference_v2_on_MC_instance"] = list(heft_actions)
        makespans["heft_reference_v2_nominal"].append(float(heft_nominal))
        for name, actions in plans.items():
            result, _e, _m = env._schedule(index, actions, mc)
            makespans[name].append(float(result.makespan_seconds))
            channels = evaluate_constraints(_constraint_view(graph, mc), result)
            hard_violations[name].append(
                float(channels["C_GRAPH_HARD_DEADLINE"]["violation"]) > 0.0)
    rows = {}
    for name in names:
        vals = makespans[name]
        hv = hard_violations[name]
        rows[name] = {
            "mean_s": statistics.fmean(vals),
            "median_s": statistics.median(vals),
            "min_s": min(vals),
            "max_s": max(vals),
            "hard_deadline_violation_rate": (sum(hv) / len(hv)) if hv else None,
        }
    return {
        "split": "validation_query",
        "graphs": len(graphs),
        "base_seed": VALIDATION_BASE_SEED,
        "note": ("deterministic reference plans on the SAME graphs and the SAME MC "
                 "realizations as the validation evaluator; compare against the model's "
                 "logged validation/query_mean_latency_seconds_k3 and _k0 (k0 latency = "
                 "-validation/objective_discounted_return_k0)"),
        "references": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    out = run()
    print("%-36s %9s %9s %9s %12s" % ("reference (same 40 val-query graphs)", "mean_s",
                                      "median_s", "min_s", "hard_viol"))
    for name, row in out["references"].items():
        hv = row["hard_deadline_violation_rate"]
        print("%-36s %9.5f %9.5f %9.5f %12s" % (
            name, row["mean_s"], row["median_s"], row["min_s"],
            "n/a" if hv is None else "%.3f" % hv))
    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
        print("written", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
