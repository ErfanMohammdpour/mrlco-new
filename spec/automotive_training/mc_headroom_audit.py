#!/usr/bin/env python3
"""Gate I: does mixed placement still pay off under the MC runtime?

Motivation: on the validation split, all-MEC is a very strong reference (0.0262 s) and
the trained policy collapses to it. Before claiming V2V/HELPER-aware optimisation we must
measure, under the FROZEN mixed-criticality runtime, how often a mixed placement actually
beats the best pure plan (all-UE / all-MEC / all-HELPER), and how much.

Every method is evaluated on the SAME graph and the SAME MC realization.

CLI: python3 spec/automotive_training/mc_headroom_audit.py [--meta-train N] [--validation N]
                                                            [--json PATH]
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

PURE = {"all_UE": 0, "all_MEC": 1, "all_HELPER": 2}


def greedy_coordinate_descent(env, index, mc, n=20):
    chosen = []
    for k in range(n):
        best = None
        for action in (0, 1, 2):
            plan = chosen + [action] + [0] * (n - k - 1)
            result, _e, _m = env._schedule(index, plan, mc)
            value = float(result.makespan_seconds)
            if best is None or value < best[0] - 1e-12 or (
                    abs(value - best[0]) <= 1e-12 and action < best[1]):
                best = (value, action)
        chosen.append(int(best[1]))
    return chosen


def audit(graphs, base_seed: int, label: str) -> dict:
    env = AutomotiveEnv(graphs, AutomotiveResourceCluster(), role="validation",
                        single_dist=True, slots_per_task=len(graphs), base_seed=base_seed)
    env.set_task({"dist_index": 0, "graph_indices": np.arange(len(graphs), dtype=np.int32)})
    env.reset()
    rows = []
    for index, graph in enumerate(graphs):
        mc = env._slot_mc[index]
        makespans, hard = {}, {}
        for name, action in PURE.items():
            result, _e, _m = env._schedule(index, [action] * 20, mc)
            makespans[name] = float(result.makespan_seconds)
            channels = evaluate_constraints(_constraint_view(graph, mc), result)
            hard[name] = bool(float(channels["C_GRAPH_HARD_DEADLINE"]["violation"]) > 0.0)
        greedy = greedy_coordinate_descent(env, index, mc)
        gr, _e, _m = env._schedule(index, greedy, mc)
        makespans["greedy_cd"] = float(gr.makespan_seconds)
        h_actions, h_nominal, _r = heft_reference_v2_plan(graph.as_record(), co_physical=True)
        hr, _e, _m = env._schedule(index, list(h_actions), mc)
        makespans["heft_v2"] = float(hr.makespan_seconds)
        best_pure = min((makespans[n], n) for n in PURE)
        best_mixed = min((makespans["greedy_cd"], "greedy_cd"),
                         (makespans["heft_v2"], "heft_v2"))
        best_plan = greedy if best_mixed[1] == "greedy_cd" else list(h_actions)
        mix = {a: best_plan.count(a) / float(len(best_plan)) for a in (0, 1, 2)}
        rows.append({
            "graph_id": graph.graph_id, "split": label,
            "best_pure": best_pure[1], "best_pure_s": best_pure[0],
            "best_mixed": best_mixed[1], "best_mixed_s": best_mixed[0],
            "improvement_pct": 100.0 * (best_pure[0] - best_mixed[0]) / max(best_pure[0], 1e-12),
            "all_MEC_s": makespans["all_MEC"], "greedy_s": makespans["greedy_cd"],
            "heft_v2_s": makespans["heft_v2"],
            "best_plan_fraction_ue": mix[0], "best_plan_fraction_mec": mix[1],
            "best_plan_fraction_helper": mix[2],
            "uses_helper": mix[2] > 0.0,
            "hard_violation_best_pure": hard[best_pure[1]],
            "all_MEC_is_best_pure": best_pure[1] == "all_MEC",
        })
    n = float(len(rows))
    return {
        "split": label, "graphs": len(rows), "base_seed": base_seed,
        "share_mixed_strictly_better": sum(1 for r in rows if r["improvement_pct"] > 1e-9) / n,
        "mean_improvement_pct": statistics.fmean([r["improvement_pct"] for r in rows]),
        "median_improvement_pct": statistics.median([r["improvement_pct"] for r in rows]),
        "max_improvement_pct": max(r["improvement_pct"] for r in rows),
        "share_all_MEC_best_pure": sum(1 for r in rows if r["all_MEC_is_best_pure"]) / n,
        "share_best_plan_uses_helper": sum(1 for r in rows if r["uses_helper"]) / n,
        "mean_helper_fraction_in_best_plan": statistics.fmean([r["best_plan_fraction_helper"] for r in rows]),
        "best_plan_winners": {w: sum(1 for r in rows if r["best_mixed"] == w)
                              for w in ("greedy_cd", "heft_v2")},
        "mean_all_MEC_s": statistics.fmean([r["all_MEC_s"] for r in rows]),
        "mean_best_pure_s": statistics.fmean([r["best_pure_s"] for r in rows]),
        "mean_best_mixed_s": statistics.fmean([r["best_mixed_s"] for r in rows]),
        "rows": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta-train", type=int, default=40)
    ap.add_argument("--validation", type=int, default=40)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    ds = load_dataset()
    meta = ds.meta_train()
    val = ds.validation_query()
    step_m = max(1, len(meta) // int(args.meta_train))
    step_v = max(1, len(val) // int(args.validation))
    out = {
        "schema": "automotive_mc_headroom_audit_v1",
        "note": ("pure vs mixed placement under the frozen MC runtime; every method is "
                 "evaluated on the same graph and the same MC realization"),
        "meta_train": audit(meta[::step_m][: int(args.meta_train)], 101, "meta_train"),
        "validation_query": audit(val[::step_v][: int(args.validation)], 303, "validation_query"),
    }
    for split in ("meta_train", "validation_query"):
        block = out[split]
        print("== %s (n=%d) ==" % (split, block["graphs"]))
        for key in ("share_mixed_strictly_better", "mean_improvement_pct", "median_improvement_pct",
                    "max_improvement_pct", "share_all_MEC_best_pure", "share_best_plan_uses_helper",
                    "mean_helper_fraction_in_best_plan", "mean_all_MEC_s", "mean_best_pure_s",
                    "mean_best_mixed_s"):
            print("   %-34s %.5f" % (key, block[key]))
        print("   best_plan_winners               %s" % (block["best_plan_winners"],))
    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=2, sort_keys=True, default=str) + "\n")
        print("written", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
