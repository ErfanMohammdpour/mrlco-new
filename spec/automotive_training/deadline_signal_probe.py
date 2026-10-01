#!/usr/bin/env python3
"""Gate 19A: are the deadline / mixed-criticality training signals non-degenerate?

The frozen dataset is feasibility-saturated at the GRAPH level, so a deadline-aware
training claim needs evidence that the channels are not identically zero. This probe
measures, on the meta-train split and under DELIBERATELY BAD policies:

    graph hard-deadline violation rate   (makespan > D_G)
    HIGH task tardiness rate             (availability > frozen d_i)
    MEDIUM task tardiness rate
    task firm-miss rate
    mode-switch rate                     (LO -> HI under execution_uncertainty_v1)
    HC task tardiness / graph violation magnitudes

Policies: all_UE, all_MEC, all_HELPER, seeded random plans, and a greedy plan.
Nothing is tightened and no deadline is edited.

CLI: python3 -m spec.automotive_training.deadline_signal_probe [--graphs N] [--json PATH]
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

from spec.automotive_training.automotive_env import AutomotiveEnv  # noqa: E402
from spec.automotive_training.automotive_primary import AutomotiveResourceCluster  # noqa: E402
from spec.automotive_training.automotive_loader import load_dataset  # noqa: E402
from spec.automotive_training.automotive_constraints import CONSTRAINT_NAMES  # noqa: E402
from spec.automotive_training.automotive_constraints import evaluate_constraints  # noqa: E402

GREEDY_GRAPHS_DEFAULT = 40


def _metrics(env: AutomotiveEnv, index: int, actions, mc):
    result, _energy, _macro = env._schedule(index, actions, mc)
    graph = env.graph_objects[index]
    from spec.automotive_training.automotive_env import _constraint_view

    channels = evaluate_constraints(_constraint_view(graph, mc), result)
    makespan = float(result.makespan_seconds)
    return {
        "graph_violation": float(channels["C_GRAPH_HARD_DEADLINE"]["violation"]),
        "high_tardiness": float(channels["C_HI_TASK_TARDINESS"]["violation"]),
        "medium_tardiness": float(channels["C_MED_TASK_TARDINESS"]["violation"]),
        "high_misses": int(channels["C_HI_TASK_TARDINESS"]["n_violating_tasks"]),
        "medium_misses": int(channels["C_MED_TASK_TARDINESS"]["n_violating_tasks"]),
        "makespan": makespan,
        "mode": "NA" if mc is None else str(mc["final_mode"]),
        "switches": 0 if mc is None else len(mc["switches"]),
        "high_preserved": True if mc is None else bool(mc["high_preserved"]),
    }


def greedy_plan(env: AutomotiveEnv, index: int):
    order = env.orders[index]
    chosen: list[int] = []
    for k in range(len(order)):
        best = None
        for action in (0, 1, 2):
            plan = chosen + [action] + [0] * (len(order) - k - 1)
            result, _e, _m = env._schedule(index, plan, None)
            value = float(result.makespan_seconds)
            if best is None or value < best[0] - 1e-12 or (
                    abs(value - best[0]) <= 1e-12 and action < best[1]):
                best = (value, action)
        chosen.append(int(best[1]))
    return chosen


def summarise(rows: list[dict], label: str, mc_enabled: bool) -> dict:
    n = max(1, len(rows))
    return {
        "policy": label,
        "mode_realization": "MC on (execution_uncertainty_v1)" if mc_enabled else "MC off (nominal instance)",
        "graphs": len(rows),
        "graph_hard_violation_rate": sum(1 for r in rows if r["graph_violation"] > 0) / n,
        "high_task_tardiness_rate": sum(1 for r in rows if r["high_tardiness"] > 0) / n,
        "medium_task_tardiness_rate": sum(1 for r in rows if r["medium_tardiness"] > 0) / n,
        "firm_task_miss_rate": (sum(r["high_misses"] + r["medium_misses"] for r in rows)
                                / (20.0 * n)),
        "mode_switch_rate": sum(1 for r in rows if r["switches"] > 0) / n,
        "hi_mode_rate": sum(1 for r in rows if r["mode"] == "HI") / n,
        "mean_graph_violation_s": statistics.fmean(r["graph_violation"] for r in rows),
        "mean_high_tardiness_s": statistics.fmean(r["high_tardiness"] for r in rows),
        "mean_medium_tardiness_s": statistics.fmean(r["medium_tardiness"] for r in rows),
        "mean_makespan_s": statistics.fmean(r["makespan"] for r in rows),
        "high_preservation_violations": sum(1 for r in rows if not r["high_preserved"]),
    }


def run(graphs: int | None = None, greedy_graphs: int = GREEDY_GRAPHS_DEFAULT) -> dict:
    dataset = load_dataset()
    train = dataset.meta_train()
    if graphs:
        train = train[: int(graphs)]
    env = AutomotiveEnv(train, AutomotiveResourceCluster(), slots_per_task=1, base_seed=0)
    rng = np.random.RandomState(1234)
    out = {"dataset": "MARGO-AUTOMOTIVE-MC-v1", "split": "meta_train",
           "graphs_probed": len(train), "results": []}

    for mc_enabled in (False, True):
        env.reset_count = 0
        env.set_task({"dist_index": 0, "graph_indices": np.zeros(1, dtype=np.int32)})
        env.reset()
        for label in ("all_UE", "all_MEC", "all_HELPER", "random"):
            rows = []
            for index in range(len(train)):
                env.task_id = index
                env.graph_indices = np.zeros(1, dtype=np.int32)
                env.reset_count = 1
                env._draw_slots()
                mc = env._slot_mc[0] if mc_enabled else None
                if label == "random":
                    plans = [list(rng.randint(0, 3, size=20)) for _ in range(3)]
                else:
                    action = {"all_UE": 0, "all_MEC": 1, "all_HELPER": 2}[label]
                    plans = [[action] * 20]
                for plan in plans:
                    rows.append(_metrics(env, index, plan, mc))
            out["results"].append(summarise(rows, label, mc_enabled))
        # greedy on a smaller subset (60 schedule evaluations per graph)
        rows = []
        for index in range(min(greedy_graphs, len(train))):
            env.task_id = index
            env.graph_indices = np.zeros(1, dtype=np.int32)
            env.reset_count = 1
            env._draw_slots()
            mc = env._slot_mc[0] if mc_enabled else None
            rows.append(_metrics(env, index, greedy_plan(env, index), mc))
        out["results"].append(summarise(rows, "greedy", mc_enabled))
    out["status"] = "PASS"
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs", type=int, default=None)
    ap.add_argument("--json", default=str(Path(__file__).resolve().parent / "reports"
                                          / "deadline_signal_probe.json"))
    args = ap.parse_args()
    result = run(args.graphs)
    path = Path(args.json)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    for row in result["results"]:
        print("%-10s %-46s hard=%.3f high=%.3f med=%.3f firm=%.3f switch=%.3f" % (
            row["policy"], row["mode_realization"], row["graph_hard_violation_rate"],
            row["high_task_tardiness_rate"], row["medium_task_tardiness_rate"],
            row["firm_task_miss_rate"], row["mode_switch_rate"]))
    print("written", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
