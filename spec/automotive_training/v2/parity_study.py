#!/usr/bin/env python3
"""v1 <-> v2 single-DAG parity study.

Purpose: the geometry gate's candidate-panel headroom under v2 is ~6% on the same graphs
where v1's panel headroom is ~2%. Before attributing that to "richer geometry", we must
know how far the v2 scheduler departs from the frozen v1 engine on identical inputs
(one DAG, one plan, same MC realization, no concurrency, no link process, no helper state,
no reliability gate).

Reports the relative makespan difference per plan and the set of structural differences
(transfers, cut edges, MEC busy time). Any systematic bias is a MODEL DIFFERENCE that must
be documented, not silently counted as headroom.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spec.automotive_training.automotive_env import AutomotiveEnv  # noqa: E402
from spec.automotive_training.automotive_loader import load_dataset  # noqa: E402
from spec.automotive_training.automotive_primary import AutomotiveResourceCluster  # noqa: E402
from spec.automotive_training.v2.compat import fmean  # noqa: E402
from spec.automotive_training.v2.adapters import (  # noqa: E402
    compute_spec, dag_spec_from_graph, link_spec, plan_map_from_actions, pure_plan,
)
from spec.automotive_training.v2.helper_model import HelperState  # noqa: E402
from spec.automotive_training.v2.shared_scheduler import schedule_shared  # noqa: E402
from spec.automotive_training.v2.world import V2WorldConfig, build_world  # noqa: E402

PLANS = {
    "all_UE": [0] * 20,
    "all_MEC": [1] * 20,
    "all_HELPER": [2] * 20,
    "alternate_MEC_HELPER": [1, 2] * 10,
    "alternate_MEC_UE": [1, 0] * 10,
    "front_MEC_back_UE": [1] * 10 + [0] * 10,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs", type=int, default=12)
    ap.add_argument("--json", default=str(ROOT / "spec/automotive_training/reports/"
                                              "v2_system_model/V1_V2_PARITY.json"))
    args = ap.parse_args()
    ds = load_dataset()
    graphs = ds.validation_query()[: int(args.graphs)]
    env = AutomotiveEnv(graphs, AutomotiveResourceCluster(), single_dist=True,
                        slots_per_task=len(graphs), base_seed=303)
    env.set_task({"dist_index": 0, "graph_indices": np.arange(len(graphs), dtype=np.int32)})
    env.reset()
    rows = []
    for gi, graph in enumerate(graphs):
        mc = env._slot_mc[gi]
        # DEGENERATE configuration: no background, no link process, no reliability gate and
        # an always-available helper, i.e. the assumptions under which parity with the frozen
        # v1 engine is a meaningful requirement. Built by the SAME canonical builder.
        world = build_world(graph, slot_id=gi, world_id="parity_g%d" % gi, mc=mc,
                            config=V2WorldConfig(background_dags=0, helper_id=0),
                            helper_seed=0)
        world.helpers = {0: HelperState(0, world.compute.helper_cpu_bytes_per_s[0],
                                        contact_end_s=float("inf"),
                                        predicted_contact_end_s=float("inf"))}
        for plan_name, actions in PLANS.items():
            v1, _e, _m = env._schedule(gi, actions, mc)
            v1_ms = float(v1.makespan_seconds)
            res = world.with_foreground_actions(graph, actions).schedule()
            v2_ms = float(res.makespan_s)
            v1_transfers = len(getattr(v1, "transfers", []) or [])
            rows.append({
                "graph_id": graph.graph_id, "plan": plan_name,
                "v1_makespan_s": v1_ms, "v2_makespan_s": v2_ms,
                "rel_diff_pct": 100.0 * (v2_ms - v1_ms) / max(v1_ms, 1e-12),
                "v1_transfers": v1_transfers, "v2_transfers": res.mechanics["radio_events"],
            })
    diffs = [r["rel_diff_pct"] for r in rows]
    per_plan = {}
    for plan_name in PLANS:
        vals = [r["rel_diff_pct"] for r in rows if r["plan"] == plan_name]
        per_plan[plan_name] = {
            "mean_rel_diff_pct": fmean(vals),
            "median_rel_diff_pct": statistics.median(vals),
            "max_abs_rel_diff_pct": max(abs(v) for v in vals),
            "share_v2_faster": sum(1 for v in vals if v < -1e-9) / len(vals),
        }
    transfer_mismatch = sum(1 for r in rows if r["v1_transfers"] != r["v2_transfers"])
    out = {
        "schema": "v1_v2_parity_study_v1",
        "graphs": len(graphs), "plans": list(PLANS), "rows": len(rows),
        "overall": {"mean_rel_diff_pct": fmean(diffs),
                    "median_rel_diff_pct": statistics.median(diffs),
                    "max_abs_rel_diff_pct": max(abs(v) for v in diffs),
                    "share_v2_faster": sum(1 for v in diffs if v < -1e-9) / len(diffs)},
        "per_plan": per_plan,
        "transfer_count_mismatch_rows": transfer_mismatch,
        "interpretation": ("A systematic v2-faster bias means the v2 model is not "
                           "numerically parity-equivalent to the frozen v1 engine; the "
                           "geometry-gate headroom must then be reported as a MODEL "
                           "difference, not as pure geometry."),
    }
    path = Path(args.json)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({"overall": out["overall"], "transfer_count_mismatch_rows": transfer_mismatch},
                     indent=1, sort_keys=True))
    for name, block in per_plan.items():
        print("%-22s mean=%+6.2f%% median=%+6.2f%% maxabs=%5.2f%% v2_faster=%.2f" % (
            name, block["mean_rel_diff_pct"], block["median_rel_diff_pct"],
            block["max_abs_rel_diff_pct"], block["share_v2_faster"]))
    print("written", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
