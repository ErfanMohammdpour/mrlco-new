#!/usr/bin/env python3
"""Gate I re-analysis: candidate-oracle headroom and where mixed placement wins.

The first Gate I report compared the best mixed heuristic against the best pure plan and
concluded "no average advantage". That framing penalises the heuristics for the graphs
where they are bad, because all-MEC was not in the mixed candidate set. The honest
question for the benchmark is: how much headroom does the FULL candidate panel have
relative to all-MEC, i.e.

    oracle_panel = min(all_MEC, best mixed heuristic)

and separately: when a mixed plan wins, is the win V2V/HELPER driven or UE+MEC mixing?

CLI: python3 spec/automotive_training/headroom_reanalysis.py [--json PATH]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SRC = ROOT / "spec/automotive_training/reports/gateI/mc_headroom_audit.json"


def analyse(block: dict) -> dict:
    rows = block["rows"]
    pure = {"all_UE", "all_MEC", "all_HELPER"}
    mixed = {"greedy_cd", "heft_v2"}
    per_graph = []
    for r in rows:
        winners = {"all_UE": r.get("all_UE_s"), "all_MEC": r.get("all_MEC_s"),
                   "all_HELPER": r.get("all_HELPER_s"), "greedy_cd": r.get("greedy_s"),
                   "heft_v2": r.get("heft_v2_s")}
        winners = {k: v for k, v in winners.items() if v is not None}
        best_pure_name = min((v, k) for k, v in winners.items() if k in pure)
        best_mixed_name = min((v, k) for k, v in winners.items() if k in mixed)
        oracle, oracle_name = min((v, k) for k, v in winners.items())
        per_graph.append({
            "graph_id": r["graph_id"],
            "all_MEC_s": r["all_MEC_s"], "best_pure_s": best_pure_name[0],
            "best_pure": best_pure_name[1],
            "best_mixed_s": best_mixed_name[0], "best_mixed": best_mixed_name[1],
            "oracle_s": oracle, "oracle": oracle_name,
            "oracle_improvement_vs_all_MEC_pct": 100.0 * (r["all_MEC_s"] - oracle) / max(r["all_MEC_s"], 1e-12),
            "mixed_beats_all_MEC": best_mixed_name[0] < r["all_MEC_s"] - 1e-12,
            "mixed_helper_fraction": r["best_plan_fraction_helper"],
            "uses_helper": r["uses_helper"],
        })
    n = float(len(per_graph))
    mixed_wins = [g for g in per_graph if g["mixed_beats_all_MEC"]]
    return {
        "graphs": len(per_graph),
        "all_MEC_mean_s": statistics.fmean([g["all_MEC_s"] for g in per_graph]),
        "best_pure_mean_s": statistics.fmean([g["best_pure_s"] for g in per_graph]),
        "best_mixed_mean_s": statistics.fmean([g["best_mixed_s"] for g in per_graph]),
        "oracle_panel_mean_s": statistics.fmean([g["oracle_s"] for g in per_graph]),
        "oracle_panel_headroom_ms_vs_all_MEC": 1000.0 * (
            statistics.fmean([g["all_MEC_s"] for g in per_graph])
            - statistics.fmean([g["oracle_s"] for g in per_graph])),
        "oracle_panel_headroom_pct_vs_all_MEC": statistics.fmean(
            [g["oracle_improvement_vs_all_MEC_pct"] for g in per_graph]),
        "share_all_MEC_is_oracle": sum(1 for g in per_graph if g["oracle"] == "all_MEC") / n,
        "share_all_MEC_is_best_pure": sum(1 for g in per_graph if g["best_pure"] == "all_MEC") / n,
        "share_mixed_beats_all_MEC": len(mixed_wins) / n,
        "share_of_mixed_wins_using_helper": (
            sum(1 for g in mixed_wins if g["uses_helper"]) / len(mixed_wins)) if mixed_wins else 0.0,
        "mean_helper_fraction_in_mixed_wins": (
            statistics.fmean([g["mixed_helper_fraction"] for g in mixed_wins])
            if mixed_wins else 0.0),
        "median_oracle_improvement_pct": statistics.median(
            [g["oracle_improvement_vs_all_MEC_pct"] for g in per_graph]),
        "max_oracle_improvement_pct": max(g["oracle_improvement_vs_all_MEC_pct"] for g in per_graph),
        "per_graph": per_graph,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=str(ROOT / "spec/automotive_training/reports/gateI/"
                                              "headroom_reanalysis.json"))
    args = ap.parse_args()
    src = json.loads(SRC.read_text())
    out = {
        "schema": "automotive_mc_headroom_reanalysis_v1",
        "source": str(SRC.relative_to(ROOT)),
        "note": ("oracle_panel = min(all_MEC, greedy, HEFT) on the SAME graph+realization; "
                 "all-MEC is included as a candidate, so the oracle measures the real "
                 "headroom of selective mixed placement rather than the quality of any "
                 "single unconditional heuristic"),
    }
    for key in ("meta_train", "validation_query"):
        out[key] = analyse(src[key])
        b = out[key]
        print("== %s (n=%d) ==" % (key, b["graphs"]))
        for k in ("all_MEC_mean_s", "best_pure_mean_s", "best_mixed_mean_s", "oracle_panel_mean_s",
                  "oracle_panel_headroom_ms_vs_all_MEC", "oracle_panel_headroom_pct_vs_all_MEC",
                  "share_all_MEC_is_oracle", "share_all_MEC_is_best_pure",
                  "share_mixed_beats_all_MEC", "share_of_mixed_wins_using_helper",
                  "mean_helper_fraction_in_mixed_wins", "median_oracle_improvement_pct",
                  "max_oracle_improvement_pct"):
            print("   %-42s %s" % (k, b[k] if isinstance(b[k], int) else round(b[k], 5)))
    path = Path(args.json)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, sort_keys=True, default=str) + "\n")
    print("written", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
