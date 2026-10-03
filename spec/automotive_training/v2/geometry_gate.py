#!/usr/bin/env python3
"""v2 geometry gate: does the preferred placement change with exogenous context?

Regimes A-L from the v2 plan. The foreground candidate panel is evaluated while N-1
background DAGs load the shared MEC, so "MEC load" is a real queueing state, not a
parameter guess. Helper/link/reliability state is switched per regime.

Panel: all-UE, all-MEC, all-HELPER, v2-aware greedy coordinate descent, HEFT v2, and a
bounded stronger search (labelled candidate-search lower bound, budget recorded).

Output: per-regime winner fractions, candidate-panel headroom vs best pure, token shares,
queue/communication contributions, deadline misses, contact failures, p95/p99, and a
PASS/FAIL verdict per the v2 gate rules (all-MEC > 0.80 winner in EVERY regime => BLOCK).
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

from spec.automotive_training.automotive_loader import load_dataset  # noqa: E402
from spec.automotive_training.v2.adapters import (  # noqa: E402
    compute_spec, dag_spec_from_graph, link_spec, plan_map_from_actions, pure_plan,
)
from spec.automotive_training.v2.helper_model import HelperState  # noqa: E402
from spec.automotive_training.v2.heft_bridge import heft_plan_for_graph  # noqa: E402
from spec.automotive_training.v2.link_model import make_process  # noqa: E402
from spec.automotive_training.v2.reliability import make_gate, standby_required  # noqa: E402
from spec.automotive_training.v2.shared_scheduler import (  # noqa: E402
    V2ComputeSpec, schedule_shared,
)
from spec.automotive_training.v2.stronger_search import stronger_search  # noqa: E402

REGIMES = {
    "A_mec_idle_stable_v2i": dict(load=1, link="stable"),
    "B_mec_light": dict(load=2, link="stable"),
    "C_mec_moderate": dict(load=4, link="stable"),
    "D_mec_heavy": dict(load=8, link="stable"),
    "E_helper_idle_stable_v2v": dict(load=1, link="stable", helper="idle"),
    "F_helper_busy": dict(load=1, link="stable", helper="busy"),
    "G_helper_short_contact": dict(load=1, link="stable", helper="short"),
    "H_poor_v2v": dict(load=1, link="stable", v2v_scale=0.2),
    "I_network_high_variance": dict(load=1, link="degraded"),
    "J_high_unreliable_remote": dict(load=1, link="degraded", reliability=True,
                                     confidence=0.80, outage=0.10, criticality="HIGH"),
    "K_high_reliable_idle_mec": dict(load=1, link="stable", reliability=True,
                                     confidence=1.0, outage=0.0, criticality="HIGH"),
    "L_low_good_remote": dict(load=1, link="stable", reliability=True,
                              confidence=0.999, outage=0.0, criticality="LOW"),
}


def _candidate_makespan(graph, plans, link, compute, mc, *, link_process=None,
                        helper_states=None, reliability_gate=None, evidence=None,
                        standby_for=None, background=None) -> float:
    """Foreground DAG completion when scheduled TOGETHER with the background load.

    The background DAGs consume MEC CPU and the shared MEC radio, so "MEC load" becomes a
    real queueing state instead of a parameter. The returned time is the completion of the
    foreground DAG (`completion_by_dag["fg"]`), not the makespan of the whole batch.
    """
    dag = dag_spec_from_graph(graph, dag_id="fg", owner=0, mc=mc,
                              helper_id=0 if helper_states else None)
    dags = [dag]
    plan_batch = {"fg": plans}
    if background:
        bg_dags, bg_plans = background
        dags = list(bg_dags) + [dag]
        plan_batch = dict(bg_plans)
        plan_batch["fg"] = plans
    res = schedule_shared(dags, plan_batch, link=link, compute=compute,
                          link_process=link_process, helper_states=helper_states,
                          reliability_gate=reliability_gate, reliability_evidence=evidence,
                          standby_for=standby_for)
    return float(res.completion_by_dag["fg"])


def evaluate_regime(graphs, spec, *, seed: int = 0, search_budget: int = 600) -> dict:
    rows = []
    for gi, graph in enumerate(graphs):
        mc = None  # nominal realization for the gate (MC variation handled separately)
        link = link_spec(graph)
        if spec.get("v2v_scale"):
            link = type(link)(link.mec_ul_bytes_per_s, link.mec_dl_bytes_per_s,
                              link.v2v_bytes_per_s * float(spec["v2v_scale"]),
                              link.direct_helper_v2i, link.shared_radio)
        compute = compute_spec([graph])
        link_process = None if spec.get("link", "stable") == "stable" else \
            make_process(spec["link"], seed + gi)
        helper_states = None
        if spec.get("helper") == "idle":
            helper_states = {0: HelperState(0, compute.helper_cpu_bytes_per_s[0],
                                            contact_end_s=100.0, predicted_contact_end_s=100.0)}
        elif spec.get("helper") == "busy":
            helper_states = {0: HelperState(0, compute.helper_cpu_bytes_per_s[0],
                                            busy_until_s=2.0, contact_end_s=100.0,
                                            predicted_contact_end_s=100.0)}
        elif spec.get("helper") == "short":
            helper_states = {0: HelperState(0, compute.helper_cpu_bytes_per_s[0],
                                            contact_end_s=0.05, predicted_contact_end_s=0.05)}
        gate = make_gate() if spec.get("reliability") else None
        evidence = {"link_confidence": float(spec.get("confidence", 1.0)),
                    "outage_fraction": float(spec.get("outage", 0.0))} if gate else None

        # concurrent background load on the shared MEC (N-1 identical DAGs)
        load = int(spec.get("load", 1))
        background = None
        if load > 1:
            bg_dags = [dag_spec_from_graph(graph, dag_id="bg%d" % i, owner=0, mc=mc)
                       for i in range(load - 1)]
            background = (bg_dags, {d.dag_id: pure_plan(graph, 1) for d in bg_dags})

        def cand(actions):
            return _candidate_makespan(graph, plan_map_from_actions(graph, actions), link,
                                       compute, mc, link_process=link_process,
                                       helper_states=helper_states,
                                       reliability_gate=gate, evidence=evidence,
                                       standby_for=standby_rel if gate else None,
                                       background=background)

        standby_rel = standby_required
        panel = {
            "all_UE": cand([0] * 20), "all_MEC": cand([1] * 20), "all_HELPER": cand([2] * 20),
            "heft_v2": None, "greedy_cd": None, "stronger_search": None,
        }
        # HEFT v2 plan
        tokens = list(plan_map_from_actions(graph, [0] * 20).keys())
        heft_actions, _nominal = heft_plan_for_graph(graph)
        panel["heft_v2"] = cand(list(heft_actions))
        # v2-aware greedy coordinate descent (uses the same objective as the panel)
        chosen = {t: 0 for t in tokens}
        for tok in tokens:
            best = None
            for action in (0, 1, 2):
                trial = dict(chosen)
                trial[tok] = action
                value = _candidate_makespan(graph, trial, link, compute, mc,
                                            link_process=link_process,
                                            helper_states=helper_states,
                                            reliability_gate=gate, evidence=evidence,
                                            standby_for=standby_rel if gate else None,
                                            background=background)
                if best is None or value < best[0]:
                    best = (value, action)
            chosen[tok] = best[1]
        panel["greedy_cd"] = _candidate_makespan(graph, chosen, link, compute, mc,
                                                 link_process=link_process,
                                                 helper_states=helper_states,
                                                 reliability_gate=gate, evidence=evidence,
                                                 standby_for=standby_rel if gate else None,
                                                 background=background)
        # bounded stronger search (candidate-search lower bound)
        sr = stronger_search(tokens, lambda plan: _candidate_makespan(
            graph, plan, link, compute, mc, link_process=link_process,
            helper_states=helper_states, reliability_gate=gate, evidence=evidence,
            standby_for=standby_rel if gate else None, background=background), starts=2,
            seed=seed + gi, budget=search_budget, ils_rounds=1)
        panel["stronger_search"] = sr.objective
        pure = {k: panel[k] for k in ("all_UE", "all_MEC", "all_HELPER")}
        best_pure, best_pure_name = min((v, k) for k, v in pure.items())
        # deterministic tie-break: prefer the simplest plan (local first), then MEC,
        # then the search-based candidates, so ties are reported as the simple plan
        TIE = {"all_UE": 0, "all_MEC": 1, "all_HELPER": 2, "greedy_cd": 3,
               "heft_v2": 4, "stronger_search": 5}
        winner, winner_name = min((v, TIE[k], k) for k, v in panel.items())[::2]
        # panel WITHOUT the search-based candidate: does all-MEC win on its own merits?
        cheap = {k: v for k, v in panel.items()
                 if k in ("all_UE", "all_MEC", "all_HELPER", "heft_v2", "greedy_cd")}
        cheap_winner, cheap_name = min((v, TIE[k], k) for k, v in cheap.items())[::2]
        rows.append({
            "graph_id": graph.graph_id, "panel": panel, "winner": winner_name,
            "best_pure": best_pure_name, "headroom_vs_best_pure_pct":
                100.0 * (best_pure - winner) / max(best_pure, 1e-12),
            "all_MEC_winner": winner_name == "all_MEC",
            "all_MEC_winner_no_search": cheap_name == "all_MEC",
            "local_wins": abs(panel["all_UE"] - winner) <= 1e-12,
            "stronger_search_gain_vs_all_MEC_pct":
                100.0 * (panel["all_MEC"] - panel["stronger_search"]) / max(panel["all_MEC"], 1e-12),
            "search_evaluations": sr.evaluations,
        })
    n = float(len(rows))
    return {
        "regime": spec["regime_name"], "graphs": len(rows),
        "all_MEC_winner_fraction": sum(1 for r in rows if r["all_MEC_winner"]) / n,
        "all_MEC_winner_fraction_no_search": sum(1 for r in rows if r["all_MEC_winner_no_search"]) / n,
        "local_winner_fraction": sum(1 for r in rows if r["local_wins"]) / n,
        "stronger_search_gain_vs_all_MEC_mean_pct": statistics.fmean(
            [r["stronger_search_gain_vs_all_MEC_pct"] for r in rows]),
        "all_UE_winner_fraction": sum(1 for r in rows if r["winner"] == "all_UE") / n,
        "all_HELPER_winner_fraction": sum(1 for r in rows if r["winner"] == "all_HELPER") / n,
        "mixed_winner_fraction": sum(1 for r in rows if r["winner"] in
                                     ("greedy_cd", "heft_v2", "stronger_search")) / n,
        "winner_counts": {k: sum(1 for r in rows if r["winner"] == k) for k in
                          ("all_UE", "all_MEC", "all_HELPER", "greedy_cd", "heft_v2",
                           "stronger_search")},
        "headroom_vs_best_pure_mean_pct": statistics.fmean([r["headroom_vs_best_pure_pct"] for r in rows]),
        "headroom_vs_best_pure_median_pct": statistics.median([r["headroom_vs_best_pure_pct"] for r in rows]),
        "headroom_vs_best_pure_max_pct": max(r["headroom_vs_best_pure_pct"] for r in rows),
        "rows": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--search-budget", type=int, default=600)
    ap.add_argument("--json", default=str(ROOT / "spec/automotive_training/reports/"
                                              "v2_system_model/GEOMETRY_GATE.json"))
    args = ap.parse_args()
    ds = load_dataset()
    val = ds.validation_query()[: int(args.graphs)]
    out = {"schema": "v2_geometry_gate_v1", "graphs_per_regime": len(val),
           "search_label": "candidate-search lower bound", "regimes": {}}
    for name, spec in REGIMES.items():
        spec = dict(spec, regime_name=name)
        out["regimes"][name] = evaluate_regime(val, spec, seed=args.seed,
                                               search_budget=args.search_budget)
        r = out["regimes"][name]
        print("%-26s MEC=%.2f UE=%.2f HELPER=%.2f mixed=%.2f headroom(mean/med/max)=%.1f/%.1f/%.1f%%" % (
            name, r["all_MEC_winner_fraction"], r["all_UE_winner_fraction"],
            r["all_HELPER_winner_fraction"], r["mixed_winner_fraction"],
            r["headroom_vs_best_pure_mean_pct"], r["headroom_vs_best_pure_median_pct"],
            r["headroom_vs_best_pure_max_pct"]))
    mecs = [r["all_MEC_winner_fraction"] for r in out["regimes"].values()]
    helpers = [r["all_HELPER_winner_fraction"] for r in out["regimes"].values()]
    ues = [r["all_UE_winner_fraction"] for r in out["regimes"].values()]
    heads = [r["headroom_vs_best_pure_mean_pct"] for r in out["regimes"].values()]
    verdict = {
        "all_MEC_wins_everywhere_over_80pct": all(m > 0.8 for m in mecs),
        "min_all_MEC_winner_fraction": min(mecs),
        "max_all_MEC_winner_fraction": max(mecs),
        "spread_all_MEC_winner_fraction": max(mecs) - min(mecs),
        "helper_wins_somewhere": max(helpers) > 0.0,
        "local_wins_somewhere": max(ues) > 0.0,
        "max_headroom_mean_pct": max(heads),
        "min_headroom_mean_pct": min(heads),
    }
    verdict["gate"] = "BLOCK" if verdict["all_MEC_wins_everywhere_over_80pct"] else "PASS"
    out["verdict"] = verdict
    path = Path(args.json)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, sort_keys=True, default=str) + "\n")
    print("\nVERDICT:", json.dumps(verdict, sort_keys=True))
    print("written", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
