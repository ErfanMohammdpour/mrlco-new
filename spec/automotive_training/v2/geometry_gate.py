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
from spec.automotive_training.v2.compat import fmean  # noqa: E402
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
from spec.automotive_training.v2.world import V2WorldConfig, build_world  # noqa: E402

REGIMES = {
    # regimes A-D, H, I, K, L: a helper EXISTS (idle, effectively unlimited contact)
    # unless the regime explicitly makes it busy/short/unusable
    "A_mec_idle_stable_v2i": dict(load=1, link="stable", helper="stable"),
    "B_mec_light": dict(load=2, link="stable", helper="stable"),
    "C_mec_moderate": dict(load=4, link="stable", helper="stable"),
    "D_mec_heavy": dict(load=8, link="stable", helper="stable"),
    "E_helper_idle_stable_v2v": dict(load=1, link="stable", helper="idle"),
    "F_helper_busy": dict(load=1, link="stable", helper="busy"),
    "G_helper_short_contact": dict(load=1, link="stable", helper="short"),
    "H_poor_v2v": dict(load=1, link="stable", helper="stable", v2v_scale=0.2),
    "I_network_high_variance": dict(load=1, link="degraded", helper="stable"),
    "J_high_unreliable_remote": dict(load=1, link="degraded", reliability=True,
                                     confidence=0.80, outage=0.10, criticality="HIGH"),
    "K_high_reliable_idle_mec": dict(load=1, link="stable", helper="stable",
                                     reliability=True, confidence=1.0, outage=0.0,
                                     criticality="HIGH"),
    "L_low_good_remote": dict(load=1, link="stable", helper="stable", reliability=True,
                              confidence=0.999, outage=0.0, criticality="LOW"),
}


def _evaluate_plan(world, plan_map, *, link_process=None, reliability_gate=None,
                   evidence=None, standby_for=None):
    """Schedule a candidate plan in the SAME world and return the annotated result."""
    result = world.with_foreground_plan_map(plan_map).schedule(
        link_process=link_process, reliability_gate=reliability_gate,
        reliability_evidence=evidence, standby_for=standby_for)
    world.annotate(result)
    return result


def _candidate_plan(world, plan_map, *, link_process=None, reliability_gate=None,
                    evidence=None, standby_for=None) -> float:
    """Foreground episode latency for a candidate plan given as {task_id: action}."""
    return float(_evaluate_plan(world, plan_map, link_process=link_process,
                                reliability_gate=reliability_gate, evidence=evidence,
                                standby_for=standby_for).episode_latency_s)


def _winner_evidence(world, chosen, *, link_process=None, reliability_gate=None,
                     evidence=None, standby_for=None) -> dict:
    """Executed-action evidence for the winning plan plus a NO-HELPER ablation.

    Winners are determined from EXECUTED locations (admission may rewrite a token to UE), and
    the ablation re-evaluates the same plan with every HELPER action forced to UE, so helper
    usefulness is measured instead of inferred from an algorithm name.
    """
    from spec.automotive_training.v2.energy import schedule_energy

    def run(plan_map):
        return _evaluate_plan(world, plan_map, link_process=link_process,
                              reliability_gate=reliability_gate, evidence=evidence,
                              standby_for=standby_for)

    result = run(chosen)
    fg = [tm for tm in result.timings.values() if tm.dag_id == world.foreground_id]
    mix = {"UE": 0, "MEC": 0, "HELPER": 0}
    for tm in fg:
        mix[tm.location] = mix.get(tm.location, 0) + 1
    ledger = schedule_energy(result, background_dag_ids=world.background_ids)
    no_helper_plan = {int(k): (0 if int(v) == 2 else int(v)) for k, v in dict(chosen).items()}
    no_helper = run(no_helper_plan)
    no_helper_ledger = schedule_energy(no_helper, background_dag_ids=world.background_ids)
    return {
        "winner_locations": mix,
        "winner_helper_task_fraction": (float(mix["HELPER"]) / float(len(fg))) if fg else 0.0,
        "winner_system_joules": float(ledger.system_joules),
        "winner_background_joules": float(ledger.background_joules),
        "winner_contact_failures": int(result.queue_stats["helper_contact_failures"]),
        "winner_reliability_rejections": int(result.queue_stats["reliability_rejections"]),
        "winner_wasted_work_bytes": float(result.queue_stats["wasted_work_bytes_total"]),
        "winner_energy_ratio_vs_no_helper":
            float(ledger.system_joules) / max(1e-12, float(no_helper_ledger.system_joules)),
        "no_helper_makespan_s": float(no_helper.episode_latency_s),
        "helper_ablation_gain_pct": 100.0 * (float(no_helper.episode_latency_s)
                                             - float(result.episode_latency_s)) \
            / max(1e-12, float(no_helper.episode_latency_s)),
        "helper_used_in_winning_plan": bool(mix["HELPER"]),
    }


def _candidate_makespan(world, graph, actions, *, link_process=None, reliability_gate=None,
                        evidence=None, standby_for=None) -> float:
    """Foreground DAG completion for a candidate plan inside its CANONICAL world.

    The world — background DAGs of OTHER owners, helper states, arrivals, link and compute
    specs — is built ONCE by `v2.world.build_world` and only the foreground PLAN changes per
    candidate. The gate therefore evaluates exactly the object the environment trains on
    (audited defect 3.1: the gate used to construct a different scientific problem), and no
    candidate can regenerate the background or draw a different exogenous stream.
    """
    result = world.with_foreground_actions(graph, actions).schedule(
        link_process=link_process, reliability_gate=reliability_gate,
        reliability_evidence=evidence, standby_for=standby_for)
    world.annotate(result)
    return float(result.episode_latency_s)


def _helper_regime(name, rate):
    """The declared helper contact regime for the gate (an explicit test condition)."""
    inf = float("inf")
    if name == "stable":
        return {0: HelperState(0, rate, contact_end_s=inf, predicted_contact_end_s=inf)}
    if name == "idle":
        return {0: HelperState(0, rate, contact_end_s=100.0, predicted_contact_end_s=100.0)}
    if name == "busy":
        return {0: HelperState(0, rate, busy_until_s=2.0, contact_end_s=100.0,
                               predicted_contact_end_s=100.0)}
    if name == "short":
        return {0: HelperState(0, rate, contact_end_s=0.05, predicted_contact_end_s=0.05)}
    raise ValueError("unknown helper regime %r" % (name,))


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
        helper_name = spec.get("helper")
        helper_states = (_helper_regime(helper_name, compute.helper_cpu_bytes_per_s[0])
                         if helper_name else None)
        gate = make_gate() if spec.get("reliability") else None
        evidence = {"link_confidence": float(spec.get("confidence", 1.0)),
                    "outage_fraction": float(spec.get("outage", 0.0))} if gate else None

        # concurrent background load: N-1 DAGs of OTHER owners on the SAME shared MEC,
        # built by the canonical world builder (workload-derived, not carbon copies)
        load = int(spec.get("load", 1))
        world = build_world(
            graph, slot_id=gi, world_id="gate_g%d" % gi, mc=mc, config=V2WorldConfig(
                background_dags=max(0, load - 1),
                background_owner_offset=1,
                background_policy="all_mec",
                helper_id=0,
                enable_helpers=helper_states is not None),
            helper_seed=seed + gi, link=link, compute=compute)
        if helper_states is not None:
            # the gate's declared contact regime overrides the default helper draw
            world.helpers = dict(helper_states)
            world.provenance["helper_regime"] = str(helper_name)

        def cand(actions):
            return _candidate_makespan(world, graph, actions, link_process=link_process,
                                       reliability_gate=gate, evidence=evidence,
                                       standby_for=standby_rel if gate else None)

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
                value = _candidate_plan(world, trial, link_process=link_process,
                                        reliability_gate=gate, evidence=evidence,
                                        standby_for=standby_rel if gate else None)
                if best is None or value < best[0]:
                    best = (value, action)
            chosen[tok] = best[1]
        panel["greedy_cd"] = _candidate_plan(world, chosen, link_process=link_process,
                                             reliability_gate=gate, evidence=evidence,
                                             standby_for=standby_rel if gate else None)
        # bounded stronger search (candidate-search lower bound)
        sr = stronger_search(tokens, lambda plan: _candidate_plan(
            world, plan, link_process=link_process, reliability_gate=gate,
            evidence=evidence, standby_for=standby_rel if gate else None), starts=2,
            seed=seed + gi, budget=search_budget, ils_rounds=1)
        panel["stronger_search"] = sr.objective
        pure = {k: panel[k] for k in ("all_UE", "all_MEC", "all_HELPER")}
        best_pure, best_pure_name = min((v, k) for k, v in pure.items())
        # deterministic tie-break: prefer the simplest plan (local first), then MEC,
        # then the search-based candidates, so ties are reported as the simple plan
        TIE = {"all_UE": 0, "all_MEC": 1, "all_HELPER": 2, "greedy_cd": 3,
               "heft_v2": 4, "stronger_search": 5}
        winner, winner_name = min((v, TIE[k], k) for k, v in panel.items())[::2]
        # The evidence must describe the plan that ACTUALLY won. `stronger_search` returns its
        # own plan (`sr.plan`), not the greedy `chosen` map: attributing the winner's helper use
        # and no-helper ablation to `chosen` mis-report the search winner's plan.
        if winner_name == "stronger_search":
            winning_plan = {int(k): int(v) for k, v in dict(sr.plan).items()}
        elif winner_name == "greedy_cd":
            winning_plan = {int(k): int(v) for k, v in chosen.items()}
        elif winner_name == "heft_v2":
            winning_plan = plan_map_from_actions(graph, list(heft_actions))
        else:
            winning_plan = plan_map_from_actions(
                graph, [{"all_UE": 0, "all_MEC": 1, "all_HELPER": 2}[winner_name]] * 20)
        if winner_name == "stronger_search":
            # verify the reported plan really reproduces the reported objective
            reproduced = _candidate_plan(world, winning_plan, link_process=link_process,
                                         reliability_gate=gate, evidence=evidence,
                                         standby_for=standby_rel if gate else None)
            if abs(reproduced - winner) > 1e-9 * max(1.0, abs(winner)):
                raise RuntimeError(
                    "gate winner evidence does not reproduce the winning objective for %s: "
                    "%.9f vs %.9f" % (graph.graph_id, reproduced, winner))
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
            "winner_chosen_plan": {str(k): int(v) for k, v in
                                   winning_plan.items()},
            "winner_evidence": _winner_evidence(
                world, winning_plan, link_process=link_process, reliability_gate=gate,
                evidence=evidence, standby_for=standby_rel if gate else None),
        })
    n = float(len(rows))
    return {
        "regime": spec["regime_name"], "graphs": len(rows),
        "all_MEC_winner_fraction": sum(1 for r in rows if r["all_MEC_winner"]) / n,
        "all_MEC_winner_fraction_no_search": sum(1 for r in rows if r["all_MEC_winner_no_search"]) / n,
        "local_winner_fraction": sum(1 for r in rows if r["local_wins"]) / n,
        "stronger_search_gain_vs_all_MEC_mean_pct": fmean(
            [r["stronger_search_gain_vs_all_MEC_pct"] for r in rows]),
        "all_UE_winner_fraction": sum(1 for r in rows if r["winner"] == "all_UE") / n,
        "all_HELPER_winner_fraction": sum(1 for r in rows if r["winner"] == "all_HELPER") / n,
        "mixed_winner_fraction": sum(1 for r in rows if r["winner"] in
                                     ("greedy_cd", "heft_v2", "stronger_search")) / n,
        "winner_counts": {k: sum(1 for r in rows if r["winner"] == k) for k in
                          ("all_UE", "all_MEC", "all_HELPER", "greedy_cd", "heft_v2",
                           "stronger_search")},
        "headroom_vs_best_pure_mean_pct": fmean([r["headroom_vs_best_pure_pct"] for r in rows]),
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
