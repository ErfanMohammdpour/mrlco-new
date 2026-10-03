#!/usr/bin/env python3
"""Problem-geometry sensitivity study (NO training, frozen dataset untouched).

Question: is the all-MEC collapse a property of the LEARNING algorithm, or of the
problem geometry (dedicated MEC, near-free internal communication, no energy cost in the
objective, no cross-DAG contention, static V2V weaker than V2I)?

This script keeps MARGO-AUTOMOTIVE-MC-v1 exactly as frozen and varies ONLY the analysis
knobs, per graph, on the same MC realizations:

  helper_xK        scale f_helper_hz by K                (tier strength)
  v2v_xK           scale r_v2v_bps by K                  (V2V vs V2I penalty)
  mec_share_N      divide f_mec_hz and the MEC radio rates by N
                   (SURROGATE for N concurrent DAGs sharing one MEC; v1 schedules a
                    single DAG per instance, so this models processor sharing, not a
                    queueing simulation)
  helper_direct    give HELPER a direct V2I link to MEC instead of the frozen
                   MEC_DL+V2V / V2V+MEC_UL two-hop route (sensitivity only)

Reported per setting: P(all-MEC is the panel oracle), mixed headroom vs all-MEC, HELPER
token share of the winning plan, communication time, cut edges, MEC busy share and a MEC
wait proxy. Also prints the raw geometry facts (compute/transfer times) so magnitude
claims can be checked.

CLI: python3 spec/automotive_training/geometry_sensitivity.py [--graphs N] [--json PATH]
"""

from __future__ import annotations

from spec.automotive_training.v2.compat import fmean  # noqa: E402
import argparse
import copy
import json
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spec.automotive_training.automotive_env import AutomotiveEnv  # noqa: E402
from spec.automotive_training.automotive_loader import load_dataset  # noqa: E402
from spec.automotive_training.automotive_primary import (  # noqa: E402
    AutomotiveResourceCluster,
    _greedy_plan,
)
from spec.automotive_training.automotive_resources import (  # noqa: E402
    co_physical_config_for_graph,
)
from spec.automotive_training.heft_reference_v2 import heft_reference_v2_plan  # noqa: E402

SETTINGS = {
    "baseline_frozen": {},
    "helper_x0.5": {"helper": 0.5},
    "helper_x2": {"helper": 2.0},
    "helper_x4": {"helper": 4.0},
    "v2v_x0.25": {"v2v": 0.25},
    "v2v_x4": {"v2v": 4.0},
    "mec_share_2": {"mec_share": 2},
    "mec_share_4": {"mec_share": 4},
    "mec_share_8": {"mec_share": 8},
    "helper_direct_v2i": {"direct": True},
    "contention4_helper_x2": {"mec_share": 4, "helper": 2.0},
    "contention4_helper_x2_direct": {"mec_share": 4, "helper": 2.0, "direct": True},
}


def scaled_resource(resource: dict, spec: dict) -> dict:
    out = dict(resource)
    if "helper" in spec:
        out["f_helper_hz"] = float(resource["f_helper_hz"]) * float(spec["helper"])
    if "v2v" in spec:
        out["r_v2v_bps"] = float(resource["r_v2v_bps"]) * float(spec["v2v"])
    if "mec_share" in spec:
        share = 1.0 / float(spec["mec_share"])
        out["f_mec_hz"] = float(resource["f_mec_hz"]) * share
        out["r_mec_ul_bps"] = float(resource["r_mec_ul_bps"]) * share
        out["r_mec_dl_bps"] = float(resource["r_mec_dl_bps"]) * share
    return out


_ROUTE_SNAPSHOT = None


def patch_routes(direct: bool):
    """Install or remove the sensitivity-only direct HELPER<->MEC V2I route.

    The snapshot is taken ONCE so that toggling off always restores the true frozen
    routes (a previous version re-snapshotted the patched table and leaked the patch into
    every later setting).
    """
    global _ROUTE_SNAPSHOT
    from env.mec_offloaing_envs.scheduler import engine, radio as radio_mod, routes
    from env.mec_offloaing_envs.scheduler.model import Location

    if _ROUTE_SNAPSHOT is None:
        _ROUTE_SNAPSHOT = {
            "route_mh": list(routes.ROUTE_TABLE[(Location.MEC, Location.HELPER)]),
            "route_hm": list(routes.ROUTE_TABLE[(Location.HELPER, Location.MEC)]),
            "hop_destination": routes.hop_destination,
        }
    snap = _ROUTE_SNAPSHOT
    routes.ROUTE_TABLE[(Location.MEC, Location.HELPER)] = ["MEC_DLH"] if direct else list(snap["route_mh"])
    routes.ROUTE_TABLE[(Location.HELPER, Location.MEC)] = ["MEC_ULH"] if direct else list(snap["route_hm"])
    routes.HOP_TO_RESOURCE["MEC_DLH"] = "MEC_DL"
    routes.HOP_TO_RESOURCE["MEC_ULH"] = "MEC_UL"
    radio_mod.HOP_TO_LINK["MEC_DLH"] = "v2i_dl"
    radio_mod.HOP_TO_LINK["MEC_ULH"] = "v2i_ul"
    original = snap["hop_destination"]

    def patched(cur, hop):
        if hop == "MEC_DLH":
            return Location.HELPER
        if hop == "MEC_ULH":
            return Location.MEC
        return original(cur, hop)

    routes.hop_destination = patched if direct else original
    engine.hop_destination = patched if direct else original
    return snap


def hops_of(result) -> list:
    return [str(getattr(t, "hop", "?")) for t in (getattr(result, "transfers", []) or [])]


def schedule_metrics(env, index, plan, mc) -> dict:
    result, _energy, _macro = env._schedule(index, list(plan), mc)
    makespan = float(result.makespan_seconds)
    transfers = list(getattr(result, "transfers", []) or [])
    comm = sum(float(t.end) - float(t.start) for t in transfers)
    intervals = [i for i in (getattr(result, "resource_intervals", []) or [])
                 if getattr(i, "resource", "") == "MEC_CPU"]
    mec_busy = sum(float(i.end) - float(i.start) for i in intervals)
    # DAG-free wait proxy: the MEC was reserved (first..last interval) but busy only part
    # of that span, so the remainder is time the MEC spent idle waiting on dependencies.
    mec_span = (max(float(i.end) for i in intervals) - min(float(i.start) for i in intervals)
                if intervals else 0.0)
    return {"makespan_s": makespan, "comm_s": comm, "cut_edges": len(transfers),
            "mec_busy_s": mec_busy, "mec_wait_s": max(0.0, mec_span - mec_busy),
            "helper_tokens": int(sum(1 for a in plan if int(a) == 2)),
            "plan": [int(a) for a in plan]}


def evaluate_graph(env, index, resource_base, heft_plan, spec) -> dict:
    from spec.automotive_training.automotive_env import AutomotiveEnv as Env

    graph = env.graph_objects[index]
    mc = env._slot_mc[index]
    config = co_physical_config_for_graph(scaled_resource(resource_base, spec),
                                          source_sha256="sensitivity")
    env._configs_override = getattr(env, "_configs_override", {})
    # temporary per-graph config swap (analysis only; the env keeps its own bookkeeping)
    env.configs[index] = config
    entry = {}
    for name, action in (("all_UE", 0), ("all_MEC", 1), ("all_HELPER", 2)):
        entry[name] = schedule_metrics(env, index, [action] * 20, mc)
    greedy = _greedy_plan(env, index, mc)
    entry["greedy_cd"] = schedule_metrics(env, index, greedy, mc)
    entry["heft_v2"] = schedule_metrics(env, index, heft_plan, mc)
    candidates = {k: v["makespan_s"] for k, v in entry.items()}
    oracle, oracle_name = min((v, k) for k, v in candidates.items())
    allmec = entry["all_MEC"]["makespan_s"]
    winner = entry[oracle_name]
    entry["_summary"] = {
        "oracle_name": oracle_name, "oracle_s": oracle, "all_MEC_s": allmec,
        "all_MEC_optimal": bool(oracle_name == "all_MEC"),
        "headroom_pct": 100.0 * (allmec - oracle) / max(allmec, 1e-12),
        "helper_token_share": winner["helper_tokens"] / 20.0,
        "comm_s": winner["comm_s"], "cut_edges": winner["cut_edges"],
        "mec_busy_s": winner["mec_busy_s"], "mec_wait_s": winner["mec_wait_s"],
        "makespan_s": oracle,
    }
    return entry


def run_split(graphs, label: str, settings: dict) -> dict:
    env = AutomotiveEnv(graphs, AutomotiveResourceCluster(), role="validation",
                        single_dist=True, slots_per_task=len(graphs), base_seed=303)
    env.set_task({"dist_index": 0, "graph_indices": np.arange(len(graphs), dtype=np.int32)})
    env.reset()
    heft_cache = {g.graph_id: heft_reference_v2_plan(g.as_record(), co_physical=True)[0]
                  for g in graphs}
    out = {}
    for name, spec in settings.items():
        frozen = patch_routes(bool(spec.get("direct")))
        rows = []
        for index, graph in enumerate(graphs):
            entry = evaluate_graph(env, index, graph.resource, heft_cache[graph.graph_id], spec)
            rows.append(entry["_summary"])
        patch_routes(False)
        n = float(len(rows))
        out[name] = {
            "graphs": len(rows),
            "all_MEC_optimal_share": sum(1 for r in rows if r["all_MEC_optimal"]) / n,
            "mean_headroom_pct": fmean([r["headroom_pct"] for r in rows]),
            "median_headroom_pct": statistics.median([r["headroom_pct"] for r in rows]),
            "max_headroom_pct": max(r["headroom_pct"] for r in rows),
            "mean_helper_token_share": fmean([r["helper_token_share"] for r in rows]),
            "mean_comm_ms": 1000.0 * fmean([r["comm_s"] for r in rows]),
            "mean_cut_edges": fmean([r["cut_edges"] for r in rows]),
            "mean_mec_busy_ms": 1000.0 * fmean([r["mec_busy_s"] for r in rows]),
            "mean_mec_wait_ms": 1000.0 * fmean([r["mec_wait_s"] for r in rows]),
            "mean_all_MEC_ms": 1000.0 * fmean([r["all_MEC_s"] for r in rows]),
            "oracle_mix": {k: sum(1 for r in rows if r["oracle_name"] == k) for k in
                           ("all_UE", "all_MEC", "all_HELPER", "greedy_cd", "heft_v2")},
            "split": label,
        }
    return out


def geometry_facts(graphs, n: int = 20) -> dict:
    """Raw magnitudes: per-tier compute time and per-hop transfer time (medians)."""
    from env.mec_offloaing_envs.scheduler import radio as radio_mod

    compute = {"UE": [], "MEC": [], "HELPER": []}
    hops = {"MEC_UL": [], "MEC_DL": [], "V2V": []}
    cut_extra = []
    for graph in graphs[:n]:
        cfg = co_physical_config_for_graph(graph.resource, source_sha256="facts")
        rates = {"UE": cfg.ue_cpu_bytes_per_second, "MEC": cfg.mec_cpu_bytes_per_second,
                 "HELPER": cfg.helper_cpu_bytes_per_second}
        hop_rates = {"MEC_UL": cfg.mec_uplink_bytes_per_second,
                     "MEC_DL": cfg.mec_downlink_bytes_per_second,
                     "V2V": cfg.v2v_bytes_per_second}
        for task in graph.as_record()["tasks"]:
            work = float(task["compute_workload_bytes"])
            for tier, rate in rates.items():
                compute[tier].append(1000.0 * work / rate)
        inter = [float(e["payload_bytes"]) for e in graph.as_record()["edges"]] or [12000.0]
        hop_rates = {name: hop_rates[name] for name in hop_rates}
        for hop in hops:
            hops[hop].extend([1000.0 * b / hop_rates[hop] for b in inter])
        # cost of routing an intermediate node off MEC: MEC_DL + V2V (+ V2V + MEC_UL back)
        cut_extra.extend([1000.0 * b / hop_rates["MEC_DL"] + 1000.0 * b / hop_rates["V2V"]
                          for b in inter])
    return {
        "median_compute_ms": {k: statistics.median(v) for k, v in compute.items()},
        "median_transfer_ms": {k: statistics.median(v) for k, v in hops.items()},
        "median_mec_dl_plus_v2v_ms": statistics.median(cut_extra) if cut_extra else None,
        "n_graphs": min(n, len(graphs)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs", type=int, default=40)
    ap.add_argument("--json", default=str(ROOT / "spec/automotive_training/reports/"
                                              "gateI/geometry_sensitivity.json"))
    args = ap.parse_args()
    ds = load_dataset()
    val = ds.validation_query()[: int(args.graphs)]
    # family-stratified meta-train sample (the training draw is uniform over 60 map + 60
    # ppc, so a stratified half/half sample is the low-variance version of the same thing)
    import re as _re

    def _fam(g):
        m = _re.match(r"mcv1_([a-z]+)_", g.graph_id)
        return m.group(1) if m else "other"

    per_family = max(1, int(args.graphs) // 2)
    meta = []
    for family in ("map", "ppc"):
        members = [g for g in ds.meta_train() if _fam(g) == family]
        meta.extend(members[:per_family])
    meta = sorted(meta, key=lambda g: g.graph_id)
    out = {
        "schema": "automotive_geometry_sensitivity_v1",
        "note": ("v1 dataset untouched; only analysis knobs vary. mec_share_N scales the MEC "
                 "compute and MEC radio rates by 1/N as a processor-sharing surrogate for N "
                 "concurrent DAGs (v1 schedules one DAG per instance). helper_direct_v2i "
                 "installs a direct HELPER<->MEC V2I route as a sensitivity variant only."),
        "geometry_facts_validation_query": geometry_facts(ds.validation_query()),
        "validation_query": run_split(val, "validation_query", SETTINGS),
        "meta_train_sample": run_split(meta, "meta_train_sample", SETTINGS),
    }
    # self-check: the direct-link patch must actually change the used hops
    from spec.automotive_training.automotive_env import AutomotiveEnv as _Env
    from spec.automotive_training.automotive_resources import co_physical_config_for_graph as _cfg
    _probe_graphs = ds.validation_query()[:1]
    _probe = _Env(_probe_graphs, AutomotiveResourceCluster(), role="validation",
                  single_dist=True, slots_per_task=1, base_seed=303)
    _probe.set_task({"dist_index": 0, "graph_indices": np.array([0], dtype=np.int32)})
    _probe.reset()
    _mc = _probe._slot_mc[0]
    _hops = {}
    for _direct in (False, True):
        patch_routes(_direct)
        _probe.configs[0] = _cfg(_probe_graphs[0].resource, source_sha256="selfcheck")
        # a MIXED plan is required: an all-HELPER plan never uses MEC->HELPER routes
        _res, _e, _m = _probe._schedule(0, ([1, 2] * 10), _mc)
        _hops["direct_%s" % _direct] = sorted(set(hops_of(_res)))
    patch_routes(False)
    out["route_patch_self_check"] = _hops
    print("route patch self-check (MIXED plan hops):", json.dumps(_hops, sort_keys=True))
    for split in ("validation_query", "meta_train_sample"):
        print("== %s (n=%d) ==" % (split, out[split]["baseline_frozen"]["graphs"]))
        print("   %-30s %8s %10s %10s %10s %10s %10s %10s %10s" % (
            "setting", "P(MEC*)", "head%", "med%", "max%", "helper%", "comm_ms", "cuts", "mecbusy"))
        for name, b in out[split].items():
            print("   %-30s %8.3f %10.2f %10.2f %10.2f %10.4f %10.3f %10.2f %10.3f" % (
                name, b["all_MEC_optimal_share"], b["mean_headroom_pct"], b["median_headroom_pct"],
                b["max_headroom_pct"], b["mean_helper_token_share"], b["mean_comm_ms"],
                b["mean_cut_edges"], b["mean_mec_busy_ms"]))
        print()
    print("geometry facts (medians, ms):", json.dumps(out["geometry_facts_validation_query"]["median_compute_ms"]),
          json.dumps(out["geometry_facts_validation_query"]["median_transfer_ms"]),
          "mec_dl+v2v =", out["geometry_facts_validation_query"]["median_mec_dl_plus_v2v_ms"])
    path = Path(args.json)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, sort_keys=True, default=str) + "\n")
    print("written", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
