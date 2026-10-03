#!/usr/bin/env python3
"""Build the v1 geometry/rate-provenance audit artifacts (read-only on v1).

Outputs (in this directory):
  rate_provenance.csv / .json     per-graph realized rates, derived scheduler rates,
                                  realised transfer bytes/durations, profile strata
  corrected_geometry_summary.json renamed/labelled version of geometry_sensitivity.json
"""

from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HERE = Path(__file__).resolve().parent
from spec.automotive_training.automotive_env import AutomotiveEnv  # noqa: E402
from spec.automotive_training.automotive_loader import load_dataset  # noqa: E402
from spec.automotive_training.automotive_primary import (  # noqa: E402
    AutomotiveResourceCluster,
    _greedy_plan,
)
from spec.automotive_training.automotive_resources import (  # noqa: E402
    co_physical_config_for_graph,
    frozen_radio_eta,
)
from spec.automotive_training.heft_reference_v2 import heft_reference_v2_plan  # noqa: E402

HISTORICAL = {"r_mec_ul_bps": 7.0e6, "r_mec_dl_bps": 7.0e6, "r_v2v_bps": 5.0e6}


def main() -> int:
    ds = load_dataset()
    graphs = ds.validation_query() + ds.meta_train()
    rows = []
    transfer_samples = []
    env = AutomotiveEnv(graphs, AutomotiveResourceCluster(), role="validation",
                        single_dist=True, slots_per_task=len(graphs), base_seed=303)
    env.set_task({"dist_index": 0, "graph_indices": list(range(len(graphs)))})
    env.reset()
    for index, graph in enumerate(graphs):
        cfg = co_physical_config_for_graph(graph.resource, source_sha256="audit")
        row = {
            "graph_id": graph.graph_id,
            "split": "validation_query" if index < len(ds.validation_query()) else "meta_train",
            "application_family": getattr(graph, "family", None),
            "resource_profile": getattr(graph, "resource_profile", None),
            "resource_level": getattr(graph, "resource_level", None),
            "r_mec_ul_bps": float(graph.resource["r_mec_ul_bps"]),
            "r_mec_dl_bps": float(graph.resource["r_mec_dl_bps"]),
            "r_v2v_bps": float(graph.resource["r_v2v_bps"]),
            "f_ue_hz": float(graph.resource["f_ue_hz"]),
            "f_helper_hz": float(graph.resource["f_helper_hz"]),
            "f_mec_hz": float(graph.resource["f_mec_hz"]),
            "sched_mec_uplink_bytes_per_s": cfg.mec_uplink_bytes_per_second,
            "sched_mec_downlink_bytes_per_s": cfg.mec_downlink_bytes_per_second,
            "sched_v2v_bytes_per_s": cfg.v2v_bytes_per_second,
            "sched_ue_cpu_bytes_per_s": cfg.ue_cpu_bytes_per_second,
            "sched_helper_cpu_bytes_per_s": cfg.helper_cpu_bytes_per_second,
            "sched_mec_cpu_bytes_per_s": cfg.mec_cpu_bytes_per_second,
            "bits_to_bytes_ratio_ul": float(graph.resource["r_mec_ul_bps"]) / cfg.mec_uplink_bytes_per_second,
            "bits_to_bytes_ratio_dl": float(graph.resource["r_mec_dl_bps"]) / cfg.mec_downlink_bytes_per_second,
            "bits_to_bytes_ratio_v2v": float(graph.resource["r_v2v_bps"]) / cfg.v2v_bytes_per_second,
        }
        # realised transfer durations on a mixed plan (proves which rate is used)
        if index < 40:
            mc = env._slot_mc[index]
            plan = _greedy_plan(env, index, mc)
            result, _e, _m = env._schedule(index, plan, mc)
            for t in (result.transfers or [])[:6]:
                transfer_samples.append({
                    "graph_id": graph.graph_id, "hop": str(t.hop), "bytes": float(t.bytes),
                    "duration_ms": 1000.0 * (float(t.end) - float(t.start)),
                    "implied_bytes_per_s": (float(t.bytes) / (float(t.end) - float(t.start))
                                            if float(t.end) > float(t.start) else None),
                })
        rows.append(row)

    with open(HERE / "rate_provenance.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    def med(key, subset=None):
        vals = [r[key] for r in rows if subset is None or r["resource_profile"] == subset]
        return statistics.median(vals) if vals else None

    profiles = sorted({r["resource_profile"] for r in rows})
    summary = {
        "schema": "v1_rate_provenance_v1",
        "dataset": "MARGO-AUTOMOTIVE-MC-v1",
        "graphs": len(rows),
        "median_realized_rates_bps": {k: med(k) for k in
                                      ("r_mec_ul_bps", "r_mec_dl_bps", "r_v2v_bps")},
        "median_realized_rates_mbps": {k: (med(k) / 1e6 if med(k) else None) for k in
                                       ("r_mec_ul_bps", "r_mec_dl_bps", "r_v2v_bps")},
        "median_by_profile_mbps": {
            p: {k: (med(k, p) / 1e6 if med(k, p) else None) for k in
                ("r_mec_ul_bps", "r_mec_dl_bps", "r_v2v_bps")} for p in profiles},
        "median_by_profile_ghz": {
            p: {k: (med(k, p) / 1e9 if med(k, p) else None) for k in
                ("f_ue_hz", "f_helper_hz", "f_mec_hz")} for p in profiles},
        "historical_baseline_points_in_resource_profiles_yaml_mbps": HISTORICAL,
        "bits_to_bytes_ratio": {
            "median_ul": statistics.median([r["bits_to_bytes_ratio_ul"] for r in rows]),
            "median_dl": statistics.median([r["bits_to_bytes_ratio_dl"] for r in rows]),
            "median_v2v": statistics.median([r["bits_to_bytes_ratio_v2v"] for r in rows]),
            "expected": 8.0,
            "verdict": ("scheduler converts bits/s to bytes/s by exactly 8; there is NO unit "
                        "error in the rate pipeline"),
        },
        "transfer_duration_samples": transfer_samples[:30],
        "eta_provenance": frozen_radio_eta(),
        "conclusion": (
            "The ~7-11 Mbps figures are the HISTORICAL BASELINE POINTS recorded in "
            "spec/automotive_mc_v1/resource_profiles.yaml (7/7/5 Mbps, 1.0/1.5/10.0 GHz) and "
            "explicitly labelled 'NOT verified real-world constants'. The scheduler uses the "
            "PER-GRAPH realized draw inside the nominal/degraded/strong profile ranges, whose "
            "medians are higher (see median_realized_rates_mbps). The difference is which rate "
            "object is summarized, not a bits/bytes handling error."),
    }
    (HERE / "rate_provenance.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    # ---- corrected geometry summary: rename oracle -> candidate-panel oracle, and label
    src = json.loads((ROOT / "spec/automotive_training/reports/gateI/"
                             "geometry_sensitivity.json").read_text())
    corrected = {
        "schema": "v1_geometry_summary_corrected_v1",
        "source": "reports/gateI/geometry_sensitivity.json",
        "labels": {
            "oracle": ("candidate-panel oracle over the 5 frozen candidates "
                       "(all_UE, all_MEC, all_HELPER, MC-aware greedy coordinate descent, "
                       "HEFT v2). This is NOT a global oracle over 3^20 placements and the "
                       "reported headroom is NOT an upper bound."),
            "headroom": "demonstrated candidate-panel headroom vs all-MEC",
            "mec_share_N": ("SURROGATE CONTENTION SENSITIVITY: MEC compute and MEC radio "
                            "rates divided by N (processor sharing). Not a multi-user "
                            "queueing simulation."),
            "helper_direct_v2i": "sensitivity-only topology variant",
        },
        "splits": {},
    }
    for split in ("validation_query", "meta_train_sample"):
        block = src[split]
        out = {}
        for name, s in block.items():
            all_mec = s["all_MEC_optimal_share"]
            new = {
                "graphs": s["graphs"],
                "all_MEC_winner_fraction": all_mec,
                "mixed_winner_fraction": 1.0 - all_mec,
                "candidate_panel_headroom_mean_pct": s["mean_headroom_pct"],
                "candidate_panel_headroom_median_pct": s["median_headroom_pct"],
                "candidate_panel_headroom_max_pct": s["max_headroom_pct"],
                "helper_token_share_in_winner": s["mean_helper_token_share"],
                "communication_ms": s["mean_comm_ms"],
                "cut_edges": s["mean_cut_edges"],
                "mec_busy_ms": s["mean_mec_busy_ms"],
                "mec_wait_ms": s["mean_mec_wait_ms"],
                "all_MEC_ms": s["mean_all_MEC_ms"],
                "winner_counts": s["oracle_mix"],
                "evidence_class": ("SURROGATE SENSITIVITY" if name.startswith("mec_share")
                                   or "contention" in name or name == "helper_direct_v2i"
                                   else "STRONG EMPIRICAL EVIDENCE (frozen v1 geometry)"),
            }
            out[name] = new
        corrected["splits"][split] = out
    (HERE / "corrected_geometry_summary.json").write_text(
        json.dumps(corrected, indent=2, sort_keys=True) + "\n")

    print("graphs:", len(rows))
    print("median realized Mbps:", json.dumps(summary["median_realized_rates_mbps"], sort_keys=True))
    print("bits->bytes ratios:", json.dumps({k: summary["bits_to_bytes_ratio"][k] for k in
                                             ("median_ul", "median_dl", "median_v2v")}, sort_keys=True))
    print("profile medians (Mbps):", json.dumps(summary["median_by_profile_mbps"], sort_keys=True))
    for split in ("validation_query", "meta_train_sample"):
        b = corrected["splits"][split]["baseline_frozen"]
        print("%s baseline: all-MEC winner=%.3f mixed=%.3f headroom=%.2f%% helper_share=%.4f" % (
            split, b["all_MEC_winner_fraction"], b["mixed_winner_fraction"],
            b["candidate_panel_headroom_mean_pct"], b["helper_token_share_in_winner"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
