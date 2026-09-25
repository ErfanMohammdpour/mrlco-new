#!/usr/bin/env python3
"""Build the Pilot A evidence JSON from the committed CSV + log bundle."""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

BASE = Path(__file__).resolve().parents[1] / "reports" / "v0.3-audit" / "pilot"
rows = [r for r in csv.DictReader(open(BASE / "pilot_a_progress.csv"))]
bundle = (BASE / "pilot_a_bundle.txt").read_text(errors="replace")
meta = dict(
    l.split("=", 1)
    for l in re.search(r"---META---\n(.*?)\n---EXIT---", bundle, re.S).group(1).strip().splitlines()
    if "=" in l
)
ex = dict(
    l.split("=", 1)
    for l in re.search(r"---EXIT---\n(.*)", bundle, re.S).group(1).strip().splitlines()
    if "=" in l
)


def vals(key):
    return [float(r[key]) for r in rows if r.get(key) not in (None, "")]


def summ(key):
    v = vals(key)
    return {"first": v[0], "last": v[-1], "min": min(v), "max": max(v)} if v else None


SERIES_KEYS = (
    "Average reward, ", "Average latency,", "action_fraction/local",
    "action_fraction/mec", "action_fraction/v2v", "policy/entropy_valid",
    "policy/policy_loss_mean", "policy/value_loss_mean", "policy/approx_kl",
    "policy/clip_fraction", "policy/grad_norm", "critic/value_abs_max",
    "collapse/flag",
)
series = {k: summ(k) for k in SERIES_KEYS if vals(k)}
collapse_seen = any(float(r["collapse/flag"]) != 0.0 for r in rows if r.get("collapse/flag") not in (None, ""))
mec, v2v, loc = vals("action_fraction/mec"), vals("action_fraction/v2v"), vals("action_fraction/local")
checks = {
    "no_collapse_watchdog": not collapse_seen,
    "mec_share_below_095": max(mec) < 0.95,
    "local_not_eliminated": min(loc) > 0.0,
    "v2v_not_eliminated": min(v2v) > 0.0,
    "reward_trend_up": series["Average reward, "]["last"] > series["Average reward, "]["first"],
    "latency_trend_down": series["Average latency,"]["last"] < series["Average latency,"]["first"],
    "all_series_finite": all(all(x == x for x in vals(k)) for k in series),
    "value_abs_max_below_limit": max(vals("critic/value_abs_max")) < 1e3,
    "csv_row_count_25": len(rows) == 25,
}
verdict_note = (
    "No collapse (flag never set, MEC share max %.3f) and a clear training trend "
    "(reward %.3f->%.3f, latency %.1f->%.1f) with Local and V2V preserved. However the "
    "frozen validation_interval=50 exceeds the 25-iteration pilot, so there is only ONE "
    "validation point (itr 0) and the PASS criteria 'validation improves vs init' and "
    "'best validation beats all-MEC' cannot be evaluated; the contract says extend."
) % (
    max(mec),
    series["Average reward, "]["first"], series["Average reward, "]["last"],
    series["Average latency,"]["first"], series["Average latency,"]["last"],
)
evidence = {
    "schema": "pilot_a_evidence_v1",
    "code_sha": meta.get("sha"),
    "tree_dirty": meta.get("dirty"),
    "image": meta.get("image"),
    "tf": meta.get("tf"),
    "gpu": meta.get("gpu"),
    "exit": int(ex.get("exit", -1)),
    "wall_s": int(ex.get("wall_s", -1)),
    "iterations": len(rows),
    "config": {
        "reward_mode": "latency_only", "objective_mode": "log_only", "constraints": "off",
        "mask_mode": "off", "obs_version": "v3", "seed": 0, "iterations": 25,
        "energy_model": "physical_v1", "radio_model": "physical_v1",
        "timing": "legacy_frozen_rates", "decoder_order": "legacy_current",
        "entropy_coefficient": 0.0,
    },
    "series": series,
    "validation_itr0": {
        "gap_to_all_mec": rows[0].get("validation/validation_gap_to_all_mec"),
        "gap_to_greedy": rows[0].get("validation/validation_gap_to_greedy"),
        "all_mec_latency": rows[0].get("validation/validation_all_mec_latency"),
        "greedy_latency": rows[0].get("validation/validation_greedy_latency"),
        "co_location_rate": rows[0].get("validation/co_location_rate"),
        "cross_location_edges": rows[0].get("validation/cross_location_edges"),
        "task_fraction_mec": rows[0].get("validation/task_fraction/mec"),
        "utilization_mean": rows[0].get("validation/utilization_mean"),
    },
    "validation_points": 1,
    "checks": checks,
    "all_checks_pass": all(checks.values()),
    "verdict": "INCONCLUSIVE_VALIDATION_UNMEASURABLE",
    "verdict_note": verdict_note,
}
(BASE / "pilot_a_evidence.json").write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
print(json.dumps(checks, indent=1, sort_keys=True))
print("all_checks_pass:", evidence["all_checks_pass"])
print("mec max %.3f | v2v min %.4f | local min %.4f" % (max(mec), min(v2v), min(loc)))
print("reward %.3f -> %.3f | latency %.1f -> %.1f" % (
    series["Average reward, "]["first"], series["Average reward, "]["last"],
    series["Average latency,"]["first"], series["Average latency,"]["last"]))
