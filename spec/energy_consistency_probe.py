#!/usr/bin/env python3
"""Measure the timing/energy physics mismatch and record it as evidence.

The primary config pairs `timing_model=legacy_frozen_rates` with
`energy_model=physical_v1`, so the reported compute joules come from the workload
(`E = kappa*C*f^2`) while the scheduled duration comes from the frozen rate table.
The two are only comparable through `E = P(f)*T` when both use the same machine.

This probe schedules real frozen graphs under BOTH configs and reports, per tier:
  * workload-derived joules and duration-consistent joules,
  * the measured ratio and the expected `R_scheduled / R_physical`,
  * the latency (makespan) under each config, so the effect of the physics choice
    on the timing axis is recorded separately rather than silently mixed in.

Usage (CPU is enough):
  python -m spec.energy_consistency_probe --json reports/v0.3-audit/energy_consistency/energy_consistency_evidence.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

SCHEMA = "energy_consistency_v1"
TOLERANCE = 1e-9
DATA_ROOT = Path("env/mec_offloaing_envs/data/meta_offloading_20")


def round_robin_plan(decoder_order):
    """Deterministic local/MEC/helper plan used by the other probes and tests."""
    return [(int(tid), int(k % 3)) for k, tid in enumerate(decoder_order)]


def summarise(samples: list[dict]) -> dict:
    """Pure aggregation of per-(config, graph) consistency records."""
    if not samples:
        raise ValueError("no samples to summarise")
    tiers = sorted({tier for s in samples for tier in s["tiers"]})
    out = {}
    for tier in tiers:
        rows = [s["tiers"][tier] for s in samples if tier in s["tiers"]]
        expected = [s["expected_ratio"].get(tier) for s in samples if tier in s["tiers"]]
        measured = [
            row["ratio_workload_over_duration"] for row in rows
            if row["duration_consistent_joules"] > 0.0
        ]
        exp = [e for e, row in zip(expected, rows) if row["duration_consistent_joules"] > 0.0]
        out[tier] = {
            "n_graphs": len(rows),
            "mean_busy_seconds": _mean([row["busy_seconds"] for row in rows]),
            "mean_workload_joules": _mean([row["workload_joules"] for row in rows]),
            "mean_duration_consistent_joules": _mean(
                [row["duration_consistent_joules"] for row in rows]
            ),
            "mean_measured_ratio": _mean(measured),
            "mean_expected_ratio": _mean(exp),
            "max_abs_identity_error": max(
                (abs(m - e) for m, e in zip(measured, exp)), default=0.0
            ),
        }
    return out


def _mean(values):
    values = [float(v) for v in values]
    if not values:
        return 0.0
    return sum(values) / float(len(values))


def verdict(primary: dict, co_physical: dict, *, tolerance: float = 1e-9) -> dict:
    """Judge whether the mixed-physics effect is real and quantified."""
    primary_tiers = primary.get("tiers", primary)
    physical_tiers = co_physical.get("tiers", co_physical)
    primary_measured = [row["mean_measured_ratio"] for row in primary_tiers.values()]
    physical_measured = [row["mean_measured_ratio"] for row in physical_tiers.values()]
    identity_errors = [
        row["max_abs_identity_error"] for row in primary_tiers.values()
    ] + [row["max_abs_identity_error"] for row in physical_tiers.values()]
    mixed_off_unity = any(abs(r - 1.0) > 1e-6 for r in primary_measured)
    co_physical_unity = all(abs(r - 1.0) <= 1e-6 for r in physical_measured if r > 0.0)
    identity_ok = all(err <= 1e-6 for err in identity_errors)
    label = (
        "MIXED_PHYSICS_MEASURED"
        if (mixed_off_unity and co_physical_unity and identity_ok)
        else "INCONSISTENT_EVIDENCE"
    )
    return {
        "verdict": label,
        "primary_ratio_off_unity": mixed_off_unity,
        "co_physical_ratio_is_unity": co_physical_unity,
        "ratio_identity_holds": identity_ok,
        "max_abs_identity_error": max(identity_errors, default=0.0),
        "tolerance": float(tolerance),
        "requirement": (
            "any run that makes an energy claim must either use the co-physical "
            "config (timing_model=physical_rates) or report the ratio as an "
            "explicit caveat; see spec/primary_config.require_energy_timing_consistency"
        ),
    }


def _sample(result, resources) -> dict:
    from env.mec_offloaing_envs.scheduler.energy_telemetry import duration_consistency

    record = duration_consistency(result, resources)
    return {
        "tiers": record["tiers"],
        "expected_ratio": record["ratio_expected_scheduled_over_physical"],
        "duration_consistent": record["duration_consistent"],
        "makespan_seconds": float(result.makespan_seconds),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", default="reports/v0.3-audit/energy_consistency/energy_consistency_evidence.json")
    parser.add_argument("--distributions", type=int, default=3)
    parser.add_argument("--graphs-per-distribution", type=int, default=5)
    args = parser.parse_args(argv)

    from env.mec_offloaing_envs.offloading_task_graph import OffloadingTaskGraph
    from env.mec_offloaing_envs.scheduler.adapter import schedule_via_adapter
    from spec.deadline_sweep import PrioritizeCluster
    from env.mec_offloaing_envs.scheduler.primary_config import (
        energy_timing_consistency,
        resolved_physical_scheduler_config,
        resolved_primary_scheduler_config,
    )
    from env.mec_offloaing_envs.scheduler.resources import resolved_config_sha256

    configs = {
        "primary_mixed": resolved_primary_scheduler_config(),
        "co_physical": resolved_physical_scheduler_config(),
    }
    graph_paths = []
    for dist in range(1, int(args.distributions) + 1):
        for index in range(int(args.graphs_per_distribution)):
            graph_paths.append(DATA_ROOT / ("offload_random20_%d" % dist) / ("random.20.%d.gv" % index))

    payload = {
        "schema": SCHEMA,
        "data_root": str(DATA_ROOT),
        "configs": {
            label: {
                **energy_timing_consistency(config),
                "scheduler_config_sha256": str(resolved_config_sha256(config)),
            }
            for label, config in configs.items()
        },
        "graphs": [str(p) for p in graph_paths],
        "per_config": {},
        "training_code_sha": os.environ.get("MARGO_TRAINING_CODE_SHA", ""),
        "cpu_only": True,
    }

    samples = {label: [] for label in configs}
    for path in graph_paths:
        if not path.is_file():
            raise SystemExit("missing graph: %s" % path)
        for label, config in configs.items():
            # HEFT priority needs the rate surface of THIS config, so the decoder
            # order is derived per config (and recorded) rather than shared.
            graph = OffloadingTaskGraph(str(path))
            graph.prioritize_tasks(PrioritizeCluster(config))
            order = [int(t) for t in graph.prioritize_sequence]
            payload.setdefault("decoder_order_sha", {})[label] = hashlib.sha256(
                ("\n".join(str(t) for t in order) + "|" + path.name).encode()
            ).hexdigest()
            plan = round_robin_plan(order)
            result, _deltas, _energy = schedule_via_adapter(graph, plan, config)
            samples[label].append(_sample(result, config))

    summary = {label: summarise(rows) for label, rows in samples.items()}
    makespans = {
        label: _mean([row["makespan_seconds"] for row in rows])
        for label, rows in samples.items()
    }
    latency_delta = makespans["co_physical"] - makespans["primary_mixed"]
    payload["per_config"] = summary
    payload["mean_makespan_seconds"] = makespans
    payload["physical_minus_primary_makespan_seconds"] = latency_delta
    payload["physical_minus_primary_makespan_percent"] = (
        100.0 * latency_delta / makespans["primary_mixed"] if makespans["primary_mixed"] else 0.0
    )
    payload.update(verdict(summary["primary_mixed"], summary["co_physical"], tolerance=TOLERANCE))
    payload["evidence_sha256_of_itself"] = None  # self-hash is meaningless; manifest covers it

    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    manifest = out.parent / "manifest.json"
    manifest.write_text(json.dumps({
        "files": [{
            "relative_path": out.name,
            "bytes": out.stat().st_size,
            "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        }],
        "schema": "energy_consistency_manifest_v1",
    }, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "verdict": payload["verdict"],
        "primary_mixed": {
            tier: round(row["mean_measured_ratio"], 6)
            for tier, row in summary["primary_mixed"]["tiers"].items()
        },
        "co_physical": {
            tier: round(row["mean_measured_ratio"], 6)
            for tier, row in summary["co_physical"]["tiers"].items()
        },
        "makespan_delta_percent": round(payload["physical_minus_primary_makespan_percent"], 4),
    }, indent=1, sort_keys=True))
    print("wrote", out)
    return 0 if payload["verdict"] == "MIXED_PHYSICS_MEASURED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
