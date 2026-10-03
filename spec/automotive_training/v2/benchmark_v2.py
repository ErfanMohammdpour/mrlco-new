#!/usr/bin/env python3
"""v2 CPU-side throughput benchmark + artifact generation (no TensorFlow required).

Measures the phases that do not need the TF policy/sampler:
  world build / env.reset / observation encoding / one validated rollout step
  (telescoping prefixes + final validated schedule + energy ledger + constraints)
and the raw schedule throughput. It also writes the energy spec and observation schema
artifacts from the LIVE frozen configuration so they cannot drift from the code.

    python3 -m spec.automotive_training.v2.benchmark_v2 --graphs 10 --iters 3
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REPORTS = ROOT / "spec" / "automotive_training" / "reports" / "v2_system_model"

from spec.automotive_training.automotive_loader import load_dataset  # noqa: E402
from spec.automotive_training.automotive_primary import AutomotiveResourceCluster  # noqa: E402
from spec.automotive_training.v2.energy import (  # noqa: E402
    UNMODELED, frozen_energy_spec, reference_ranges_from_plans, schedule_energy,
)
from spec.automotive_training.v2.env import V2AutomotiveEnv  # noqa: E402
from spec.automotive_training.v2.world import (  # noqa: E402
    V2WorldConfig, build_world, pure_location_worlds,
)

TOKENS = 20


def _time(fn, repeat=1):
    best = float("inf")
    out = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t0)
    return best, out


def benchmark(graphs: int, iters: int, slots: int, background: int) -> dict:
    dataset = load_dataset()
    val = dataset.validation_query()[: int(graphs)]
    out: dict = {
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "tensorflow": "NOT INSTALLED (TF phases are NOT RUN)",
            "cuda": "none",
        },
        "config": {"graphs": len(val), "slots": int(slots), "background_dags": int(background),
                   "tokens_per_trajectory": TOKENS, "iterations_measured": int(iters)},
        "phases": {},
        "schedules_per_second": {},
        "notes": [],
    }

    # ---- world build -----------------------------------------------------
    def build():
        return [build_world(g, slot_id=i, mc=None,
                            config=V2WorldConfig(background_dags=background), helper_seed=i)
                for i, g in enumerate(val)]

    dt_build, worlds = _time(build, repeat=1)
    out["phases"]["world_build_s"] = dt_build

    # ---- raw schedule throughput ----------------------------------------
    world = worlds[0]
    graph0 = val[0]
    actions = [1] * TOKENS

    def one_schedule(validate):
        w = world.with_foreground_actions(graph0, actions)
        return w.schedule(validate=validate)

    r1, _ = _time(lambda: one_schedule(True), repeat=5)
    r2, _ = _time(lambda: one_schedule(False), repeat=5)
    out["phases"]["schedule_validated_s"] = r1
    out["phases"]["schedule_unvalidated_s"] = r2
    out["schedules_per_second"]["validated"] = 1.0 / max(r1, 1e-12)
    out["schedules_per_second"]["unvalidated"] = 1.0 / max(r2, 1e-12)

    # ---- env.reset / step / encoding ------------------------------------
    env = V2AutomotiveEnv(val, AutomotiveResourceCluster(), role="validation",
                          slots_per_task=int(slots), base_seed=303, single_dist=True,
                          background_dags=int(background))
    env.set_task({"dist_index": 0, "graph_indices": np.arange(min(slots, len(val)),
                                                              dtype=np.int32)})
    dt_reset, _ = _time(env.reset, repeat=1)
    out["phases"]["env_reset_s"] = dt_reset
    n_slots = len(np.asarray(env.graph_indices).reshape(-1))
    plan = np.ones((n_slots, TOKENS), dtype=int)

    dt_step, _ = _time(lambda: env.step(plan), repeat=max(1, int(iters)))
    out["phases"]["env_step_s"] = dt_step
    # per-phase decomposition of one step
    slot = 0
    dt_tel, _ = _time(lambda: env._telescoping(slot, list(plan[slot])), repeat=3)
    out["phases"]["telescoping_prefixes_s_per_slot"] = dt_tel
    dt_final, _ = _time(lambda: env._schedule_slot(slot, list(plan[slot]), validate=True),
                        repeat=3)
    out["phases"]["final_validated_schedule_s_per_slot"] = dt_final
    res = env._schedule_slot(slot, list(plan[slot]), validate=True)
    dt_energy, led = _time(lambda: schedule_energy(res), repeat=5)
    out["phases"]["energy_ledger_s_per_slot"] = dt_energy
    out["energy_example"] = led.as_dict()
    dt_refs, refs = _time(lambda: reference_ranges_from_plans(
        {name: w.schedule(validate=False) for name, w in pure_location_worlds(
            val[0], slot_id=0, config=V2WorldConfig(background_dags=0), helper_seed=1).items()},
        scheduler_config_sha256="0" * 64), repeat=1)
    out["phases"]["reference_ranges_s_per_slot"] = dt_refs

    # throughput extrapolation (CPU phases only, explicitly NOT a GPU training estimate)
    per_iter = (dt_reset + 10 * dt_step)     # 10 meta-batch slots, one step each
    out["cpu_iteration_estimate_s"] = per_iter
    out["cpu_iteration_estimate_note"] = (
        "CPU rollout phases only (reset + 10 slots x one validated step incl. 21 telescoping "
        "prefixes, energy ledger and constraints). PPO inner steps, the MRLCO outer update "
        "and validation need TensorFlow and are NOT RUN here; they must be measured in the "
        "TF environment.")
    out["phases"]["schedule_throughput_vs_prior_claim"] = {
        "prior_measured": "~0.4-3.5 ms per schedule, 91.7 s per support rollout (CPU)",
        "now_ms_per_validated_schedule": 1000.0 * r1,
        "now_ms_per_unvalidated_schedule": 1000.0 * r2,
    }
    return out


def write_energy_spec(root: Path, spec) -> dict:
    rows = []
    for tier, ts in sorted(spec.tiers.items()):
        rows.append({
            "parameter": "kappa_%s" % tier, "value": ts.kappa, "unit": "J/(cycle*Hz^2)",
            "source": ts.source,
        })
        rows.append({"parameter": "f_%s" % tier, "value": ts.f_hz, "unit": "Hz",
                     "source": "frozen tier specification"})
        rows.append({"parameter": "P_%s" % tier, "value": ts.implied_dynamic_power_w,
                     "unit": "W", "source": "derived: kappa * f^3"})
    for name, value, unit in (("ue_tx_w", spec.ue_tx_w, "W"), ("mec_tx_w", spec.mec_tx_w, "W"),
                              ("helper_tx_w", spec.helper_tx_w, "W"),
                              ("ue_rx_w", spec.ue_rx_w, "W"),
                              ("helper_rx_w", spec.helper_rx_w, "W")):
        rows.append({"parameter": name, "value": value, "unit": unit,
                     "source": "frozen radio configuration"})
    rows.append({"parameter": "cycles_per_bit", "value": spec.cycles_per_bit,
                 "unit": "cycles/bit", "source": "frozen task data model"})
    doc = {
        "schema": "v2_energy_spec_v1",
        "model": spec.model,
        "primary_scope": spec.energy_scope,
        "include_rx_energy": bool(spec.include_rx_energy),
        "equations": {
            "cycles": "C_executed = executed_bytes * 8 * cycles_per_bit",
            "cpu_power": "P_cpu(tier) = kappa_tier * f_tier^3",
            "cpu_energy": "E_cpu = P_cpu(tier) * cpu_service_seconds  (== integral P dt; "
                          "equals kappa*C*f^2 exactly when the allocation is physical)",
            "tx_energy": "E_tx = P_tx(transmitter tier) * ACTIVE_transfer_service_seconds",
        },
        "boundaries": {
            "requester": "UE CPU + UE radio",
            "mobile": "requester + helper CPU + helper radio",
            "system": "mobile + MEC compute + MEC TX (frozen primary scope)",
        },
        "parameters": rows,
        "unmodeled_components": list(UNMODELED),
        "known_modelling_note":
            "the frozen rate table (MEC 10 MiB/s, UE/helper 1 MiB/s) is NOT the physical "
            "rate implied by f and cycles_per_bit (4.1667 MiB/s at the MEC), so the ledger "
            "uses the DURATION form of the same allocation the scheduler executed and "
            "reports the work form and the ratio as provenance.",
        "evidence_class": {
            "kappa_ue": "frozen config: chosen inside the verified band (kappa 1e-28..1e-26, "
                        "f 0.2..2.5 GHz); corroborated by ST-HO p_local = 1 W at 1 GHz",
            "kappa_helper": "frozen config cites Liu et al. DCN 9(6) 2023 Table 1 - the "
                            "source could NOT be re-verified in this session (publisher "
                            "403): treat as an unverified transplant pending a check",
            "kappa_mec": "same source, same caveat; implied dynamic power is 1 kW",
            "tx_powers": "frozen config cites Liu et al. DCN 2023 (unverified this session)",
            "rx_powers": "NOT MODELLED (include_rx_energy=false); ST-HO Table 1 gives "
                         "p_down = 0.2 W, so a RX sensitivity sweep is possible",
            "helper_v2v_relay_energy": "no accessible 2024-2026 primary source publishes a "
                                       "helper/V2V relay energy equation: the helper tier is "
                                       "a transparent assumption, not literature-pinned",
        },
    }
    (root / "V2_ENERGY_SPEC.json").write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    lines = ["# V2_ENERGY_SPEC - physical energy for the v2 system model", "",
             "Generated from the LIVE frozen configuration by "
             "`python3 -m spec.automotive_training.v2.benchmark_v2`.", "",
             "## Model", "",
             "```", "C_executed = executed_bytes * 8 * cycles_per_bit", "P_cpu(tier) = "
             "kappa_tier * f_tier^3", "E_cpu = P_cpu(tier) * cpu_service_seconds   "
             "(= integral P dt)", "E_tx  = P_tx(transmitter) * ACTIVE transfer service",
             "```", "",
             "Primary scope: **%s** (frozen). RX energy: %s."
             % (spec.energy_scope, "on" if spec.include_rx_energy else "OFF"), "",
             "## Parameters", "", "| parameter | value | unit | source |", "|---|---|---|---|"]
    for r in rows:
        lines.append("| %s | %s | %s | %s |" % (r["parameter"], r["value"], r["unit"],
                                                str(r["source"]).replace("\n", " ")))
    lines += ["", "## Boundaries", "",
              "* `E_requester` = UE CPU + UE radio",
              "* `E_mobile` = requester + helper CPU + helper radio",
              "* `E_system` = mobile + MEC compute + MEC TX  (primary)", "",
              "## Explicitly OUT of scope (declared, never reported as a measured zero)", ""]
    lines += ["* " + u for u in UNMODELED]
    lines += ["", "## Known modelling note", "",
              "The frozen rate table is not the physical rate implied by `f` and "
              "`cycles_per_bit`. The ledger therefore uses the duration form of the SAME "
              "allocation the scheduler executed, and reports the work form and the ratio "
              "as provenance rather than mixing two inconsistent allocations.", ""]
    (root / "V2_ENERGY_SPEC.md").write_text("\n".join(lines) + "\n")
    return doc


def write_obs_schema(root: Path) -> dict:
    from spec.automotive_training.v2.env import V2_CONTEXT_FIELDS
    from spec.automotive_training.v2.observation import (
        V1_FEATURE_DIM, V1_PACKED_DIM, V2_CONTEXT_DIM, V2_FEATURE_DIM, V2_OBS_VERSION,
        V2_PACKED_DIM, v2_feature_names,
    )

    doc = {
        "schema": "obs_schema_v2",
        "obs_version": V2_OBS_VERSION,
        "v1_obs_version": "automotive_mc_obs_v1",
        "v1_feature_dim": V1_FEATURE_DIM,
        "v2_context_dim": V2_CONTEXT_DIM,
        "v2_feature_dim": V2_FEATURE_DIM,
        "v1_packed_dim": V1_PACKED_DIM,
        "v2_packed_dim": V2_PACKED_DIM,
        "layout": "[v1 features | v2 context | forward neighbours | backward neighbours | mask]",
        "context_fields": list(V2_CONTEXT_FIELDS),
        "feature_names": list(v2_feature_names()),
        "dimensions_are_derived": True,
        "normalisation": ("the v2 context columns are bounded and use the identity entry "
                          "(mean 0, std 1); the 40 frozen v1 columns keep their "
                          "training-only statistics"),
        "checkpoint_contract": {
            "obs_version_required": V2_OBS_VERSION,
            "v1_checkpoints_are_incompatible": True,
            "migration": "no silent migration: a v1 checkpoint (79-wide) must be re-trained "
                         "or explicitly migrated; loading it into the v2 width raises",
        },
        "open_items": [
            "per-node epsilon: the context carries a single class epsilon; the per-node "
            "minimum/mean/max channels are NOT implemented",
            "estimate age: no age-of-estimate channel is implemented",
            "queue/load: no decision-time queue channel is implemented",
            "energy/budget/lambda conditioning: NOT implemented in the context",
            "the TF encoder consumption of this schema is NOT RUN in this environment "
            "(TensorFlow is not installed)",
        ],
    }
    (root / "OBS_SCHEMA_V2.json").write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    return doc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs", type=int, default=10)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--slots", type=int, default=10)
    ap.add_argument("--background", type=int, default=0)
    args = ap.parse_args()
    REPORTS.mkdir(parents=True, exist_ok=True)
    spec = frozen_energy_spec()
    run = benchmark(args.graphs, args.iters, args.slots, args.background)
    energy_doc = write_energy_spec(REPORTS, spec)
    obs_doc = write_obs_schema(REPORTS)
    run["artifacts"] = {"VP2_ENERGY_SPEC": "V2_ENERGY_SPEC.md/json",
                        "obs_schema": obs_doc["obs_version"]}
    run["parameters"] = {"energy": energy_doc["parameters"]}
    (REPORTS / "V2_THROUGHPUT.json").write_text(json.dumps(run, indent=2, sort_keys=True,
                                                           default=str) + "\n")
    print(json.dumps({k: v for k, v in run["phases"].items()}, indent=2, default=str))
    print("cpu_iteration_estimate_s", run["cpu_iteration_estimate_s"])
    print("written", REPORTS / "V2_THROUGHPUT.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
