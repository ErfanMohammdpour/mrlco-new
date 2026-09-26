#!/usr/bin/env python3
"""P2: `long latency-only PPO/meta-learning diagnostic` (1000 iterations).

Independent run directory (`runs/long_latency_v1/off/seed_0`), validation at
true-init, itr 0, every 50 and the final iteration, explicit checkpoints
(0/50/100/200/300/500/750/final, never overwritten), interim machine summaries
at 200/500, the pilot watchdog and provenance. `paper_result=false`.

This is a DIAGNOSTIC: no deadline, shield or energy-constrained claim may be
derived from it.

Usage (on Kish):
  python -m spec.pilot_long --root runs/long_latency_v1 --seed 0 --itr 1000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

SCHEMA = "long_latency_diagnostic_v1"
METHOD_ID = "margo_v0.4_long_latency_only_v1"
CONTRACT = {
    "reward_mode": "latency_only",
    "objective_mode": "log_only",
    "constraints": "off",
    "deadlines": "none",
    "mask_mode": "off",
    "obs_version": "v3",
    "energy_model": "physical_v1 (telemetry only)",
    "radio_model": "physical_v1",
    "timing": "legacy_frozen_rates",
    "decoder_order": "legacy_current",
    "entropy_coefficient": 0.0,
    "validation_interval": 50,
    # Parity with the 25/40-iteration Pilot A runs: same spawn-parallel env
    # executor, so the long trajectory is comparable to the short pilots.
    "parallel_env": True,
    "instability_watchdog": "inline (PilotInstabilityError)",
    "paper_result": False,
}
CHECKPOINT_ITERS = (0, 50, 100, 200, 300, 500, 750)
INTERIM_ITERS = (200, 500)
VALUE_ABS_MAX_LIMIT = 1e3
WARN_SHARE_FLOOR = 0.01
WARN_STREAK = 5


# --------------------------------------------------------------------------- #
# pure helpers (unit-tested without TensorFlow)
# --------------------------------------------------------------------------- #
def checkpoint_plan(last_itr: int) -> tuple[int, ...]:
    """Explicit checkpoint iterations plus the final one."""
    return tuple(sorted(set(CHECKPOINT_ITERS) | {last_itr}))


def assert_fresh_run_dir(path: Path) -> None:
    """Refuse to touch an existing run directory (never overwrite)."""
    if path.exists() and any(path.iterdir()):
        raise RuntimeError(
            "refusing to reuse a non-empty run directory: %s (use a new --root)"
            % path
        )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest(root: Path) -> list[dict]:
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            rows.append({
                "relative_path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })
    return rows


def configure_obs_env() -> dict:
    """The contract is obs v3 + mask off + constraints off.

    The encoder/policy modules read these env vars at import time, so they must be
    set before the stack is built; without v3 the encoder raises
    `resource_vec only valid for obs v2/v3`.
    """
    os.environ["MARGO_OBS_VERSION"] = "v3"
    os.environ["MARGO_MASK_MODE"] = "off"
    os.environ["MARGO_CONSTRAINTS"] = "off"
    return {
        "MARGO_OBS_VERSION": os.environ["MARGO_OBS_VERSION"],
        "MARGO_MASK_MODE": os.environ["MARGO_MASK_MODE"],
        "MARGO_CONSTRAINTS": os.environ["MARGO_CONSTRAINTS"],
    }


def classification(rows: list[dict], *, stopped: bool = False) -> dict:
    """`rows` = per-validation dicts with k0/k3/gaps (sorted by iteration)."""
    import math

    finite = all(
        all(
            isinstance(v, (int, float)) and math.isfinite(float(v))
            for v in (r.get("k0"), r.get("k3"), r.get("gap_to_all_mec"), r.get("gap_to_greedy"))
            if v is not None
        )
        for r in rows
    )
    if stopped:
        return {
            "verdict": "STOPPED_UNSTABLE",
            "improved_points": 0,
            "detail": "inline watchdog aborted the run on a non-finite metric or "
                      "value_abs_max >= limit; no conclusion is drawn",
        }
    if len(rows) < 2 or not finite:
        return {"verdict": "INSUFFICIENT_OR_NONFINITE", "improved_points": 0}
    init = rows[0]
    improved = [
        r for r in rows[1:]
        if r["k3"] is not None and init["k3"] is not None and r["k3"] < init["k3"]
    ]
    gap_improved = [
        r for r in rows[1:]
        if r["gap_to_all_mec"] is not None and init["gap_to_all_mec"] is not None
        and r["gap_to_all_mec"] < init["gap_to_all_mec"]
    ]
    best = min(rows, key=lambda r: r["k3"] if r["k3"] is not None else float("inf"))
    return {
        "verdict": "PASS_LATENCY_LEARNING" if len(improved) >= 2 and len(gap_improved) >= 2
        else "NO_CONSISTENT_IMPROVEMENT",
        "improved_validation_points": len(improved),
        "improved_gap_points": len(gap_improved),
        "best_itr": best.get("itr"),
        "best_k3": best.get("k3"),
        "init_k3": init.get("k3"),
        "last_gap_to_all_mec": rows[-1].get("gap_to_all_mec"),
        "last_gap_to_greedy": rows[-1].get("gap_to_greedy"),
    }


def watchdog_flags(series: dict, *, value_limit: float = VALUE_ABS_MAX_LIMIT) -> dict:
    """Stop/warn conditions from the per-iteration series (lists, oldest first)."""
    def recent(key, n=WARN_STREAK):
        return list(series.get(key, []))[-n:]

    import math

    stop = {
        "non_finite": any(
            any(
                not (isinstance(v, (int, float)) and math.isfinite(float(v)))
                for v in series.get(k, [])
            )
            for k in series
        ),
        "value_abs_max_over_limit": bool(series.get("critic/value_abs_max"))
        and max(series["critic/value_abs_max"]) >= value_limit,
    }
    warn = {
        "local_below_floor": len(recent("action_fraction/local")) == WARN_STREAK
        and all(v < WARN_SHARE_FLOOR for v in recent("action_fraction/local")),
        "v2v_below_floor": len(recent("action_fraction/v2v")) == WARN_STREAK
        and all(v < WARN_SHARE_FLOOR for v in recent("action_fraction/v2v")),
        "entropy_near_zero": len(recent("policy/entropy_valid")) == WARN_STREAK
        and all(v < 0.01 for v in recent("policy/entropy_valid")),
    }
    return {"stop": stop, "warn": warn, "stop_any": any(stop.values())}


def _run_dir(root: str, seed: int) -> Path:
    return Path(root) / "off" / ("seed_%d" % int(seed))


def _validation_rows(csv_path: Path) -> list[dict]:
    """Extract the per-validation trajectory from the trainer CSV."""
    import csv

    rows = list(csv.DictReader(open(csv_path)))
    out = []
    for row in rows:
        k0 = row.get("validation_query_mean_latency_k0")
        k3 = row.get("validation_query_mean_latency_k3")
        if k0 in (None, "") or k3 in (None, ""):
            continue
        def f(name):
            v = row.get(name)
            try:
                return float(v)
            except (TypeError, ValueError):
                return None
        out.append({
            "itr": int(float(row.get("Itr", -1))),
            "k0": f("validation_query_mean_latency_k0"),
            "k3": f("validation_query_mean_latency_k3"),
            "all_mec": f("validation/validation_all_mec_latency"),
            "greedy": f("validation/validation_greedy_latency"),
            "gap_to_all_mec": f("validation/validation_gap_to_all_mec"),
            "gap_to_greedy": f("validation/validation_gap_to_greedy"),
            "action_fraction_local": f("action_fraction/local"),
            "action_fraction_mec": f("action_fraction/mec"),
            "action_fraction_v2v": f("action_fraction/v2v"),
            "entropy": f("policy/entropy_valid"),
            "objective_J": f("objective/J"),
            "objective_feasible": f("objective/feasible"),
        })
    return out


def interim_summary(rows: list[dict], true_init: dict, itr: int) -> dict:
    return {
        "schema": "long_latency_interim_v1",
        "itr": int(itr),
        "true_init": true_init,
        "validation_trajectory": rows,
        "better_than_true_init": bool(
            rows and rows[-1]["k3"] is not None and true_init.get("k3") is not None
            and rows[-1]["k3"] < true_init["k3"]
        ),
        "gap_reduction_to_all_mec": (
            None if not rows or true_init.get("gap_to_all_mec") is None
            or rows[-1]["gap_to_all_mec"] is None
            else true_init["gap_to_all_mec"] - rows[-1]["gap_to_all_mec"]
        ),
        "k3_better_than_k0": bool(rows and rows[-1]["k3"] is not None
                                  and rows[-1]["k0"] is not None
                                  and rows[-1]["k3"] < rows[-1]["k0"]),
        "classification": classification(rows),
    }


# --------------------------------------------------------------------------- #
# driver (TensorFlow)
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="runs/long_latency_v1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--itr", type=int, default=1000)
    parser.add_argument("--i-allow-gpu", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="contract/plan only")
    args = parser.parse_args(argv)

    run_dir = _run_dir(args.root, args.seed)
    last_itr = int(args.itr) - 1
    plan = checkpoint_plan(last_itr)
    if args.dry_run:
        print(json.dumps({
            "schema": SCHEMA, "contract": CONTRACT, "run_dir": str(run_dir),
            "iterations": int(args.itr), "last_itr": last_itr,
            "checkpoint_iters": list(plan), "interim_iters": list(INTERIM_ITERS),
            "validation_iters": sorted(set(range(0, int(args.itr), 50)) | {last_itr}),
        }, indent=2, sort_keys=True))
        return 0

    assert_fresh_run_dir(run_dir)
    obs_env = configure_obs_env()

    import numpy as np
    import tensorflow as tf
    from utils import logger
    from meta_trainer import PilotInstabilityError, build_frozen_primary_stack
    from env.mec_offloaing_envs.scheduler.primary_config import (
        resolved_primary_scheduler_config,
    )
    from spec.phase4_train_driver import require_gpu_permission

    require_gpu_permission(bool(args.i_allow_gpu))
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    logger.configure(dir=str(run_dir / "logs"), format_strs=["stdout", "log", "csv"])

    trainer, algo = build_frozen_primary_stack(
        seed=int(args.seed), n_itr=int(args.itr), ckpt_dir=str(run_dir / "ckpt"),
        audit_writer=None, print_action_choices=False, parallel=True,
        reward_mode="latency_only", learning_mode="publication", vocab_size=3,
        use_energy=True, constraints=None, constraint_dual_lr=None,
        objective_mode="log_only",
        objective_spec={"latency_ref": "all_ue", "energy_budget_j": 1e12},
        scheduler_config=resolved_primary_scheduler_config(),
        strict_scheduler_config=True,
    )
    trainer.pilot_checkpoint_iters = set(plan)
    trainer.pilot_interim_iters = tuple(INTERIM_ITERS)
    # Inline abort: stop at the first non-finite metric or value_abs_max over limit.
    trainer.pilot_watchdog = True
    trainer.pilot_value_abs_max_limit = VALUE_ABS_MAX_LIMIT
    held = trainer.held_out_evaluator
    initial = {}
    instability = None
    final_row = None

    def _flat(metrics):
        keep = {
            "query_mean_latency": "k0_or_k3",
            "query_all_mec_latency": "all_mec",
            "query_greedy_latency": "greedy",
            "query_entropy_valid": "entropy",
            "query_action_fraction/local": "action_fraction_local",
            "query_action_fraction/mec": "action_fraction_mec",
            "query_action_fraction/v2v": "action_fraction_v2v",
            "query_co_location_rate": "co_location_rate",
            "query_cross_location_edges": "cross_location_edges",
            "query_utilization_mean": "utilization_mean",
            "query_utilization_max": "utilization_max",
        }
        return {out: metrics.get(key) for key, out in keep.items()}

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        algo.sync_task_policies_from_core()
        # true-init validation BEFORE any update
        k0 = held.evaluate_all(k_steps=0, sess=sess)
        k3 = held.evaluate_all(k_steps=3, sess=sess)
        initial = {"itr": -1, "true_init": True, **_flat(k0), "k0": k0.get("query_mean_latency"),
                   "k3": k3.get("query_mean_latency")}
        initial["gap_to_all_mec"] = (
            None if initial.get("k3") is None or initial.get("all_mec") is None
            else initial["k3"] - initial["all_mec"]
        )
        initial["gap_to_greedy"] = (
            None if initial.get("k3") is None or initial.get("greedy") is None
            else initial["k3"] - initial["greedy"]
        )
        (run_dir / "true_init_validation.json").write_text(
            json.dumps(initial, indent=2, sort_keys=True) + "\n"
        )

        def _interim(itr, _trainer):
            rows = _validation_rows(run_dir / "logs" / "progress.csv")
            (run_dir / ("interim_itr%d.json" % itr)).write_text(
                json.dumps(interim_summary(rows, initial, itr), indent=2, sort_keys=True) + "\n"
            )

        trainer.pilot_interim_hook = _interim
        try:
            trainer.train()
        except PilotInstabilityError as exc:
            # The inline watchdog fired: no final adaptation, no final checkpoint.
            instability = str(exc)
            print("WATCHDOG ABORT: %s" % instability)

        if instability is None:
            final = held.evaluate_all(k_steps=3, sess=sess)
            final_k0 = held.evaluate_all(k_steps=0, sess=sess)
            final_row = {
                "itr": last_itr,
                "final": True,
                "k0": final_k0.get("query_mean_latency"),
                "k3": final.get("query_mean_latency"),
                "all_mec": final.get("query_all_mec_latency"),
                "greedy": final.get("query_greedy_latency"),
            }
            final_row["gap_to_all_mec"] = (
                None if final_row["k3"] is None or final_row["all_mec"] is None
                else final_row["k3"] - final_row["all_mec"]
            )
            final_row["gap_to_greedy"] = (
                None if final_row["k3"] is None or final_row["greedy"] is None
                else final_row["k3"] - final_row["greedy"]
            )
            (run_dir / "final_validation.json").write_text(
                json.dumps({"itr": last_itr, "final": _flat(final), **final_row},
                           indent=2, sort_keys=True) + "\n"
            )
            final_ckpt = run_dir / "ckpt" / "meta_model_final.ckpt"
            if not final_ckpt.exists():
                trainer.policy.core_policy.save_variables(save_path=str(final_ckpt))

    rows = _validation_rows(run_dir / "logs" / "progress.csv")
    trajectory = rows + ([final_row] if final_row else [])
    # the final iteration belongs in the verdict
    series_keys = ("action_fraction/local", "action_fraction/mec", "action_fraction/v2v",
                   "policy/entropy_valid", "critic/value_abs_max", "policy/approx_kl",
                   "policy/clip_fraction", "policy/grad_norm", "collapse/flag")
    import csv as _csv
    raw_rows = list(_csv.DictReader(open(run_dir / "logs" / "progress.csv")))
    series = {}
    for key in series_keys:
        series[key] = []
        for row in raw_rows:
            try:
                series[key].append(float(row[key]))
            except (TypeError, ValueError):
                pass
    evidence = {
        "schema": SCHEMA, "method_id": METHOD_ID, "contract": CONTRACT,
        "run_dir": str(run_dir), "iterations": int(args.itr),
        "training_code_sha": os.environ.get("MARGO_TRAINING_CODE_SHA", ""),
        "evaluation_code_sha": os.environ.get("MARGO_EVAL_CODE_SHA", ""),
        "true_init": initial, "validation_trajectory": trajectory,
        "classification": classification([initial] + trajectory, stopped=instability is not None),
        "instability": instability,
        "obs_env": obs_env,
        "watchdog": watchdog_flags(series),
        "checkpoint_selection_metric": (
            "objective_contract_v1: validation/objective_discounted_return (k3) - the "
            "discounted return PPO optimises - via Trainer.best_val_objective; the "
            "chosen value is recorded in ckpt/meta_model_best_val.metric.json. "
            "objective_mode=log_only does NOT change the saved best checkpoint "
            "(lexicographic selection is off). The pre-contract undiscounted sum is "
            "still logged as validation/objective_legacy_undiscounted_sum_*."
        ),
        "paper_result": False,
    }
    (run_dir / "long_run_evidence.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n"
    )
    (run_dir / "manifest.json").write_text(
        json.dumps({"schema": "long_latency_manifest_v1", "files": manifest(run_dir)},
                   indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(evidence["classification"], indent=1, sort_keys=True))
    return 3 if instability else 0


if __name__ == "__main__":
    raise SystemExit(main())
