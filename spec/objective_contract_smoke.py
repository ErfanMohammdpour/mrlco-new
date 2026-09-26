#!/usr/bin/env python3
"""CPU smoke: objective_contract_v1 must be the logged AND selecting criterion.

Runs ONE outer iteration (mask off, constraints off, latency_only, log_only) and
then asserts, from the artifacts the run actually wrote:

  * the CSV carries the contract columns,
  * the scalar that selected `meta_model_best_val.ckpt` equals
    `validation/objective_discounted_return_k3` exactly,
  * the sidecar `ckpt/meta_model_best_val.metric.json` agrees with that row,
  * the legacy undiscounted sum is a *different* number and is not used.

Usage (CPU container, no GPU needed):
  python -m spec.objective_contract_smoke --run-dir runs/objective_contract_smoke/seed_0
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

SCHEMA = "objective_contract_smoke_v1"


def configure_obs_env() -> dict:
    os.environ["MARGO_OBS_VERSION"] = "v3"
    os.environ["MARGO_MASK_MODE"] = "off"
    os.environ["MARGO_CONSTRAINTS"] = "off"
    return {
        "MARGO_OBS_VERSION": os.environ["MARGO_OBS_VERSION"],
        "MARGO_MASK_MODE": os.environ["MARGO_MASK_MODE"],
        "MARGO_CONSTRAINTS": os.environ["MARGO_CONSTRAINTS"],
    }


def assert_fresh_run_dir(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise RuntimeError("refusing to reuse a non-empty run dir: %s" % path)


def _row_with_objective(csv_path: Path) -> dict:
    with open(csv_path) as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        if row.get("validation/objective_discounted_return_k3") not in (None, ""):
            return row
    raise AssertionError("no validation row with the contract objective in %s" % csv_path)


def verify(run_dir: Path) -> dict:
    """Pure checks over the written artifacts; returns the evidence payload."""
    from spec.objective_contract import (
        METRIC_PRIMARY,
        SCHEMA as contract_schema,
        SELECTION_METRIC_DEFAULT,
    )

    row = _row_with_objective(run_dir / "logs" / "progress.csv")
    objective_k3 = float(row["validation/objective_discounted_return_k3"])
    objective_k0 = float(row["validation/objective_discounted_return_k0"])
    legacy_k3 = float(row["validation/objective_legacy_undiscounted_sum_k3"])
    latency_k3 = float(row["validation_query_mean_latency_k3"])
    scalar = float(row["checkpoint_selection_scalar"])

    sidecar_path = run_dir / "ckpt" / "meta_model_best_val.metric.json"
    sidecar = json.loads(sidecar_path.read_text()) if sidecar_path.is_file() else None

    checks = {
        "contract_schema_in_csv": row.get("objective_contract/schema") == contract_schema,
        "selection_metric_is_primary": (
            row.get("checkpoint_selection_metric_name") == SELECTION_METRIC_DEFAULT
            and SELECTION_METRIC_DEFAULT == METRIC_PRIMARY
        ),
        "selection_scalar_equals_objective_k3": abs(scalar - objective_k3) <= 1e-12,
        "selection_column_is_the_objective": (
            row.get("checkpoint_selection_metric")
            == "validation/objective_discounted_return"
        ),
        "legacy_scalar_is_a_different_number": abs(legacy_k3 - objective_k3) > 1e-9,
        "best_val_checkpoint_selected": float(row.get("checkpoint_is_best_val", 0.0)) == 1.0,
        "sidecar_written": sidecar is not None,
        "sidecar_names_the_contract": bool(sidecar) and sidecar.get("contract") == contract_schema,
        "sidecar_metric_is_primary": bool(sidecar)
        and sidecar.get("metric_name") == SELECTION_METRIC_DEFAULT,
        "sidecar_value_matches_csv": bool(sidecar)
        and abs(float(sidecar.get("value", float("nan"))) - objective_k3) <= 1e-12,
        "latency_companion_present": latency_k3 > 0.0,
    }
    payload = {
        "schema": SCHEMA,
        "run_dir": str(run_dir),
        "objective_k0": objective_k0,
        "objective_k3": objective_k3,
        "legacy_undiscounted_sum_k3": legacy_k3,
        "latency_k3_seconds": latency_k3,
        "selection_scalar": scalar,
        "checks": checks,
        "passed": all(checks.values()),
        "sidecar": sidecar,
    }
    if not payload["passed"]:
        payload["failures"] = sorted(k for k, v in checks.items() if not v)
    return payload


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="runs/objective_contract_smoke/seed_0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--itr", type=int, default=1)
    parser.add_argument("--i-allow-gpu", action="store_true")
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    assert_fresh_run_dir(run_dir)
    obs_env = configure_obs_env()

    import tensorflow as tf
    from meta_trainer import build_frozen_primary_stack
    from env.mec_offloaing_envs.scheduler.primary_config import (
        resolved_primary_scheduler_config,
    )
    from spec.phase4_train_driver import require_gpu_permission
    from utils import logger

    require_gpu_permission(bool(args.i_allow_gpu))
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    logger.configure(dir=str(run_dir / "logs"), format_strs=["stdout", "log", "csv"])

    trainer, _algo = build_frozen_primary_stack(
        seed=int(args.seed), n_itr=int(args.itr), ckpt_dir=str(run_dir / "ckpt"),
        audit_writer=None, print_action_choices=False, parallel=False,
        reward_mode="latency_only", learning_mode="publication", vocab_size=3,
        use_energy=True, constraints=None, constraint_dual_lr=None,
        objective_mode="log_only",
        objective_spec={"latency_ref": "all_ue", "energy_budget_j": 1e12},
        scheduler_config=resolved_primary_scheduler_config(),
        strict_scheduler_config=True,
    )
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        _algo.sync_task_policies_from_core()
        trainer.train()

    payload = verify(run_dir)
    payload["obs_env"] = obs_env
    payload["training_code_sha"] = os.environ.get("MARGO_TRAINING_CODE_SHA", "")
    out = run_dir / "objective_contract_smoke.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: payload[k] for k in ("passed", "objective_k3", "legacy_undiscounted_sum_k3",
                                              "selection_scalar", "checks")},
                     indent=1, sort_keys=True))
    print("wrote", out)
    if not payload["passed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
