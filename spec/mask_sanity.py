"""⑥b mask sanity runs: dedicated trainer entrypoint for v3 obs.

Its own run directories, its own preflight, and its own post-run validation.
The legacy `par500` target is left alone: it writes an old v0.1 run dir, does not
separate off/static, and would mix artifacts.

    python -m spec.mask_sanity --mode off    --itr 1   --seed 0 --i-allow-gpu
    python -m spec.mask_sanity --mode static --itr 500 --seed 0 --i-allow-gpu
    python -m spec.mask_sanity --mode static --itr 1   --preflight-only

Run directories:
    runs/mask_sanity_v3/<mode>/seed_<seed>/

Preflight (hard assertions, all fatal):
    MARGO_OBS_VERSION == v3            MARGO_MASK_MODE == <mode>
    constraints disabled               n_itr in {1, 500}
    git tree clean, SHA recorded       deadline count in the training env == 0

The preflight runs twice: once before anything heavy (env/config/git), and again
against the LIVE stack, where the deadline count is counted over the actual
training graphs and the policy's mask mode and placeholder are checked.

After training the progress CSV is validated: all seven metric columns must be
present, every row must be finite, the five control rates must be exactly zero
and `critic/value_abs_max` must stay below 1e3 -- the agreed stop conditions.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

MODES = ("off", "static")
ALLOWED_ITERS = (1, 500)
METHOD_ID = "margo_v0.4_mask_sanity_v3"
DEFAULT_RUNS_ROOT = Path("runs") / "mask_sanity_v3"
VALUE_ABS_MAX_LIMIT = 1e3

METRIC_KEYS = (
    "mask/active_rate",
    "mask/forced_rate",
    "mask/all_invalid_rate",
    "policy/invalid_action_rate",
    "policy/argmax_masked_rate",
    "policy/entropy_valid",
    "critic/value_abs_max",
)
ZERO_RATE_KEYS = METRIC_KEYS[:5]


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=MODES)
    parser.add_argument("--itr", type=int, required=True, choices=ALLOWED_ITERS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--runs-root", default=str(DEFAULT_RUNS_ROOT))
    parser.add_argument("--i-allow-gpu", action="store_true")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="run the cheap preflight and exit without building the stack",
    )
    parser.add_argument(
        "--skip-git-check",
        action="store_true",
        help="diagnostics only: do not require a clean tree",
    )
    return parser.parse_args(argv)


def run_dir(mode, seed, runs_root):
    return Path(runs_root) / str(mode) / ("seed_%d" % int(seed))


def git_state():
    """(sha, dirty_files) -- tolerates a missing git binary."""
    def _git(*args):
        proc = subprocess.run(["git", *args], capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            return None
        return proc.stdout.strip()

    sha = _git("rev-parse", "HEAD")
    porcelain = _git("status", "--porcelain")
    dirty = [] if porcelain is None else [l for l in porcelain.splitlines() if l.strip()]
    return sha, dirty


def configure_env(mode):
    """Env vars must be set BEFORE the policy/encoder modules are imported."""
    os.environ["MARGO_OBS_VERSION"] = "v3"
    os.environ["MARGO_MASK_MODE"] = str(mode)
    # constraints are out of scope for the sanity run: make it explicit rather
    # than inheriting whatever the shell happens to export
    os.environ["MARGO_CONSTRAINTS"] = "off"


def static_preflight(mode, itr, seed, runs_root, skip_git_check):
    failures = []

    def require(name, ok, detail=None):
        if not ok:
            failures.append({"check": name, "detail": detail})

    require("mode_supported", mode in MODES, mode)
    require("itr_supported", int(itr) in ALLOWED_ITERS, itr)

    from env.mec_offloaing_envs.scheduler import encoder_obs as eo
    from env.mec_offloaing_envs.scheduler import masking

    require("obs_version_v3", eo.OBS_VERSION == "v3", eo.OBS_VERSION)
    require("obs_dim_v3", int(eo.PACKED_DIM) == 70, int(eo.PACKED_DIM))
    require("mask_mode_matches", masking.resolve_mask_mode() == mode, masking.resolve_mask_mode())

    from spec.constraints_config import constraints_from_env

    constraint_spec, dual_lr = constraints_from_env()
    require("constraints_disabled", constraint_spec is None, dual_lr)

    sha, dirty = git_state()
    if not skip_git_check:
        require("git_tree_clean", not dirty, dirty[:5])
    require("git_sha_recorded", bool(sha), sha)

    rd = run_dir(mode, seed, runs_root)
    return {
        "method_id": METHOD_ID,
        "mode": mode,
        "itr": int(itr),
        "seed": int(seed),
        "run_dir": str(rd),
        "obs_version": eo.OBS_VERSION,
        "obs_dim": int(eo.PACKED_DIM),
        "mask_mode": masking.resolve_mask_mode(),
        "constraints": "off",
        "git_sha": sha,
        "git_dirty": bool(dirty),
        "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "failures": failures,
    }


def verify_live_stack(trainer, mode):
    """Assertions that need the actual training environment."""
    failures = []

    def require(name, ok, detail=None):
        if not ok:
            failures.append({"check": name, "detail": detail})

    from env.mec_offloaing_envs.scheduler import encoder_obs as eo
    from env.mec_offloaing_envs.scheduler.adapter import to_canonical_dag
    from env.mec_offloaing_envs.scheduler.static_bounds import deadline_vector

    require("live_obs_version_v3", eo.OBS_VERSION == "v3", eo.OBS_VERSION)

    core = getattr(trainer.policy, "core_policy", None)
    if core is not None:
        require("live_mask_mode", getattr(core, "mask_mode", None) == mode,
                getattr(core, "mask_mode", None))
        require(
            "live_placeholder_matches_mode",
            (core.feasible_mask is not None) == (mode != "off"),
            {"has_placeholder": core.feasible_mask is not None},
        )

    env = trainer.env
    n_tasks = 0
    n_deadline = 0
    for graphs in getattr(env, "task_graphs_batchs", []):
        for tg in graphs:
            dag = to_canonical_dag(tg)
            order = [int(tid) for tid in tg.prioritize_sequence]
            for deadline, dtype in deadline_vector(dag, order):
                n_tasks += 1
                if deadline is not None or str(dtype) != "none":
                    n_deadline += 1
    require("training_graphs_seen", n_tasks > 0, n_tasks)
    require("deadline_count_zero", n_deadline == 0, {"tasks": n_tasks, "with_deadline": n_deadline})
    return {"failures": failures, "tasks": n_tasks, "deadlines": n_deadline}


def _train_masked(args, payload, rd):
    """Mirror of phase4_train_driver._train with the live preflight inserted."""
    import tensorflow as tf
    from utils import logger
    from meta_trainer import build_frozen_primary_stack
    from spec.phase4_train_driver import require_gpu_permission
    from spec.train_audit import TrainAuditWriter

    require_gpu_permission(bool(args.i_allow_gpu))

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    logger.configure(dir=str(rd / "logs"), format_strs=["stdout", "log", "csv"])
    writer = TrainAuditWriter(rd)
    trainer, algo = build_frozen_primary_stack(
        seed=int(args.seed),
        n_itr=int(args.itr),
        ckpt_dir=str(rd / "ckpt"),
        audit_writer=writer,
        print_action_choices=False,
        parallel=True,
        reward_mode="publication",
        learning_mode="publication",
        vocab_size=3,
        use_energy=True,
        constraints=None,
        constraint_dual_lr=None,
    )

    live = verify_live_stack(trainer, args.mode)
    payload["live_preflight"] = live
    if live["failures"]:
        _write_payload(rd, payload)
        raise SystemExit(
            "live preflight failed: %s" % json.dumps(live["failures"], sort_keys=True)
        )

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        algo.sync_task_policies_from_core()
        trainer.train()
    return payload


def validate_progress_csv(rd, itr):
    """All seven columns present, every row finite, stop conditions respected."""
    import csv

    csv_path = Path(rd) / "logs" / "progress.csv"
    failures = []
    if not csv_path.is_file():
        return {"failures": [{"check": "progress_csv_exists", "detail": str(csv_path)}],
                "rows": 0}

    with csv_path.open() as handle:
        reader = csv.DictReader(handle)
        header = list(reader.fieldnames or [])
        rows = list(reader)

    if len(set(header)) != len(header):
        failures.append({"check": "csv_header_unique", "detail": header})
    if any(not str(k).strip() for k in header):
        failures.append({"check": "csv_header_no_blank_keys", "detail": header})
    for key in METRIC_KEYS:
        if key not in header:
            failures.append({"check": "metric_column_present", "detail": key})
    if len(rows) < int(itr):
        failures.append({"check": "row_count", "detail": {"rows": len(rows), "itr": int(itr)}})

    worst = {"entropy_min": None, "value_abs_max": None}
    for i, row in enumerate(rows):
        for key in METRIC_KEYS:
            if key not in row or row[key] in ("", None):
                failures.append({"check": "metric_value_present", "detail": {"row": i, "key": key}})
                continue
            try:
                value = float(row[key])
            except (TypeError, ValueError):
                failures.append({"check": "metric_value_numeric", "detail": {"row": i, "key": key, "value": row[key]}})
                continue
            if value != value or value in (float("inf"), float("-inf")):
                failures.append({"check": "metric_value_finite", "detail": {"row": i, "key": key, "value": value}})
            if key in ZERO_RATE_KEYS and value != 0.0:
                failures.append({"check": "control_rate_zero", "detail": {"row": i, "key": key, "value": value}})
            if key == "policy/entropy_valid":
                prev = worst["entropy_min"]
                worst["entropy_min"] = value if prev is None else min(prev, value)
            if key == "critic/value_abs_max":
                prev = worst["value_abs_max"]
                worst["value_abs_max"] = value if prev is None else max(prev, value)

    if worst["value_abs_max"] is not None and worst["value_abs_max"] >= VALUE_ABS_MAX_LIMIT:
        failures.append(
            {"check": "value_abs_max_below_limit", "detail": worst["value_abs_max"]}
        )
    return {"failures": failures, "rows": len(rows), "header": header, **worst}


def _write_payload(rd, payload):
    rd.mkdir(parents=True, exist_ok=True)
    (rd / "config.resolved.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    configure_env(args.mode)
    rd = run_dir(args.mode, args.seed, args.runs_root)

    payload = static_preflight(args.mode, args.itr, args.seed, args.runs_root, args.skip_git_check)
    _write_payload(rd, payload)
    if payload["failures"]:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 2
    if args.preflight_only:
        payload["preflight_only"] = True
        _write_payload(rd, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    payload["started_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    _train_masked(args, payload, rd)
    payload["finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    payload["csv_validation"] = validate_progress_csv(rd, args.itr)
    payload["gpu_finished"] = True
    payload["paper_result"] = False
    payload["failures"] = list(payload["csv_validation"]["failures"])
    _write_payload(rd, payload)
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 1 if payload["failures"] else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
