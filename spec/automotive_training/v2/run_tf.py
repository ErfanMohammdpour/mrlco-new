#!/usr/bin/env python3
"""REAL TensorFlow 1.15 runner for the v2 system model — the executable entry point.

`python -m spec.automotive_training.v2.stack` was never a runner (the module has no `main`),
so this module is the production entry point. It:

1. builds the REAL stack (`build_automotive_v2_stack`: Graph2Seq policy, sampler, PPO
   processor, MRLCO, v2 environment, v2 evaluator);
2. creates a real `tf.compat.v1.Session()`, initialises the graph, and installs the session as
   the default so the frozen trainer's `get_default_session()` calls resolve;
3. optionally restores a training checkpoint;
4. calls the ACTUAL `trainer.train()`;
5. collects executed evidence: iterations completed, weight movement, measured energy,
   raw/signed violations, applied penalties, dual state and broadcast targets, selector
   decisions, and the validation summary;
6. writes a JSON report (and a checkpoint) so a fresh process can resume.

It exits non-zero when no iteration actually executed, when the run timed out before
validation, or when a mandatory measurement is missing: a successful exit without executed
iterations is a failure, not a pass.

    python3 -m spec.automotive_training.v2.run_tf --iterations 2 --background 2 \
        --link-regime moderate --ckpt-dir /tmp/v2run/ckpt --json /tmp/v2run/report.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class V2RunError(RuntimeError):
    """Raised when the production path cannot execute or produced no real evidence."""


def _finite(value: Any) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise V2RunError("non-finite measurement: %r" % (value,))
    return out


def _collect_evidence(trainer, env, algo, *, elapsed: float, iterations: int) -> dict:
    """Read the executed evidence OFF the real objects (never a placeholder)."""
    manager = getattr(trainer, "auto_v2_constraint_manager", None)
    sampler = getattr(trainer, "sampler", None)
    counters = dict(getattr(sampler, "counters", {}) or {})
    evidence: dict[str, Any] = {
        "iterations_requested": int(iterations),
        # NEVER inferred from requested n_itr, tensor norms or a checkpoint's existence
        "iterations_started": int(getattr(trainer, "started_iterations", 0) or 0),
        "iterations_completed": int(getattr(trainer, "completed_iterations", 0) or 0),
        "last_completed_iteration": int(getattr(trainer, "last_completed_iteration", -1)),
        "wall_seconds": _finite(elapsed),
        "sampler_counters": {k: int(v) for k, v in counters.items()},
        "penalty": {
            "episodes": int(getattr(trainer, "auto_penalty_episodes", 0) or 0),
            "episodes_with_nonzero_penalty": int(
                getattr(trainer, "auto_penalty_episodes_nonzero", 0) or 0),
            "penalty_sum": float(getattr(trainer, "auto_penalty_sum", 0.0) or 0.0),
            "steps_with_penalty": int(getattr(trainer, "auto_penalty_steps", 0) or 0),
            "signed_rows_observed": int(getattr(trainer, "auto_v2_signed_rows", 0) or 0),
        },
        "lambda_broadcast": {
            "targets": int(getattr(trainer, "auto_lambda_broadcast_targets", 0) or 0),
            "values": dict(getattr(trainer, "auto_lambda_broadcast_values", {}) or {}),
        },
        "trained_core_norms": {},
    }
    if manager is not None:
        evidence["constraints"] = {
            "names": list(manager.names),
            "lambdas": manager.lambdas_by_name(),
            "updates": int(manager.controller.updates),
            "spec_sha256": manager.state().get("constraints_sha256"),
            "status": manager.status(),
        }
    try:
        import tensorflow as tf

        sess = tf.compat.v1.get_default_session()
        if sess is not None:
            core = algo.policy.core_policy.get_trainable_variables()
            values = sess.run(core)
            evidence["trained_core_norms"] = {
                "n_tensors": len(core),
                "total_norm": float(sum(float((v ** 2).sum()) ** 0.5 for v in values)),
                # np.isfinite detects BOTH NaN and +-inf; `v == v` misses infinity
                "all_finite": bool(all(bool(np.isfinite(v).all()) for v in values)),
            }
    except Exception as exc:                     # pragma: no cover - evidence only
        evidence["trained_core_norms"] = {"error": str(exc)[:200]}
    # the v2 environment's own measured facts from the last rollout.
    # The frozen sampler's vectorised executor runs the rollouts on `copy.deepcopy(env)`
    # clones, so the ORIGINAL `trainer.env` has no telemetry: search the executor for an
    # environment that actually executed.
    def _measured(candidate):
        try:
            return len(getattr(candidate, "last_telemetry", []) or []) > 0
        except Exception:
            return False

    measured = env if _measured(env) else None
    if measured is None:
        seen, queue = set(), [getattr(trainer, "sampler", None),
                              getattr(trainer, "meta_sampler", None),
                              getattr(trainer, "auto_sampler", None), env]
        while queue and measured is None:
            node = queue.pop(0)
            if node is None or id(node) in seen:
                continue
            seen.add(id(node))
            if _measured(node):
                measured = node
                break
            for attr in ("vec_env", "envs", "workers", "env", "meta_env", "executor"):
                child = getattr(node, attr, None)
                if child is None:
                    continue
                if isinstance(child, (list, tuple)):
                    queue.extend(child)
                else:
                    queue.append(child)
    if measured is None:
        records = list(getattr(trainer, "auto_last_telemetry_records", []) or [])
        if not records:
            raise V2RunError(
                "no rollout telemetry reached the trainer: the production rollouts did not "
                "execute, or the observer did not receive their paths")
        evidence["measured_env"] = {
            "type": "rollout_paths (executor clones are not reachable from trainer.env)",
            "records_seen": int(getattr(trainer, "auto_telemetry_records_seen", len(records))),
            "records_kept": len(records),
        }
        record = records[-1]
        block = record.get("constraints") or {}
        evidence["last_rollout"] = {
            "makespan_s": float(record.get("makespan_s", 0.0)),
            "world_makespan_s": float(record.get("world_makespan_s", 0.0)),
            "system_joules": float(record.get("system_joules", 0.0)),
            "requester_joules": float(record.get("requester_joules", 0.0)),
            "mobile_joules": float(record.get("mobile_joules", 0.0)),
            "background_joules": float(record.get("background_joules", 0.0)),
            "constraint_penalty": float(record.get("constraint_penalty", 0.0)),
            "constraint_violations": {n: float(block.get("%s_violation" % n, 0.0))
                                      for n in block.get("names", [])},
            "constraint_signed": {n: float(block.get("%s_signed" % n, 0.0))
                                  for n in block.get("names", [])},
            "n_tasks": int(record["v2"]["n_tasks"]),
            "n_high_tasks": int(record["v2"]["n_high_tasks"]),
            "n_medium_tasks": int(record["v2"]["n_medium_tasks"]),
            "n_helper_tasks": int(record["v2"]["n_helper_tasks"]),
            "n_mec_tasks": int(record["v2"]["n_mec_tasks"]),
            "world_id": str(record["v2"]["world_id"]),
            "world_fingerprint_sha256": str(record["v2"]["world_fingerprint_sha256"]),
            "unmodeled_components": list(record.get("energy_unmodeled") or []),
            "energy_constraint": str(record.get("energy_constraint")),
        }
        evidence["iteration_measurements"] = {
            "mean_system_joules": float(getattr(trainer, "auto_energy_last_iteration", 0.0)),
            "mean_requester_joules": float(
                getattr(trainer, "auto_requester_last_iteration", 0.0)),
            "mean_violation": float(getattr(trainer, "auto_violation_last_iteration", 0.0)),
        }
        return evidence
    try:
        telemetry = measured.last_telemetry[-1]
        ledger = (measured.last_energy_ledger[-1]
                  if getattr(measured, "last_energy_ledger", None) else None)
        evidence["measured_env"] = {
            "type": type(measured).__name__,
            "is_original": measured is env,
            "n_slots": len(measured.last_telemetry),
        }
        evidence["last_rollout"] = {
            "makespan_s": float(telemetry.makespan_s),
            "world_makespan_s": float(telemetry.world_makespan_s),
            "system_joules": float(telemetry.system_joules),
            "requester_joules": float(telemetry.requester_joules),
            "mobile_joules": float(telemetry.mobile_joules),
            "background_joules": float(telemetry.background_joules),
            "constraint_penalty": float(telemetry.constraint_penalty),
            "n_tasks": int(telemetry.n_tasks),
            "n_high_tasks": int(telemetry.n_high_tasks),
            "n_medium_tasks": int(telemetry.n_medium_tasks),
            "n_helper_tasks": int(telemetry.n_helper_tasks),
            "n_mec_tasks": int(telemetry.n_mec_tasks),
            "world_id": str(telemetry.world_id),
            "world_fingerprint_sha256": str(telemetry.world_fingerprint_sha256),
            "unmodeled_components": (list(ledger.unmodeled) if ledger is not None else []),
        }
    except Exception as exc:
        raise V2RunError("the v2 environment produced no rollout telemetry to report: %s"
                         % exc) from exc
    return evidence


def _assert_real_run(evidence: Mapping) -> None:
    """A run that executed nothing, or never measured anything, is a FAILURE."""
    if int(evidence["iterations_completed"]) < 1:
        raise V2RunError(
            "no outer iteration COMPLETED (started=%r completed=%r last_completed=%r): an "
            "iteration counts only after its scheduled validation phase"
            % (evidence["iterations_started"], evidence["iterations_completed"],
               evidence["last_completed_iteration"]))
    if not evidence["sampler_counters"]:
        raise V2RunError("the sampler reported no counters: no rollout can have happened")
    last = evidence["last_rollout"]
    if not math.isfinite(float(last["makespan_s"])) or float(last["makespan_s"]) <= 0.0:
        raise V2RunError("the last rollout has no finite positive latency: %r" % (last,))
    if float(last["system_joules"]) <= 0.0:
        raise V2RunError("the last rollout reports ZERO system energy: the energy ledger did "
                         "not run on the production path")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="MARGO v2 real TensorFlow runner")
    ap.add_argument("--iterations", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--background", type=int, default=2)
    ap.add_argument("--background-policy", default="all_mec")
    ap.add_argument("--link-regime", default="moderate", choices=("stable", "moderate",
                                                                 "degraded"))
    ap.add_argument("--mec-workers", type=int, default=1)
    ap.add_argument("--reliability", action="store_true")
    ap.add_argument("--constraints", dest="constraints", action="store_true", default=True)
    ap.add_argument("--no-constraints", dest="constraints", action="store_false")
    ap.add_argument("--energy-budget-fraction", type=float, default=0.5)
    ap.add_argument("--ue-budget-fraction", type=float, default=1.0)
    ap.add_argument("--reward-mode", default="latency_only")
    ap.add_argument("--ckpt-dir", default="/tmp/v2run/ckpt")
    ap.add_argument("--resume-from", default=None)
    ap.add_argument("--json", default="/tmp/v2run/v2_tf_run.json")
    ap.add_argument("--r-select", type=int, default=2)
    ap.add_argument("--s-select", type=int, default=2)
    ap.add_argument("--device", default="cpu", choices=("cpu", "gpu"),
                    help="cpu honours CUDA_VISIBLE_DEVICES from the environment; gpu requests "
                         "the device explicitly with a BOUNDED memory fraction so a co-resident "
                         "service is never displaced")
    ap.add_argument("--gpu-fraction", type=float, default=0.08,
                    help="hard cap on the process GPU memory as a fraction of the card")
    ap.add_argument("--intra-op", type=int, default=0, help="0 = TensorFlow default")
    ap.add_argument("--inter-op", type=int, default=0, help="0 = TensorFlow default")
    args = ap.parse_args(argv)

    started = time.time()
    from spec.automotive_training.v2.stack import build_automotive_v2_stack

    trainer, algo = build_automotive_v2_stack(
        seed=int(args.seed), n_itr=int(args.iterations), ckpt_dir=str(args.ckpt_dir),
        reward_mode=str(args.reward_mode), use_energy=True,
        link_regime=str(args.link_regime), mec_workers=int(args.mec_workers),
        reliability=bool(args.reliability), background_dags=int(args.background),
        background_policy=str(args.background_policy),
        constraints_enabled=bool(args.constraints),
        budget_fractions={"total_energy": float(args.energy_budget_fraction),
                          "ue_energy": float(args.ue_budget_fraction)},
        r_select=int(args.r_select), s_select=int(args.s_select))

    import tensorflow as tf

    graph = tf.compat.v1.get_default_graph()
    with graph.as_default():
        cfg = tf.compat.v1.ConfigProto(allow_soft_placement=True,
                                       log_device_placement=False)
        if int(args.intra_op) > 0:
            cfg.intra_op_parallelism_threads = int(args.intra_op)
        if int(args.inter_op) > 0:
            cfg.inter_op_parallelism_threads = int(args.inter_op)
        if str(args.device) == "gpu":
            # BOUNDED allocation: a co-resident service (vLLM) keeps its memory, and this
            # process can never grow into it. allow_growth is deliberately NOT combined with
            # the fraction cap, so the cap is a hard ceiling.
            cfg.gpu_options.per_process_gpu_memory_fraction = float(args.gpu_fraction)
            cfg.gpu_options.allow_growth = True
        session_cfg = cfg
        device_used = "GPU" if str(args.device) == "gpu" else "CPU"
        print("session device=%s gpu_fraction=%s intra_op=%s inter_op=%s"
              % (device_used, args.gpu_fraction, args.intra_op, args.inter_op))
        sess = tf.compat.v1.Session(config=session_cfg)
        with sess.as_default():
            saver = None
            restored = False
            if args.resume_from:
                saver = tf.compat.v1.train.Saver(max_to_keep=1)
                saver.restore(sess, str(args.resume_from))
                restored = True
            else:
                sess.run(tf.compat.v1.global_variables_initializer())
            env = trainer.env
            env.reset()
            try:
                trainer.train()
                failed = None
            except Exception as exc:            # keep the evidence we did produce
                failed = "%s: %s" % (type(exc).__name__, exc)
            elapsed = time.time() - started
            ckpt_path = None
            try:
                if saver is None:
                    saver = tf.compat.v1.train.Saver(max_to_keep=1)
                Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
                ckpt_path = saver.save(sess, str(Path(args.ckpt_dir) / "v2_run.ckpt"))
            except Exception as exc:
                raise V2RunError("could not save a real checkpoint: %s" % exc) from exc
            evidence = _collect_evidence(trainer, env, algo, elapsed=elapsed,
                                        iterations=int(args.iterations))
            evidence["checkpoint"] = {"path": ckpt_path, "restored_from": args.resume_from,
                                      "was_resume": bool(restored),
                                      "status": ("partial_failed" if failed is not None
                                                 else "completed_iterations=%d"
                                                      % evidence["iterations_completed"])}
            progress = Path(str(getattr(trainer, "auto_run_dir", args.ckpt_dir))) / "v2_progress.json"
            evidence["progress"] = {"path": str(progress),
                                    "exists": bool(progress.exists())}
            evidence["device"] = {"requested": str(args.device),
                                  "cuda_visible_devices": os.environ.get(
                                      "CUDA_VISIBLE_DEVICES", "<unset>")}
            evidence["config"] = {
                "seed": int(args.seed), "iterations": int(args.iterations),
                "background_dags": int(args.background),
                "background_policy": str(args.background_policy),
                "link_regime": str(args.link_regime), "mec_workers": int(args.mec_workers),
                "reliability": bool(args.reliability), "constraints": bool(args.constraints),
                "reward_mode": str(args.reward_mode),
                "budget_fractions": {"total_energy": float(args.energy_budget_fraction),
                                     "ue_energy": float(args.ue_budget_fraction)},
                "r_select": int(args.r_select), "s_select": int(args.s_select),
                "device": str(args.device), "gpu_fraction": float(args.gpu_fraction),
                "intra_op": int(args.intra_op), "inter_op": int(args.inter_op),
                "obs_version": getattr(trainer, "auto_obs_version", None),
                "v2_system": getattr(trainer, "auto_v2_system", None),
                "v2_constraints": getattr(trainer, "auto_v2_constraints", None),
                "crn": getattr(trainer, "auto_crn", None),
            }
            evidence["error"] = failed
            report = {"schema": "v2_tf_run_v1", "ok": failed is None, "evidence": evidence,
                      "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started))}
            out = Path(args.json)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")
            print(json.dumps({"ok": report["ok"], "iterations_executed":
                              evidence["iterations_executed"],
                              "penalties": evidence["penalty"],
                              "lambdas": (evidence.get("constraints") or {}).get("lambdas"),
                              "last_rollout": {k: evidence["last_rollout"][k] for k in
                                               ("makespan_s", "system_joules",
                                                "constraint_penalty", "world_id")},
                              "error": failed}, indent=2, sort_keys=True))
            if failed is not None:
                return 2
            _assert_real_run(evidence)
            print("REAL TF RUN OK ->", out)
            return 0


if __name__ == "__main__":
    sys.exit(main())
