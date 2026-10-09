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
import hashlib
import json
import math
import shutil
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
        # `--iterations` is the TOTAL target. A resumed run must finish at exactly that
        # count without repeating completed work, so the next iteration to execute is
        # reported explicitly and is what `resume` must be launched with.
        "next_iteration": int(getattr(trainer, "next_iteration", 0) or 0),
        "outer_updates_completed": int(getattr(trainer, "outer_updates_completed", 0) or 0),
        "validations_completed": int(getattr(trainer, "validations_completed", 0) or 0),
        # alias kept so the terminal summary and any consumer written against the old key
        # cannot raise KeyError after a long run (that defect is exactly what aborted the
        # gate2 report AFTER training had finished)
        "iterations_executed": int(getattr(trainer, "completed_iterations", 0) or 0),
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


def _resume_bundle_path(ckpt_dir) -> Path:
    return Path(str(ckpt_dir)) / "v2_resume_state.json"


def _set_sampler_contract(trainer, contract) -> bool:
    """Apply the requested sampling contract to the object that OWNS the draw decision.

    `AutomotiveMetaSampler` forwards READS to its inner sampler but a WRITE lands on the wrapper
    and is invisible to the code that reads it, so the contract is set through the resolved owner
    (same rule as the draw cursor).
    """
    path, owner = _attribute_owner_path(trainer, "sampling_contract_version")
    if owner is None:
        _p, owner = _attribute_owner_path(trainer, "draw_branch")
    if owner is None:
        return False
    setter = getattr(owner, "set_sampling_contract", None)
    if callable(setter):
        setter(contract)
    else:
        owner.sampling_contract_version = str(contract)
        if hasattr(owner, "token_keyed_draws"):
            owner.token_keyed_draws = (str(contract) == _sampling_contract_default())
    return True


def _sampling_evidence(trainer, resume_state=None) -> dict:
    """The sampler branch that EXECUTED, plus the draw identity that was restored.

    `enable_crn=True` says a stateless branch EXISTS; it does not say the rollout took it. The
    meta sampler counts the branch of every rollout (`draw_branch_counts`) and reports the last
    one, so this is a measurement. A legacy-path rollout is surfaced explicitly rather than
    buried inside a boolean.
    """
    sampler = getattr(trainer, "auto_sampler", None) or getattr(trainer, "sampler", None)
    counts = dict(getattr(sampler, "draw_branch_counts", {}) or {})
    policy = getattr(trainer, "algo", None)
    policy = getattr(policy, "policy", None)
    out = {
        "requested_contract": getattr(trainer, "auto_sampling_contract", None),
        "observed_contract": _sampling_contract_of(trainer),
        "last_branch": getattr(sampler, "draw_branch", None),
        "branch_counts": {str(k): int(v) for k, v in counts.items()},
        "policy_sampling_contract": getattr(policy, "sampling_contract_version", None),
        "mask_mode": getattr(policy, "mask_mode", None),
        "draw_identity_seed": getattr(sampler, "draw_identity_seed", None),
        "draw_cursor": _read_draw_cursor(trainer),
        "draw_cursor_path": _draw_identity_owner_path(trainer)[0],
        "last_draw_key_prefix": getattr(sampler, "draw_key_prefix", None),
    }
    if resume_state is not None:
        out["resume_draw_cursor_path"] = resume_state.get("recorded_draw_cursor_path")
        out["resume_target_increased"] = bool(resume_state.get("target_increased"))
        out["resume_stored_iteration_target"] = resume_state.get("stored_iteration_target")
    legacy = int(counts.get("legacy_stateful_categorical", 0) or 0)
    keyed = int(counts.get("crn_token_keyed_v2", 0) or 0)
    out["all_rollouts_token_keyed"] = bool(keyed > 0 and legacy == 0)
    out["legacy_rollouts"] = legacy
    if legacy:
        out["legacy_warning"] = ("%d rollout(s) fell back to the STATEFUL categorical sampler: "
                                 "those draws are not reproducible across a process boundary"
                                 % legacy)
    # the draw cursor must be owned by the object the sampler advances, not by a wrapper
    out["draw_cursor_owner_is_sampler"] = bool(
        out["draw_cursor_path"] and str(out["draw_cursor_path"]).endswith("sampler"))
    return out


def _contract_record(trainer, args) -> dict:
    """The resume-relevant contract as THIS process would write it (F19)."""
    return {
        "obs_version": getattr(trainer, "auto_obs_version", None),
        "dataset_fingerprint": getattr(trainer, "auto_dataset_fingerprint", None),
        "scheduler_fingerprint": getattr(trainer, "auto_scheduler_fingerprint", None),
        "sampling_contract": _sampling_contract_of(trainer, args),
        "validation_interval": int(getattr(args, "validation_interval", 0) or 0),
        "objective_contract": _objective_contract_of(trainer),
        "constraint_contract": _constraint_contract_of(trainer),
        "protocol_id": getattr(trainer, "auto_protocol_id", None),
    }


#: Resume-bundle schema. v1 carried the iteration ledger, duals, sampler counters and the
#: fingerprints but NOT the environment identity or any RNG state, so a restored process rebuilt
#: the SAME iteration NUMBER over DIFFERENT worlds (measured: control `ep11_slot19` vs resume
#: `ep3_slot19`, and a 34% difference in mean_system_joules). v2 adds an `environment` section and
#: an `rng` section, and v1 bundles are REFUSED for exact resume.
RESUME_SCHEMA_V1 = "v2_resume_state_v1"
RESUME_SCHEMA = "v2_resume_state_v2"


# --- F34: OBSERVED kernel placement -----------------------------------------------------------------
# `allow_soft_placement=True` lets an op that was PLACED on `/gpu:0` silently EXECUTE on the CPU.
# The runner used to print the REQUESTED device and record the config, which asserts a device
# without measuring one. When `--kernel-profile <path.json>` is set, ONE run is executed with
# `RunOptions(trace_level=FULL_TRACE)` + `RunMetadata`, and the OBSERVED per-op device (from
# `step_stats.dev_stats`, which is the device the kernel actually ran on) is written out with a
# GPU/CPU summary. It is OFF by default, and RunOptions/RunMetadata change neither the graph nor
# the random streams.
def _profile_run_kwargs() -> dict:
    """`run(..., options=..., run_metadata=...)` for a full trace under TF 1.15."""
    import tensorflow as tf
    options = tf.compat.v1.RunOptions(trace_level=tf.compat.v1.RunOptions.FULL_TRACE)
    metadata = tf.compat.v1.RunMetadata()
    return {"options": options, "run_metadata": metadata}, metadata


def _profiled_device_record(metadata, graph=None):
    """OBSERVED per-op device placement from a `RunMetadata` full trace.

    `step_stats.dev_stats` groups the executed kernels BY DEVICE, and every `node_stats` there
    is a kernel that RAN on that device -- so this measures execution, not placement. The
    `device` field of each node is the placement the placer chose, kept alongside so a
    placed-on-GPU / ran-on-CPU fallback is visible.
    """
    step = getattr(metadata, "step_stats", None)
    if step is None:
        return {"ops": [], "device_counts": {}, "gpu_ops": 0, "cpu_ops": 0,
                "other_ops": 0, "n_ops": 0, "note": "no step_stats in RunMetadata"}
    placed = {}
    if graph is not None:
        try:
            for op in graph.get_operations():
                placed[op.name] = getattr(op, "device", "") or ""
        except Exception:                                      # pragma: no cover - graph guard
            placed = {}
    ops = []
    device_counts: dict = {}
    for dev in step.dev_stats:
        name = str(dev.device)
        for node in dev.node_stats:
            memory = getattr(node, "memory", None) or []
            first = memory[0] if len(memory) else None
            total_bytes = getattr(first, "total_bytes", None)
            if total_bytes is None and isinstance(first, dict):
                total_bytes = first.get("total_bytes", 0)
            ops.append({
                "op": str(node.node_name),
                "device": name,
                "placed_device": placed.get(str(node.node_name), ""),
                "all_start_micros": int(node.all_start_micros),
                "op_time_micros": float(node.op_end_rel_micros - node.op_start_rel_micros),
                "memory_bytes": int(total_bytes or 0),
            })
            device_counts[name] = device_counts.get(name, 0) + 1
    gpu = sum(n for d, n in device_counts.items() if ":GPU:" in d or "/gpu:" in d.lower())
    cpu = sum(n for d, n in device_counts.items() if ":CPU:" in d or "/cpu:" in d.lower())
    return {
        "ops": ops,
        "device_counts": device_counts,
        "gpu_ops": int(gpu),
        "cpu_ops": int(cpu),
        "other_ops": int(sum(device_counts.values()) - gpu - cpu),
        "n_ops": int(sum(device_counts.values())),
    }


def _visible_device_info() -> dict:
    """The devices this process can actually see (requested/visible/observed are distinct)."""
    import tensorflow as tf
    info = {"cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
            "tf_force_gpu_allow_growth": os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH", "<unset>"),
            "physical_devices": []}
    try:
        cfg = tf.compat.v1.ConfigProto()
        info["config_device_count"] = {str(k): int(v) for k, v in cfg.device_count.items()}
        info["allow_soft_placement"] = bool(cfg.allow_soft_placement)
        info["log_device_placement"] = bool(cfg.log_device_placement)
    except Exception as exc:                                   # pragma: no cover - config guard
        info["config_error"] = str(exc)[:200]
    try:
        from tensorflow.python.client import device_lib
        info["physical_devices"] = [{"name": d.name, "device_type": d.device_type}
                                    for d in device_lib.list_local_devices()]
    except Exception as exc:                                   # pragma: no cover - probe guard
        info["physical_devices_error"] = str(exc)[:200]
    try:
        info["gpu_devices"] = [d.name for d in tf.config.experimental.list_physical_devices("GPU")]
    except Exception:                                          # pragma: no cover - TF1.15 has none
        info["gpu_devices"] = []
    return info


class KernelProfiler:
    """Capture OBSERVED device placement for ONE run without touching training semantics.

    `sess.run` is wrapped: the FIRST run that executed a non-trivial number of kernels is traced
    with `RunOptions(FULL_TRACE)` and its `RunMetadata` is kept. The traced call returns exactly
    what the untraced call returns, and `RunOptions`/`RunMetadata` are not part of any random
    op's inputs, so the RNG stream and every fetched value are unchanged.
    """

    def __init__(self, path, requested_device="cpu", min_ops=10):
        self.path = Path(str(path))
        self.requested_device = str(requested_device)
        self.min_ops = int(min_ops)
        self.captured = None
        self.captured_run_number = None
        self.runs_seen = 0
        self._orig_run = None

    def start(self, sess):
        import tensorflow as tf
        self._tf = tf
        self._orig_run = sess.run

        def traced_run(fetches, feed_dict=None, **kwargs):
            self.runs_seen += 1
            if self.captured is not None or "options" in kwargs \
                    or "run_metadata" in kwargs:
                return self._orig_run(fetches, feed_dict=feed_dict, **kwargs)
            extra, metadata = _profile_run_kwargs()
            out = self._orig_run(fetches, feed_dict=feed_dict, **dict(kwargs, **extra))
            try:
                n_ops = sum(len(d.node_stats) for d in metadata.step_stats.dev_stats)
            except Exception:                                  # pragma: no cover - trace guard
                n_ops = 0
            if n_ops >= self.min_ops:
                self.captured = metadata
                self.captured_run_number = self.runs_seen
            return out

        sess.run = traced_run

    def stop(self, sess):
        if self._orig_run is not None:
            sess.run = self._orig_run
            self._orig_run = None

    def write(self, graph=None) -> dict:
        if self.captured is None:
            doc = {"schema": "v2_kernel_profile_v1", "ok": False,
                   "reason": "no run reached the profiling op-count threshold (%d)" % self.min_ops,
                   "runs_seen": int(self.runs_seen),
                   "requested_device": self.requested_device,
                   "visible": _visible_device_info()}
        else:
            observed = _profiled_device_record(self.captured, graph=graph)
            doc = {"schema": "v2_kernel_profile_v1", "ok": True,
                   "requested_device": self.requested_device,
                   "profiled_run_number": int(self.captured_run_number),
                   "runs_seen": int(self.runs_seen),
                   "visible": _visible_device_info(),
                   "observed": {k: v for k, v in observed.items() if k != "ops"},
                   "op_device": observed["ops"]}
            doc["gpu_share"] = (float(observed["gpu_ops"]) / float(observed["n_ops"])
                                if observed["n_ops"] else 0.0)
            if self.requested_device == "gpu" and int(observed["gpu_ops"]) == 0:
                doc["warning"] = ("requested GPU but NO op executed on a GPU device: soft "
                                  "placement fell back to CPU")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(doc, indent=2, sort_keys=True, default=str) + "\n")
        os.replace(str(tmp), str(self.path))
        return doc


def _sampler_clone_envs(trainer) -> list:
    """Every environment CLONE reachable from the trainer.

    `MetaIterativeEnvExecutor.__init__` builds its worker list ONCE with
    `copy.deepcopy(env)` (vectorized_env_executor.py:24), BEFORE any resume is applied. Those
    clones, not `trainer.env`, are what actually run the rollouts and advance the episode
    counter that names the world (`world_id = "ep<episodes>_slot<flat>"`). Restoring only the
    primary env therefore left the world sequence unchanged -- the measured F33 divergence
    (`ep11_slot19` control vs `ep3_slot19` resume).
    """
    found: list = []
    seen: set = set()

    def _visit(obj, depth=0):
        if obj is None or depth > 4 or id(obj) in seen:
            return
        seen.add(id(obj))
        envs = getattr(obj, "envs", None)
        if isinstance(envs, (list, tuple)) or (hasattr(envs, "__len__") and hasattr(envs, "tolist")):
            try:
                for e in list(envs):
                    if hasattr(e, "episodes"):
                        found.append(e)
            except TypeError:
                pass
        for attr in ("sampler", "meta_sampler", "vec_env", "env", "base_sampler"):
            try:
                child = getattr(obj, attr, None)
            except Exception:                                  # pragma: no cover - proxy guard
                child = None
            if child is not None and child is not obj:
                _visit(child, depth + 1)

    _visit(getattr(trainer, "sampler", None))
    _visit(getattr(trainer, "algo", None))
    return found


def _env_identity(env, trainer=None) -> dict:
    """The state that decides WHICH world an iteration sees.

    `V2AutomotiveEnv` derives `world_id`, the helper seeds and the link seed from `episodes` (and
    from `_world_key` when one is pinned), while `AutomotiveEnv` derives the MC realization from
    `reset_count`. Restoring the iteration counter without these replays a different world
    sequence -- the F33 divergence.
    """
    if env is None:
        return {}
    base = getattr(env, "base", None)
    out = {
        "episodes": int(getattr(env, "episodes", 0) or 0),
        "world_key": getattr(env, "_world_key", None),
        "reset_count": int(getattr(base, "reset_count", 0) or 0) if base is not None else None,
        "realization_epoch": (getattr(base, "realization_epoch", None)
                              if base is not None else None),
        "task_id": int(getattr(base, "task_id", -1)) if base is not None else None,
        "base_seed": int(getattr(env, "base_seed", 0) or 0),
        "slots_per_task": int(getattr(env, "slots_per_task", 0) or 0),
        "link_regime": str(getattr(env, "link_regime", "")),
        "background_dags": int(getattr(env, "background_dags", 0) or 0),
    }
    # EXACT per-clone counters: the workers can reset independently (`vectorized_env_executor`
    # resets a single env when its path terminates), so a single scalar is not sufficient.
    if trainer is not None:
        clones = _sampler_clone_envs(trainer)
        out["clone_count"] = len(clones)
        out["clone_episodes"] = [int(getattr(c, "episodes", 0) or 0) for c in clones]
        out["clone_reset_counts"] = [
            int(getattr(getattr(c, "base", None), "reset_count", 0) or 0) for c in clones]
        out["clone_task_ids"] = [
            int(getattr(getattr(c, "base", None), "task_id", -1)) for c in clones]
        def _gi(c):
            gi = getattr(getattr(c, "base", None), "graph_indices", None)
            if gi is None:
                return []
            return [int(x) for x in np.asarray(gi).reshape(-1)]

        out["clone_graph_indices"] = [_gi(c) for c in clones]
    inner = getattr(env, "env", None)
    if inner is not None and inner is not env:
        out["wrapped"] = _env_identity(inner)
    return out


def _rng_state(trainer) -> dict:
    """The Python/NumPy random streams the training loop consumes.

    A seed alone cannot reconstruct an advanced `RandomState`. `MRLCO.rng` drives support-row
    selection (and therefore which trajectories the inner/outer update sees), so it must be
    carried across the process boundary.
    """
    out: dict = {}
    algo = getattr(trainer, "algo", None)
    rng = getattr(algo, "rng", None) if algo is not None else None
    if rng is not None and hasattr(rng, "get_state"):
        state = rng.get_state()
        out["algo"] = {
            "kind": "numpy_randomstate",
            "bit_generator": str(state[0]),
            "keys": [str(x) for x in state[1].tolist()],
            "pos": int(state[2]),
            "has_gauss": int(state[3]),
            "cached_gaussian": float(state[4]),
        }
    # The v2 stack seeds the stdlib `random` module during construction
    # (`meta_trainer.py: `random.seed(SEED)`), so its state at a checkpoint boundary is part of
    # the run's identity. The first-divergence trace showed it differing while every shared TF
    # variable was bit-identical; it is restored here so the boundary is complete whether or not
    # any current consumer advances it.
    try:
        import random as _random
        st = _random.getstate()
        out["python_random"] = {"kind": "python_random", "version": int(st[0]),
                                "keys": [str(x) for x in st[1]], "gauss": st[2]}
    except Exception:                                          # pragma: no cover
        pass
    np_state = np.random.get_state()
    out["numpy_global"] = {
        "kind": "numpy_randomstate",
        "bit_generator": str(np_state[0]),
        "keys": [str(x) for x in np_state[1].tolist()],
        "pos": int(np_state[2]),
        "has_gauss": int(np_state[3]),
        "cached_gaussian": float(np_state[4]),
    }
    return out


def _restore_rng_state(trainer, doc) -> None:
    doc = doc or {}
    algo = getattr(trainer, "algo", None)
    entry = doc.get("algo")
    if algo is not None and entry is not None and hasattr(getattr(algo, "rng", None), "set_state"):
        algo.rng.set_state(_rng_tuple(entry))
    entry = doc.get("numpy_global")
    if entry is not None:
        np.random.set_state(_rng_tuple(entry))
    entry = doc.get("python_random")
    if entry is not None:
        import random as _random
        version = int(entry["version"])
        _random.setstate((version, tuple(int(x) for x in entry["keys"]),
                          entry.get("gauss")))


def _rng_tuple(entry) -> tuple:
    keys = np.array([int(x) for x in entry["keys"]], dtype=np.uint32)
    return (str(entry["bit_generator"]), keys, int(entry["pos"]),
            int(entry["has_gauss"]), float(entry["cached_gaussian"]))


def _attribute_owner_path(trainer, name, roots=("sampler", "meta_sampler", "auto_sampler",
                                                "algo")):
    """`(dotted_path, object)` of the object whose `__dict__` actually holds `name`.

    `AutomotiveMetaSampler` forwards READS to its inner sampler through `__getattr__`, but a WRITE
    on the wrapper lands on the wrapper and is invisible to the code that reads the attribute off
    the inner object. Resolving the true owner through `__dict__` is what makes the restore
    effective; using `hasattr` alone silently wrote the draw cursor (and the counters) to the
    wrapper, which is why the resumed process still replayed draws from the start.

    The search is a BREADTH-FIRST walk over the known wrapper attributes instead of the previous
    two-level probe: `trainer.auto_sampler.sampler` (the v2 stack) is three levels deep, and the
    old probe never reached it, so a restore was silently a no-op. The returned path is recorded
    in the bundle so the restore can be applied to the SAME attribute it was read from.
    """
    import collections

    queue = collections.deque()
    seen = set()
    for attr in roots:
        obj = getattr(trainer, attr, None)
        if obj is not None:
            queue.append((attr, obj))
    while queue:
        path, obj = queue.popleft()
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        if name in getattr(obj, "__dict__", {}):
            return path, obj
        for attr in ("sampler", "meta_sampler", "base_sampler", "inner"):
            try:
                child = getattr(obj, attr, None)
            except Exception:                                  # pragma: no cover - proxy guard
                child = None
            if child is not None and child is not obj:
                queue.append(("%s.%s" % (path, attr), child))
    return None, None


def _attribute_owner(trainer, name):
    return _attribute_owner_path(trainer, name)[1]


def _draw_identity_owner(trainer):
    """The object that owns `_draw_cursor` (the meta sampler, however it is wrapped)."""
    return _draw_identity_owner_path(trainer)[1]


def _draw_identity_owner_path(trainer, recorded_path=None):
    """`(path, object)` for `_draw_cursor`, preferring the path the bundle recorded.

    A resume that restored the cursor on a DIFFERENT object than the one that advances it would
    replay draws; the recorded path makes that impossible to reintroduce silently.
    """
    if recorded_path:
        obj = trainer
        for part in str(recorded_path).split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if obj is not None and "_draw_cursor" in getattr(obj, "__dict__", {}):
            return str(recorded_path), obj
    return _attribute_owner_path(trainer, "_draw_cursor")


def _read_draw_cursor(trainer) -> int:
    obj = _draw_identity_owner(trainer)
    return int(getattr(obj, "_draw_cursor", 0) or 0)


def _read_draw_cursor_with_path(trainer) -> tuple:
    """`(value, dotted_path)` -- the path is persisted so the restore is verifiable."""
    path, obj = _draw_identity_owner_path(trainer)
    return int(getattr(obj, "_draw_cursor", 0) or 0), path


def _restore_draw_cursor(trainer, value, recorded_path=None) -> bool:
    """Write the cursor to its ACTUAL owning attribute. Returns whether it was applied."""
    if value is None:
        return False
    path, obj = _draw_identity_owner_path(trainer, recorded_path=recorded_path)
    if obj is None:
        return False
    obj._draw_cursor = int(value)
    return True


def _restore_sampler_counters(trainer, counters) -> None:
    """Re-apply the persisted sampler counters to whichever object owns them."""
    if not counters:
        return
    owner = _attribute_owner(trainer, "counters")
    if owner is None:
        for attr in ("sampler", "meta_sampler"):
            obj = getattr(trainer, attr, None)
            if obj is not None and isinstance(getattr(obj, "counters", None), dict):
                owner = obj
                break
    if owner is None:
        return
    existing = getattr(owner, "counters", None)
    if isinstance(existing, dict):
        for k, v in dict(counters).items():
            if k in existing:
                existing[k] = int(v)


def _restore_env_identity(trainer, doc) -> None:
    """Apply the restored environment identity. Must run BEFORE any reset/selection/sampling."""
    env = getattr(trainer, "env", None)
    if env is None or not doc:
        return
    root = doc
    while True:                                   # unwrap nested trainer-visible wrappers
        if "episodes" in doc:
            break
        inner = doc.get("wrapped")
        if not inner:
            return
        doc = inner
    env.episodes = int(doc.get("episodes", 0))
    if doc.get("world_key") is not None and hasattr(env, "set_world_key"):
        env.set_world_key(doc["world_key"])
    base = getattr(env, "base", None)
    if base is not None:
        if doc.get("reset_count") is not None:
            base.reset_count = int(doc["reset_count"])
        if hasattr(env, "set_realization_epoch"):
            env.set_realization_epoch(doc.get("realization_epoch"))
        elif doc.get("realization_epoch") is not None:
            base.realization_epoch = int(doc["realization_epoch"])
    # the CLONES are the objects that run the rollouts and advance the counters the world id is
    # derived from, so they must be restored too, positionally and exactly.
    clones = _sampler_clone_envs(trainer)
    eps = list(root.get("clone_episodes") or [])
    rcs = list(root.get("clone_reset_counts") or [])
    if eps and len(eps) != len(clones):
        raise V2RunError(
            "resume bundle carries %d clone episode counters but this process has %d sampler "
            "clones: the world sequence cannot be restored exactly, refusing to resume"
            % (len(eps), len(clones)))
    tids = list(root.get("clone_task_ids") or [])
    gids = list(root.get("clone_graph_indices") or [])
    for i, clone in enumerate(clones):
        if eps:
            clone.episodes = int(eps[i])
        cbase = getattr(clone, "base", None)
        if rcs and cbase is not None:
            cbase.reset_count = int(rcs[i])
        # the clone's CURRENT task assignment is part of the boundary too: it is what the
        # next `sync_task_policies_from_core` / reset reads before `update_tasks` replaces it
        if tids and cbase is not None:
            cbase.task_id = int(tids[i])
        if gids and cbase is not None and gids[i]:
            cbase.graph_indices = np.asarray(gids[i], dtype=np.int32)
        if hasattr(clone, "set_realization_epoch"):
            clone.set_realization_epoch(doc.get("realization_epoch"))
        elif rcs and cbase is not None and doc.get("realization_epoch") is not None:
            cbase.realization_epoch = int(doc["realization_epoch"])


#: Transactional checkpoint generations. A generation is a directory that contains the TF
#: shards AND the auxiliary state bundle, committed together. `CURRENT` points at the newest
#: COMPLETE generation and is replaced atomically, so a crash during a save can never publish a
#: mixed old-TF / new-state snapshot: the half-written generation is never referenced.
GEN_DIR = "generations"
CURRENT_POINTER = "CURRENT"
GENERATION_MANIFEST = "manifest.json"


def _sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _generation_dir(ckpt_dir, itr) -> Path:
    return Path(ckpt_dir) / GEN_DIR / ("gen_%08d" % int(itr))


def _write_generation(sess, saver, trainer, args, itr) -> Path:
    """Write one COMPLETE generation, then commit it atomically."""
    import tensorflow as tf  # noqa: F401  (session comes from the caller)
    root = Path(args.ckpt_dir) / GEN_DIR
    root.mkdir(parents=True, exist_ok=True)
    final = _generation_dir(args.ckpt_dir, itr)
    staging = final.with_name(final.name + ".tmp")
    if staging.exists():
        shutil.rmtree(str(staging))
    staging.mkdir(parents=True)
    # 1) TF shards
    if saver is None:
        saver = tf.compat.v1.train.Saver(max_to_keep=1)
    saver.save(sess, str(staging / "v2_run.ckpt"))
    # 2) auxiliary state bundle (written atomically inside the staging dir)
    state = _build_resume_state(trainer, args)
    state["generation"] = {"iteration": int(itr), "complete": True}
    bundle = staging / "v2_resume_state.json"
    tmp = bundle.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(str(tmp), str(bundle))
    # 3) manifest over every required file
    required = sorted(p for p in staging.iterdir() if p.is_file())
    manifest = {
        "schema": "v2_checkpoint_generation_v1",
        "iteration": int(itr),
        "next_iteration": int(itr) + 1,
        "sampling_contract": _sampling_contract_of(trainer, args),
        # F19: the manifest carries the whole contract, so a resume from a generation whose
        # bundle is damaged can still be checked against the manifest (and a manifest that
        # disagrees with its own bundle is itself a mismatch).
        "obs_version": getattr(trainer, "auto_obs_version", None),
        "dataset_fingerprint": getattr(trainer, "auto_dataset_fingerprint", None),
        "scheduler_fingerprint": getattr(trainer, "auto_scheduler_fingerprint", None),
        "validation_interval": int(getattr(args, "validation_interval", 0) or 0),
        "objective_contract": _objective_contract_of(trainer),
        "constraint_contract": _constraint_contract_of(trainer),
        "files": {p.name: _sha256_file(p) for p in required},
        "complete": True,
    }
    (staging / GENERATION_MANIFEST).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    # 4) atomic publish: the rename is the commit point
    if final.exists():
        shutil.rmtree(str(final))
    os.replace(str(staging), str(final))
    # 5) atomic pointer replacement
    ptr = Path(args.ckpt_dir) / CURRENT_POINTER
    tmp = ptr.with_suffix(".tmp")
    tmp.write_text(json.dumps({"schema": "v2_checkpoint_pointer_v1",
                               "generation": final.name,
                               "iteration": int(itr)}, sort_keys=True) + "\n")
    os.replace(str(tmp), str(ptr))
    return final


def _verify_generation(gen_dir) -> tuple:
    """(ok, reason). A generation is usable only if it is EXACTLY the manifest it declares.

    Checks, in order: the manifest exists and parses; its schema is the expected one; every
    declared file is present with the declared hash; no UNDECLARED file was added (an extra
    binary or a foreign state bundle would otherwise be trusted as part of the snapshot); the
    TF index and the state bundle exist. Every failure is a precise reason, never a boolean.
    """
    gen_dir = Path(gen_dir)
    man = gen_dir / GENERATION_MANIFEST
    if not man.is_file():
        return False, "manifest missing"
    try:
        doc = json.loads(man.read_text())
    except Exception as exc:
        return False, "manifest unreadable: %s" % exc
    if doc.get("schema") != "v2_checkpoint_generation_v1":
        return False, "unexpected manifest schema %r" % (doc.get("schema"),)
    declared = doc.get("files") or {}
    for name, want in declared.items():
        f = gen_dir / name
        if not f.is_file():
            return False, "missing file %s" % name
        got = _sha256_file(f)
        if got != want:
            return False, "hash mismatch for %s (manifest %s, file %s)" % (name, want[:12], got[:12])
    present = sorted(p.name for p in gen_dir.iterdir() if p.is_file())
    extra = [name for name in present
             if name not in declared and name != GENERATION_MANIFEST]
    if extra:
        return False, "undeclared file(s) in generation: %s" % ", ".join(extra)
    if not (gen_dir / "v2_run.ckpt.index").is_file():
        return False, "tf index missing"
    if not (gen_dir / "v2_resume_state.json").is_file():
        return False, "state bundle missing"
    return True, "complete"


def _resolve_resume_generation(ckpt_dir) -> Path:
    """CURRENT if it is complete, else the newest complete generation, else a precise failure."""
    ckpt_dir = Path(ckpt_dir)
    ptr = ckpt_dir / CURRENT_POINTER
    candidates = []
    if ptr.is_file():
        try:
            name = json.loads(ptr.read_text()).get("generation")
        except Exception:
            name = None
        if name:
            candidates.append(ckpt_dir / GEN_DIR / str(name))
    gen_root = ckpt_dir / GEN_DIR
    if gen_root.is_dir():
        for d in sorted((p for p in gen_root.iterdir() if p.is_dir()
                         and not p.name.endswith(".tmp")), reverse=True):
            if d not in candidates:
                candidates.append(d)
    for cand in candidates:
        ok, reason = _verify_generation(cand)
        if ok:
            return cand
        print("checkpoint generation %s rejected: %s" % (cand.name, reason))
    raise V2RunError(
        "no COMPLETE checkpoint generation under %s: refusing to resume from a partial or "
        "mixed snapshot. Refusing rather than guessing." % (ckpt_dir,))


def _resolve_resume_ckpt(ckpt_dir) -> str:
    """The TF checkpoint PREFIX to restore for an exact continuation."""
    legacy = Path(ckpt_dir) / "v2_run.ckpt"
    if (Path(ckpt_dir) / GEN_DIR).is_dir() or (Path(ckpt_dir) / CURRENT_POINTER).is_file():
        gen = _resolve_resume_generation(ckpt_dir)
        return str(gen / "v2_run.ckpt")
    if legacy.is_file() or legacy.with_suffix(".ckpt.index").is_file():
        return str(legacy)
    raise V2RunError("no checkpoint found under %s" % (ckpt_dir,))


def _read_generation_manifest(gen_dir):
    """The generation's manifest as a dict, or None when it is absent/unreadable."""
    man = Path(gen_dir) / GENERATION_MANIFEST
    if not man.is_file():
        return None
    try:
        doc = json.loads(man.read_text())
    except Exception:
        return None
    return doc if isinstance(doc, dict) else None


def _complete_generations(ckpt_dir) -> list:
    """Every COMPLETE generation, newest first. `.tmp` staging directories are never included."""
    gen_root = Path(ckpt_dir) / GEN_DIR
    if not gen_root.is_dir():
        return []
    out = []
    for d in sorted((p for p in gen_root.iterdir() if p.is_dir()
                     and not p.name.endswith(".tmp")), reverse=True):
        ok, _reason = _verify_generation(d)
        if ok:
            out.append(d)
    return out


def _prune_generations(ckpt_dir, keep=3, protect=()) -> list:
    """Delete OLD complete generations, always keeping at least `keep` recoverable ones.

    `protect` names generations that must survive pruning regardless of age (the best-validation
    generation and the final generation). The NEWEST complete generation is never pruned, so a
    crash can always resume from the latest boundary. Returns the names that were deleted.
    """
    keep = max(1, int(keep))
    complete = _complete_generations(ckpt_dir)
    if len(complete) <= keep:
        return []
    protected = {str(p) for p in protect if p}
    newest = complete[0].name
    survivors = {d.name for d in complete[:keep]}
    survivors.add(newest)
    survivors |= {name for name in protected if (Path(ckpt_dir) / GEN_DIR / name).is_dir()}
    deleted = []
    for gen in complete:
        if gen.name in survivors:
            continue
        _ok, reason = _verify_generation(gen)
        if not _ok:                                  # never delete something already unusable
            print("prune: leaving %s in place (%s)" % (gen.name, reason))
            continue
        shutil.rmtree(str(gen))
        deleted.append(gen.name)
    return deleted


def _build_resume_state(trainer, args) -> dict:
    """The NON-TensorFlow half of a resumable checkpoint.

    `tf.compat.v1.train.Saver` already carries the model variables AND every optimizer slot
    that lives in the default graph's `GLOBAL_VARIABLES`/`TRAINABLE_VARIABLES` collections
    (Adam `m`/`v` and their `beta*_power` counters are created by `tf.train.AdamOptimizer`
    inside the policy's `variable_scope`s, so they ARE in the Saver). What the Saver cannot
    carry is the Python-side state: the iteration ledger, the dual variables, the sampler
    counters, the constraint schema and the split/config fingerprints. Without those a
    restore silently restarts the duals from zero and repeats completed iterations.
    """
    manager = getattr(trainer, "auto_v2_constraint_manager", None)
    sampler = getattr(trainer, "sampler", None)
    draw_cursor, draw_cursor_path = _read_draw_cursor_with_path(trainer)
    return {
        "schema": RESUME_SCHEMA,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "iteration_target": int(args.iterations),
        "started_iterations": int(getattr(trainer, "started_iterations", 0) or 0),
        "completed_iterations": int(getattr(trainer, "completed_iterations", 0) or 0),
        "last_completed_iteration": int(getattr(trainer, "last_completed_iteration", -1)),
        "next_iteration": int(getattr(trainer, "next_iteration", 0) or 0),
        "outer_updates_completed": int(getattr(trainer, "outer_updates_completed", 0) or 0),
        "dual_updates_completed": int(getattr(trainer, "dual_updates_completed", 0) or 0),
        "validations_completed": int(getattr(trainer, "validations_completed", 0) or 0),
        "seed": int(args.seed),
        "background_dags": int(args.background),
        "link_regime": str(args.link_regime),
        "background_policy": str(args.background_policy),
        "mec_workers": int(args.mec_workers),
        "r_select": int(args.r_select),
        "s_select": int(args.s_select),
        "obs_version": getattr(trainer, "auto_obs_version", None),
        "protocol_id": getattr(trainer, "auto_protocol_id", None),
        "dataset_fingerprint": getattr(trainer, "auto_dataset_fingerprint", None),
        "scheduler_fingerprint": getattr(trainer, "auto_scheduler_fingerprint", None),
        "v2_system": getattr(trainer, "auto_v2_system", None),
        # F19: the CONTRACT the numbers were produced under. Each of these is written AND
        # compared on resume, because a checkpoint that is silently re-interpreted under a
        # different observation schema, dataset, scheduler, sampling contract or objective is
        # not a continuation of the same run even when every seed matches.
        "sampling_contract": _sampling_contract_of(trainer, args),
        "validation_interval": int(getattr(args, "validation_interval", 0) or 0),
        "objective_contract": _objective_contract_of(trainer),
        "constraint_contract": _constraint_contract_of(trainer, manager),
        "constraints": (manager.state() if manager is not None else None),
        "sampler_counters": {str(k): int(v) for k, v in
                             dict(getattr(sampler, "counters", {}) or {}).items()},
        # F33: the IDENTITY-KEYED draw cursor. The rollout pins
        # [draw_identity_seed, _draw_cursor] before every `get_actions`, so this counter IS the
        # draw sequence; without it a resumed process would restart the identity at 0 and replay
        # the same draws it already made.
        "draw_cursor": draw_cursor,
        # ... and the ATTRIBUTE it was read from, so the restore lands on the same owner
        # (`trainer.auto_sampler.sampler` in the v2 stack) instead of a forwarding wrapper.
        "draw_cursor_path": draw_cursor_path,
        "best_val_objective": getattr(trainer, "best_val_objective", None),
        # F33 additions: the identity that decides WHICH worlds the next iteration sees, and the
        # advanced Python/NumPy random streams. Without these the loop resumes at the right
        # iteration NUMBER over a different world sequence.
        "environment": _env_identity(getattr(trainer, "env", None), trainer),
        "rng": _rng_state(trainer),
    }


def _sampling_contract_default() -> str:
    """The v2 sampling contract: the stateless, token-keyed categorical draw."""
    return "crn_token_keyed_v2"


def _sampling_contract_of(trainer, args=None) -> str:
    """The ACTUAL sampling contract: the requested one, else the sampler's own report.

    The trainer attribute is the request (`--sampling-contract`, default
    `crn_token_keyed_v2`); `draw_branch` / `sampling_contract_version` is what executed. When the
    sampler reports a branch it actually took, that branch wins, so the bundle can never advertise
    a contract the sampler did not run.
    """
    requested = (getattr(args, "sampling_contract", None)
                 or getattr(trainer, "auto_sampling_contract", None)
                 or _sampling_contract_default())
    sampler = getattr(trainer, "auto_sampler", None) or getattr(trainer, "sampler", None)
    observed = None
    for attr in ("sampling_contract_version", "draw_branch"):
        observed = getattr(sampler, attr, None)
        if observed:
            break
    if str(requested) == _sampling_contract_default() and observed:
        return str(observed)
    return str(requested)


def _objective_contract_of(trainer) -> str:
    """A stable signature of the OBJECTIVE/unit contract, independent of the dual values."""
    doc = {
        "v2_system": getattr(trainer, "auto_v2_system", None),
        "reward_mode": getattr(trainer, "auto_reward_mode", None),
        "decoding": getattr(trainer, "auto_decoding", None),
        "run_kind": getattr(trainer, "auto_run_kind", None),
        "method_id": getattr(trainer, "auto_method_id", None),
        "axes": getattr(trainer, "auto_axes", None),
    }
    return hashlib.sha256(json.dumps(doc, sort_keys=True, default=str).encode()).hexdigest()


def _constraint_contract_of(trainer, manager=None) -> str:
    """The constraint SCHEMA (names + spec hash + enabled flag), never the dual values."""
    if manager is None:
        manager = getattr(trainer, "auto_v2_constraint_manager", None)
    cfg = getattr(trainer, "auto_v2_constraints", None) or {}
    names = cfg.get("names")
    if names is None and manager is not None:
        names = list(getattr(manager, "names", []) or [])
    doc = {
        "enabled": bool(cfg.get("enabled", manager is not None)),
        "names": [str(n) for n in (names or [])],
        "spec_sha256": cfg.get("spec_sha256"),
        "objectives": cfg.get("objectives"),
    }
    return hashlib.sha256(json.dumps(doc, sort_keys=True, default=str).encode()).hexdigest()


def _write_resume_state(trainer, args) -> Path:
    """Atomic write: a torn state file must never look like a completed boundary."""
    path = _resume_bundle_path(args.ckpt_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = _build_resume_state(trainer, args)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(str(tmp), str(path))
    return path


def _resume_expected_contract(raw, args, trainer=None, manifest=None) -> dict:
    """Every contract key whose change makes a resume a DIFFERENT run.

    F19: the bundle already WROTE `obs_version`, `dataset_fingerprint` and
    `scheduler_fingerprint`, but the comparison only looked at seed/background/link/
    background_policy/mec_workers, so a checkpoint produced under a different observation
    schema, dataset, scheduler or sampling contract was silently accepted. The generation
    manifest records `sampling_contract` too, so it is surfaced here as an independent check:
    a manifest that disagrees with the bundle is a mismatch of its own.
    """
    manifest = manifest or {}
    manifest_contract = manifest.get("sampling_contract")
    bundle_contract = raw.get("sampling_contract")
    # The contract THIS run requests is the authority on the current side: in production
    # `trainer.auto_sampling_contract` is set from `--sampling-contract`, so reading the trainer
    # would compare the request against itself and never detect a change.
    requested_contract = str(getattr(args, "sampling_contract", None)
                             or _sampling_contract_of(trainer, args)
                             or _sampling_contract_default())
    checkpoint_contract = (manifest_contract if manifest_contract is not None
                           else bundle_contract)
    expected = {
        "seed": int(args.seed),
        "background_dags": int(args.background),
        "link_regime": str(args.link_regime),
        "background_policy": str(args.background_policy),
        "mec_workers": int(args.mec_workers),
        "obs_version": getattr(trainer, "auto_obs_version", None),
        "dataset_fingerprint": getattr(trainer, "auto_dataset_fingerprint", None),
        "scheduler_fingerprint": getattr(trainer, "auto_scheduler_fingerprint", None),
        "sampling_contract": requested_contract,
        "validation_interval": int(getattr(args, "validation_interval", 0) or 0),
        "objective_contract": _objective_contract_of(trainer) if trainer is not None else None,
        "constraint_contract": (_constraint_contract_of(trainer)
                                if trainer is not None else None),
    }
    if requested_contract != _sampling_contract_default():
        # ... and when a non-default contract is REQUESTED, the checkpoint's own value is what it
        # must match: surface it explicitly so the message names both sides.
        expected["sampling_contract@checkpoint"] = checkpoint_contract
    if (manifest_contract is not None and bundle_contract is not None
            and str(manifest_contract) != str(bundle_contract)):
        expected["sampling_contract@manifest"] = manifest_contract
    return expected


def _resume_mismatches(raw, args, trainer=None, manifest=None) -> list:
    """The list of `key: checkpoint=... requested=...` contract mismatches. Pure function.

    Split out of `_restore_resume_state` so the rejection rules can be unit-tested with
    synthetic bundles and no TensorFlow, no session and no checkpoint on disk.
    """
    mismatches = []
    manifest = manifest if isinstance(manifest, dict) else {}
    for key, value in _resume_expected_contract(raw, args, trainer, manifest).items():
        if key == "sampling_contract@manifest":
            # the manifest and the bundle disagree: name both, never silently pick a winner
            mismatches.append(
                "sampling_contract: manifest=%r bundle=%r"
                % (manifest.get("sampling_contract"), raw.get("sampling_contract")))
            continue
        if key == "sampling_contract@checkpoint":
            # a non-default contract was requested: the CHECKPOINT's own record must match it
            if raw.get("sampling_contract") != value:
                mismatches.append("sampling_contract: checkpoint=%r requested=%r"
                                  % (raw.get("sampling_contract"), value))
            continue
        if key == "objective_contract" and not raw.get("objective_contract"):
            # a bundle written before the objective contract was persisted cannot be compared
            continue
        if key == "constraint_contract" and not raw.get("constraint_contract"):
            continue
        if key == "validation_interval" and raw.get("validation_interval") is None:
            # older bundles recorded the interval only in the generation manifest
            manifest_interval = manifest.get("validation_interval")
            if manifest_interval is None:
                continue
            if int(manifest_interval) != int(value):
                mismatches.append("validation_interval: checkpoint=%r requested=%r"
                                  % (manifest_interval, value))
            continue
        if key in raw and raw[key] != value:
            mismatches.append("%s: checkpoint=%r requested=%r" % (key, raw[key], value))
    return mismatches


def _restore_resume_state(trainer, path, args, manifest=None) -> dict:
    """Restore the Python-side state and REJECT an incompatible checkpoint."""
    raw = json.loads(Path(str(path)).read_text())
    schema = raw.get("schema")
    if schema == RESUME_SCHEMA_V1:
        raise V2RunError(
            "this checkpoint carries the LEGACY resume bundle %r, which did not persist the "
            "environment identity (episode/reset counters, world key, realization epoch) or the "
            "RNG streams. Resuming it would rebuild the same iteration NUMBER over a DIFFERENT "
            "world sequence, so exact continuation cannot be honoured. Use the explicitly named "
            "weights-only warm start (--warm-start-weights-from) to start a NEW run from these "
            "weights, or supply a %r checkpoint." % (RESUME_SCHEMA_V1, RESUME_SCHEMA))
    if schema != RESUME_SCHEMA:
        raise V2RunError("unsupported resume-state schema %r (expected %r)"
                         % (schema, RESUME_SCHEMA))
    if manifest is None:
        manifest = _read_generation_manifest(Path(str(path)).parent)
    mismatches = _resume_mismatches(raw, args, trainer=trainer, manifest=manifest)
    manager = getattr(trainer, "auto_v2_constraint_manager", None)
    if manager is not None and raw.get("constraints"):
        manager.load_state(raw["constraints"])       # raises on a spec/schema mismatch
    elif manager is not None and raw.get("constraints") is None:
        mismatches.append("checkpoint carries no dual state but this run has constraints")
    if mismatches:
        raise V2RunError("incompatible checkpoint, refusing to resume: %s"
                         % "; ".join(mismatches))
    # F33: the environment identity and the RNG streams are applied HERE, i.e. before the loop
    # can reset, select a task, build a realization or sample. Applying them later (or not at all)
    # is what made the resumed process replay different worlds.
    _restore_env_identity(trainer, raw.get("environment"))
    _restore_rng_state(trainer, raw.get("rng"))
    # The bundle WROTE `sampler_counters` but nothing ever applied them, so a resumed process
    # reported zero support/query calls and zero unique graphs/rollout seeds for work it had
    # already done (the first-divergence trace at B1 showed them all zero). They are the run's
    # own accounting of how many trajectories and distinct identities were consumed, so they are
    # restored with the rest of the boundary.
    _restore_sampler_counters(trainer, raw.get("sampler_counters"))
    # ... and the draw cursor is written to the attribute it was READ FROM (`draw_cursor_path`),
    # never to a forwarding wrapper: a wrapper write is invisible to the sampler that advances it.
    _restore_draw_cursor(trainer, raw.get("draw_cursor"),
                         recorded_path=raw.get("draw_cursor_path"))
    trainer.last_completed_iteration = int(raw.get("last_completed_iteration", -1))
    trainer.completed_iterations = int(raw.get("completed_iterations", 0))
    trainer.next_iteration = int(raw.get("next_iteration", 0))
    trainer.started_iterations = int(raw.get("started_iterations", 0))
    trainer.outer_updates_completed = int(raw.get("outer_updates_completed", 0))
    trainer.dual_updates_completed = int(raw.get("dual_updates_completed", 0))
    trainer.validations_completed = int(raw.get("validations_completed", 0))
    if trainer.best_val_objective is None and raw.get("best_val_objective") is not None:
        trainer.best_val_objective = float(raw["best_val_objective"])
    # RESUME SEMANTICS: `--iterations` is the TOTAL target, so the loop must start at the
    # first iteration that has NOT completed, not at 0.
    trainer.start_itr = int(trainer.next_iteration)
    # INCREASING the total target is a legitimate continuation ("run 10 more"); DECREASING it
    # below work already done is not, and `trainer.start_itr` is the authority for that.
    if trainer.start_itr > int(args.iterations):
        raise V2RunError("checkpoint is already past the requested total target: next=%d "
                         "target=%d" % (trainer.start_itr, int(args.iterations)))
    stored_target = raw.get("iteration_target")
    resume_state = dict(raw)
    resume_state["target_increased"] = bool(
        stored_target is not None and int(args.iterations) > int(stored_target))
    resume_state["stored_iteration_target"] = (
        None if stored_target is None else int(stored_target))
    resume_state["recorded_draw_cursor_path"] = raw.get("draw_cursor_path")
    return resume_state


def _assert_real_run(evidence: Mapping) -> None:
    """A run that executed nothing, or never measured anything, is a FAILURE."""
    if int(evidence["iterations_completed"]) < 1:
        raise V2RunError(
            "no outer iteration COMPLETED (started=%r completed=%r last_completed=%r): an "
            "iteration counts only after EVERY required phase for that iteration has "
            "succeeded, including the scheduled validation when one is due"
            % (evidence["iterations_started"], evidence["iterations_completed"],
               evidence["last_completed_iteration"]))
    # `--iterations` is the TOTAL target, so a run that resumed below the target must not
    # present itself as finished, and a run that overshot is a counter bug, not a pass.
    if int(evidence["iterations_completed"]) != int(evidence["next_iteration"]):
        raise V2RunError("completion counter disagrees with the next iteration to execute: "
                         "completed=%r next=%r"
                         % (evidence["iterations_completed"], evidence["next_iteration"]))
    if int(evidence["iterations_completed"]) > int(evidence["iterations_requested"]):
        raise V2RunError("completed %d iterations for a target of %d: --iterations is a TOTAL "
                         "target, so overshooting is a counter defect"
                         % (evidence["iterations_completed"],
                            evidence["iterations_requested"]))
    if not evidence["sampler_counters"]:
        raise V2RunError("the sampler reported no counters: no rollout can have happened")
    last = evidence["last_rollout"]
    if not math.isfinite(float(last["makespan_s"])) or float(last["makespan_s"]) <= 0.0:
        raise V2RunError("the last rollout has no finite positive latency: %r" % (last,))
    if float(last["system_joules"]) <= 0.0:
        raise V2RunError("the last rollout reports ZERO system energy: the energy ledger did "
                         "not run on the production path")
    # THE LAGRANGIAN LOOP MUST HAVE CLOSED. Without these a run whose lambda stayed at zero,
    # whose broadcast reached nothing, or which never observed a signed constraint row could
    # still print "REAL TF RUN OK" and be promoted as a constrained-RL result.
    constraints = evidence.get("constraints") or {}
    if constraints:
        if int(constraints.get("updates", 0)) < 1:
            raise V2RunError(
                "constraints are configured but the dual never updated (updates=%r): the "
                "budget cannot have influenced training"
                % (constraints.get("updates"),))
        if int(evidence["penalty"]["signed_rows_observed"]) < 1:
            raise V2RunError("no signed constraint row reached the trainer: the dual has no "
                             "training data to ascend on")
        if int(evidence["lambda_broadcast"]["targets"]) < 1:
            raise V2RunError(
                "the constraint multipliers were broadcast to ZERO environments: the "
                "rollouts cannot have been penalised with the active lambda")
        # NOTE (F14/F33): a valid constraint need NOT generate a positive penalty in a given
        # finite run. A feasible production batch may legitimately pass with zero penalties, and
        # a process that resumed at an already-feasible boundary sees only its own increments.
        # Requiring "some episode must be penalised" therefore rejected correct runs. The
        # penalty PATH is verified instead by the controlled feasible/violating/mixed fixtures
        # (test_v2_constraint_penalty_path.py); here we only require that the machinery which
        # WOULD apply a penalty is live: an authoritative dual exists, signed rows reached it,
        # it updated at least once, and the multipliers were broadcast to every clone.
        if int(evidence["lambda_broadcast"]["targets"]) < 1:
            raise V2RunError(
                "the constraint multipliers were broadcast to ZERO environments: the "
                "rollouts cannot have been penalised with the active lambda")
        if "penalty" in evidence:
            evidence["penalty"]["note"] = (
                "zero penalised episodes is a legitimate outcome for a feasible batch; the "
                "penalty path is covered by the controlled fixture tests")


def _build_arg_parser() -> argparse.ArgumentParser:
    """The runner's argument parser (exposed so tests can assert the CLI surface)."""
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
    ap.add_argument("--resume-from", default=None,
                    help="EXACT continuation from a v2 generation (TF checkpoint + v2 resume "
                         "bundle). Requires a v2_resume_state_v2 bundle; a legacy v1 bundle is "
                         "refused because it cannot reproduce the world sequence.")
    ap.add_argument("--validation-interval", type=int, default=50,
                    help="V2-only scheduled-validation interval. Production default 50; V1 keeps "
                         "the frozen 50. A short acceptance configuration may use a smaller value, "
                         "which is recorded in the run fingerprint.")
    ap.add_argument("--checkpoint-interval", type=int, default=10,
                    help="Commit a TRANSACTIONAL full-state checkpoint generation every N "
                         "completed outer iterations (plus validation points and completion).")
    ap.add_argument("--state-trace", default=None,
                    help="F33 first-divergence trace: append per-boundary state fingerprints "
                         "(B1 saved/restored, B2 worlds, B3 observations, B6 costs) to this "
                         "JSONL so a control and a resume can be diffed boundary by boundary.")
    ap.add_argument("--warm-start-weights-from", default=None,
                    help="WEIGHTS-ONLY warm start: load the TF variables from this checkpoint and "
                         "begin a NEW run (iteration 0, fresh duals, fresh world sequence). This "
                         "is NOT equivalent continuation and is labelled as such in run.json.")
    ap.add_argument("--json", default="/tmp/v2run/v2_tf_run.json")
    ap.add_argument("--kernel-profile", default=None,
                    help="F34: write the OBSERVED per-op device placement of ONE profiled run to "
                         "this JSON path (op name, device, GPU/CPU counts, visible devices). OFF "
                         "by default; it changes no training semantics and no RNG stream.")
    ap.add_argument("--r-select", type=int, default=2)
    ap.add_argument("--s-select", type=int, default=2)
    ap.add_argument("--device", default="cpu", choices=("cpu", "gpu"),
                    help="cpu honours CUDA_VISIBLE_DEVICES from the environment; gpu requests "
                         "the device explicitly with a BOUNDED memory fraction so a co-resident "
                         "service is never displaced")
    ap.add_argument("--gpu-fraction", type=float, default=0.03,
                    help="hard cap on the process GPU memory as a fraction of the card")
    ap.add_argument("--gpu-allow-growth", dest="gpu_allow_growth", action="store_true",
                    default=False,
                    help="grow incrementally instead of pre-allocating the whole cap")
    ap.add_argument("--intra-op", type=int, default=0, help="0 = TensorFlow default")
    ap.add_argument("--inter-op", type=int, default=0, help="0 = TensorFlow default")
    ap.add_argument("--sampling-contract", default=_sampling_contract_default(),
                    help="the categorical sampling contract this run REQUESTS. The v2 default is "
                         "the stateless token-keyed draw (%s); a checkpoint recorded under a "
                         "different contract is refused on resume (F19)."
                         % _sampling_contract_default())
    return ap


def main(argv=None) -> int:
    ap = _build_arg_parser()
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
        budget_fractions={"total_energy_mec": float(args.energy_budget_fraction),
                          "ue_energy": float(args.ue_budget_fraction)},
        r_select=int(args.r_select), s_select=int(args.s_select),
        validation_interval=int(args.validation_interval),
        checkpoint_interval=int(args.checkpoint_interval))

    import tensorflow as tf

    from spec.automotive_training.v2.state_trace import StateTracer, boundary1

    tracer = StateTracer(getattr(args, "state_trace", None))

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
            # A pre-allocated HARD CAP (allow_growth False) avoids growth-triggered allocator
            # failures when another service owns most of the card. allow_growth is opt-in.
            cfg.gpu_options.per_process_gpu_memory_fraction = float(args.gpu_fraction)
            cfg.gpu_options.allow_growth = bool(args.gpu_allow_growth)
        session_cfg = cfg
        if str(args.device) == "gpu":
            # the image exports TF_FORCE_GPU_ALLOW_GROWTH=1, which TF 1.15 cannot parse and
            # which silently contradicts the fraction cap
            os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true" if args.gpu_allow_growth else "false"
        device_used = "GPU" if str(args.device) == "gpu" else "CPU"
        print("session device=%s gpu_fraction=%s intra_op=%s inter_op=%s"
              % (device_used, args.gpu_fraction, args.intra_op, args.inter_op))
        # F34: OFF unless asked. `allow_soft_placement=True` means a GPU REQUEST is not evidence
        # that a GPU EVER RAN, so the requested device is only one half of what is recorded.
        profiler = (KernelProfiler(args.kernel_profile, requested_device=str(args.device))
                    if getattr(args, "kernel_profile", None) else None)
        sess = tf.compat.v1.Session(config=session_cfg)
        if profiler is not None:
            profiler.start(sess)
        with sess.as_default():
            if args.resume_from and args.warm_start_weights_from:
                raise V2RunError("--resume-from (exact continuation) and "
                                 "--warm-start-weights-from (weights-only) are mutually exclusive")
            saver = None
            restored = False
            resume_mode = "fresh"
            if args.resume_from:
                _prefix = str(args.resume_from)
                if Path(_prefix).is_dir():
                    _prefix = _resolve_resume_ckpt(_prefix)
                saver = tf.compat.v1.train.Saver(max_to_keep=1)
                saver.restore(sess, _prefix)
                restored = True
                resume_mode = "exact_continuation"
                print("restored TF checkpoint prefix %s" % (_prefix,))
            elif args.warm_start_weights_from:
                # WEIGHTS ONLY. Deliberately NOT a resume: the iteration ledger, duals, world
                # identity and RNG stay at their fresh values, and run.json records the mode so
                # this can never be reported as equivalent continuation.
                saver = tf.compat.v1.train.Saver(max_to_keep=1)
                saver.restore(sess, str(args.warm_start_weights_from))
                resume_mode = "warm_start_weights"
                print("WARM START (weights only) from %s: new run at iteration 0, fresh duals, "
                      "fresh world sequence -- NOT equivalent continuation"
                      % (args.warm_start_weights_from,))
            else:
                sess.run(tf.compat.v1.global_variables_initializer())
            env = trainer.env
            # RESTORE BEFORE THE FIRST reset / sampling draw / rollout. A TensorFlow restore
            # alone is not a resume: it leaves the iteration ledger, the dual variables and
            # the sampler counters at their initial values, so the run would re-execute
            # completed iterations and restart dual ascent from zero.
            resume_state = None
            if args.resume_from:
                # `_restore_resume_state` reads the GENERATION MANIFEST sitting next to the
                # bundle (F19): a bundle that is intact but whose manifest advertises a different
                # sampling contract is not a continuation of this run.
                resume_state = _restore_resume_state(
                    trainer, _resume_bundle_path(args.ckpt_dir), args)
                print("resumed from %s at next_iteration=%d target=%d%s"
                      % (args.resume_from, trainer.start_itr, int(args.iterations),
                         " (target increased from %s)"
                         % (resume_state.get("stored_iteration_target"),)
                         if resume_state.get("target_increased") else ""))
            # B1: the saved/restored boundary, BEFORE any reset consumes it. On a resume this
            # is the state the first post-resume iteration will start from; on a control (no
            # --resume-from) it is the initialised state, which is how the very first boundary
            # gets pinned for the fresh run too.
            trainer.auto_state_tracer = tracer
            # The REQUESTED contract. It is applied to the object that OWNS the draw decision
            # (the inner meta sampler, never a forwarding wrapper); the sampler then reports the
            # branch it actually took, and `_sampling_contract_of` records the executed one.
            trainer.auto_sampling_contract = str(args.sampling_contract)
            _set_sampler_contract(trainer, str(args.sampling_contract))

            def _maybe_generation_checkpoint(itr, _sess=sess, _args=args):
                every = max(1, int(getattr(_args, "checkpoint_interval", 10) or 10))
                if (int(itr) + 1) % every != 0:
                    return None
                _saver = tf.compat.v1.train.Saver(max_to_keep=1)
                gen = _write_generation(_sess, _saver, trainer, _args, int(itr))
                print("committed checkpoint generation %s" % (gen.name,))
                # F34/retention: keep at least three COMPLETE recoverable generations and never
                # the newest one; the final generation is additionally protected below.
                _deleted = _prune_generations(_args.ckpt_dir, keep=3)
                if _deleted:
                    print("pruned checkpoint generations: %s" % (", ".join(_deleted),))
                return gen

            trainer.auto_generation_checkpoint = _maybe_generation_checkpoint
            if tracer.path is not None:
                tracer.record(
                    "B1_iter%d" % int(getattr(trainer, "start_itr", 0) or 0),
                    boundary1(sess, trainer))
            env.reset()
            try:
                trainer.train()
                failed = None
            except Exception as exc:            # keep the evidence we did produce
                failed = "%s: %s" % (type(exc).__name__, exc)
            elapsed = time.time() - started
            ckpt_path = None
            resume_path = None
            # F34: stop tracing FIRST so no later run is traced, then write the OBSERVED artifact.
            if profiler is not None:
                profiler.stop(sess)
            kernel_profile = (profiler.write(graph=graph) if profiler is not None else None)
            try:
                if saver is None:
                    saver = tf.compat.v1.train.Saver(max_to_keep=1)
                Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
                ckpt_path = saver.save(sess, str(Path(args.ckpt_dir) / "v2_run.ckpt"))
                resume_path = _write_resume_state(trainer, args)
                _fin = _write_generation(sess, saver, trainer, args,
                                         int(getattr(trainer, "last_completed_iteration",
                                                     int(args.iterations) - 1)))
                print("committed FINAL checkpoint generation %s" % (_fin.name,))
                # the FINAL and best-validation generations are protected from retention pruning
                _prune_generations(args.ckpt_dir, keep=3, protect=(_fin.name,))
            except Exception as exc:
                raise V2RunError("could not save a real checkpoint: %s" % exc) from exc
            evidence = _collect_evidence(trainer, env, algo, elapsed=elapsed,
                                        iterations=int(args.iterations))
            # the provenance of this run's starting weights: a fresh run, an EXACT continuation,
            # or a deliberately non-equivalent weights-only warm start
            evidence["resume_mode"] = str(resume_mode)
            evidence["resume_from"] = (str(args.resume_from) if args.resume_from
                                       else (str(args.warm_start_weights_from)
                                             if args.warm_start_weights_from else None))
            evidence["environment_identity"] = _env_identity(env, trainer)
            completed = int(evidence["iterations_completed"])
            reached_target = completed >= int(args.iterations)
            status = "completed_iterations=%d" % completed
            if failed is not None:
                status = "partial_failed"
            elif not reached_target:
                # A run that stopped short of the TOTAL target is NOT a validated checkpoint.
                status = "partial_incomplete=%d_of_%d" % (completed, int(args.iterations))
            evidence["checkpoint"] = {"path": ckpt_path, "restored_from": args.resume_from,
                                      "was_resume": bool(restored),
                                      "resume_state_path": (str(resume_path) if resume_path
                                                            else None),
                                      "status": status,
                                      "promotable": bool(failed is None and reached_target)}
            progress = Path(str(getattr(trainer, "auto_run_dir", args.ckpt_dir))) / "v2_progress.json"
            evidence["progress"] = {"path": str(progress),
                                    "exists": bool(progress.exists())}
            evidence["device"] = {"requested": str(args.device),
                                  "cuda_visible_devices": os.environ.get(
                                      "CUDA_VISIBLE_DEVICES", "<unset>")}
            # F34: the OBSERVED placement, not the requested one. `ok=False` records that the
            # profile could not capture a run; it never aborts training.
            if kernel_profile is not None:
                evidence["device"]["kernel_profile"] = {
                    "path": str(args.kernel_profile),
                    "ok": bool(kernel_profile.get("ok")),
                    "gpu_ops": (kernel_profile.get("observed") or {}).get("gpu_ops"),
                    "cpu_ops": (kernel_profile.get("observed") or {}).get("cpu_ops"),
                    "n_ops": (kernel_profile.get("observed") or {}).get("n_ops"),
                    "gpu_share": kernel_profile.get("gpu_share"),
                    "warning": kernel_profile.get("warning"),
                }
            # SAMPLER BRANCH EVIDENCE: report the fetch that ACTUALLY executed (token-keyed v2 vs
            # the legacy stateful categorical) and how many rollouts took each branch, instead of
            # restating `enable_crn=True`.
            evidence["sampling"] = _sampling_evidence(trainer, resume_state)
            evidence["config"] = {
                "seed": int(args.seed), "iterations": int(args.iterations),
                "background_dags": int(args.background),
                "background_policy": str(args.background_policy),
                "link_regime": str(args.link_regime), "mec_workers": int(args.mec_workers),
                "reliability": bool(args.reliability), "constraints": bool(args.constraints),
                "reward_mode": str(args.reward_mode),
                "budget_fractions": {"total_energy_mec": float(args.energy_budget_fraction),
                                     "ue_energy": float(args.ue_budget_fraction)},
                "r_select": int(args.r_select), "s_select": int(args.s_select),
                "device": str(args.device), "gpu_fraction": float(args.gpu_fraction),
                "gpu_allow_growth": bool(args.gpu_allow_growth),
                "intra_op": int(args.intra_op), "inter_op": int(args.inter_op),
                "obs_version": getattr(trainer, "auto_obs_version", None),
                "v2_system": getattr(trainer, "auto_v2_system", None),
                "v2_constraints": getattr(trainer, "auto_v2_constraints", None),
                "crn": getattr(trainer, "auto_crn", None),
                "validation_interval": int(getattr(args, "validation_interval", 0) or 0),
                "checkpoint_interval": int(getattr(args, "checkpoint_interval", 0) or 0),
            }
            evidence["error"] = failed
            # F31: the verdict must be FINAL before any success artifact is written. The
            # assertions used to run AFTER `run.json` had already been committed with
            # `ok = (failed is None)`, so a run that failed `_assert_real_run` still left a
            # truthful-looking `{"ok": true, "error": null}` on disk. Failures are now folded
            # into the verdict first, and the machine-readable result carries the reason.
            if failed is None:
                try:
                    _assert_real_run(evidence)
                except Exception as exc:                 # noqa: BLE001 - verdict boundary
                    failed = "%s: %s" % (type(exc).__name__, exc)
            incomplete_reason = None
            if failed is None and not reached_target:
                incomplete_reason = ("target not reached: %d of %d iterations completed"
                                     % (completed, int(args.iterations)))
            report = {"schema": "v2_tf_run_v1", "ok": failed is None and incomplete_reason is None,
                      "error": failed, "incomplete_reason": incomplete_reason,
                      "contract": _contract_record(trainer, args),
                      "evidence": evidence,
                      "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started))}
            out = Path(args.json)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")
            print(json.dumps({"ok": report["ok"],
                              "iterations_requested": evidence["iterations_requested"],
                              "iterations_started": evidence["iterations_started"],
                              "iterations_completed": evidence["iterations_completed"],
                              "last_completed_iteration": evidence["last_completed_iteration"],
                              "next_iteration": evidence["next_iteration"],
                              "outer_updates_completed": evidence["outer_updates_completed"],
                              "validations_completed": evidence["validations_completed"],
                              "penalties": evidence["penalty"],
                              "lambdas": (evidence.get("constraints") or {}).get("lambdas"),
                              "last_rollout": {k: evidence["last_rollout"][k] for k in
                                               ("makespan_s", "system_joules",
                                                "constraint_penalty", "world_id")},
                              "error": failed}, indent=2, sort_keys=True))
            if failed is not None:
                print("REAL TF RUN FAILED -> %s : %s" % (out, failed), file=sys.stderr)
                return 2
            if incomplete_reason is not None:
                # Everything asserted above passed, but `--iterations` is the TOTAL target
                # and it was not reached: this is an incomplete run with a resumable
                # boundary, not a successfully finished one.
                print("REAL TF RUN INCOMPLETE (%d of %d iterations) -> %s"
                      % (completed, int(args.iterations), out), file=sys.stderr)
                return 3
            print("REAL TF RUN OK ->", out)
            return 0


if __name__ == "__main__":
    sys.exit(main())
