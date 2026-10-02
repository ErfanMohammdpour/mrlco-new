#!/usr/bin/env python3
"""Trainer-side automotive integration: live constraint duals + the frozen
lexicographic checkpoint rule.

Two things the legacy Trainer cannot do on the automotive contract:

1. **Live duals.** The legacy constrained path observes violations inside worker env
   copies, so the trainer's controller buffer stays empty and lambda never moves.
   Here the trainer OBSERVES the violations carried in the rollout telemetry (the
   same numbers that were penalised), steps the duals once per outer iteration, and
   broadcasts the new lambda back to the env so the NEXT iteration's reward uses it.
2. **Checkpoint rule.** The saved best-validation model is chosen by the frozen
   lexicographic rule (`automotive_lexicographic_v1`), never by a single scalar, and
   the rule is recorded with its sha so a later meta-test run cannot silently change
   the selection.

The mixin is combined with `meta_trainer.Trainer` by
`automotive_primary.build_automotive_primary_stack`; nothing else is overridden.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from .automotive_constraints import CONSTRAINT_NAMES
from .automotive_primary import (
    CHECKPOINT_RULE,
    CHECKPOINT_RULE_ID,
    checkpoint_rule_sha,
    lexicographic_key,
)


def _iter_telemetry(paths):
    """Yield every energy/MC telemetry record in one rollout batch."""
    if not paths:
        return
    groups = paths.values() if isinstance(paths, Mapping) else [paths]
    for group in groups:
        records = group if isinstance(group, (list, tuple)) else [group]
        for path in records:
            if not isinstance(path, Mapping):
                continue
            telemetry = path.get("energy_telemetry")
            if isinstance(telemetry, Mapping):
                yield telemetry
            elif isinstance(telemetry, (list, tuple)):
                for record in telemetry:
                    if isinstance(record, Mapping):
                        yield record


class AutomotiveTrainingError(RuntimeError):
    """Raised when the automotive training contract is violated at runtime."""


def constraint_violations_from_telemetry(paths) -> dict:
    """Collect per-episode violations (one value per rollout) from the telemetry."""
    out: dict[str, list[float]] = {name: [] for name in CONSTRAINT_NAMES}
    if not paths:
        return {}
    for task_paths in paths.values():
        for path in task_paths:
            telemetry = path.get("energy_telemetry")
            if isinstance(telemetry, (list, tuple)) and telemetry:
                telemetry = telemetry[0] if isinstance(telemetry[0], Mapping) else telemetry
            if not isinstance(telemetry, Mapping):
                continue
            for name in CONSTRAINT_NAMES:
                value = telemetry.get("violation/%s" % name)
                if value is not None:
                    out[name].append(float(value))
    return {k: v for k, v in out.items() if v}


class AutomotiveTrainerMixin(object):
    """Mixin for `meta_trainer.Trainer` (the primary validation/checkpoint chain)."""

    # -- constraint observer -------------------------------------------------
    def constraint_observer(self, samples_data=None, task_specs=None, paths=None) -> dict:
        """Observe this iteration's violations on the TRAINING controller.

        Called once per outer iteration before the dual step. Violations come from the
        rollout telemetry (the exact values that were penalised), never from a cloned
        worker controller.
        """
        controller = getattr(self, "auto_controller", None)
        if controller is None:
            return {}
        costs = constraint_violations_from_telemetry(paths)
        if costs:
            controller.observe(costs, split="meta_train")
        # penalty statistics straight from the rollout telemetry (the numbers the env
        # actually used), so a run can prove the Lagrangian feedback reached the reward
        penalties = [float(r.get("constraint_penalty", 0.0) or 0.0)
                     for r in _iter_telemetry(paths)]
        self.auto_penalty_episodes = int(getattr(self, "auto_penalty_episodes", 0)) + len(penalties)
        self.auto_penalty_episodes_nonzero = int(getattr(self, "auto_penalty_episodes_nonzero", 0)) + sum(
            1 for p in penalties if abs(p) > 1e-12)
        self.auto_penalty_sum = float(getattr(self, "auto_penalty_sum", 0.0)) + float(sum(penalties))
        self.auto_penalty_steps = int(getattr(self, "auto_penalty_steps", 0)) + (
            1 if any(abs(p) > 1e-12 for p in penalties) else 0)
        return costs

    def broadcast_constraint_lambdas(self) -> None:
        """Push the current multipliers into EVERY env a rollout can run on.

        The sampler's vectorised executor creates `copy.deepcopy(env)` clones, so
        updating only the original env leaves the clones at lambda=0 and the policy
        never receives the penalty. Returns the number of environments updated.
        """
        controller = getattr(self, "auto_controller", None)
        if controller is None:
            return 0
        lambdas = controller.lambdas
        updated = 0
        env = getattr(self, "auto_env", None)
        if env is not None and hasattr(env, "set_constraint_lambdas"):
            env.set_constraint_lambdas(lambdas)
            updated += 1
        sampler = getattr(self, "meta_sampler", None) or getattr(self, "sampler", None)
        executor = getattr(sampler, "vec_env", None)
        if executor is not None and hasattr(executor, "set_constraint_lambdas"):
            updated += int(executor.set_constraint_lambdas(lambdas))
        self.auto_lambda_broadcast_targets = updated
        self.auto_lambda_broadcast_values = dict(lambdas) if hasattr(lambdas, "items") else list(lambdas)
        if updated == 0:
            raise AutomotiveTrainingError(
                "no environment accepted the constraint multipliers; the Lagrangian "
                "feedback would be silently dropped")
        return updated

    # -- lexicographic validation -------------------------------------------
    def _run_validation(self, itr):
        from utils import logger

        evaluator = self.held_out_evaluator
        if evaluator is None:
            raise RuntimeError("automotive validation requires a held-out evaluator")
        k0 = evaluator.evaluate_all(k_steps=0)
        k3 = evaluator.evaluate_all(k_steps=3)
        self.algo.sync_task_policies_from_core()

        logger.logkv("validation/objective_discounted_return_k0",
                     float(k0.get("query_discounted_return", 0.0)))
        logger.logkv("validation/objective_discounted_return_k3",
                     float(k3.get("query_discounted_return", 0.0)))
        logger.logkv("validation/query_mean_latency_seconds_k3",
                     float(k3.get("query_mean_latency_seconds", float("nan"))))
        # k=0 (not adapted) is logged as a first-class metric; the previous run only
        # logged its return, so the k0 latency had to be recovered as -return.
        logger.logkv("validation/query_mean_latency_seconds_k0",
                     float(k0.get("query_mean_latency_seconds", float("nan"))))
        logger.logkv("validation/rollouts_k0", float(k0.get("rollouts", 0)))
        logger.logkv("validation/query_graphs", float(k3.get("query_graphs", 0)))
        for name in ("graph_hard_violation_rate",
                     "graph_high_tardiness_incidence_rate",
                     "graph_medium_tardiness_incidence_rate",
                     "high_task_tardiness_task_rate",
                     "medium_task_tardiness_task_rate",
                     "firm_task_miss_rate", "firm_task_miss_count",
                     "n_high_tasks", "n_medium_tasks",
                     "n_tardy_high_tasks", "n_tardy_medium_tasks",
                     "mode_switch_rate", "hi_mode_rate", "mc_policy_violation_count"):
            logger.logkv("validation/%s_k3" % name, float(k3.get(name, 0.0)))
        # correctness telemetry (Gate B/C/D): these must hold on every validation
        logger.logkv("correctness/core_scratch_sync_count", float(getattr(evaluator, "sync_count", 0)))
        logger.logkv("correctness/core_scratch_sync_max_abs_diff",
                     float(getattr(evaluator, "last_sync_max_abs_diff", float("nan"))))
        logger.logkv("correctness/core_unchanged_after_adaptation",
                     1.0 if getattr(evaluator, "core_unchanged_after_adaptation", False) else 0.0)
        logger.logkv("correctness/lambda_broadcast_targets",
                     float(getattr(self, "auto_lambda_broadcast_targets", 0)))
        logger.logkv("correctness/penalty_steps",
                     float(getattr(self, "auto_penalty_steps", 0)))
        self.auto_correctness = {
            "core_scratch_sync_count": int(getattr(evaluator, "sync_count", 0)),
            "core_scratch_sync_max_abs_diff": float(getattr(evaluator, "last_sync_max_abs_diff", float("nan"))),
            "core_unchanged_after_adaptation": bool(getattr(evaluator, "core_unchanged_after_adaptation", False)),
            "adaptation_ppo_constructions": int(getattr(evaluator, "adaptation_ppo_constructions", 0)),
            "lambda_broadcast_targets": int(getattr(self, "auto_lambda_broadcast_targets", 0)),
            "penalty_steps": int(getattr(self, "auto_penalty_steps", 0)),
            "dual_batch_sizes_after_reset": list(getattr(self, "auto_dual_batch_sizes", [])),
        }
        logger.logkv("checkpoint_selection_metric", "validation/" + CHECKPOINT_RULE_ID)
        logger.logkv("checkpoint_selection_rule_id", CHECKPOINT_RULE_ID)
        logger.logkv("checkpoint_selection_rule_sha", checkpoint_rule_sha())

        self.auto_last_validation = dict(k3)
        key = lexicographic_key(k3)
        logger.logkv("checkpoint_selection_key_hash",
                     float(abs(hash(key)) % 1000000007))
        logger.logkv("checkpoint_selection_latency_s",
                     float(k3.get("query_mean_latency_seconds", float("nan"))))
        save = self.best_selection_key is None or key < tuple(self.best_selection_key)
        logger.logkv("checkpoint_is_best_val", 1.0 if save else 0.0)
        if save:
            self.best_selection_key = tuple(key)
            self.best_val_objective = float(k3.get("query_discounted_return", 0.0))
            path = self._ckpt_path("meta_model_best_val.ckpt")
            self.policy.core_policy.save_variables(save_path=path)
            sidecar = {
                "schema": "best_val_metric_v1",
                "contract": "automotive_checkpoint_rule_v1",
                "rule_id": CHECKPOINT_RULE_ID,
                "rule_steps": list(CHECKPOINT_RULE),
                "rule_sha": checkpoint_rule_sha(),
                "selection_key": [str(x) for x in key],
                "metric_name": "validation/" + CHECKPOINT_RULE_ID,
                "csv_key": "validation/query_mean_latency_seconds_k3",
                "value": float(k3.get("query_mean_latency_seconds", float("nan"))),
                "itr": int(itr),
                "higher_is_better": False,
                "reward_mode": "latency_only",
                "discount": 0.99,
                "query_discounted_return": float(k3.get("query_discounted_return", 0.0)),
                "graph_hard_violation_rate": float(k3.get("graph_hard_violation_rate", 0.0)),
                "mc_policy_violation_count": float(k3.get("mc_policy_violation_count", 0.0)),
                "energy_constraint": "not_configured",
            }
            with open(self._ckpt_path("meta_model_best_val.metric.json"), "w") as fh:
                json.dump(sidecar, fh, indent=2, sort_keys=True)
                fh.write("\n")
        return k0, k3

def _mixin_train(self):
    raise RuntimeError("placeholder")


class _AutomotiveReportMixin(object):
    """Writes the smoke/fingerprint artifact after `train()` returns."""

    def train(self):
        import json as _json
        import time as _time

        started = _time.time()
        result = super().train()
        wall = _time.time() - started
        self.auto_wall_seconds = float(wall)
        self.write_auto_report()
        return result

    def write_auto_report(self) -> dict:
        import json as _json
        from pathlib import Path as _Path

        from .automotive_fingerprint import write_fingerprint
        from .automotive_primary import (
            CHECKPOINT_RULE,
            CHECKPOINT_RULE_ID,
            OBS_VERSION,
            SUPPORT_TRAJECTORIES_PER_META_TASK,
            TOKENS_PER_TRAJECTORY,
            checkpoint_rule_sha,
        )

        run_dir = _Path(getattr(self, "auto_run_dir", self.ckpt_dir))
        sampler = getattr(self, "auto_sampler", None)
        controller = getattr(self, "auto_controller", None)
        report = {
            "method_id": str(getattr(self, "auto_method_id",
                                      "margo_automotive_mc_v1_%s" % getattr(self, "auto_run_kind", "primary"))),
            "dataset": "MARGO-AUTOMOTIVE-MC-v1",
            "obs_version": OBS_VERSION,
            "scheduler_axes": dict(getattr(self, "auto_axes", {})),
            "legacy_mixed_axes": dict(getattr(self, "auto_legacy_axes", {})),
            "graph_specific_resources": True,
            "scheduler_config_sha256_first_graph": getattr(self, "auto_scheduler_fingerprint", None),
            "executed_outer_iterations": int(getattr(self, "n_itr", 0)),
            "run_kind": str(getattr(self, "auto_run_kind", "primary")),
            "wall_seconds": float(getattr(self, "auto_wall_seconds", 0.0)),
            "sampler_counters": dict(getattr(sampler, "counters", {}) if sampler else {}),
            "sampler_budget": {
                "support_trajectories_per_meta_task": SUPPORT_TRAJECTORIES_PER_META_TASK,
                "tokens_per_trajectory": TOKENS_PER_TRAJECTORY,
                "meta_batch_size": 10,
                "total_samples_tokens": (getattr(sampler.sampler, "total_samples", None)
                                         if sampler else None),
                "max_path_length_role": "episode_cap_only",
            },
            "constraints": {
                "lambdas": (controller.lambdas if controller else {}),
                "state": (controller.as_dict() if controller else {}),
                "energy_constraint": "not_configured",
            },
            "checkpoint_rule": {"rule_id": CHECKPOINT_RULE_ID, "steps": list(CHECKPOINT_RULE),
                                "sha": checkpoint_rule_sha()},
            "meta_test_access_count": None,
            "deadline_mask": "off",
            "queue_blind_mask_used": False,
            "objective_mode": "latency_only",
            "last_validation": dict(getattr(self, "auto_last_validation", {}) or {}),
            "mc_fixture": dict(getattr(self, "auto_mc_fixture", {}) or {}),
            "peak_gpu_bytes": _peak_gpu_bytes(),
            "query_graph_count": int(getattr(self, "auto_query_graph_count",
                                             len(getattr(getattr(self, "held_out_evaluator", None),
                                                         "query_graphs", []) or []))),
            "penalty_feedback": {
                "episodes": int(getattr(self, "auto_penalty_episodes", 0)),
                "episodes_with_nonzero_penalty": int(getattr(self, "auto_penalty_episodes_nonzero", 0)),
                "penalty_sum": float(getattr(self, "auto_penalty_sum", 0.0)),
                "iteration_with_nonzero_penalty": int(getattr(self, "auto_penalty_steps", 0)),
            },
            "lambda_broadcast_targets": int(getattr(self, "auto_lambda_broadcast_targets", 0)),
            "dual_batch_size_at_end": {
                name: (controller.batch_size(name) if controller is not None
                       and hasattr(controller, "batch_size") else None)
                for name in _controller_names(controller)},
            "correctness": dict(getattr(self, "auto_correctness", {}) or {}),
            "sampler_counter_contract": {
                "support_calls": int((getattr(sampler, "counters", {}) or {}).get("support_calls", 0)),
                "query_calls": int((getattr(sampler, "counters", {}) or {}).get("query_calls", 0)),
                "balanced": bool((getattr(sampler, "counters", {}) or {}).get("counter_role_contract_ok", False)),
                "expected_calls_each": int(getattr(self, "n_itr", 0)),
            },
        }
        try:
            from .automotive_loader import META_TEST_GUARD

            report["meta_test_access_count"] = META_TEST_GUARD.meta_test_access_count()
        except Exception:
            pass
        run_dir.mkdir(parents=True, exist_ok=True)
        try:
            fp = write_fingerprint(
                run_dir, obs_version=OBS_VERSION,
                scheduler_axes=dict(getattr(self, "auto_axes", {})),
                checkpoint_rule_sha=checkpoint_rule_sha(),
                sampler_budget=report["sampler_budget"],
                run_kind=str(getattr(self, "auto_run_kind", "primary")),
                outer_iterations=int(getattr(self, "n_itr", 0)))
            report["training_fingerprint"] = fp["fingerprint"]
            report["training_fingerprint_parts"] = {
                k: fp["parts"][k] for k in ("git_sha", "code_dirty", "dataset_manifest_sha",
                                            "graphs_sha", "splits_sha",
                                            "calibration_report_sha", "obs_version",
                                            "scheduler_axes", "checkpoint_rule_sha",
                                            "run_kind", "outer_iterations")}
        except Exception as exc:  # pragma: no cover - report only
            report["training_fingerprint_error"] = repr(exc)
        (run_dir / "automotive_smoke_report.json").write_text(
            _json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")
        return report


def _controller_names(controller):
    """Channel names of the raw controller or of its trainer-facing adapter."""
    if controller is None:
        return list(CONSTRAINT_NAMES)
    names = getattr(controller, "names", None)
    if names:
        return [str(n) for n in names]
    specs = getattr(controller, "specs", None)
    if specs:
        return [str(n) for n in specs.keys()]
    return list(CONSTRAINT_NAMES)


def _peak_gpu_bytes():
    """Peak GPU bytes, or None on a CPU-only run.

    `MaxBytesInUse` only has a GPU kernel, so asking for it on a CPU session raises
    `InvalidArgumentError: No OpKernel was registered ... MaxBytesInUse`. The CPU
    smoke and the TF-gated tests must therefore short-circuit before touching it.
    """
    try:
        import tensorflow as tf

        try:
            gpus = tf.config.experimental.list_physical_devices("GPU")
        except Exception:
            gpus = []
        if not gpus:
            return None
        with tf.compat.v1.Session() as sess:
            return int(sess.run(tf.contrib.memory_stats.MaxBytesInUse()))
    except Exception:
        return None


AutomotiveTrainerMixin = type(
    "AutomotiveTrainerMixin",
    (globals()["AutomotiveTrainerMixin"], _AutomotiveReportMixin),
    {},
)
