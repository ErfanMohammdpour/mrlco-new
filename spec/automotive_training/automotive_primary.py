#!/usr/bin/env python3
"""Automotive primary stack: the SAME policy / sampler / MRLCO / Trainer chain as
`meta_trainer.build_frozen_primary_stack`, but wired to the frozen automotive dataset.

What changes (and only this):

* graph source: frozen `MARGO-AUTOMOTIVE-MC-v1` graphs, one graph = one meta-task;
* scheduler: per-graph CO-PHYSICAL ResourceConfig (graph rates), no legacy rebuild;
* observation: `automotive_mc_obs_v1` (v3 + 9 mixed-criticality columns);
* MC runtime: frozen `execution_uncertainty_v1` + the M7 mode/drop policy;
* sampler budget: explicit in TRAJECTORY COUNT (`support_trajectories_per_task` x
  `tokens_per_trajectory`), never inferred from `max_path_length`;
* validation: the frozen validation split (20 support / 40 query), k=0 and k=3,
  with a lexicographic checkpoint rule frozen before any meta-test evaluation;
* constraints: the three truthful channels with working Lagrangian duals whose
  violations are observed from the rollout telemetry (the legacy dual path was inert
  because worker envs own cloned controllers).

What does NOT change: the autoregressive 20-token plan, actions {0=UE,1=MEC,2=HELPER},
the Graph2Seq encoder, the LSTM decoder + attention, the inner PPO schedule
(lr 5e-4, exactly 3 applies, clip 0.2, value clip 0.2, vf 0.5, grad norm 0.5,
gamma 0.99, lambda 0.95, entropy 0), meta-batch 10, and the first-order outer update.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .automotive_constraints import (
    CONSTRAINT_NAMES,
    AutomotiveDualController,
    default_constraint_specs,
)
from .automotive_env import AutomotiveEnv
from .automotive_resources import (
    CO_PHYSICAL_AXES,
    LEGACY_MIXED_AXES,
    config_fingerprint,
)

OBS_VERSION = "automotive_mc_obs_v1"
DATASET_ID = "MARGO-AUTOMOTIVE-MC-v1"
SUPPORT_TRAJECTORIES_PER_META_TASK = 20
TOKENS_PER_TRAJECTORY = 20
META_BATCH_SIZE = 10
VALIDATION_SUPPORT_COUNT = 20
VALIDATION_QUERY_COUNT = 40
CHECKPOINT_RULE_ID = "automotive_lexicographic_v1"
DUAL_LR = 0.05

CHECKPOINT_RULE = (
    "1 zero graph hard-deadline violation rate",
    "2 zero MC policy violations (HIGH drop/degrade must always be zero)",
    "3 lowest HIGH task tardiness violation",
    "4 lowest MEDIUM firm tardiness violation",
    "5 energy feasibility only if configured (energy is not_configured here)",
    "6 minimum validation latency among the remaining feasible checkpoints",
)


class AutomotivePrimaryError(RuntimeError):
    """Refused. Never silently repaired."""


def checkpoint_rule_sha() -> str:
    material = {
        "rule_id": CHECKPOINT_RULE_ID,
        "steps": list(CHECKPOINT_RULE),
        "energy_constraint": "not_configured",
        "dataset": DATASET_ID,
        "obs_version": OBS_VERSION,
    }
    return hashlib.sha256(json.dumps(material, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


class AutomotiveResourceCluster(object):
    """Minimal cluster handle for the automotive env.

    The automotive env never resolves a config from the cluster (it uses each graph's
    own co-physical config), so this carries only what the sampler/logger read.
    """

    def __init__(self, energy_config: Mapping[str, Any] | None = None,
                 constraint_controller: Any = None):
        self.use_energy = True
        self.energy_config = dict(energy_config or {})
        self.scheduler_config = None
        self.constraint_controller = constraint_controller

    def reset(self) -> None:
        return None


class _ConstraintSpecProxy(object):
    """Minimal legacy-shaped spec so the frozen Trainer's dual block runs unchanged."""

    def __init__(self, names):
        self.enabled = bool(names)
        self.mode = "lagrangian"
        self.active_names = list(names)


class AutomotiveDualAdapter(object):
    """Expose `AutomotiveDualController` through the interface the Trainer expects.

    The frozen Trainer calls `controller.spec.enabled`, `controller.names`,
    `controller.lambdas` and `dual_step()` (expecting a flat float dict). The real
    controller keeps its own richer API; this adapter only translates.
    """

    def __init__(self, controller, names):
        self._controller = controller
        self.names = list(names)
        self.spec = _ConstraintSpecProxy(self.names)

    @property
    def lambdas(self):
        return [float(self._controller.lambdas.get(name, 0.0)) for name in self.names]

    def observe(self, costs, split="meta_train"):
        return self._controller.observe(costs, split=split)

    def reset_batch(self):
        return self._controller.reset_batch()

    def as_dict(self):
        return self._controller.as_dict()

    def dual_step(self):
        diag = self._controller.dual_step() or {}
        means = diag.get("means") or {}
        deltas = diag.get("deltas") or {}

        def _by_name(container, name, index):
            if isinstance(container, Mapping):
                value = container.get(name, 0.0)
            elif isinstance(container, (list, tuple)) and index < len(container):
                value = container[index]
            else:
                value = 0.0
            try:
                return float(value)
            except (TypeError, ValueError):
                return 0.0

        out = {}
        for index, name in enumerate(self.names):
            out["constraint/mean_violation_%s" % name] = _by_name(means, name, index)
            out["constraint/delta_%s" % name] = _by_name(deltas, name, index)
        updated = diag.get("updated", 0)
        out["constraint/updates"] = float(len(updated)) if isinstance(updated, (list, tuple)) else _by_name(
            {"updated": updated}, "updated", 0)
        skipped = diag.get("skipped", 0)
        out["constraint/skipped"] = float(len(skipped)) if isinstance(skipped, (list, tuple)) else _by_name(
            {"skipped": skipped}, "skipped", 0)
        return out


class AutomotiveMetaSampler(object):
    """Thin wrapper adding an EXPLICIT trajectory budget + exact counters."""

    def __init__(self, sampler, *, meta_batch_size: int, trajectories_per_meta_task: int,
                 tokens_per_trajectory: int):
        self.sampler = sampler
        self.meta_batch_size = int(meta_batch_size)
        self.trajectories_per_meta_task = int(trajectories_per_meta_task)
        self.tokens_per_trajectory = int(tokens_per_trajectory)
        # explicit budget: tokens, derived from the trajectory count (never from a
        # max-path-length constant)
        self.sampler.total_samples = (self.meta_batch_size * self.trajectories_per_meta_task
                                      * self.tokens_per_trajectory)
        self.counters = {"support_calls": 0, "query_calls": 0,
                         "selected_meta_tasks": 0, "support_trajectories": 0,
                         "support_tokens": 0, "query_trajectories": 0, "query_tokens": 0,
                         "unique_graph_ids": 0, "unique_rollout_seeds": 0}

    def __getattr__(self, item):
        return getattr(self.sampler, item)

    def update_tasks(self):
        tasks = self.sampler.update_tasks()
        self.counters["selected_meta_tasks"] += len(tasks)
        return tasks

    def obtain_samples(self, log=False, log_prefix=""):
        paths = self.sampler.obtain_samples(log=log, log_prefix=log_prefix)
        role = "support" if self.counters["support_calls"] == 0 else "query"
        self.counters[f"{role}_calls"] += 1
        trajectories = tokens = 0
        graph_ids, seeds = set(), set()
        for task_paths in paths.values():
            for path in task_paths:
                trajectories += 1
                length = int(np.asarray(path["rewards"]).shape[-1])
                tokens += length
                if length != self.tokens_per_trajectory:
                    raise AutomotivePrimaryError(
                        "trajectory token count %d != %d (budget is not explicit)"
                        % (length, self.tokens_per_trajectory))
                telemetry = path.get("energy_telemetry")
                if isinstance(telemetry, Mapping):
                    candidates = [telemetry]
                elif isinstance(telemetry, (list, tuple)) and telemetry:
                    candidates = [t for t in telemetry if isinstance(t, Mapping)]
                else:
                    candidates = []
                for record in candidates:
                    seeds.add(int(record.get("rollout_seed", -1)))
                    graph_ids.add(int(record.get("graph_index", -1)))
        self.counters[f"{role}_trajectories"] += trajectories
        self.counters[f"{role}_tokens"] += tokens
        self.counters["unique_graph_ids"] = len(graph_ids)
        self.counters["unique_rollout_seeds"] = len(seeds)
        return paths


def _iter_paths(paths):
    """Accept both the meta-sampler dict-of-lists and the plain sampler list."""
    if isinstance(paths, Mapping):
        for task_paths in paths.values():
            for path in task_paths:
                yield path
    else:
        for path in paths:
            yield path


def metrics_from_paths(paths) -> dict:
    """Aggregate the telemetry of one rollout batch into validation metrics."""
    makespans, violations, modes, switches = [], [], [], []
    high_tard, med_tard, firm_miss, mc_violations, dropped = [], [], [], [], []
    for path in _iter_paths(paths):
        if True:
            telemetry = path.get("energy_telemetry")
            if isinstance(telemetry, (list, tuple)) and telemetry:
                telemetry = telemetry[0] if isinstance(telemetry[0], Mapping) else telemetry
            if not isinstance(telemetry, Mapping):
                continue
            makespans.append(float(telemetry.get("makespan_s", np.nan)))
            violations.append(float(telemetry.get("violation/C_GRAPH_HARD_DEADLINE", 0.0)))
            high_tard.append(float(telemetry.get("violation/C_HI_TASK_TARDINESS", 0.0)))
            med_tard.append(float(telemetry.get("violation/C_MED_TASK_TARDINESS", 0.0)))
            firm_miss.append(float(telemetry.get("firm_miss_count", 0)))
            modes.append(str(telemetry.get("mode", "NA")))
            switches.append(float(telemetry.get("mode_switch_count", 0)))
            dropped.append(float(telemetry.get("dropped_task_count", 0)))
            if not bool(telemetry.get("high_preserved", 1.0)):
                mc_violations.append(1.0)
    n = max(1, len(makespans))
    return {
        "query_mean_latency_seconds": float(np.mean(makespans)) if makespans else float("nan"),
        "query_discounted_return": float(-np.mean(makespans)) if makespans else float("nan"),
        "graph_hard_violation_rate": float(np.mean([v > 0.0 for v in violations])) if violations else 0.0,
        "high_task_tardiness_rate": float(np.mean([v > 0.0 for v in high_tard])) if high_tard else 0.0,
        "medium_task_tardiness_rate": float(np.mean([v > 0.0 for v in med_tard])) if med_tard else 0.0,
        "firm_task_miss_rate": float(np.sum(firm_miss) / (20.0 * n)),
        "mean_graph_violation_s": float(np.mean(violations)) if violations else 0.0,
        "mean_high_tardiness_s": float(np.mean(high_tard)) if high_tard else 0.0,
        "mean_medium_tardiness_s": float(np.mean(med_tard)) if med_tard else 0.0,
        "mode_switch_rate": float(np.mean([s > 0.0 for s in switches])) if switches else 0.0,
        "mean_mode_switch_count": float(np.mean(switches)) if switches else 0.0,
        "hi_mode_rate": float(np.mean([m == "HI" for m in modes])) if modes else 0.0,
        "mc_policy_violation_count": float(np.sum(mc_violations)),
        "mean_dropped_task_count": float(np.mean(dropped)) if dropped else 0.0,
        "rollouts": int(n),
    }


class AutomotiveHeldOutEvaluator(object):
    """Validation on the frozen validation split (20 support / 40 query), k=0 and k=3."""

    def __init__(self, *, support_graphs, query_graphs, policy, source_policy,
                 ppo_batch_size: int = 20, enable_query_rollout: bool = True):
        self.support_graphs = list(support_graphs)
        self.query_graphs = list(query_graphs)
        self.policy = policy
        self.source_policy = source_policy
        self.ppo_batch_size = int(ppo_batch_size)
        self.enable_query_rollout = bool(enable_query_rollout)

    def _env(self, graphs, slots: int, seed: int) -> AutomotiveEnv:
        return AutomotiveEnv(graphs, AutomotiveResourceCluster(), role="validation",
                             slots_per_task=slots, base_seed=seed, mc_enabled=True)

    def _rollout(self, graphs, slots: int, policy, seed: int, *, adapt_steps: int):
        """One rollout batch through the PLAIN policy path (rank-3 observations).

        The validation policy is a `Seq2SeqPolicy`, which consumes a single
        (n_graphs, 20, packed) batch. `Seq2SeqSampler` drives ONE env directly, so the
        validation env is built in single-distribution mode with every graph in the
        slot axis — exactly the legacy held-out layout.
        """
        from samplers.seq2seq_sampler import Seq2SeqSampler
        from samplers.seq2seq_sampler_process import Seq2SeSamplerProcessor
        from baselines.vf_baseline import ValueFunctionBaseline
        from meta_algos.ppo_offloading import PPO

        env = AutomotiveEnv(graphs, AutomotiveResourceCluster(), role="validation",
                            single_dist=True, slots_per_task=len(graphs),
                            base_seed=seed, mc_enabled=True)
        env.set_task({"dist_index": 0,
                      "graph_indices": np.arange(len(graphs), dtype=np.int32)})
        sampler = Seq2SeqSampler(env, policy, rollouts_per_meta_task=1,
                                 max_path_length=TOKENS_PER_TRAJECTORY, parallel=False)
        sampler.total_samples = len(graphs) * TOKENS_PER_TRAJECTORY
        processor = Seq2SeSamplerProcessor(baseline=ValueFunctionBaseline(),
                                           discount=0.99, gae_lambda=0.95,
                                           normalize_adv=True, positive_adv=False)
        paths = sampler.obtain_samples()
        if adapt_steps > 0:
            processed = processor.process_samples(paths)
            samples = processed[0] if isinstance(processed, tuple) else processed
            ppo = PPO(policy=policy, meta_sampler=sampler,
                      meta_sampler_process=processor, lr=5e-4,
                      num_inner_grad_steps=3, clip_value=0.2, max_grad_norm=0.5,
                      rng=np.random.RandomState(seed + 1))
            ppo.UpdatePPOTarget(samples, batch_size=min(20, len(graphs)),
                                k_steps=int(adapt_steps))
            env.reset()
            paths = sampler.obtain_samples()
        return paths

    def evaluate_all(self, k_steps: int) -> dict:
        if k_steps not in (0, 3):
            raise AutomotivePrimaryError("k_steps must be 0 or 3, got %r" % (k_steps,))
        if k_steps == 0:
            paths = self._rollout(self.query_graphs, 1, self.policy, seed=101,
                                  adapt_steps=0)
        else:
            # scratch adaptation on the validation SUPPORT only, then a DISJOINT
            # query rollout (different graphs, different MC realizations)
            assign = getattr(self.policy, "assign_trainable", None)
            if assign is not None:
                assign(self.source_policy, self.policy)
            self._rollout(self.support_graphs, 1, self.policy, seed=202, adapt_steps=3)
            paths = self._rollout(self.query_graphs, 1, self.policy, seed=303,
                                  adapt_steps=0)
        metrics = metrics_from_paths(paths)
        metrics["k_steps"] = int(k_steps)
        metrics["support_graphs"] = len(self.support_graphs)
        metrics["query_graphs"] = len(self.query_graphs)
        metrics["split"] = "validation"
        return metrics


def lexicographic_key(metrics: Mapping[str, Any]) -> tuple:
    """Frozen checkpoint rule: lower is better, evaluated in this exact order."""
    return (
        1 if float(metrics.get("graph_hard_violation_rate", 1.0)) > 0 else 0,
        1 if float(metrics.get("mc_policy_violation_count", 1.0)) > 0 else 0,
        float(metrics.get("mean_high_tardiness_s", 0.0)),
        float(metrics.get("mean_medium_tardiness_s", 0.0)),
        float(metrics.get("query_mean_latency_seconds", float("inf"))),
    )


def build_automotive_primary_stack(*, seed: int, n_itr: int, ckpt_dir: str,
                                   dataset_dir: str | None = None,
                                   reward_mode: str = "latency_only",
                                   use_energy: bool = True,
                                   constraints: Sequence[str] = CONSTRAINT_NAMES,
                                   constraint_dual_lr: float = DUAL_LR,
                                   obs_version: str = OBS_VERSION,
                                   energy_config: Mapping[str, Any] | None = None,
                                   meta_batch_size: int = META_BATCH_SIZE,
                                   support_trajectories: int = SUPPORT_TRAJECTORIES_PER_META_TASK,
                                   run_kind: str = "primary"):
    """Build (Trainer, MRLCO) on the frozen automotive dataset."""
    if obs_version != OBS_VERSION:
        raise AutomotivePrimaryError("obs_version must be %r" % OBS_VERSION)
    # MRLCO enforces the frozen primary budgets at construction (meta_batch 10,
    # support_trajectories 20, ppo_batch 20, exactly 3 inner applies). A "tiny
    # plumbing" stack is therefore impossible without changing the frozen algorithm,
    # which the integration contract forbids: the CPU plumbing smoke uses the SAME
    # canonical structural budgets, and only the hardware differs.
    if int(meta_batch_size) != META_BATCH_SIZE:
        raise AutomotivePrimaryError(
            "MRLCO requires meta_batch_size == %d (frozen); a reduced-budget stack "
            "would change the primary algorithm" % META_BATCH_SIZE)
    if int(support_trajectories) != SUPPORT_TRAJECTORIES_PER_META_TASK:
        raise AutomotivePrimaryError(
            "MRLCO requires support_trajectories == %d (frozen)"
            % SUPPORT_TRAJECTORIES_PER_META_TASK)
    os.environ["MARGO_OBS_VERSION"] = OBS_VERSION
    from env.mec_offloaing_envs.scheduler import encoder_obs

    encoder_obs.set_obs_version(OBS_VERSION)

    import tensorflow as tf
    from policies.meta_seq2seq_policy import MetaSeq2SeqPolicy
    from policies.meta_seq2seq_policy import Seq2SeqPolicy
    from samplers.seq2seq_meta_sampler import Seq2SeqMetaSampler
    from samplers.seq2seq_meta_sampler_process import Seq2SeqMetaSamplerProcessor
    from baselines.vf_baseline import ValueFunctionBaseline
    from meta_algos.MRLCO import MRLCO
    from meta_algos.held_out_eval import HeldOutQueryEvaluator  # noqa: F401  (interface ref)

    from .automotive_loader import (
        DATASET_ID as LOADER_DATASET_ID,
        META_TEST_GUARD,
        load_dataset,
    )

    if LOADER_DATASET_ID != DATASET_ID:
        raise AutomotivePrimaryError("loader/dataset id mismatch")
    seed = int(seed)
    np.random.seed(seed)
    tf.compat.v1.set_random_seed(seed)

    dataset = load_dataset(dataset_dir) if dataset_dir else load_dataset()
    META_TEST_GUARD.assert_unopened()
    train_graphs = dataset.meta_train()
    val_support = dataset.validation_support()
    val_query = dataset.validation_query()
    if len(val_support) != VALIDATION_SUPPORT_COUNT or len(val_query) != VALIDATION_QUERY_COUNT:
        raise AutomotivePrimaryError("the frozen validation split must be 20 support / 40 query")

    spec_map = default_constraint_specs()
    # the automotive contract enables the three truthful channels by default; energy
    # stays not_configured. Passing an explicit list narrows it.
    requested = list(constraints) if constraints else list(CONSTRAINT_NAMES)
    enabled = [name for name in requested if name in spec_map]
    controller = AutomotiveDualController(specs={k: spec_map[k] for k in enabled},
                                          dual_lr=float(constraint_dual_lr))
    adapter = AutomotiveDualAdapter(controller, enabled)
    cluster = AutomotiveResourceCluster(energy_config=dict(energy_config or {}),
                                        constraint_controller=adapter)
    env = AutomotiveEnv(train_graphs, cluster, role="meta_train",
                        slots_per_task=int(support_trajectories),
                        base_seed=seed, mc_enabled=True,
                        constraint_lambdas=controller.lambdas)
    env.constraint_controller = adapter
    env.constraint_spec = None

    _greedy_plan, greedy_finish, _greedy_energy = env.greedy_solution()
    baseline = ValueFunctionBaseline()
    meta_policy = MetaSeq2SeqPolicy(meta_batch_size=int(meta_batch_size),
                                    obs_dim=env.input_dim, encoder_units=128,
                                    decoder_units=128, vocab_size=3)
    sampler = Seq2SeqMetaSampler(env, meta_policy, rollouts_per_meta_task=1,
                                 meta_batch_size=int(meta_batch_size),
                                 max_path_length=TOKENS_PER_TRAJECTORY, parallel=False)
    budgeted = AutomotiveMetaSampler(
        sampler, meta_batch_size=int(meta_batch_size),
        trajectories_per_meta_task=int(support_trajectories),
        tokens_per_trajectory=TOKENS_PER_TRAJECTORY)
    processor = Seq2SeqMetaSamplerProcessor(baseline=baseline, discount=0.99,
                                            gae_lambda=0.95, normalize_adv=True,
                                            positive_adv=False)
    processor.mask_mode = getattr(meta_policy, "mask_mode", None)
    algo = MRLCO(policy=meta_policy, meta_sampler=budgeted, meta_sampler_process=processor,
                 inner_lr=5e-4, outer_lr=5e-4, meta_batch_size=int(meta_batch_size),
                 num_inner_grad_steps=3, clip_value=0.2, value_clip_epsilon=0.2,
                 support_trajectories=int(support_trajectories),
                 ppo_batch_size_trajectories=int(support_trajectories),
                 rng=np.random.RandomState(seed), support_select="random")

    held_out = AutomotiveHeldOutEvaluator(
        support_graphs=val_support, query_graphs=val_query,
        policy=Seq2SeqPolicy(obs_dim=env.input_dim, encoder_units=128,
                             decoder_units=128, vocab_size=3, name="validation_policy"),
        source_policy=meta_policy.core_policy, ppo_batch_size=int(support_trajectories))

    from meta_trainer import Trainer
    from .automotive_trainer import AutomotiveTrainerMixin

    AutomotiveTrainerImpl = type("AutomotiveTrainer", (AutomotiveTrainerMixin, Trainer), {})
    trainer = AutomotiveTrainerImpl(
        algo=algo, env=env, sampler=budgeted, sample_processor=processor,
        policy=meta_policy, n_itr=int(n_itr), greedy_finish_time=greedy_finish,
        start_itr=0, inner_batch_size=int(support_trajectories),
        print_action_choices=False, action_print_interval=0, seed=seed,
        validation_interval=50, held_out_evaluator=held_out, ckpt_dir=ckpt_dir,
        audit_writer=None)
    trainer.auto_spec_map = {k: spec_map[k] for k in enabled}
    trainer.auto_controller = controller
    trainer.auto_dataset_fingerprint = dataset.fingerprint()
    trainer.auto_sampler = budgeted
    trainer.auto_scheduler_fingerprint = config_fingerprint(env.configs[0])
    trainer.auto_axes = dict(CO_PHYSICAL_AXES)
    trainer.auto_legacy_axes = dict(LEGACY_MIXED_AXES)
    trainer.auto_env = env
    trainer.auto_run_dir = Path(ckpt_dir).resolve().parent
    trainer.auto_run_kind = str(run_kind)
    mc_probe_graph = next((g for g in train_graphs
                           if any(str(t.criticality) == "HIGH" for t in g.tasks)),
                          train_graphs[0])
    trainer.auto_mc_fixture = _mc_fixture_probe(mc_probe_graph)
    from .automotive_fingerprint import write_fingerprint

    trainer.auto_fingerprint = write_fingerprint(
        trainer.auto_run_dir, obs_version=OBS_VERSION, scheduler_axes=dict(CO_PHYSICAL_AXES),
        checkpoint_rule_sha=checkpoint_rule_sha(),
        sampler_budget={"support_trajectories_per_meta_task": int(support_trajectories),
                        "tokens_per_trajectory": TOKENS_PER_TRAJECTORY,
                        "meta_batch_size": int(meta_batch_size)},
        run_kind=str(run_kind), outer_iterations=int(n_itr),
        dataset_dir=dataset_dir)["fingerprint"]
    return trainer, algo


def _mc_fixture_probe(graph) -> dict:
    """Deterministic MC stress fixtures built from the FROZEN budgets.

    below C_LO / within [C_LO, C_HI] / above C_HI for the first HIGH task, plus a LOW
    task that is a required ancestor of a HIGH task (must NOT be dropped).
    """
    from . import mc_runtime

    out = {"graph_id": graph.graph_id, "fixtures": {}}
    high = [t for t in graph.tasks if str(t.criticality) == "HIGH"]
    if not high:
        out["note"] = "graph declares no HIGH task; the MC contract is vacuously met"
        return out
    ref = mc_runtime.load_uncertainty()
    f_ref, xi = mc_runtime._active_reference(ref)
    target = high[0]
    lo = mc_runtime.equivalent_work(float(target.empirical_execution_budget_lo_s), f_ref, xi)
    hi = (mc_runtime.equivalent_work(float(target.empirical_execution_budget_hi_s), f_ref, xi)
          if target.empirical_execution_budget_hi_s is not None else None)
    base = {int(t.task_id): {"realized_equiv": mc_runtime.equivalent_work(
        float(t.empirical_execution_budget_lo_s), f_ref, xi) * 0.5,
        "demand_lo_equiv": mc_runtime.equivalent_work(
            float(t.empirical_execution_budget_lo_s), f_ref, xi),
        "demand_hi_equiv": (mc_runtime.equivalent_work(
            float(t.empirical_execution_budget_hi_s), f_ref, xi)
            if t.empirical_execution_budget_hi_s is not None else None)}
        for t in graph.tasks}
    cases = {"below_clo": lo * 0.5, "at_or_below_chi": lo * 1.1,
             "above_chi": (hi * 1.3 if hi is not None else lo * 1.5)}
    for name, value in cases.items():
        realized = {k: dict(v) for k, v in base.items()}
        realized[int(target.task_id)]["realized_equiv"] = max(1.0, float(value))
        try:
            mc = mc_runtime.resolve_mode_and_execution(graph, realized, None)
            out["fixtures"][name] = {
                "final_mode": mc["final_mode"],
                "switches": len(mc["switches"]),
                "triggering_task_id": (mc["switches"][0]["triggering_task_id"]
                                       if mc["switches"] else None),
                "high_preserved": bool(mc["high_preserved"]),
                "dropped": len(mc["dropped_task_ids"]),
                "capped_to_hi": len(mc["capped_to_hi"]),
            }
        except Exception as exc:  # pragma: no cover - diagnostic only
            out["fixtures"][name] = {"error": repr(exc)}
    return out


def long_run_config_path() -> Path:
    return Path(__file__).resolve().parent / "frozen_automotive_primary.yaml"
