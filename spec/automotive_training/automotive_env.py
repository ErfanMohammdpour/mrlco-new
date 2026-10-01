#!/usr/bin/env python3
"""Automotive primary environment: frozen MARGO-AUTOMOTIVE-MC-v1 graphs on the
canonical co-physical scheduler, with the frozen mixed-criticality runtime.

Duck-type compatible with `env.mec_offloaing_envs.offloading_env.OffloadingEnvironment`
for every method the primary sampler/executor/policy actually use, so the policy, the
sampler, MRLCO and the Trainer stay unchanged:

    sample_tasks, set_task, _slice_current, reset, step, input_dim, greedy_actions,
    support_graphs_per_task, resource_cluster, distribution_ids, greedy_solution,
    validation_plan_payload, validation_plan_identities

Deliberate differences from the legacy env:

* one frozen graph = one meta-task; `graph_indices` is a `slots_per_task` vector of the
  SAME graph, so one sampler step yields exactly `slots_per_task` support trajectories
  of `len(order)` tokens with no hidden over-generation;
* each slot draws its own execution-uncertainty realization at `reset()` (seed =
  sha256(base_seed | graph_id | slot | reset_count)), so support and query rollouts get
  DISJOINT realizations and the trajectories are independent instances;
* scheduling uses each graph's own co-physical ResourceConfig (graph f_UE/f_HELPER/
  f_MEC and R_MEC_UL/R_MEC_DL/R_V2V);
* MC runtime comes from `mc_runtime` (frozen `execution_uncertainty_v1`), never from a
  second ad-hoc model: LO/HI mode, dependency-safe dropping, HI-only HIGH capping,
  degradation declared unsupported and never applied, HIGH never dropped;
* the SAME executed-task policy is used for the reward and for the schedule;
* reward = telescoping latency (latency_only) plus a terminal Lagrangian penalty over
  the enabled constraint channels; the raw latency objective, each violation and each
  lambda also travel in the telemetry so they are never hidden inside a latency metric;
* the static observation encodes the frozen MC budgets/flags with `mc_mode_is_hi = 0`:
  the mode is an execution-time event and the decoder is open-loop, so the policy can
  never observe it (declared limitation, not an accident).
"""

from __future__ import annotations

import hashlib
import inspect
from typing import Any, Mapping, Sequence

import numpy as np

from .automotive_constraints import (
    CONSTRAINT_NAMES,
    AutomotiveDualController,
    evaluate_constraints,
)
from .automotive_dag import canonical_dag, decoder_order, resources_for_graph
from . import mc_runtime

TELESCOPING_COMPLETION = "all_UE"
AUTOMOTIVE_OBS_VERSION = "automotive_mc_obs_v1"


class AutomotiveEnvError(ValueError):
    """Refused. Never silently repaired."""


def _hash_seed(*parts: Any) -> int:
    material = "|".join(str(p) for p in parts)
    return int(hashlib.sha256(material.encode()).hexdigest()[:15], 16)


class AutomotiveEnv(object):
    """Meta-env over frozen automotive graphs (one graph = one meta-task)."""

    def __init__(self, graphs: Sequence[Any], resource_cluster: Any,
                 *, role: str = "support", slots_per_task: int = 20,
                 base_seed: int = 0, mc_enabled: bool = True,
                 energy_telemetry: bool = True,
                 constraint_lambdas: Mapping[str, float] | None = None,
                 uncertainty: Mapping[str, Any] | None = None,
                 single_dist: bool = False):
        if not graphs:
            raise AutomotiveEnvError("the automotive env needs at least one graph")
        # NOTE: the automotive obs version is NOT set globally here. It must be active
        # before policies/graph2seq_encoder is imported (that binds FEATURE_DIM/
        # PACKED_DIM at import time) and that is the stack builder's job. Observation
        # building below sets it temporarily and RESTORES it, so constructing an
        # automotive env can never leak the version into other (legacy) tests/runs.
        self.graph_objects = list(graphs)
        self.records = [g.as_record() for g in self.graph_objects]
        self.resource_cluster = resource_cluster
        self.role = str(role)
        self.slots_per_task = int(slots_per_task)
        self.support_graphs_per_task = int(slots_per_task)
        self.base_seed = int(base_seed)
        self.mc_enabled = bool(mc_enabled)
        self.energy_telemetry_enabled = bool(energy_telemetry)
        self.constraint_lambdas = dict(constraint_lambdas or {})
        self.single_dist = bool(single_dist)
        self.constraint_controller = None
        self.uncertainty = dict(uncertainty) if uncertainty is not None else mc_runtime.load_uncertainty()
        self.greedy_actions = (0, 1, 2)
        self.task_id = -1
        self.graph_indices = None
        self.reset_count = 0
        self.last_realization: list[dict] = []
        self.last_mc_result: list[dict] = []
        self.last_constraint_costs = None

        self.configs = [resources_for_graph(g) for g in self.records]
        self.dags = [canonical_dag(g) for g in self.records]
        self.orders = [decoder_order(g) for g in self.records]
        observations = [self._observation(i) for i in range(len(self.records))]
        if self.single_dist:
            # one "distribution" holding every graph: the layout the plain
            # Seq2SeqSampler/Seq2SeqPolicy validation path expects (rank-3 obs)
            self.total_task = 1
            self.distribution_ids = [0]
            self.task_graphs_batchs = [list(self.records)]
            self.encoder_batchs = [list(observations)]
            self.decoder_full_lengths = [[len(self.orders[i]) for i in range(len(self.records))]]
        else:
            self.total_task = len(self.records)
            self.distribution_ids = list(range(self.total_task))
            self.task_graphs_batchs = [[g] for g in self.records]
            self.encoder_batchs = [[o] for o in observations]
            self.decoder_full_lengths = [[len(self.orders[i])] for i in range(self.total_task)]
        self.max_running_time_batchs = [[1.0] for _ in self.records]
        self.min_running_time_batchs = [[0.0] for _ in self.records]
        self.input_dim = int(np.asarray(self.encoder_batchs[0][0]).shape[-1])
        self._slot_realizations = []
        self._slot_mc = []
        self._draw_slots()

    # -- observation --------------------------------------------------------
    def _observation(self, index: int) -> np.ndarray:
        from env.mec_offloaing_envs.scheduler import encoder_obs

        kwargs = {}
        if "mc_context" in inspect.signature(encoder_obs.encode_canonical_dag).parameters:
            kwargs["mc_context"] = self._mc_context(self.graph_objects[index])
        previous = str(getattr(encoder_obs, "OBS_VERSION", "v1"))
        try:
            encoder_obs.set_obs_version(AUTOMOTIVE_OBS_VERSION)
            arr = encoder_obs.encode_canonical_dag(
                self.dags[index], self.orders[index],
                stats=self._stats(),
                resources=self.configs[index], **kwargs)
        finally:
            if previous != AUTOMOTIVE_OBS_VERSION:
                encoder_obs.set_obs_version(previous)
        return np.asarray(arr, dtype=np.float32)

    def _axes_fingerprint(self) -> str:
        if getattr(self, "_axes_sha", None) is None:
            from .automotive_resources import axes_fingerprint

            self._axes_sha = axes_fingerprint()
        return self._axes_sha

    def _stats(self):
        """Frozen automotive feature stats with degenerate fitted stds neutralised.

        The frozen v3 fit has `std(log_ue_cpu) = 6.99e-11` because the legacy corpus
        held the UE CPU rate constant. Standardising with that value explodes a
        bounded feature to ~1e10 and destroys the policy input. The automotive caller
        therefore treats any fitted std below 1e-6 as IDENTITY (divide by 1.0); the
        change is caller-side, deterministic, and never mutates the frozen stats file
        or the shared encoder.
        """
        if getattr(self, "_stats_cache", None) is None:
            from env.mec_offloaing_envs.scheduler import encoder_obs

            base = encoder_obs.default_feature_stats()
            std = np.asarray(base.std, dtype=np.float64).copy()
            degenerate = std < 1e-6
            if degenerate.any():
                std[degenerate] = 1.0
            self._stats_cache = encoder_obs.FeatureStats(
                feature_names=base.feature_names,
                mean=np.asarray(base.mean, dtype=np.float64),
                std=std,
                n_graphs=int(base.n_graphs),
                n_nodes=int(base.n_nodes),
                role=str(base.role),
                max_indegree_unique=int(base.max_indegree_unique),
                max_outdegree_unique=int(base.max_outdegree_unique),
                dataset_manifest_sha256=str(base.dataset_manifest_sha256),
                split_policy_sha256=str(base.split_policy_sha256),
            )
            self._stats_degenerate_columns = [
                str(base.feature_names[i]) for i in np.flatnonzero(degenerate)]
        return self._stats_cache

    def _mc_context(self, graph: Any) -> dict:
        per_task = {}
        max_c_lo = 0.0
        for t in graph.tasks:
            c_lo = float(t.empirical_execution_budget_lo_s)
            c_hi = t.empirical_execution_budget_hi_s
            max_c_lo = max(max_c_lo, c_lo)
            span = float(t.L_s) - float(t.E_s)
            per_task[int(t.task_id)] = {
                "c_lo_s": c_lo,
                "c_hi_s": None if c_hi is None else float(c_hi),
                "has_c_hi": c_hi is not None,
                "c_hi_over_lo": 0.0 if c_hi is None else float(c_hi) / c_lo,
                "drop_allowed": bool(t.drop_allowed_hi_mode),
                "degrade_allowed": bool(t.degrade_allowed_hi_mode),
                "slack_ratio": 0.0 if span <= 0 else max(0.0, min(1.0, float(t.slack_s) / span)),
            }
        return {"per_task": per_task, "mode_is_hi": False,
                "D_G_s": float(graph.D_G_s), "max_c_lo_s": max(max_c_lo, 1e-12)}

    # -- meta-task API ------------------------------------------------------
    def sample_tasks(self, n_tasks: int) -> list[dict]:
        if self.single_dist:
            if int(n_tasks) != 1:
                raise AutomotiveEnvError(
                    "single_dist mode exposes exactly one distribution (the whole graph set)")
            return [{"dist_index": 0,
                     "graph_indices": np.arange(len(self.records), dtype=np.int32)}]
        ids = np.random.choice(np.arange(self.total_task), int(n_tasks), replace=False)
        return [{"dist_index": int(i),
                 "graph_indices": np.zeros(self.slots_per_task, dtype=np.int32)}
                for i in ids]

    def set_task(self, task: Mapping[str, Any]) -> None:
        self.task_id = int(task["dist_index"])
        self.graph_indices = np.asarray(task["graph_indices"], dtype=np.int32)

    def _slice_current(self, batch):
        item = batch[self.task_id]
        if self.graph_indices is None:
            return item
        idx = np.asarray(self.graph_indices, dtype=np.int32)
        if isinstance(item, np.ndarray):
            return item[idx]
        return [item[int(i)] for i in idx]

    def reset(self):
        self.resource_cluster.reset()
        self.reset_count += 1
        self._draw_slots()
        return np.array(self._slice_current(self.encoder_batchs))

    # -- MC realization -----------------------------------------------------
    def rollout_seed(self, graph_id: str, slot: int, reset_count: int) -> int:
        return _hash_seed("margo-rollout", self.base_seed, graph_id, slot, reset_count)

    def _graph_index(self, slot: int) -> int:
        """Global graph index for one slot (single_dist mode maps slots -> graphs)."""
        return int(self.graph_indices[slot]) if self.single_dist else int(self.task_id)

    def _draw_slots(self) -> None:
        if self.task_id < 0 or self.graph_indices is None:
            self._slot_realizations, self._slot_mc = [], []
            return
        self._slot_realizations, self._slot_mc = [], []
        for slot in self.graph_indices:
            graph = self.graph_objects[self._graph_index(int(slot))]
            seed = self.rollout_seed(graph.graph_id, int(slot), self.reset_count)
            if self.mc_enabled:
                realized = mc_runtime.draw_realized_demands(graph, seed, self.uncertainty)
                mc = mc_runtime.resolve_mode_and_execution(
                    graph, realized, self.orders[self.task_id], self.uncertainty)
                if any(str(t.criticality) == "HIGH" for t in graph.tasks):
                    mc_runtime.assert_high_preserved(mc, graph)
            else:
                realized, mc = None, None
            self._slot_realizations.append(realized)
            self._slot_mc.append(mc)

    def set_constraint_lambdas(self, lambdas: Mapping[str, float]) -> None:
        self.constraint_lambdas = {str(k): float(v) for k, v in dict(lambdas).items()}

    # -- scheduling / reward ------------------------------------------------
    def _schedule(self, index: int, actions: Sequence[int], mc: Mapping[str, Any] | None):
        from env.mec_offloaing_envs.scheduler.energy_scope import energy_scalar
        from env.mec_offloaing_envs.scheduler.engine import schedule
        from env.mec_offloaing_envs.scheduler.model import CanonicalDAG, CanonicalTask

        record = self.records[index]
        dag = self.dags[index]
        order = self.orders[index]
        config = self.configs[index]
        if mc is None:
            result = schedule(dag, order, [int(a) for a in actions], config)
            return result, float(energy_scalar(result, scope="system")), macro_plan(actions, order)

        survivors = list(mc["executed_task_ids"])
        execution = mc["execution"]
        tasks, macro = [], []
        for tid in survivors:
            src = dag.tasks[int(tid)]
            tasks.append(CanonicalTask(
                task_id=int(tid),
                compute_workload_bytes=max(1, int(round(float(execution[int(tid)]["effective_equiv"])))),
                task_output_bytes=src.task_output_bytes,
                external_input_bytes=src.external_input_bytes,
                cycles_per_bit=src.cycles_per_bit,
                deadline_s=src.deadline_s,
                deadline_type=src.deadline_type,
                criticality_class=src.criticality_class,
                tardiness_weight=src.tardiness_weight))
        edges = [(e.src_task_id, e.dst_task_id, e.edge_output_bytes)
                 for e in dag.edges
                 if e.src_task_id in survivors and e.dst_task_id in survivors]
        dag_use = CanonicalDAG.from_records(tasks, edges)
        order_use = [t for t in order if t in survivors]
        actions_use = [int(actions[order.index(t)]) for t in order_use]
        result = schedule(dag_use, order_use, actions_use, config)
        return result, float(energy_scalar(result, scope="system")), macro_plan(actions, order)

    def _telescoping(self, index: int, actions: Sequence[int],
                     mc: Mapping[str, Any] | None, l_scale: float):
        order = self.orders[index]
        rewards: list[float] = []
        # L_0 = the all-UE completion makespan: the constant the telescoping identity
        # sums to (sum_t r_t = -(L_final - L_0)/L_scale)
        previous, _e, _m = self._schedule(index, [0] * len(order), mc)
        previous = float(previous.makespan_seconds)
        for k in range(len(order)):
            prefix = [int(a) for a in actions[:k + 1]] + [0] * (len(order) - k - 1)
            result, _energy, _macro = self._schedule(index, prefix, mc)
            makespan = float(result.makespan_seconds)
            rewards.append(-(makespan - previous) / max(l_scale, 1e-12))
            previous = makespan
        return rewards

    def step(self, action):
        if self.single_dist:
            return self._step_single_dist(action)
        index = int(self.task_id)
        graph = self.graph_objects[index]
        order = self.orders[index]
        action = np.asarray(action)
        if action.ndim != 2 or action.shape[0] != len(self.graph_indices):
            raise AutomotiveEnvError(
                "action shape %r does not match %d slots" % (action.shape, len(self.graph_indices)))
        l_scale = float(graph.D_G_s)
        reward_batch, finish_batch, energy_batch, telemetry_batch = [], [], [], []
        self.last_realization, self.last_mc_result = [], []
        for slot, actions in enumerate(action):
            mc = self._slot_mc[slot] if self._slot_mc else None
            realization = self._slot_realizations[slot] if self._slot_realizations else None
            rewards = self._telescoping(index, actions, mc, l_scale)
            result, system_j, macro = self._schedule(index, actions, mc)
            makespan = float(result.makespan_seconds)
            channels = evaluate_constraints(_constraint_view(graph, mc), result)
            violations = {name: float(channels[name]["violation"]) for name in CONSTRAINT_NAMES}
            n_violating = {name: int(channels[name]["n_violating_tasks"]) for name in CONSTRAINT_NAMES}
            penalty = sum(float(self.constraint_lambdas.get(name, 0.0)) * violations[name]
                          for name in CONSTRAINT_NAMES)
            if penalty:
                rewards[-1] = rewards[-1] - penalty / max(l_scale, 1e-12)
            telemetry = self._telemetry(index, graph, order, result, mc, realization,
                                        rewards, slot, system_j)
            penalty = float(telemetry["constraint_penalty"])
            if penalty:
                rewards[-1] = rewards[-1] - penalty / max(l_scale, 1e-12)
            reward_batch.append(np.asarray(rewards, dtype=np.float32))
            finish_batch.append(makespan)
            energy_batch.append(np.full(len(order), float(system_j), dtype=np.float32))
            telemetry_batch.append(telemetry)
        self.last_realization = list(self._slot_realizations)
        self.last_mc_result = list(self._slot_mc)
        observation = np.array(self._slice_current(self.encoder_batchs))
        info = (finish_batch, energy_batch, telemetry_batch)
        return observation, reward_batch, True, info

    def _telemetry(self, index, graph, order, result, mc, realization, rewards, slot,
                   system_j) -> dict:
        """Energy + constraint + MC telemetry for one episode (penalty NOT applied)."""
        from env.mec_offloaing_envs.scheduler.energy_scope import energy_scalar
        from env.mec_offloaing_envs.scheduler.energy_telemetry import (
            TELEMETRY_SCHEMA_VERSION,
        )

        makespan = float(result.makespan_seconds)
        l_scale = max(float(graph.D_G_s), 1e-12)
        channels = evaluate_constraints(_constraint_view(graph, mc), result)
        violations = {name: float(channels[name]["violation"]) for name in CONSTRAINT_NAMES}
        n_violating = {name: int(channels[name]["n_violating_tasks"]) for name in CONSTRAINT_NAMES}
        penalty = sum(float(self.constraint_lambdas.get(name, 0.0)) * violations[name]
                      for name in CONSTRAINT_NAMES)
        latency_term = -makespan / l_scale
        scoped = {
            "requester_joules": energy_scalar(result, scope="requester"),
            "mobile_joules": energy_scalar(result, scope="mobile"),
            "system_joules": energy_scalar(result, scope="system"),
        }
        return {
            "schema_version": TELEMETRY_SCHEMA_VERSION,
            **scoped,
            "primary_scope": "system",
            "primary_joules": float(scoped["system_joules"]),
            "scheduler_config_sha256": self._axes_fingerprint(),
            "graph_scheduler_config_sha256": str(self.configs[index].source_config_sha256),
            "makespan_s": makespan,
            "latency_only_objective": latency_term,
            **{f"violation/{name}": violations[name] for name in CONSTRAINT_NAMES},
            **{f"lambda/{name}": float(self.constraint_lambdas.get(name, 0.0))
               for name in CONSTRAINT_NAMES},
            **{f"n_violating/{name}": n_violating[name] for name in CONSTRAINT_NAMES},
            "constraint_penalty": float(penalty),
            "penalized_objective": latency_term - penalty / l_scale,
            "firm_miss_count": int(n_violating["C_HI_TASK_TARDINESS"]
                                   + n_violating["C_MED_TASK_TARDINESS"]),
            "task_count": len(order),
            "mode": "NA" if mc is None else str(mc["final_mode"]),
            "mode_switch_count": 0 if mc is None else len(mc["switches"]),
            "dropped_task_count": 0 if mc is None else len(mc["dropped_task_ids"]),
            "capped_to_hi_count": 0 if mc is None else len(mc["capped_to_hi"]),
            "high_preserved": 1.0 if (mc is None or bool(mc["high_preserved"])) else 0.0,
            "degradation_applied_count": 0.0 if mc is None else float(
                sum(1 for v in (mc.get("degrade_applied") or {}).values() if v)
                if isinstance(mc.get("degrade_applied"), dict)
                else len(mc.get("degrade_applied") or [])),
            "slot": int(slot),
            "graph_index": int(index),
            "reset_count": int(self.reset_count),
            "rollout_seed": -1 if realization is None else int(
                self.rollout_seed(graph.graph_id, int(slot), self.reset_count)),
            "graph_id_hash": float(int(hashlib.sha256(graph.graph_id.encode())
                                       .hexdigest()[:8], 16) % 1000003),
        }

    def _step_single_dist(self, action):
        action = np.asarray(action)
        if action.ndim != 2 or action.shape[0] != len(self.graph_indices):
            raise AutomotiveEnvError(
                "action shape %r does not match %d slots" % (action.shape, len(self.graph_indices)))
        reward_batch, finish_batch, energy_batch, telemetry_batch = [], [], [], []
        self.last_realization, self.last_mc_result = [], []
        for slot, actions in enumerate(action):
            index = self._graph_index(slot)
            graph = self.graph_objects[index]
            order = self.orders[index]
            mc = self._slot_mc[slot] if self._slot_mc else None
            realization = self._slot_realizations[slot] if self._slot_realizations else None
            rewards = self._telescoping(index, actions, mc, float(graph.D_G_s))
            result, system_j, _macro = self._schedule(index, actions, mc)
            telemetry = self._telemetry(index, graph, order, result, mc, realization,
                                        rewards, slot, system_j)
            if telemetry["constraint_penalty"]:
                rewards[-1] = rewards[-1] - telemetry["constraint_penalty"] / max(
                    float(graph.D_G_s), 1e-12)
            reward_batch.append(np.asarray(rewards, dtype=np.float32))
            finish_batch.append(float(result.makespan_seconds))
            energy_batch.append(np.full(len(order), float(system_j), dtype=np.float32))
            telemetry_batch.append(telemetry)
        self.last_realization = list(self._slot_realizations)
        self.last_mc_result = list(self._slot_mc)
        observation = np.array(self._slice_current(self.encoder_batchs))
        return observation, reward_batch, True, (finish_batch, energy_batch, telemetry_batch)

    # -- stack helpers ------------------------------------------------------
    def greedy_solution(self):
        """(action plan, per-meta-task greedy makespans, per-meta-task energies).

        The Trainer indexes the second element by `dist_index` (one meta-task = one
        frozen graph), so both lists are per-graph, never a scalar.
        """
        makespans, energies, best_actions = [], [], None
        for index in range(self.total_task):
            best = None
            for action in (0, 1, 2):
                result, energy, _macro = self._schedule(
                    index, [action] * len(self.orders[index]), None)
                candidate = (action, float(result.makespan_seconds), float(energy))
                if best is None or candidate[1] < best[1]:
                    best = candidate
            makespans.append(best[1])
            energies.append(best[2])
            if best_actions is None:
                best_actions = [best[0]] * len(self.orders[index])
        return best_actions or [0] * 20, makespans, energies

    def validation_plan_payload(self):
        return None

    def validation_plan_identities(self):
        return [{"graph_fingerprint": g.canonical_sha256,
                 "scheduler_config_sha256": c.source_config_sha256,
                 "energy_scope": c.energy_scope}
                for g, c in zip(self.graph_objects, self.configs)]

    def render(self, mode="human"):
        pass


def _constraint_view(graph: Any, mc: Mapping[str, Any] | None):
    """Graph view restricted to the EXECUTED tasks.

    A task dropped by the frozen HI policy has no executed deadline obligation, and
    the constraint module refuses a schedule result that silently lacks a task it was
    asked about. HIGH is never dropped and MEDIUM drop is disabled in the frozen
    policy, so the two tardiness channels are unaffected.
    """
    if mc is None:
        return graph
    survivors = set(int(t) for t in mc["executed_task_ids"])
    from types import SimpleNamespace

    return SimpleNamespace(tasks=[t for t in graph.tasks if int(t.task_id) in survivors],
                           D_G_s=float(graph.D_G_s), graph_id=str(graph.graph_id))


def macro_plan(actions: Sequence[int], order: Sequence[int]) -> list[tuple[int, int]]:
    """Macro actions (UE/MEC/HELPER per task) implied by the 20-token plan."""
    return [(int(t), int(a)) for t, a in zip(order, [int(a) for a in actions])]


def budget_for_one_meta_task(slots_per_task: int, tokens_per_trajectory: int) -> int:
    """Explicit token budget for ONE meta-task (never inferred from max_path_length)."""
    return int(slots_per_task) * int(tokens_per_trajectory)
