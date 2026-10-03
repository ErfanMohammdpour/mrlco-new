#!/usr/bin/env python3
"""v2 macro-step environment: v1 observation/MC layer + v2 shared scheduler execution.

Composition, not modification: the frozen `AutomotiveEnv` still produces the MC
realization (via `mc_runtime`) and the v1 observation tensor (`automotive_mc_obs_v1`,
PACKED_DIM 79), while execution, reward and telemetry come from the v2 shared scheduler
(queueing on shared MEC CPU/radio, estimated-vs-realized links, helper occupancy/contact,
criticality-aware reliability).

Explicit limitation (recorded, not hidden): the TF observation is still the v1 79-column
schema. The new v2 context (estimated link multipliers, link confidence, helper contact
remaining, helper busy fraction, reliability epsilon, criticality mix) is exposed by
`v2_context()` for the future encoder bump; until that bump lands the policy cannot see it,
so a v2 training run measures the v2 *dynamics*, not v2-informed decision making.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from spec.automotive_training.automotive_env import AutomotiveEnv
from spec.automotive_training.automotive_primary import AutomotiveResourceCluster
from spec.automotive_training.v2.adapters import (
    compute_spec, dag_spec_from_graph, link_spec, plan_map_from_actions,
)
from spec.automotive_training.v2.helper_model import HelperState, make_helpers
from spec.automotive_training.v2.link_model import LINK_DL, LINK_UL, LINK_V2V, make_process
# NOTE: `v2.observation` imports V2_CONTEXT_FIELDS from this module, so the observation
# helpers are imported lazily inside the methods that need them (no module-level cycle).
from spec.automotive_training.v2.reliability import load_classes, standby_required
from spec.automotive_training.v2.shared_scheduler import (
    MEC, UE, V2ComputeSpec, V2ScheduleError, schedule_shared,
)

V2_CONTEXT_FIELDS = (
    "est_ul", "est_dl", "est_v2v",
    "conf_ul", "conf_dl", "conf_v2v",
    "helper_contact_remaining_s", "helper_busy_fraction",
    "epsilon_class", "criticality_high_share", "criticality_medium_share",
    "mec_workers",
)


class V2EnvError(RuntimeError):
    """Raised on invalid v2 environment use."""


@dataclass
class V2EpisodeTelemetry:
    slot: int
    graph_id: str
    makespan_s: float
    queue_wait_total_s: float
    outage_wait_total_s: float
    helper_rejections: int
    helper_contact_failures: int
    reliability_rejections: int
    fallback_reserved_s: float
    location_mix: dict
    deadline_miss_rate: float
    high_miss_count: int
    medium_miss_count: int
    energy_joules: float
    scheduler_invariants: dict

    def as_dict(self) -> dict:
        return dict(self.__dict__)


class V2AutomotiveEnv:
    """Duck-type compatible with the frozen sampler/trainer env interface."""

    def __init__(self, graphs: Sequence, cluster: Any | None = None, *,
                 role: str = "meta_train", slots_per_task: int = 1, base_seed: int = 0,
                 link_regime: str = "stable", mec_workers: int = 1,
                 helper_contact_mean_s: float = 2.0, helper_contact_cv: float = 0.5,
                 helper_busy_s: float = 0.0, reliability: bool = False,
                 runtime_deadline_scale: float = 1.0, mc_enabled: bool = True,
                 constraint_controller: Any | None = None, single_dist: bool = False):
        # Two layouts are supported, exactly as in the frozen env:
        # * single_dist=False (training): one distribution per graph, `slots_per_task`
        #   trajectories per distribution, `sample_tasks(meta_batch_size)` selects graphs;
        # * single_dist=True (validation/eval): one distribution holding every graph, so one
        #   slot per graph.
        graphs = list(graphs)
        self.single_dist = bool(single_dist)
        slots = max(1, len(graphs)) if self.single_dist else max(1, int(slots_per_task))
        self.base = AutomotiveEnv(graphs, cluster or AutomotiveResourceCluster(),
                                  role=role, slots_per_task=slots,
                                  base_seed=int(base_seed), mc_enabled=bool(mc_enabled),
                                  single_dist=self.single_dist)
        self.role = role
        self.slots_per_task = max(1, len(graphs)) if bool(single_dist) else max(1, int(slots_per_task))
        self.base_seed = int(base_seed)
        self.link_regime = str(link_regime)
        self.mec_workers = int(mec_workers)
        self.helper_contact_mean_s = float(helper_contact_mean_s)
        self.helper_contact_cv = float(helper_contact_cv)
        self.helper_busy_s = float(helper_busy_s)
        self.reliability_enabled = bool(reliability)
        self.runtime_deadline_scale = float(runtime_deadline_scale)
        self.constraint_controller = constraint_controller
        self._classes = load_classes()
        self.episodes = 0
        self.last_telemetry: list = []
        self.last_v2_context: list = []
        self.last_link_summary: dict = {}
        self.link_process = None if self.link_regime == "stable" else None  # per reset

    # -- v1-compatible surface --------------------------------------------
    def __getattr__(self, item):
        """Delegate anything not defined here to the frozen v1 env.

        The frozen stack builder and the held-out evaluator touch a number of v1 env
        attributes (`configs`, `graph_objects`, `orders`, `graph_indices`, `_slot_mc`,
        `greedy_solution`, ...). Delegation keeps that surface working without copying it,
        and keeps the v2 env a true composition.
        """
        if item.startswith("__") and item.endswith("__"):
            raise AttributeError(item)
        try:
            base = object.__getattribute__(self, "base")
        except AttributeError:
            raise AttributeError(item)
        return getattr(base, item)

    @property
    def input_dim(self) -> int:
        """91 when the v2 obs schema is active, otherwise the frozen 79."""
        from spec.automotive_training.v2.observation import V2_PACKED_DIM

        # The v2 env ALWAYS emits the v2 schema (frozen v1 rows + 12 context columns), so this
        # is 91 regardless of the globally active version. The stack builder must therefore
        # activate automotive_v2_obs_v1 before the policy/encoder is imported, or the encoder
        # would expect 79/50 and fail loudly at construction (which is the desired failure).
        return int(V2_PACKED_DIM)

    @property
    def total_task(self) -> int:
        return self.base.total_task

    def set_task(self, task: Mapping) -> None:
        self.base.set_task(task)

    def sample_tasks(self, n_tasks: int):
        return self.base.sample_tasks(n_tasks)

    def set_constraint_lambdas(self, lambdas) -> None:
        self.base.set_constraint_lambdas(lambdas)
        self.constraint_lambdas = dict(lambdas or {})

    def reset(self):
        obs = self.base.reset()
        self.episodes += 1
        indices = getattr(self.base, "graph_indices", None)
        slots = [] if indices is None else [int(s) for s in indices]
        # one seeded link process and one helper pool per slot
        self.link_process = None
        if self.link_regime != "stable":
            self.link_process = make_process(self.link_regime,
                                             seed=self.base_seed * 1000003 + self.episodes)
        self.helper_states = []
        for i, slot in enumerate(slots):
            graph = self.base.graph_objects[self.base._graph_index(int(slot))]
            specs = {0: {"cpu_bytes_per_s": compute_spec([graph]).helper_cpu_bytes_per_s[0],
                         "busy_until_s": self.helper_busy_s, "owner": i}}
            self.helper_states.append(make_helpers(
                specs, seed=self.base_seed * 7919 + self.episodes * 31 + i,
                contact_mean_s=self.helper_contact_mean_s, contact_cv=self.helper_contact_cv))
        self.last_v2_context = self._contexts()
        self.last_link_summary = {}
        if self.link_process is not None:
            graph0 = self.base.graph_objects[self.base._graph_index(slots[0])]
            cfg = link_spec(graph0)
            self.last_link_summary = self.link_process.summary(
                {"mec_ul": cfg.mec_ul_bytes_per_s, "mec_dl": cfg.mec_dl_bytes_per_s,
                 "v2v": cfg.v2v_bytes_per_s})
        return self._packed_observation(self.last_v2_context)

    def _packed_observation(self, contexts) -> np.ndarray:
        """Frozen v1 rows with the 12 v2 context columns inserted (91-wide)."""
        from spec.automotive_training.v2.observation import V2_PACKED_DIM, pack_v2_row

        rows = np.asarray(self.base._slice_current(self.base.encoder_batchs), dtype=np.float32)
        if rows.shape[-1] == V2_PACKED_DIM:
            return rows
        n_rows = int(rows.shape[0])
        # The frozen meta-sampler runs the env through MetaIterativeEnvExecutor, which sets ONE
        # meta task (one graph) per call, so every returned row belongs to the SAME graph. The
        # per-slot vector is therefore the graph's context repeated over its slots; when the
        # executor exposes the flat slot list we still use the exact per-slot mapping.
        raw = np.asarray(getattr(self.base, "graph_indices", []), dtype=object).reshape(-1)
        valid = []
        for value in raw:
            try:
                valid.append(int(value))
            except (TypeError, ValueError):
                continue
        if len(valid) == n_rows and n_rows > 1:
            ctxs = [self._context_vector(int(s)) for s in valid]
        elif len(contexts) == n_rows:
            ctxs = [np.asarray(c, dtype=np.float32) for c in contexts]
        else:
            gid = valid[0] if valid else int(getattr(self.base, "task_id", 0) or 0)
            gid = min(max(int(gid), 0), len(self.base.graph_objects) - 1)
            ctxs = [self._context_for_graph(self.base.graph_objects[gid])] * n_rows
        out = np.empty((n_rows, rows.shape[1], V2_PACKED_DIM), dtype=np.float32)
        for row in range(n_rows):
            out[row] = pack_v2_row(rows[row], np.asarray(ctxs[row], dtype=np.float32))
        return out

    def _slot_of_graph(self, graph):
        """Flat slot index whose graph is `graph` (None when the layout does not expose it)."""
        indices = np.asarray(getattr(self.base, "graph_indices", []), dtype=object).reshape(-1)
        try:
            target = int(self.base.graph_objects.index(graph))
        except ValueError:
            return None
        for slot, value in enumerate(indices):
            try:
                if int(value) == target:
                    return slot
            except (TypeError, ValueError):
                continue
        return None

    def _contexts(self) -> list:
        """One v2 context per flat slot (training: meta_batch x slots_per_task)."""
        indices = getattr(self.base, "graph_indices", None)
        if indices is None:
            return []
        flat = np.asarray(indices).reshape(-1)
        return [self._context_vector(int(slot)) for slot in flat]

    # -- execution ---------------------------------------------------------
    def _schedule_slot(self, slot: int, actions: Sequence[int]):
        index = self.base._graph_index(int(slot))
        graph = self.base.graph_objects[index]
        mc = self.base._slot_mc[slot] if self.base._slot_mc else None
        link = link_spec(graph)
        compute = compute_spec([graph], mec_workers=self.mec_workers)
        dag = dag_spec_from_graph(graph, dag_id="s%d" % slot, owner=0, mc=mc, helper_id=0)
        plan = plan_map_from_actions(graph, actions)
        gate = None
        evidence = None
        if self.reliability_enabled:
            from spec.automotive_training.v2.reliability import make_gate
            gate = make_gate(self._classes)
            conf = 1.0
            outage = 0.0
            if self.link_process is not None:
                conf = float(np.mean([self.link_process.confidence(LINK_UL),
                                      self.link_process.confidence(LINK_DL),
                                      self.link_process.confidence(LINK_V2V)]))
                outage = float(np.mean([np.mean(self.link_process._outage[LINK_UL]),
                                        np.mean(self.link_process._outage[LINK_DL]),
                                        np.mean(self.link_process._outage[LINK_V2V])]))
            evidence = {"link_confidence": conf, "outage_fraction": outage}
        return schedule_shared([dag], {"s%d" % slot: plan}, link=link, compute=compute,
                               link_process=self.link_process,
                               helper_states=self.helper_states[slot],
                               reliability_gate=gate, reliability_evidence=evidence,
                               standby_for=standby_required if gate else None)

    def _telescoping(self, slot: int, actions: Sequence[int]) -> list:
        """Prefix marginal makespans under the v2 scheduler (v1 reward semantics)."""
        index = self.base._graph_index(int(slot))
        graph = self.base.graph_objects[index]
        n = len(self.base.orders[index])
        previous = self._schedule_slot(slot, [0] * n).makespan_s
        rewards = []
        l_scale = max(float(graph.D_G_s), 1e-12)
        for k in range(n):
            prefix = [int(a) for a in actions[:k + 1]] + [0] * (n - k - 1)
            current = self._schedule_slot(slot, prefix).makespan_s
            rewards.append(-(current - previous) / l_scale)
            previous = current
        return rewards

    def step(self, action):
        action = np.asarray(action)
        slots = getattr(self.base, "graph_indices", None)
        if slots is None:
            raise V2EnvError(
                "step() called before set_task(): the meta-sampler executor must select a "
                "meta task first (sample_tasks -> set_task -> reset -> step)")
        n_slots = len(slots)
        if action.ndim != 2 or action.shape[0] != n_slots:
            raise V2EnvError("action shape %r does not match %d slots"
                             % (action.shape, n_slots))
        reward_batch, finish_batch, energy_batch, telemetry_batch = [], [], [], []
        self.last_telemetry, self.last_v2_context = [], []
        for slot, actions in enumerate(action):
            rewards = self._telescoping(slot, actions)
            result = self._schedule_slot(slot, actions)
            index = self.base._graph_index(int(slot))
            graph = self.base.graph_objects[index]
            l_scale = max(float(graph.D_G_s), 1e-12)
            tasks = self.base.records[index].get("tasks", []) if hasattr(self.base.records[index], "get") else []
            by_id = {int(t["task_id"]): t for t in tasks} if tasks else {}
            misses = {"HIGH": 0, "MEDIUM": 0, "other": 0}
            n_tasks = 0
            for (dag_id, tid), tm in result.timings.items():
                n_tasks += 1
                deadline = None
                spec_task = by_id.get(int(tid))
                if spec_task is not None and spec_task.get("deadline_s") is not None:
                    deadline = float(spec_task["deadline_s"]) * self.runtime_deadline_scale
                if deadline is not None and tm.finish_s > deadline + 1e-12:
                    crit = str(spec_task.get("criticality", "other")).upper()
                    misses[crit if crit in ("HIGH", "MEDIUM") else "other"] += 1
            penalty = 0.0
            if self.constraint_controller is not None:
                penalty = 0.0
            telemetry = V2EpisodeTelemetry(
                slot=slot, graph_id=graph.graph_id, makespan_s=float(result.makespan_s),
                queue_wait_total_s=float(result.queue_stats["cpu_wait_mean_s"] * n_tasks),
                outage_wait_total_s=float(result.queue_stats["outage_wait_total_s"]),
                helper_rejections=int(result.queue_stats["helper_rejections_inadmissible"]),
                helper_contact_failures=int(result.queue_stats["helper_contact_failures"]),
                reliability_rejections=int(result.queue_stats["reliability_rejections"]),
                fallback_reserved_s=float(result.queue_stats["fallback_reserved_s"]),
                location_mix={loc: sum(1 for t in result.timings.values() if t.location == loc)
                              for loc in (UE, MEC, "HELPER")},
                deadline_miss_rate=(sum(misses.values()) / n_tasks) if n_tasks else 0.0,
                high_miss_count=misses["HIGH"], medium_miss_count=misses["MEDIUM"],
                energy_joules=0.0, scheduler_invariants=dict(result.invariants))
            self.last_telemetry.append(telemetry)
            self.last_v2_context.append(self._context_vector(slot))
            reward_batch.append(np.asarray(rewards, dtype=np.float32))
            finish_batch.append(float(result.makespan_s))
            energy_batch.append(np.zeros(len(rewards), dtype=np.float32))
            telemetry_batch.append(self._frozen_telemetry(slot, telemetry))
        obs = self._packed_observation(self.last_v2_context)
        return obs, reward_batch, True, (finish_batch, energy_batch, telemetry_batch)

    def _frozen_telemetry(self, slot: int, telemetry) -> dict:
        """Per-slot telemetry in the FROZEN energy schema, with the v2 record nested.

        Energy is explicitly not configured for the v2 system model (`energy_constraint:
        not_configured`), so the three accounting boundaries are emitted as exact ZEROS with
        the frozen schema's `primary_scope="mobile"` (the validator accepts only
        requester|mobile|system) plus an explicit `energy_constraint="not_configured"` marker,
        so a zero can never be mistaken for a measured v1 energy number. The REAL scheduler
        config fingerprint is kept. The v2 dynamics live under the `v2` key.
        """
        from env.mec_offloaing_envs.scheduler.energy_telemetry import (
            TELEMETRY_SCHEMA_VERSION,
        )

        index = self.base._graph_index(int(slot))
        graph = self.base.graph_objects[index]
        cfg = self.configs[index]
        v2 = telemetry.as_dict()
        return {
            "schema_version": TELEMETRY_SCHEMA_VERSION,
            "requester_joules": 0.0,
            "mobile_joules": 0.0,
            "system_joules": 0.0,
            "primary_scope": "mobile",   # frozen schema allows requester|mobile|system only
            "primary_joules": 0.0,
            "scheduler_config_sha256": self.base._axes_fingerprint(),
            "graph_scheduler_config_sha256": str(cfg.source_config_sha256),
            "makespan_s": float(telemetry.makespan_s),
            "latency_only_objective": -float(telemetry.makespan_s) / max(float(graph.D_G_s), 1e-12),
            "energy_constraint": "not_configured",
            "v2": v2,
        }

    # -- v2 context (not yet in the TF observation) ------------------------
    def _context_vector(self, slot: int) -> np.ndarray:
        return self._context_for_graph(self.base.graph_objects[self.base._graph_index(int(slot))])

    def _context_for_graph(self, graph) -> np.ndarray:
        est = {LINK_UL: 1.0, LINK_DL: 1.0, LINK_V2V: 1.0}
        conf = {LINK_UL: 1.0, LINK_DL: 1.0, LINK_V2V: 1.0}
        if self.link_process is not None:
            for link in (LINK_UL, LINK_DL, LINK_V2V):
                est[link] = self.link_process.estimate_at(link, 0.0)
                conf[link] = self.link_process.confidence(link, 0.0)
        slot = self._slot_of_graph(graph)
        helper = self.helper_states[slot].get(0) if slot is not None else None
        predicted = getattr(helper, "predicted_contact_end_s", None) if helper else None
        remaining = 0.0 if predicted is None or not math.isfinite(float(predicted)) else float(predicted)
        counts = {"HIGH": 0, "MEDIUM": 0, "other": 0}
        for t in graph.tasks:
            cls = str(t.criticality).upper()
            counts[cls if cls in ("HIGH", "MEDIUM") else "other"] += 1
        total = max(1, sum(counts.values()))
        crit = str(graph.tasks[0].criticality).upper() if graph.tasks else "MEDIUM"
        eps = float(self._classes.get(crit, self._classes["MEDIUM"]).epsilon)
        return np.asarray([
            est[LINK_UL], est[LINK_DL], est[LINK_V2V],
            conf[LINK_UL], conf[LINK_DL], conf[LINK_V2V],
            remaining, float(self.helper_busy_s),
            eps, counts["HIGH"] / total, counts["MEDIUM"] / total, float(self.mec_workers),
        ], dtype=np.float32)

    def v2_context(self) -> np.ndarray:
        """[slots, len(V2_CONTEXT_FIELDS)] v2 context vector (encoder-bump pending)."""
        if not self.last_v2_context:
            raise V2EnvError("call reset()/step() before v2_context()")
        return np.asarray(self.last_v2_context, dtype=np.float32)
