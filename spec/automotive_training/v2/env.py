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
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from spec.automotive_training.automotive_env import AutomotiveEnv
from spec.automotive_training.automotive_primary import AutomotiveResourceCluster
from spec.automotive_training.v2.adapters import (
    compute_spec, dag_spec_from_graph, link_spec, plan_map_from_actions,
)
from spec.automotive_training.v2.constraint_channels import channel_telemetry
from spec.automotive_training.v2.constraints_v2 import (
    ConstraintCostBatch, V2ConstraintError, V2ConstraintManager, spec_from_fractions,
)
from spec.automotive_training.v2.energy import reference_ranges_from_plans, schedule_energy
from spec.automotive_training.v2.helper_model import HelperState, make_helpers
from spec.automotive_training.v2.link_model import LINK_DL, LINK_UL, LINK_V2V, make_process
# NOTE: `v2.observation` imports V2_CONTEXT_FIELDS from this module, so the observation
# helpers are imported lazily inside the methods that need them (no module-level cycle).
from spec.automotive_training.v2.reliability import load_classes, standby_required
from spec.automotive_training.v2.shared_scheduler import (
    MEC, UE, V2ComputeSpec, V2ScheduleError, schedule_shared,
)
from spec.automotive_training.v2.world import (
    V2World, V2WorldConfig, V2WorldError, build_world, pure_location_worlds,
)

#: The v2 decision-time context contract lives in ONE leaf module so the environment and the
#: frozen encoder cannot drift apart; see `v2/context_fields.py` for what each entry means
#: and for the observability rules every entry must satisfy.
from spec.automotive_training.v2.context_fields import V2_CONTEXT_FIELDS  # noqa: E402


class V2EnvError(RuntimeError):
    """Raised on invalid v2 environment use."""


def stable_seed(*parts) -> int:
    """Seed keyed by STABLE identities, never by how many RNG calls have been consumed."""
    import hashlib

    payload = "|".join(str(p) for p in parts).encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:12], 16) % (2 ** 31 - 1)


def _crit_of(by_id: Mapping, timing) -> str:
    """Criticality of a timing record, from the graph record (never from a global guess)."""
    rec = by_id.get(int(getattr(timing, "task_id", -1)))
    if not isinstance(rec, Mapping):
        return "other"
    return str(rec.get("criticality", "other")).upper()


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
    # -- v2 stage E: real energy, constraints and denominators -------------
    world_makespan_s: float = 0.0
    requester_joules: float = 0.0
    mobile_joules: float = 0.0
    system_joules: float = 0.0
    background_joules: float = 0.0
    energy_primary_scope: str = "system"
    energy_model_sha256: str = ""
    constraint_penalty: float = 0.0
    constraint_violations: dict = field(default_factory=dict)
    constraint_signed: dict = field(default_factory=dict)
    constraint_lambdas: dict = field(default_factory=dict)
    n_tasks: int = 0
    n_helper_tasks: int = 0
    n_mec_tasks: int = 0
    n_high_tasks: int = 0
    n_medium_tasks: int = 0
    world_id: str = ""
    world_fingerprint_sha256: str = ""

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
                 constraint_controller: Any | None = None, single_dist: bool = False,
                 background_dags: int = 0, background_owner_offset: int = 1,
                 world_config: V2WorldConfig | None = None,
                 background_policy: str = "all_mec",
                 background_workload_jitter: float = 0.5,
                 background_arrival_spacing_s: float = 0.0,
                 shared_radio: bool = True, direct_helper_v2i: bool = False,
                 energy_enabled: bool = True,
                 constraints_enabled: bool = False,
                 constraint_spec: Any | None = None,
                 budget_fractions: Mapping | None = None,
                 dual_lr: float = 0.05, max_lambda: float = 1e3,
                 scheduler_config_sha256: str | None = None):
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
        #: number of background DAGs (other owners) that compete with the foreground DAG in
        #: the SAME world/scheduler call. 0 keeps the single-DAG behaviour.
        self.background_dags = max(0, int(background_dags))
        self.background_owner_offset = max(1, int(background_owner_offset))
        # ---- ONE canonical world configuration (training, gate, parity, evaluator) ----
        self.world_config = world_config or V2WorldConfig(
            background_dags=self.background_dags,
            background_owner_offset=self.background_owner_offset,
            background_policy=str(background_policy),
            background_workload_jitter=float(background_workload_jitter),
            background_arrival_spacing_s=float(background_arrival_spacing_s),
            mec_workers=self.mec_workers,
            shared_radio=bool(shared_radio),
            direct_helper_v2i=bool(direct_helper_v2i),
            helper_contact_mean_s=self.helper_contact_mean_s,
            helper_contact_cv=self.helper_contact_cv,
            helper_busy_s=self.helper_busy_s)
        self.background_dags = int(self.world_config.background_dags)
        self.energy_enabled = bool(energy_enabled)
        self.scheduler_config_sha256 = scheduler_config_sha256
        self._classes = load_classes()
        self.episodes = 0
        self.last_telemetry: list = []
        self.last_v2_context: list = []
        self.last_link_summary: dict = {}
        self.last_energy_ledger: list = []
        self.last_constraint_costs = ConstraintCostBatch()
        self.last_constraint_batch = self.last_constraint_costs
        self._last_results: dict = {}
        self.link_process = None if self.link_regime == "stable" else None  # per reset
        self._worlds: dict = {}
        self._world_key = None
        self._v1_schedule_trap = False
        self._reference_ranges: dict = {}
        self._reference_results: dict = {}
        self._reference_ledgers: dict = {}
        # ---- constraints: one manager per worker; lambdas held fixed inside a rollout ----
        from env.mec_offloaing_envs.scheduler.constraints import ConstraintSpec

        self.budget_fractions = dict(budget_fractions
                                     or {"total_energy": 0.5, "ue_energy": 0.5})
        if constraint_spec is not None:
            self.constraint_spec = constraint_spec
        elif constraints_enabled:
            self.constraint_spec = spec_from_fractions(self.budget_fractions)
        else:
            self.constraint_spec = ConstraintSpec(mode="off")
        if constraints_enabled and not self.constraint_spec.enabled:
            raise V2EnvError(
                "constraints_enabled=True but the constraint spec has no budget configured; "
                "refusing to run a constraint channel that can never bind")
        self.constraint_manager = (
            V2ConstraintManager(spec=self.constraint_spec, dual_lr=float(dual_lr),
                                max_lambda=float(max_lambda),
                                scheduler_config_sha256=scheduler_config_sha256)
            if constraints_enabled else None)

    # -- v1-compatible surface --------------------------------------------
    @property
    def realization_epoch(self):
        """Pinned MC-realization epoch of the frozen base env (None = counter-keyed)."""
        return getattr(self.base, "realization_epoch", None)

    @realization_epoch.setter
    def realization_epoch(self, value):
        """FORWARD the assignment to the base env.

        `__getattr__` only covers reads, so `env.realization_epoch = 3` used to create a
        wrapper attribute that the frozen env never saw; the MC realization stayed keyed by
        the mutable `reset_count`. The audited requirement is explicit that this assignment
        must reach the base environment.
        """
        self.base.realization_epoch = value

    def set_realization_epoch(self, epoch) -> None:
        self.base.realization_epoch = None if epoch is None else int(epoch)

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
        """Broadcast the dual vector to this environment before a rollout.

        Held FIXED for the whole rollout; the environment only ever OBSERVES costs. The dual
        step happens once per iteration in the trainer, on training data only.
        """
        self.base.set_constraint_lambdas(lambdas)
        self.constraint_lambdas = dict(lambdas or {})
        if self.constraint_manager is not None and self.constraint_manager.enabled:
            self.constraint_manager.set_lambdas(self.constraint_lambdas)

    def constraint_state(self) -> dict:
        if self.constraint_manager is None:
            return {"enabled": False}
        return self.constraint_manager.state()

    def load_constraint_state(self, state) -> None:
        if self.constraint_manager is not None:
            self.constraint_manager.load_state(state)

    def reset(self):
        if self._world_key is not None:
            # The MC realization is drawn INSIDE `self.base.reset()`, so the epoch must be
            # pinned BEFORE that call. Setting it afterwards pinned the epoch for the NEXT
            # reset and made a replayed iteration draw a different executed sub-DAG.
            self.base.realization_epoch = stable_seed(self.base_seed, "mc_epoch",
                                                      self._world_key)
        obs = self.base.reset()
        self.episodes += 1
        indices = getattr(self.base, "graph_indices", None)
        slots = [] if indices is None else [int(s) for s in indices]
        # one seeded link process per episode (identity-keyed, not call-order keyed)
        self.link_process = None
        if self._world_key is not None:
            link_seed = stable_seed(self.base_seed, "link", self._world_key)
        else:
            link_seed = self.base_seed * 1000003 + self.episodes
        if self.link_regime != "stable":
            self.link_process = make_process(self.link_regime, seed=link_seed)
        # ONE canonical world per FLAT SLOT POSITION (not per dataset graph index: two slots
        # may legitimately hold the same dataset graph under different conditions, and keying
        # by the graph index would collapse them into one world)
        self._worlds = {}
        self._reference_ranges = {}
        self._reference_results = {}
        self._reference_ledgers = {}
        self.helper_states = []
        for flat in range(len(slots)):
            world = self._build_slot_world(flat)
            self._worlds[flat] = world
            self.helper_states.append(world.helpers)
        self._slot_spec_cache = {}
        self.last_v2_context = self._contexts()
        self.last_link_summary = {}
        self.last_energy_ledger = []
        self.last_constraint_costs = ConstraintCostBatch()
        self.last_constraint_batch = self.last_constraint_costs
        self._last_results = {}
        if self.link_process is not None and slots:
            graph0 = self.base.graph_objects[self.base._graph_index(slots[0])]
            cfg = link_spec(graph0)
            self.last_link_summary = self.link_process.summary(
                {"mec_ul": cfg.mec_ul_bytes_per_s, "mec_dl": cfg.mec_dl_bytes_per_s,
                 "v2v": cfg.v2v_bytes_per_s})
        return self._packed_observation(self.last_v2_context)

    def set_v1_schedule_trap(self, armed: bool) -> None:
        """Arm a trap that makes the V1 `_schedule` path impossible to reach silently.

        `V2AutomotiveEnv` deliberately does not implement `_schedule`; `__getattr__` would
        otherwise hand the call to the frozen v1 base environment and return v1 numbers under a
        v2 label. Arming this makes such a delegation raise.
        """
        self._v1_schedule_trap = bool(armed)

    def _schedule(self, *args, **kwargs):
        if getattr(self, "_v1_schedule_trap", False):
            raise V2EnvError(
                "the V1 scheduler `_schedule` was reached on a V2 environment: a v2 evaluation "
                "must score every plan on the canonical v2 world (this is the audited silent "
                "v1-scoring fallback)")
        return self.base._schedule(*args, **kwargs)

    def set_world_realization(self, key) -> None:
        """Fix the environmental realization by a STABLE identity key.

        The realized link process and the helper draws are derived from this key, NOT from the
        mutable episode counter. Without it, replay, CRN pairing and checkpoint resume cannot
        reproduce a world: the counter advances with every `reset()` (including the resets that
        support/query/validation perform), so the resumed run draws a different environment and
        the weights diverge by O(learning rate). Passing `None` restores counter-keyed
        behaviour (the historical default).
        """
        self._world_key = None if key is None else str(key)

    def world_realization_key(self):
        return self._world_key

    # -- canonical world construction --------------------------------------
    def _build_slot_world(self, flat: int) -> "V2World":
        """Build the world for FLAT SLOT POSITION `flat` using the SHARED builder.

        Identity is explicit and never conflated: `flat` is the batch slot position,
        `dataset_graph_id` is the graph's index in the loaded corpus (read from
        `graph_indices[flat]`), and `world_id` carries the episode and the flat position.
        """
        flat = int(flat)
        indices = np.asarray(getattr(self.base, "graph_indices", []), dtype=object).reshape(-1)
        if flat >= indices.size:
            raise V2EnvError("flat slot %d is outside graph_indices (%d entries)"
                             % (flat, indices.size))
        try:
            # `graph_indices[flat]` IS the dataset graph index of slot `flat` (v1 layout: the
            # base env resolves a slot with `graph_objects[graph_indices[slot]]`). It is NOT a
            # slot position, so it must never be fed back through `_graph_index`.
            dataset_graph_id = int(indices[flat])
        except (TypeError, ValueError) as exc:
            raise V2EnvError("graph_indices[%d] is not an integer: %r"
                             % (flat, indices[flat])) from exc
        try:
            graph = self.base.graph_objects[dataset_graph_id]
        except Exception as exc:                      # pragma: no cover - layout guard
            raise V2EnvError("flat slot %d -> graph %d has no object: %s"
                             % (flat, dataset_graph_id, exc)) from exc
        mc = self.base._slot_mc[flat] if self.base._slot_mc else None
        if self._world_key is not None:
            helper_seed = stable_seed(self.base_seed, "helper", self._world_key, flat)
        else:
            helper_seed = self.base_seed * 7919 + self.episodes * 31 + flat
        if self._world_key is not None:
            # the world identity itself is keyed by the realization, so background workloads
            # (which are derived from world_id) are reproduced too
            world_id = "w_%s_slot%d" % (self._world_key, flat)
        else:
            world_id = "ep%d_slot%d" % (self.episodes, flat)
        return build_world(
            graph, slot_id=flat, world_id=world_id, mc=mc,
            dataset_graph_id=dataset_graph_id, config=self.world_config,
            helper_seed=helper_seed)

    def world_for_slot(self, slot: int) -> "V2World":
        world = self._worlds.get(int(slot))
        if world is None:
            raise V2EnvError("no world built for slot %s (call reset() first)" % slot)
        return world

    def reference_ranges_for_slot(self, slot: int):
        """TRAIN-ONLY budget/normalisation references, measured on the v2 scheduler.

        The three pure-location worlds are built from the SLOT'S OWN graph and cached per
        episode; no validation/meta-test statistic is ever consulted.
        """
        slot = int(slot)
        if slot in self._reference_ranges:
            return self._reference_ranges[slot]
        if not self.energy_enabled and self.constraint_manager is None:
            return None
        world = self.world_for_slot(slot)
        graph = self.base.graph_objects[world.dataset_graph_id]
        mc = self.base._slot_mc[slot] if self.base._slot_mc else None
        refs_cfg = self.world_config
        if self._world_key is not None:
            ref_helper_seed = stable_seed(self.base_seed, "helper", self._world_key, slot)
        else:
            ref_helper_seed = self.base_seed * 7919 + self.episodes * 31 + slot
        worlds = pure_location_worlds(
            graph, slot_id=slot, mc=mc, dataset_graph_id=world.dataset_graph_id,
            config=refs_cfg, helper_seed=ref_helper_seed)
        plans = {name: w.schedule(validate=False) for name, w in worlds.items()}
        refs = reference_ranges_from_plans(
            plans, scheduler_config_sha256=self._scheduler_fingerprint())
        self._reference_ranges[slot] = refs
        # keep the plans AND their energy ledgers: the observation's decision-time queue/load
        # and energy channels read these, and they are pure plan-time quantities (no realized
        # link process is involved), so they cannot leak a future
        self._reference_results[slot] = plans
        self._reference_ledgers[slot] = (
            {name: schedule_energy(res) for name, res in plans.items()}
            if self.energy_enabled else {})
        return refs

    def _scheduler_fingerprint(self) -> str:
        """64-hex scheduler fingerprint required by a SCOPED `ReferenceRanges`.

        Must be a real sha256: a non-hex string is rejected by the energy scope contract, and
        a silently-swallowed rejection used to make every context call rebuild the reference
        worlds (measured: ~7 s per step).
        """
        if self.scheduler_config_sha256:
            return str(self.scheduler_config_sha256)
        import hashlib

        payload = "%s|%s" % (self.world_config.sha256(),
                             str(self.base._axes_fingerprint()))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

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
        # `graph_indices[flat]` is the DATASET GRAPH of flat slot `flat`, NOT a slot position.
        # Using it as a slot id collapsed two slots that share a graph (e.g. [0, 0]) onto the
        # SAME context, hiding differences in contact, world or helper state between
        # trajectories. The context is resolved by FLAT SLOT POSITION here.
        raw = np.asarray(getattr(self.base, "graph_indices", []), dtype=object).reshape(-1)
        if raw.size == n_rows and n_rows > 1:
            ctxs = [self._context_vector(flat) for flat in range(n_rows)]
        elif len(contexts) == n_rows:
            ctxs = [np.asarray(c, dtype=np.float32) for c in contexts]
        else:
            # single-row layout (one meta task per call): the graph is the task's graph
            gid = int(self.base.graph_objects.index(self.base.graph_objects[0])) if False else 0
            try:
                value = int(raw[0]) if raw.size else int(getattr(self.base, "task_id", 0) or 0)
                gid = min(max(value, 0), len(self.base.graph_objects) - 1)
            except (TypeError, ValueError):
                gid = 0
            ctxs = [self._context_for_graph(self.base.graph_objects[gid])] * n_rows
        out = np.empty((n_rows, rows.shape[1], V2_PACKED_DIM), dtype=np.float32)
        for row in range(n_rows):
            out[row] = pack_v2_row(rows[row], np.asarray(ctxs[row], dtype=np.float32))
        return out

    def _slot_of_graph(self, graph):
        """FIRST flat slot position whose DATASET graph is `graph` (None when unavailable).

        Resolved by dataset graph identity, never by Python object identity. When the same
        dataset graph occupies several slots this returns the first one, which is why every
        internal caller that knows its slot passes it explicitly
        (`_context_vector(flat) -> _context_for_graph(graph, slot=flat, world=...)`).
        """
        indices = np.asarray(getattr(self.base, "graph_indices", []), dtype=object).reshape(-1)
        try:
            target = int(self.base.graph_objects.index(graph))
        except ValueError:
            return None
        for flat, value in enumerate(indices):
            try:
                if int(value) == target:
                    return flat
            except (TypeError, ValueError):
                continue
        return None

    def _contexts(self) -> list:
        """One v2 context per FLAT SLOT POSITION (never per dataset graph index)."""
        indices = getattr(self.base, "graph_indices", None)
        if indices is None:
            return []
        return [self._context_vector(flat)
                for flat in range(int(np.asarray(indices).reshape(-1).size))]

    # -- execution ---------------------------------------------------------
    def _slot_specs(self, slot: int):
        """REMOVED as a construction path: kept only as a loud error.

        Training used to build each slot from `compute_spec([graph])` and schedule a single
        DAG, while the geometry gate added background DAGs — two different scientific
        problems. `V2World.build_world` is now the only builder (audited defect 3.1/3.2).
        """
        raise V2EnvError(
            "_slot_specs() is gone: the canonical world builder is v2.world.build_world "
            "(use world_for_slot(%s))" % slot)

    def _schedule_slot(self, slot: int, actions: Sequence[int], *, validate: bool = True):
        """Schedule the SLOT'S WORLD with a candidate foreground plan.

        Candidates only swap the foreground plan; the background DAGs, helper states,
        arrivals and identities are identical for every candidate, so a search or prefix
        replay can never draw a different world or consume a different exogenous stream.
        """
        world = self.world_for_slot(slot)
        graph = self.base.graph_objects[world.dataset_graph_id]
        gate = None
        evidence = None
        if self.reliability_enabled:
            from spec.automotive_training.v2.reliability import make_gate
            gate = make_gate(self._classes)
            conf = 1.0
            outage = 0.0
            if self.link_process is not None:
                # DECISION-TIME evidence only: confidence comes from the estimation model and
                # the outage fraction from the PAST window [0, t_now] (t_now = plan time = 0).
                conf = float(np.mean([self.link_process.confidence(LINK_UL),
                                      self.link_process.confidence(LINK_DL),
                                      self.link_process.confidence(LINK_V2V)]))
                outage = float(np.mean([self.link_process.past_outage_fraction(LINK_UL, 0.0),
                                        self.link_process.past_outage_fraction(LINK_DL, 0.0),
                                        self.link_process.past_outage_fraction(LINK_V2V, 0.0)]))
            evidence = {"link_confidence": conf, "outage_fraction": outage,
                        "evidence_window": "past_only:[0,plan_time]"}
        result = world.with_foreground_actions(graph, actions).schedule(
            link_process=self.link_process, reliability_gate=gate,
            reliability_evidence=evidence,
            standby_for=(standby_required if gate else None), validate=bool(validate))
        world.annotate(result)
        # the EPISODE is the foreground DAG: its completion (not the batch makespan) is the
        # latency the reward and telemetry refer to; `world_makespan_s` keeps the batch value
        result.makespan_s = float(result.episode_latency_s)
        return result

    def _telescoping(self, slot: int, actions: Sequence[int]) -> list:
        """Prefix marginal makespans under the v2 scheduler (v1 reward semantics)."""
        index = self.base._graph_index(int(slot))
        graph = self.base.graph_objects[index]
        n = len(self.base.orders[index])
        previous = self._schedule_slot(slot, [0] * n, validate=False).makespan_s
        rewards = []
        l_scale = max(float(graph.D_G_s), 1e-12)
        for k in range(n):
            prefix = [int(a) for a in actions[:k + 1]] + [0] * (n - k - 1)
            # prefixes are evaluated WITHOUT the invariant validator (pure engineering
            # speed-up: identical schedule arithmetic, skipped O(tasks x edges) re-check).
            # The episode's final schedule IS validated in step().
            current = self._schedule_slot(slot, prefix, validate=False).makespan_s
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
        self.last_energy_ledger, self.last_constraint_costs = [], []
        for slot, actions in enumerate(action):
            rewards = self._telescoping(slot, actions)
            result = self._schedule_slot(slot, actions, validate=True)
            self._last_results[int(slot)] = result
            world = self.world_for_slot(slot)
            index = self.base._graph_index(int(slot))
            graph = self.base.graph_objects[index]
            tasks = (self.base.records[index].get("tasks", [])
                     if hasattr(self.base.records[index], "get") else [])
            by_id = {int(t["task_id"]): t for t in tasks} if tasks else {}
            misses = {"HIGH": 0, "MEDIUM": 0, "other": 0}
            n_tasks = 0
            for (dag_id, tid), tm in result.timings.items():
                if dag_id != world.foreground_id:
                    continue          # the episode is the FOREGROUND DAG: denominators too
                n_tasks += 1
                deadline = None
                spec_task = by_id.get(int(tid))
                if spec_task is not None and spec_task.get("deadline_s") is not None:
                    deadline = float(spec_task["deadline_s"]) * self.runtime_deadline_scale
                if deadline is not None and tm.finish_s > deadline + 1e-12:
                    crit = str(spec_task.get("criticality", "other")).upper()
                    misses[crit if crit in ("HIGH", "MEDIUM") else "other"] += 1
            # ---- REAL energy from the event ledger (no zero-joule shim) ----
            ledger = None
            if self.energy_enabled:
                ledger = schedule_energy(result)
                self.last_energy_ledger.append(ledger)
            else:
                self.last_energy_ledger.append(None)
            # ---- constraints: measured, signed, applied once on the terminal token ----
            costs = None
            penalty = 0.0
            if self.constraint_manager is not None:
                refs = self.reference_ranges_for_slot(slot)
                self.constraint_manager.references = refs
                costs = self.constraint_manager.evaluate(result, ledger)
                penalty = self.constraint_manager.apply_penalty(rewards, costs)
                self.constraint_manager.observe(costs)
                self.last_constraint_costs.append(costs)
            else:
                self.last_constraint_costs.append(None)
            reference_energy = 0.0 if ledger is None else float(ledger.system_joules)
            telemetry = V2EpisodeTelemetry(
                slot=slot, graph_id=graph.graph_id, makespan_s=float(result.makespan_s),
                queue_wait_total_s=float(result.queue_stats["cpu_wait_mean_s"] * n_tasks),
                outage_wait_total_s=float(result.queue_stats["outage_wait_total_s"]),
                helper_rejections=int(result.queue_stats["helper_rejections_inadmissible"]),
                helper_contact_failures=int(result.queue_stats["helper_contact_failures"]),
                reliability_rejections=int(result.queue_stats["reliability_rejections"]),
                fallback_reserved_s=float(result.queue_stats["fallback_reserved_s"]),
                location_mix={loc: sum(1 for t in result.timings.values()
                                       if t.location == loc and t.dag_id == world.foreground_id)
                              for loc in (UE, MEC, "HELPER")},
                deadline_miss_rate=(sum(misses.values()) / n_tasks) if n_tasks else 0.0,
                high_miss_count=misses["HIGH"], medium_miss_count=misses["MEDIUM"],
                energy_joules=float(reference_energy), scheduler_invariants=dict(result.invariants),
                world_makespan_s=float(result.world_makespan_s or 0.0),
                requester_joules=(0.0 if ledger is None else float(ledger.requester_joules)),
                mobile_joules=(0.0 if ledger is None else float(ledger.mobile_joules)),
                system_joules=float(reference_energy),
                background_joules=(0.0 if ledger is None else float(ledger.background_joules)),
                energy_primary_scope=("" if ledger is None else str(ledger.primary_scope)),
                energy_model_sha256=("" if ledger is None else str(ledger.spec_sha256)),
                constraint_penalty=float(penalty),
                constraint_violations=(
                    {} if costs is None else {n: float(costs.violations[i])
                                              for i, n in enumerate(costs.names)}),
                constraint_signed=(
                    {} if costs is None else {n: float(costs.signed[i])
                                              for i, n in enumerate(costs.names)}),
                constraint_lambdas=(
                    {} if self.constraint_manager is None
                    else self.constraint_manager.lambdas_by_name()),
                n_tasks=int(n_tasks),
                n_helper_tasks=int(sum(1 for t in result.timings.values()
                                       if t.location == "HELPER"
                                       and t.dag_id == world.foreground_id)),
                n_mec_tasks=int(sum(1 for t in result.timings.values()
                                    if t.location == "MEC"
                                    and t.dag_id == world.foreground_id)),
                n_high_tasks=int(sum(1 for t in result.timings.values()
                                     if t.dag_id == world.foreground_id
                                     and _crit_of(by_id, t) == "HIGH")),
                n_medium_tasks=int(sum(1 for t in result.timings.values()
                                       if t.dag_id == world.foreground_id
                                       and _crit_of(by_id, t) == "MEDIUM")),
                world_id=str(world.world_id),
                world_fingerprint_sha256=str(world.fingerprint_sha256()))
            self.last_telemetry.append(telemetry)
            self.last_v2_context.append(self._context_vector(slot))
            reward_batch.append(np.asarray(rewards, dtype=np.float32))
            finish_batch.append(float(result.makespan_s))
            # the energy channel of the frozen interface now carries REAL joules
            energy_batch.append(np.full(len(rewards), float(reference_energy),
                                        dtype=np.float32))
            telemetry_batch.append(self._frozen_telemetry(slot, telemetry))
        # The FROZEN trainer reads `env.last_constraint_costs.active`/`.as_dict()`. The v2 env
        # has one cost per SLOT, so the attribute itself is a `ConstraintCostBatch` (a list
        # subclass): it stays indexable per slot AND exposes the aggregated `.active`/
        # `.as_dict()` the production loop calls. (A plain list here crashed the trainer with
        # "'list' object has no attribute 'active'" at the end of the first real iteration.)
        self.last_constraint_costs = ConstraintCostBatch(
            [c for c in self.last_constraint_costs if c is not None])
        self.last_constraint_batch = self.last_constraint_costs
        obs = self._packed_observation(self.last_v2_context)
        return obs, reward_batch, True, (finish_batch, energy_batch, telemetry_batch)

    def _frozen_telemetry(self, slot: int, telemetry) -> dict:
        """Per-slot telemetry: REAL energy in the frozen schema plus the constraint record.

        The three accounting boundaries are the measured event-based joules at the frozen
        primary scope (`system`). `energy_constraint` is `configured` when a constraint is
        active and `telemetry_only` when energy is measured but not constrained — the old
        `not_configured` + exact-zero combination is gone, so a zero can no longer be
        mistaken for a measured v1 number.
        """
        from env.mec_offloaing_envs.scheduler.energy_telemetry import (
            TELEMETRY_SCHEMA_VERSION,
        )

        index = self.base._graph_index(int(slot))
        graph = self.base.graph_objects[index]
        cfg = self.configs[index]
        v2 = telemetry.as_dict()
        primary_scope = str(telemetry.energy_primary_scope) or "system"
        ledger = (self.last_energy_ledger[slot]
                  if slot < len(self.last_energy_ledger) else None)
        constraint = (self.constraint_manager.telemetry(self.last_constraint_costs[slot])
                      if (self.constraint_manager is not None
                          and slot < len(self.last_constraint_costs)
                          and self.last_constraint_costs[slot] is not None)
                      else {"enabled": False})
        # ---- the FROZEN trainer observer reads `violation/<NAME>` at the TOP level ----
        # Without these keys `constraint_violations_from_telemetry` returns {} and the dual
        # never moves, so a budget cannot influence training at all.
        world = self._worlds.get(int(slot))
        result = self._last_results.get(int(slot))
        channel_keys: dict = {}
        if world is not None and result is not None:
            try:
                foreground = world.foreground
                criticality = {int(t.task_id): t.criticality for t in foreground.tasks}
                channel_keys = channel_telemetry(
                    graph, result, foreground_dag=foreground,
                    d_g_s=float(getattr(graph, "D_G_s", 0.0)),
                    lambdas=(self.constraint_manager.lambdas_by_name()
                             if self.constraint_manager is not None else {}),
                    penalty=float(telemetry.constraint_penalty),
                    l_scale=max(float(getattr(graph, "D_G_s", 1.0) or 1.0), 1e-12),
                    task_criticality=criticality)
            except Exception as exc:
                # a missing channel is an ERROR, never a silent zero violation
                raise V2EnvError(
                    "cannot produce the trainer constraint channels for slot %s: %s"
                    % (slot, exc)) from exc
        return {
            "schema_version": TELEMETRY_SCHEMA_VERSION,
            "requester_joules": float(telemetry.requester_joules),
            "mobile_joules": float(telemetry.mobile_joules),
            "system_joules": float(telemetry.system_joules),
            "primary_scope": primary_scope,
            "primary_joules": float(telemetry.energy_joules),
            "background_joules": float(telemetry.background_joules),
            "energy_model_sha256": str(telemetry.energy_model_sha256),
            # components that are OUT of the modelled boundary, so a missing joule is never
            # read as a measured zero
            "energy_unmodeled": list(self._unmodeled_components(slot)),
            "constraints": constraint,
            "scheduler_config_sha256": self.base._axes_fingerprint(),
            "graph_scheduler_config_sha256": str(cfg.source_config_sha256),
            "world_fingerprint_sha256": str(telemetry.world_fingerprint_sha256),
            "makespan_s": float(telemetry.makespan_s),
            "world_makespan_s": float(telemetry.world_makespan_s),
            "latency_only_objective": -float(telemetry.makespan_s) / max(float(graph.D_G_s), 1e-12),
            "energy_constraint": ("configured" if constraint.get("enabled")
                                  else "telemetry_only"),
            **channel_keys,
            "v2": v2,
        }

    def helper_busy_fraction(self, slot: int) -> float:
        """Committed helper busy time as a FRACTION of its predicted contact window.

        Separate from the `helper_busy_s` context channel on purpose: the audit flagged
        labelling busy SECONDS as a busy FRACTION.
        """
        world = self._worlds.get(int(slot))
        helper = world.helpers.get(0) if world is not None else None
        predicted = getattr(helper, "predicted_contact_end_s", None) if helper else None
        if helper is None or predicted is None or not math.isfinite(float(predicted)):
            return 0.0
        window = max(1e-9, float(predicted) - float(helper.contact_start_s))
        return float(max(0.0, min(1.0, float(helper.busy_until_s) / window)))

    def _unmodeled_components(self, slot: int) -> tuple:
        from spec.automotive_training.v2.energy import UNMODELED

        return tuple(UNMODELED)

    # -- v2 context (not yet in the TF observation) ------------------------
    def _context_vector(self, slot: int) -> np.ndarray:
        """The v2 context of ONE flat slot, resolved by IDENTITY.

        The slot's own world carries `world_id`/`slot_id`/`dataset_graph_id`, so two slots that
        contain the SAME dataset graph but different contact conditions (or a nonzero dataset
        index) each get their own correct vector: the lookup is by slot identity, never by
        object identity inside `graph_objects`.
        """
        slot = int(slot)
        world = self._worlds.get(slot)
        if world is None:
            graph = self.base.graph_objects[self.base._graph_index(slot)]
            return self._context_for_graph(graph, slot=None)
        graph = self.base.graph_objects[world.dataset_graph_id]
        return self._context_for_graph(graph, slot=slot, world=world)

    def _context_for_graph(self, graph, *, slot: int | None = None, world=None) -> np.ndarray:
        """Decision-time context. `slot` (hence the world) is optional only for legacy calls."""
        links = (LINK_UL, LINK_DL, LINK_V2V)
        est = {link: 1.0 for link in links}
        conf = {link: 1.0 for link in links}
        age = {link: 0.0 for link in links}
        observed_outage = {link: 0.0 for link in links}
        if self.link_process is not None:
            for link in links:
                est[link] = float(self.link_process.estimate_at(link, 0.0))
                conf[link] = float(self.link_process.confidence(link, 0.0))
                age[link] = float(self.link_process.estimate_age_s(link, 0.0))
                observed_outage[link] = float(
                    self.link_process.past_outage_fraction(link, 0.0))
        if world is None and slot is not None:
            world = self._worlds.get(int(slot))
        if world is None and slot is None:
            slot = self._slot_of_graph(graph)
            world = self._worlds.get(int(slot)) if slot is not None else None
        # ---- helper contact: remaining, committed BUSY SECONDS and the busy FRACTION ----
        helper = None
        if world is not None:
            helper = world.helpers.get(0)
        elif slot is not None and slot < len(self.helper_states):
            helper = self.helper_states[int(slot)].get(0)
        predicted = getattr(helper, "predicted_contact_end_s", None) if helper else None
        contact_start = float(getattr(helper, "contact_start_s", 0.0)) if helper else 0.0
        busy_s = float(getattr(helper, "busy_until_s", 0.0)) if helper else 0.0
        if predicted is None or not math.isfinite(float(predicted)):
            remaining = 0.0
            busy_fraction = 0.0
            window = float("inf")
        else:
            remaining = max(0.0, float(predicted) - 0.0)
            window = max(1e-9, float(predicted) - contact_start)
            busy_fraction = float(max(0.0, min(1.0, busy_s / window)))
        # ---- per-NODE epsilons from the criticality class of EVERY task ----------------
        counts = {"HIGH": 0, "MEDIUM": 0, "other": 0}
        eps_nodes = []
        for task in graph.tasks:
            cls = str(task.criticality).upper()
            cls = cls if cls in ("HIGH", "MEDIUM") else "LOW"
            counts[cls if cls in ("HIGH", "MEDIUM") else "other"] += 1
            eps_nodes.append(float(self._classes[cls].epsilon))
        total = max(1, sum(counts.values()))
        crit = str(graph.tasks[0].criticality).upper() if graph.tasks else "MEDIUM"
        eps_class = float(self._classes.get(crit, self._classes["MEDIUM"]).epsilon)
        eps_min = float(min(eps_nodes)) if eps_nodes else eps_class
        eps_max = float(max(eps_nodes)) if eps_nodes else eps_class
        eps_mean = float(sum(eps_nodes) / len(eps_nodes)) if eps_nodes else eps_class
        # ---- decision-time queue/load and energy of the three reference plans ----------
        waits = {"all_mec": 0.0, "all_ue": 0.0, "all_helper": 0.0}
        energies = {"all_mec": 0.0, "all_ue": 0.0, "all_helper": 0.0}
        if slot is not None and int(slot) not in self._reference_ranges:
            # FAIL FAST: the context's queue/energy/budget channels need the reference plans.
            # Swallowing a failure here silently left the cache empty and rebuilt three worlds
            # per context call on every step.
            self.reference_ranges_for_slot(int(slot))
        refs = (self._reference_ranges.get(int(slot)) if slot is not None else None)
        results = (self._reference_results.get(int(slot)) if slot is not None else None) or {}
        ledgers = (self._reference_ledgers.get(int(slot)) if slot is not None else None) or {}
        for name in waits:
            result = results.get(name)
            if result is not None:
                waits[name] = float(result.queue_stats["cpu_wait_mean_s"])
            ledger = ledgers.get(name)
            if ledger is not None:
                energies[name] = float(ledger.system_joules)
        budget_ratio = 0.0
        if self.constraint_manager is not None and refs is not None:
            spec = self.constraint_manager.spec
            if getattr(spec, "total_energy_frac_of_all_ue", None) is not None:
                budget = float(spec.total_energy_frac_of_all_ue) * float(refs.E_ue)
                budget_ratio = float(budget / max(max(energies.values()), 1e-12))
        lambdas = (self.constraint_manager.lambdas_by_name()
                   if self.constraint_manager is not None else {})
        slack = 1.0
        if helper is not None and world is not None:
            from spec.automotive_training.v2.reliability import contact_slack

            fg = world.foreground
            payload_in = max([float(t.external_input_bytes) for t in fg.tasks] or [0.0])
            slack = float(contact_slack(
                predicted_contact_end_s=predicted, now_s=0.0, payload_in_bytes=payload_in,
                compute_bytes=float(sum(t.compute_bytes for t in fg.tasks)),
                v2v_bytes_per_s=float(world.link.v2v_bytes_per_s),
                helper_bytes_per_s=float(world.compute.helper_cpu_bytes_per_s[0]),
                output_bytes=float(max([t.output_bytes for t in fg.tasks] or [0.0]))))
        background = float(world.config.background_dags) if world is not None else 0.0
        vector = [
            est[LINK_UL], est[LINK_DL], est[LINK_V2V],
            conf[LINK_UL], conf[LINK_DL], conf[LINK_V2V],
            remaining, busy_s,
            eps_class, counts["HIGH"] / total, counts["MEDIUM"] / total,
            float(world.config.mec_workers if world is not None else self.mec_workers),
            age[LINK_UL], age[LINK_DL], age[LINK_V2V],
            eps_min, eps_mean, eps_max,
            waits["all_mec"], waits["all_ue"], waits["all_helper"],
            energies["all_ue"], energies["all_mec"], energies["all_helper"],
            budget_ratio,
            float(lambdas.get("ue_energy", 0.0)), float(lambdas.get("total_energy", 0.0)),
            float(lambdas.get("helper_energy", 0.0)),
            slack, background,
        ]
        vector = np.asarray(vector, dtype=np.float32)
        if vector.shape[0] != len(V2_CONTEXT_FIELDS):
            raise V2EnvError("context width %d != %d declared fields"
                             % (vector.shape[0], len(V2_CONTEXT_FIELDS)))
        if not np.all(np.isfinite(vector)):
            raise V2EnvError("non-finite v2 context entry: %r" % (vector,))
        return vector

    def v2_context(self) -> np.ndarray:
        """[slots, len(V2_CONTEXT_FIELDS)] v2 context vector (encoder-bump pending)."""
        if not self.last_v2_context:
            raise V2EnvError("call reset()/step() before v2_context()")
        return np.asarray(self.last_v2_context, dtype=np.float32)
