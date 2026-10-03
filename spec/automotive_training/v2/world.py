#!/usr/bin/env python3
"""v2 canonical world builder — ONE construction path for training, validation, baselines,
candidate search, the geometry gate, parity and replay.

Audited defect 3.1: training built each slot from `compute_spec([graph])` and scheduled
`[dag]` alone, while the geometry gate added background DAGs — so training and the gate were
solving different scientific problems. This module removes that divergence: every consumer
calls `build_world` and receives the same structure.

World semantics (declared):

* an INDEPENDENT world per batch slot; unrelated slots are never merged;
* each world has exactly ONE foreground DAG (the episode the policy is scored on) plus
  `background_dags` DAGs owned by OTHER owners, scheduled in the SAME
  `schedule_shared` call against the SAME MEC CPU, the SAME MEC radio and the SAME V2V
  channel. Background is a declared, FROZEN workload generator (policy `background_policy`),
  not a jointly learned competitor — no multi-user learning claim is made;
* background DAGs are WORKLOAD-DERIVED from the foreground graph but not carbon copies:
  their compute/output/external byte counts and their arrival times come from a per-world
  deterministic RNG keyed by stable identities (world id, background index), so repeated
  graph instances still have distinct DAG, owner and dataset identities;
* each owner has its OWN local CPU calendar; MEC and the radios are genuinely shared;
* arrival times, tie-breaking (time, dag_id, task_id) and the scheduling discipline are
  explicit and frozen in `provenance`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence

import numpy as np

from spec.automotive_training.automotive_dag import decoder_order
from spec.automotive_training.v2.adapters import (
    compute_spec, dag_spec_from_graph, link_spec, plan_map_from_actions, pure_plan,
)
from spec.automotive_training.v2.helper_model import make_helpers
from spec.automotive_training.v2.shared_scheduler import (
    V2ComputeSpec, V2DAGSpec, V2LinkSpec, V2ScheduleResult, schedule_shared,
)

BACKGROUND_POLICIES = ("all_mec", "all_ue", "all_helper")


class V2WorldError(RuntimeError):
    """Raised on an invalid world configuration or identity collision."""


@dataclass(frozen=True)
class V2WorldConfig:
    """Everything that decides the shape of a world. Frozen and fingerprinted."""

    background_dags: int = 0
    background_owner_offset: int = 1
    background_policy: str = "all_mec"
    #: workload-derived variation band for background DAGs (fraction of the foreground
    #: byte counts). 0.0 would make every background graph an exact copy of the foreground.
    background_workload_jitter: float = 0.5
    background_arrival_spacing_s: float = 0.0
    mec_workers: int = 1
    shared_radio: bool = True
    direct_helper_v2i: bool = False
    foreground_owner: int = 0
    helper_id: int | None = 0
    helper_contact_mean_s: float = 2.0
    helper_contact_cv: float = 0.5
    helper_busy_s: float = 0.0
    enable_helpers: bool = True

    def __post_init__(self) -> None:
        if self.background_policy not in BACKGROUND_POLICIES:
            raise V2WorldError("background_policy must be one of %s, got %r"
                               % (list(BACKGROUND_POLICIES), self.background_policy))
        if int(self.background_dags) < 0:
            raise V2WorldError("background_dags must be >= 0")
        if int(self.mec_workers) < 1:
            raise V2WorldError("mec_workers must be >= 1")
        if not (0.0 <= float(self.background_workload_jitter) < 1.0):
            raise V2WorldError("background_workload_jitter must be in [0, 1)")
        if float(self.background_arrival_spacing_s) < 0.0:
            raise V2WorldError("background_arrival_spacing_s must be >= 0")

    @property
    def n_owners(self) -> int:
        return int(self.background_owner_offset) + int(self.background_dags)

    def as_dict(self) -> dict:
        return dict(self.__dict__)

    def sha256(self) -> str:
        payload = json.dumps(self.as_dict(), sort_keys=True, default=str).encode()
        return hashlib.sha256(payload).hexdigest()


@dataclass
class V2World:
    """One independent multi-DAG world, ready to schedule."""

    world_id: str
    slot_id: int
    dataset_graph_id: int
    foreground_id: str
    dags: tuple
    plans: dict
    link: V2LinkSpec
    compute: V2ComputeSpec
    helpers: dict
    arrivals: dict
    background_ids: tuple
    config: V2WorldConfig
    #: decoder order of the FOREGROUND graph, cached at build time. `plan_map_from_actions`
    #: re-canonicalises the DAG on every call, which dominated the CPU step cost once the
    #: reward evaluates one schedule per token.
    foreground_order: tuple = ()
    provenance: dict = field(default_factory=dict)

    def plan_from_actions(self, actions: Sequence[int]) -> dict:
        """{task_id: action} from a decoder-order action list, using the CACHED order."""
        actions = list(actions)
        if len(actions) != len(self.foreground_order):
            raise V2WorldError(
                "plan length %d != cached decoder order %d for world %s"
                % (len(actions), len(self.foreground_order), self.world_id))
        return {int(tid): int(a) for tid, a in zip(self.foreground_order, actions)}

    @property
    def foreground(self) -> V2DAGSpec:
        return next(d for d in self.dags if d.dag_id == self.foreground_id)

    @property
    def owners(self) -> tuple:
        return tuple(sorted({int(d.owner) for d in self.dags}))

    def schedule(self, *, link_process=None, reliability_gate=None,
                 reliability_evidence: Mapping | None = None, standby_for=None,
                 contact_margin: float = 0.9, validate: bool = True,
                 max_service_steps: int | None = None) -> V2ScheduleResult:
        """Execute THIS world. `foreground_completion` is the episode latency."""
        kwargs: dict[str, Any] = {}
        if max_service_steps is not None:
            kwargs["max_service_steps"] = int(max_service_steps)
        return schedule_shared(
            list(self.dags), dict(self.plans), link=self.link, compute=self.compute,
            link_process=link_process, helper_states=self.helpers,
            reliability_gate=reliability_gate, reliability_evidence=reliability_evidence,
            standby_for=standby_for, contact_margin=float(contact_margin),
            validate=bool(validate), **kwargs)

    def episode_latency(self, result: V2ScheduleResult) -> float:
        """The FOREGROUND DAG's completion, never the batch makespan."""
        return float(result.completion_by_dag[self.foreground_id])

    def annotate(self, result: V2ScheduleResult) -> V2ScheduleResult:
        """Record BOTH latencies: world makespan and the foreground episode latency."""
        result.world_makespan_s = float(result.makespan_s)
        result.episode_latency_s = self.episode_latency(result)
        result.foreground_dag_id = self.foreground_id
        return result

    def with_foreground_plan_map(self, plan_map: Mapping) -> "V2World":
        """Snapshot with a foreground plan given directly as {task_id: action}."""
        plans = dict(self.plans)
        plans[self.foreground_id] = {int(k): int(v) for k, v in dict(plan_map).items()}
        return replace(self, plans=plans)

    def with_foreground_actions(self, graph, actions) -> "V2World":
        """A snapshot with a DIFFERENT foreground plan and everything else untouched.

        Candidate search, prefix replay and baseline scoring all evaluate the SAME world: the
        background DAGs, helper states, arrivals, identities and the exogenous link stream do
        not depend on the candidate plan, so no candidate can draw a different world. The
        decoder order is the CACHED one, so no DAG is re-canonicalised per candidate.
        """
        plans = dict(self.plans)
        actions = list(actions)
        if len(actions) == len(self.foreground_order):
            plans[self.foreground_id] = self.plan_from_actions(actions)
        else:
            plans[self.foreground_id] = plan_map_from_actions(graph, actions)
        return replace(self, plans=plans)

    def structure_fingerprint_sha256(self) -> str:
        """Identity of the WORLD STRUCTURE: everything except the foreground plan.

        A candidate change must leave this unchanged, which is what "no candidate regenerates
        the world" means operationally. `fingerprint_sha256()` includes the plans and is the
        full identity of a scored configuration.
        """
        payload = {
            "config": self.config.as_dict(), "world_id": self.world_id,
            "slot_id": self.slot_id, "dataset_graph_id": self.dataset_graph_id,
            "foreground_id": self.foreground_id,
            "dags": sorted(d.dag_id for d in self.dags),
            "owners": sorted(int(d.owner) for d in self.dags),
            "arrivals": {k: float(v) for k, v in sorted(self.arrivals.items())},
            "helpers": {k: [v.cpu_bytes_per_s, v.busy_until_s, v.contact_start_s,
                            v.contact_end_s, v.predicted_contact_end_s]
                        for k, v in sorted(self.helpers.items())},
            "link": vars(self.link),
            "compute": [self.compute.mec_cpu_bytes_per_s, self.compute.mec_workers,
                        list(self.compute.ue_cpu_bytes_per_s),
                        list(self.compute.helper_cpu_bytes_per_s)],
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()

    def fingerprint_sha256(self) -> str:
        payload = {
            "config": self.config.as_dict(),
            "world_id": self.world_id,
            "slot_id": self.slot_id,
            "dataset_graph_id": self.dataset_graph_id,
            "foreground_id": self.foreground_id,
            "dags": [{"dag_id": d.dag_id, "owner": d.owner, "arrival_s": d.arrival_s,
                      "helper_id": d.helper_id,
                      "tasks": [[t.task_id, t.compute_bytes, t.output_bytes,
                                 list(t.predecessors), t.is_root, t.is_sink,
                                 t.criticality, t.deadline_s, t.owner,
                                 t.external_input_bytes] for t in d.tasks]}
                     for d in self.dags],
            "focus_order_len": len(self.foreground_order),
            "plans": {k: dict(sorted(v.items())) if isinstance(v, Mapping) else list(v)
                      for k, v in sorted(self.plans.items())},
            "helpers": {k: {"cpu": v.cpu_bytes_per_s, "busy": v.busy_until_s,
                            "start": v.contact_start_s, "end": v.contact_end_s,
                            "predicted": v.predicted_contact_end_s}
                        for k, v in sorted(self.helpers.items())},
            "link": vars(self.link), "compute": {
                "mec_cpu_bytes_per_s": self.compute.mec_cpu_bytes_per_s,
                "mec_workers": self.compute.mec_workers,
                "ue_cpu_bytes_per_s": list(self.compute.ue_cpu_bytes_per_s),
                "helper_cpu_bytes_per_s": list(self.compute.helper_cpu_bytes_per_s)},
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def _jitter_bytes(value: float, rng: np.random.RandomState, band: float) -> float:
    """Workload-derived background byte count inside [1-band, 1+band] of the foreground."""
    if band <= 0.0:
        return float(value)
    factor = 1.0 + float(rng.uniform(-band, band))
    return float(max(1.0, round(float(value) * factor)))


def build_background_dag(graph, *, dag_id: str, owner: int, mc, config: V2WorldConfig,
                         rng: np.random.RandomState, arrival_s: float,
                         direct_helper_v2i: bool = False) -> V2DAGSpec:
    """A workload-derived background DAG with its OWN identity and byte counts."""
    dag = dag_spec_from_graph(graph, dag_id=dag_id, owner=int(owner), mc=mc,
                              helper_id=(config.helper_id if config.enable_helpers
                                         and config.background_policy == "all_helper" else None),
                              arrival_s=float(arrival_s))
    band = float(config.background_workload_jitter)
    tasks = []
    for task in dag.tasks:
        tasks.append(replace(
            task,
            compute_bytes=_jitter_bytes(task.compute_bytes, rng, band),
            output_bytes=_jitter_bytes(task.output_bytes, rng, band),
            external_input_bytes=_jitter_bytes(task.external_input_bytes, rng, band)))
    return replace(dag, tasks=tasks)


def build_world(graph, *, slot_id: int = 0, world_id: str | None = None,
                mc: Mapping | None = None, dataset_graph_id: int | None = None,
                config: V2WorldConfig | None = None,
                foreground_actions: Sequence[int] | None = None,
                helper_seed: int = 0, link: V2LinkSpec | None = None,
                compute: V2ComputeSpec | None = None) -> V2World:
    """Build one independent world: the foreground DAG plus its background competitors."""
    config = config or V2WorldConfig()
    slot_id = int(slot_id)
    world_id = str(world_id if world_id is not None else "w%d" % slot_id)
    dataset_graph_id = int(dataset_graph_id if dataset_graph_id is not None else slot_id)
    owner = int(config.foreground_owner)
    n_helpers = max(1, max(owner, config.n_owners))
    link = link or link_spec(graph, shared_radio=config.shared_radio,
                             direct_helper_v2i=config.direct_helper_v2i)
    compute = compute or compute_spec([graph], mec_workers=config.mec_workers,
                                      helper_ids=list(range(n_helpers)))
    fg_id = "%s_fg" % world_id
    foreground_order = tuple(int(t) for t in decoder_order(graph.as_record()))
    fg = dag_spec_from_graph(
        graph, dag_id=fg_id, owner=owner, mc=mc,
        helper_id=(int(config.helper_id) if config.enable_helpers
                   and config.helper_id is not None else None))
    dags = [fg]
    plans = {fg_id: (plan_map_from_actions(graph, foreground_actions)
                     if foreground_actions is not None else pure_plan(graph, 0))}
    if config.background_policy == "all_mec":
        bg_action = 1
    elif config.background_policy == "all_ue":
        bg_action = 0
    else:
        bg_action = 2
    background_ids = []
    for b in range(int(config.background_dags)):
        bg_id = "%s_bg%d" % (world_id, b)
        rng = np.random.RandomState(_stable_seed(world_id, b))
        arrival = float(b) * float(config.background_arrival_spacing_s)
        bg = build_background_dag(graph, dag_id=bg_id,
                                  owner=int(config.background_owner_offset) + b, mc=mc,
                                  config=config, rng=rng, arrival_s=arrival)
        if not bg.tasks:
            raise V2WorldError("background DAG %s has no surviving tasks" % bg_id)
        dags.append(bg)
        plans[bg_id] = {t.task_id: bg_action for t in bg.tasks}
        background_ids.append(bg_id)
    # identity check: every DAG in the world must be uniquely identifiable
    ids = [d.dag_id for d in dags]
    if len(set(ids)) != len(ids):
        raise V2WorldError("duplicate DAG identities in world %s: %s" % (world_id, ids))
    # helpers: one per owner index, seeded per world (deterministic under replay)
    helpers: dict = {}
    if config.enable_helpers:
        ids_needed = sorted({int(d.helper_id) for d in dags if d.helper_id is not None})
        specs = {}
        for hid in ids_needed:
            idx = min(int(hid), len(compute.helper_cpu_bytes_per_s) - 1)
            specs[int(hid)] = {
                "cpu_bytes_per_s": float(compute.helper_cpu_bytes_per_s[idx]),
                "busy_until_s": float(config.helper_busy_s),
                "owner": int(hid) + int(config.background_owner_offset),
            }
        helpers = make_helpers(specs, seed=int(helper_seed),
                               contact_mean_s=float(config.helper_contact_mean_s),
                               contact_cv=float(config.helper_contact_cv))
    provenance = {
        "builder": "v2.world.build_world",
        "config_sha256": config.sha256(),
        "link_regime_direct_helper_v2i": bool(config.direct_helper_v2i),
        "background_policy": config.background_policy,
        "background_policy_frozen": True,
        "background_identity_rule": "world_id + '_bg' + index",
        "background_jitter": float(config.background_workload_jitter),
        "tie_break": "(time, dag_id, task_id)",
        "calendar_discipline": "non-preemptive earliest-fit, one booking per transfer",
        "channel_discipline": "interrupted transmission RETAINS the channel",
        "episode_latency": "foreground DAG completion (not the batch makespan)",
        "independent_world_per_slot": True,
        "multi_user_learning_claim": False,
    }
    return V2World(world_id=world_id, slot_id=slot_id,
                   dataset_graph_id=dataset_graph_id, foreground_id=fg_id,
                   dags=tuple(dags), plans=plans, link=link, compute=compute,
                   helpers=helpers, arrivals={d.dag_id: d.arrival_s for d in dags},
                   background_ids=tuple(background_ids), config=config,
                   foreground_order=foreground_order, provenance=provenance)


def _stable_seed(world_id: str, index: int) -> int:
    """Exogenous randomness keyed by STABLE identity, never by RNG call order."""
    payload = ("%s|bg|%d" % (world_id, int(index))).encode()
    return int(hashlib.sha256(payload).hexdigest()[:12], 16) % (2 ** 31 - 1)


def pure_location_worlds(graph, *, slot_id: int = 0, mc: Mapping | None = None,
                         dataset_graph_id: int | None = None,
                         config: V2WorldConfig | None = None,
                         helper_seed: int = 0) -> dict:
    """The three all-UE / all-MEC / all-HELPER reference worlds for one graph.

    Used for TRAIN-ONLY budget calibration and normalisation. Background is disabled here on
    purpose: a reference plan is a property of the foreground graph, and adding background
    would make the reference depend on the competitor load.
    """
    base = config or V2WorldConfig()
    ref_config = replace(base, background_dags=0)
    out = {}
    for name, action in (("all_ue", 0), ("all_mec", 1), ("all_helper", 2)):
        n_tasks = len(pure_plan(graph, action))
        out[name] = build_world(graph, slot_id=slot_id, world_id="ref_%s" % name, mc=mc,
                                dataset_graph_id=dataset_graph_id, config=ref_config,
                                foreground_actions=[action] * n_tasks,
                                helper_seed=helper_seed)
    return out


__all__ = [
    "BACKGROUND_POLICIES", "V2World", "V2WorldConfig", "V2WorldError",
    "build_background_dag", "build_world", "pure_location_worlds",
]
