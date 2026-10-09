#!/usr/bin/env python3
"""Adapters: frozen v1 graphs -> v2 shared-scheduler inputs.

Explicit v2 modelling choices (recorded, not hidden):

* **Shared MEC capacity.** In v1 the MEC rate is a per-graph profile draw. A shared MEC is
  an infrastructure property, so v2 uses one capacity for the concurrent set. Default:
  the MEDIAN of the participating graphs' `mec_cpu_bytes_per_second` (overridable). This
  is an explicit assumption, not a literature constant.
* **Per-vehicle resources.** UE CPU and the radio rates stay per-vehicle (each graph keeps
  its own realized profile) because they belong to the vehicle, not to the infrastructure.
* **Helper CPU.** One calendar per helper id, rate from the owning graph's
  `helper_cpu_bytes_per_second`.
"""

from __future__ import annotations

import statistics
from typing import Any, Mapping, Sequence

from spec.automotive_training.automotive_dag import canonical_dag, decoder_order
from spec.automotive_training.automotive_resources import co_physical_config_for_graph

from .shared_scheduler import (
    HELPER,
    MEC,
    UE,
    V2ComputeSpec,
    V2DAGSpec,
    V2LinkSpec,
    V2ScheduleError,
    V2TaskSpec,
)


#: V2 resource/timing/energy contract. v1 scheduled with the PER-GRAPH dataset frequency
#: (`f_dataset / (8*xi)`) while the event ledger priced CPU energy at the FROZEN tier frequency,
#: so the two identities `E = P*T` and `E = kappa*C*f^2` referred to different `f`
#: (measured: dataset 422 MHz / 845 MHz / 6.74 GHz vs frozen 1 / 1.5 / 10 GHz, i.e.
#: `E_duration/E_work = R_physical/R_scheduled != 1`). v2 uses ONE frequency for scheduling AND
#: pricing: the rate is the physical rate at the frozen tier frequency,
#: `R = f_frozen / (8*xi)`, which makes `E_duration == E_work` exactly by construction.
V2_RESOURCE_CONTRACT = "frozen_tier_f_v2"


def config_for(graph) -> Any:
    """V2 scheduler config under the resolved resource contract (one frequency for T and E)."""
    cfg = co_physical_config_for_graph(graph.resource, source_sha256="v2-adapter")
    return _apply_v2_resource_contract(cfg, graph)


def _apply_v2_resource_contract(cfg, graph):
    """Rewrite the CPU service rates to the physical rates at the ENERGY-PRICING frequency.

    The ledger prices dynamic power as `P = kappa*f^3` with the frozen tier frequency, so the
    duration must be `T = C/f_frozen = bytes*8*xi/f_frozen`. Scheduling with any other frequency
    silently breaks `E = P*T == kappa*C*f^2`.
    """
    from spec.automotive_training.automotive_resources import dataset_reference_parameters
    from spec.automotive_training.v2.energy import frozen_energy_spec
    xi = float(dataset_reference_parameters().get("xi_cycles_per_bit"))
    if not (xi > 0.0):
        raise ValueError("xi_cycles_per_bit must be positive, got %r" % (xi,))
    tiers = frozen_energy_spec().tiers
    rates = {
        "ue_cpu_bytes_per_second": float(tiers["ue"].f_hz) / (8.0 * xi),
        "helper_cpu_bytes_per_second": float(tiers["helper"].f_hz) / (8.0 * xi),
        "mec_cpu_bytes_per_second": float(tiers["mec"].f_hz) / (8.0 * xi),
    }
    try:
        import dataclasses
        if dataclasses.is_dataclass(cfg):
            return dataclasses.replace(cfg, **rates)
        for k, v in rates.items():
            setattr(cfg, k, v)
        return cfg
    except Exception as exc:                                     # pragma: no cover
        raise RuntimeError(
            "could not apply the V2 resource contract %r to the scheduler config: %s"
            % (V2_RESOURCE_CONTRACT, exc)) from exc


def shared_mec_capacity(graphs: Sequence, *, mode: str = "median") -> float:
    values = [config_for(g).mec_cpu_bytes_per_second for g in graphs]
    if mode == "median":
        return float(statistics.median(values))
    if mode == "min":
        return float(min(values))
    if mode == "max":
        return float(max(values))
    raise V2ScheduleError("unknown shared MEC capacity mode %r" % mode)


def compute_spec(graphs: Sequence, *, mec_workers: int = 1, mec_capacity: float | None = None,
                 helper_ids: Sequence[int] | None = None) -> V2ComputeSpec:
    caps = [config_for(g) for g in graphs]
    n = len(graphs)
    helpers = helper_ids if helper_ids is not None else [None] * n
    helper_rates = []
    for i, g in enumerate(graphs):
        helper_rates.append(float(config_for(g).helper_cpu_bytes_per_second))
    return V2ComputeSpec(
        mec_cpu_bytes_per_s=float(mec_capacity if mec_capacity is not None
                                  else statistics.median([c.mec_cpu_bytes_per_second for c in caps])),
        mec_workers=int(mec_workers),
        ue_cpu_bytes_per_s=tuple(float(c.ue_cpu_bytes_per_second) for c in caps),
        helper_cpu_bytes_per_s=tuple(helper_rates),
    )


def link_spec(graph, *, shared_radio: bool = True, direct_helper_v2i: bool = False) -> V2LinkSpec:
    cfg = config_for(graph)
    return V2LinkSpec(mec_ul_bytes_per_s=float(cfg.mec_uplink_bytes_per_second),
                      mec_dl_bytes_per_s=float(cfg.mec_downlink_bytes_per_second),
                      v2v_bytes_per_s=float(cfg.v2v_bytes_per_second),
                      direct_helper_v2i=bool(direct_helper_v2i),
                      shared_radio=bool(shared_radio))


def dag_spec_from_graph(graph, *, dag_id: str, owner: int, arrival_s: float = 0.0,
                        helper_id: int | None = None, mc: Mapping | None = None,
                        helper_contact_window: tuple | None = None) -> V2DAGSpec:
    """Build a V2DAGSpec, applying the same MC survivors/effective work as the v1 env."""
    record = graph.as_record()
    dag = canonical_dag(record)
    survivors = None
    execution = None
    if mc is not None:
        survivors = set(int(t) for t in mc["executed_task_ids"])
        execution = mc["execution"]
    tasks = []
    for tid, task in sorted(dag.tasks.items()):
        if survivors is not None and int(tid) not in survivors:
            continue
        if execution is not None:
            compute_bytes = max(1, int(round(float(execution[int(tid)]["effective_equiv"]))))
        else:
            compute_bytes = float(task.compute_workload_bytes)
        preds = [int(e.src_task_id) for e in (dag.predecessors().get(int(tid), []) or [])]
        if survivors is not None:
            preds = [p for p in preds if p in survivors]
        tasks.append(V2TaskSpec(
            task_id=int(tid), compute_bytes=float(compute_bytes),
            output_bytes=float(task.task_output_bytes), predecessors=tuple(preds),
            is_root=not preds,
            is_sink=int(tid) in set(int(s) for s in (dag.sinks() if callable(dag.sinks) else dag.sinks)),
            criticality=str(task.criticality_class).upper(), deadline_s=float(task.deadline_s)
            if task.deadline_s is not None else None,
            owner=int(owner),
            external_input_bytes=float(task.external_input_bytes)))
    # Sink semantics under the MC realization (v1 engine parity): a task must deliver its
    # output to the vehicle if it has no SURVIVING successor, not only if it is a static
    # sink. When the MC filter drops every successor of task X, X becomes the endpoint of
    # the executed sub-DAG and its output has to be returned; taking only `dag.sinks()`
    # silently under-booked those returns (measured: one 185 B return per graph).
    kept = {int(t.task_id) for t in tasks}
    static_sinks = {int(t.task_id) for t in tasks if t.is_sink}
    def _succ_ids(tid):
        raw = dag.successors().get(int(tid), []) or []
        out = set()
        for item in raw:
            out.add(int(getattr(item, "dst_task_id", item)))
        return out

    successor_ids = {int(tid): _succ_ids(tid) for tid in kept}
    tasks = [
        V2TaskSpec(
            task_id=t.task_id, compute_bytes=t.compute_bytes, output_bytes=t.output_bytes,
            predecessors=t.predecessors, is_root=t.is_root,
            is_sink=(t.task_id in static_sinks
                     or not (successor_ids.get(t.task_id, set()) & kept)),
            criticality=t.criticality, deadline_s=t.deadline_s, owner=t.owner,
            external_input_bytes=t.external_input_bytes)
        for t in tasks
    ]
    return V2DAGSpec(dag_id=str(dag_id), owner=int(owner), tasks=tasks,
                     arrival_s=float(arrival_s), helper_id=helper_id,
                     helper_contact_window=helper_contact_window)


def plan_map_from_actions(graph, actions: Sequence[int]) -> dict:
    """Decoder-order action list -> {task_id: action}."""
    order = decoder_order(graph.as_record())
    if len(actions) != len(order):
        raise V2ScheduleError("plan length %d != decoder order %d" % (len(actions), len(order)))
    return {int(tid): int(a) for tid, a in zip(order, actions)}


def pure_plan(graph, action: int) -> dict:
    return {int(tid): int(action) for tid in decoder_order(graph.as_record())}
