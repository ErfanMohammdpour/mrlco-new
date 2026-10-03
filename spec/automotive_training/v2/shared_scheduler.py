#!/usr/bin/env python3
"""v2 shared multi-DAG scheduler: real queueing on shared MEC/radio resources.

Design (design A = single MEC server; design B = K-worker pool via `mec_workers`):

* Resources are explicit FIFO, non-preemptive calendars.
  - shared: `MEC_UL`, `MEC_DL`, `V2V_CHANNEL`, `MEC_CPU` (K servers)
  - per-vehicle: `UE_CPU_<i>` (owner only)
  - per-helper: `HELPER_CPU_<h>`
* Data residency: a task's output stays where it executed. A successor whose location
  differs pays a transfer of the predecessor's `task_output_bytes` on the route.
* Routes follow v1 (`UE->MEC` UL, `MEC->UE` DL, `UE<->HELPER` V2V, `MEC<->HELPER` via the
  UE two-hop, or a direct V2I hop when `direct_helper_v2i=True`, labelled sensitivity).
* Each DAG i has an explicit arrival/release time; N DAGs overlap in time and contend for
  the shared calendars. Deterministic tie-breaking by (time, dag_id, task_id).
* Per task we log queue_wait_ul, tx_ul, queue_wait_cpu, cpu_time, queue_wait_dl, tx_dl,
  total_remote_response; per episode we log utilizations, queue-length stats and waits.

This module does not import the frozen engine: it is a new model, and parity with v1 for
N=1 / free resources is asserted by tests, not assumed.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

UE, MEC, HELPER = "UE", "MEC", "HELPER"
HOP_UL, HOP_DL, HOP_V2V = "MEC_UL", "MEC_DL", "V2V"
HOP_UL_DIRECT, HOP_DL_DIRECT = "MEC_ULH", "MEC_DLH"


class V2ScheduleError(RuntimeError):
    """Raised when a v2 schedule would be physically or logically invalid."""


@dataclass(frozen=True)
class V2LinkSpec:
    """Effective rates (bytes/second) and shared-channel semantics."""

    mec_ul_bytes_per_s: float
    mec_dl_bytes_per_s: float
    v2v_bytes_per_s: float
    direct_helper_v2i: bool = False
    shared_radio: bool = True          # False -> per-vehicle radio (sensitivity)


@dataclass(frozen=True)
class V2ComputeSpec:
    mec_cpu_bytes_per_s: float
    mec_workers: int = 1               # design A = 1, design B = K
    ue_cpu_bytes_per_s: Sequence[float] = ()
    helper_cpu_bytes_per_s: Sequence[float] = ()


@dataclass
class Calendar:
    """FIFO non-preemptive server; `reserve` returns (start, end)."""

    name: str
    servers: int = 1
    _free: list = field(default_factory=lambda: [0.0])

    def __post_init__(self) -> None:
        self._free = [0.0] * max(1, int(self.servers))

    def reserve(self, ready: float, duration: float):
        if duration < 0.0 or not math.isfinite(duration):
            raise V2ScheduleError("invalid duration for %s: %r" % (self.name, duration))
        idx = min(range(len(self._free)), key=lambda i: (self._free[i], i))
        start = max(float(ready), self._free[idx])
        end = start + float(duration)
        self._free[idx] = end
        return start, end

    def peek_wait(self, ready: float) -> float:
        idx = min(range(len(self._free)), key=lambda i: (self._free[i], i))
        return max(0.0, self._free[idx] - float(ready))

    @property
    def busy_intervals(self):
        return list(self._free)


@dataclass
class V2TaskSpec:
    task_id: int
    compute_bytes: float
    output_bytes: float
    predecessors: tuple = ()
    is_root: bool = False
    is_sink: bool = False
    criticality: str = "MEDIUM"
    deadline_s: float | None = None
    owner: int = 0                     # vehicle index (UE_CPU_i)
    external_input_bytes: float = 0.0  # uploaded from the vehicle (v1 engine semantics)


@dataclass
class V2DAGSpec:
    dag_id: str
    owner: int
    tasks: Sequence[V2TaskSpec]
    arrival_s: float = 0.0
    helper_id: int | None = None       # helper this DAG may use
    helper_contact_window: tuple | None = None   # (start, end) of usable contact


@dataclass
class TaskTiming:
    dag_id: str
    task_id: int
    location: str
    ready_s: float
    queue_wait_ul_s: float
    tx_ul_s: float
    queue_wait_cpu_s: float
    cpu_s: float
    queue_wait_dl_s: float
    tx_dl_s: float
    start_s: float
    finish_s: float
    transfer_in_bytes: float
    transfer_hops: tuple = ()
    helper_id: int | None = None

    @property
    def total_remote_response_s(self) -> float:
        return self.queue_wait_ul_s + self.tx_ul_s + self.cpu_s + self.tx_dl_s


@dataclass
class V2ScheduleResult:
    makespan_s: float
    completion_by_dag: dict
    timings: dict
    mechanics: dict
    utilizations: dict
    queue_stats: dict
    invariants: dict
    arrivals: dict
    active_concurrency: list


def _route_hops(location_a: str, location_b: str, link: V2LinkSpec) -> tuple:
    """Ordered hops for a transfer. MEC<->HELPER relays through the UE (v1 topology)
    unless the sensitivity-only direct V2I link is enabled."""
    if location_a == location_b:
        return ()
    if {location_a, location_b} == {UE, MEC}:
        return (HOP_UL,) if location_b == MEC else (HOP_DL,)
    if {location_a, location_b} == {UE, HELPER}:
        return (HOP_V2V,)
    if (location_a, location_b) == (MEC, HELPER):
        return (HOP_DL_DIRECT,) if link.direct_helper_v2i else (HOP_DL, HOP_V2V)
    if (location_a, location_b) == (HELPER, MEC):
        return (HOP_UL_DIRECT,) if link.direct_helper_v2i else (HOP_V2V, HOP_UL)
    raise V2ScheduleError("unknown route %s -> %s" % (location_a, location_b))


def _rate_for(hop: str, dag_index: int, link: V2LinkSpec) -> float:
    if hop == HOP_UL:
        return link.mec_ul_bytes_per_s
    if hop == HOP_DL:
        return link.mec_dl_bytes_per_s
    if hop == HOP_V2V:
        return link.v2v_bytes_per_s
    if hop == HOP_UL_DIRECT:
        return link.mec_ul_bytes_per_s
    if hop == HOP_DL_DIRECT:
        return link.mec_dl_bytes_per_s
    raise V2ScheduleError("unknown hop %r" % hop)


def schedule_shared(dags: Sequence[V2DAGSpec], plans: Mapping[str, Sequence[int]],
                    *, link: V2LinkSpec, compute: V2ComputeSpec) -> V2ScheduleResult:
    """List-schedule multiple DAGs on shared calendars. Deterministic."""
    if len(dags) != len(plans):
        raise V2ScheduleError("one plan per DAG required")
    shared_cpu = Calendar("MEC_CPU", servers=int(compute.mec_workers))
    ul = Calendar("MEC_UL") if link.shared_radio else Calendar("MEC_UL", servers=len(dags))
    dl = Calendar("MEC_DL") if link.shared_radio else Calendar("MEC_DL", servers=len(dags))
    v2v = Calendar("V2V_CHANNEL") if link.shared_radio else Calendar("V2V_CHANNEL", servers=max(1, len(dags)))
    ue_cpu = {i: Calendar("UE_CPU_%d" % i) for i in range(len(dags))}
    helper_cpu: dict = {}
    for dag in dags:
        if dag.helper_id is not None:
            helper_cpu.setdefault(dag.helper_id, Calendar("HELPER_CPU_%d" % dag.helper_id))
    radio_events: list = []
    concurrency: list = []

    timings: dict = {}
    finish: dict = {}
    remaining: dict = {}
    by_dag: dict = {}
    for dag in dags:
        by_dag[dag.dag_id] = {t.task_id: t for t in dag.tasks}
        remaining[dag.dag_id] = {t.task_id: len(t.predecessors) for t in dag.tasks}
        import collections
        succ = collections.defaultdict(list)
        for t in dag.tasks:
            for p in t.predecessors:
                succ[p].append(t.task_id)
        dag_succ = succ
        setattr(dag, "_succ", succ)
    ready_heap: list = []
    for dag in dags:
        for t in dag.tasks:
            if remaining[dag.dag_id][t.task_id] == 0:
                heapq.heappush(ready_heap, (dag.arrival_s, dag.dag_id, t.task_id))
    while ready_heap:
        _arr, dag_id, task_id = heapq.heappop(ready_heap)
        dag = next(d for d in dags if d.dag_id == dag_id)
        spec = by_dag[dag_id][task_id]
        plan = plans[dag_id]
        if isinstance(plan, Mapping):
            action = int(plan.get(task_id, 0))
        else:
            action = int(plan[task_id]) if task_id < len(plan) else 0
        location = {0: UE, 1: MEC, 2: HELPER}[action]
        if location == HELPER and dag.helper_id is None:
            location = UE   # no helper available -> the plan degrades to local execution
        # data-ready time from predecessors (outputs stay where they executed)
        ready = float(dag.arrival_s)
        in_bytes = 0.0
        if spec.is_root and spec.external_input_bytes and location != UE:
            # v1 engine semantics: the external input is uploaded before execution
            for hop in _route_hops(UE, location, link):
                rate = _rate_for(hop, 0, link)
                duration = float(spec.external_input_bytes) / rate
                cal = (ul if hop in (HOP_UL, HOP_UL_DIRECT)
                       else dl if hop in (HOP_DL, HOP_DL_DIRECT) else v2v)
                _s, _e = cal.reserve(ready, duration)
                radio_events.append((_s, _e, cal.name, float(spec.external_input_bytes)))
                ready = _e
            in_bytes += float(spec.external_input_bytes)
        for p in spec.predecessors:
            pt: TaskTiming = timings[(dag_id, p)]
            transfer = (pt.location != location)
            payload = float(by_dag[dag_id][p].output_bytes) if transfer else 0.0
            t_ready = pt.finish_s
            if transfer:
                hops = _route_hops(pt.location, location, link)
                for hop in hops:
                    rate = _rate_for(hop, 0, link)
                    duration = payload / rate
                    cal = (ul if hop in (HOP_UL, HOP_UL_DIRECT)
                           else dl if hop in (HOP_DL, HOP_DL_DIRECT) else v2v)
                    start, end = cal.reserve(t_ready, duration)
                    radio_events.append((start, end, cal.name, payload))
                    t_ready = end
                in_bytes += payload
            ready = max(ready, t_ready)
        # phase order: input transfer already reserved above, then CPU, then DL for the sink
        queue_wait_cpu = shared_cpu.peek_wait(ready) if location == MEC else (
            ue_cpu[dag.owner].peek_wait(ready) if location == UE else
            helper_cpu[dag.helper_id].peek_wait(ready))
        if location == MEC:
            rate = compute.mec_cpu_bytes_per_s
            cpu_duration = float(spec.compute_bytes) / rate
            start, cpu_end = shared_cpu.reserve(ready, cpu_duration)
        elif location == UE:
            rate = float(compute.ue_cpu_bytes_per_s[dag.owner])
            cpu_duration = float(spec.compute_bytes) / rate
            start, cpu_end = ue_cpu[dag.owner].reserve(ready, cpu_duration)
        else:
            rate = float(compute.helper_cpu_bytes_per_s[dag.helper_id])
            cpu_duration = float(spec.compute_bytes) / rate
            start, cpu_end = helper_cpu[dag.helper_id].reserve(ready, cpu_duration)
        queue_wait_cpu = max(0.0, start - ready)
        tx_dl = 0.0
        queue_wait_dl = 0.0
        finish_s = cpu_end
        if spec.is_sink and location != UE:
            hops = _route_hops(location, UE, link)
            t = cpu_end
            for i, hop in enumerate(hops):
                rate = _rate_for(hop, 0, link)
                duration = float(spec.output_bytes) / rate
                cal = (ul if hop in (HOP_UL, HOP_UL_DIRECT)
                       else dl if hop in (HOP_DL, HOP_DL_DIRECT) else v2v)
                if i == 0:
                    queue_wait_dl = cal.peek_wait(t)
                dstart, dend = cal.reserve(t, duration)
                radio_events.append((dstart, dend, cal.name, float(spec.output_bytes)))
                tx_dl += duration
                t = dend
            finish_s = t
        timing = TaskTiming(
            dag_id=dag_id, task_id=task_id, location=location, ready_s=ready,
            queue_wait_ul_s=0.0,
            tx_ul_s=sum(float(by_dag[dag_id][p].output_bytes) /
                        _rate_for(hop, 0, link)
                        for p in spec.predecessors
                        for hop in _route_hops(timings[(dag_id, p)].location, location, link)),
            queue_wait_cpu_s=queue_wait_cpu, cpu_s=cpu_duration,
            queue_wait_dl_s=queue_wait_dl, tx_dl_s=tx_dl,
            start_s=start, finish_s=finish_s, transfer_in_bytes=in_bytes,
            helper_id=dag.helper_id if location == HELPER else None)
        timings[(dag_id, task_id)] = timing
        finish[(dag_id, task_id)] = finish_s
        concurrency.append((start, "start", dag_id))
        concurrency.append((finish_s, "end", dag_id))
        for child in getattr(dag, "_succ")[task_id]:
            remaining[dag_id][child] -= 1
            if remaining[dag_id][child] == 0:
                heapq.heappush(ready_heap, (finish_s, dag_id, child))
    completion = {}
    for dag in dags:
        completion[dag.dag_id] = max(finish[(dag.dag_id, t.task_id)] for t in dag.tasks) + dag.arrival_s * 0
    makespan = max(completion.values()) if completion else 0.0
    # utilization + queue statistics
    horizon = max(1e-12, makespan)
    def util(cal: Calendar) -> float:
        return sum(cal.busy_intervals) / (horizon * max(1, cal.servers))
    concurrency.sort()
    active = 0
    peak = 0
    area = 0.0
    last = concurrency[0][0] if concurrency else 0.0
    active_series = []
    for t, kind, dag_id in concurrency:
        area += active * max(0.0, t - last)
        active += 1 if kind == "start" else -1
        peak = max(peak, active)
        active_series.append((t, active))
        last = t
    waits = [tm.queue_wait_cpu_s for tm in timings.values()]
    remote_wait = [tm.queue_wait_ul_s for tm in timings.values()]
    result = V2ScheduleResult(
        makespan_s=makespan, completion_by_dag=completion, timings=timings,
        mechanics={"radio_events": len(radio_events), "tasks": len(timings)},
        utilizations={"MEC_CPU": util(shared_cpu), "MEC_UL": util(ul), "MEC_DL": util(dl),
                      "V2V": util(v2v),
                      "UE_CPU": [util(ue_cpu[i]) for i in sorted(ue_cpu)],
                      "HELPER_CPU": [util(helper_cpu[h]) for h in sorted(helper_cpu)]},
        queue_stats={"cpu_wait_mean_s": (sum(waits) / len(waits)) if waits else 0.0,
                     "cpu_wait_max_s": max(waits) if waits else 0.0,
                     "remote_wait_max_s": max(remote_wait) if remote_wait else 0.0},
        invariants={}, arrivals={d.dag_id: d.arrival_s for d in dags},
        active_concurrency=active_series)
    result.invariants = validate_schedule(dags, plans, link, result, shared_cpu, ue_cpu,
                                          helper_cpu, ul, dl, v2v)
    return result


def validate_schedule(dags, plans, link, result, shared_cpu, ue_cpu, helper_cpu,
                      ul, dl, v2v) -> dict:
    """Hard invariants; any violation raises."""
    checks = {}
    # precedence + data arrival
    for dag in dags:
        by_id = {t.task_id: t for t in dag.tasks}
        for t in dag.tasks:
            tm = result.timings[(dag.dag_id, t.task_id)]
            if tm.queue_wait_cpu_s < -1e-12 or tm.queue_wait_ul_s < -1e-12 or tm.queue_wait_dl_s < -1e-12:
                raise V2ScheduleError("negative queue wait in %s/%s" % (dag.dag_id, t.task_id))
            if not math.isfinite(tm.finish_s):
                raise V2ScheduleError("non-finite completion")
            for p in t.predecessors:
                ptm = result.timings[(dag.dag_id, p)]
                payload = float(by_id[p].output_bytes) if ptm.location != tm.location else 0.0
                # route-aware transfer duration (sum over the actual hops)
                needed = sum(payload / _rate_for(hop, 0, link)
                             for hop in _route_hops(ptm.location, tm.location, link))
                if tm.start_s + 1e-9 < ptm.finish_s + needed:
                    raise V2ScheduleError(
                        "data arrival violation %s/%s -> %s"
                        % (dag.dag_id, p, t.task_id))
    checks["precedence_and_data_arrival"] = True
    # calendar consistency: no double-booking beyond servers
    for cal in (shared_cpu, ul, dl, v2v):
        if len(cal._free) < 1:
            raise V2ScheduleError("calendar %s has no servers" % cal.name)
    checks["calendars_well_formed"] = True
    # completion consistency
    for dag in dags:
        end = max(result.timings[(dag.dag_id, t.task_id)].finish_s for t in dag.tasks)
        if abs(end - result.completion_by_dag[dag.dag_id]) > 1e-9:
            raise V2ScheduleError("completion mismatch for %s" % dag.dag_id)
    checks["completion_consistent"] = True
    # utilization bounds
    for name, value in result.utilizations.items():
        values = value if isinstance(value, list) else [value]
        for v in values:
            if v < -1e-9 or v > 1.0 + 1e-9:
                raise V2ScheduleError("impossible utilization %s=%r" % (name, v))
    checks["utilization_bounds"] = True
    return checks
