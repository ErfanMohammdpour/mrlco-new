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
    """Non-preemptive server pool with EARLIEST-FIT reservation.

    Reservations are inserted in time order (not arrival order), so booking a transfer
    that only starts later cannot block a short transfer that is ready earlier. This
    matters for shared radio channels where the sink return of one DAG is booked before
    the ingress of the next DAG: a FIFO-by-arrival calendar would serialise them.
    """

    name: str
    servers: int = 1
    _intervals: list = field(default_factory=list)

    def __post_init__(self) -> None:
        self._intervals = [[] for _ in range(max(1, int(self.servers)))]

    def _earliest_fit(self, ready: float, duration: float):
        best = None
        for idx, intervals in enumerate(self._intervals):
            t = float(ready)
            for start, end in intervals:            # sorted by start
                if t + duration <= start + 1e-12:
                    break
                t = max(t, end)
            end = t + duration
            if best is None or (end, idx) < (best[0], best[1]):
                best = (end, idx, t)
        return best[0], best[1], best[2]

    def reserve(self, ready: float, duration: float):
        if duration < 0.0 or not math.isfinite(duration):
            raise V2ScheduleError("invalid duration for %s: %r" % (self.name, duration))
        end, idx, start = self._earliest_fit(float(ready), float(duration))
        intervals = self._intervals[idx]
        intervals.append((start, end))
        intervals.sort()
        return start, end

    def peek_wait(self, ready: float) -> float:
        _end, _idx, start = self._earliest_fit(float(ready), 0.0)
        return max(0.0, start - float(ready))

    @property
    def busy_intervals(self):
        """Total busy time per server (used for utilisation)."""
        return [sum(e - s for s, e in intervals) for intervals in self._intervals]


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
    outage_wait_s: float = 0.0
    outage_events: int = 0
    helper_rejected_inadmissible: bool = False
    reliability_rejected: bool = False
    reliability_p_success: float | None = None
    fallback_reserved_s: float = 0.0
    fallback_used_s: float = 0.0
    contact_failure: bool = False
    restart_penalty_s: float = 0.0

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


HOP_TO_LINK = {HOP_UL: "mec_ul", HOP_DL: "mec_dl", HOP_V2V: "v2v",
               HOP_UL_DIRECT: "mec_ul", HOP_DL_DIRECT: "mec_dl"}


def schedule_shared(dags: Sequence[V2DAGSpec], plans: Mapping[str, Sequence[int]],
                    *, link: V2LinkSpec, compute: V2ComputeSpec,
                    link_process=None, helper_states: Mapping | None = None,
                    contact_margin: float = 0.9, reliability_gate=None,
                    reliability_evidence: Mapping | None = None,
                    standby_for=None) -> V2ScheduleResult:
    """List-schedule multiple DAGs on shared calendars. Deterministic.

    `link_process` (v2 stage 3) supplies **realized** link multipliers per (link, time).
    The plan is executed open-loop: when a link is out, execution waits for recovery and
    the wait is logged (no re-planning).
    """
    if len(dags) != len(plans):
        raise V2ScheduleError("one plan per DAG required")
    shared_cpu = Calendar("MEC_CPU", servers=int(compute.mec_workers))
    ul = Calendar("MEC_UL") if link.shared_radio else Calendar("MEC_UL", servers=len(dags))
    dl = Calendar("MEC_DL") if link.shared_radio else Calendar("MEC_DL", servers=len(dags))
    v2v = Calendar("V2V_CHANNEL") if link.shared_radio else Calendar("V2V_CHANNEL", servers=max(1, len(dags)))
    ue_cpu = {i: Calendar("UE_CPU_%d" % i) for i in range(len(dags))}
    helper_cpu: dict = {}
    helper_busy_until: dict = {}
    for dag in dags:
        if dag.helper_id is not None:
            cal = helper_cpu.setdefault(dag.helper_id, Calendar("HELPER_CPU_%d" % dag.helper_id))
            if dag.helper_id not in helper_busy_until:
                state = (helper_states or {}).get(dag.helper_id)
                offset = float(getattr(state, "busy_until_s", 0.0)) if state is not None else 0.0
                helper_busy_until[dag.helper_id] = offset
                cal.reserve(0.0, offset)   # reserve the helper's own workload first
    radio_events: list = []
    concurrency: list = []

    def reserve_transfer(cal: Calendar, ready: float, payload: float, hop: str):
        link_name = HOP_TO_LINK[hop]
        base = _rate_for(hop, 0, link)
        t = float(ready)
        outage_wait = 0.0
        outage_events = 0
        if link_process is not None:
            dt = link_process.regime.dt_s
            for _ in range(4096):
                if link_process.realized(link_name, t) > 0.0:
                    break
                t += dt
                outage_wait += dt
                outage_events += 1
            else:
                raise V2ScheduleError("link %s stayed out for the whole horizon" % link_name)
            rate = link_process.realized_rate(base, link_name, t)
        else:
            rate = base
        duration = float(payload) / rate
        start, end = cal.reserve(t, duration)
        return start, end, duration, outage_wait, outage_events

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
        helper_rejected = False
        if location == HELPER:
            state = (helper_states or {}).get(dag.helper_id)
            if dag.helper_id is None or state is None:
                location = UE   # no helper available -> the plan degrades to local execution
            else:
                # admissibility against the PREDICTED (planner-visible) contact window
                from .helper_model import admissible
                predecessors_payload = sum(float(by_dag[dag_id][p].output_bytes)
                                           for p in spec.predecessors)
                if not admissible(state, now_s=float(dag.arrival_s),
                                  payload_in_bytes=predecessors_payload or float(spec.external_input_bytes),
                                  compute_bytes=float(spec.compute_bytes),
                                  v2v_bytes_per_s=link.v2v_bytes_per_s,
                                  margin=float(contact_margin)):
                    location = UE
                    helper_rejected = True
        # criticality-aware remote admissibility (stage 5): a remote token is only kept if
        # the reliability envelope of its class admits the estimated success probability
        reliability_rejected = False
        p_success = None
        if location in (MEC, HELPER) and reliability_gate is not None:
            evidence = dict(reliability_evidence or {})
            evidence.setdefault("link_confidence", 1.0)
            evidence.setdefault("outage_fraction", 0.0)
            if location == HELPER:
                state_ev = (helper_states or {}).get(dag.helper_id)
                predicted = getattr(state_ev, "predicted_contact_end_s", None)
                need = (float(spec.compute_bytes) /
                        max(1e-9, float(compute.helper_cpu_bytes_per_s[dag.helper_id])))
                if predicted is not None and math.isfinite(float(predicted)):
                    window = max(1e-9, float(predicted) - float(dag.arrival_s))
                    evidence.setdefault("contact_margin", min(1.0, need / window))
            ok, p_success = reliability_gate(location, str(spec.criticality).upper(), evidence)
            if not ok:
                location = UE
                reliability_rejected = True

        # data-ready time from predecessors (outputs stay where they executed)
        ready = float(dag.arrival_s)
        in_bytes = 0.0
        tx_dl = 0.0
        queue_wait_dl = 0.0
        task_outage_wait = 0.0
        task_outage_events = 0
        if spec.is_root and spec.external_input_bytes and location != UE:
            # v1 engine semantics: the external input is uploaded before execution
            root_outage_wait = 0.0
            root_outage_events = 0
            for hop in _route_hops(UE, location, link):
                cal = (ul if hop in (HOP_UL, HOP_UL_DIRECT)
                       else dl if hop in (HOP_DL, HOP_DL_DIRECT) else v2v)
                _s, _e, _d, _ow, _oe = reserve_transfer(cal, ready,
                                                        float(spec.external_input_bytes), hop)
                radio_events.append((_s, _e, cal.name, float(spec.external_input_bytes)))
                ready = _e
                root_outage_wait += _ow
                root_outage_events += _oe
            in_bytes += float(spec.external_input_bytes)
            task_outage_wait += root_outage_wait
            task_outage_events += root_outage_events
        for p in spec.predecessors:
            pt: TaskTiming = timings[(dag_id, p)]
            transfer = (pt.location != location)
            payload = float(by_dag[dag_id][p].output_bytes) if transfer else 0.0
            t_ready = pt.finish_s
            if transfer:
                hops = _route_hops(pt.location, location, link)
                for hop in hops:
                    cal = (ul if hop in (HOP_UL, HOP_UL_DIRECT)
                           else dl if hop in (HOP_DL, HOP_DL_DIRECT) else v2v)
                    start, end, _dur, _ow, _oe = reserve_transfer(cal, t_ready, payload, hop)
                    radio_events.append((start, end, cal.name, payload))
                    t_ready = end
                    task_outage_wait += _ow
                    task_outage_events += _oe
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
            state = (helper_states or {}).get(dag.helper_id)
            contact_failure = False
            restart_penalty = 0.0
            if state is not None and math.isfinite(float(state.contact_end_s)):
                ready_contact = max(ready, float(state.contact_start_s))
                start, cpu_end = helper_cpu[dag.helper_id].reserve(ready_contact, cpu_duration)
                if cpu_end > float(state.contact_end_s) + 1e-12:
                    # minimal documented fallback: the remainder restarts locally
                    contact_failure = True
                    done = max(0.0, (float(state.contact_end_s) - start)) * rate
                    remaining_bytes = max(0.0, float(spec.compute_bytes) - done)
                    restart_penalty = remaining_bytes / max(1e-9, float(compute.ue_cpu_bytes_per_s[dag.owner]))
                    # the data (and any partial output) must come back to the vehicle
                    # before the local restart: book the return transfer explicitly
                    local_ready = float(state.contact_end_s)
                    if in_bytes > 0:
                        for hop in _route_hops(HELPER, UE, link):
                            cal = (ul if hop in (HOP_UL, HOP_UL_DIRECT)
                                   else dl if hop in (HOP_DL, HOP_DL_DIRECT) else v2v)
                            rs, re_, _rd, _row, _roe = reserve_transfer(cal, local_ready,
                                                                        float(in_bytes), hop)
                            radio_events.append((rs, re_, cal.name, float(in_bytes)))
                            local_ready = re_
                            tx_dl += _rd
                    _us, ue_end = ue_cpu[dag.owner].reserve(local_ready, restart_penalty)
                    cpu_duration = restart_penalty
                    cpu_end = ue_end
                    location = UE
            else:
                start, cpu_end = helper_cpu[dag.helper_id].reserve(ready, cpu_duration)
        queue_wait_cpu = max(0.0, start - ready)
        fallback_reserved = 0.0
        fallback_used = 0.0
        if (location in (MEC, HELPER) and standby_for is not None
                and standby_for(str(spec.criticality).upper())):
            # warm standby hook: book the local worst-case duration so the vehicle can
            # take over if the remote execution fails (reserved, not necessarily used)
            worst = float(spec.compute_bytes) / max(1e-9, float(compute.ue_cpu_bytes_per_s[dag.owner]))
            # the standby keeps the vehicle committed for at most the remote execution
            # window (it must be able to take over while the remote attempt runs)
            window = max(0.0, float(cpu_end) - float(ready))
            if window > 0.0:
                standby = min(worst, window)
                _fs, _fe = ue_cpu[dag.owner].reserve(ready, standby)
                fallback_reserved = standby
        finish_s = cpu_end
        if spec.is_sink and location != UE:
            hops = _route_hops(location, UE, link)
            t = cpu_end
            for i, hop in enumerate(hops):
                cal = (ul if hop in (HOP_UL, HOP_UL_DIRECT)
                       else dl if hop in (HOP_DL, HOP_DL_DIRECT) else v2v)
                if i == 0:
                    queue_wait_dl = cal.peek_wait(t)
                dstart, dend, duration, _ow, _oe = reserve_transfer(
                    cal, t, float(spec.output_bytes), hop)
                radio_events.append((dstart, dend, cal.name, float(spec.output_bytes)))
                tx_dl += duration
                t = dend
                task_outage_wait += _ow
                task_outage_events += _oe
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
            helper_id=dag.helper_id if location == HELPER else None,
            helper_rejected_inadmissible=bool(helper_rejected),
            reliability_rejected=bool(reliability_rejected),
            reliability_p_success=(None if p_success is None else float(p_success)),
            fallback_reserved_s=float(fallback_reserved),
            fallback_used_s=float(locals().get("restart_penalty", 0.0)),
            contact_failure=bool(locals().get("contact_failure", False)),
            restart_penalty_s=float(locals().get("restart_penalty", 0.0)),
            outage_wait_s=float(task_outage_wait),
            outage_events=int(task_outage_events))
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
    # utilization + queue statistics: the horizon covers the makespan AND any reservation
    # that extends past it (a warm standby booked on the vehicle calendar)
    ends = [makespan]
    for cal in (shared_cpu, ul, dl, v2v):
        for intervals in cal._intervals:
            if intervals:
                ends.append(max(e for _s, e in intervals))
    for cal in list(ue_cpu.values()) + list(helper_cpu.values()):
        for intervals in cal._intervals:
            if intervals:
                ends.append(max(e for _s, e in intervals))
    horizon = max(1e-12, max(ends))
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
                     "remote_wait_max_s": max(remote_wait) if remote_wait else 0.0,
                     "outage_wait_total_s": sum(tm.outage_wait_s for tm in timings.values()),
                     "outage_events_total": sum(tm.outage_events for tm in timings.values()),
                     "helper_rejections_inadmissible": sum(
                         1 for tm in timings.values() if tm.helper_rejected_inadmissible),
                     "helper_contact_failures": sum(
                         1 for tm in timings.values() if tm.contact_failure),
                     "helper_restart_penalty_s": sum(tm.restart_penalty_s for tm in timings.values()),
                     "reliability_rejections": sum(
                         1 for tm in timings.values() if tm.reliability_rejected),
                     "fallback_reserved_s": sum(tm.fallback_reserved_s for tm in timings.values()),
                     "fallback_used_s": sum(tm.fallback_used_s for tm in timings.values())},
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
            # Causality + booking consistency. The scheduler records `ready_s` as the
            # time at which every input transfer actually completed, so the invariant is
            # checked against that (process-aware) value rather than re-deriving transfer
            # durations from base rates, which is invalid for time-varying links.
            if tm.start_s + 1e-9 < tm.ready_s - 1e-12:
                raise V2ScheduleError(
                    "task %s/%s starts before its data is ready (start=%.9f ready=%.9f)"
                    % (dag.dag_id, t.task_id, tm.start_s, tm.ready_s))
            for p in t.predecessors:
                ptm = result.timings[(dag.dag_id, p)]
                if tm.ready_s + 1e-9 < ptm.finish_s:
                    raise V2ScheduleError(
                        "data arrived before the producer finished %s/%s -> %s"
                        % (dag.dag_id, p, t.task_id))
    checks["precedence_and_data_arrival"] = True
    # calendar consistency: no double-booking beyond servers
    for cal in (shared_cpu, ul, dl, v2v):
        if len(cal._intervals) < 1:
            raise V2ScheduleError("calendar %s has no servers" % cal.name)
        for intervals in cal._intervals:
            ordered = sorted(intervals)
            for (s1, e1), (s2, e2) in zip(ordered, ordered[1:]):
                if s2 + 1e-9 < e1:
                    raise V2ScheduleError(
                        "double booking on %s: [%.6f,%.6f] and [%.6f,%.6f]"
                        % (cal.name, s1, e1, s2, e2))
    checks["calendars_well_formed"] = True
    checks["no_double_booking"] = True
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
