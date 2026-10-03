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
    #: number of `plan_service` searches that had to fall back to the best feasible candidate
    #: instead of a converged fixed point (surfaced in `V2ScheduleResult.mechanics`)
    fixed_point_fallbacks: int = 0

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
        return self.reserve_at(idx, start, end)

    def _fits(self, idx: int, start: float, end: float) -> bool:
        """True when [start, end) does not overlap any reservation of server `idx`."""
        for s, e in self._intervals[int(idx)]:
            if start < e - 1e-12 and s < end - 1e-12:
                return False
        return True

    def plan_service(self, ready: float, duration_fn, *, tol: float = 1e-9,
                     max_iter: int = 64):
        """Non-mutating fixed point for a transfer whose duration depends on its start.

        Problem: the earliest free slot can start later than `ready`, and a later start can
        see a different served-rate history (outages, time-varying links), so the duration is
        itself a function of the start:

            start_{k+1} = earliest_fit(ready, duration(start_k))

        Two iterations are NOT a proof of convergence. This routine therefore
        * never mutates the calendar (a failed search leaves no provisional reservation),
        * accepts only when |start_{k+1}-start_k| <= tol and |duration_{k+1}-duration_k| <=
          tol*max(1,|duration|) AND the interval provably fits its server,
        * bounds the search at `max_iter`, and
        * on a detected cycle / bound exhaustion returns the best **provably feasible**
          candidate found and increments `fixed_point_fallbacks` (reported in the result
          mechanics) instead of silently booking a non-converged interval. When no feasible
          candidate exists at all it raises `V2ScheduleError`.

        Returns (server_index, start_s, duration_s, end_s).
        """
        seen = set()
        candidates = []
        start = float(ready)
        duration = float(duration_fn(start))
        for _ in range(max(1, int(max_iter))):
            if duration < 0.0 or not math.isfinite(duration):
                raise V2ScheduleError(
                    "non-finite or negative service duration %r on %s: a transfer that can "
                    "never complete must not be booked" % (duration, self.name))
            _end, idx, new_start = self._earliest_fit(float(ready), duration)
            new_duration = float(duration_fn(new_start))
            if new_duration < 0.0 or not math.isfinite(new_duration):
                raise V2ScheduleError(
                    "non-finite or negative service duration %r at start %.9f on %s"
                    % (new_duration, new_start, self.name))
            end = new_start + new_duration
            feasible = self._fits(idx, new_start, end)
            if feasible:
                candidates.append((end, int(idx), float(new_start), float(new_duration)))
            stable = (abs(new_start - start) <= tol * max(1.0, abs(start))
                      and abs(new_duration - duration) <= tol * max(1.0, abs(duration)))
            if stable and feasible:
                return int(idx), float(new_start), float(new_duration), float(end)
            key = (round(float(new_start), 12), round(float(new_duration), 12))
            if key in seen:
                break
            seen.add(key)
            start, duration = new_start, new_duration
        if candidates:
            self.fixed_point_fallbacks += 1
            end, idx, s, d = min(candidates)
            return idx, s, d, end
        raise V2ScheduleError(
            "no feasible reservation on %s for ready=%.9f (fixed point did not converge in "
            "%d iterations and no candidate interval fits)" % (self.name, ready, max_iter))

    def reserve_at(self, idx: int, start: float, end: float):
        """Commit an interval that `plan_service` already proved feasible (booked once)."""
        s, e = float(start), float(end)
        if e < s - 1e-12 or not math.isfinite(s) or not math.isfinite(e):
            raise V2ScheduleError("invalid reservation on %s: [%r, %r]" % (self.name, s, e))
        intervals = self._intervals[int(idx)]
        intervals.append((s, e))
        intervals.sort()
        return s, e

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
    # -- v2 stage-4/5 accounting (all defaulted, additive) -----------------
    #: active radio service seconds by hop class. `tx_*` counts ONLY time in which bytes
    #: were actually served; outage pauses are never counted as active TX.
    tx_v2v_s: float = 0.0
    queue_wait_v2v_s: float = 0.0
    #: bytes actually executed on a CPU at this task's location
    cpu_executed_bytes: float = 0.0
    #: work executed at a remote location and then discarded because the result could not
    #: be returned before the contact ended (never counted as useful work)
    cpu_wasted_bytes: float = 0.0
    #: bytes re-executed locally after a helper contact failure (full restart: v2 does not
    #: implement checkpointing, so a restart re-runs the whole task)
    cpu_restart_bytes: float = 0.0
    #: where the work was first attempted (differs from `location` after a fallback)
    attempted_location: str = ""
    #: bytes of checkpoint state transferred to the requester before disconnection.
    #: ALWAYS 0 in v2: checkpoint transfer is NOT implemented (labelled unsupported, so it
    #: can never be mistaken for a free recovery mechanism).
    checkpoint_transfer_bytes: float = 0.0

    @property
    def total_remote_response_s(self) -> float:
        return (self.queue_wait_ul_s + self.tx_ul_s + self.queue_wait_v2v_s + self.tx_v2v_s
                + self.cpu_s + self.tx_dl_s)

    @property
    def tx_service_s(self) -> float:
        return self.tx_ul_s + self.tx_dl_s + self.tx_v2v_s


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
    #: (start_s, end_s, calendar_name, bytes) for every booked radio transfer. The interval is
    #: the CHANNEL OCCUPANCY (service + mid-transfer outage pauses under the declared
    #: retain-the-channel model); active service is in `radio_ledger`.
    radio_events: list = field(default_factory=list)
    #: one record per radio transfer, the event ledger the energy model consumes:
    #: {"dag_id","task_id","hop","direction","src","dst","bytes","start_s","end_s",
    #:  "service_s","outage_s","queue_wait_s","retained_channel"}
    radio_ledger: list = field(default_factory=list)


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


def _route_steps(location_a: str, location_b: str, link: V2LinkSpec) -> tuple:
    """`_route_hops` with the per-hop (src, dst) locations: ((hop, src, dst), ...)."""
    if location_a == location_b:
        return ()
    if {location_a, location_b} == {UE, MEC}:
        return ((HOP_UL, UE, MEC),) if location_b == MEC else ((HOP_DL, MEC, UE),)
    if {location_a, location_b} == {UE, HELPER}:
        return (((HOP_V2V, UE, HELPER),) if location_b == HELPER
                else ((HOP_V2V, HELPER, UE),))
    if (location_a, location_b) == (MEC, HELPER):
        return (((HOP_DL_DIRECT, MEC, HELPER),) if link.direct_helper_v2i
                else ((HOP_DL, MEC, UE), (HOP_V2V, UE, HELPER)))
    if (location_a, location_b) == (HELPER, MEC):
        return (((HOP_UL_DIRECT, HELPER, MEC),) if link.direct_helper_v2i
                else ((HOP_V2V, HELPER, UE), (HOP_UL, UE, MEC)))
    raise V2ScheduleError("unknown route %s -> %s" % (location_a, location_b))


def _hop_class(hop: str) -> str:
    if hop in (HOP_UL, HOP_UL_DIRECT):
        return "ul"
    if hop in (HOP_DL, HOP_DL_DIRECT):
        return "dl"
    if hop == HOP_V2V:
        return "v2v"
    raise V2ScheduleError("unknown hop %r" % hop)


#: explicit bound on the number of rate steps one transfer may take before it is declared
#: horizon-exhausted (never silently truncated)
MAX_SERVICE_STEPS = 1_000_000


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
                    standby_for=None, validate: bool = True,
                    max_service_steps: int = MAX_SERVICE_STEPS) -> V2ScheduleResult:
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
    radio_ledger: list = []
    concurrency: list = []
    calendar_of = {HOP_UL: ul, HOP_UL_DIRECT: ul, HOP_DL: dl, HOP_DL_DIRECT: dl, HOP_V2V: v2v}

    def _service_detail(start, payload, base, link_name):
        """(channel occupancy s, outage wait s, outage events) for a transfer starting at `start`.

        `occupancy` is the interval the channel is held for = active service + outage pauses.
        The active-service time is `occupancy - outage_wait`; the two are kept separate so an
        outage pause can never be counted as active TX.
        """
        if payload <= 0.0 or base <= 0.0:
            return 0.0, 0.0, 0
        if link_process is None:
            return float(payload) / float(base), 0.0, 0
        dt = float(link_process.regime.dt_s)
        if not math.isfinite(dt) or dt <= 0.0:
            raise V2ScheduleError("link regime dt_s must be positive and finite")
        t = float(start)
        served = 0.0
        outage_wait = 0.0
        outage_events = 0
        steps = 0
        while served + 1e-12 < payload:
            steps += 1
            if steps > int(max_service_steps):
                raise V2ScheduleError(
                    "transfer on %s starting at %.6fs did not complete within %d rate steps "
                    "(horizon exhaustion: the open-loop plan cannot finish)"
                    % (link_name, start, int(max_service_steps)))
            rate = base * float(link_process.realized(link_name, t))
            if rate <= 0.0:
                t += dt
                outage_wait += dt
                outage_events += 1
                continue
            step = min(dt, (payload - served) / rate)
            served += rate * step
            t += step
        return (t - float(start)), outage_wait, outage_events

    def reserve_transfer(cal: Calendar, ready: float, payload: float, hop: str):
        """Book exactly ONE interval for a transfer (rate at the ACTUAL booked start).

        Returns (booked_start, booked_end, active_service_s, queue_wait_s, outage_wait_s,
        outage_events). `Calendar.plan_service` resolves the start/duration fixed point
        WITHOUT touching the calendar, so no provisional reservation can be left behind and
        the same transfer is never booked twice (the audited over-reservation defect).
        """
        link_name = HOP_TO_LINK[hop]
        base = float(_rate_for(hop, 0, link))
        payload = float(payload)
        ready = float(ready)
        if payload <= 0.0:
            return ready, ready, 0.0, 0.0, 0.0, 0
        cache = {}

        def duration_fn(start):
            key = round(float(start), 12)
            hit = cache.get(key)
            if hit is None:
                hit = _service_detail(float(start), payload, base, link_name)
                cache[key] = hit
            return hit[0]

        idx, start, duration, end = cal.plan_service(ready, duration_fn)
        cal.reserve_at(idx, start, end)
        detail = cache.get(round(float(start), 12))
        if detail is None:
            detail = _service_detail(start, payload, base, link_name)
        if abs(detail[0] - duration) > 1e-9 * max(1.0, abs(duration)):
            raise V2ScheduleError(
                "booked occupancy %.12f disagrees with the service integral %.12f on %s"
                % (duration, detail[0], cal.name))
        active = max(0.0, float(detail[0]) - float(detail[1]))
        return start, end, active, max(0.0, start - ready), detail[1], detail[2]

    def _new_acc():
        return {"tx": {"ul": 0.0, "dl": 0.0, "v2v": 0.0},
                "wait": {"ul": 0.0, "dl": 0.0, "v2v": 0.0},
                "outage_wait": 0.0, "outage_events": 0}

    def book(src, dst, ready, payload, dag_id, task_id, acc):
        """Reserve every hop of a route, record the event ledger, return the delivery time."""
        t = float(ready)
        for hop, hsrc, hdst in _route_steps(src, dst, link):
            cal = calendar_of[hop]
            start, end, service, wait, outage_wait, outage_events = reserve_transfer(
                cal, t, payload, hop)
            if float(payload) > 0.0:
                radio_events.append((start, end, cal.name, float(payload)))
                radio_ledger.append({
                    "dag_id": dag_id, "task_id": int(task_id), "hop": hop,
                    "direction": "%s->%s" % (hsrc, hdst), "src": hsrc, "dst": hdst,
                    "bytes": float(payload), "start_s": start, "end_s": end,
                    "service_s": float(service), "outage_s": float(outage_wait),
                    "queue_wait_s": float(wait), "outage_events": int(outage_events),
                    "retained_channel": True,
                })
                cls = _hop_class(hop)
                acc["tx"][cls] += float(service)
                acc["wait"][cls] += float(wait)
                acc["outage_wait"] += float(outage_wait)
                acc["outage_events"] += int(outage_events)
            t = end
        return t

    timings: dict = {}
    finish: dict = {}
    remaining: dict = {}
    by_dag: dict = {}
    succ_by_dag: dict = {}
    import collections
    for dag in dags:
        by_dag[dag.dag_id] = {t.task_id: t for t in dag.tasks}
        remaining[dag.dag_id] = {t.task_id: len(t.predecessors) for t in dag.tasks}
        succ = collections.defaultdict(list)
        for t in dag.tasks:
            for p in t.predecessors:
                succ[p].append(t.task_id)
        succ_by_dag[dag.dag_id] = succ
    dag_by_id = {d.dag_id: d for d in dags}
    from .helper_model import admissible, required_helper_time_s

    def _plan_action(dag_id, task_id):
        plan = plans[dag_id]
        if isinstance(plan, Mapping):
            return int(plan.get(task_id, 0))
        return int(plan[task_id]) if task_id < len(plan) else 0

    # ---- pass 0: effective location per token (plan + admission), timing-independent ----
    # Deciding locations for the WHOLE plan before scheduling makes the "must the helper
    # return this result before contact ends?" question answerable at that task's turn.
    effective_loc: dict = {}
    admission: dict = {}
    for dag in dags:
        state = (helper_states or {}).get(dag.helper_id)
        for spec in dag.tasks:
            key = (dag.dag_id, spec.task_id)
            location = {0: UE, 1: MEC, 2: HELPER}[_plan_action(dag.dag_id, spec.task_id)]
            helper_rejected = False
            reliability_rejected = False
            p_success = None
            if location == HELPER:
                if dag.helper_id is None or state is None:
                    location = UE   # no helper available -> the plan degrades to local
                else:
                    pred_payload = sum(float(by_dag[dag.dag_id][p].output_bytes)
                                       for p in spec.predecessors)
                    if not admissible(
                            state, now_s=float(dag.arrival_s),
                            payload_in_bytes=pred_payload or float(spec.external_input_bytes),
                            compute_bytes=float(spec.compute_bytes),
                            v2v_bytes_per_s=link.v2v_bytes_per_s,
                            margin=float(contact_margin)):
                        location = UE
                        helper_rejected = True
            if location in (MEC, HELPER) and reliability_gate is not None:
                evidence = dict(reliability_evidence or {})
                evidence.setdefault("link_confidence", 1.0)
                evidence.setdefault("outage_fraction", 0.0)
                if location == HELPER:
                    predicted = getattr(state, "predicted_contact_end_s", None)
                    from .reliability import contact_slack
                    evidence.setdefault("contact_margin", contact_slack(
                        predicted_contact_end_s=predicted, now_s=float(dag.arrival_s),
                        payload_in_bytes=(pred_payload if spec.predecessors
                                          else float(spec.external_input_bytes)),
                        compute_bytes=float(spec.compute_bytes),
                        v2v_bytes_per_s=link.v2v_bytes_per_s,
                        helper_bytes_per_s=float(compute.helper_cpu_bytes_per_s[dag.helper_id]),
                        output_bytes=float(spec.output_bytes)))
                ok, p_success = reliability_gate(
                    location, str(spec.criticality).upper(), evidence)
                if not ok:
                    location = UE
                    reliability_rejected = True
            effective_loc[key] = location
            admission[key] = (bool(helper_rejected), bool(reliability_rejected), p_success)

    ready_heap: list = []
    for dag in dags:
        for t in dag.tasks:
            if remaining[dag.dag_id][t.task_id] == 0:
                heapq.heappush(ready_heap, (dag.arrival_s, dag.dag_id, t.task_id))
    while ready_heap:
        _arr, dag_id, task_id = heapq.heappop(ready_heap)
        dag = dag_by_id[dag_id]
        spec = by_dag[dag_id][task_id]
        location = effective_loc[(dag_id, task_id)]
        helper_rejected, reliability_rejected, p_success = admission[(dag_id, task_id)]
        acc = _new_acc()

        # data-ready time from predecessors (outputs stay where they executed)
        ready = float(dag.arrival_s)
        in_bytes = 0.0
        if spec.is_root and float(spec.external_input_bytes) > 0.0 and location != UE:
            # v1 engine semantics: the external input is uploaded before execution
            ready = book(UE, location, ready, float(spec.external_input_bytes),
                         dag_id, task_id, acc)
            in_bytes += float(spec.external_input_bytes)
        for p in spec.predecessors:
            pt: TaskTiming = timings[(dag_id, p)]
            if pt.location != location:
                payload = float(by_dag[dag_id][p].output_bytes)
                ready = max(ready, book(pt.location, location, pt.finish_s, payload,
                                        dag_id, task_id, acc))
                in_bytes += payload
            else:
                ready = max(ready, pt.finish_s)

        # ---- CPU phase ----
        contact_failure = False
        restart_penalty = 0.0
        wasted_bytes = 0.0
        restart_bytes = 0.0
        executed_bytes = float(spec.compute_bytes)
        attempted_location = location
        state = (helper_states or {}).get(dag.helper_id)
        if location == MEC:
            rate = float(compute.mec_cpu_bytes_per_s)
            cpu_duration = executed_bytes / rate
            start, cpu_end = shared_cpu.reserve(ready, cpu_duration)
        elif location == UE:
            rate = float(compute.ue_cpu_bytes_per_s[dag.owner])
            cpu_duration = executed_bytes / rate
            start, cpu_end = ue_cpu[dag.owner].reserve(ready, cpu_duration)
        else:
            rate = float(compute.helper_cpu_bytes_per_s[dag.helper_id])
            cpu_duration = executed_bytes / rate
            if state is None or not math.isfinite(float(state.contact_end_s)):
                start, cpu_end = helper_cpu[dag.helper_id].reserve(ready, cpu_duration)
            else:
                contact_end = float(state.contact_end_s)
                ready_contact = max(ready, float(state.contact_start_s))
                start, cpu_end = helper_cpu[dag.helper_id].reserve(ready_contact, cpu_duration)
                # Returning the RESULT must also be possible inside the contact window: CPU
                # completion alone is not sufficient (audited defect 4.6).
                return_required = bool(spec.is_sink) or any(
                    effective_loc[(dag_id, c)] != HELPER for c in succ_by_dag[dag_id][task_id])
                return_needed = 0.0
                if return_required and float(spec.output_bytes) > 0.0:
                    return_needed = _service_detail(
                        cpu_end, float(spec.output_bytes),
                        float(link.v2v_bytes_per_s), HOP_TO_LINK[HOP_V2V])[0]
                if cpu_end + return_needed > contact_end + 1e-12:
                    contact_failure = True
                    # Work the helper actually performed before disconnection. It is WASTED:
                    # once contact has ended no transfer can carry the result back, and v2
                    # implements NO checkpointing, so a free partial recovery would be
                    # physically invalid. Recovery is a FULL local restart from the task's
                    # inputs, which the requester already owns.
                    served_s = max(0.0, min(cpu_end, contact_end) - start)
                    wasted_bytes = min(executed_bytes, served_s * rate)
                    restart_bytes = executed_bytes
                    restart_penalty = restart_bytes / max(
                        1e-9, float(compute.ue_cpu_bytes_per_s[dag.owner]))
                    _us, ue_end = ue_cpu[dag.owner].reserve(contact_end, restart_penalty)
                    cpu_duration = restart_penalty
                    cpu_end = ue_end
                    # the FINAL execution re-runs the whole task locally, so the executed
                    # work of this task is the restart; `wasted_bytes` stays separate
                    executed_bytes = restart_bytes
                    location = UE
        queue_wait_cpu = max(0.0, start - ready)
        fallback_reserved = 0.0
        if (location in (MEC, HELPER) and standby_for is not None
                and standby_for(str(spec.criticality).upper())):
            # warm standby hook: book the local worst-case duration so the vehicle can
            # take over if the remote execution fails (reserved, not necessarily used)
            worst = executed_bytes / max(1e-9, float(compute.ue_cpu_bytes_per_s[dag.owner]))
            window = max(0.0, float(cpu_end) - float(ready))
            if window > 0.0:
                standby = min(worst, window)
                _fs, _fe = ue_cpu[dag.owner].reserve(ready, standby)
                fallback_reserved = standby
        finish_s = cpu_end
        if spec.is_sink and location != UE:
            finish_s = book(location, UE, cpu_end, float(spec.output_bytes),
                            dag_id, task_id, acc)
        timing = TaskTiming(
            dag_id=dag_id, task_id=task_id, location=location, ready_s=ready,
            queue_wait_ul_s=float(acc["wait"]["ul"]),
            tx_ul_s=float(acc["tx"]["ul"]),
            queue_wait_cpu_s=queue_wait_cpu, cpu_s=cpu_duration,
            queue_wait_dl_s=float(acc["wait"]["dl"]), tx_dl_s=float(acc["tx"]["dl"]),
            start_s=start, finish_s=finish_s, transfer_in_bytes=in_bytes,
            helper_id=dag.helper_id if location == HELPER else None,
            helper_rejected_inadmissible=bool(helper_rejected),
            reliability_rejected=bool(reliability_rejected),
            reliability_p_success=(None if p_success is None else float(p_success)),
            fallback_reserved_s=float(fallback_reserved),
            fallback_used_s=float(restart_penalty),
            contact_failure=bool(contact_failure),
            restart_penalty_s=float(restart_penalty),
            outage_wait_s=float(acc["outage_wait"]),
            outage_events=int(acc["outage_events"]),
            tx_v2v_s=float(acc["tx"]["v2v"]),
            queue_wait_v2v_s=float(acc["wait"]["v2v"]),
            cpu_executed_bytes=float(executed_bytes),
            cpu_wasted_bytes=float(wasted_bytes),
            cpu_restart_bytes=float(restart_bytes),
            attempted_location=attempted_location,
            checkpoint_transfer_bytes=0.0)
        timings[(dag_id, task_id)] = timing
        finish[(dag_id, task_id)] = finish_s
        concurrency.append((start, "start", dag_id))
        concurrency.append((finish_s, "end", dag_id))
        for child in succ_by_dag[dag_id][task_id]:
            remaining[dag_id][child] -= 1
            if remaining[dag_id][child] == 0:
                heapq.heappush(ready_heap, (finish_s, dag_id, child))
    completion = {}
    for dag in dags:
        completion[dag.dag_id] = max(finish[(dag.dag_id, t.task_id)] for t in dag.tasks)
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
    remote_wait = [tm.queue_wait_ul_s + tm.queue_wait_v2v_s for tm in timings.values()]
    fallbacks = sum(cal.fixed_point_fallbacks
                    for cal in (shared_cpu, ul, dl, v2v, *ue_cpu.values(), *helper_cpu.values()))
    result = V2ScheduleResult(
        makespan_s=makespan, completion_by_dag=completion, timings=timings,
        mechanics={"radio_events": len(radio_events), "tasks": len(timings),
                   "radio_bytes": float(sum(e[3] for e in radio_events)),
                   "booking_fixed_point_fallbacks": int(fallbacks)},
        radio_events=list(radio_events),
        radio_ledger=list(radio_ledger),
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
                     "fallback_used_s": sum(tm.fallback_used_s for tm in timings.values()),
                     "wasted_work_bytes_total": sum(
                         tm.cpu_wasted_bytes for tm in timings.values()),
                     "restart_work_bytes_total": sum(
                         tm.cpu_restart_bytes for tm in timings.values()),
                     "checkpoint_transfer_bytes_total": sum(
                         tm.checkpoint_transfer_bytes for tm in timings.values()),
                     "tx_service_total_s": sum(tm.tx_service_s for tm in timings.values())},
        invariants={}, arrivals={d.dag_id: d.arrival_s for d in dags},
        active_concurrency=active_series)
    if validate:
        result.invariants = validate_schedule(dags, plans, link, result, shared_cpu, ue_cpu,
                                              helper_cpu, ul, dl, v2v)
    else:
        # prefix/telescoping evaluation: the same schedule computation without the
        # O(tasks x edges) route re-verification (the final episode schedule is validated)
        result.invariants = {"validated": False}
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
    # radio ledger: channel occupancy must equal ACTIVE SERVICE + OUTAGE pause under the
    # declared retain-the-channel model. The audited over-reservation (booking
    # wait+service as one interval and re-reserving the same transfer) breaks this identity,
    # so this check is the regression guard for it.
    for rec in result.radio_ledger:
        span = float(rec["end_s"]) - float(rec["start_s"])
        service = float(rec["service_s"])
        outage = float(rec["outage_s"])
        if span < -1e-12:
            raise V2ScheduleError("negative transfer interval in %r" % (rec,))
        if service < -1e-12 or outage < -1e-12:
            raise V2ScheduleError("negative service/outage time in %r" % (rec,))
        if abs((service + outage) - span) > 1e-9 * max(1.0, span):
            raise V2ScheduleError(
                "channel occupancy (%.9f) != active service (%.9f) + outage (%.9f) for %r: "
                "the booked interval does not describe the transfer"
                % (span, service, outage, rec))
        if float(rec["queue_wait_s"]) < -1e-12:
            raise V2ScheduleError("negative transfer queue wait in %r" % (rec,))
        if float(rec["bytes"]) < 0.0:
            raise V2ScheduleError("negative transfer payload in %r" % (rec,))
    checks["transfer_ledger_consistent"] = True
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
