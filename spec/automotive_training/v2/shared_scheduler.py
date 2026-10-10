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

Declared F08 semantics (each pinned by a test in `scheduler/tests/test_v2_*`):

* **Count-once broadcast.** A producer's output is transmitted ONCE per
  (dag, producer, destination location) and shared by every successor placed in that
  location; distinct destinations keep distinct transfers. Only the first consumer books the
  transfer, so its radio time, its ledger bytes and its TX energy are charged once.
* **A discarded reservation is released.** `Calendar.release` removes/shrinks a committed
  interval. On a helper contact failure the helper CPU interval is truncated to the served
  prefix `[start, start + served_s]` (the part the energy ledger bills as
  `cpu_attempt_seconds`); helper CPU is never reserved if it cannot start inside the contact
  window.
* **Infeasible ingress is never booked.** A HELPER placement whose input cannot ARRIVE inside
  the realized contact window falls back to the UE before anything is transmitted to (or
  billed for) that helper, and no transfer to a helper starts before `contact_start`.
* **Contact feasibility is route-aware.** The return pre-check asks when the data actually
  LEAVES the helper (last HELPER-source hop end) for EVERY required successor destination, so
  it sees the direct-V2I `MEC_ULH` hop instead of assuming a `HELPER -> UE` relay.
* **Ingress is serialised, not planned in isolation.** The ingress probe of one task chains
  its required routes on a NON-MUTATING overlay of the shared calendars (every pending leg is
  visible to the next), so a second/third simultaneous ingress to the same helper cannot
  silently slip past `contact_end` and be booked: the probe sees the real serialised
  delivery time, exactly like the later `book()` does.
* **Egress feasibility is enforced before a HELPER -> UE relay is booked.** The departure
  pre-check asks when the result actually LEAVES the helper (last HELPER-source hop end,
  `_helper_departure`) for EVERY required destination, with the CPU end the helper really
  reserves. A placement whose departure exceeds the realized `contact_end` is abandoned
  INSIDE the HELPER branch: the attempt is attributed and billed (served prefix released, full
  local restart), and **no** HELPER -> UE relay is booked after disconnection -- the planner
  never commits a transfer that `validate_schedule` would reject. The pre-fix planner instead
  committed the relay and aborted a 1500-iteration campaign at iteration 321.
* **A helper predecessor must be able to reach the requester.** Every task that has
  successors also requires its output to be able to reach the UE, because any successor can
  later be forced to restart locally and v2 implements no checkpointing (there is no
  requester-local copy of a helper result). That single requirement closes the audited
  dead-end: a successor whose HELPER-resident predecessor could not relay in time is refused
  the helper (and the predecessor then fails its own egress check and falls back too), so no
  stranded data, no fabricated relay, and a valid DAG. `validate_schedule` keeps the same
  invariant as an independent check, with the shared `CONTACT_SLACK_S`.
* **A local restart owns its inputs.** On a fallback, any predecessor whose FINAL location is
  not the UE and whose payload was not already relayed through the UE is transferred to the
  requester (and priced) BEFORE the restart is reserved.
* **Warm standby is declared planning state only.** `fallback_reserved_s` is reported but the
  reserve does NOT occupy the vehicle's real CPU calendar: a reservation that costs no joules
  (warm-standby energy is in `energy.UNMODELED`) must not cost time either.

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
    #: NON-MUTATING forecast overlay: [(server_index, start_s, end_s), ...] reservations that a
    #: planning probe is about to commit. `plan_service` sees them (so chained probes can
    #: serialise against each other) but they are never written to `_intervals`; the real
    #: booking commits them through `reserve_at` only once the whole probe is admissible. See
    #: the multi-ingress serialisation rule in the module docstring.
    _plan_overlay: list = field(default_factory=list)

    def __post_init__(self) -> None:
        self._intervals = [[] for _ in range(max(1, int(self.servers)))]

    def set_plan_overlay(self, reservations) -> None:
        """Install the pending-reservation overlay used by the NEXT `plan_service` call."""
        self._plan_overlay = [(int(i), float(s), float(e)) for i, s, e in reservations]

    def clear_plan_overlay(self) -> None:
        self._plan_overlay = []

    def _overlaid(self, idx: int):
        """`_intervals[idx]` plus the pending overlay entries for that server, sorted."""
        if not self._plan_overlay:
            return self._intervals[idx]
        extra = [(s, e) for i, s, e in self._plan_overlay if i == int(idx)]
        if not extra:
            return self._intervals[idx]
        return sorted(list(self._intervals[idx]) + extra)

    def _earliest_fit(self, ready: float, duration: float):
        best = None
        for idx, intervals in enumerate(self._intervals):
            intervals = self._overlaid(idx)
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
        """Reserve the earliest feasible interval; returns (start_s, end_s)."""
        _idx, start, end = self.reserve_indexed(ready, duration)
        return start, end

    def reserve_indexed(self, ready: float, duration: float):
        """`reserve` for callers that may later have to RELEASE the booking.

        Returns (server_index, start_s, end_s); `reserve` is the 2-tuple convenience wrapper.
        A caller that can discover after booking that the reserved time is worthless (a
        helper CPU attempt discarded by a contact failure, defect F08/a) needs the server
        index to hand the exact interval back to `release`.
        """
        if duration < 0.0 or not math.isfinite(duration):
            raise V2ScheduleError("invalid duration for %s: %r" % (self.name, duration))
        end, idx, start = self._earliest_fit(float(ready), float(duration))
        self.reserve_at(idx, start, end)
        return int(idx), float(start), float(end)

    def release(self, idx: int, start: float, end: float) -> float:
        """Remove/shrink the committed reservation(s) overlapping [start, end) on `idx`.

        A calendar with no release primitive can only grow: the audited F08 defect booked a
        helper CPU interval for an attempt that was then DISCARDED (contact ended before the
        result could leave the helper) and left it in place forever, so a completely
        unrelated later DAG was blocked by CPU time that provably produced nothing, and a
        warm-standby booking held the vehicle's own calendar while being billed as zero
        joules (a time/energy inconsistency).

        Semantics: the overlap of [start, end) with every committed interval of server `idx`
        is cut out; any left/right remainder of a partially covered interval is kept, and the
        server list is re-sorted, so the calendar stays well ordered and non-overlapping.
        Returns the released seconds (0.0 when nothing overlapped). Releasing is safe to call
        with an empty range and never moves a surviving interval in time.
        """
        intervals = self._intervals[int(idx)]
        s_req, e_req = float(start), float(end)
        if e_req < s_req - 1e-12 or not math.isfinite(s_req) or not math.isfinite(e_req):
            raise V2ScheduleError("invalid release on %s: [%r, %r]"
                                  % (self.name, s_req, e_req))
        kept = []
        released = 0.0
        for s, e in intervals:
            if e <= s_req + 1e-12 or s >= e_req - 1e-12:
                kept.append((s, e))                       # no overlap
                continue
            if s < s_req - 1e-12:
                kept.append((s, min(e, s_req)))           # left remainder
            if e > e_req + 1e-12:
                kept.append((max(s, e_req), e))           # right remainder
            released += min(e, e_req) - max(s, s_req)
        kept.sort()
        self._intervals[int(idx)] = kept
        return max(0.0, released)

    def _fits(self, idx: int, start: float, end: float) -> bool:
        """True when [start, end) overlaps neither a reservation nor a pending overlay entry."""
        for s, e in self._overlaid(int(idx)):
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
    #: CPU seconds actually spent at `attempted_location` before a contact failure (0.0 when
    #: the attempt succeeded). Energy is charged on the REAL executed work, so a failed
    #: remote attempt is billed at the helper tier and the local restart at the UE tier.
    cpu_attempt_seconds: float = 0.0
    #: bytes/second at which the discarded attempt ran (0.0 when no attempt was made). It lets
    #: the energy ledger re-derive the partial-attempt byte conservation
    #: (`cpu_attempt_seconds * cpu_attempt_rate_bytes_per_s == cpu_wasted_bytes`) without being
    #: handed the compute spec.
    cpu_attempt_rate_bytes_per_s: float = 0.0
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
    #: the batch makespan of the whole world (foreground + background). Reported separately
    #: from the episode latency: the reward and telemetry of the FOREGROUND episode use
    #: `episode_latency_s`, while `world_makespan_s` describes the shared world.
    world_makespan_s: float | None = None
    #: the FOREGROUND DAG's completion time (the episode latency), when a world defines one
    episode_latency_s: float | None = None
    #: foreground DAG identity, when a world defines one
    foreground_dag_id: str | None = None
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

#: the one slack every contact-window gate shares: the ingress-arrival gate, the egress
#: departure gate and `validate_schedule`'s helper-contact invariant must agree, otherwise a
#: plan can be "feasible" for one edge of the window and inadmissible for the other, leaving
#: the task stranded between them.
CONTACT_SLACK_S = 1e-9


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


def _declared_horizon(link_process) -> float | None:
    """The realized trace length in seconds, or None when the process does not declare one.

    Accepts either the real `LinkProcess.horizon_s()` method or a plain numeric attribute, so
    duck-typed test doubles work without silently inventing a horizon.
    """
    for attr in ("horizon_s", "horizon"):
        value = getattr(link_process, attr, None)
        if value is None:
            continue
        if callable(value):
            value = value()
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value) and value > 0.0:
            return value
    return None


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

    def _planned_route(src, dst, ready, payload):
        """Non-mutating plan of a whole route on the ACTUAL shared calendars.

        The F07 defect was to estimate the helper's result return with the queue-free
        `_service_detail`, which ignored that another DAG may already hold the V2V channel;
        the check passed, then `book()` reserved the return AFTER contact end. This plans the
        whole route with `Calendar.plan_service` (same fixed point `book()` uses) WITHOUT
        committing, so the feasibility check sees the real queued delivery time. A later
        `book()` of the same route yields the same intervals because the calendar state has
        not changed in between.

        Returns [(hop, src_loc, dst_loc, start_s, end_s, ready_s, duration_s, server_index),
        ...] in order (empty when src == dst). `ready_s` and `server_index` are exactly what
        the booking needs to reproduce the probed interval, so the multi-ingress probe can
        commit what it checked without re-deriving (and possibly changing) the start.
        """
        t = float(ready)
        cache = {}
        legs = []
        for hop, hsrc, hdst in _route_steps(src, dst, link):
            base = float(_rate_for(hop, 0, link))
            cal = calendar_of[hop]
            hop_ready = float(t)

            def hop_duration_fn(start, _base=base, _hop=hop):
                k = round(float(start), 12)
                hit = cache.get((_hop, k))
                if hit is None:
                    hit = _service_detail(float(start), payload, _base,
                                          HOP_TO_LINK[_hop])[0]
                    cache[(_hop, k)] = hit
                return hit

            idx, start, dur, end = cal.plan_service(t, hop_duration_fn)
            legs.append((hop, hsrc, hdst, float(start), float(end), hop_ready, float(dur),
                         int(idx)))
            t = end
        return legs

    def _planned_delivery(src, dst, ready, payload):
        """Non-mutating ARRIVAL time at `dst` for the route src -> dst (see `_planned_route`)."""
        legs = _planned_route(src, dst, ready, payload)
        return legs[-1][4] if legs else float(ready)

    def _helper_departure(dst, ready, payload):
        """When the data actually LEAVES the helper on the route HELPER -> `dst`.

        `_planned_delivery` returns the ARRIVAL time at the destination. The contact window
        binds the helper's OWN transmissions, and on a relayed route (HELPER -> UE -> MEC) the
        helper is free as soon as its first hop ends. The pre-fix F08/d check used
        `HELPER -> UE` unconditionally: correct for the relay topology, but blind to the
        direct V2I link. With `direct_helper_v2i=True` the route to a MEC destination is the
        single MEC_ULH HELPER -> MEC hop, so the pre-check passed and the successor then booked
        a helper departure after `contact_end` (which `validate=True` rejected, while
        `validate=False` silently accepted it).
        """
        legs = _planned_route(HELPER, dst, ready, payload)
        departure = float(ready)
        for _hop, hsrc, _hdst, _start, end, _r, _d, _i in legs:
            if hsrc == HELPER:
                departure = end
        return departure

    def _required_dsts(spec, dag_id, task_id):
        """FINAL destination locations that must RECEIVE this task's output.

        A successor placed elsewhere needs the output there; a task with no successor that is
        not the sink needs nothing. A sink's output always ends at its requester's UE.

        A task WITH successors also requires the UE: any of those successors can later be
        forced to restart locally (an ingress that no longer fits, or a downgrade cascade), and
        v2 implements no checkpointing, so the only copy it could restart from is one that has
        already reached the requester. Requiring the UE makes the successor gate's fallback
        feasible BY CONSTRUCTION instead of leaving a HELPER -> UE relay that
        `validate_schedule` would have to reject after the fact.
        """
        dsts = []
        if spec.is_sink or succ_by_dag[dag_id][task_id]:
            dsts.append(UE)
        for c in succ_by_dag[dag_id][task_id]:
            dst = effective_loc[(dag_id, c)]
            if dst != HELPER:
                dsts.append(dst)
        return dsts

    def _last_departure(cpu_end, payload, dsts):
        """Latest time at which the task's output LEAVES the helper (never before `cpu_end`).

        Returns `cpu_end` when no transfer is required (same-location successors only), so the
        caller can compare the value directly with `contact_end`.
        """
        latest = float(cpu_end)
        if payload <= 0.0:
            return latest
        for dst in dsts:
            latest = max(latest, _helper_departure(dst, cpu_end, payload))
        return latest

    #: planner-side downgrade counters (surfaced in `queue_stats`, see below)
    egress_infeasible = 0
    ingress_infeasible = 0

    def _fallback_input_available(pred_key):
        """Can predecessor `pred_key` still hand its output to the requester's UE in time?

        The question is asked about the PREDECESSOR's own window: its result must be able to
        leave the helper (`_helper_departure` from its own finish time, F08/d) before
        `contact_end`. Any later deadline would be wrong -- the block below runs at the
        SUCCESSOR's turn only to decide the successor's placement, and using the successor's
        CPU end would ask for a relay that the predecessor's window never had to contain.

        Three honest answers, in order:
          * it finished on the UE or the MEC: nothing has to move, so True;
          * it finished on the HELPER and can still relay `HELPER -> UE` inside the window: True;
          * it finished on the HELPER and CANNOT: False. Its data would be stranded on a
            disconnected node, so any successor that could have to restart from it must not be
            placed remotely (v2 has no checkpointing and no requester-local copy).
        """
        tm = timings.get(pred_key)
        if tm is None or tm.location != HELPER:
            return True
        payload = float(by_dag[pred_key[0]][pred_key[1]].output_bytes)
        if payload <= 0.0:
            return True
        return (_helper_departure(UE, tm.finish_s, payload)
                <= contact_end + CONTACT_SLACK_S)

    #: calendars the pending-ingress overlay may touch (the shared radio + the two CPU pools do
    #: not carry transfers, so only the radio calendars are overlaid)
    def _radio_calendars():
        return (ul, dl, v2v)

    def _set_overlay(pending):
        for cal in _radio_calendars():
            cal.set_plan_overlay([(idx, s, e) for c, idx, s, e in pending if c is cal])

    def _clear_overlay():
        for cal in _radio_calendars():
            cal.clear_plan_overlay()

    def probe_ingress(requests, contact_window):
        """Plan every REQUIRED ingress route TOGETHER and decide whether it is admissible.

        `requests` is an ordered [(src, dst, ready_s, payload_bytes, key), ...]. Every route is
        planned on a NON-MUTATING overlay of the radio calendars that already holds every leg
        planned for the earlier requests, and the planning cursor advances past each planned
        leg, so a second/third simultaneous ingress to one helper sees the real serialised
        delivery time instead of the (too optimistic) queue-free one. No real calendar is
        touched: a rejected probe leaves nothing behind.

        `contact_window` is (start, end) for a helper destination or (None, None) for a MEC
        destination: no ingress leg may start before `start` and no ARRIVAL may end after `end`.
        Returns (admissible, pending, ready_for); `pending` is the [(calendar, server_index,
        start, end), ...] the probe proved feasible and `ready_for[key]` the delivery time that
        `book_ingress` then commits verbatim.
        """
        pending = []
        pending_by_key = {}
        ready_for = {}
        window_start, window_end = contact_window
        cursor = None
        for src, dst, ready, payload, key in requests:
            base_ready = float(ready)
            if window_start is not None:
                base_ready = max(base_ready, float(window_start))
            if cursor is not None:
                base_ready = max(base_ready, cursor)
            _set_overlay(pending)
            route_legs = _planned_route(src, dst, base_ready, float(payload))
            legs_here = []
            for hop, _hsrc, _hdst, start, end, _r, _d, idx in route_legs:
                entry = (calendar_of[hop], idx, float(start), float(end))
                legs_here.append(entry)
                pending.append(entry)
                cursor = float(end)
            pending_by_key[key] = legs_here
            arrival = float(route_legs[-1][4]) if route_legs else base_ready
            # the SAME slack the egress gate uses (`CONTACT_SLACK_S`): an ingress that is
            # "feasible" past the window while the matching departure is refused would strand
            # the task between the two gates
            if (window_end is not None
                    and arrival > float(window_end) + CONTACT_SLACK_S):
                _clear_overlay()
                return False, [], {}
            ready_for[key] = arrival
        _clear_overlay()
        return True, pending_by_key, ready_for

    def book_ingress(requests, pending_by_key, ready_for, dag_id, task_id, acc,
                     window_start=None):
        """Commit exactly the intervals `probe_ingress` proved feasible, or raise.

        The probe and this commit run back to back with no reservation in between, so
        re-planning each request against the same pending overlay must reproduce the probed
        interval leg by leg. Re-planning request k sees the overlay of the legs that are still
        UN-BOOKED at that moment (the later requests) -- and NOT its own already probed entry,
        which would push the re-plan behind itself -- which is exactly the calendar state the
        probe saw when it planned request k. A divergence would mean committing an interval
        that was never feasibility-checked, so it raises instead of booking silently.

        Every leg is booked (and logged/priced) even when its payload is zero, because a
        multi-hop route must still occupy each hop in order; only the ledger record of a
        zero-byte transfer is skipped, exactly as `book()` does. Returns {key: arrival_s}.
        """
        arrived = {}
        cursor = None
        #: legs the probe checked that this commit has not booked yet: the overlay the next
        #: re-plan must use (the probe saw exactly the later, still un-booked requests' legs).
        pending = [entry for req in requests for entry in pending_by_key.get(req[4], ())]
        for src, dst, base_ready, payload, key in requests:
            ready = float(base_ready)
            if window_start is not None:
                ready = max(ready, float(window_start))
            if cursor is not None:
                ready = max(ready, cursor)
            # the current request's own probed legs are NOT part of the overlay it is planned
            # against (the probe planned them against exactly these un-booked later legs)
            own = list(pending_by_key.get(key, ()))
            _set_overlay([entry for entry in pending if all(entry is not o for o in own)])
            route_legs = _planned_route(src, dst, ready, float(payload))
            if len(route_legs) != len(own):
                _clear_overlay()
                raise V2ScheduleError(
                    "ingress probe/commit mismatch for %r: %d legs probed but the commit "
                    "re-planned %d" % (key, len(own), len(route_legs)))
            for (hop, hsrc, hdst, start, end, leg_ready, _dur, idx), entry in zip(
                    route_legs, own):
                if (entry[0] is not calendar_of[hop] or entry[1] != idx
                        or abs(entry[2] - start) > 1e-12 or abs(entry[3] - end) > 1e-12):
                    _clear_overlay()
                    raise V2ScheduleError(
                        "ingress probe/commit mismatch for %r: probed %r but the commit "
                        "re-planned [%s %r, %r]"
                        % (key, entry, hop, start, end))
                booked = reserve_transfer(calendar_of[hop], leg_ready, float(payload), hop)
                if (abs(booked[0] - start) > 1e-9 * max(1.0, abs(start))
                        or abs(booked[1] - end) > 1e-9 * max(1.0, abs(end))):
                    _clear_overlay()
                    raise V2ScheduleError(
                        "ingress commit did not reproduce the probed interval on %s for %r: "
                        "probed [%.12f, %.12f] but booked [%.12f, %.12f]"
                        % (calendar_of[hop].name, key, start, end, booked[0], booked[1]))
                for pos, cand in enumerate(pending):
                    if cand is entry:
                        pending.pop(pos)
                        break
                if float(payload) > 0.0:
                    radio_events.append((booked[0], booked[1], calendar_of[hop].name,
                                         float(payload)))
                    radio_ledger.append({
                        "dag_id": dag_id, "task_id": int(task_id), "hop": hop,
                        "direction": "%s->%s" % (hsrc, hdst), "src": hsrc, "dst": hdst,
                        "bytes": float(payload), "start_s": booked[0], "end_s": booked[1],
                        "service_s": float(booked[2]), "outage_s": float(booked[4]),
                        "queue_wait_s": float(booked[3]), "outage_events": int(booked[5]),
                        "retained_channel": True,
                    })
                    cls = _hop_class(hop)
                    acc["tx"][cls] += float(booked[2])
                    acc["wait"][cls] += float(booked[3])
                    acc["outage_wait"] += float(booked[4])
                    acc["outage_events"] += int(booked[5])
                cursor = float(booked[1])
            arrival = float(route_legs[-1][4]) if route_legs else ready
            if abs(arrival - float(ready_for[key])) > 1e-9 * max(1.0, abs(arrival)):
                _clear_overlay()
                raise V2ScheduleError(
                    "ingress probe/commit mismatch for %r: probed arrival %.12f but the "
                    "commit re-planned %.12f (an unverified interval would have been booked)"
                    % (key, float(ready_for[key]), arrival))
            arrived[key] = arrival
        _clear_overlay()
        if pending:
            raise V2ScheduleError(
                "ingress probe/commit mismatch: %d probed legs were never booked"
                % len(pending))
        return arrived

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
        # The rate trace is PIECEWISE CONSTANT on [k*dt, (k+1)*dt). Integrating with a full-dt
        # step from an arbitrary start applied one interval's rate across the next interval's
        # boundary (F06): a transfer starting at 0.05 s with dt=0.1 s and a x10 rate jump at
        # 0.1 s finished at 0.160 s instead of the analytic 0.115 s. Every step must therefore
        # end at the NEXT TRACE BOUNDARY, an outage transition, or payload completion.
        guard = 0
        while served + 1e-12 < payload:
            steps += 1
            if steps > int(max_service_steps):
                raise V2ScheduleError(
                    "transfer on %s starting at %.6fs did not complete within %d rate steps "
                    "(horizon exhaustion: the open-loop plan cannot finish)"
                    % (link_name, start, int(max_service_steps)))
            if guard > int(max_service_steps) + 1:
                raise V2ScheduleError("service integration did not advance on %s" % link_name)
            guard += 1
            # --- finite horizon, stated explicitly (no silent infinite extension) ---
            # Only enforced when the process actually declares a horizon: the real LinkProcess
            # exposes `horizon_s()`; lightweight test doubles may expose nothing, in which case
            # the horizon is unknown and cannot be checked.
            horizon = _declared_horizon(link_process)
            if horizon is not None and t >= horizon - 1e-12:
                raise V2ScheduleError(
                    "transfer on %s starting at %.6fs reached the end of the realized link "
                    "trace (%.6fs) with %.6f of %.6f bytes served: the frozen horizon does not "
                    "cover this plan, so the rate beyond it is undefined"
                    % (link_name, start, horizon, served, payload))
            rate = base * float(link_process.realized(link_name, t))
            # next trace boundary strictly after `t`
            boundary = (math.floor(t / dt + 1e-9) + 1) * dt
            if rate <= 0.0:
                # an outage pause also stops AT the boundary; it never straddles one
                advance = max(boundary - t, 0.0)
                if advance <= 0.0:
                    advance = dt
                t += advance
                outage_wait += advance
                outage_events += 1
                continue
            step = min(boundary - t, dt, (payload - served) / rate)
            if step <= 0.0:
                # `t` is exactly on the boundary: the next rate applies from here
                t = boundary if boundary > t else t + dt
                continue
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
        cal.clear_plan_overlay()        # a real booking never plans against a pending overlay
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
        _clear_overlay()                # never book against a stale pending overlay
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
    #: (dag_id, predecessor_task_id, destination_location) -> booked delivery end. A
    #: producer's output is transmitted ONCE per destination location and shared by every
    #: successor placed there (a broadcast); see the reuse branch in the booking loop.
    delivered: dict = {}
    while ready_heap:
        _arr, dag_id, task_id = heapq.heappop(ready_heap)
        dag = dag_by_id[dag_id]
        spec = by_dag[dag_id][task_id]
        location = effective_loc[(dag_id, task_id)]
        helper_rejected, reliability_rejected, p_success = admission[(dag_id, task_id)]
        acc = _new_acc()
        state = (helper_states or {}).get(dag.helper_id)
        contact_start = 0.0
        contact_end = float("inf")
        if state is not None:
            contact_start = float(getattr(state, "contact_start_s", 0.0))
            contact_end = float(getattr(state, "contact_end_s", float("inf")))

        # ---- data-ready time from predecessors (outputs stay where they executed) ----
        # The nominal ready time is the latest predecessor finish. Booking happens AFTER the
        # runtime contact feasibility decision below, so that no ingress is ever booked (or
        # billed) for a helper placement that cannot receive it.
        ready = float(dag.arrival_s)
        for p in spec.predecessors:
            ready = max(ready, float(timings[(dag_id, p)].finish_s))

        # ---- ingress planning + helper-placement gates (F08/e, items 1 and 3) ----
        # A HELPER placement is realizable only if (a) the task's INPUT can actually arrive
        # inside the realized contact window and (b) the task's OUTPUT can actually leave it
        # again for every required destination. The pre-fix scheduler did neither: it booked
        # ingress as early as the calendar allowed and discovered the disconnection afterwards
        # (`UE -> HELPER` booked and priced entirely after `contact_end`), it planned each
        # predecessor route in ISOLATION so a second simultaneous ingress to one helper slipped
        # past the window, and it committed the helper before the first forbidden DEPARTURE,
        # leaving a successor to discover it and book a HELPER -> UE relay that could not
        # happen (the 1500-iteration campaign abort).
        #
        # The ingress probe is a non-mutating forecast over the real radio calendars that
        # chains the pending legs of the same task on an overlay so simultaneous ingresses
        # serialise, and `book_ingress` then commits exactly the intervals it checked. The
        # placement decision is structural, not a validation step, so it holds for
        # `validate=False` (the prefix/telescoping path PPO scores) exactly as it does for
        # `validate=True`: no inadmissible helper placement is ever committed.

        def _ingress_requests_for(final_location):
            """The transfers this task needs when it finally runs at `final_location`.

            Rebuilt from the FINAL location, never from the abandoned HELPER attempt, so a
            downgrade keeps every predecessor that is not already resident at the requester and
            drops the transfers nobody performs (the root's own external input is local to the
            UE, and a helper predecessor that stayed on the helper still needs its relay).
            """
            out = []
            if (spec.is_root and float(spec.external_input_bytes) > 0.0
                    and final_location != UE):
                # v1 engine semantics: the external input is uploaded before execution; it may
                # be transmitted to the helper only once the contact window has opened.
                out.append((UE, final_location, float(dag.arrival_s),
                            float(spec.external_input_bytes), ("root", dag_id, task_id)))
            for p in spec.predecessors:
                pt = timings[(dag_id, p)]
                payload = float(by_dag[dag_id][p].output_bytes)
                key = (dag_id, p, final_location)
                if pt.location == final_location:
                    continue
                if key in delivered:
                    # already transmitted once for this destination location (broadcast)
                    continue
                out.append((pt.location, final_location, float(pt.finish_s), payload, key))
            return out

        in_bytes = 0.0
        ingress_requests = _ingress_requests_for(location)
        for p in spec.predecessors:
            pt = timings[(dag_id, p)]
            if pt.location == location:
                ready = max(ready, pt.finish_s)
            elif (dag_id, p, location) in delivered:
                # ---- count-once broadcast (F08/c) ----
                # Two successors in the SAME destination location consume one physical
                # transmission of the producer's output. The transmission is shared; only the
                # first consumer books it (and pays its radio time/energy), the second reuses
                # the already-booked delivery and only records that it received the bytes.
                ready = max(ready, delivered[(dag_id, p, location)])
                in_bytes += float(by_dag[dag_id][p].output_bytes)
        if spec.is_root and float(spec.external_input_bytes) > 0.0 and location != UE:
            in_bytes += float(spec.external_input_bytes)

        def _refuse_helper_placement():
            """Abandon the HELPER placement and rebuild the ingress for the requester."""
            nonlocal location, ingress_requests, in_bytes
            location = UE
            effective_loc[(dag_id, task_id)] = UE
            ingress_requests = _ingress_requests_for(UE)
            in_bytes = sum(float(e[3]) for e in ingress_requests)

        if (location == HELPER and state is not None and math.isfinite(contact_end)
                and ingress_requests):
            ok, _pending, _ready = probe_ingress(ingress_requests,
                                                 (contact_start, contact_end))
            if not ok:
                # (F08/e) the ingress cannot arrive inside the contact window: plan locally.
                # The overlay was discarded, so nothing was booked and nothing is billed.
                _refuse_helper_placement()
                ingress_infeasible += 1

        # ---- EGRESS feasibility is enforced where it binds: the HELPER CPU branch ----
        # The departure check (`_last_departure`) runs with the CPU end the helper actually
        # reserves; when the result cannot leave in time the placement is abandoned INSIDE that
        # branch, before any HELPER -> UE relay is booked. A projected gate HERE would decide
        # the same question with a CPU end the helper might never get (another DAG can still
        # take the interval) and would have to re-derive the failure accounting.

        # ---- a successor may only be admitted to a helper when its fallback is realizable ----
        # If a HELPER-resident predecessor can no longer hand its output to the requester's UE
        # before disconnection, a successor placed on the helper could never be restarted from
        # it (v2 has no checkpointing and no requester-local copy), and a later fallback would
        # try to book a HELPER -> UE relay that cannot happen. Downgrade the successor BEFORE
        # its ingress is committed: the predecessor then fails its own egress gate and falls
        # back too, so no stranded data and no fabricated transfer is ever booked.
        if location == HELPER and state is not None and math.isfinite(contact_end):
            for p in spec.predecessors:
                if timings[(dag_id, p)].location != HELPER:
                    continue
                if not _fallback_input_available((dag_id, p)):
                    _refuse_helper_placement()
                    ingress_infeasible += 1
                    break

        # ---- book the input transfers for the FINAL location ----
        if ingress_requests:
            window = ((contact_start, contact_end)
                      if location == HELPER and state is not None
                      and math.isfinite(contact_end) else (None, None))
            ok, pending_by_key, ready_for = probe_ingress(ingress_requests, window)
            if not ok:
                raise V2ScheduleError(
                    "ingress for %s/%s cannot be planned after the placement was fixed"
                    % (dag_id, task_id))
            arrived = book_ingress(ingress_requests, pending_by_key, ready_for, dag_id,
                                   task_id, acc,
                                   window_start=(contact_start if location == HELPER
                                                 else None))
            for entry in ingress_requests:
                ready = max(ready, arrived[entry[4]])
                if entry[4] != ("root", dag_id, task_id) and location != UE:
                    # the bytes this placement actually carried (a broadcast reuse was counted
                    # above, a HELPER downgrade rebuilt the list from scratch)
                    in_bytes += float(entry[3])
                if entry[4] != ("root", dag_id, task_id):
                    delivered[entry[4]] = float(arrived[entry[4]])

        # ---- CPU phase ----
        contact_failure = False
        restart_penalty = 0.0
        wasted_bytes = 0.0
        restart_bytes = 0.0
        attempt_seconds = 0.0
        attempt_rate = 0.0
        executed_bytes = float(spec.compute_bytes)
        attempted_location = location
        helper_slot = None
        if location == MEC:
            rate = float(compute.mec_cpu_bytes_per_s)
            cpu_duration = executed_bytes / rate
            start, cpu_end = shared_cpu.reserve(ready, cpu_duration)
        elif location == UE:
            rate = float(compute.ue_cpu_bytes_per_s[dag.owner])
            cpu_duration = executed_bytes / rate
            start, cpu_end = ue_cpu[dag.owner].reserve(ready, cpu_duration)
        elif state is None or not math.isfinite(contact_end):
            rate = float(compute.helper_cpu_bytes_per_s[dag.helper_id])
            cpu_duration = executed_bytes / rate
            start, cpu_end = helper_cpu[dag.helper_id].reserve(ready, cpu_duration)
        else:
            # ---- HELPER placement inside a realized contact window ----
            rate = float(compute.helper_cpu_bytes_per_s[dag.helper_id])
            cpu_duration = executed_bytes / rate
            cal_h = helper_cpu[dag.helper_id]
            ready_contact = max(ready, contact_start)
            required_dsts = _required_dsts(spec, dag_id, task_id)
            # (F08/a) NEVER reserve helper CPU that cannot start while the helper is in
            # contact: the attempt would be discarded, and the reservation would block the
            # helper's calendar for work that provably cannot happen. `helper_slot` is the
            # committed interval (or None when nothing was reserved) so that a discarded
            # attempt can be handed back to `Calendar.release`.
            if ready_contact < contact_end - 1e-12:
                helper_slot = cal_h.reserve_indexed(ready_contact, cpu_duration)
                start, cpu_end = helper_slot[1], helper_slot[2]
            else:
                start = cpu_end = ready_contact
            # (F08/a + item 1) The worker is only useful if its RESULT can leave the helper
            # before disconnection. The window binds the helper's OWN transmissions
            # (`_helper_departure`, F08/d), so the failure condition is checked against the
            # route every REQUIRED destination must take -- never against the queue-free
            # `HELPER -> UE` relay the pre-F08 check assumed. A start at/after `contact_end`
            # (queued behind the helper's own workload) is the same failure with zero usable
            # work. In BOTH cases the placement is abandoned INSIDE this branch: the task goes
            # local, so no HELPER -> UE relay is ever booked after the link is gone and no
            # successor is left waiting for data that cannot leave the helper.
            no_window = helper_slot is None or start >= contact_end - 1e-12
            if no_window or (
                    _last_departure(cpu_end, float(spec.output_bytes), required_dsts)
                    > contact_end + CONTACT_SLACK_S):
                contact_failure = True
                if not no_window:
                    # the required output transfer could not leave inside the contact window:
                    # this is the planner-side downgrade surfaced as `helper_egress_infeasible`
                    egress_infeasible += 1
                # Work the helper actually performed before disconnection. It is WASTED: once
                # contact has ended no transfer can carry the result back, and v2 implements NO
                # checkpointing, so a free partial recovery would be physically invalid.
                # Recovery is a FULL local restart from the task's inputs, which the requester
                # already owns.
                served_s = 0.0 if helper_slot is None else max(
                    0.0, min(cpu_end, contact_end) - start)
                wasted_bytes = min(executed_bytes, served_s * rate)
                restart_bytes = executed_bytes
                restart_penalty = restart_bytes / max(
                    1e-9, float(compute.ue_cpu_bytes_per_s[dag.owner]))
                # (F08/a) the discarded interval must be RELEASED: keep only the served prefix
                # [start, start+served_s] (whose energy is billed through `cpu_attempt_seconds`)
                # and give the remainder back to the helper's calendar, so a second DAG is not
                # blocked by work that produced nothing.
                if helper_slot is not None:
                    cal_h.release(helper_slot[0], start + served_s, cpu_end)
                # (F08/e) the local restart must own its INPUTS. A predecessor whose FINAL
                # location is not the UE and whose payload was not already relayed through the
                # UE on its way to the attempted location still sits there; the restart would
                # run on data that is not resident on the requester. The relay is booked (and
                # priced) BEFORE the local CPU is reserved, and its delivery time bounds the
                # restart's ready time. This relay is feasible by construction: the successor
                # gate above only admitted helper placements whose HELPER-resident
                # predecessors can still reach the UE before disconnection.
                for p in spec.predecessors:
                    pt = timings[(dag_id, p)]
                    if pt.location == UE:
                        continue
                    if any(hsrc == UE or hdst == UE for _h, hsrc, hdst
                           in _route_steps(pt.location, attempted_location, link)):
                        continue        # already relayed through the requester on ingress
                    payload = float(by_dag[dag_id][p].output_bytes)
                    ready = max(ready, book(pt.location, UE, max(ready, pt.finish_s),
                                            payload, dag_id, task_id, acc))
                start, cpu_end = ue_cpu[dag.owner].reserve(
                    max(contact_end, ready), restart_penalty)
                cpu_duration = restart_penalty
                # the FINAL execution re-runs the whole task locally, so the executed work of
                # this task is the restart; `wasted_bytes` stays separate
                executed_bytes = restart_bytes
                attempt_seconds = float(served_s)
                attempt_rate = float(rate)
                location = UE
        queue_wait_cpu = max(0.0, start - ready)
        fallback_reserved = 0.0
        if (location in (MEC, HELPER) and standby_for is not None
                and standby_for(str(spec.criticality).upper())):
            # Warm standby hook (F08/a): the requester's worst-case local duration is DECLARED
            # as a planning reserve (`fallback_reserved_s`) but does NOT occupy the vehicle's
            # real CPU calendar. The pre-fix code booked it and never released it, so a standby
            # that was never used moved a later UE task's start (0.0 -> 0.6 s) and the whole
            # makespan while the blocked CPU time was billed as exactly zero joules (warm-standby
            # energy is in `energy.UNMODELED`): time and energy disagreed. A reservation that
            # costs no joules must not cost time either. Keeping the booking instead would
            # require a priced idle/hold power at the requester tier, which the frozen model
            # does not define, so the standby is reported, not committed.
            worst = executed_bytes / max(1e-9, float(compute.ue_cpu_bytes_per_s[dag.owner]))
            window = max(0.0, float(cpu_end) - float(ready))
            if window > 0.0:
                fallback_reserved = min(worst, window)
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
            cpu_attempt_seconds=float(attempt_seconds),
            cpu_attempt_rate_bytes_per_s=float(attempt_rate),
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
    # that extends past it (e.g. a local restart booked after the last task finished)
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
                     # planner-side helper downgrades that used to be discovered only by the
                     # validator (or not at all under `validate=False`): a HELPER placement
                     # whose required output transfer could not leave inside the contact
                     # window (`helper_egress_infeasible`, item 1) or whose serialised ingress
                     # could not arrive inside it (`helper_ingress_infeasible`, item 3). Both
                     # are monotonically accumulated per schedule so a production run can be
                     # checked for "did this rule fire at all".
                     "helper_egress_infeasible": int(egress_infeasible),
                     "helper_ingress_infeasible": int(ingress_infeasible),
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
                                              helper_cpu, ul, dl, v2v,
                                              helper_states=helper_states, compute=compute)
    else:
        # prefix/telescoping evaluation: the same schedule computation without the
        # O(tasks x edges) route re-verification (the final episode schedule is validated)
        result.invariants = {"validated": False}
    return result


def validate_schedule(dags, plans, link, result, shared_cpu, ue_cpu, helper_cpu,
                      ul, dl, v2v, helper_states=None, compute=None) -> dict:
    """Hard invariants; any violation raises.

    `compute` supplies the CPU rates the scheduler actually allocated. When it is given, the
    partial-attempt accounting of every contact failure is re-derived and checked (F08/b);
    without it that check is skipped (the validator stays usable with the pre-F08 signature).
    """
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
    # ---- data residency (F08/e) ----
    # `ready_s` is when a task's inputs are all available, but the check above only compares it
    # with the producers' finish times: it never asks WHERE those outputs ended up. A task that
    # fell back to the UE while its predecessor's result still sat on the disconnected helper
    # therefore validated clean (`validate=True` passed with NO HELPER -> UE transfer at all).
    # Every edge whose endpoints executed at DIFFERENT locations must be covered by an actually
    # booked route from the producer's FINAL location to the consumer's FINAL location, starting
    # after the producer finished and arriving by the consumer's ready time. A producer ->
    # destination transmission is a broadcast shared by every successor in that location, so the
    # witness is a route, not a per-task ledger record. (The witness search can reuse one route
    # for two distinct producers that share a start window; it is a soundness guard for MISSING
    # data movement, not a proof of uniqueness.)
    ledger_by_dag: dict = {}
    for rec in result.radio_ledger:
        ledger_by_dag.setdefault(str(rec.get("dag_id")), []).append(rec)

    def _route_witness(dag_id, route, not_before, deadline):
        cursor = float(not_before)
        for hop, hsrc, hdst in route:
            hits = [r for r in ledger_by_dag.get(dag_id, ())
                    if str(r.get("hop")) == hop and str(r.get("src")) == hsrc
                    and str(r.get("dst")) == hdst
                    and float(r.get("start_s")) >= cursor - 1e-9
                    and float(r.get("end_s")) <= deadline + 1e-9]
            if not hits:
                return False
            cursor = min(float(r["end_s"]) for r in hits)
        return True

    for dag in dags:
        by_id = {t.task_id: t for t in dag.tasks}
        for t in dag.tasks:
            tm = result.timings[(dag.dag_id, t.task_id)]
            for p in t.predecessors:
                ptm = result.timings[(dag.dag_id, p)]
                if ptm.location == tm.location:
                    continue
                if float(by_id[p].output_bytes) <= 0.0:
                    continue                     # nothing to carry: no transfer expected
                route = _route_steps(ptm.location, tm.location, link)
                if not _route_witness(dag.dag_id, route, ptm.finish_s, tm.ready_s):
                    raise V2ScheduleError(
                        "task %s/%s is ready at %.9f on %s but its predecessor %s/%s finished "
                        "at %.9f on %s and no booked %s -> %s route carries its %r output "
                        "bytes: the consumer would run on data that is not resident there"
                        % (dag.dag_id, t.task_id, tm.ready_s, tm.location, dag.dag_id, p,
                           ptm.finish_s, ptm.location, ptm.location, tm.location,
                           float(by_id[p].output_bytes)))
    checks["data_residency"] = True
    # ---- partial-attempt conservation (F08/b) ----
    # `energy.schedule_energy` prices a contact failure as
    #     P(attempted_tier) * cpu_attempt_seconds + P(final_tier) * cpu_s
    # so the scheduler must conserve both the attempted work
    # (`cpu_attempt_seconds * rate(attempted_location) == cpu_wasted_bytes`) and the recovery
    # work (the restart re-runs the WHOLE task because v2 implements no checkpointing). Neither
    # assertion existed before F08: only non-negativity/finiteness was checked, so a scheduler
    # that mis-stated the attempt would have been priced as fact. The residual tolerance is
    # scaled by the byte magnitude because `executed / rate * rate` is not exactly `executed`
    # in binary floating point.
    if compute is not None:
        for dag in dags:
            for t in dag.tasks:
                tm = result.timings[(dag.dag_id, t.task_id)]
                if not tm.contact_failure:
                    continue
                attempted = str(tm.attempted_location)
                if attempted == HELPER and dag.helper_id is not None:
                    rate = float(compute.helper_cpu_bytes_per_s[dag.helper_id])
                elif attempted == MEC:
                    rate = float(compute.mec_cpu_bytes_per_s)
                elif attempted == UE:
                    rate = float(compute.ue_cpu_bytes_per_s[dag.owner])
                else:
                    raise V2ScheduleError(
                        "contact failure %s/%s reports attempted location %r, which has no CPU "
                        "rate in the compute spec" % (dag.dag_id, t.task_id, attempted))
                served = float(tm.cpu_attempt_seconds) * rate
                wasted = float(tm.cpu_wasted_bytes)
                if abs(served - wasted) > 1e-9 * max(1.0, abs(wasted)):
                    raise V2ScheduleError(
                        "partial-attempt billing is not conserved for %s/%s: attempted location "
                        "%s served %.12f s at %r B/s = %r bytes but cpu_wasted_bytes = %r "
                        "(energy would price a fiction)"
                        % (dag.dag_id, t.task_id, attempted, tm.cpu_attempt_seconds, rate,
                           served, wasted))
                computes = float(t.compute_bytes)
                if abs(float(tm.cpu_restart_bytes) - computes) > 1e-9 * max(1.0, abs(computes)):
                    raise V2ScheduleError(
                        "contact failure %s/%s restarts %r bytes but the task computes %r: v2 "
                        "implements no checkpointing, so the local restart re-runs the WHOLE "
                        "task" % (dag.dag_id, t.task_id, tm.cpu_restart_bytes, t.compute_bytes))
                if wasted > computes + 1e-9 * max(1.0, abs(computes)):
                    raise V2ScheduleError(
                        "contact failure %s/%s wasted %r bytes, more than the %r bytes the task "
                        "computes" % (dag.dag_id, t.task_id, wasted, t.compute_bytes))
        checks["cpu_attempt_conservation"] = True
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
    # helper CONTACT invariant (F07/F08). The validator previously accepted a schedule whose
    # helper result was delivered AFTER the helper disconnected, because the return was planned
    # with the queue-free service duration and no invariant re-checked the booked interval. Every
    # transfer that LEAVES a helper must complete inside that helper's contact window; otherwise
    # the task must have been reported as a contact failure / fallback, not as a helper success.
    # F08/f: the SAME window binds transfers that ARRIVE at the helper. The pre-fix scope checked
    # only `src == HELPER`, so ingress booked before the window opened or after it closed was
    # invisible (and the `src == HELPER` start half was vacuous because the helper CPU is
    # reserved at `max(ready, contact_start)`).
    if helper_states:
        dag_by_id = {d.dag_id: d for d in dags}
        checked_helper_transfers = 0
        for rec in result.radio_ledger:
            src, dst = str(rec.get("src")), str(rec.get("dst"))
            if src != HELPER and dst != HELPER:
                continue
            dag = dag_by_id.get(str(rec.get("dag_id")))
            hid = getattr(dag, "helper_id", None) if dag is not None else None
            state = (helper_states or {}).get(hid) if hid is not None else None
            if state is None:
                continue
            contact_start = float(getattr(state, "contact_start_s", 0.0))
            contact_end = float(getattr(state, "contact_end_s", float("inf")))
            start = float(rec.get("start_s"))
            end = float(rec.get("end_s"))
            if math.isfinite(contact_end):
                if end > contact_end + 1e-9:
                    raise V2ScheduleError(
                        "helper %r transfer for %s/%s (%s -> %s) completes at %.9f, after "
                        "contact ends at %.9f: no link to the helper exists after disconnection"
                        % (hid, rec.get("dag_id"), rec.get("task_id"), src, dst, end,
                           contact_end))
                if start > contact_end + 1e-9:
                    raise V2ScheduleError(
                        "helper %r transfer for %s/%s (%s -> %s) starts at %.9f, after contact "
                        "ends at %.9f: ingress that arrives after disconnection is billed bytes "
                        "no task can read"
                        % (hid, rec.get("dag_id"), rec.get("task_id"), src, dst, start,
                           contact_end))
            if start + 1e-9 < contact_start:
                raise V2ScheduleError(
                    "helper %r transfer for %s/%s (%s -> %s) starts at %.9f, before contact "
                    "begins at %.9f" % (hid, rec.get("dag_id"), rec.get("task_id"), src, dst,
                                        start, contact_start))
            checked_helper_transfers += 1
        checks["helper_contact_window"] = True
        checks["helper_transfers_checked"] = checked_helper_transfers
    return checks
