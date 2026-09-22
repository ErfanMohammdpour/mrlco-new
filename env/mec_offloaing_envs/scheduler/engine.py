"""Spec-faithful production scheduling engine (Phase 1)."""

from __future__ import annotations

import heapq
from collections.abc import Sequence

from .calendar import make_calendars
from .model import (
    CanonicalDAG,
    EnergyBreakdown,
    Location,
    ResourceInterval,
    ScheduleResult,
    TaskExecutionRecord,
    TransferRecord,
)
from .resources import ResourceConfig
from .energy_model import hop_energy_fields
from .routes import HOP_TO_RESOURCE, hop_destination, route


def _topo_order(graph: CanonicalDAG, decoder_rank: dict[int, int]) -> list[int]:
    preds = graph.predecessors()
    succs = graph.successors()
    indeg = {tid: len(preds[tid]) for tid in graph.tasks}
    heap: list[tuple[int, int]] = []
    for tid, d in indeg.items():
        if d == 0:
            heapq.heappush(heap, (decoder_rank[tid], tid))
    order: list[int] = []
    while heap:
        _, u = heapq.heappop(heap)
        order.append(u)
        for v in succs[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                heapq.heappush(heap, (decoder_rank[v], v))
    if len(order) != len(graph.tasks):
        raise ValueError("cyclic DAG")
    return order


def _cpu_resource(loc: Location) -> str:
    return {
        Location.UE: "UE_CPU",
        Location.MEC: "MEC_CPU",
        Location.HELPER: "HELPER_CPU",
    }[loc]


def schedule(
    graph: CanonicalDAG,
    decoder_order: Sequence[int],
    actions: Sequence[int | str | Location],
    resources: ResourceConfig,
) -> ScheduleResult:
    """Schedule a full action plan under frozen MARGO semantics.

    Actions align to `decoder_order` (one location per task id in that order).
    Output residency = execution location. Only sinks return to UE for makespan.
    Dependency transfers use `edge_output_bytes`; roots use `external_input_bytes`.
    """
    if len(decoder_order) != len(graph.tasks):
        raise ValueError("decoder_order length mismatch")
    if len(actions) != len(decoder_order):
        raise ValueError("actions length mismatch")
    if set(decoder_order) != set(graph.tasks):
        raise ValueError("decoder_order must cover all task ids exactly once")

    decoder_rank = {tid: i for i, tid in enumerate(decoder_order)}
    locs = {
        tid: Location.from_action(actions[i]) for i, tid in enumerate(decoder_order)
    }
    order = _topo_order(graph, decoder_rank)
    preds = graph.predecessors()
    succs = graph.successors()

    cals = make_calendars()
    energy = EnergyBreakdown()
    transfers: list[TransferRecord] = []
    intervals: list[ResourceInterval] = []
    finish: dict[int, float] = {}
    start: dict[int, float] = {}
    loc_out: dict[int, Location] = {}
    # ②A: when a task's OUTPUT becomes usable by each consumer.
    deliveries: dict[int, list[float]] = {tid: [] for tid in graph.tasks}

    def add_energy_hop(hop: str, duration: float, src_loc: Location) -> None:
        """Delegates to energy_model.hop_energy_fields so the legacy and
        physical_v1 (TX-payer, optional RX) accounting never diverge."""
        for field_name, joules in hop_energy_fields(
            hop, duration, src_loc, resources
        ).items():
            setattr(energy, field_name, getattr(energy, field_name) + joules)

    def move_bytes(
        nbytes: int,
        hops: list[str],
        earliest: float,
        src_loc: Location,
        edge_src: int | None,
        edge_dst: int | None,
    ) -> float:
        if not hops or nbytes == 0:
            return earliest
        t = earliest
        cur = src_loc
        for hop_i, hop in enumerate(hops):
            dur = nbytes / resources.hop_rate(hop)
            res_name = HOP_TO_RESOURCE[hop]
            s, e = cals[res_name].reserve(dur, t)
            dst = hop_destination(cur, hop)
            # Zero-duration hops do not occupy capacity; do not emit intervals/transfers.
            if dur > 0.0:
                intervals.append(
                    ResourceInterval(resource=res_name, start=s, end=e, hop=hop)
                )
                add_energy_hop(hop, dur, cur)
                transfers.append(
                    TransferRecord(
                        hop=hop,
                        hop_index=hop_i,
                        bytes=nbytes,
                        start=s,
                        end=e,
                        src_location=cur,
                        dst_location=dst,
                        src_task_id=edge_src,
                        dst_task_id=edge_dst,
                    )
                )
            t = e
            cur = dst
        return t

    for tid in order:
        task = graph.tasks[tid]
        loc = locs[tid]
        ready = 0.0

        ext = int(task.external_input_bytes)
        if ext > 0:
            hops = route(Location.UE, loc)
            ready = max(ready, move_bytes(ext, hops, 0.0, Location.UE, None, tid))

        for edge in sorted(
            preds[tid],
            key=lambda e: (decoder_rank[e.src_task_id], e.src_task_id),
        ):
            src = edge.src_task_id
            hops = route(loc_out[src], loc)
            arrival = move_bytes(
                edge.edge_output_bytes,
                hops,
                finish[src],
                loc_out[src],
                src,
                tid,
            )
            ready = max(ready, arrival)
            deliveries[src].append(arrival)

        dur = task.compute_workload_bytes / resources.cpu_rate_for_task(loc, task)
        res_name = _cpu_resource(loc)
        s, e = cals[res_name].reserve(dur, ready)
        if dur > 0.0:
            intervals.append(
                ResourceInterval(resource=res_name, start=s, end=e, task_id=tid)
            )
        start[tid] = s
        finish[tid] = e
        loc_out[tid] = loc

        # Compute energy: legacy = duration*rho*f^zeta (MEC compute = 0);
        # physical_v1 = kappa * C * f^2 on every tier, including MEC.
        cpu_field = resources.compute_energy_field(loc)
        setattr(
            energy,
            cpu_field,
            getattr(energy, cpu_field)
            + resources.compute_energy_joules(
                loc, task.compute_workload_bytes, dur, task.cycles_per_bit
            ),
        )

    result_at_ue = 0.0
    for tid in sorted(graph.sinks(), key=lambda x: (decoder_rank[x], x)):
        out_b = int(graph.tasks[tid].task_output_bytes)
        hops = route(loc_out[tid], Location.UE)
        if hops:
            returned = move_bytes(out_b, hops, finish[tid], loc_out[tid], tid, None)
        else:
            returned = finish[tid]
        deliveries[tid].append(returned)
        result_at_ue = max(result_at_ue, returned)

    global_xi = getattr(
        getattr(resources, "energy_model", None), "cycles_per_bit", None
    )
    task_records: dict[int, TaskExecutionRecord] = {}
    soft_sum = 0.0
    soft_norm_sum = 0.0
    tardiness_values: list[float] = []
    firm_miss = 0
    hard_miss = 0
    n_typed = 0
    for tid in graph.tasks:
        task = graph.tasks[tid]
        # ②A.1: two distinct availability notions.
        #   first_available     = earliest arrival at ANY consumer (diagnostic)
        #   all_consumers_ready = arrival at EVERY required consumer
        #                         (max over successors; UE return for a sink)
        #                       -> PRIMARY deadline basis: the output is not
        #                       usable while one consumer still cannot read it.
        seen = sorted(deliveries[tid])
        first_available = seen[0] if seen else finish[tid]
        all_consumers_ready = seen[-1] if seen else finish[tid]
        deadline = task.deadline_s
        tardiness = 0.0
        if deadline is not None:
            tardiness = max(0.0, all_consumers_ready - float(deadline))
        dtype = task.deadline_type
        if dtype != "none":
            n_typed += 1
            tardiness_values.append(tardiness)
            if dtype == "soft":
                w = float(task.tardiness_weight)
                soft_sum += w * tardiness
                if deadline and float(deadline) > 0.0:
                    soft_norm_sum += w * tardiness / float(deadline)
            elif dtype == "firm":
                if tardiness > 0.0:
                    firm_miss += 1
            elif dtype == "hard":
                if tardiness > 0.0:
                    hard_miss += 1
        task_records[tid] = TaskExecutionRecord(
            task_id=tid,
            location=locs[tid],
            start=start[tid],
            finish=finish[tid],
            output_location=loc_out[tid],
            first_available=first_available,
            all_consumers_ready=all_consumers_ready,
            deadline_s=deadline,
            deadline_type=dtype,
            criticality_class=str(task.criticality_class),
            tardiness_weight=float(task.tardiness_weight),
            tardiness_s=tardiness,
        )

    return ScheduleResult(
        tasks=task_records,
        transfers=transfers,
        resource_intervals=intervals,
        energy=energy,
        makespan_seconds=result_at_ue,
        terminal_return_time=result_at_ue,
        topo_order=order,
        soft_tardiness_s=soft_sum,
        soft_tardiness_normalized=soft_norm_sum,
        firm_miss_count=firm_miss,
        firm_miss_rate=(firm_miss / n_typed) if n_typed else 0.0,
        hard_miss_count=hard_miss,
        hard_miss_rate=(hard_miss / n_typed) if n_typed else 0.0,
        hard_feasible=(hard_miss == 0),
        mean_tardiness_s=(sum(tardiness_values) / len(tardiness_values)) if tardiness_values else 0.0,
        max_tardiness_s=max(tardiness_values) if tardiness_values else 0.0,
    )
