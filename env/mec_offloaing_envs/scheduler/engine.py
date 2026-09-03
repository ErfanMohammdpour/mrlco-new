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

    def add_energy_hop(hop: str, duration: float, src_loc: Location) -> None:
        if hop == "MEC_UL":
            energy.ue_mec_uplink_joules += duration * resources.ptx_mec_w
        elif hop == "MEC_DL":
            energy.ue_mec_downlink_joules += duration * resources.prx_mec_w
        elif hop == "V2V":
            # Both endpoints are mobile (UE + HELPER) → tx + rx each hop.
            if src_loc == Location.UE:
                energy.ue_v2v_tx_joules += duration * resources.ptx_v2v_w
                energy.helper_v2v_rx_joules += duration * resources.prx_v2v_w
            elif src_loc == Location.HELPER:
                energy.helper_v2v_tx_joules += duration * resources.ptx_v2v_w
                energy.ue_v2v_rx_joules += duration * resources.prx_v2v_w
            else:
                # MEC_DL then V2V: data at UE sending to HELPER
                energy.ue_v2v_tx_joules += duration * resources.ptx_v2v_w
                energy.helper_v2v_rx_joules += duration * resources.prx_v2v_w

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
            ready = max(
                ready,
                move_bytes(
                    edge.edge_output_bytes,
                    hops,
                    finish[src],
                    loc_out[src],
                    src,
                    tid,
                ),
            )

        dur = task.compute_workload_bytes / resources.cpu_rate(loc)
        res_name = _cpu_resource(loc)
        s, e = cals[res_name].reserve(dur, ready)
        if dur > 0.0:
            intervals.append(
                ResourceInterval(resource=res_name, start=s, end=e, task_id=tid)
            )
        start[tid] = s
        finish[tid] = e
        loc_out[tid] = loc

        if loc == Location.UE:
            energy.ue_local_cpu_joules += (
                dur * resources.rho_ue * (resources.f_l**resources.zeta)
            )
        elif loc == Location.HELPER:
            energy.helper_compute_joules += (
                dur * resources.rho_helper * (resources.f_v2v**resources.zeta)
            )

    result_at_ue = 0.0
    for tid in sorted(graph.sinks(), key=lambda x: (decoder_rank[x], x)):
        out_b = int(graph.tasks[tid].task_output_bytes)
        hops = route(loc_out[tid], Location.UE)
        if hops:
            result_at_ue = max(
                result_at_ue,
                move_bytes(out_b, hops, finish[tid], loc_out[tid], tid, None),
            )
        else:
            result_at_ue = max(result_at_ue, finish[tid])

    task_records = {
        tid: TaskExecutionRecord(
            task_id=tid,
            location=locs[tid],
            start=start[tid],
            finish=finish[tid],
            output_location=loc_out[tid],
        )
        for tid in graph.tasks
    }

    return ScheduleResult(
        tasks=task_records,
        transfers=transfers,
        resource_intervals=intervals,
        energy=energy,
        makespan_seconds=result_at_ue,
        terminal_return_time=result_at_ue,
        topo_order=order,
    )
