"""Adapter: OffloadingTaskGraph → CanonicalDAG + legacy plan validation."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Any

from .engine import schedule
from .model import CanonicalDAG, CanonicalTask, Location, ScheduleResult
from .resources import ResourceConfig


class AdapterValidationError(ValueError):
    """Invalid plan or task graph for canonical scheduling."""


def _raw_edges_from_task_graph(task_graph: Any) -> list[tuple[int, int, int]]:
    """Use edge_set records, not dependency_matrix (matrix overwrites duplicates)."""
    edges: list[tuple[int, int, int]] = []
    for edge in task_graph.edge_set:
        # edge = [src, src_depth, src_proc, transmission_cost, dst, dst_depth, dst_proc]
        src = int(edge[0])
        dst = int(edge[4])
        nbytes = int(edge[3])
        edges.append((src, dst, nbytes))
    return edges


def to_canonical_dag(task_graph: Any) -> CanonicalDAG:
    tasks: list[CanonicalTask] = []
    for i, task in enumerate(task_graph.task_list):
        is_root = len(task_graph.pre_task_sets[i]) == 0
        external = int(task.processing_data_size) if is_root else 0
        tasks.append(
            CanonicalTask(
                task_id=i,
                compute_workload_bytes=int(task.processing_data_size),
                task_output_bytes=int(task.transmission_data_size),
                external_input_bytes=external,
            )
        )
    return CanonicalDAG.from_records(tasks, _raw_edges_from_task_graph(task_graph))


def validate_plan(task_graph: Any, plan: Sequence[tuple[int, int]]) -> tuple[list[int], list[int]]:
    n = int(task_graph.task_number)
    if len(plan) != n:
        raise AdapterValidationError(f"plan length {len(plan)} != task_number {n}")

    decoder_order = [int(tid) for tid, _ in plan]
    actions = [int(a) for _, a in plan]

    if sorted(decoder_order) != list(range(n)):
        raise AdapterValidationError("decoder_order must be a permutation of task ids")

    for a in actions:
        if a not in (0, 1, 2):
            raise AdapterValidationError(f"action must be 0/1/2, got {a}")

    # Topological: every raw edge src before dst in decoder_order
    rank = {tid: i for i, tid in enumerate(decoder_order)}
    for src, dst, _ in _raw_edges_from_task_graph(task_graph):
        if rank[src] >= rank[dst]:
            raise AdapterValidationError(
                f"decoder_order not topological for edge {src}->{dst}"
            )

    return decoder_order, actions


def resource_config_from_cluster(resource_cluster: Any) -> ResourceConfig:
    cfg = resource_cluster.energy_config or {}
    mbps_to_Bps = 1024.0 * 1024.0 / 8.0
    return ResourceConfig(
        ue_cpu_bytes_per_second=float(resource_cluster.mobile_process_capable),
        mec_cpu_bytes_per_second=float(resource_cluster.mec_process_capable),
        helper_cpu_bytes_per_second=float(resource_cluster.v2v_process_capable),
        mec_uplink_bytes_per_second=float(resource_cluster.bandwidth_up) * mbps_to_Bps,
        mec_downlink_bytes_per_second=float(resource_cluster.bandwidth_dl) * mbps_to_Bps,
        v2v_bytes_per_second=float(resource_cluster.v2v_bandwidth) * mbps_to_Bps,
        rho_ue=float(cfg.get("rho", 1.0)),
        f_l=float(cfg.get("f_l", 1.0)),
        zeta=float(cfg.get("zeta", 2.0)),
        ptx_mec_w=float(cfg.get("ptx", 0.1)),
        prx_mec_w=float(cfg.get("prx", 0.05)),
        ptx_v2v_w=float(cfg.get("ptx_v2v", 0.06)),
        prx_v2v_w=float(cfg.get("prx_v2v", 0.03)),
        rho_helper=float(cfg.get("rho_v2v", 0.7)),
        f_v2v=float(cfg.get("f_v2v", 1.0)),
    )


def _per_task_energy(result: ScheduleResult, resources: ResourceConfig) -> dict[int, float]:
    out: dict[int, float] = defaultdict(float)
    for tid, rec in result.tasks.items():
        dur = rec.finish - rec.start
        if rec.location == Location.UE:
            out[tid] += dur * resources.rho_ue * (resources.f_l**resources.zeta)
        elif rec.location == Location.HELPER:
            out[tid] += dur * resources.rho_helper * (resources.f_v2v**resources.zeta)
    for t in result.transfers:
        owner = t.dst_task_id if t.dst_task_id is not None else t.src_task_id
        if owner is None:
            continue
        dur = t.end - t.start
        if t.hop == "MEC_UL":
            out[owner] += dur * resources.ptx_mec_w
        elif t.hop == "MEC_DL":
            out[owner] += dur * resources.prx_mec_w
        elif t.hop == "V2V":
            out[owner] += dur * (resources.ptx_v2v_w + resources.prx_v2v_w)
    return dict(out)


def shaped_latency_deltas(result: ScheduleResult, decoder_order: Sequence[int]) -> list[float]:
    """Compatibility deltas along decoder order; sum equals makespan_seconds."""
    current = 0.0
    deltas: list[float] = []
    for tid in decoder_order:
        event = result.tasks[int(tid)].finish
        nxt = max(current, event)
        deltas.append(nxt - current)
        current = nxt
    remainder = result.makespan_seconds - current
    if remainder > 1e-12:
        deltas[-1] += remainder
    elif remainder < -1e-9:
        # Numerical / sink-before-last-task: clamp last
        deltas[-1] = max(0.0, deltas[-1] + remainder)
    return deltas


def schedule_via_adapter(
    task_graph: Any,
    plan: Sequence[tuple[int, int]],
    resources: ResourceConfig,
) -> tuple[ScheduleResult, list[float], list[float]]:
    """Validate, schedule, return (result, latency_deltas, per_step_energy)."""
    decoder_order, actions = validate_plan(task_graph, plan)
    dag = to_canonical_dag(task_graph)
    result = schedule(dag, decoder_order, actions, resources)
    deltas = shaped_latency_deltas(result, decoder_order)
    energy_map = _per_task_energy(result, resources)
    energy_list = [float(energy_map.get(tid, 0.0)) for tid in decoder_order]
    return result, deltas, energy_list
