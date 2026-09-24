"""Adapter: OffloadingTaskGraph → CanonicalDAG + legacy plan validation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .engine import schedule
from .energy_api import attribute_energy_by_task
from .model import CanonicalDAG, CanonicalTask, ScheduleResult
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
        # Per-task physics/deadline fields are optional on the legacy task
        # object; absent attributes keep the schema defaults (global
        # cycles_per_bit, no deadline).
        tasks.append(
            CanonicalTask(
                task_id=i,
                compute_workload_bytes=int(task.processing_data_size),
                task_output_bytes=int(task.transmission_data_size),
                external_input_bytes=external,
                cycles_per_bit=getattr(task, "cycles_per_bit", None),
                deadline_s=getattr(task, "deadline_s", None),
                deadline_type=str(getattr(task, "deadline_type", "none")),
                criticality_class=str(getattr(task, "criticality_class", "medium")),
                tardiness_weight=float(
                    getattr(task, "tardiness_weight", getattr(task, "criticality", 1.0))
                ),
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


class SchedulerConfigError(ValueError):
    """A resolved scheduler config is missing, dropped or inconsistent."""


def _check_cluster_consistency(resource_cluster: Any, config: ResourceConfig) -> None:
    """Legacy-facing rate fields on `Resources` must agree with the config.

    No silent "config wins": a mismatch means the process would schedule with one
    set of rates while the env reports another.
    """
    mbps_to_Bps = 1024.0 * 1024.0 / 8.0
    checks = (
        ("mec_process_capable", float(resource_cluster.mec_process_capable),
         config.mec_cpu_bytes_per_second),
        ("mobile_process_capable", float(resource_cluster.mobile_process_capable),
         config.ue_cpu_bytes_per_second),
        ("v2v_process_capable", float(resource_cluster.v2v_process_capable),
         config.helper_cpu_bytes_per_second),
        ("bandwidth_up", float(resource_cluster.bandwidth_up) * mbps_to_Bps,
         config.mec_uplink_bytes_per_second),
        ("bandwidth_dl", float(resource_cluster.bandwidth_dl) * mbps_to_Bps,
         config.mec_downlink_bytes_per_second),
        ("v2v_bandwidth", float(resource_cluster.v2v_bandwidth) * mbps_to_Bps,
         config.v2v_bytes_per_second),
    )
    for name, actual, expected in checks:
        if abs(float(actual) - float(expected)) > 1e-9 * max(1.0, abs(expected)):
            raise SchedulerConfigError(
                "Resources.%s disagrees with scheduler_config (%r != %r)"
                % (name, actual, expected)
            )


def resource_config_from_cluster(
    resource_cluster: Any, *, strict: bool = False
) -> ResourceConfig:
    """Return the scheduler config for this cluster.

    With `Resources.scheduler_config` set, the SAME immutable object is returned:
    nothing is reconstructed from `energy_config` or rate fields, so no model, spec
    or hash can be dropped. `strict=True` (primary production paths) refuses the
    legacy fallback outright.
    """
    resolved = getattr(resource_cluster, "scheduler_config", None)
    if resolved is not None:
        if not isinstance(resolved, ResourceConfig):
            raise SchedulerConfigError(
                "Resources.scheduler_config must be a ResourceConfig, got %r"
                % type(resolved).__name__
            )
        _check_cluster_consistency(resource_cluster, resolved)
        return resolved
    if strict:
        raise SchedulerConfigError(
            "no scheduler_config on this cluster: primary training/validation "
            "paths must be given a resolved config, not the legacy fallback"
        )
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
        # provenance: this object was rebuilt from legacy fields, not resolved
        energy_scope=str(cfg.get("energy_scope", "") or ""),
    )


def legacy_adapter_fallback(config: ResourceConfig) -> bool:
    """True when the config came from the legacy cluster-rebuild path."""
    return not config.source_config_sha256 and config.timing_model == "legacy_frozen_rates"


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
    energy_map = attribute_energy_by_task(result, resources)
    energy_list = [float(energy_map.get(tid, 0.0)) for tid in decoder_order]
    return result, deltas, energy_list
