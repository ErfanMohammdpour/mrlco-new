#!/usr/bin/env python3
"""Canonical scheduling objects built from a FROZEN automotive graph record.

This is the single construction site used by training, the parity test and the
HEFT-v2 baseline. It never mutates the record and never resizes a payload.

  T_i^x  = 8 * xi * W_i / f_x            (bytes / (f/(8*xi)))
  T_tx,e = 8 * B_e / R_link              (edge-specific B_e, link-specific R)
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence


class AutomotiveDagError(ValueError):
    """Refused. Never silently repaired."""


def canonical_dag(graph: Mapping[str, Any]):
    from env.mec_offloaing_envs.scheduler.model import (
        CanonicalDAG,
        CanonicalTask,
    )

    from .automotive_resources import dataset_reference_parameters

    xi = dataset_reference_parameters()["xi_cycles_per_bit"]
    tasks = []
    for t in graph["tasks"]:
        tasks.append(
            CanonicalTask(
                task_id=int(t["task_id"]),
                compute_workload_bytes=int(t["compute_workload_bytes"]),
                task_output_bytes=int(t["task_output_bytes"]),
                external_input_bytes=int(t.get("external_input_bytes", 0)),
                cycles_per_bit=float(xi),
                deadline_s=float(t["deadline_s"]),
                deadline_type=str(t.get("deadline_type", "firm")),
                criticality_class=str(t["criticality"]).lower(),
                tardiness_weight=float(t.get("tardiness_weight", 1.0)),
            )
        )
    edges = [(int(e["src"]), int(e["dst"]), int(e["payload_bytes"])) for e in graph["edges"]]
    return CanonicalDAG.from_records(tasks, edges)


def decoder_order(graph: Mapping[str, Any]) -> list[int]:
    """Stable topological order (the dataset stores tasks 0..19 topologically)."""
    order = [int(t["task_id"]) for t in graph["tasks"]]
    if sorted(order) != list(range(len(order))):
        raise AutomotiveDagError("task ids must be exactly 0..n-1")
    rank = {tid: i for i, tid in enumerate(order)}
    for e in graph["edges"]:
        if rank[int(e["src"])] >= rank[int(e["dst"])]:
            raise AutomotiveDagError(
                "stored task order is not topological for edge %s->%s"
                % (e["src"], e["dst"])
            )
    return order


def resources_for_graph(graph: Mapping[str, Any], *, co_physical: bool = True):
    from .automotive_resources import (
        co_physical_config_for_graph,
        legacy_mixed_config_for_graph,
    )

    builder = co_physical_config_for_graph if co_physical else legacy_mixed_config_for_graph
    return builder(graph["resource"], source_sha256=str(graph.get("canonical_sha256", "")))


def plan_from_actions(graph: Mapping[str, Any], actions: Sequence[int]) -> list[tuple[int, int]]:
    order = decoder_order(graph)
    if len(actions) != len(order):
        raise AutomotiveDagError("plan length != task count")
    for a in actions:
        if int(a) not in (0, 1, 2):
            raise AutomotiveDagError("action must be 0/1/2, got %r" % (a,))
    return [(tid, int(a)) for tid, a in zip(order, actions)]


def schedule_actions(graph: Mapping[str, Any], actions: Sequence[int], *,
                     co_physical: bool = True):
    """Schedule `actions` (aligned to `decoder_order`) under frozen MARGO semantics.

    The plan is validated through `plan_from_actions` first: a wrong-length plan or an
    action outside {0, 1, 2} is refused here rather than silently reinterpreted by the
    engine's positional zip.
    """
    from env.mec_offloaing_envs.scheduler.engine import schedule

    dag = canonical_dag(graph)
    order = decoder_order(graph)
    plan = plan_from_actions(graph, actions)
    return schedule(dag, order, [a for _tid, a in plan], resources_for_graph(
        graph, co_physical=co_physical))


def tier_frequencies(graph: Mapping[str, Any]) -> dict:
    r = graph["resource"]
    return {0: float(r["f_ue_hz"]), 1: float(r["f_mec_hz"]), 2: float(r["f_helper_hz"])}
