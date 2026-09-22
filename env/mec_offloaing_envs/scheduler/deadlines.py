"""Deterministic deadline assignment rules (②A) — DIAGNOSTIC ONLY.

    *** These are synthetic diagnostic rules, NOT the final benchmark deadline
    generator. ***  The final dataset plan is
        D_G -> EFT/LFT -> per-task subdeadlines   (feasibility-aware generation),
    which is a later phase. Everything below exists so unit tests and
    diagnostics can attach reproducible deadlines to the current dataset.

    Deadline basis: a task is measured against `all_consumers_ready`
    (max_j delivery(i->j); UE return for a sink) — see scheduler/engine.py.

The frozen dataset carries no deadlines, so deadlines must be *assigned* by an
explicit, reproducible rule.  Measurement semantics are fixed in the engine:
a task misses its deadline when its OUTPUT is not usable in time
(`F_available > deadline_s`), not when its compute finishes late.

Rules (all deterministic; no randomness, no dataset mutation):

    none                     -> every task deadline_type="none" (default; the
                                scheduler result is byte-identical to before)
    uniform_of_all_mec       -> d_i = factor * L_allMEC            (same for all)
    uniform_of_all_ue        -> d_i = factor * L_allUE
    per_task_slack_of_ref    -> d_i = factor * finish_i(reference plan)
    per_task_slack_of_ready  -> d_i = factor * all_consumers_ready_i(reference)
    per_task_slack_of_first_avail -> d_i = factor * first_available_i(reference)
                                (diagnostic comparison only; NOT the primary
                                 deadline basis)

`criticality` is a weight used by soft tardiness; `deadline_type` selects the
channel ("none" | "soft" | "firm" | "hard").
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from .adapter import schedule_via_adapter
from .energy_api import compute_reference_ranges, pure_location_plan
from .model import CanonicalDAG
from .resources import ResourceConfig

DIAGNOSTIC_ONLY = True

DEADLINE_RULES = (
    "none",
    "uniform_of_all_mec",
    "uniform_of_all_ue",
    "per_task_slack_of_ref",
    "per_task_slack_of_ready",
    "per_task_slack_of_first_avail",
)


def assign_deadlines(
    task_graph: Any,
    resources: ResourceConfig,
    *,
    rule: str = "none",
    factor: float = 1.2,
    deadline_type: str = "soft",
    tardiness_weight: float = 1.0,
    tardiness_weight_by_task: dict[int, float] | None = None,
    criticality_class: str = "medium",
    criticality_class_by_task: dict[int, str] | None = None,
    reference_action: int = 1,
) -> Any:
    """Attach `deadline_s` / `deadline_type` / `criticality` to a task graph.

    Returns the same object mutated in place (the legacy task graph is a plain
    Python object used only for scheduling). `rule="none"` only stamps
    `deadline_type` and returns immediately.
    """
    if rule not in DEADLINE_RULES:
        raise ValueError("rule must be one of %s, got %r" % (DEADLINE_RULES, rule))
    factor = float(factor)
    if factor <= 0.0:
        raise ValueError("factor must be > 0")

    order = [int(t) for t in task_graph.prioritize_sequence]
    if rule == "none":
        for tid in order:
            task = task_graph.task_list[tid]
            task.deadline_type = "none"
            task.deadline_s = None
            task.tardiness_weight = 1.0
            task.criticality_class = "medium"
        return task_graph

    refs = compute_reference_ranges(task_graph, resources)
    if rule == "uniform_of_all_mec":
        value = factor * float(refs.L_mec)
    elif rule == "uniform_of_all_ue":
        value = factor * float(refs.L_ue)
    else:
        plan = pure_location_plan(order, int(reference_action))
        result, _, _ = schedule_via_adapter(task_graph, plan, resources)
        value = None
        per_task = {}
        for tid in order:
            rec = result.tasks[tid]
            if rule == "per_task_slack_of_ref":
                base = rec.finish
            elif rule == "per_task_slack_of_ready":
                base = rec.availability_seconds      # all_consumers_ready
            else:
                base = (
                    rec.first_available
                    if rec.first_available is not None
                    else rec.finish
                )
            per_task[tid] = factor * float(base)

    for tid in order:
        task = task_graph.task_list[tid]
        task.deadline_s = float(value) if value is not None else float(per_task[tid])
        task.deadline_type = str(deadline_type)
        task.tardiness_weight = float(
            (tardiness_weight_by_task or {}).get(tid, tardiness_weight)
        )
        task.criticality_class = str(
            (criticality_class_by_task or {}).get(tid, criticality_class)
        )
    return task_graph


def with_deadlines(
    dag: CanonicalDAG,
    deadlines_s: dict[int, float],
    *,
    deadline_type: str = "soft",
    tardiness_weight: float = 1.0,
    tardiness_weight_by_task: dict[int, float] | None = None,
    criticality_class: str = "medium",
) -> CanonicalDAG:
    """Pure variant: return a new CanonicalDAG with deadlines stamped."""
    tasks = {
        tid: replace(
            task,
            deadline_s=float(deadlines_s[tid]) if tid in deadlines_s else task.deadline_s,
            deadline_type=deadline_type if tid in deadlines_s else task.deadline_type,
            criticality_class=criticality_class,
            tardiness_weight=float(
                (tardiness_weight_by_task or {}).get(tid, tardiness_weight)
            ),
        )
        for tid, task in dag.tasks.items()
    }
    return CanonicalDAG(
        tasks=tasks,
        edges=list(dag.edges),
        edge_record_count=dag.edge_record_count,
        unique_edge_count=dag.unique_edge_count,
    )
