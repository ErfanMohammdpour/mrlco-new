#!/usr/bin/env python3
"""The trainer's constraint channels, computed for a v2 schedule by the FROZEN v1 evaluator.

The v2 environment used to emit only miss COUNTS. The training loop, however, observes
`violation/<NAME>` keys produced by `spec.automotive_training.automotive_constraints`; with no
such keys `constraint_violations_from_telemetry` returned `{}` and the dual never moved — a
budget could not influence training at all.

Rather than reimplementing the formulas (which would drift), this module hands the frozen
`evaluate_constraints` the v2 foreground task availability as a plain
`{task_id: seconds}` mapping, which is an accepted input form of that function. The channels,
budgets and `n_violating_tasks` counts are therefore produced by the SAME code the v1 trainer
uses.
"""

from __future__ import annotations

from typing import Any, Mapping

from spec.automotive_training.automotive_constraints import (
    CONSTRAINT_NAMES, evaluate_constraints,
)

from spec.automotive_training.v2.shared_scheduler import V2ScheduleResult


class V2ConstraintChannelError(RuntimeError):
    """Raised when the v1 channels cannot be evaluated for a v2 schedule."""


class _GraphView:
    """A minimal (tasks, D_G_s) view the frozen evaluator accepts.

    The v2 world executes the MC-surviving sub-DAG, so only those tasks are scheduled and only
    those can report an availability. The frozen evaluator requires a deadline for every task
    it is shown, and a task with no declared subdeadline has no firm requirement to violate, so
    such tasks are excluded from the FIRM channels and reported as `unjudged_tasks` instead of
    being silently assigned a fabricated deadline.
    """

    def __init__(self, tasks, d_g_s):
        self.tasks = list(tasks)
        self.D_G_s = float(d_g_s)


def v1_channels(graph, result: V2ScheduleResult, *, foreground_dag_id: str | None = None,
                foreground_dag=None, d_g_s: float | None = None) -> dict:
    """{channel: {violation, value, budget, n_violating_tasks}} via the frozen evaluator.

    Only FOREGROUND tasks enter the availability map: the episode (and therefore every
    deadline/tardiness requirement) is the foreground DAG, and including background tasks
    would charge the episode for other owners' deadlines. `foreground_dag` is the world's
    `V2DAGSpec` for the foreground, which already carries exactly the surviving tasks.
    """
    if foreground_dag is not None:
        if foreground_dag_id is None:
            foreground_dag_id = str(foreground_dag.dag_id)
        if d_g_s is None:
            d_g_s = float(getattr(graph, "D_G_s", 0.0))
        tasks = [t for t in foreground_dag.tasks if t.deadline_s is not None]
        unjudged = [int(t.task_id) for t in foreground_dag.tasks if t.deadline_s is None]
    else:
        tasks = []
        unjudged = []
    if foreground_dag_id is None:
        foreground_dag_id = max(result.completion_by_dag,
                                key=lambda k: result.completion_by_dag[k]) \
            if result.completion_by_dag else None
    availability: dict = {}
    for (dag_id, task_id), timing in result.timings.items():
        if foreground_dag_id is not None and dag_id != foreground_dag_id:
            continue
        availability[int(task_id)] = float(timing.finish_s)
    if not availability:
        raise V2ConstraintChannelError(
            "no foreground timings to evaluate (foreground_dag_id=%r)" % (foreground_dag_id,))
    if tasks:
        availability = {tid: value for tid, value in availability.items()
                        if tid in {int(t.task_id) for t in tasks}}
        view = _GraphView(tasks, d_g_s if d_g_s is not None
                          else float(getattr(graph, "D_G_s", 0.0)))
    else:
        view = graph
    try:
        out = evaluate_constraints(view, availability)
    except Exception as exc:                    # pragma: no cover - contract guard
        raise V2ConstraintChannelError(
            "the frozen v1 constraint evaluator rejected the v2 availability map "
            "(%d tasks): %s" % (len(availability), exc)) from exc
    out["_unjudged_tasks"] = unjudged
    return out


def channel_telemetry(graph, result: V2ScheduleResult, *,
                      foreground_dag_id: str | None = None, foreground_dag=None,
                      d_g_s: float | None = None,
                      lambdas: Mapping[str, float] | None = None,
                      penalty: float = 0.0, l_scale: float = 1.0,
                      task_criticality: Mapping | None = None) -> dict:
    """The exact top-level keys the frozen trainer observer reads, plus useful extras."""
    channels = v1_channels(graph, result, foreground_dag_id=foreground_dag_id,
                           foreground_dag=foreground_dag, d_g_s=d_g_s)
    lambdas = dict(lambdas or {})
    out: dict[str, Any] = {}
    n_violating = {}
    for name in CONSTRAINT_NAMES:
        channel = channels.get(name)
        if channel is None:
            continue
        out["violation/%s" % name] = float(channel["violation"])
        out["lambda/%s" % name] = float(lambdas.get(name, 0.0))
        out["n_violating/%s" % name] = int(channel["n_violating_tasks"])
        n_violating[name] = int(channel["n_violating_tasks"])
    out["constraint_penalty"] = float(penalty)
    out["penalized_objective"] = -float(result.makespan_s) / max(float(l_scale), 1e-12) \
        - float(penalty) / max(float(l_scale), 1e-12)
    out["firm_miss_count"] = int(
        n_violating.get("C_HI_TASK_TARDINESS", 0) + n_violating.get("C_MED_TASK_TARDINESS", 0))
    out["task_count"] = int(sum(1 for _ in result.timings))
    unjudged = channels.get("_unjudged_tasks") or []
    out["unjudged_tasks_no_subdeadline"] = int(len(unjudged))
    if task_criticality:
        counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
        for value in task_criticality.values():
            key = str(value).strip().upper()
            if key in counts:
                counts[key] += 1
        out["n_tasks_high"] = counts["HIGH"]
        out["n_tasks_medium"] = counts["MEDIUM"]
        out["n_tasks_low"] = counts["LOW"]
    return out


__all__ = ["CONSTRAINT_NAMES", "V2ConstraintChannelError", "channel_telemetry", "v1_channels"]
