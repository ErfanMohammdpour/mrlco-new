"""Greedy baseline: plan search only; every candidate evaluated by schedule()."""

from __future__ import annotations

from typing import Any

from .adapter import schedule_via_adapter
from .model import ScheduleResult
from .resources import ResourceConfig

# Latency baseline metric. Energy-aware J_report greedy is a later spec choice.
GREEDY_METRIC = "makespan_seconds"
FILL_UNASSIGNED = 0  # all_UE completion policy for unevaluated suffix
ACTION_TIE_BREAK = (0, 1, 2)  # UE → MEC → HELPER


def greedy_plan(
    task_graph: Any,
    resources: ResourceConfig,
) -> tuple[list[tuple[int, int]], ScheduleResult]:
    """Build a greedy plan in decoder order.

    At each position, try UE/MEC/HELPER, fill remaining tasks with all_UE,
    score complete plans with `schedule()`, pick lowest makespan.
    Tie-break: UE then MEC then HELPER.
    """
    order = [int(tid) for tid in task_graph.prioritize_sequence]
    n = len(order)
    chosen: list[int] = []
    for k in range(n):
        best_metric: float | None = None
        best_action: int | None = None
        for action in ACTION_TIE_BREAK:
            fill = chosen + [action] + [FILL_UNASSIGNED] * (n - k - 1)
            plan = list(zip(order, fill))
            result, _, _ = schedule_via_adapter(task_graph, plan, resources)
            metric = result.makespan_seconds
            if (
                best_metric is None
                or metric + 1e-12 < best_metric
                or (abs(metric - best_metric) <= 1e-12 and action < int(best_action))
            ):
                best_metric = metric
                best_action = action
        assert best_action is not None
        chosen.append(best_action)

    plan = list(zip(order, chosen))
    result, _, _ = schedule_via_adapter(task_graph, plan, resources)
    return plan, result
