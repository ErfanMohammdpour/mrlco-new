"""Greedy baseline: plan search only; every candidate evaluated by schedule()."""

from __future__ import annotations

from typing import Any

from .adapter import schedule_via_adapter
from .model import ScheduleResult
from .resources import ResourceConfig

# Latency baseline metric. Energy-aware J_report greedy is a later spec choice.
GREEDY_METRIC = "makespan_seconds"
FILL_UNASSIGNED = 0  # all_UE completion policy for unevaluated suffix
ACTION_TIE_BREAK = (0, 1, 2)  # UE → MEC → HELPER (publication ternary)
BINARY_ACTIONS = (0, 1)  # UE → MEC; no V2V


def resolve_greedy_actions(actions=None):
    if actions is None:
        return ACTION_TIE_BREAK
    out = tuple(int(a) for a in actions)
    allowed = set(ACTION_TIE_BREAK)
    if not out:
        raise ValueError("greedy actions empty")
    bad = [a for a in out if a not in allowed]
    if bad:
        raise ValueError("greedy actions %s not in %s" % (bad, ACTION_TIE_BREAK))
    return out


def greedy_plan(
    task_graph: Any,
    resources: ResourceConfig,
    actions=None,
) -> tuple[list[tuple[int, int]], ScheduleResult]:
    """Build a greedy plan in decoder order.

    At each position, try UE/MEC/HELPER (or a restricted action set), fill
    remaining tasks with all_UE, score complete plans with `schedule()`,
    pick lowest makespan. Tie-break: lower action id.
    """
    action_set = resolve_greedy_actions(actions)
    order = [int(tid) for tid in task_graph.prioritize_sequence]
    n = len(order)
    chosen: list[int] = []
    for k in range(n):
        best_metric: float | None = None
        best_action: int | None = None
        for action in action_set:
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


def greedy_from_mec_plan(
    task_graph: Any,
    resources: ResourceConfig,
    max_passes: int = 2,
    actions=None,
    metric_fn=None,
) -> tuple[list[tuple[int, int]], ScheduleResult]:
    """Local search from all-MEC. Flip a token only if metric strictly drops.

    Default metric is makespan (Phase 4 latency track). Pass metric_fn(result)
    for J_λ. Not the publication `greedy_plan` (that starts empty with all_UE fill).
    """
    action_set = resolve_greedy_actions(actions)
    order = [int(tid) for tid in task_graph.prioritize_sequence]
    n = len(order)
    if n < 1:
        raise ValueError("empty task graph")
    max_passes = int(max_passes)
    if max_passes < 1:
        raise ValueError("max_passes must be positive")
    chosen = [1] * n

    def score(actions: list[int]) -> tuple[float, ScheduleResult]:
        plan = list(zip(order, actions))
        result, _, _ = schedule_via_adapter(task_graph, plan, resources)
        if metric_fn is None:
            return float(result.makespan_seconds), result
        return float(metric_fn(result)), result

    best_m, _ = score(chosen)
    for _ in range(max_passes):
        changed = False
        for k in range(n):
            local_a = chosen[k]
            local_m = best_m
            for action in action_set:
                if action == chosen[k]:
                    continue
                trial = list(chosen)
                trial[k] = action
                m, _ = score(trial)
                if m + 1e-12 < local_m:
                    local_m = m
                    local_a = action
            if local_a != chosen[k]:
                chosen[k] = local_a
                best_m = local_m
                changed = True
        if not changed:
            break
    plan = list(zip(order, chosen))
    result, _, _ = schedule_via_adapter(task_graph, plan, resources)
    return plan, result
