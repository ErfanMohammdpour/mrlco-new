"""Witness plans: a REAL schedulable plan that meets every hard deadline.

A deadline regime is only meaningful if the instance stays schedulable. Feasibility
of the admissible relaxation in `static_bounds.py` is NOT feasibility: that
relaxation assumes zero contention and free transfers, while the engine schedules
the whole DAG on six single-capacity non-preemptive resources
(`calendar.py`), and it measures a miss on `all_consumers_ready` -- the sink
return hop included. So every trainable graph must carry a witness plan that was
actually REPLAYED through `schedule_via_adapter` and came back with
`hard_miss_count == 0`.

Search: `greedy_from_mec_plan` with a miss-first lexicographic score
(`hard_miss_count * BIG + makespan`), then a deterministic best-improvement
H1/H2 local search, then verification by replay. The greedy seed alone succeeds
on most graphs, so the expensive pass only runs on failures.

Nothing here mutates the graph: deadlines must already be stamped
(`deadline_regime.stamp_task_graph`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .adapter import schedule_via_adapter
from .greedy import greedy_from_mec_plan
from .model import ScheduleResult
from .resources import ResourceConfig

ACTION_SET = (0, 1, 2)
DEFAULT_BIG_M = 1e6


def miss_first_score(result: ScheduleResult, big_m: float = DEFAULT_BIG_M) -> float:
    """Lexicographic: hard misses dominate, then makespan."""
    return float(result.hard_miss_count) * float(big_m) + float(result.makespan_seconds)


@dataclass(frozen=True)
class PlanResult:
    """A concrete plan and its real replay, with no deadline requirement."""

    actions: tuple[int, ...]
    makespan_s: float
    method: str
    evaluations: int
    ready_s: tuple[float, ...]          # all_consumers_ready per decoder position
    task_ids: tuple[int, ...]
    table: Mapping[int, Mapping[str, float]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "actions": [int(a) for a in self.actions],
            "makespan_s": float(self.makespan_s),
            "method": self.method,
            "evaluations": int(self.evaluations),
            "ready_s": [float(v) for v in self.ready_s],
        }


@dataclass(frozen=True)
class WitnessResult:
    found: bool
    actions: tuple[int, ...]
    makespan_s: float
    hard_miss_count: int
    method: str
    evaluations: int
    per_task: Mapping[int, Mapping[str, float]] | None = None   # replay table
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        blob: dict[str, Any] = {
            "found": bool(self.found),
            "actions": [int(a) for a in self.actions],
            "makespan_s": float(self.makespan_s),
            "hard_miss_count": int(self.hard_miss_count),
            "method": self.method,
            "evaluations": int(self.evaluations),
        }
        if self.per_task is not None:
            blob["per_task"] = {
                str(tid): {k: float(v) for k, v in row.items()}
                for tid, row in sorted(self.per_task.items())
            }
        if self.reason:
            blob["reason"] = self.reason
        return blob

def plan_for(task_graph: Any, actions: Sequence[int]) -> list[tuple[int, int]]:
    order = [int(t) for t in task_graph.prioritize_sequence]
    if len(order) != len(actions):
        raise ValueError(
            "actions length %d != task count %d" % (len(actions), len(order))
        )
    return list(zip(order, [int(a) for a in actions]))


def replay_plan(
    task_graph: Any,
    actions: Sequence[int],
    resources: ResourceConfig,
) -> tuple[ScheduleResult, dict[int, dict[str, float]]]:
    """Schedule a concrete plan and return the per-task deadline table."""
    result, _deltas, _energy = schedule_via_adapter(
        task_graph, plan_for(task_graph, actions), resources
    )
    table: dict[int, dict[str, float]] = {}
    for tid, record in result.tasks.items():
        deadline = record.deadline_s
        ready = float(record.all_consumers_ready)
        table[int(tid)] = {
            "finish_s": float(record.finish),
            "all_consumers_ready_s": ready,
            "deadline_s": float(deadline) if deadline is not None else float("nan"),
            "slack_s": float(deadline) - ready if deadline is not None else float("nan"),
            "missed": 1.0 if record.tardiness_s > 0.0 else 0.0,
        }
    return result, table


def find_fastest_plan(
    task_graph: Any,
    resources: ResourceConfig,
    *,
    max_passes: int = 4,
    max_pair_rounds: int = 2,
    max_pairs: int = 40,
) -> PlanResult:
    """Deterministic makespan search: greedy H1 from all-MEC, then best H1/H2.

    Used to ANCHOR the deadline generation: the deadline regime is built from a
    schedule the real engine accepts, never from the (contention-free) relaxation.
    Requires no deadlines on the graph.
    """
    if not hasattr(task_graph, "prioritize_sequence"):
        raise ValueError("task_graph must expose prioritize_sequence")
    order = [int(t) for t in task_graph.prioritize_sequence]
    evaluations = 0

    plan, result = greedy_from_mec_plan(
        task_graph, resources, max_passes=max_passes, actions=ACTION_SET, metric_fn=None
    )
    evaluations += max_passes * len(order) * (len(ACTION_SET) - 1) + 1
    actions = [int(a) for _tid, a in plan]
    best = float(result.makespan_seconds)
    method = "greedy_from_mec_makespan"

    def evaluate(candidate: Sequence[int]) -> tuple[float, ScheduleResult]:
        nonlocal evaluations
        evaluations += 1
        res, _d, _e = schedule_via_adapter(
            task_graph, plan_for(task_graph, candidate), resources
        )
        return float(res.makespan_seconds), res

    for _round in range(int(max_pair_rounds)):
        improved = False
        for k in range(len(actions)):
            for action in ACTION_SET:
                if action == actions[k]:
                    continue
                trial = list(actions)
                trial[k] = action
                score, res = evaluate(trial)
                if score + 1e-12 < best:
                    best, actions, improved = score, trial, True

    result, table = replay_plan(task_graph, actions, resources)
    ready = tuple(float(table[int(t)]["all_consumers_ready_s"]) for t in order)
    return PlanResult(
        actions=tuple(actions),
        makespan_s=float(result.makespan_seconds),
        method=method,
        evaluations=evaluations,
        ready_s=ready,
        task_ids=tuple(order),
        table=table,
    )


def find_witness(
    task_graph: Any,
    resources: ResourceConfig,
    *,
    big_m: float = DEFAULT_BIG_M,
    max_passes: int = 4,
    max_pair_rounds: int = 3,
    max_pairs: int = 60,
    relaxed_lb_s: float | None = None,
    seed_actions: Sequence[int] | None = None,
) -> WitnessResult:
    """Deterministic witness search followed by a real replay.

    Returns `found=False` with a reason when no plan meets all hard deadlines;
    callers must then exclude the graph from a trainable regime.

    `relaxed_lb_s` (the admissible graph lower bound, when known) is checked as an
    invariant: a witness cannot be faster than its own relaxation, so a violation
    means one of the two models is wrong.
    """
    if not hasattr(task_graph, "prioritize_sequence"):
        raise ValueError("task_graph must expose prioritize_sequence")
    tasks = getattr(task_graph, "task_list", None) or []
    if not any(str(getattr(task, "deadline_type", "none")) == "hard" for task in tasks):
        raise ValueError(
            "no hard deadlines are stamped on this graph; refusing to certify a "
            "witness for a regime that would make it trivially true"
        )

    evaluations = 0
    metric = lambda result: miss_first_score(result, big_m)  # noqa: E731
    order = [int(t) for t in task_graph.prioritize_sequence]

    # the anchored plan the deadlines were derived from is checked FIRST: if it
    # still meets every deadline there is nothing to search for
    if seed_actions is not None:
        seed = [int(a) for a in seed_actions]
        if len(seed) != len(order):
            raise ValueError(
                "seed_actions length %d != task count %d" % (len(seed), len(order))
            )
        result, table = replay_plan(task_graph, seed, resources)
        evaluations += 1
        if int(result.hard_miss_count) == 0:
            return WitnessResult(
                found=True,
                actions=tuple(seed),
                makespan_s=float(result.makespan_seconds),
                hard_miss_count=0,
                method="anchored_seed",
                evaluations=evaluations,
                per_task=table,
            )

    plan, result = greedy_from_mec_plan(
        task_graph, resources, max_passes=max_passes, actions=ACTION_SET, metric_fn=metric
    )
    evaluations += max_passes * len(plan) * (len(ACTION_SET) - 1) + 1
    actions = [action for _tid, action in plan]
    best_score = metric(result)
    method = "greedy_from_mec"

    if result.hard_miss_count > 0:
        # deterministic best-improvement local search: H1 over every token, then
        # best H2 pair, with a strict improvement rule and a stable tie-break
        def evaluate(candidate: Sequence[int]) -> tuple[float, ScheduleResult]:
            nonlocal evaluations
            evaluations += 1
            res, _d, _e = schedule_via_adapter(
                task_graph, plan_for(task_graph, candidate), resources
            )
            return metric(res), res

        for _round in range(int(max_pair_rounds)):
            improved = False
            for k in range(len(actions)):
                current = actions[k]
                for action in ACTION_SET:
                    if action == current:
                        continue
                    trial = list(actions)
                    trial[k] = action
                    score, res = evaluate(trial)
                    if score + 1e-12 < best_score:
                        best_score = score
                        actions = trial
                        result = res
                        improved = True
            pairs = [
                (i, j)
                for i in range(len(actions))
                for j in range(i + 1, len(actions))
            ][: int(max_pairs)]
            for i, j in pairs:
                base_i, base_j = actions[i], actions[j]
                for a in ACTION_SET:
                    for b in ACTION_SET:
                        if (a, b) == (base_i, base_j):
                            continue
                        trial = list(actions)
                        trial[i], trial[j] = a, b
                        score, res = evaluate(trial)
                        if score + 1e-12 < best_score:
                            best_score = score
                            actions = trial
                            result = res
                            improved = True
            if not improved:
                break
        method = "greedy_from_mec+h1h2"

    # final verification: replay the chosen plan and read the engine's own verdict
    result, table = replay_plan(task_graph, actions, resources)
    found = int(result.hard_miss_count) == 0
    if not all(math.isfinite(row["all_consumers_ready_s"]) for row in table.values()):
        raise ValueError("witness replay produced a non-finite timestamp")
    if relaxed_lb_s is not None and found:
        if float(result.makespan_seconds) + 1e-9 < float(relaxed_lb_s):
            raise ValueError(
                "witness makespan %.9f is below the admissible lower bound %.9f: "
                "the bound or the scheduler is wrong"
                % (result.makespan_seconds, relaxed_lb_s)
            )

    reason = ""
    if not found:
        misses = [
            tid for tid, row in table.items() if row["missed"] > 0.0
        ]
        reason = "no plan met every hard deadline (%d tasks missed, e.g. %s)" % (
            len(misses),
            sorted(misses)[:5],
        )

    return WitnessResult(
        found=found,
        actions=tuple(int(a) for a in actions),
        makespan_s=float(result.makespan_seconds),
        hard_miss_count=int(result.hard_miss_count),
        method=method,
        evaluations=evaluations,
        per_task=table,
        reason=reason,
    )


def mixed_action_witness(actions: Sequence[int]) -> bool:
    """True when the witness does not use a single action everywhere."""
    return len(set(int(a) for a in actions)) > 1
