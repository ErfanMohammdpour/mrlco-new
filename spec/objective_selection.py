#!/usr/bin/env python3
"""②B-2 adapter: PER-GRAPH objective/constraints -> aggregate -> lexicographic choice.

Why per-graph (blocker fix):

    E[L] / E[L_ref]  !=  E[ L_g / L_ref,g ]              # ratio-of-means is biased
    mean(E_g) <= mean(B_E)  does NOT imply  c_E,g <= 0   for every g

A checkpoint can look "energy feasible" on averages while individual episodes
violate the per-episode budget, which is exactly the constraint we claim to
enforce.  So every quantity is built per graph first and only then aggregated:

    J_val            = mean_g J_g
    soft_tard_norm   = mean_g soft_tard_norm_g
    c_H              = sum_g hard_miss_g / sum_g N_hard,g      (0 required)
    c_F              = sum_g firm_miss_g / sum_g N_firm,g      (<= eps_F)
    c_E_max          = max_g c_E,g            (per-episode hard constraint)
    energy_viol_rate = mean_g 1[c_E,g > 0]    (0 required)

Task counts are accumulated from the plans, never reconstructed from the number
of graphs.  A ratio-of-means shortcut is intentionally absent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from env.mec_offloaing_envs.scheduler.objective import (
    CheckpointCandidate,
    ObjectiveSpec,
    PlanObjective,
    choose_checkpoint,
    evaluate_plan_objective,
)

OBJECTIVE_MODE_OFF = "off"
OBJECTIVE_MODE_LOG_ONLY = "log_only"
OBJECTIVE_MODE_LEXICOGRAPHIC = "lexicographic"
OBJECTIVE_MODES = (OBJECTIVE_MODE_OFF, OBJECTIVE_MODE_LOG_ONLY, OBJECTIVE_MODE_LEXICOGRAPHIC)

SELECTION_METRIC_NAME = "lexicographic_feasible_then_J"


@dataclass(frozen=True)
class AggregatedObjective:
    """Validation-level objective after per-graph evaluation."""

    n_graphs: int
    J: float
    latency_norm_mean: float
    soft_tardiness_norm_mean: float
    energy_violation_rate: float
    c_E_max: float
    c_E_raw_max: float
    hard_miss_total: int
    hard_task_total: int
    c_H: float
    firm_miss_total: int
    firm_task_total: int
    c_F: float
    hard_miss_epsilon: float
    firm_miss_epsilon: float
    per_graph: tuple[PlanObjective, ...] = field(default=())

    @property
    def hard_ok(self) -> bool:
        return self.hard_miss_total <= 0 or self.c_H <= self.hard_miss_epsilon + 1e-12

    @property
    def energy_ok(self) -> bool:
        """Per-episode hard constraint: not a single graph may exceed the budget."""
        return self.energy_violation_rate <= 0.0 and self.c_E_max <= 0.0

    @property
    def firm_ok(self) -> bool:
        return self.c_F <= self.firm_miss_epsilon + 1e-12

    @property
    def feasible(self) -> bool:
        return self.hard_ok and self.energy_ok and self.firm_ok

    @property
    def total_violation(self) -> float:
        return float(
            max(0.0, self.c_H - self.hard_miss_epsilon)
            + self.energy_violation_rate
            + max(0.0, self.c_F - self.firm_miss_epsilon)
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "J": self.J,
            "latency_norm_mean": self.latency_norm_mean,
            "soft_tardiness_norm_mean": self.soft_tardiness_norm_mean,
            "energy_violation_rate": self.energy_violation_rate,
            "c_E_max": self.c_E_max,
            "c_E_raw_max": self.c_E_raw_max,
            "c_H": self.c_H,
            "c_F": self.c_F,
            "hard_miss_total": float(self.hard_miss_total),
            "hard_task_total": float(self.hard_task_total),
            "firm_miss_total": float(self.firm_miss_total),
            "firm_task_total": float(self.firm_task_total),
            "feasible": float(self.feasible),
            "total_violation": self.total_violation,
            "n_graphs": float(self.n_graphs),
        }

    def selection_key(self) -> tuple:
        if self.feasible:
            return (0, self.J, self.total_violation)
        return (1, self.total_violation, self.J)


def aggregate_plan_objectives(
    objectives: Sequence[PlanObjective], spec: ObjectiveSpec
) -> AggregatedObjective:
    """Aggregate per-graph objectives exactly as specified."""
    if not objectives:
        raise ValueError("no per-graph objectives to aggregate")
    n = len(objectives)
    hard_miss = sum(o.hard_miss_count for o in objectives)
    hard_tasks = sum(o.hard_task_count for o in objectives)
    firm_miss = sum(o.firm_miss_count for o in objectives)
    firm_tasks = sum(o.firm_task_count for o in objectives)
    viol = sum(1 for o in objectives if o.c_E > 0.0) / float(n)
    return AggregatedObjective(
        n_graphs=n,
        J=sum(o.J for o in objectives) / n,
        latency_norm_mean=sum(o.latency_norm for o in objectives) / n,
        soft_tardiness_norm_mean=sum(o.soft_tardiness_norm for o in objectives) / n,
        energy_violation_rate=viol,
        c_E_max=max(o.c_E for o in objectives),
        c_E_raw_max=max(getattr(o, "c_E_raw", o.c_E) for o in objectives),
        hard_miss_total=int(hard_miss),
        hard_task_total=int(hard_tasks),
        c_H=(hard_miss / hard_tasks) if hard_tasks else 0.0,
        firm_miss_total=int(firm_miss),
        firm_task_total=int(firm_tasks),
        c_F=(firm_miss / firm_tasks) if firm_tasks else 0.0,
        hard_miss_epsilon=float(spec.hard_miss_epsilon),
        firm_miss_epsilon=float(spec.firm_miss_epsilon),
        per_graph=tuple(objectives),
    )


def objective_from_plans(
    plans: Sequence[tuple[Any, Any]], spec: ObjectiveSpec
) -> AggregatedObjective:
    """`plans` = sequence of (ScheduleResult, ReferenceRanges), one per graph.

    The only supported entry point: per-graph evaluation first, aggregation second.
    """
    objectives = [evaluate_plan_objective(result, refs, spec) for result, refs in plans]
    return aggregate_plan_objectives(objectives, spec)


def objective_log_kvs(obj: AggregatedObjective, prefix: str = "objective") -> dict[str, float]:
    return {f"{prefix}/{k}": v for k, v in obj.as_dict().items()}


def selection_decision(
    history: Sequence[tuple[str, AggregatedObjective]],
) -> dict[str, Any]:
    """Lexicographic decision over validation history (feasible first, then min J)."""
    proxies = [
        CheckpointCandidate(
            label=label,
            objective=PlanObjective(
                latency_s=obj.J,
                latency_ref_s=1.0,
                latency_norm=obj.J,
                soft_tardiness_s=0.0,
                soft_tardiness_norm=obj.soft_tardiness_norm_mean,
                beta_soft=1.0,
                J=obj.J,
                energy_system_j=0.0,
                energy_budget_j=1.0,
                c_E=0.0 if obj.energy_ok else max(obj.c_E_max, 1e-9),
                hard_miss_count=obj.hard_miss_total,
                hard_task_count=max(1, obj.hard_task_total),
                c_H=obj.c_H,
                firm_miss_count=obj.firm_miss_total,
                firm_task_count=max(1, obj.firm_task_total),
                c_F=obj.c_F,
                hard_miss_epsilon=obj.hard_miss_epsilon,
                firm_miss_epsilon=obj.firm_miss_epsilon,
            ),
        )
        for label, obj in history
    ]
    choice = choose_checkpoint(proxies)
    return {
        "metric": SELECTION_METRIC_NAME,
        "winner": choice.winner,
        "feasible": choice.feasible,
        "ranked": choice.ranked,
        "best_infeasible": choice.best_infeasible,
        "note": choice.note,
    }
