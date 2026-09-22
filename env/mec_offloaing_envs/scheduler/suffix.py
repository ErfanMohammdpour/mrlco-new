"""③ suffix layer: optimistic DAG lookahead, constructive fastest-feasible
suffix, and the shaped potential. Pure scheduler — no TF, no PPO, no Lagrangian.

Two SEPARATE mechanisms, never mixed:

    PROOF layer (mask authority)     `dag_lower_bound_masks()`
        A relaxation over every remaining completion: free placement of
        undecided tasks, zero resource contention, co-location assumed for
        dependency transfers.  If even this relaxation misses a hard deadline,
        NO completion exists and the action may be masked.

    CONSTRUCTIVE layer (potential only)  `construct_fastest_feasible_suffix()`
        One deterministic greedy suffix.  Used to build the potential Ĵ(s).
        If it fails, the action stays OPEN: heuristic failure is not proof.

Potential (only the objective; energy/firm stay in their own channels):

    Ĵ(s) = L̂(s)/L_ref + beta_soft * T̂_soft(s)
    r_t  = Ĵ(s_{t-1}) - gamma * Ĵ(s_t)          (see reward.py, discount=gamma)

with Ĵ(s_T) == J_actual at a terminal state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .engine import schedule
from .feasibility import (
    FeasibilityContext,
    ParentInput,
    finish_lower_bound,
    transfer_lower_bound,
)
from .model import CanonicalDAG, Location, ScheduleResult
from .resources import ResourceConfig

REASON_PROOF_MASK = "dag_lower_bound_infeasible"
REASON_HEURISTIC_FAILED = "heuristic_only_not_used"
REASON_SUFFIX_FOUND = "suffix_found"


@dataclass(frozen=True)
class TaskDeadline:
    deadline_s: float
    deadline_type: str = "hard"
    tardiness_weight: float = 1.0


@dataclass(frozen=True)
class ObjectiveContext:
    """Objective/budget context, decoupled from the energy model.

    Callers pass `latency_ref_s` and `energy_budget_j` explicitly so the suffix
    layer never reaches into the energy model or the panel budget machinery.
    """

    latency_ref_s: float
    beta_soft: float = 1.0
    energy_budget_j: float | None = None
    deadlines: Mapping[int, TaskDeadline] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if float(self.latency_ref_s) <= 0.0:
            raise ValueError("latency_ref_s must be positive")
        if self.energy_budget_j is not None and float(self.energy_budget_j) <= 0.0:
            raise ValueError("energy_budget_j must be positive when given")

    def deadline_of(self, task_id: int) -> TaskDeadline | None:
        return self.deadlines.get(int(task_id))

    def hard_deadlines(self) -> dict[int, float]:
        return {
            int(tid): float(d.deadline_s)
            for tid, d in self.deadlines.items()
            if d.deadline_type == "hard"
        }


@dataclass(frozen=True)
class SuffixContext:
    """Everything the suffix construction needs about the current state."""

    dag: CanonicalDAG
    resources: ResourceConfig
    order: Sequence[int]              # topological decoder/HEFT order
    objective: ObjectiveContext
    decisions: Mapping[int, Location] = field(default_factory=dict)
    prefix_finish: Mapping[int, float] = field(default_factory=dict)
    index: int = 0                    # index of the task being decided
    cycles_per_bit: float | None = None
    soft_tardiness_norm_so_far: float = 0.0

    @property
    def current_task(self) -> int:
        return int(self.order[self.index])

    @property
    def remaining(self) -> list[int]:
        return [int(t) for t in self.order[self.index :]]

    def workload(self, task_id: int) -> int:
        return int(self.dag.tasks[int(task_id)].compute_workload_bytes)

    def is_sink(self, task_id: int) -> bool:
        tid = int(task_id)
        return not any(e.src_task_id == tid for e in self.dag.edges)

    def successor_bytes(self, task_id: int) -> int:
        tid = int(task_id)
        return min(
            (int(e.edge_output_bytes) for e in self.dag.edges if e.src_task_id == tid),
            default=0,
        )

    def parent_inputs(self, task_id: int, placements: Mapping[int, Location]) -> tuple[ParentInput, ...]:
        tid = int(task_id)
        out = []
        for e in self.dag.edges:
            if e.dst_task_id != tid:
                continue
            src = int(e.src_task_id)
            out.append(
                ParentInput(
                    source_location=placements.get(src, Location.UE),
                    source_ready_s=float(self.prefix_finish.get(src, 0.0)),
                    bytes=int(e.edge_output_bytes),
                )
            )
        return tuple(out)


@dataclass(frozen=True)
class SuffixResult:
    found: bool
    reason: str
    placements: dict[int, Location] = field(default_factory=dict)
    plan: list[tuple[int, int]] = field(default_factory=list)
    result: ScheduleResult | None = None
    estimated_latency_s: float | None = None
    estimated_soft_tardiness_norm: float | None = None
    estimated_J: float | None = None
    failed_at_task: int | None = None

    @property
    def feasible_suffix(self) -> bool:
        return bool(self.found)


# ---------------------------------------------------------------------------
# PROOF layer: optimistic DAG relaxation
# ---------------------------------------------------------------------------
def _cheapest_compute_seconds(ctx: SuffixContext, task_id: int) -> float:
    """min over tiers of C_i / f (the relaxation may place the task anywhere)."""
    workload = ctx.workload(task_id)
    best = float("inf")
    for action in (0, 1, 2):
        loc = Location.from_action(action)
        rate = ctx.resources.cpu_rate_for_task(loc, ctx.dag.tasks[int(task_id)])
        if rate > 0:
            best = min(best, workload / rate)
    return best


def dag_lower_bound_ready(
    ctx: SuffixContext, action: int
) -> dict[int, float]:
    """Admissible (optimistic) all-consumers-ready bound for every remaining task.

    Relaxation: the current task takes `action`; every later task may be placed
    on its cheapest/fastest tier; dependency transfers are free when the two
    tasks could be co-located (which the relaxation may always assume); zero
    resource contention.  Sinks still pay the cheapest possible return hop.
    """
    current = ctx.current_task
    lb_finish: dict[int, float] = {}
    # decided prefix: use known finishes
    for tid, fin in ctx.prefix_finish.items():
        if int(tid) in ctx.dag.tasks:
            lb_finish[int(tid)] = float(fin)

    parents = ctx.parent_inputs(current, {**dict(ctx.decisions), current: Location.from_action(action)})
    lb_finish[current] = finish_lower_bound(
        action,
        FeasibilityContext(
            parents=parents,
            deadline_s=None,
            is_sink=ctx.is_sink(current),
            return_bytes=ctx.successor_bytes(current) or int(
                ctx.dag.tasks[current].task_output_bytes
            ),
        ),
        ctx.resources,
        workload_bytes=ctx.workload(current),
        global_cycles_per_bit=ctx.cycles_per_bit,
    )

    for tid in ctx.remaining:
        tid = int(tid)
        if tid == current:
            continue
        deps = [int(e.src_task_id) for e in ctx.dag.edges if e.dst_task_id == tid]
        start = max((lb_finish.get(p, 0.0) for p in deps), default=0.0)
        lb_finish[tid] = start + _cheapest_compute_seconds(ctx, tid)

    ready: dict[int, float] = {}
    for tid in ctx.remaining:
        tid = int(tid)
        if ctx.is_sink(tid):
            out_bytes = int(ctx.dag.tasks[tid].task_output_bytes)
            cheapest = min(
                transfer_lower_bound(out_bytes, Location.from_action(a), Location.UE, ctx.resources)
                for a in (0, 1, 2)
            )
            ready[tid] = lb_finish[tid] + cheapest
        else:
            ready[tid] = lb_finish[tid]
    return ready


def dag_lower_bound_masks(ctx: SuffixContext, action: int) -> tuple[bool, str | None, int | None]:
    """True when the relaxation itself misses a hard deadline -> mask is sound."""
    hard = ctx.objective.hard_deadlines()
    if not hard:
        return False, None, None
    ready = dag_lower_bound_ready(ctx, action)
    for tid, deadline in hard.items():
        if tid not in ready:
            continue
        if ready[tid] > deadline:
            return True, REASON_PROOF_MASK, int(tid)
    return False, None, None


# ---------------------------------------------------------------------------
# CONSTRUCTIVE layer: fastest feasible suffix (potential only)
# ---------------------------------------------------------------------------
def construct_fastest_feasible_suffix(
    ctx: SuffixContext, action: int | None = None
) -> SuffixResult:
    """Deterministic greedy suffix: at each remaining task take the feasible
    placement with the smallest optimistic finish.

    `action` forces the CURRENT task's placement (action-conditioned lookahead,
    used by `evaluate_action`).  `action=None` leaves the current task to the
    heuristic as well, which is what makes Ĵ a function of the STATE alone
    (`state_potential`) — required for the telescoping identity to hold exactly.

    Failure is reported as `found=False` and MUST NOT be used to mask.
    """
    placements: dict[int, Location] = dict(ctx.decisions)
    current = ctx.current_task
    if action is not None:
        placements[current] = Location.from_action(action)
    order_actions: list[tuple[int, int]] = [
        (int(t), int(placements[int(t)].to_action()))
        for t in ctx.order
        if int(t) in placements
    ]
    finish = dict(ctx.prefix_finish)

    for tid in ctx.remaining:
        tid = int(tid)
        if tid == current and action is not None:
            compute = ctx.workload(tid) / max(
                ctx.resources.cpu_rate_for_task(placements[tid], ctx.dag.tasks[tid]), 1e-12
            )
            start = max(
                (float(finish.get(int(e.src_task_id), 0.0)) for e in ctx.dag.edges if e.dst_task_id == tid),
                default=0.0,
            )
            finish[tid] = start + compute
            continue

        parents = ctx.parent_inputs(tid, placements)
        deadline = ctx.objective.deadline_of(tid)
        candidates: list[tuple[float, int]] = []
        for a in (0, 1, 2):
            lb_finish = finish_lower_bound(
                a,
                FeasibilityContext(
                    parents=parents,
                    deadline_s=None,
                    is_sink=ctx.is_sink(tid),
                    return_bytes=int(ctx.dag.tasks[tid].task_output_bytes),
                ),
                ctx.resources,
                workload_bytes=ctx.workload(tid),
                global_cycles_per_bit=ctx.cycles_per_bit,
            )
            ready_est = lb_finish
            # Only hard/firm deadlines constrain feasibility. A SOFT deadline is
            # a cost measured by the objective, never a reason to reject a
            # placement (rejecting it would turn a penalty into a constraint).
            if (
                deadline is not None
                and deadline.deadline_type in ("hard", "firm")
                and ready_est > float(deadline.deadline_s)
            ):
                continue
            candidates.append((lb_finish, a))
        if not candidates:
            return SuffixResult(
                found=False,
                reason=REASON_HEURISTIC_FAILED,
                failed_at_task=tid,
                placements=placements,
            )
        best_finish, best_action = min(candidates)
        placements[tid] = Location.from_action(best_action)
        finish[tid] = best_finish
        order_actions.append((tid, best_action))

    order_actions.sort(key=lambda item: list(ctx.order).index(item[0]))
    plan = [(tid, a) for tid, a in order_actions]
    actions = [a for _tid, a in plan]
    try:
        result = schedule(ctx.dag, list(ctx.order), actions, ctx.resources)
    except Exception:
        return SuffixResult(
            found=False, reason=REASON_HEURISTIC_FAILED, placements=placements
        )

    J = suffix_potential(result, ctx.objective)
    return SuffixResult(
        found=True,
        reason=REASON_SUFFIX_FOUND,
        placements=placements,
        plan=plan,
        result=result,
        estimated_latency_s=float(result.makespan_seconds),
        estimated_soft_tardiness_norm=float(result.soft_tardiness_normalized),
        estimated_J=float(J),
    )


def state_potential(ctx: SuffixContext) -> SuffixResult:
    """Ĵ(s): heuristic completion from the state alone (no forced action).

    r_t = Ĵ(s_{t-1}) - gamma * Ĵ(s_t) telescopes exactly with gamma=1, and
    Ĵ(s_0) is identical for every candidate plan, so plan ordering is decided by
    the terminal objective only.
    """
    return construct_fastest_feasible_suffix(ctx, None)


def suffix_potential(result: ScheduleResult, objective: ObjectiveContext) -> float:
    """Ĵ(s) = L̂/L_ref + beta_soft * T̂_soft_norm (objective only)."""
    return float(
        float(result.makespan_seconds) / float(objective.latency_ref_s)
        + float(objective.beta_soft) * float(result.soft_tardiness_normalized)
    )


# ---------------------------------------------------------------------------
# Per-action evaluation + mask
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ActionSuffixEvaluation:
    action: int
    location: str
    mask: bool                       # True = allowed
    mask_reason: str
    lb_ready_s: float
    suffix: SuffixResult
    potential: float | None


def evaluate_action(ctx: SuffixContext, action: int) -> ActionSuffixEvaluation:
    """LB (sound) + constructive suffix (potential). Mask needs proof only."""
    parents = ctx.parent_inputs(ctx.current_task, dict(ctx.decisions))
    lb = finish_lower_bound(
        action,
        FeasibilityContext(
            parents=parents,
            deadline_s=(
                ctx.objective.deadline_of(ctx.current_task).deadline_s
                if ctx.objective.deadline_of(ctx.current_task)
                else None
            ),
            is_sink=ctx.is_sink(ctx.current_task),
            return_bytes=int(ctx.dag.tasks[ctx.current_task].task_output_bytes),
        ),
        ctx.resources,
        workload_bytes=ctx.workload(ctx.current_task),
        global_cycles_per_bit=ctx.cycles_per_bit,
    )
    task_deadline = ctx.objective.deadline_of(ctx.current_task)
    lb_masks = task_deadline is not None and lb > float(task_deadline.deadline_s)
    proof_masks, proof_reason, _failed = (
        (False, None, None) if lb_masks else dag_lower_bound_masks(ctx, action)
    )
    suffix = construct_fastest_feasible_suffix(ctx, action)
    masked = bool(lb_masks or proof_masks)
    reason = (
        "lb_exceeds_deadline"
        if lb_masks
        else (proof_reason or ("not_proven_infeasible" if suffix.found else REASON_HEURISTIC_FAILED))
    )
    return ActionSuffixEvaluation(
        action=int(action),
        location=Location.from_action(action).value,
        mask=not masked,
        mask_reason=reason,
        lb_ready_s=float(lb),
        suffix=suffix,
        potential=suffix.estimated_J if suffix.found else None,
    )


def evaluate_all_actions(ctx: SuffixContext) -> list[ActionSuffixEvaluation]:
    return [evaluate_action(ctx, a) for a in (0, 1, 2)]


def feasible_action_mask(ctx: SuffixContext) -> list[bool]:
    """Mask from PROOF sources only (LB + DAG relaxation). Never from heuristics."""
    return [ev.mask for ev in evaluate_all_actions(ctx)]


def best_by_potential(
    evaluations: Sequence[ActionSuffixEvaluation],
) -> ActionSuffixEvaluation | None:
    usable = [e for e in evaluations if e.mask]
    if not usable:
        return None
    with_potential = [e for e in usable if e.potential is not None]
    pool = with_potential or usable
    return min(pool, key=lambda e: (e.potential if e.potential is not None else float("inf")))
