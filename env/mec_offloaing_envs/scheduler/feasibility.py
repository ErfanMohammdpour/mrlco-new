"""②B-3 / ③: hard-deadline feasibility — the PROOF layer only.

Soundness rule (non-negotiable):

    optimistic lower bound > deadline   ->  SAFE TO MASK   (proof)
    constructive heuristic failed       ->  NO MASK        (not evidence)

A mask is only allowed when the bound proves that NO schedule can meet the
deadline, even with zero resource contention and optimal placement of everything
that comes later.  Anything weaker may remove a feasible action: that is a shield
bug, not a conservative choice.

Bound (per action `a` for the current task `i`), exactly as approved:

    ParentInput: source_location, source_ready_s (= parent COMPUTE finish), bytes
    A_{p->i}(a) = t_p_source + transfer_lb(loc_p, loc_a, bytes_p)
    S_i^LB(a)   = max_p A_{p->i}(a)                 # parents transfer in PARALLEL
    F_i^LB(a)   = S_i^LB(a) + C_i / f_a
    R_i^LB(a)   = F_i^LB(a)                                   if i is not a sink
                = F_i^LB(a) + transfer_lb(loc_a, UE, out)     if i is a sink

Why this shape:

* `max` over parents, not a sum: with zero contention the inbound transfers do
  not serialize, so summing them would over-estimate the start time and could
  mask a feasible action.
* `source_ready_s` is the parent's COMPUTE finish (the moment the bytes exist at
  the parent's location), NOT its all-consumers-ready time.  Waiting for every
  other consumer of the parent is not required to start this child.
* For non-sinks the delivery term is ZERO: successors may be co-located (all on
  MEC pays no downlink), so any positive constant would be an over-estimate.
  A loose bound only weakens the mask; a tight-but-wrong bound breaks the shield.
* Only sink returns must reach the UE, so only there is the hop unavoidable.

③ adds DAG lookahead on top (tightening only, never loosening) plus the
constructive fastest-feasible suffix, which feeds the potential and NEVER the mask.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .model import Location
from .resources import ResourceConfig

MASK_REASONS = (
    "no_deadline",
    "not_proven_infeasible",
    "lb_exceeds_deadline",
)
HEURISTIC_ONLY_REASON = "heuristic_only_not_used"


@dataclass(frozen=True)
class ParentInput:
    """One predecessor's contribution to this task's earliest start."""

    source_location: Location
    source_ready_s: float          # parent compute finish (bytes exist there)
    bytes: int                     # bytes that must travel parent -> this task

    def __post_init__(self) -> None:
        if float(self.source_ready_s) < 0.0:
            raise ValueError("source_ready_s must be non-negative")
        if int(self.bytes) < 0:
            raise ValueError("bytes must be non-negative")


@dataclass(frozen=True)
class FeasibilityContext:
    """Per-task context. Parents stay parent-by-parent; bytes are never aggregated."""

    parents: tuple[ParentInput, ...] = ()
    deadline_s: float | None = None
    is_sink: bool = False
    criticality_class: str = "medium"
    return_bytes: int = 0          # used only when is_sink

    def __post_init__(self) -> None:
        if self.deadline_s is not None and float(self.deadline_s) < 0.0:
            raise ValueError("deadline_s must be non-negative")
        if int(self.return_bytes) < 0:
            raise ValueError("return_bytes must be non-negative")


@dataclass(frozen=True)
class ActionFeasibility:
    action: int
    location: str
    mask: bool                     # True = allowed
    lower_bound_ready_s: float
    start_lower_bound_s: float
    finish_lower_bound_s: float
    reason: str

    @property
    def allowed(self) -> bool:
        return self.mask


def _route_hops(src: Location, dst: Location) -> list[str]:
    from .routes import route

    return list(route(src, dst))


def transfer_lower_bound(
    nbytes: int, src: Location, dst: Location, resources: ResourceConfig
) -> float:
    """Zero-contention transfer time over the route: sum over its (serial) hops.

    The sum is the true no-queueing time, hence <= any real transfer, so it is
    sound and tighter than ignoring hops.
    """
    nbytes = int(nbytes)
    if nbytes <= 0 or src == dst:
        return 0.0
    hops = _route_hops(src, dst)
    if not hops:
        return 0.0
    total = 0.0
    for hop in hops:
        rate = resources.hop_rate(hop)
        if rate <= 0:
            return float("inf")
        total += nbytes / rate
    return float(total)


def _cpu_rate(resources: ResourceConfig, location: Location, cycles_per_bit) -> float:
    if getattr(resources, "physical", False):
        from .energy_model import tier_for_location

        spec = resources.energy_model
        xi = float(spec.cycles_per_bit) if cycles_per_bit is None else float(cycles_per_bit)
        return spec.tier(tier_for_location(location)).cpu_rate_bytes_per_second(xi)
    return resources.cpu_rate(location)


def start_lower_bound(
    action: int,
    ctx: FeasibilityContext,
    resources: ResourceConfig,
) -> float:
    """S_i^LB(a) = max_p (parent ready + transfer LB). 0.0 when no parents."""
    if not ctx.parents:
        return 0.0
    location = Location.from_action(action)
    return max(
        float(p.source_ready_s)
        + transfer_lower_bound(int(p.bytes), p.source_location, location, resources)
        for p in ctx.parents
    )


def finish_lower_bound(
    action: int,
    ctx: FeasibilityContext,
    resources: ResourceConfig,
    *,
    workload_bytes: int,
    global_cycles_per_bit: float | None = None,
) -> float:
    """F_i^LB(a) = S_i^LB(a) + C_i / f_a (zero-queueing compute)."""
    location = Location.from_action(action)
    rate = _cpu_rate(resources, location, global_cycles_per_bit)
    compute_s = (float(workload_bytes) / rate) if rate > 0 else float("inf")
    return float(start_lower_bound(action, ctx, resources) + compute_s)


def optimistic_lower_bound_ready(
    action: int,
    ctx: FeasibilityContext,
    resources: ResourceConfig,
    *,
    workload_bytes: int = 0,
    global_cycles_per_bit: float | None = None,
) -> float:
    """R_i^LB(a): the only quantity a mask is allowed to use.

    non-sink: R = F  (successors may be co-located -> zero delivery assumed)
    sink:     R = F + unavoidable return hop to the UE
    """
    finish = finish_lower_bound(
        action,
        ctx,
        resources,
        workload_bytes=workload_bytes,
        global_cycles_per_bit=global_cycles_per_bit,
    )
    if not ctx.is_sink:
        return finish
    location = Location.from_action(action)
    return float(
        finish
        + transfer_lower_bound(int(ctx.return_bytes), location, Location.UE, resources)
    )


def feasible_actions(
    ctx: FeasibilityContext,
    resources: ResourceConfig,
    *,
    actions: tuple[int, ...] = (0, 1, 2),
    workload_bytes: int = 0,
    global_cycles_per_bit: float | None = None,
) -> list[ActionFeasibility]:
    """Sound per-action mask for the current task (proof layer)."""
    out: list[ActionFeasibility] = []
    if ctx.deadline_s is None:
        for action in actions:
            out.append(
                ActionFeasibility(
                    action=action,
                    location=Location.from_action(action).value,
                    mask=True,
                    lower_bound_ready_s=0.0,
                    start_lower_bound_s=0.0,
                    finish_lower_bound_s=0.0,
                    reason="no_deadline",
                )
            )
        return out

    deadline = float(ctx.deadline_s)
    for action in actions:
        start_lb = start_lower_bound(action, ctx, resources)
        finish_lb = finish_lower_bound(
            action,
            ctx,
            resources,
            workload_bytes=workload_bytes,
            global_cycles_per_bit=global_cycles_per_bit,
        )
        ready_lb = optimistic_lower_bound_ready(
            action,
            ctx,
            resources,
            workload_bytes=workload_bytes,
            global_cycles_per_bit=global_cycles_per_bit,
        )
        proven_infeasible = ready_lb > deadline
        out.append(
            ActionFeasibility(
                action=action,
                location=Location.from_action(action).value,
                mask=not proven_infeasible,
                lower_bound_ready_s=ready_lb,
                start_lower_bound_s=start_lb,
                finish_lower_bound_s=finish_lb,
                reason="lb_exceeds_deadline" if proven_infeasible else "not_proven_infeasible",
            )
        )
    return out


def mask_vector(feasibilities: Sequence[ActionFeasibility]) -> list[bool]:
    """[3] boolean mask ordered by action id (0=UE, 1=MEC, 2=HELPER)."""
    by_action = {f.action: bool(f.mask) for f in feasibilities}
    return [by_action.get(a, True) for a in (0, 1, 2)]


def all_actions_masked(feasibilities: Sequence[ActionFeasibility]) -> bool:
    return bool(feasibilities) and all(not f.mask for f in feasibilities)


def heuristic_failure_must_not_mask(heuristic_found: bool) -> str:
    """③ guard: a failed constructive search is NOT evidence of infeasibility."""
    return "not_proven_infeasible" if heuristic_found else HEURISTIC_ONLY_REASON
