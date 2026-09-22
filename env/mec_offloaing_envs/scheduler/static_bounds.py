"""⑤a — static (plan-independent) per-task / per-action optimistic bounds.

Used by:
  * observation v3  — the policy must SEE why an action is infeasible
  * and as a cheap pre-filter ahead of the runtime shield

Semantics are the same relaxation as `suffix.dag_lower_bound_ready`, but with no
prefix state: predecessors are undecided, so each takes its OWN best action
(min over actions) and dependency transfers are free because co-location may
always be assumed.  The result is admissible (<= any real schedule) and depends
only on the DAG, the resource/radio model and `cycles_per_bit` — never on the
policy, so it is safe to put in the observation (no leakage of future actions).

Unlike the runtime mask, these numbers are attached to the observation BEFORE a
plan is generated; they are features, not decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from .feasibility import transfer_lower_bound
from .model import CanonicalDAG, Location
from .resources import ResourceConfig

ACTION_LOCATIONS = (Location.UE, Location.MEC, Location.HELPER)
N_ACTIONS = 3


@dataclass(frozen=True)
class StaticBounds:
    """[N, 3] arrays aligned to the decoder order."""

    order: tuple[int, ...]
    finish_lb: tuple[tuple[float, float, float], ...]
    ready_lb: tuple[tuple[float, float, float], ...]   # sink: + return hop
    min_ready_lb: tuple[float, ...]
    is_sink: tuple[bool, ...]
    max_ready_lb: float

    @property
    def n(self) -> int:
        return len(self.order)

    def position(self, task_id: int) -> int:
        try:
            return self.order.index(int(task_id))
        except ValueError as exc:
            raise KeyError("task %r not in decoder order" % (task_id,)) from exc

    def feasible_by_deadline(self, deadline_per_position: Sequence[float | None]) -> list[list[bool]]:
        """[N,3] feasibility under a per-position deadline (None -> all feasible)."""
        if len(deadline_per_position) != self.n:
            raise ValueError("deadline vector length mismatch")
        out: list[list[bool]] = []
        for i, deadline in enumerate(deadline_per_position):
            if deadline is None:
                out.append([True, True, True])
                continue
            out.append([self.ready_lb[i][a] <= float(deadline) for a in range(N_ACTIONS)])
        return out

    def min_slack_ratio(self, deadline_per_position: Sequence[float | None]) -> list[float]:
        """(d - min_a ready_lb) / d, clipped to [-1, 1]; 0.0 when no deadline."""
        if len(deadline_per_position) != self.n:
            raise ValueError("deadline vector length mismatch")
        out = []
        for i, deadline in enumerate(deadline_per_position):
            if deadline is None or float(deadline) <= 0.0:
                out.append(0.0)
                continue
            d = float(deadline)
            ratio = (d - self.min_ready_lb[i]) / d
            out.append(float(min(1.0, max(-1.0, ratio))))
        return out


def _cpu_rate(resources: ResourceConfig, location: Location, cycles_per_bit) -> float:
    if getattr(resources, "physical", False):
        from .energy_model import tier_for_location

        spec = resources.energy_model
        xi = float(spec.cycles_per_bit) if cycles_per_bit is None else float(cycles_per_bit)
        return spec.tier(tier_for_location(location)).cpu_rate_bytes_per_second(xi)
    return resources.cpu_rate(location)


def static_action_bounds(
    dag: CanonicalDAG,
    order: Sequence[Any],
    resources: ResourceConfig,
    *,
    cycles_per_bit: float | None = None,
) -> StaticBounds:
    """Per-task per-action optimistic finish / ready bounds (plan-independent)."""
    order_tuple = tuple(int(t) for t in order)
    if set(order_tuple) != set(dag.tasks) or len(order_tuple) != len(dag.tasks):
        raise ValueError("decoder order must be a permutation of the DAG task ids")

    preds = dag.predecessors()
    rank = {tid: i for i, tid in enumerate(order_tuple)}
    finish: list[list[float]] = [[0.0] * N_ACTIONS for _ in order_tuple]
    ready: list[list[float]] = [[0.0] * N_ACTIONS for _ in order_tuple]

    for pos, tid in enumerate(order_tuple):
        task = dag.tasks[tid]
        # predecessors are free (min over their actions); transfers are free
        start = 0.0
        for edge in preds[tid]:
            src_pos = rank[int(edge.src_task_id)]
            start = max(start, min(finish[src_pos]))
        is_sink = not any(e.src_task_id == tid for e in dag.edges)
        for action in range(N_ACTIONS):
            location = ACTION_LOCATIONS[action]
            rate = _cpu_rate(resources, location, cycles_per_bit)
            if rate <= 0:
                raise ValueError("non-positive cpu rate for %s" % location)
            compute = float(task.compute_workload_bytes) / rate
            finish[pos][action] = start + compute
            if is_sink:
                out_bytes = int(task.task_output_bytes)
                ready[pos][action] = finish[pos][action] + transfer_lower_bound(
                    out_bytes, location, Location.UE, resources
                )
            else:
                # successors may be co-located -> zero delivery is admissible
                ready[pos][action] = finish[pos][action]

    return StaticBounds(
        order=order_tuple,
        finish_lb=tuple(tuple(row) for row in finish),
        ready_lb=tuple(tuple(row) for row in ready),
        min_ready_lb=tuple(min(row) for row in ready),
        is_sink=tuple(
            not any(e.src_task_id == tid for e in dag.edges) for tid in order_tuple
        ),
        max_ready_lb=max((max(row) for row in ready), default=0.0),
    )


def deadline_vector(
    dag: CanonicalDAG, order: Sequence[Any]
) -> list[tuple[float | None, str]]:
    """(deadline_s, deadline_type) per decoder position, for observation use."""
    out = []
    for tid in order:
        task = dag.tasks[int(tid)]
        out.append((task.deadline_s, str(task.deadline_type)))
    return out
