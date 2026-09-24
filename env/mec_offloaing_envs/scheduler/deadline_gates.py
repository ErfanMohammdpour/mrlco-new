"""Shield diagnostics: what a deadline regime's mask actually closes.

`active_rate > 0` is not enough to judge a regime. The question is WHICH action
gets closed: the deadline-free policy collapses onto MEC (`MASKED_PPO_INTERFACE_6b`
§17), so a regime that never closes MEC cannot test that collapse. These pure
functions turn (bounds, deadlines) into the breakdown the gate report needs, per
graph and per task, split by depth / sink / criticality.

All functions are pure and TF-free so the numbers in the report are reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .masking import N_ACTIONS

ACTION_NAMES = ("ue", "mec", "helper")
ACTION_INDEX = {"ue": 0, "mec": 1, "helper": 2}


@dataclass(frozen=True)
class MaskBreakdown:
    """Token counts for one graph (or an aggregate of graphs)."""

    tokens: int = 0
    active: int = 0
    forced: int = 0
    all_invalid: int = 0
    closed: tuple[int, int, int] = (0, 0, 0)      # ue, mec, helper
    sink_tokens: int = 0
    sink_active: int = 0
    depth_active: Mapping[int, int] = field(default_factory=dict)
    depth_tokens: Mapping[int, int] = field(default_factory=dict)
    criticality_active: Mapping[str, int] = field(default_factory=dict)
    criticality_tokens: Mapping[str, int] = field(default_factory=dict)

    def __add__(self, other: "MaskBreakdown") -> "MaskBreakdown":
        if not isinstance(other, MaskBreakdown):
            return NotImplemented

        def merge(a: Mapping[Any, int], b: Mapping[Any, int]) -> dict[Any, int]:
            out = dict(a)
            for key, value in b.items():
                out[key] = out.get(key, 0) + value
            return out

        return MaskBreakdown(
            tokens=self.tokens + other.tokens,
            active=self.active + other.active,
            forced=self.forced + other.forced,
            all_invalid=self.all_invalid + other.all_invalid,
            closed=tuple(a + b for a, b in zip(self.closed, other.closed)),
            sink_tokens=self.sink_tokens + other.sink_tokens,
            sink_active=self.sink_active + other.sink_active,
            depth_active=merge(self.depth_active, other.depth_active),
            depth_tokens=merge(self.depth_tokens, other.depth_tokens),
            criticality_active=merge(self.criticality_active, other.criticality_active),
            criticality_tokens=merge(self.criticality_tokens, other.criticality_tokens),
        )

    def rates(self) -> dict[str, float]:
        n = float(self.tokens) if self.tokens else 0.0
        if not n:
            return {
                "tokens": 0,
                "active_rate": 0.0, "forced_rate": 0.0, "all_invalid_rate": 0.0,
                "mec_closed_rate": 0.0, "ue_closed_rate": 0.0, "helper_closed_rate": 0.0,
                "sink_active_rate": 0.0,
            }
        return {
            "tokens": int(self.tokens),
            "active_rate": self.active / n,
            "forced_rate": self.forced / n,
            "all_invalid_rate": self.all_invalid / n,
            "ue_closed_rate": self.closed[0] / n,
            "mec_closed_rate": self.closed[1] / n,
            "helper_closed_rate": self.closed[2] / n,
            "sink_active_rate": (self.sink_active / self.sink_tokens) if self.sink_tokens else 0.0,
        }


def mask_from_deadlines(
    bounds: Any, deadlines_by_position: Sequence[float | None]
) -> list[list[bool]]:
    """[N, 3] feasibility: an action stays open while its ready bound fits.

    Mirrors `StaticBounds.feasible_by_deadline`; the deadline basis is the ready
    bound (sink return hop included) because that is what the engine compares.
    """
    if len(deadlines_by_position) != bounds.n:
        raise ValueError(
            "deadline vector length %d != task count %d"
            % (len(deadlines_by_position), bounds.n)
        )
    out: list[list[bool]] = []
    for i, deadline in enumerate(deadlines_by_position):
        if deadline is None:
            out.append([True] * N_ACTIONS)
            continue
        out.append([bounds.ready_lb[i][a] <= float(deadline) for a in range(N_ACTIONS)])
    return out


def breakdown_for_graph(
    bounds: Any,
    deadlines_by_position: Sequence[float | None],
    *,
    depths: Mapping[int, int] | None = None,
    criticality: Mapping[int, str] | None = None,
) -> MaskBreakdown:
    """Pre-guard breakdown for one graph, in decoder-position order."""
    feasible = mask_from_deadlines(bounds, deadlines_by_position)
    closed = [0, 0, 0]
    active = forced = all_invalid = 0
    sink_tokens = sink_active = 0
    depth_active: dict[int, int] = {}
    depth_tokens: dict[int, int] = {}
    crit_active: dict[str, int] = {}
    crit_tokens: dict[str, int] = {}

    for pos in range(bounds.n):
        row = feasible[pos]
        n_open = sum(1 for flag in row if flag)
        is_active = n_open < N_ACTIONS
        if is_active:
            active += 1
        if n_open == 1:
            forced += 1
        if n_open == 0:
            all_invalid += 1
        for action in range(N_ACTIONS):
            if not row[action]:
                closed[action] += 1
        if bounds.is_sink[pos]:
            sink_tokens += 1
            if is_active:
                sink_active += 1
        if depths is not None:
            depth = int(depths.get(pos, -1))
            depth_tokens[depth] = depth_tokens.get(depth, 0) + 1
            if is_active:
                depth_active[depth] = depth_active.get(depth, 0) + 1
        if criticality is not None:
            cls = str(criticality.get(pos, "unknown"))
            crit_tokens[cls] = crit_tokens.get(cls, 0) + 1
            if is_active:
                crit_active[cls] = crit_active.get(cls, 0) + 1

    return MaskBreakdown(
        tokens=bounds.n,
        active=active,
        forced=forced,
        all_invalid=all_invalid,
        closed=tuple(closed),
        sink_tokens=sink_tokens,
        sink_active=sink_active,
        depth_active=depth_active,
        depth_tokens=depth_tokens,
        criticality_active=crit_active,
        criticality_tokens=crit_tokens,
    )


def gate_summary(
    per_graph: Mapping[str, MaskBreakdown],
    witness_found: Mapping[str, bool],
    witness_mixed: Mapping[str, bool],
    excluded: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Aggregate one (regime, split) into the numbers the gate report needs."""
    total = len(per_graph) + len(excluded)
    aggregate = MaskBreakdown()
    for breakdown in per_graph.values():
        aggregate = aggregate + breakdown
    mec_closed_graphs = sum(
        1 for breakdown in per_graph.values() if breakdown.closed[1] > 0
    )
    summary = aggregate.rates()
    summary.update(
        {
            "graphs": len(per_graph),
            "graphs_total": total,
            "witness_rate": (len(per_graph) / total) if total else 0.0,
            "witness_found": sum(1 for key in per_graph if witness_found.get(key, False)),
            "witness_mixed_rate": (
                sum(1 for key in per_graph if witness_mixed.get(key, False)) / len(per_graph)
                if per_graph
                else 0.0
            ),
            "graphs_with_mec_closure": mec_closed_graphs,
            "graphs_with_mec_closure_rate": mec_closed_graphs / len(per_graph) if per_graph else 0.0,
            "excluded": [
                {"graph": item.get("graph"), "reason": item.get("reason")}
                for item in excluded[:20]
            ],
            "depth_active": dict(sorted(aggregate.depth_active.items())),
            "depth_tokens": dict(sorted(aggregate.depth_tokens.items())),
            "criticality_active": dict(sorted(aggregate.criticality_active.items())),
            "criticality_tokens": dict(sorted(aggregate.criticality_tokens.items())),
        }
    )
    return summary


def passes_trainable_gate(
    summary: Mapping[str, Any],
    *,
    active_lo: float = 0.05,
    active_hi: float = 0.40,
    preferred_lo: float = 0.10,
    preferred_hi: float = 0.30,
    all_invalid_hi: float = 0.01,
    witness_min: float = 0.95,
) -> dict[str, Any]:
    """The acceptance rules, evaluated rather than assumed."""
    checks = {
        "active_rate_in_range": active_lo <= float(summary["active_rate"]) <= active_hi,
        "active_rate_preferred": preferred_lo <= float(summary["active_rate"]) <= preferred_hi,
        "all_invalid_below_limit": float(summary["all_invalid_rate"]) < all_invalid_hi,
        "witness_rate_ok": float(summary["witness_rate"]) >= witness_min,
        "mec_closure_present": float(summary["mec_closed_rate"]) > 0.0,
    }
    checks["passes"] = (
        checks["active_rate_in_range"]
        and checks["all_invalid_below_limit"]
        and checks["witness_rate_ok"]
    )
    return checks
