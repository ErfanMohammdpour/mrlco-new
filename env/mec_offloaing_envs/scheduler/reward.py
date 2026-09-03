"""Post-hoc telescoping token rewards (OBJECTIVE_AND_ENERGY.md §6).

Training reward is NOT clipped. Scientific `J_report` stays separate and clipped
and is opt-in via `compute_j_report` (off on the training path).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .adapter import schedule_via_adapter, validate_plan
from .energy_api import (
    ENERGY_WEIGHT,
    LATENCY_WEIGHT,
    ReferenceRanges,
    attribute_energy_by_task,
    compute_reference_ranges,
    j_report,
    require_publication_weights,
)
from .model import ScheduleResult
from .resources import ResourceConfig

FILL_UNASSIGNED = 0  # all_UE completion policy


@dataclass(frozen=True)
class TelescopingRewardResult:
    """Token rewards r_1..r_N plus provisional schedule traces."""

    rewards: list[float]
    makespans: list[float]  # L_0..L_N
    energies: list[float]  # E_0..E_N
    refs: ReferenceRanges
    final_result: ScheduleResult
    final_per_task_energy: list[float]
    j_report_value: float | None  # None unless compute_j_report=True

    @property
    def final_makespan(self) -> float:
        return self.makespans[-1]

    @property
    def final_energy(self) -> float:
        return self.energies[-1]


def provisional_plan(
    decoder_order: Sequence[int],
    decided_actions: Sequence[int],
    *,
    fill: int = FILL_UNASSIGNED,
) -> list[tuple[int, int]]:
    """Build P_t: prefix decided_actions, suffix filled with `fill` (default all_UE)."""
    order = [int(tid) for tid in decoder_order]
    n = len(order)
    decided = [int(a) for a in decided_actions]
    if len(decided) > n:
        raise ValueError(f"decided_actions length {len(decided)} > N={n}")
    for a in decided:
        if a not in (0, 1, 2):
            raise ValueError(f"action must be 0/1/2, got {a}")
    if fill not in (0, 1, 2):
        raise ValueError(f"fill must be 0/1/2, got {fill}")
    actions = list(decided) + [fill] * (n - len(decided))
    return list(zip(order, actions))


def telescoping_token_rewards(
    task_graph: Any,
    plan: Sequence[tuple[int, int]],
    resources: ResourceConfig,
    *,
    include_energy: bool = True,
    latency_weight: float | None = None,
    energy_weight: float | None = None,
    refs: ReferenceRanges | None = None,
    compute_j_report: bool = False,
) -> TelescopingRewardResult:
    """Post-hoc telescoping with completion policy all_UE.

    Schedules P_1..P_N (P_0 metrics reused from pure-location all_UE refs).
    Deltas are unclipped. Token reward:

        r_t = -(w_L * (L_t - L_{t-1}) / L_scale + w_E * (E_t - E_{t-1}) / E_scale)

    Publication mode freezes w_L/w_E at 0.5/0.5. Training path leaves
    `compute_j_report=False` to avoid clip_and_log warning floods.
    """
    decoder_order, actions = validate_plan(task_graph, plan)
    n = len(decoder_order)

    if include_energy:
        if latency_weight is None and energy_weight is None:
            lw, ew = LATENCY_WEIGHT, ENERGY_WEIGHT
            require_publication_weights(lw, ew)
        else:
            lw, ew = require_publication_weights(
                LATENCY_WEIGHT if latency_weight is None else latency_weight,
                ENERGY_WEIGHT if energy_weight is None else energy_weight,
            )
    else:
        lw = LATENCY_WEIGHT if latency_weight is None else float(latency_weight)
        ew = 0.0

    if refs is None:
        refs = compute_reference_ranges(task_graph, resources)

    # Reuse all_UE reference metrics as P_0 — no extra schedule call.
    makespans: list[float] = [refs.L_ue]
    energies: list[float] = [refs.E_ue]
    final_result: ScheduleResult | None = None

    for t in range(1, n + 1):
        prov = provisional_plan(decoder_order, actions[:t], fill=FILL_UNASSIGNED)
        result, _, _ = schedule_via_adapter(task_graph, prov, resources)
        makespans.append(result.makespan_seconds)
        energies.append(result.total_mobile_joules)
        if t == n:
            final_result = result

    assert final_result is not None

    rewards: list[float] = []
    for t in range(1, n + 1):
        delta_l = makespans[t] - makespans[t - 1]
        delta_e = energies[t] - energies[t - 1]
        term = lw * (delta_l / refs.L_scale)
        if include_energy:
            term += ew * (delta_e / refs.E_scale)
        rewards.append(-term)

    energy_map = attribute_energy_by_task(final_result, resources)
    per_task = [float(energy_map.get(tid, 0.0)) for tid in decoder_order]
    j_val = (
        j_report(makespans[-1], energies[-1], refs) if compute_j_report else None
    )

    return TelescopingRewardResult(
        rewards=rewards,
        makespans=makespans,
        energies=energies,
        refs=refs,
        final_result=final_result,
        final_per_task_energy=per_task,
        j_report_value=j_val,
    )


def expected_episode_return(
    makespans: Sequence[float],
    energies: Sequence[float],
    refs: ReferenceRanges,
    *,
    include_energy: bool = True,
    latency_weight: float = LATENCY_WEIGHT,
    energy_weight: float = ENERGY_WEIGHT,
) -> float:
    """Closed form: sum_t r_t == -(w_L*(L_N-L_0)/L_scale + w_E*(E_N-E_0)/E_scale)."""
    term = latency_weight * ((makespans[-1] - makespans[0]) / refs.L_scale)
    if include_energy:
        term += energy_weight * ((energies[-1] - energies[0]) / refs.E_scale)
    return -term
