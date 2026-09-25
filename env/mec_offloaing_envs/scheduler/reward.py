"""Post-hoc telescoping token rewards (OBJECTIVE_AND_ENERGY.md §6).

Training reward is NOT clipped. Scientific `J_report` stays separate and clipped
and is opt-in via `compute_j_report` (off on the training path).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .adapter import schedule_via_adapter, validate_plan
from .constraints import (
    ATTRIBUTION_TELESCOPED,
    EMPTY_COSTS,
    ConstraintCosts,
    ConstraintMetrics,
    ConstraintSpec,
    costs_from_metrics,
    evaluate_constraints,
)
from .energy_api import (
    ENERGY_WEIGHT,
    LATENCY_WEIGHT,
    ReferenceRanges,
    attribute_scoped_energy_by_task,
    compute_reference_ranges,
    j_report,
    require_publication_weights,
)
from .energy_scope import SCOPE_MOBILE, energy_scalar
from .model import ScheduleResult
from .resources import ResourceConfig

FILL_UNASSIGNED = 0  # all_UE completion policy
LATENCY_REF_L_SCALE = "l_scale"  # frozen v0.1: divide by pure-location L_scale
LATENCY_REF_L_MEC = "l_mec"  # diagnostic: divide by all-MEC makespan; not publication
REWARD_MODE_PUBLICATION = "publication"  # LEGACY: explicit opt-in only (mobile energy)
REWARD_MODE_LATENCY_ONLY = "latency_only"  # PRIMARY (E3.1): no energy term at all
REWARD_MODE_LATENCY_TMEC = "latency_over_all_mec"  # diagnostic
REWARD_MODES = (
    REWARD_MODE_PUBLICATION,
    REWARD_MODE_LATENCY_ONLY,
    REWARD_MODE_LATENCY_TMEC,
)


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
    constraint_costs: ConstraintCosts = EMPTY_COSTS  # EMPTY unless constraints enabled
    constraint_penalty: float = 0.0  # sum_i lambda_i * violation_i actually applied

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


def _all_ue_constraint_metrics(refs: ReferenceRanges, n_tasks: int) -> ConstraintMetrics:
    """P_0 (all-UE) metrics, derived from the reference ranges — no extra schedule.

    The all-UE plan has no radio and no helper work, so its UE energy IS `E_ue`
    and its makespan IS `L_ue`.
    """
    return ConstraintMetrics(
        ue_energy_j=float(refs.E_ue),
        helper_energy_j=0.0,
        total_energy_j=float(refs.E_ue),
        helper_compute_j=0.0,
        v2v_airtime_s=0.0,
        v2v_task_fraction=0.0,
        makespan_s=float(refs.L_ue),
        n_tasks=int(n_tasks),
        n_helper_tasks=0,
    )


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
    latency_ref: str = LATENCY_REF_L_SCALE,
    reward_mode: str = REWARD_MODE_PUBLICATION,
    constraints: ConstraintSpec | None = None,
    duals: Sequence[float] | None = None,
    discount: float = 1.0,
    reference_mode: str = "pure_location",
) -> TelescopingRewardResult:
    """Post-hoc telescoping with completion policy all_UE.

    Schedules P_1..P_N (P_0 metrics reused from pure-location all_UE refs).
    Deltas are unclipped. Token reward (publication):

        r_t = J_{t-1} - discount * J_t
        J_t = w_L * L_t / L_scale + w_E * E_t / E_scale        (potential)

    With `discount=1.0` this is exactly the historical form
    `-(w_L * (L_t - L_{t-1}) / L_scale + w_E * (E_t - E_{t-1}) / E_scale)`.
    With `discount=gamma<1` the shaping becomes potential-based
    (`F = gamma*Phi(s') - Phi(s)`, `Phi = -J`) and the discounted return
    telescopes exactly:  `sum_t gamma^(t-1) r_t = J_0 - gamma^N J_N`
    so maximising the PPO return is maximising the final schedule objective.

    Diagnostic `latency_ref=l_mec` (not v0.1 publication):

        J_t = L_t / T_allMEC

    Constrained mode (`constraints.enabled`) adds the Lagrangian penalty
    `- sum_i lambda_i * max(0, (c_i - b_i)/scale_i)`:

    * `attribution="terminal"` (default): the whole penalty lands on the last
      token — exact plan-level constraint semantics.
    * `attribution="telescoped"`: the penalty is spread over tokens as deltas of
      the *signed* cost, so `sum_t r_t` equals the unconstrained return minus
      `sum_i lambda_i * (signed_i(P_N) - signed_i(P_0))`.

    Publication mode freezes w_L/w_E at 0.5/0.5. Training path leaves
    `compute_j_report=False` to avoid clip_and_log warning floods.

    `reward_mode` (E3.1):

    * `latency_only` (PRIMARY): `J_t = L_t / L_scale`, NO energy term at any
      boundary. `include_energy`/energy weights are ignored.
    * `publication` (LEGACY, explicit opt-in only): the historical 0.5/0.5
      potential with the MOBILE energy term, byte-exact.
    * `latency_over_all_mec`: the existing diagnostic.
    """
    decoder_order, actions = validate_plan(task_graph, plan)
    n = len(decoder_order)
    if latency_ref not in (LATENCY_REF_L_SCALE, LATENCY_REF_L_MEC):
        raise ValueError("latency_ref must be l_scale or l_mec, got %r" % (latency_ref,))
    if reward_mode not in REWARD_MODES:
        raise ValueError("reward_mode must be one of %s, got %r" % (REWARD_MODES, reward_mode))
    discount = float(discount)
    if not 0.0 < discount <= 1.0:
        raise ValueError("discount must be in (0, 1], got %s" % discount)
    latency_only = reward_mode == REWARD_MODE_LATENCY_ONLY
    diagnostic_tmec = reward_mode == REWARD_MODE_LATENCY_TMEC or latency_ref == LATENCY_REF_L_MEC
    if latency_only:
        # energy-free primary: always the pure-location latency scale
        latency_ref = LATENCY_REF_L_SCALE
        diagnostic_tmec = False
    constrained = bool(constraints is not None and constraints.enabled)
    lagrangian = [float(l) for l in (duals or [])]
    if constrained and len(lagrangian) != len(constraints.active_names):
        raise ValueError(
            "duals length %d != active constraints %d"
            % (len(lagrangian), len(constraints.active_names))
        )

    if latency_only:
        lw, ew = 1.0, 0.0
        include_energy = False
    elif diagnostic_tmec:
        lw, ew = 1.0, 0.0
        include_energy = False
    elif include_energy:
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
        refs = compute_reference_ranges(task_graph, resources, mode=reference_mode)

    # Reuse all_UE reference metrics as P_0 — no extra schedule call.
    makespans: list[float] = [refs.L_ue]
    energies: list[float] = [refs.E_ue]
    final_result: ScheduleResult | None = None
    prefix_signed: list[tuple[float, ...]] = []
    if constrained and constraints.attribution == ATTRIBUTION_TELESCOPED:
        base = costs_from_metrics(_all_ue_constraint_metrics(refs, n), refs, constraints)
        prefix_signed.append(base.signed)

    for t in range(1, n + 1):
        prov = provisional_plan(decoder_order, actions[:t], fill=FILL_UNASSIGNED)
        result, _, _ = schedule_via_adapter(task_graph, prov, resources)
        makespans.append(result.makespan_seconds)
        energies.append(energy_scalar(result, scope=SCOPE_MOBILE))
        if constrained and constraints.attribution == ATTRIBUTION_TELESCOPED:
            costs_t = evaluate_constraints(result, resources, refs, constraints)
            prefix_signed.append(costs_t.signed)
        if t == n:
            final_result = result

    assert final_result is not None

    # -- token rewards: potential-based telescoping --------------------------
    # J_t is the provisional-plan objective (unclipped). Offsets cancel in the
    # differences, so r_t = J_{t-1} - discount*J_t reduces to the historical
    # delta form when discount == 1.
    denom_l = max(float(refs.L_mec), 1e-12) if diagnostic_tmec else refs.L_scale
    potentials: list[float] = []
    for t in range(n + 1):
        j_t = lw * (makespans[t] / denom_l)
        if include_energy:
            j_t += ew * (energies[t] / refs.E_scale)
        potentials.append(j_t)

    rewards: list[float] = []
    for t in range(1, n + 1):
        rewards.append(potentials[t - 1] - discount * potentials[t])

    # -- constraints ---------------------------------------------------------
    final_costs = EMPTY_COSTS
    penalty_applied = 0.0
    if constrained:
        final_costs = evaluate_constraints(final_result, resources, refs, constraints)
        if constraints.attribution == ATTRIBUTION_TELESCOPED:
            for t in range(1, n + 1):
                step_penalty = 0.0
                for lam, prev, cur in zip(lagrangian, prefix_signed[t - 1], prefix_signed[t]):
                    step_penalty += lam * (cur - prev)
                rewards[t - 1] -= step_penalty
                penalty_applied += step_penalty
        else:
            penalty_applied = final_costs.penalty(lagrangian)
            rewards[-1] -= penalty_applied

    energy_map = attribute_scoped_energy_by_task(
        final_result, resources, scope=SCOPE_MOBILE
    )
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
        constraint_costs=final_costs,
        constraint_penalty=float(penalty_applied),
    )


def expected_episode_return(
    makespans: Sequence[float],
    energies: Sequence[float],
    refs: ReferenceRanges,
    *,
    include_energy: bool = True,
    latency_weight: float = LATENCY_WEIGHT,
    energy_weight: float = ENERGY_WEIGHT,
    latency_ref: str = LATENCY_REF_L_SCALE,
    reward_mode: str = REWARD_MODE_PUBLICATION,
    discount: float = 1.0,
) -> float:
    """Closed form of the (optionally discounted) token-reward sum.

    `discount=1`: `-(w_L*(L_N-L_0)/L_scale + w_E*(E_N-E_0)/E_scale)`.
    `discount=gamma`: `J_0 - gamma^N * J_N` with `J_t = w_L*L_t/L_scale(+E term)`.
    Diagnostic l_mec uses T_allMEC, no energy term. `reward_mode="latency_only"`
    is the energy-free primary: `J_t = L_t / L_scale`.
    """
    n = len(makespans) - 1
    discount = float(discount)
    if reward_mode not in REWARD_MODES:
        raise ValueError("reward_mode must be one of %s, got %r" % (REWARD_MODES, reward_mode))
    if reward_mode == REWARD_MODE_LATENCY_ONLY:
        include_energy = False
        latency_weight, energy_weight = 1.0, 0.0
        latency_ref = LATENCY_REF_L_SCALE
    elif reward_mode == REWARD_MODE_LATENCY_TMEC:
        latency_ref = LATENCY_REF_L_MEC
    if latency_ref == LATENCY_REF_L_MEC:
        denom = max(float(refs.L_mec), 1e-12)
        j0 = makespans[0] / denom
        jn = makespans[-1] / denom
        return float(j0 - (discount ** n) * jn if discount != 1.0 else -(jn - j0))
    denom_l = refs.L_scale

    def _potential(idx: int) -> float:
        j_t = latency_weight * (makespans[idx] / denom_l)
        if include_energy:
            j_t += energy_weight * (energies[idx] / refs.E_scale)
        return j_t

    j0, jn = _potential(0), _potential(n)
    if discount == 1.0:
        return float(-(jn - j0))
    return float(j0 - (discount ** n) * jn)
