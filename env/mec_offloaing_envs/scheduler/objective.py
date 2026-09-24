"""②B-1/B-2: plan-level objective, constraint cost channels, and lexicographic
checkpoint selection. Pure numpy — no TF, no PPO, no Lagrangian.

②B-1 objective (approved):

    L_norm          = L / L_ref                 (L_ref = per-graph reference)
    T_soft_norm     = sum_{i in S} w_i * [F_ready_i - d_i]^+ / d_i
    J               = L_norm + beta_s * T_soft_norm

Constraint cost channels (computed and logged only in ②B-1):

    c_E = [ E_system / B_E - 1 ]^+            (capped at `cap`)
    c_H = #hard_miss / max(1, N_H)
    c_F = #firm_miss / max(1, N_F)      with budget  c_F <= eps_F

L_ref is a *fixed per-graph reference built before policy evaluation* (default:
the all-UE plan) — no min-max composite, so the objective is monotone in latency
and does not saturate.

②B-2 checkpoint selection is lexicographic, never a single scalar:

    feasible(pi) := c_H == 0 and c_E == 0 and c_F <= eps_F
    pi* = argmin_{pi feasible} J(pi)

If no checkpoint is feasible we report that explicitly (with the smallest
normalized constraint violation for debugging) and refuse to name a winner.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from .energy_api import ReferenceRanges
from .energy_scope import SCOPE_SYSTEM, energy_scalar
from .model import ScheduleResult
from .validate import require_finite, require_nonneg_float

LATENCY_REF_ALL_UE = "all_ue"
LATENCY_REF_ALL_MEC = "all_mec"
LATENCY_REF_FIXED = "fixed"
LATENCY_REFS = (LATENCY_REF_ALL_UE, LATENCY_REF_ALL_MEC, LATENCY_REF_FIXED)

SELECTION_METRIC_NAME = "lexicographic_feasible_then_J"
DEFAULT_COST_CAP = 10.0


@dataclass(frozen=True)
class ObjectiveSpec:
    """Plan objective + constraint budgets (no Lagrange multipliers here)."""

    latency_ref: str = LATENCY_REF_ALL_UE
    latency_denominator_s: float | None = None  # required for LATENCY_REF_FIXED
    beta_soft: float = 1.0
    # Energy budget B_E. EXACTLY ONE of the two knobs must be set, and it must be
    # strictly positive (a zero budget is not a budget, it is a typo).
    energy_budget_j: float | None = None
    energy_budget_frac_of_all_ue: float | None = None
    # Hard deadline is a hard constraint in the primary configuration: epsilon is
    # frozen at 0. A probabilistic hard deadline would be a DIFFERENT semantic
    # ("chance_hard"), not a relaxed epsilon.
    hard_miss_epsilon: float = 0.0
    firm_miss_epsilon: float = 0.0
    cost_cap: float = DEFAULT_COST_CAP
    # Reserved for a future chance-constrained formulation. Must stay None in v1.
    chance_hard_epsilon: float | None = None

    def __post_init__(self) -> None:
        if self.latency_ref not in LATENCY_REFS:
            raise ValueError(
                "latency_ref must be one of %s, got %r" % (LATENCY_REFS, self.latency_ref)
            )
        if self.latency_ref == LATENCY_REF_FIXED:
            if self.latency_denominator_s is None or float(self.latency_denominator_s) <= 0.0:
                raise ValueError("latency_ref='fixed' requires a positive denominator")
        require_nonneg_float("beta_soft", float(self.beta_soft))
        require_nonneg_float("cost_cap", float(self.cost_cap))
        if float(self.hard_miss_epsilon) != 0.0:
            raise ValueError(
                "hard_miss_epsilon must stay 0 in the primary configuration: a hard "
                "deadline is a hard constraint. A probabilistic form is a different "
                "semantic (chance_hard), not a relaxed epsilon."
            )
        require_nonneg_float("firm_miss_epsilon", float(self.firm_miss_epsilon))
        if self.chance_hard_epsilon is not None:
            raise NotImplementedError(
                "chance_hard_epsilon is reserved and not implemented in v1"
            )
        set_j = self.energy_budget_j is not None
        set_frac = self.energy_budget_frac_of_all_ue is not None
        if set_j == set_frac:
            raise ValueError(
                "set EXACTLY ONE of energy_budget_j / energy_budget_frac_of_all_ue "
                "(got j=%r, frac=%r)" % (self.energy_budget_j, self.energy_budget_frac_of_all_ue)
            )
        for name in ("energy_budget_j", "energy_budget_frac_of_all_ue"):
            value = getattr(self, name)
            if value is not None and float(value) <= 0.0:
                raise ValueError("%s must be strictly positive, got %r" % (name, value))

    @classmethod
    def from_dict(cls, doc: dict[str, Any] | None) -> "ObjectiveSpec":
        if not doc:
            raise ValueError("ObjectiveSpec.from_dict requires a mapping")
        known = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        unknown = sorted(set(doc) - known)
        if unknown:
            raise ValueError("unknown objective keys: %s" % unknown)
        kwargs: dict[str, Any] = {}
        for key, value in doc.items():
            if key == "latency_ref":
                kwargs[key] = str(value)
            elif value is None:
                kwargs[key] = None
            else:
                kwargs[key] = float(value)
        return cls(**kwargs)

    def as_dict(self) -> dict[str, Any]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__  # type: ignore[attr-defined]
        }

    def latency_denominator(self, refs: ReferenceRanges) -> float:
        if self.latency_ref == LATENCY_REF_ALL_MEC:
            return max(float(refs.L_mec), 1e-12)
        if self.latency_ref == LATENCY_REF_FIXED:
            return max(float(self.latency_denominator_s), 1e-12)
        return max(float(refs.L_ue), 1e-12)

    def energy_budget(self, refs: ReferenceRanges) -> float:
        if self.energy_budget_j is not None:
            return float(self.energy_budget_j)
        return float(self.energy_budget_frac_of_all_ue) * max(float(refs.E_ue), 1e-12)


@dataclass(frozen=True)
class PlanObjective:
    """Evaluated plan: objective value + every constraint channel."""

    latency_s: float
    latency_ref_s: float
    latency_norm: float
    soft_tardiness_s: float
    soft_tardiness_norm: float
    beta_soft: float
    J: float
    energy_system_j: float
    energy_budget_j: float
    c_E: float                     # capped cost (usable as a PPO cost signal)
    hard_miss_count: int
    hard_task_count: int
    c_H: float
    firm_miss_count: int
    firm_task_count: int
    c_F: float
    hard_miss_epsilon: float
    firm_miss_epsilon: float
    n_soft_tasks: int = 0
    # Uncapped violation: the cap is a PPO convenience, never a reporting one.
    c_E_raw: float = 0.0

    # -- feasibility gate ------------------------------------------------
    @property
    def energy_ok(self) -> bool:
        return self.c_E <= 0.0

    @property
    def hard_ok(self) -> bool:
        return self.c_H <= self.hard_miss_epsilon + 1e-12

    @property
    def firm_ok(self) -> bool:
        return self.c_F <= self.firm_miss_epsilon + 1e-12

    @property
    def feasible(self) -> bool:
        return self.hard_ok and self.energy_ok and self.firm_ok

    @property
    def total_violation(self) -> float:
        """Normalized violation magnitude (debug ordering only)."""
        return float(
            max(0.0, self.c_H - self.hard_miss_epsilon)
            + max(0.0, self.c_E)
            + max(0.0, self.c_F - self.firm_miss_epsilon)
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "J": self.J,
            "latency_s": self.latency_s,
            "latency_ref_s": self.latency_ref_s,
            "latency_norm": self.latency_norm,
            "soft_tardiness_s": self.soft_tardiness_s,
            "soft_tardiness_norm": self.soft_tardiness_norm,
            "beta_soft": self.beta_soft,
            "energy_system_j": self.energy_system_j,
            "energy_budget_j": self.energy_budget_j,
            "c_E": self.c_E,
            "c_E_raw": self.c_E_raw,
            "c_H": self.c_H,
            "c_F": self.c_F,
            "hard_miss_count": float(self.hard_miss_count),
            "firm_miss_count": float(self.firm_miss_count),
            "n_hard_tasks": float(self.hard_task_count),
            "n_firm_tasks": float(self.firm_task_count),
            "n_soft_tasks": float(self.n_soft_tasks),
            "feasible": float(self.feasible),
            "total_violation": self.total_violation,
        }


def evaluate_plan_objective(
    result: ScheduleResult,
    refs: ReferenceRanges,
    spec: ObjectiveSpec,
) -> PlanObjective:
    """J = L_norm + beta_s * T_soft_norm, plus the three cost channels."""
    latency = require_finite("makespan", result.makespan_seconds)
    denom = spec.latency_denominator(refs)
    hard_count = 0
    firm_count = 0
    soft_count = 0
    for rec in result.tasks.values():
        if rec.deadline_type == "hard":
            hard_count += 1
        elif rec.deadline_type == "firm":
            firm_count += 1
        elif rec.deadline_type == "soft":
            soft_count += 1
    # SYSTEM boundary (E2.1): the numerator is explicit. Its budget/reference
    # still comes from the mobile-built `refs` until 4.3 aligns them on system;
    # that mismatch is recorded in the consumer inventory, never hidden.
    energy = energy_scalar(result, scope=SCOPE_SYSTEM)
    budget = spec.energy_budget(refs)
    c_e_raw = (energy / budget) - 1.0 if budget > 0 else float("inf")
    c_e = min(max(0.0, c_e_raw), spec.cost_cap)
    c_h = float(result.hard_miss_count) / float(max(1, hard_count))
    c_f = float(result.firm_miss_count) / float(max(1, firm_count))
    latency_norm = latency / denom
    t_soft = float(result.soft_tardiness_normalized)
    return PlanObjective(
        latency_s=latency,
        latency_ref_s=denom,
        latency_norm=latency_norm,
        soft_tardiness_s=float(result.soft_tardiness_s),
        soft_tardiness_norm=t_soft,
        beta_soft=float(spec.beta_soft),
        J=latency_norm + float(spec.beta_soft) * t_soft,
        energy_system_j=energy,
        energy_budget_j=budget,
        c_E=c_e,
        c_E_raw=float(c_e_raw),
        hard_miss_count=int(result.hard_miss_count),
        hard_task_count=int(hard_count),
        c_H=c_h,
        firm_miss_count=int(result.firm_miss_count),
        firm_task_count=int(firm_count),
        c_F=c_f,
        hard_miss_epsilon=float(spec.hard_miss_epsilon),
        firm_miss_epsilon=float(spec.firm_miss_epsilon),
        n_soft_tasks=int(soft_count),
    )


def objective_log_kvs(obj: PlanObjective, prefix: str = "objective") -> dict[str, float]:
    return {f"{prefix}/{k}": v for k, v in obj.as_dict().items()}


# ---------------------------------------------------------------------------
# ②B-2 lexicographic checkpoint selection
# ---------------------------------------------------------------------------
def selection_key(obj: PlanObjective) -> tuple:
    """Sort key implementing the approved lexicographic rule.

    Feasible checkpoints always rank before infeasible ones; among feasible ones
    the smallest J wins; among infeasible ones the smallest total violation is
    shown first (debug ordering only — never reported as the scientific winner).
    """
    if obj.feasible:
        return (0, obj.J, obj.total_violation)
    return (1, obj.total_violation, obj.J)


@dataclass
class CheckpointCandidate:
    label: str
    objective: PlanObjective


@dataclass
class CheckpointChoice:
    winner: str | None
    feasible: bool
    ranked: list[str] = field(default_factory=list)
    best_infeasible: str | None = None
    note: str = ""


def choose_checkpoint(candidates: Sequence[CheckpointCandidate]) -> CheckpointChoice:
    """Lexicographic selection. Reports infeasibility instead of inventing a winner."""
    if not candidates:
        return CheckpointChoice(winner=None, feasible=False, note="no candidates")
    ordered = sorted(candidates, key=lambda c: selection_key(c.objective))
    ranked = [c.label for c in ordered]
    best = ordered[0]
    if best.objective.feasible:
        return CheckpointChoice(
            winner=best.label,
            feasible=True,
            ranked=ranked,
            note="feasible: min J among feasible checkpoints",
        )
    return CheckpointChoice(
        winner=None,
        feasible=False,
        ranked=ranked,
        best_infeasible=best.label,
        note=(
            "NO FEASIBLE CHECKPOINT: no candidate satisfies hard/energy/firm "
            "constraints; '%s' has the smallest normalized violation and is NOT a "
            "scientific winner" % best.label
        ),
    )


# ---------------------------------------------------------------------------
# Candidate-panel energy budget (③ support)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PanelBudget:
    """B_E(rho) = E_min + rho * (E_fast - E_min) from a candidate panel.

    A fraction-of-all-UE budget is a poor primary anchor in the physical model
    (MEC compute dwarfs all-UE energy, so the budget trips immediately).  The
    panel form interpolates between the cheapest panel member and the energy of
    the fastest panel member, both of which are achievable plans:

        E_min  = min over the panel of E_s
        E_fast = E of argmin_s T_s

    This is still a *diagnostic* budget: the final joint energy+deadline budget
    must come from the dataset generator's jointly feasible anchor schedule.
    """

    e_min: float
    e_fast: float
    rho: float

    def __post_init__(self) -> None:
        for name in ("e_min", "e_fast"):
            if float(getattr(self, name)) < 0.0:
                raise ValueError("%s must be non-negative" % name)
        if not 0.0 <= float(self.rho) <= 1.0:
            raise ValueError("rho must be in [0,1], got %r" % (self.rho,))
        if float(self.e_fast) < float(self.e_min) - 1e-12:
            raise ValueError("e_fast must be >= e_min (fastest plan cannot be cheapest)")

    @property
    def budget_j(self) -> float:
        return float(self.e_min) + float(self.rho) * (
            float(self.e_fast) - float(self.e_min)
        )

    def as_dict(self) -> dict[str, float]:
        return {"e_min": self.e_min, "e_fast": self.e_fast,
                "rho": self.rho, "budget_j": self.budget_j}


def candidate_panel_budget(
    panel: Sequence[tuple[float, float]], rho: float
) -> PanelBudget:
    """Build B_E(rho) from a panel of (makespan, energy_system) pairs."""
    if not panel:
        raise ValueError("empty candidate panel")
    energies = [float(e) for _t, e in panel]
    fastest = min(panel, key=lambda p: (float(p[0]), float(p[1])))
    return PanelBudget(
        e_min=min(energies), e_fast=float(fastest[1]), rho=float(rho)
    )
