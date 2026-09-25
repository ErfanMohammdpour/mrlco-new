"""V2V / energy constraints as a constrained-MDP layer (Lagrangian dual).

Motivation (measured — see `spec/energy_reward_audit.py`):

* the MOBILE boundary = UE + HELPER and MEC compute is free (ADR-001), so an
  all-MEC plan costs only ~3% of the all-UE plan.  A scalar 0.5/0.5 objective
  therefore does not express "keep the mobile devices' energy under a budget";
  it just tilts the search.
* V2V/HELPER help is not free either: helper CPU costs 0.7 W and helper radio is
  paid by the helper.  Nothing in the scalar objective stops the policy from
  draining the helper.

So the budgets live here, not in the weights.  Formulation:

    minimize   J(pi) = E[ 0.5*L_norm + 0.5*E_norm ]        (unchanged objective)
    subject to E[ c_i(pi) ] <= b_i          for each active constraint i

with a Lagrangian penalty  sum_i lambda_i * max(0, (c_i - b_i)/scale_i)  added to
the (already telescoped) token reward, and dual ascent on lambda at the OUTER
(meta) level:

    lambda_i <- max(0, lambda_i + eta * mean_batch( (c_i - b_i)/scale_i ))

Budgets are per-graph.  They may be absolute (joules / seconds) or expressed as a
fraction of an episode-local reference produced by
`energy_api.compute_reference_ranges` (E_ue, L_mec, ...), which keeps a budget
comparable across graphs of very different size.

Mode `off` (default) is a strict no-op: `enabled()` is False, no metrics are
measured, and `telescoping_token_rewards` must return bit-identical rewards.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from .energy_api import ReferenceRanges
from .energy_scope import (
    SCOPE_MOBILE,
    SCOPE_REQUESTER,
    SCOPE_SYSTEM,
    energy_scalar,
    require_reference_scope,
)
from .model import Location, ScheduleResult
from .resources import ResourceConfig
from .validate import require_finite, require_nonneg_float

CONSTRAINT_MODE_OFF = "off"
CONSTRAINT_MODE_LAGRANGIAN = "lagrangian"
CONSTRAINT_MODES = (CONSTRAINT_MODE_OFF, CONSTRAINT_MODE_LAGRANGIAN)

ATTRIBUTION_TERMINAL = "terminal"
ATTRIBUTION_TELESCOPED = "telescoped"
ATTRIBUTIONS = (ATTRIBUTION_TERMINAL, ATTRIBUTION_TELESCOPED)

# Active constraint names (stable order — the lambda vector follows this order).
C_UE_ENERGY = "ue_energy"
C_HELPER_ENERGY = "helper_energy"
C_TOTAL_ENERGY = "total_energy"
C_V2V_AIRTIME = "v2v_airtime"
C_V2V_TASK_FRACTION = "v2v_task_fraction"
C_MEC_TASK_FRACTION = "mec_task_fraction"
C_DEADLINE = "deadline"

ALL_CONSTRAINTS = (
    C_UE_ENERGY,
    C_HELPER_ENERGY,
    C_TOTAL_ENERGY,
    C_V2V_AIRTIME,
    C_V2V_TASK_FRACTION,
    C_MEC_TASK_FRACTION,
    C_DEADLINE,
)


def _opt_float(value: Any, name: str) -> float | None:
    if value is None:
        return None
    return require_finite(name, float(value))


@dataclass(frozen=True)
class ConstraintSpec:
    """Which constraints are active and what their budgets are.

    Absolute budgets:  ``ue_energy_budget_j``, ``helper_energy_budget_j``,
    ``total_energy_budget_j``, ``v2v_airtime_budget_s``, ``deadline_s``.
    Fractional budgets (episode-local reference):  ``*_frac_of_all_ue``,
    ``*_frac_of_makespan``, ``deadline_frac_of_all_mec``.
    A constraint is active iff at least one of its budget knobs is set.
    """

    mode: str = CONSTRAINT_MODE_OFF
    attribution: str = ATTRIBUTION_TERMINAL

    ue_energy_budget_j: float | None = None
    ue_energy_frac_of_all_ue: float | None = None

    helper_energy_budget_j: float | None = None
    helper_energy_frac_of_all_ue: float | None = None

    total_energy_budget_j: float | None = None
    total_energy_frac_of_all_ue: float | None = None

    v2v_airtime_budget_s: float | None = None
    v2v_airtime_frac_of_makespan: float | None = None

    v2v_task_fraction_max: float | None = None

    # MEC server capacity. Measured reason this exists: with MEC compute out of the
    # energy scope (ADR-001) an all-MEC plan is the argmin of BOTH latency and
    # mobile energy, so V2V/energy budgets alone are never binding — every
    # constraint is trivially satisfied by "send everything to MEC". A MEC-side
    # capacity budget is what makes the constrained problem non-degenerate.
    mec_task_fraction_max: float | None = None

    deadline_s: float | None = None
    deadline_frac_of_all_mec: float | None = None

    def __post_init__(self) -> None:
        if self.mode not in CONSTRAINT_MODES:
            raise ValueError(f"mode must be one of {CONSTRAINT_MODES}, got {self.mode!r}")
        if self.attribution not in ATTRIBUTIONS:
            raise ValueError(
                f"attribution must be one of {ATTRIBUTIONS}, got {self.attribution!r}"
            )
        for name in (
            "ue_energy_budget_j",
            "helper_energy_budget_j",
            "total_energy_budget_j",
            "v2v_airtime_budget_s",
            "deadline_s",
        ):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative, got {value}")
        for name in (
            "ue_energy_frac_of_all_ue",
            "helper_energy_frac_of_all_ue",
            "total_energy_frac_of_all_ue",
            "v2v_airtime_frac_of_makespan",
            "v2v_task_fraction_max",
            "mec_task_fraction_max",
            "deadline_frac_of_all_mec",
        ):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative, got {value}")

    # -- construction ------------------------------------------------------
    @classmethod
    def from_dict(cls, doc: dict[str, Any] | None) -> "ConstraintSpec":
        if not doc:
            return cls()
        known = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        unknown = sorted(set(doc) - known)
        if unknown:
            raise ValueError(f"unknown constraint keys: {unknown}")
        kwargs: dict[str, Any] = {}
        for key, value in doc.items():
            if key in ("mode", "attribution"):
                kwargs[key] = str(value)
            else:
                kwargs[key] = _opt_float(value, key)
        return cls(**kwargs)

    @classmethod
    def from_config(cls, energy_config: dict[str, Any] | None) -> "ConstraintSpec":
        cfg = energy_config or {}
        return cls.from_dict(cfg.get("constraints"))

    # -- queries -----------------------------------------------------------
    @property
    def enabled(self) -> bool:
        return self.mode == CONSTRAINT_MODE_LAGRANGIAN and bool(self.active_names)

    @property
    def active_names(self) -> tuple[str, ...]:
        active = []
        if self.ue_energy_budget_j is not None or self.ue_energy_frac_of_all_ue is not None:
            active.append(C_UE_ENERGY)
        if (
            self.helper_energy_budget_j is not None
            or self.helper_energy_frac_of_all_ue is not None
        ):
            active.append(C_HELPER_ENERGY)
        if (
            self.total_energy_budget_j is not None
            or self.total_energy_frac_of_all_ue is not None
        ):
            active.append(C_TOTAL_ENERGY)
        if (
            self.v2v_airtime_budget_s is not None
            or self.v2v_airtime_frac_of_makespan is not None
        ):
            active.append(C_V2V_AIRTIME)
        if self.v2v_task_fraction_max is not None:
            active.append(C_V2V_TASK_FRACTION)
        if self.mec_task_fraction_max is not None:
            active.append(C_MEC_TASK_FRACTION)
        if self.deadline_s is not None or self.deadline_frac_of_all_mec is not None:
            active.append(C_DEADLINE)
        return tuple(active)

    def constraint_status(self) -> dict[str, str]:
        """Explicit per-constraint status: `active` or `not_configured`.

        A constraint without a budget is NOT silently satisfied: it is reported
        as not_configured and contributes no violation/penalty, so the Lagrangian
        stays off for it.
        """
        active = set(self.active_names)
        return {
            name: ("active" if name in active else "not_configured")
            for name in ALL_CONSTRAINTS
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "attribution": self.attribution,
            "active": list(self.active_names),
            "constraint_status": self.constraint_status(),
            **{
                name: getattr(self, name)
                for name in self.__dataclass_fields__  # type: ignore[attr-defined]
                if getattr(self, name) is not None
            },
        }


@dataclass(frozen=True)
class ConstraintMetrics:
    """Plan-level quantities the constraints are expressed on."""

    ue_energy_j: float
    helper_energy_j: float
    total_energy_j: float
    helper_compute_j: float
    v2v_airtime_s: float
    v2v_task_fraction: float
    makespan_s: float
    n_tasks: int
    n_helper_tasks: int
    n_mec_tasks: int = 0

    def as_dict(self) -> dict[str, float]:
        return {
            "ue_energy_j": self.ue_energy_j,
            "helper_energy_j": self.helper_energy_j,
            "total_energy_j": self.total_energy_j,
            "helper_compute_j": self.helper_compute_j,
            "v2v_airtime_s": self.v2v_airtime_s,
            "v2v_task_fraction": self.v2v_task_fraction,
            "mec_task_fraction": self.mec_task_fraction,
            "makespan_s": self.makespan_s,
            "n_helper_tasks": float(self.n_helper_tasks),
            "n_mec_tasks": float(self.n_mec_tasks),
        }

    @property
    def mec_task_fraction(self) -> float:
        if not self.n_tasks:
            return 0.0
        return float(self.n_mec_tasks) / float(self.n_tasks)


def measure_metrics(result: ScheduleResult) -> ConstraintMetrics:
    """Read the plan-level metrics straight off a canonical `ScheduleResult`.

    Boundary quantities come from the scope accessor only: `c_ue` is the
    REQUESTER boundary, `c_total` the MOBILE boundary and helper work is the
    difference of the two. `helper_compute_j` is a component, not a boundary.
    """
    energy = result.energy
    requester = energy_scalar(result, scope=SCOPE_REQUESTER)
    mobile = energy_scalar(result, scope=SCOPE_MOBILE)
    system = energy_scalar(result, scope=SCOPE_SYSTEM)
    n_tasks = len(result.tasks)
    n_helper = sum(1 for rec in result.tasks.values() if rec.location == Location.HELPER)
    n_mec = sum(1 for rec in result.tasks.values() if rec.location == Location.MEC)
    airtime = 0.0
    for transfer in result.transfers:
        if transfer.hop == "V2V":
            airtime += float(transfer.end - transfer.start)
    return ConstraintMetrics(
        ue_energy_j=requester,
        helper_energy_j=max(0.0, mobile - requester),
        # E3.2: the "total energy" constraint is a SYSTEM consumer.
        total_energy_j=system,
        helper_compute_j=float(energy.helper_compute_joules),
        v2v_airtime_s=float(airtime),
        v2v_task_fraction=(float(n_helper) / float(n_tasks)) if n_tasks else 0.0,
        makespan_s=float(result.makespan_seconds),
        n_tasks=int(n_tasks),
        n_helper_tasks=int(n_helper),
        n_mec_tasks=int(n_mec),
    )


def _pick(absolute: float | None, fractional: float | None, reference: float, name: str) -> float:
    """Absolute budget wins; otherwise budget = fractional * reference."""
    if absolute is not None:
        return abs(absolute)
    if fractional is not None:
        return abs(fractional) * max(reference, 0.0)
    raise ValueError(f"{name}: no budget configured")


def budgets_and_scales(
    spec: ConstraintSpec,
    refs: ReferenceRanges,
    metrics: ConstraintMetrics,
) -> dict[str, tuple[float, float, float]]:
    """name -> (raw_metric, budget, scale) for every active constraint.

    Every constraint is a SYSTEM consumer (E3.2): the reference that anchors the
    fractional budgets and scales must be system-built. C_UE_ENERGY keeps the
    REQUESTER metric (it is the requester's own battery), but its all-UE anchor is
    scope-invariant.
    """
    require_reference_scope(refs, expected_scope=SCOPE_SYSTEM)
    out: dict[str, tuple[float, float, float]] = {}
    for name in spec.active_names:
        if name == C_UE_ENERGY:
            out[name] = (
                metrics.ue_energy_j,
                _pick(spec.ue_energy_budget_j, spec.ue_energy_frac_of_all_ue, refs.E_ue, name),
                max(refs.E_ue, 1e-12),
            )
        elif name == C_HELPER_ENERGY:
            out[name] = (
                metrics.helper_energy_j,
                _pick(
                    spec.helper_energy_budget_j,
                    spec.helper_energy_frac_of_all_ue,
                    refs.E_ue,
                    name,
                ),
                max(refs.E_ue, 1e-12),
            )
        elif name == C_TOTAL_ENERGY:
            out[name] = (
                metrics.total_energy_j,
                _pick(
                    spec.total_energy_budget_j,
                    spec.total_energy_frac_of_all_ue,
                    refs.E_ue,
                    name,
                ),
                max(refs.E_ue, 1e-12),
            )
        elif name == C_V2V_AIRTIME:
            out[name] = (
                metrics.v2v_airtime_s,
                _pick(
                    spec.v2v_airtime_budget_s,
                    spec.v2v_airtime_frac_of_makespan,
                    metrics.makespan_s,
                    name,
                ),
                max(metrics.makespan_s, 1e-12),
            )
        elif name == C_V2V_TASK_FRACTION:
            require_nonneg_float("v2v_task_fraction_max", float(spec.v2v_task_fraction_max))
            out[name] = (
                metrics.v2v_task_fraction,
                float(spec.v2v_task_fraction_max),
                1.0,
            )
        elif name == C_MEC_TASK_FRACTION:
            require_nonneg_float("mec_task_fraction_max", float(spec.mec_task_fraction_max))
            out[name] = (
                metrics.mec_task_fraction,
                float(spec.mec_task_fraction_max),
                1.0,
            )
        elif name == C_DEADLINE:
            out[name] = (
                metrics.makespan_s,
                _pick(spec.deadline_s, spec.deadline_frac_of_all_mec, refs.L_mec, name),
                max(refs.L_mec, 1e-12),
            )
        else:  # pragma: no cover - active_names is closed
            raise ValueError(f"unknown constraint {name!r}")
    return out


@dataclass(frozen=True)
class ConstraintCosts:
    """Per-episode constraint costs, already normalized by their scale."""

    names: tuple[str, ...] = ()
    raw: tuple[float, ...] = ()
    budgets: tuple[float, ...] = ()
    scales: tuple[float, ...] = ()
    signed: tuple[float, ...] = ()  # (raw - budget) / scale ; can be negative
    violations: tuple[float, ...] = ()  # max(0, signed)

    @property
    def active(self) -> bool:
        return bool(self.names)

    @property
    def total_violation(self) -> float:
        return float(sum(self.violations))

    def as_dict(self) -> dict[str, float]:
        out: dict[str, float] = {"total_violation": self.total_violation}
        for name, raw, budget, signed in zip(self.names, self.raw, self.budgets, self.signed):
            out[f"{name}_raw"] = float(raw)
            out[f"{name}_budget"] = float(budget)
            out[f"{name}_signed"] = float(signed)
        return out

    def penalty(self, lambdas: Sequence[float]) -> float:
        if len(lambdas) != len(self.violations):
            raise ValueError(
                f"lambda vector length {len(lambdas)} != constraints {len(self.violations)}"
            )
        return float(sum(float(l) * v for l, v in zip(lambdas, self.violations)))


EMPTY_COSTS = ConstraintCosts()


def costs_from_metrics(
    metrics: ConstraintMetrics,
    refs: ReferenceRanges,
    spec: ConstraintSpec | None,
) -> ConstraintCosts:
    """Constraint costs from already-measured metrics (no ScheduleResult needed)."""
    if spec is None or not spec.enabled:
        return EMPTY_COSTS
    table = budgets_and_scales(spec, refs, metrics)
    names = tuple(spec.active_names)
    raw = tuple(float(table[n][0]) for n in names)
    budgets = tuple(float(table[n][1]) for n in names)
    scales = tuple(float(table[n][2]) for n in names)
    signed = tuple((r - b) / s for r, b, s in zip(raw, budgets, scales))
    violations = tuple(max(0.0, v) for v in signed)
    return ConstraintCosts(
        names=names,
        raw=raw,
        budgets=budgets,
        scales=scales,
        signed=signed,
        violations=violations,
    )


def evaluate_constraints(
    result: ScheduleResult,
    resources: ResourceConfig,
    refs: ReferenceRanges,
    spec: ConstraintSpec | None,
    *,
    metrics: ConstraintMetrics | None = None,
) -> ConstraintCosts:
    """Plan-level constraint costs. Returns EMPTY_COSTS when inactive."""
    if spec is None or not spec.enabled:
        return EMPTY_COSTS
    if metrics is None:
        metrics = measure_metrics(result)
    return costs_from_metrics(metrics, refs, spec)


@dataclass
class ConstraintController:
    """Lagrangian dual variables + dual ascent at the outer (meta) loop.

    `observe()` is called by the environment once per trajectory with the
    realized `ConstraintCosts.signed`; `dual_step()` is called by the trainer
    once per outer iteration and consumes the buffer.
    """

    spec: ConstraintSpec
    dual_lr: float = 0.05
    max_lambda: float = 1e3
    _lambdas: list[float] = field(default_factory=list)
    _buffer: list[tuple[float, ...]] = field(default_factory=list)
    updates: int = 0

    def __post_init__(self) -> None:
        require_nonneg_float("dual_lr", float(self.dual_lr))
        if not self._lambdas:
            self._lambdas = [0.0] * len(self.spec.active_names)
        if len(self._lambdas) != len(self.spec.active_names):
            raise ValueError(
                f"initial lambdas {len(self._lambdas)} != active constraints "
                f"{len(self.spec.active_names)}"
            )

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self.spec.active_names)

    @property
    def lambdas(self) -> list[float]:
        return list(self._lambdas)

    def penalty(self, costs: ConstraintCosts) -> float:
        if not costs.active:
            return 0.0
        return costs.penalty(self._lambdas)

    def observe(self, costs: ConstraintCosts) -> None:
        if costs.active:
            self._buffer.append(tuple(float(v) for v in costs.signed))

    def reset_buffer(self) -> None:
        self._buffer = []

    def dual_step(self) -> dict[str, float]:
        """lambda <- max(0, lambda + lr * mean(signed)); returns diagnostics."""
        n = len(self._lambdas)
        if not self._buffer:
            return {"constraint/updates": float(self.updates), "constraint/buffer": 0.0}
        sums = [0.0] * n
        for row in self._buffer:
            for i, value in enumerate(row):
                sums[i] += value
        mean_signed = [s / len(self._buffer) for s in sums]
        for i in range(n):
            self._lambdas[i] = min(
                self.max_lambda, max(0.0, self._lambdas[i] + self.dual_lr * mean_signed[i])
            )
        self.updates += 1
        diag = {"constraint/updates": float(self.updates), "constraint/buffer": float(len(self._buffer))}
        for name, lam, ms in zip(self.names, self._lambdas, mean_signed):
            diag[f"constraint/lambda_{name}"] = float(lam)
            diag[f"constraint/mean_signed_{name}"] = float(ms)
        self.reset_buffer()
        return diag

    def state(self) -> dict[str, Any]:
        return {
            "spec": self.spec.as_dict(),
            "lambdas": {name: lam for name, lam in zip(self.names, self._lambdas)},
            "dual_lr": float(self.dual_lr),
            "updates": int(self.updates),
        }

    def status(self) -> dict[str, str]:
        """Per-constraint status, including the ones never configured."""
        return self.spec.constraint_status()
