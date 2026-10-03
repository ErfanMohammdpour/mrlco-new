#!/usr/bin/env python3
"""v2 constraints: the SAME Lagrangian formulation, controller and cost object as v1.

There is exactly one constraint implementation. This module only *measures* v2 quantities
and feeds them into `env/mec_offloaing_envs/scheduler/constraints.py`:

    g_i        = (raw_i - budget_i) / scale_i          # signed, can be negative
    violation_i= max(0, g_i)                           # reporting only
    penalty    = sum_i lambda_i * violation_i          # added ONCE, on the terminal token
    lambda_i  <- clip(lambda_i + eta * mean_train(g_i), 0, lambda_max)   # SIGNED dual ascent

Why the separation matters (audited defects):
* raw / budget / signed / violation are kept in four distinct arrays — a selector that reads
  a missing violation must fail, never see a fabricated zero;
* dual ascent uses the SIGNED cost, so lambda can DECREASE when a constraint is satisfied
  (updating only the positive hinge makes lambda monotonically non-decreasing);
* the penalty is applied once per episode (attribution="terminal"), not once per token;
* the reference ranges used for fractional budgets and scales are measured on the V2
  scheduler itself and must carry the same energy scope and scheduler fingerprint.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from env.mec_offloaing_envs.scheduler.constraints import (
    ALL_CONSTRAINTS, ATTRIBUTION_TERMINAL, CONSTRAINT_MODE_LAGRANGIAN, CONSTRAINT_MODE_OFF,
    EMPTY_COSTS, ConstraintController, ConstraintCosts, ConstraintMetrics, ConstraintSpec,
    budgets_and_scales, costs_from_metrics,
)
from env.mec_offloaing_envs.scheduler.energy_api import ReferenceRanges
from env.mec_offloaing_envs.scheduler.energy_scope import (
    SCOPE_MOBILE, SCOPE_REQUESTER, SCOPE_SYSTEM, require_reference_scope,
)

from spec.automotive_training.v2.energy import V2EnergyLedger
from spec.automotive_training.v2.shared_scheduler import V2ScheduleResult


class V2ConstraintError(RuntimeError):
    """Raised when a constraint cannot be evaluated from real, in-scope measurements."""


def v2_metrics(result: V2ScheduleResult, ledger: V2EnergyLedger) -> ConstraintMetrics:
    """Plan-level metrics for a v2 episode, measured from the event ledgers."""
    requester = float(ledger.requester_joules)
    mobile = float(ledger.mobile_joules)
    system = float(ledger.system_joules)
    n_tasks = len(result.timings)
    n_helper = sum(1 for tm in result.timings.values() if tm.location == "HELPER")
    n_mec = sum(1 for tm in result.timings.values() if tm.location == "MEC")
    airtime = sum(float(r["service_s"]) for r in result.radio_ledger if r["hop"] == "V2V")
    return ConstraintMetrics(
        ue_energy_j=requester,
        helper_energy_j=max(0.0, mobile - requester),
        total_energy_j=system,
        helper_compute_j=float(ledger.breakdown.helper_compute_joules),
        v2v_airtime_s=float(airtime),
        v2v_task_fraction=(float(n_helper) / float(n_tasks)) if n_tasks else 0.0,
        makespan_s=float(result.makespan_s),
        n_tasks=int(n_tasks),
        n_helper_tasks=int(n_helper),
        n_mec_tasks=int(n_mec))


def require_measurable(metrics: ConstraintMetrics, *, context: str = "") -> None:
    """Every mandatory metric must be a finite non-negative number.

    A missing or non-finite metric is an ERROR, never a zero violation: silently treating an
    unmeasured constraint as satisfied is exactly how a checkpoint selector ends up choosing
    an infeasible plan while reporting full feasibility.
    """
    for name in ("ue_energy_j", "helper_energy_j", "total_energy_j", "helper_compute_j",
                 "v2v_airtime_s", "v2v_task_fraction", "makespan_s"):
        value = getattr(metrics, name)
        if value is None:
            raise V2ConstraintError(
                "constraint metric %s is MISSING%s; refusing to treat it as a zero "
                "violation" % (name, (" in " + context) if context else ""))
        value = float(value)
        if not math.isfinite(value):
            raise V2ConstraintError("constraint metric %s is not finite: %r%s"
                                    % (name, value, (" in " + context) if context else ""))
        if value < 0.0:
            raise V2ConstraintError("constraint metric %s is negative: %r%s"
                                    % (name, value, (" in " + context) if context else ""))
    if int(metrics.n_tasks) <= 0:
        raise V2ConstraintError("constraint metrics report no tasks: denominators are "
                                "undefined%s" % ((" in " + context) if context else ""))


def v2_constraint_costs(result: V2ScheduleResult, ledger: V2EnergyLedger,
                        spec: ConstraintSpec | None,
                        refs: ReferenceRanges | None,
                        *, scheduler_config_sha256: str | None = None) -> ConstraintCosts:
    """Signed/normalized costs for every active constraint (EMPTY when inactive)."""
    if spec is None or not spec.enabled:
        return EMPTY_COSTS
    if refs is None:
        raise V2ConstraintError(
            "constraints are enabled but no reference ranges were supplied: fractional "
            "budgets and scales cannot be measured")
    expected = str(getattr(spec, "reference_scope", "") or SCOPE_SYSTEM)
    require_reference_scope(refs, expected_scope=expected,
                            expected_scheduler_config_sha256=scheduler_config_sha256)
    metrics = v2_metrics(result, ledger)
    require_measurable(metrics, context="v2_constraint_costs")
    return costs_from_metrics(metrics, refs, spec)


def constraints_fingerprint(spec: ConstraintSpec, scope: str | None = None) -> str:
    """Fingerprint of the constraint SPECIFICATION and its declared reference scope.

    Deliberately depends only on CONFIGURATION, never on a live `ReferenceRanges` object: the
    object is rebuilt per episode, so including it made a checkpoint fail to restore its own
    dual state (the live scope read back as None before the first evaluation).
    """
    scope = str(scope or getattr(spec, "reference_scope", None) or SCOPE_SYSTEM)
    payload = {"spec": spec.as_dict() if spec is not None else None,
               "reference_scope": scope}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def spec_from_config(energy_config: Mapping[str, Any] | None,
                     *, mode: str = CONSTRAINT_MODE_LAGRANGIAN,
                     reference_scope: str = SCOPE_SYSTEM) -> ConstraintSpec:
    """Build the v2 constraint spec from the frozen `energy.constraints` block."""
    spec = ConstraintSpec.from_config(dict(energy_config or {}))
    if mode == CONSTRAINT_MODE_OFF:
        return ConstraintSpec(mode=CONSTRAINT_MODE_OFF)
    if not spec.enabled:
        return ConstraintSpec(mode=CONSTRAINT_MODE_OFF)
    import dataclasses

    return dataclasses.replace(spec, mode=CONSTRAINT_MODE_LAGRANGIAN,
                               attribution=ATTRIBUTION_TERMINAL)


@dataclass
class V2ConstraintManager:
    """Per-worker dual state: hold lambda fixed inside a rollout, update once per iteration."""

    spec: ConstraintSpec
    references: ReferenceRanges | None = None
    dual_lr: float = 0.05
    max_lambda: float = 1e3
    scheduler_config_sha256: str | None = None
    controller: ConstraintController | None = None
    applied_penalty: float = 0.0
    last_costs: ConstraintCosts = EMPTY_COSTS
    history: list = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.controller is None:
            self.controller = ConstraintController(spec=self.spec, dual_lr=float(self.dual_lr),
                                                   max_lambda=float(self.max_lambda))

    # -- properties --------------------------------------------------------
    @property
    def enabled(self) -> bool:
        return bool(self.spec.enabled)

    @property
    def names(self) -> tuple:
        return tuple(self.controller.names) if self.controller else ()

    @property
    def lambdas(self) -> list:
        return list(self.controller.lambdas) if self.controller else []

    def lambdas_by_name(self) -> dict:
        return {name: float(lam) for name, lam in zip(self.names, self.lambdas)}

    def set_lambdas(self, values: Mapping[str, float] | Sequence[float]) -> None:
        """Broadcast an externally-held lambda vector (checkpoint restore / trainer)."""
        if isinstance(values, Mapping):
            unknown = sorted(set(map(str, values)) - set(self.names))
            if unknown:
                raise V2ConstraintError(
                    "lambda broadcast carries unknown constraints %s (active: %s)"
                    % (unknown, list(self.names)))
            vector = [float(values.get(name, 0.0)) for name in self.names]
        else:
            vector = [float(v) for v in values]
        if len(vector) != len(self.names):
            raise V2ConstraintError("lambda vector length %d != constraints %d"
                                    % (len(vector), len(self.names)))
        for value in vector:
            if not math.isfinite(value) or value < 0.0:
                raise V2ConstraintError("lambda must be finite and >= 0, got %r" % (value,))
        self.controller._lambdas = vector       # held fixed for the whole rollout

    # -- per-episode -------------------------------------------------------
    def evaluate(self, result: V2ScheduleResult,
                 ledger: V2EnergyLedger) -> ConstraintCosts:
        costs = v2_constraint_costs(result, ledger, self.spec, self.references,
                                    scheduler_config_sha256=self.scheduler_config_sha256)
        self.last_costs = costs
        return costs

    def apply_penalty(self, rewards: Sequence[float],
                      costs: ConstraintCosts) -> float:
        """Add `-penalty` to the TERMINAL token only. Returns the applied penalty.

        The episode cost is charged exactly once: charging it on every token would multiply
        it by the token count, and charging it on the first token would break the
        `sum(rewards) == -latency_improvement - penalty` identity that the returns rely on.
        """
        penalty = float(self.controller.penalty(costs)) if costs.active else 0.0
        self.applied_penalty = penalty
        if penalty and len(rewards):
            rewards[-1] = float(rewards[-1]) - penalty
        return penalty

    def observe(self, costs: ConstraintCosts) -> None:
        self.controller.observe(costs)

    def dual_step(self) -> dict:
        diag = self.controller.dual_step()
        self.history.append({k: float(v) for k, v in diag.items()})
        return diag

    # -- persistence -------------------------------------------------------
    def state(self) -> dict:
        return {
            "constraints": self.controller.state(),
            "scheduler_config_sha256": self.scheduler_config_sha256,
            "constraints_sha256": constraints_fingerprint(
                self.spec, getattr(self.references, "energy_scope", None)),
            "reference_scope": str(getattr(self.references, "energy_scope", None)
                                   or SCOPE_SYSTEM),
        }

    def load_state(self, state: Mapping[str, Any]) -> None:
        doc = dict(state or {})
        cstate = doc.get("constraints") or {}
        lambdas = cstate.get("lambdas")
        if lambdas:
            self.set_lambdas(lambdas)
        self.controller.updates = int(cstate.get("updates", 0))
        expected = doc.get("constraints_sha256")
        if expected and expected != constraints_fingerprint(
                self.spec, doc.get("reference_scope")):
            raise V2ConstraintError(
                "constraint specification/normalisation changed since the checkpoint "
                "(expected %s): refusing to restore a mismatched dual state" % expected)

    # -- telemetry ---------------------------------------------------------
    def telemetry(self, costs: ConstraintCosts | None = None) -> dict:
        """Top-level, selector-readable constraint record with denominators.

        Everything a checkpoint selector needs is present and explicit:
        raw, budget, signed, violation, lambda — plus the task denominators.
        """
        costs = costs if costs is not None else self.last_costs
        out: dict[str, Any] = {
            "enabled": bool(self.enabled),
            "names": list(costs.names),
            "lambdas": self.lambdas_by_name(),
            "penalty_applied": float(self.applied_penalty),
            "total_violation": float(costs.total_violation),
            "updates": int(self.controller.updates),
        }
        for i, name in enumerate(costs.names):
            out["%s_raw" % name] = float(costs.raw[i])
            out["%s_budget" % name] = float(costs.budgets[i])
            out["%s_scale" % name] = float(costs.scales[i])
            out["%s_signed" % name] = float(costs.signed[i])
            out["%s_violation" % name] = float(costs.violations[i])
            out["%s_lambda" % name] = float(self.lambdas[i])
        return out


#: constraint name -> ConstraintSpec knob for fractional budgets
FRACTION_KNOBS = {
    "ue_energy": "ue_energy_frac_of_all_ue",
    "helper_energy": "helper_energy_frac_of_all_ue",
    "total_energy": "total_energy_frac_of_all_ue",
}


def spec_from_fractions(fractions: Mapping[str, float] | None,
                        *, mode: str = CONSTRAINT_MODE_LAGRANGIAN) -> ConstraintSpec:
    """Constraint spec from FRACTIONAL budgets alone.

    Fractional budgets need no reference at construction time: the reference is resolved from
    the episode's own pure-location plans when the cost is evaluated. Absolute budgets are
    also accepted by `ConstraintSpec` for callers that have a physical joule figure.
    """
    kwargs: dict[str, Any] = {"mode": mode, "attribution": ATTRIBUTION_TERMINAL}
    for name, value in dict(fractions or {}).items():
        knob = FRACTION_KNOBS.get(str(name))
        if knob is None:
            raise V2ConstraintError("unknown fractional constraint %r (known: %s)"
                                    % (name, sorted(FRACTION_KNOBS)))
        value = float(value)
        if not math.isfinite(value) or value < 0.0:
            raise V2ConstraintError("budget fraction %s must be finite and >= 0" % name)
        kwargs[knob] = value
    return ConstraintSpec(**kwargs)


def calibrate_budgets(refs: ReferenceRanges, fractions: Mapping[str, float],
                      *, reference_scope: str = SCOPE_SYSTEM) -> ConstraintSpec:
    """TRAIN-ONLY budget calibration: fractional budgets anchored on the plan references.

    Budgets are expressed as a fraction of the pure-location plan reference, so they scale
    with the graph and never need a validation/meta-test statistic.
    """
    require_reference_scope(refs, expected_scope=reference_scope)
    spec = spec_from_fractions(fractions)
    if not spec.enabled:
        raise V2ConstraintError(
            "budget calibration produced no ACTIVE constraint: a constraint channel that can "
            "never bind must not be presented as a constraint")
    return spec


def select_checkpoint(records: Sequence[Mapping[str, Any]], *,
                      objective_key: str = "objective",
                      feasibility_key: str = "feasible",
                      violation_key: str = "total_violation") -> dict:
    """Feasibility-aware checkpoint selection.

    Order (frozen):
      1. hard feasibility first — a record is feasible only when its recorded
         `feasible` flag is True AND its total violation is <= 0 (within tolerance);
      2. among the feasible records, the best objective (lower is better);
      3. when NO record is feasible, return the best infeasible one and LABEL it
         `"feasible": False` — never present an infeasible checkpoint as a feasible winner.

    A record missing the objective or the violation is an ERROR: reading a missing metric as
    zero is exactly how an infeasible checkpoint used to look feasible.
    """
    if not records:
        raise V2ConstraintError("no checkpoint records to select from")
    scored = []
    for i, rec in enumerate(records):
        if objective_key not in rec:
            raise V2ConstraintError(
                "checkpoint record %d has no %r; refusing to select on a missing objective"
                % (i, objective_key))
        if violation_key not in rec:
            raise V2ConstraintError(
                "checkpoint record %d has no %r; refusing to treat an unmeasured constraint "
                "as satisfied" % (i, violation_key))
        objective = float(rec[objective_key])
        violation = float(rec[violation_key])
        if not math.isfinite(objective) or not math.isfinite(violation):
            raise V2ConstraintError("non-finite checkpoint metric in record %d" % i)
        feasible = bool(rec.get(feasibility_key, False)) and violation <= 1e-9
        scored.append((feasible, objective, violation, i, dict(rec)))
    feasible_rows = [row for row in scored if row[0]]
    if feasible_rows:
        _f, objective, _v, i, rec = min(feasible_rows, key=lambda r: (r[1], r[3]))
        out = dict(rec)
        out.update({"selected_index": i, "feasible": True, "objective": objective,
                    "total_violation": scored[i][2]})
        return out
    _f, objective, violation, i, rec = min(scored, key=lambda r: (r[1], r[3]))
    out = dict(rec)
    out.update({"selected_index": i, "feasible": False, "objective": objective,
                "total_violation": violation,
                "selection_note": "NO feasible checkpoint: best infeasible retained and "
                                  "labelled infeasible"})
    return out


__all__ = [
    "ALL_CONSTRAINTS", "SCOPE_MOBILE", "SCOPE_REQUESTER", "SCOPE_SYSTEM",
    "V2ConstraintError", "V2ConstraintManager", "calibrate_budgets",
    "constraints_fingerprint", "require_measurable", "select_checkpoint",
    "spec_from_fractions",
    "spec_from_config",
    "v2_constraint_costs", "v2_metrics",
]
