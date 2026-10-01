#!/usr/bin/env python3
"""Truthful deadline / mixed-criticality constraint channels for MARGO-AUTOMOTIVE-MC-v1.

This module is the ONLY place where the primary-path constraint names live. It does
not import, read or enable `spec/constraints.yaml` (that file is a
`proposal_not_frozen` energy proposal), and it defines no energy budget by default.

Semantics (frozen by `spec/automotive_mc_v1/DATASET_CONTRACT.md` §8 and §9):

* ``D_G`` is the HARD graph-level requirement.  A schedule satisfies it when the
  graph makespan does not exceed it; the violation is
  ``max(0, makespan_s - D_G_s)``.
* ``d_i`` are FIRM per-task subdeadline targets (``E_i <= d_i <= L_i``).  A task
  misses a target when its availability -- ``all_consumers_ready``, i.e. arrival at
  EVERY required consumer, not compute finish -- exceeds ``d_i``.  The aggregate
  violation is the SUM of the per-task tardiness over the tasks of that criticality
  class, and the aggregate allowance (budget) is 0 s.
* ``HIGH``/``MEDIUM`` subdeadlines are FIRM.  They are deliberately NOT renamed to
  "hard": only ``D_G`` is hard.

The three channels are independent of ``tardiness_weight`` (a cost shape) and of the
energy axis; turning a lagrangian multiplier on for one of them never silently moves
another.

Energy: ``energy_constraint = "not_configured"``.  There is NO fake budget.  An
energy channel can only exist when a caller supplies an explicitly provenance-tagged
budget (non-empty provenance string + finite numeric budget); otherwise
``EnergyConstraintError`` is raised and nothing is enabled.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "CONSTRAINT_NAMES",
    "C_GRAPH_HARD_DEADLINE",
    "C_HI_TASK_TARDINESS",
    "C_MED_TASK_TARDINESS",
    "ENERGY_CONSTRAINT_NAME",
    "ENERGY_NOT_CONFIGURED",
    "energy_constraint",
    "EnergyConstraintError",
    "ConstraintSpec",
    "default_constraint_specs",
    "energy_spec",
    "specs_from_mapping",
    "evaluate_constraints",
    "AutomotiveDualController",
    "objective_breakdown",
]

# --------------------------------------------------------------------------- #
# Exact names (semantics first -- no misleading aliases)
# --------------------------------------------------------------------------- #
C_GRAPH_HARD_DEADLINE = "C_GRAPH_HARD_DEADLINE"
C_HI_TASK_TARDINESS = "C_HI_TASK_TARDINESS"
C_MED_TASK_TARDINESS = "C_MED_TASK_TARDINESS"

#: The primary-path constraint channels, in stable order.
CONSTRAINT_NAMES: tuple[str, ...] = (
    C_GRAPH_HARD_DEADLINE,
    C_HI_TASK_TARDINESS,
    C_MED_TASK_TARDINESS,
)

#: Energy is NOT one of the primary channels and has no budget of its own.
ENERGY_CONSTRAINT_NAME = "C_ENERGY_GLOBAL"
ENERGY_NOT_CONFIGURED = "not_configured"

#: Module-level truth: energy is not configured and no number is invented for it.
energy_constraint = ENERGY_NOT_CONFIGURED

HIGH = "HIGH"
MEDIUM = "MEDIUM"
LOW = "LOW"
CRITICALITY_CLASSES = (LOW, MEDIUM, HIGH)

HARD = "hard"
FIRM = "firm"

#: provenance labels -- everything either comes from the frozen dataset contract or
#: from an explicit caller-supplied provenance tag.
SOURCE_D_G = "frozen_dataset_contract:D_G_s"
SOURCE_SUBDEADLINE = "frozen_dataset_contract:d_i"
NOT_CONFIGURED_SOURCE = "not_configured"

TRAINING_SPLITS: tuple[str, ...] = ("meta_train", "train")
NON_TRAINING_SPLITS: tuple[str, ...] = ("validation", "meta_test")

_MISSING = object()


class EnergyConstraintError(ValueError):
    """Energy was asked for without an explicitly provenance-tagged budget."""


# --------------------------------------------------------------------------- #
# Small numeric guards (no NaN/Inf ever leaves this module)
# --------------------------------------------------------------------------- #
def _require_finite(value: Any, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{what} must be a real number, got {value!r}")
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{what} must be finite, got {value!r}")
    return out


def _get(obj: Any, name: str, default: Any = _MISSING) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


# --------------------------------------------------------------------------- #
# Spec
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ConstraintSpec:
    """One constraint channel: name, budget, kind, unit, enabled flag, provenance.

    ``budget`` semantics depend on ``kind``/``name``:

    * ``C_GRAPH_HARD_DEADLINE``: ``budget=None`` because the budget is the per-graph
      frozen ``D_G_s`` (see ``source``); ``evaluate_constraints`` reports it resolved.
    * ``C_HI_TASK_TARDINESS`` / ``C_MED_TASK_TARDINESS``: ``budget=0.0`` s, the
      aggregate allowance of a sum of non-negative tardiness terms.
    * energy: ``budget`` may only be set together with a non-empty provenance tag.
    """

    name: str
    budget: float | None
    kind: str
    unit: str
    enabled: bool = True
    source: str = ""

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise ValueError("ConstraintSpec.name must be a non-empty string")
        if self.kind not in (HARD, FIRM):
            raise ValueError(f"kind must be {HARD!r} or {FIRM!r}, got {self.kind!r}")
        if not str(self.unit).strip():
            raise ValueError("ConstraintSpec.unit must be a non-empty string")
        if self.budget is not None:
            budget = _require_finite(self.budget, f"{self.name}.budget")
            if budget < 0.0:
                raise ValueError(f"{self.name}.budget must be non-negative, got {budget}")
            object.__setattr__(self, "budget", budget)

    @property
    def status(self) -> str:
        """active | not_configured | disabled (never silently 'active')."""
        if self.enabled:
            return "active"
        if str(self.source).startswith(NOT_CONFIGURED_SOURCE):
            return "not_configured"
        return "disabled"


def default_constraint_specs() -> dict[str, ConstraintSpec]:
    """The three primary channels.  No energy channel is created."""
    return {
        C_GRAPH_HARD_DEADLINE: ConstraintSpec(
            name=C_GRAPH_HARD_DEADLINE,
            budget=None,
            kind=HARD,
            unit="s",
            enabled=True,
            source=SOURCE_D_G,
        ),
        C_HI_TASK_TARDINESS: ConstraintSpec(
            name=C_HI_TASK_TARDINESS,
            budget=0.0,
            kind=FIRM,
            unit="s",
            enabled=True,
            source=SOURCE_SUBDEADLINE,
        ),
        C_MED_TASK_TARDINESS: ConstraintSpec(
            name=C_MED_TASK_TARDINESS,
            budget=0.0,
            kind=FIRM,
            unit="s",
            enabled=True,
            source=SOURCE_SUBDEADLINE,
        ),
    }


def energy_spec(budget: Any = None, provenance: str | None = None, *,
                kind: str = FIRM, unit: str = "j") -> ConstraintSpec:
    """Build the energy channel -- or refuse to.

    * ``budget is None`` and no provenance: returns the ``not_configured`` spec (the
      default; enabled=False, budget=None, no invented number).
    * a budget without a non-empty provenance tag: raises ``EnergyConstraintError``.
    * both present: returns an enabled spec whose ``source`` is the provenance tag.
    """
    has_provenance = provenance is not None and bool(str(provenance).strip())
    if budget is None:
        if has_provenance:
            raise EnergyConstraintError(
                "energy provenance was supplied without a numeric budget; there is no "
                "budget to tag"
            )
        return ConstraintSpec(
            name=ENERGY_CONSTRAINT_NAME,
            budget=None,
            kind=kind,
            unit=unit,
            enabled=False,
            source=NOT_CONFIGURED_SOURCE,
        )
    if not has_provenance:
        raise EnergyConstraintError(
            "refusing to enable an energy constraint without provenance: pass a "
            "non-empty provenance string describing where the budget was measured/"
            "frozen (no fake budget is allowed)"
        )
    value = _require_finite(budget, "energy budget")
    if value < 0.0:
        raise EnergyConstraintError(f"energy budget must be non-negative, got {value}")
    return ConstraintSpec(
        name=ENERGY_CONSTRAINT_NAME,
        budget=value,
        kind=kind,
        unit=unit,
        enabled=True,
        source=f"provenance:{provenance}",
    )


def specs_from_mapping(doc: Mapping[str, Any]) -> dict[str, ConstraintSpec]:
    """Parse the (already-loaded) `constraints_automotive_v1.yaml` document.

    Kept pure: this module never opens the YAML itself, and never reads
    `spec/constraints.yaml`.
    """
    if not isinstance(doc, Mapping):
        raise ValueError("constraint document must be a mapping")
    specs: dict[str, ConstraintSpec] = {}
    for row in doc.get("constraints") or ():
        if not isinstance(row, Mapping) or "name" not in row:
            raise ValueError(f"malformed constraint row: {row!r}")
        budget = row.get("budget")
        if isinstance(budget, str) and budget.startswith("from_graph:"):
            budget = None  # resolved per graph from the frozen field
        specs[str(row["name"])] = ConstraintSpec(
            name=str(row["name"]),
            budget=budget,
            kind=str(row.get("kind", FIRM)),
            unit=str(row.get("unit", "s")),
            enabled=bool(row.get("enabled", True)),
            source=str(row.get("source", "")),
        )
    energy_doc = doc.get("energy_constraint", ENERGY_NOT_CONFIGURED)
    if energy_doc in (ENERGY_NOT_CONFIGURED, None, False):
        pass
    elif isinstance(energy_doc, Mapping) and energy_doc.get("enabled"):
        specs[ENERGY_CONSTRAINT_NAME] = energy_spec(
            energy_doc.get("budget"),
            energy_doc.get("provenance"),
            unit=str(energy_doc.get("unit", "j")),
        )
    else:
        raise EnergyConstraintError(
            "energy_constraint must be 'not_configured' or a mapping with an "
            f"enabled flag and explicit provenance, got {energy_doc!r}"
        )
    return specs


# --------------------------------------------------------------------------- #
# Evaluation against a real ScheduleResult (or an availability mapping)
# --------------------------------------------------------------------------- #
def _criticality_of(record: Any) -> str | None:
    """Result-side class (ScheduleResult TaskExecutionRecord.criticality_class)."""
    value = _get(record, "criticality_class", None)
    if value is None:
        return None
    text = str(value).strip().upper()
    return text if text in CRITICALITY_CLASSES else None


def _availability_of(record: Any, task_id: int) -> float:
    for attr in ("all_consumers_ready", "availability_seconds", "first_available", "finish"):
        value = _get(record, attr, None)
        if value is not None:
            return _require_finite(value, f"availability[{task_id}].{attr}")
    raise ValueError(f"task {task_id}: record exposes no availability time")


def _collect_availability(result: Any) -> tuple[dict[int, float], dict[int, Any]]:
    """-> (task_id -> availability seconds, task_id -> raw record).

    ``result`` may be a scheduler ``ScheduleResult`` (``.tasks`` mapping of records
    carrying ``all_consumers_ready``) or a plain mapping ``task_id -> seconds``.
    """
    if result is None:
        raise ValueError("result is required")
    if isinstance(result, Mapping) and "tasks" not in result:
        values = list(result.values())
        if not values:
            raise ValueError("availability mapping is empty; nothing to evaluate")
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
            return ({int(k): _require_finite(v, f"availability[{k}]")
                     for k, v in result.items()}, dict(result))
    tasks = _get(result, "tasks", None)
    if tasks is None:
        raise ValueError("result must expose .tasks or be a mapping task_id -> seconds")
    iterable = list(tasks.values()) if isinstance(tasks, Mapping) else list(tasks)
    avail: dict[int, float] = {}
    records: dict[int, Any] = {}
    for rec in iterable:
        tid = _get(rec, "task_id", None)
        if tid is None:
            raise ValueError(f"record without task_id: {rec!r}")
        tid = int(tid)
        avail[tid] = _availability_of(rec, tid)
        records[tid] = rec
    return avail, records


def _makespan_of(result: Any, avail: Mapping[int, float]) -> float:
    for attr in ("makespan_seconds", "makespan_s", "makespan"):
        value = _get(result, attr, None)
        if value is not None:
            return _require_finite(value, f"result.{attr}")
    if not avail:
        raise ValueError("cannot derive a makespan from an empty result")
    return max(avail.values())


def _deadline_of(graph_task: Any, record: Any, task_id: int) -> float:
    for source, attr in ((graph_task, "deadline_s"), (graph_task, "d_i"),
                         (record, "deadline_s"), (record, "d_i")):
        value = _get(source, attr, None)
        if value is not None:
            return _require_finite(value, f"task {task_id}.{attr}")
    raise ValueError(f"task {task_id}: no firm subdeadline d_i available")


def evaluate_constraints(graph: Any, result: Any) -> dict[str, dict[str, Any]]:
    """Evaluate the three primary channels for one (graph, schedule) pair.

    Returns ``{name: {"violation", "value", "budget", "n_violating_tasks"}}``.

    * hard graph channel: ``value = makespan_s``, ``budget = graph.D_G_s``,
      ``violation = max(0, value - budget)``, ``n_violating_tasks`` 1/0 because the
      requirement is graph-level (not a task count).
    * firm task channels: ``value = violation = sum_i max(0, availability_i - d_i)``
      over the tasks of that class, ``budget = 0.0`` s, ``n_violating_tasks`` counts
      the tasks that missed their own ``d_i``.
    """
    graph_tasks = _get(graph, "tasks", None)
    if graph_tasks is None:
        raise ValueError("graph must expose .tasks")
    graph_tasks = list(graph_tasks.values()) if isinstance(graph_tasks, Mapping) else list(graph_tasks)

    D_G = _get(graph, "D_G_s", None)
    if D_G is None:
        D_G = _get(graph, "D_G", None)
    if D_G is None:
        raise ValueError("graph must expose the frozen D_G_s")
    D_G = _require_finite(D_G, "graph.D_G_s")

    avail, records = _collect_availability(result)
    makespan = _makespan_of(result, avail)

    hard_violation = max(0.0, makespan - D_G)
    out: dict[str, dict[str, Any]] = {
        C_GRAPH_HARD_DEADLINE: {
            "violation": hard_violation,
            "value": makespan,
            "budget": D_G,
            "n_violating_tasks": 1 if hard_violation > 0.0 else 0,
        },
        C_HI_TASK_TARDINESS: {"violation": 0.0, "value": 0.0, "budget": 0.0,
                              "n_violating_tasks": 0},
        C_MED_TASK_TARDINESS: {"violation": 0.0, "value": 0.0, "budget": 0.0,
                               "n_violating_tasks": 0},
    }
    targets = {
        HIGH: out[C_HI_TASK_TARDINESS],
        MEDIUM: out[C_MED_TASK_TARDINESS],
    }

    for graph_task in graph_tasks:
        tid = _get(graph_task, "task_id", None)
        if tid is None:
            raise ValueError(f"graph task without task_id: {graph_task!r}")
        tid = int(tid)
        record = records.get(tid)
        if tid not in avail:
            raise ValueError(f"task {tid}: no availability reported by the schedule result")
        cls = None
        for candidate in (_get(graph_task, "criticality", None),
                          _get(graph_task, "criticality_class", None),
                          _criticality_of(record) if record is not None else None):
            if candidate is None:
                continue
            text = str(candidate).strip().upper()
            if text in CRITICALITY_CLASSES:
                cls = text
                break
        if cls is None:
            raise ValueError(f"task {tid}: no resolvable criticality class")
        if cls == LOW:
            continue
        firm = targets[cls]
        d_i = _deadline_of(graph_task, record, tid)
        tardiness = max(0.0, avail[tid] - d_i)
        firm["value"] += tardiness
        firm["violation"] += tardiness
        if tardiness > 0.0:
            firm["n_violating_tasks"] += 1

    for name in (C_HI_TASK_TARDINESS, C_MED_TASK_TARDINESS):
        out[name] = {
            "violation": float(out[name]["violation"]),
            "value": float(out[name]["value"]),
            "budget": 0.0,
            "n_violating_tasks": int(out[name]["n_violating_tasks"]),
        }
    out[C_GRAPH_HARD_DEADLINE] = {
        "violation": float(out[C_GRAPH_HARD_DEADLINE]["violation"]),
        "value": float(out[C_GRAPH_HARD_DEADLINE]["value"]),
        "budget": float(out[C_GRAPH_HARD_DEADLINE]["budget"]),
        "n_violating_tasks": int(out[C_GRAPH_HARD_DEADLINE]["n_violating_tasks"]),
    }
    return out


# --------------------------------------------------------------------------- #
# Lagrangian dual controller (ACTIVE, one projected ascent step per channel)
# --------------------------------------------------------------------------- #
class AutomotiveDualController:
    """Per-channel Lagrangian multipliers with a real, projected dual ascent step.

    ``dual_step`` applies exactly one ascent step per ENABLED channel using the MEAN
    violation of the observed batch::

        lambda <- clip(lambda + dual_lr * mean(batch_violations))

    It is a strict no-op for a channel whose batch is empty or whose batch mean is
    exactly zero, and it never fabricates a multiplier for a disabled or
    ``not_configured`` channel.  Training state only moves for
    ``split in {"meta_train", "train"}``.
    """

    def __init__(self, specs: Mapping[str, ConstraintSpec] | Sequence[ConstraintSpec] | None,
                 dual_lr: float, clip: tuple[float, float] = (0.0, 1e6)) -> None:
        self.specs: dict[str, ConstraintSpec] = _normalize_specs(specs)
        self.dual_lr = _require_finite(dual_lr, "dual_lr")
        if self.dual_lr < 0.0:
            raise ValueError(f"dual_lr must be non-negative, got {self.dual_lr}")
        lo = _require_finite(clip[0], "clip[0]")
        hi = _require_finite(clip[1], "clip[1]")
        if lo < 0.0:
            raise ValueError(f"clip lower bound must be non-negative, got {lo}")
        if hi < lo:
            raise ValueError(f"clip must satisfy lo <= hi, got {clip!r}")
        self.clip = (lo, hi)

        for name, spec in self.specs.items():
            if spec.enabled and name == ENERGY_CONSTRAINT_NAME:
                if spec.budget is None or not str(spec.source).startswith("provenance:"):
                    raise EnergyConstraintError(
                        "refusing an enabled energy constraint without an explicit "
                        "provenance-tagged budget"
                    )

        self.lambdas: dict[str, float] = {
            name: lo for name, spec in self.specs.items() if spec.enabled
        }
        self._batch: dict[str, list[float]] = {name: [] for name in self.lambdas}
        self._last_mean: dict[str, float | None] = {name: None for name in self.lambdas}
        self._updates: dict[str, int] = {name: 0 for name in self.lambdas}
        self.rejected_splits: dict[str, int] = {}
        self.ignored_names: dict[str, int] = {}
        self.last_rejected_split: str | None = None

    # -- observation -------------------------------------------------------
    def observe(self, costs: Mapping[str, Any], split: str = "meta_train") -> dict[str, Any]:
        """Record one episode's violations; returns what was accepted.

        ``costs`` maps a channel name to a violation (float or list of floats).  A
        non-finite value raises.  A non-training ``split`` is refused: nothing is
        recorded, the refusal is counted in ``rejected_splits`` and ``as_dict``.
        """
        if not isinstance(costs, Mapping):
            raise ValueError(f"costs must be a mapping name -> violation, got {type(costs)!r}")
        cleaned: dict[str, list[float]] = {}
        for name, value in costs.items():
            key = str(name)
            if isinstance(value, (list, tuple)):
                values: Iterable[Any] = value
            else:
                values = (value,)
            cleaned[key] = [_require_finite(v, f"violation[{key}]") for v in values]

        split_key = str(split)
        if split_key not in TRAINING_SPLITS:
            self.rejected_splits[split_key] = self.rejected_splits.get(split_key, 0) + 1
            self.last_rejected_split = split_key
            return {"accepted": False, "split": split_key, "recorded": [],
                    "reason": "split is not a training split"}

        recorded = []
        for name, values in cleaned.items():
            if name not in self._batch:
                self.ignored_names[name] = self.ignored_names.get(name, 0) + 1
                continue
            self._batch[name].extend(values)
            recorded.append(name)
        return {"accepted": True, "split": split_key, "recorded": recorded}

    def reset_batch(self) -> None:
        """Drop the observed batch (multipliers and counters are kept)."""
        for name in self._batch:
            self._batch[name] = []

    def batch_size(self, name: str) -> int:
        return len(self._batch.get(name, ()))

    # -- dual ascent -------------------------------------------------------
    def dual_step(self) -> dict[str, Any]:
        """One projected dual ASCENT step per enabled channel (mean over the batch)."""
        means: dict[str, float | None] = {}
        deltas: dict[str, float] = {}
        updated: list[str] = []
        skipped: list[str] = []
        for name, spec in self.specs.items():
            if not spec.enabled:
                skipped.append(name)
                continue
            batch = self._batch.get(name, [])
            if not batch:
                self._last_mean[name] = None
                means[name] = None
                continue
            mean = math.fsum(batch) / float(len(batch))
            if not math.isfinite(mean):
                raise ValueError(f"non-finite batch mean for {name}")
            self._last_mean[name] = mean
            means[name] = mean
            if mean == 0.0:
                continue  # strict no-op: no multiplier movement on a satisfied batch
            previous = self.lambdas[name]
            lo, hi = self.clip
            updated_value = min(max(previous + self.dual_lr * mean, lo), hi)
            if not math.isfinite(updated_value):
                raise ValueError(f"non-finite multiplier update for {name}")
            self.lambdas[name] = updated_value
            deltas[name] = updated_value - previous
            if updated_value != previous:
                self._updates[name] += 1
                updated.append(name)
        return {"means": means, "deltas": deltas, "updated": updated, "skipped": skipped,
                "lambdas": dict(self.lambdas)}

    def penalty(self, violations: Mapping[str, Any]) -> float:
        """sum_name lambda_name * violation_name over the ENABLED channels only."""
        if not isinstance(violations, Mapping):
            raise ValueError("violations must be a mapping name -> violation")
        terms = []
        for name, lam in self.lambdas.items():
            value = violations.get(name, 0.0)
            if value is None:
                value = 0.0
            terms.append(lam * _require_finite(value, f"violation[{name}]"))
        total = float(math.fsum(terms))
        if not math.isfinite(total):
            raise ValueError("penalty is not finite")
        return total

    # -- introspection -----------------------------------------------------
    def as_dict(self) -> dict[str, Any]:
        constraints: dict[str, dict[str, Any]] = {}
        for name, spec in self.specs.items():
            active = spec.enabled
            constraints[name] = {
                "lambda": float(self.lambdas[name]) if active else None,
                "last_mean_violation": self._last_mean.get(name),
                "updates": int(self._updates.get(name, 0)),
                "status": spec.status,
                "kind": spec.kind,
                "unit": spec.unit,
                "budget": spec.budget,
                "source": spec.source,
                "batch_size": len(self._batch.get(name, ())),
            }
        return {
            "dual_lr": self.dual_lr,
            "clip": [self.clip[0], self.clip[1]],
            "lambdas": dict(self.lambdas),
            "constraints": constraints,
            "rejected_splits": dict(self.rejected_splits),
            "ignored_names": dict(self.ignored_names),
            "energy_constraint": ENERGY_NOT_CONFIGURED,
        }


def _normalize_specs(specs: Mapping[str, ConstraintSpec] | Sequence[ConstraintSpec] | None
                     ) -> dict[str, ConstraintSpec]:
    if specs is None:
        return default_constraint_specs()
    if isinstance(specs, Mapping):
        items = list(specs.items())
    else:
        items = [(spec.name, spec) for spec in specs]
    out: dict[str, ConstraintSpec] = {}
    for key, spec in items:
        if not isinstance(spec, ConstraintSpec):
            raise TypeError(f"specs must contain ConstraintSpec values, got {type(spec)!r}")
        if str(key) != spec.name:
            raise ValueError(f"spec key {key!r} does not match spec.name {spec.name!r}")
        out[spec.name] = spec
    return out


# --------------------------------------------------------------------------- #
# Objective separation
# --------------------------------------------------------------------------- #
def objective_breakdown(latency_only_objective: float,
                        violations: Mapping[str, Any],
                        controller: AutomotiveDualController | None = None,
                        *,
                        lambdas: Mapping[str, float] | None = None) -> dict[str, Any]:
    """Keep the latency objective and the constraint penalty in DISTINCT keys.

    ``constraint_penalty`` is never folded into ``latency_only_objective``; the two are
    combined only in the separate ``penalized_objective`` key.
    """
    latency = _require_finite(latency_only_objective, "latency_only_objective")
    if not isinstance(violations, Mapping):
        raise ValueError("violations must be a mapping name -> violation")

    if controller is not None:
        if not isinstance(controller, AutomotiveDualController):
            raise TypeError("controller must be an AutomotiveDualController")
        lambda_map = {name: float(value) for name, value in controller.lambdas.items()}
        penalty = controller.penalty(violations)
    else:
        lambda_map = {str(k): _require_finite(v, f"lambda[{k}]")
                      for k, v in dict(lambdas or {}).items()}
        penalty = float(math.fsum(
            lam * _require_finite(violations.get(name, 0.0), f"violation[{name}]")
            for name, lam in lambda_map.items()
        ))

    names = list(lambda_map) + [str(n) for n in violations if str(n) not in lambda_map]
    reported_violations = {
        name: (_require_finite(violations[name], f"violation[{name}]")
               if name in violations else 0.0)
        for name in names
    }
    reported_lambdas = {name: float(lambda_map.get(name, 0.0)) for name in names}

    penalized = latency + penalty
    if not math.isfinite(penalized):
        raise ValueError("penalized_objective is not finite")
    return {
        "latency_only_objective": latency,
        "constraint_violations": reported_violations,
        "constraint_lambdas": reported_lambdas,
        "constraint_penalty": penalty,
        "penalized_objective": penalized,
    }
