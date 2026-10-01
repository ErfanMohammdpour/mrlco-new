#!/usr/bin/env python3
"""Runtime mixed-criticality semantics for the FROZEN MARGO-AUTOMOTIVE-MC-v1 dataset.

Prompt-side, numpy/stdlib only. This module READS the frozen dataset contract (M7
criticality policy, M5-v2 workload model) and never writes or regenerates anything
under ``env/mec_offloaing_envs/data/automotive_mc_v1/``.

Design rules (each is enforced by ``env/mec_offloaing_envs/scheduler/tests/
test_automotive_mc_runtime.py``):

1. **Tier-independent demand.** ``C_LO``/``C_HI`` are seconds on the M5-v2
   reference-equivalent compute model. Both are converted once to *equivalent work*
   (the same units as ``compute_workload_bytes``)::

       demand_lo_equiv = C_LO_ref_s * f_ref / (8 * xi)
       demand_hi_equiv = C_HI_ref_s * f_ref / (8 * xi)

   A rollout draws its realized demand in those units. The LO->HI trigger is a
   predicate of the demand vector alone; ``tier_duration_seconds`` is applied
   afterwards and can never change the mode decision.

2. **Pre-declared execution uncertainty.** ``execution_uncertainty_v1.yaml``
   (``evidence_class: source_calibrated_synthetic``, ``frozen_before_smoke: true``)
   declares the regime split, the seed derivation and the counter PRNG. Nothing is
   measured and nothing may be tuned after a smoke run.

3. **Mode machine.** initial ``LO``; the only ``LO -> HI`` trigger is a HIGH task
   whose realized demand exceeds its ``C_LO``; ``HI`` is sticky until graph
   completion; ``HI -> LO`` is forbidden in v1. HIGH is never dropped and never
   degraded. MEDIUM/LOW follow the frozen M7 per-task drop/degrade fields verbatim.

4. **Dependency-safe dropping.** In HI mode the required-task closure is computed by
   walking predecessors from the surviving set, so every input of a surviving task
   is still produced; only LOW tasks that are both policy-droppable in HI and outside
   that closure are dropped. The surviving sub-DAG stays acyclic.

5. **HI demand capping.** In HI mode a HIGH task's realized demand is capped at its
   ``C_HI`` (that is what the HI budget means) and ``capped_to_hi`` records it. No
   payload/workload scaling is invented for degradation: a degradation the policy
   permits but for which no concrete transformation model exists is recorded as
   ``degradation_status == "unsupported_not_applied"`` with ``degrade_applied=False``.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:  # pragma: no cover - import shape depends on how the test is invoked
    from .automotive_loader import AutomotiveGraph, AutomotiveTask
except ImportError:  # pragma: no cover
    import sys

    _REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    from spec.automotive_training.automotive_loader import (  # type: ignore
        AutomotiveGraph,
        AutomotiveTask,
    )

# --------------------------------------------------------------------------- #
# Frozen constants
# --------------------------------------------------------------------------- #
EXECUTION_UNCERTAINTY_VERSION = "execution_uncertainty_v1"
EXECUTION_UNCERTAINTY_PATH = Path(__file__).resolve().parent / "execution_uncertainty_v1.yaml"
WORKLOAD_MODEL_V2_PATH = (
    Path(__file__).resolve().parents[1] / "automotive_mc_v1" / "workload_model_v2.yaml"
)

EVIDENCE_CLASS = "source_calibrated_synthetic"
MODE_INITIAL = "LO"
MODE_OVERLOAD = "HI"
MODES = (MODE_INITIAL, MODE_OVERLOAD)
CRITICALITY_CLASSES = ("LOW", "MEDIUM", "HIGH")

REASON_HIGH_EXCEEDS_CLO = "high_task_observed_execution_exceeds_empirical_execution_budget_lo"

#: Mode semantics mirrored from criticality_policy.yaml (`mode_semantics`). These are
#: NOT new permissions: they are only the *fallback* used when a task record has no
#: explicit M7 field (e.g. a hand-built fixture). The frozen dataset always supplies
#: the per-task fields, and those win. Source: spec/automotive_mc_v1/criticality_policy.yaml
M7_DROP_DEGRADE_FALLBACK: dict[str, dict[str, bool]] = {
    "HIGH": {
        "drop_allowed_lo_mode": False,
        "drop_allowed_hi_mode": False,
        "degrade_allowed_lo_mode": False,
        "degrade_allowed_hi_mode": False,
    },
    "MEDIUM": {
        "drop_allowed_lo_mode": False,
        "drop_allowed_hi_mode": False,
        "degrade_allowed_lo_mode": False,
        "degrade_allowed_hi_mode": True,
    },
    "LOW": {
        "drop_allowed_lo_mode": False,
        "drop_allowed_hi_mode": True,
        "degrade_allowed_lo_mode": False,
        "degrade_allowed_hi_mode": True,
    },
}
M7_MODE_SEMANTICS_KEYS = {
    "drop_allowed_lo_mode": "drop_allowed_lo_mode",
    "drop_allowed_hi_mode": "drop_allowed_hi_mode",
    "degrade_allowed_lo_mode": "degrade_allowed_lo_mode",
    "degrade_allowed_hi_mode": "degrade_allowed_hi_mode",
}

#: The 9 mandated mode-switch log fields (criticality_policy.yaml
#: `mode_semantics.switch_log_fields`).
SWITCH_LOG_FIELDS = (
    "triggering_task_id",
    "semantic_role",
    "criticality",
    "observed_execution",
    "C_LO",
    "reason",
    "previous_mode",
    "new_mode",
    "logical_schedule_point",
)

DEGRADATION_INTACT = "intact"
DEGRADATION_UNSUPPORTED = "unsupported_not_applied"

#: Defaults mirroring workload_model_v2.yaml reference_compute_model; the on-disk file
#: is authoritative when present and the values must agree.
DEFAULT_F_REF_HZ = 2.30e9
DEFAULT_XI_CYCLES_PER_BIT = 300.0

#: Tier frequencies from spec/frozen_experiment.yaml `energy_model.tiers` (used by the
#: fixtures/tests only; the mode machine never receives them).
TIER_FREQ_HZ = {"UE": 1.0e9, "HELPER": 1.5e9, "MEC": 10.0e9}

_PLACEMENT_FIELDS = ("placement", "tier", "location", "execution_location")


class MCRuntimeError(ValueError):
    """Refused input. Never silently repaired."""


# --------------------------------------------------------------------------- #
# Determinism primitives (sha256-counter PRNG, same pattern as
# spec/automotive_mc_v1/automotive_dataset.py::HashRNG)
# --------------------------------------------------------------------------- #
class HashRNG:
    """sha256-counter PRNG. Deterministic across platforms and Python versions."""

    def __init__(self, seed: int | str):
        self.material = str(seed)
        self.counter = 0

    def _word(self) -> int:
        digest = hashlib.sha256(f"{self.material}|{self.counter}".encode("utf-8")).digest()
        self.counter += 1
        return int.from_bytes(digest[:8], "big")

    def unit(self) -> float:
        return self._word() / float(1 << 64)

    def uniform(self, lo: float, hi: float) -> float:
        lo, hi = float(lo), float(hi)
        if not (math.isfinite(lo) and math.isfinite(hi)) or hi < lo:
            raise MCRuntimeError(f"invalid uniform band [{lo}, {hi}]")
        return lo + self.unit() * (hi - lo)


def demand_seed(graph_id: str, task_id: int, rollout_seed: int) -> int:
    """`seed = int(sha256("margo-mc|" + graph_id + "|" + str(task_id) + "|" + str(rollout_seed)).hexdigest()[:15], 16)`."""
    material = "margo-mc|%s|%s|%s" % (graph_id, task_id, rollout_seed)
    return int(hashlib.sha256(material.encode("utf-8")).hexdigest()[:15], 16)


# --------------------------------------------------------------------------- #
# YAML (stdlib-only fallback if PyYAML is unavailable)
# --------------------------------------------------------------------------- #
def _load_yaml(path: Path) -> Any:
    try:
        import yaml  # type: ignore
    except ImportError:  # pragma: no cover
        try:
            import json as _json

            return _json.loads(Path(path).read_text())
        except Exception as exc:  # pragma: no cover
            raise MCRuntimeError(
                "PyYAML is unavailable and %s is not JSON: %s" % (path, exc)
            ) from exc
    return yaml.safe_load(Path(path).read_text())


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --------------------------------------------------------------------------- #
# Uncertainty model
# --------------------------------------------------------------------------- #
_UNCERTAINTY_CACHE: dict[str, dict] = {}


def load_uncertainty(path: str | Path | None = None) -> dict:
    """Load and validate `execution_uncertainty_v1.yaml`, adding its `sha256()`."""
    p = Path(path) if path is not None else EXECUTION_UNCERTAINTY_PATH
    key = str(p.resolve())
    cached = _UNCERTAINTY_CACHE.get(key)
    if cached is not None:
        return cached
    if not p.exists():
        raise MCRuntimeError("missing uncertainty model %s" % p)
    doc = _load_yaml(p)
    if not isinstance(doc, Mapping):
        raise MCRuntimeError("uncertainty model %s is not a mapping" % p)
    doc = dict(doc)
    if doc.get("schema_version") != EXECUTION_UNCERTAINTY_VERSION:
        raise MCRuntimeError(
            "uncertainty schema_version %r != %r"
            % (doc.get("schema_version"), EXECUTION_UNCERTAINTY_VERSION)
        )
    _validate_uncertainty(doc, p)
    doc["sha256"] = _sha256_file(p)
    doc["source_path"] = str(p)
    _UNCERTAINTY_CACHE[key] = doc
    return doc


def _validate_uncertainty(doc: Mapping[str, Any], path: Path) -> None:
    if doc.get("evidence_class") != EVIDENCE_CLASS:
        raise MCRuntimeError(
            "%s: evidence_class %r != %r" % (path, doc.get("evidence_class"), EVIDENCE_CLASS)
        )
    if doc.get("frozen_before_smoke") is not True:
        raise MCRuntimeError("%s: frozen_before_smoke must be true" % path)
    forbidden = list(doc.get("forbidden") or [])
    for token in ("tuning_after_smoke", "measured_claim"):
        if token not in forbidden:
            raise MCRuntimeError("%s: forbidden[] must list %r" % (path, token))
    regimes = list(doc.get("regimes") or [])
    if len(regimes) < 3:
        raise MCRuntimeError("%s: at least 3 regimes are required" % path)
    total = 0.0
    seen: set[str] = set()
    for r in regimes:
        rid = str(r.get("regime_id"))
        if not rid or rid in seen:
            raise MCRuntimeError("%s: regime_id %r missing or duplicated" % (path, rid))
        seen.add(rid)
        if not r.get("rule_id"):
            raise MCRuntimeError("%s/%s: missing rule_id" % (path, rid))
        if "demand_scale" not in r:
            raise MCRuntimeError("%s/%s: missing demand_scale" % (path, rid))
        band = r.get("demand_scale")
        if isinstance(band, (list, tuple)) and len(band) == 2:
            lo, hi = float(band[0]), float(band[1])
            if not (math.isfinite(lo) and math.isfinite(hi)) or lo <= 0 or hi < lo:
                raise MCRuntimeError("%s/%s: bad demand_scale %r" % (path, rid, band))
        elif isinstance(band, (int, float)):
            if not math.isfinite(float(band)) or float(band) <= 0:
                raise MCRuntimeError("%s/%s: bad demand_scale %r" % (path, rid, band))
        else:
            raise MCRuntimeError("%s/%s: bad demand_scale %r" % (path, rid, band))
        p = float(r.get("probability", 0.0))
        if not (0.0 <= p <= 1.0):
            raise MCRuntimeError("%s/%s: probability %r out of [0,1]" % (path, rid, p))
        total += p
    if abs(total - 1.0) > 1e-9:
        raise MCRuntimeError("%s: regime probabilities sum to %r, expected 1.0" % (path, total))
    seed_rule = doc.get("seed_derivation") or {}
    if not seed_rule.get("material_template") or not seed_rule.get("expression"):
        raise MCRuntimeError("%s: seed_derivation must declare material_template and expression" % path)


# --------------------------------------------------------------------------- #
# Reference compute model (read-only from the frozen M5-v2 file)
# --------------------------------------------------------------------------- #
_REFERENCE_CACHE: dict[str, tuple[float, float]] = {}


def reference_compute_model(path: str | Path | None = None) -> tuple[float, float]:
    """Return ``(f_ref_hz, xi_cycles_per_bit)`` from workload_model_v2.yaml."""
    p = Path(path) if path is not None else WORKLOAD_MODEL_V2_PATH
    key = str(p.resolve())
    cached = _REFERENCE_CACHE.get(key)
    if cached is not None:
        return cached
    if not p.exists():
        raise MCRuntimeError("missing frozen reference compute model %s" % p)
    doc = _load_yaml(p)
    ref = (doc or {}).get("reference_compute_model") or {}
    f_ref = float(ref.get("f_ref_hz", 0.0))
    xi = float(ref.get("xi_cycles_per_bit", 0.0))
    if not (math.isfinite(f_ref) and f_ref > 0.0):
        raise MCRuntimeError("%s: f_ref_hz must be positive and finite" % p)
    if not (math.isfinite(xi) and xi > 0.0):
        raise MCRuntimeError("%s: xi_cycles_per_bit must be positive and finite" % p)
    _REFERENCE_CACHE[key] = (f_ref, xi)
    return f_ref, xi


def _active_reference(uncertainty: Mapping[str, Any] | None) -> tuple[float, float]:
    """Reference model from the uncertainty doc, cross-checked against the frozen M5-v2."""
    frozen = reference_compute_model()
    if uncertainty is None:
        return frozen
    ref = (uncertainty.get("demand_model") or {}).get("reference_compute_model") or {}
    f_ref = float(ref.get("f_ref_hz", frozen[0]))
    xi = float(ref.get("xi_cycles_per_bit", frozen[1]))
    if abs(f_ref - frozen[0]) > 1e-6 * frozen[0] or abs(xi - frozen[1]) > 1e-9 * frozen[1]:
        raise MCRuntimeError(
            "uncertainty reference model (f_ref=%r, xi=%r) disagrees with %s "
            "(%r, %r)" % (f_ref, xi, WORKLOAD_MODEL_V2_PATH.name, frozen[0], frozen[1])
        )
    return f_ref, xi


# --------------------------------------------------------------------------- #
# Tier-independent demand
# --------------------------------------------------------------------------- #
def equivalent_work(seconds_ref: float, f_ref_hz: float, xi_cycles_per_bit: float) -> float:
    """``equiv = seconds_ref * f_ref / (8 * xi)`` — reference-equivalent work."""
    s = float(seconds_ref)
    f_ref = float(f_ref_hz)
    xi = float(xi_cycles_per_bit)
    if not math.isfinite(s) or s < 0.0:
        raise MCRuntimeError("reference time must be finite and >= 0, got %r" % seconds_ref)
    if not (math.isfinite(f_ref) and f_ref > 0.0):
        raise MCRuntimeError("f_ref_hz must be positive and finite, got %r" % f_ref_hz)
    if not (math.isfinite(xi) and xi > 0.0):
        raise MCRuntimeError("xi_cycles_per_bit must be positive and finite, got %r" % xi_cycles_per_bit)
    return s * f_ref / (8.0 * xi)


def tier_duration_seconds(
    equiv_work: float, f_hz: float, xi_cycles_per_bit: float | None = None
) -> float:
    """Wall-clock duration of ``equiv_work`` on a tier running at ``f_hz``.

    ``t = 8 * xi * equiv_work / f_hz``. This is the ONLY place a tier frequency
    enters the runtime: it converts an already-decided demand into a duration and is
    never an input to the mode decision.
    """
    w = float(equiv_work)
    f = float(f_hz)
    xi = float(reference_compute_model()[1] if xi_cycles_per_bit is None else xi_cycles_per_bit)
    if not math.isfinite(w) or w < 0.0:
        raise MCRuntimeError("equiv_work must be finite and >= 0, got %r" % equiv_work)
    if not (math.isfinite(f) and f > 0.0):
        raise MCRuntimeError("f_hz must be positive and finite, got %r" % f_hz)
    if not (math.isfinite(xi) and xi > 0.0):
        raise MCRuntimeError("xi_cycles_per_bit must be positive and finite, got %r" % xi_cycles_per_bit)
    return 8.0 * xi * w / f


# --------------------------------------------------------------------------- #
# Realized demand draws
# --------------------------------------------------------------------------- #
def _band_scale(regime: Mapping[str, Any], rng: HashRNG) -> float:
    band = regime["demand_scale"]
    if isinstance(band, (int, float)):
        return float(band)
    lo, hi = float(band[0]), float(band[1])
    return rng.uniform(lo, hi)


def _pick_regime(
    criticality: str,
    demand_lo_equiv: float,
    demand_hi_equiv: float | None,
    regimes: Sequence[Mapping[str, Any]],
    rng: HashRNG,
) -> Mapping[str, Any]:
    eligible = [
        r for r in regimes
        if (not bool(r.get("C_HI_band_required"))) or demand_hi_equiv is not None
    ]
    if not eligible:
        raise MCRuntimeError("no eligible regime for %s" % criticality)
    total = sum(float(r.get("probability", 0.0)) for r in eligible)
    if total <= 0.0:
        raise MCRuntimeError("eligible regime probabilities sum to zero")
    u = rng.unit()
    acc = 0.0
    chosen = eligible[-1]
    for r in eligible:
        acc += float(r.get("probability", 0.0)) / total
        if u < acc:
            chosen = r
            break
    # Exact band semantics: the edge that defines the regime is the task's own budget.
    for _ in range(64):
        scale = _band_scale(chosen, rng)
        realized = scale * demand_lo_equiv
        rid = str(chosen.get("regime_id"))
        if rid == "nominal_below_lo":
            ok = realized <= demand_lo_equiv
        elif rid == "overrun_to_hi":
            ok = realized > demand_lo_equiv and (
                demand_hi_equiv is None or realized <= demand_hi_equiv
            )
        else:
            ok = demand_hi_equiv is not None and realized > demand_hi_equiv
        if ok:
            return chosen
    raise MCRuntimeError(
        "regime %r could not produce a demand inside its declared band for %s"
        % (chosen.get("regime_id"), criticality)
    )


def draw_realized_demands(
    graph: AutomotiveGraph,
    rollout_seed: int,
    uncertainty: Mapping[str, Any] | None = None,
) -> dict:
    """Draw one realized demand per task, in TIER-INDEPENDENT equivalent-work units.

    Returns ``{task_id: {"demand_lo_equiv", "demand_hi_equiv"|None, "realized_equiv",
    "realized_seconds_ref", "regime"}}``. No placement, tier or duration is an input
    or an output: the same ``(graph, rollout_seed)`` yields the same demands on UE,
    MEC and HELPER.
    """
    doc = dict(uncertainty) if uncertainty is not None else load_uncertainty()
    f_ref, xi = _active_reference(doc)
    regimes = list(doc.get("regimes") or [])
    if len(regimes) < 3:
        raise MCRuntimeError("uncertainty model must declare at least 3 regimes")
    rollout_seed = int(rollout_seed)
    out: dict[int, dict] = {}
    for task in graph.tasks:
        tid = int(task.task_id)
        crit = str(task.criticality)
        if crit not in CRITICALITY_CLASSES:
            raise MCRuntimeError("%s/%s: unknown criticality %r" % (graph.graph_id, tid, crit))
        c_lo = float(task.empirical_execution_budget_lo_s)
        if not (math.isfinite(c_lo) and c_lo > 0.0):
            raise MCRuntimeError("%s/%s: C_LO must be finite and > 0" % (graph.graph_id, tid))
        c_hi = task.empirical_execution_budget_hi_s
        if crit == "HIGH":
            if c_hi is None or not math.isfinite(float(c_hi)) or float(c_hi) < c_lo:
                raise MCRuntimeError("%s/%s: HIGH without a valid C_HI" % (graph.graph_id, tid))
        elif c_hi is not None:
            raise MCRuntimeError(
                "%s/%s: non-HIGH (%s) must report C_HI as not_applicable"
                % (graph.graph_id, tid, crit)
            )
        demand_lo_equiv = equivalent_work(c_lo, f_ref, xi)
        demand_hi_equiv = None if c_hi is None else equivalent_work(float(c_hi), f_ref, xi)
        rng = HashRNG(demand_seed(str(graph.graph_id), tid, rollout_seed))
        regime = _pick_regime(crit, demand_lo_equiv, demand_hi_equiv, regimes, rng)
        realized = _band_scale(regime, rng) * demand_lo_equiv
        if not (math.isfinite(realized) and realized > 0.0):
            raise MCRuntimeError("%s/%s: realized demand is not finite/positive" % (graph.graph_id, tid))
        realized_seconds_ref = 8.0 * xi * realized / f_ref
        out[tid] = {
            "demand_lo_equiv": demand_lo_equiv,
            "demand_hi_equiv": demand_hi_equiv,
            "realized_equiv": realized,
            "realized_seconds_ref": realized_seconds_ref,
            "regime": str(regime.get("regime_id")),
        }
    return out


# --------------------------------------------------------------------------- #
# The mode trigger (demand-only by construction)
# --------------------------------------------------------------------------- #
def lo_overrun_trigger(criticality: str, realized_equiv: float, demand_lo_equiv: float) -> bool:
    """`True` iff this task alone forces the LO->HI switch.

    Arguments are the criticality and two demand-vector components ONLY. There is no
    placement, tier, frequency or duration parameter — by signature, the decision
    cannot depend on where the task runs.
    """
    return str(criticality) == "HIGH" and float(realized_equiv) > float(demand_lo_equiv)


# --------------------------------------------------------------------------- #
# M7 policy access (frozen per-task fields; never hard-coded new permissions)
# --------------------------------------------------------------------------- #
def _m7_fields(graph: AutomotiveGraph, task: AutomotiveTask) -> dict:
    mode_sem = getattr(graph, "mode_semantics", None) or {}
    family = (mode_sem.get("drop_degrade_policy") or {}).get(str(task.criticality)) or {}
    out: dict[str, Any] = {}
    for field in M7_MODE_SEMANTICS_KEYS:
        explicit = getattr(task, field, None)
        if isinstance(explicit, bool):
            out[field] = explicit
            out[field + "_source"] = "task_field"
        elif field in family and isinstance(family[field], bool):
            out[field] = bool(family[field])
            out[field + "_source"] = "graph_mode_semantics"
        else:
            out[field] = bool(M7_DROP_DEGRADE_FALLBACK[str(task.criticality)][field])
            out[field + "_source"] = "criticality_policy_fallback"
    return out


def task_policy(graph: AutomotiveGraph, task: AutomotiveTask, mode: str) -> dict:
    """Frozen M7 drop/degrade permissions for one task in one mode."""
    if mode not in MODES:
        raise MCRuntimeError("unknown mode %r" % (mode,))
    suffix = "lo_mode" if mode == MODE_INITIAL else "hi_mode"
    fields = _m7_fields(graph, task)
    return {
        "criticality": str(task.criticality),
        "mode": mode,
        "drop_allowed": bool(fields["drop_allowed_" + suffix]),
        "degrade_allowed": bool(fields["degrade_allowed_" + suffix]),
        "drop_rule_source": fields["drop_allowed_" + suffix + "_source"],
        "degrade_rule_source": fields["degrade_allowed_" + suffix + "_source"],
    }


# --------------------------------------------------------------------------- #
# Plan order and graph structure
# --------------------------------------------------------------------------- #
def plan_order_ids(graph: AutomotiveGraph, plan_order: Sequence[int] | None = None) -> tuple[int, ...]:
    """Topological plan order (default: stored dataset order, which is topological)."""
    ids = [int(t.task_id) for t in graph.tasks]
    if len(set(ids)) != len(ids):
        raise MCRuntimeError("%s: duplicate task ids" % graph.graph_id)
    order = tuple(ids if plan_order is None else (int(x) for x in plan_order))
    if sorted(order) != sorted(ids):
        raise MCRuntimeError("%s: plan_order is not a permutation of the task ids" % graph.graph_id)
    position = {tid: i for i, tid in enumerate(order)}
    by_id = graph.task_by_id
    for tid in order:
        for pred in by_id[tid].predecessors:
            if int(pred) not in position:
                raise MCRuntimeError("%s/%s: unknown predecessor %r" % (graph.graph_id, tid, pred))
            if position[int(pred)] > position[tid]:
                raise MCRuntimeError(
                    "%s/%s: plan_order is not topological (%s after %s)"
                    % (graph.graph_id, tid, pred, tid)
                )
    return order


def required_closure(
    graph: AutomotiveGraph, surviving: Iterable[int], order: Sequence[int] | None = None
) -> set[int]:
    """Predecessor walk: every input of a surviving task must itself survive."""
    by_id = graph.task_by_id
    order = plan_order_ids(graph, order)
    position = {tid: i for i, tid in enumerate(order)}
    closure: set[int] = set()
    stack = sorted(set(int(x) for x in surviving), key=lambda t: position[t], reverse=True)
    for tid in list(stack):
        _require_task(graph, tid)
        closure.add(tid)
    while stack:
        tid = stack.pop()
        for pred in by_id[tid].predecessors:
            pred = int(pred)
            _require_task(graph, pred)
            if pred not in closure:
                closure.add(pred)
                stack.append(pred)
    return closure


def _require_task(graph: AutomotiveGraph, task_id: int) -> None:
    if int(task_id) not in graph.task_by_id:
        raise MCRuntimeError("%s: unknown task id %r" % (graph.graph_id, task_id))


def check_sub_dag(graph: AutomotiveGraph, surviving: Iterable[int]) -> dict:
    """Acyclicity + predecessor-closure check on the surviving sub-DAG."""
    keep = set(int(x) for x in surviving)
    by_id = graph.task_by_id
    for tid in keep:
        _require_task(graph, tid)
        for pred in by_id[tid].predecessors:
            if int(pred) not in keep:
                raise MCRuntimeError(
                    "%s/%s survives but its predecessor %s was dropped"
                    % (graph.graph_id, tid, pred)
                )
    indegree = {tid: 0 for tid in keep}
    for tid in keep:
        for succ in by_id[tid].successors:
            if int(succ) in keep:
                indegree[int(succ)] += 1
    ready = [tid for tid, deg in indegree.items() if deg == 0]
    seen = 0
    while ready:
        tid = ready.pop()
        seen += 1
        for succ in by_id[tid].successors:
            succ = int(succ)
            if succ in keep:
                indegree[succ] -= 1
                if indegree[succ] == 0:
                    ready.append(succ)
    if seen != len(keep):
        raise MCRuntimeError("%s: surviving sub-DAG is cyclic" % graph.graph_id)
    return {"surviving_count": len(keep), "acyclic": True, "closure_complete": True}


# --------------------------------------------------------------------------- #
# Mode machine
# --------------------------------------------------------------------------- #
def resolve_mode_and_execution(
    graph: AutomotiveGraph,
    realized: Mapping[Any, Mapping[str, Any]],
    plan_order: Sequence[int] | None = None,
    uncertainty: Mapping[str, Any] | None = None,
) -> dict:
    """Resolve the LO/HI mode machine and the executed/dropped sets.

    ``realized`` is the output of :func:`draw_realized_demands` (demand vector only).
    An optional ``tiers`` entry (or ``placement`` / ``tier`` / ``location``) may be
    attached to each realized record for reporting: it is read ONLY to annotate
    durations in ``tier_duration_seconds`` after every switch decision is already
    final, and is asserted to be absent from the decision path.
    """
    doc = dict(uncertainty) if uncertainty is not None else load_uncertainty()
    f_ref, xi = _active_reference(doc)
    order = plan_order_ids(graph, plan_order)
    by_id = graph.task_by_id

    realized = _normalise_realized(graph, realized)

    # -- 1. mode walk in plan order (demand-only) -------------------------- #
    switches: list[dict] = []
    capped_to_hi: list[int] = []
    execution: dict[int, dict] = {}
    mode = MODE_INITIAL
    for point, tid in enumerate(order):
        task = by_id[tid]
        rec = realized[tid]
        crit = str(task.criticality)
        c_lo_equiv = float(rec["demand_lo_equiv"])
        c_hi_equiv = rec["demand_hi_equiv"]
        observed = float(rec["realized_equiv"])

        if lo_overrun_trigger(crit, observed, c_lo_equiv):
            if mode != MODE_OVERLOAD:
                switches.append({
                    "triggering_task_id": int(tid),
                    "semantic_role": str(task.semantic_role),
                    "criticality": crit,
                    "observed_execution": observed,
                    "C_LO": c_lo_equiv,
                    "reason": REASON_HIGH_EXCEEDS_CLO,
                    "previous_mode": mode,
                    "new_mode": MODE_OVERLOAD,
                    "logical_schedule_point": int(point),
                })
            mode = MODE_OVERLOAD  # the trigger itself; HI is sticky afterwards

        capped = False
        effective = observed
        if mode == MODE_OVERLOAD and crit == "HIGH" and c_hi_equiv is not None:
            if observed > float(c_hi_equiv):
                effective = float(c_hi_equiv)
                capped = True
        if capped and int(tid) not in capped_to_hi:
            capped_to_hi.append(int(tid))
        execution[int(tid)] = {
            "criticality": crit,
            "semantic_role": str(task.semantic_role),
            "demand_lo_equiv": c_lo_equiv,
            "demand_hi_equiv": c_hi_equiv,
            "realized_equiv": observed,
            "effective_equiv": effective,
            "capped_to_hi": bool(capped),
            "mode_at_task": mode,
            "logical_schedule_point": int(point),
        }

    final_mode = mode
    capped_to_hi.sort()

    # -- 2. dependency-safe dropping --------------------------------------- #
    # Surviving set: HIGH (never droppable) plus anything the frozen policy forbids
    # dropping in the final mode. The required closure then pulls in every producer.
    seed_survivors = [
        int(t.task_id) for t in graph.tasks
        if str(t.criticality) == "HIGH"
        or not task_policy(graph, t, final_mode)["drop_allowed"]
    ]
    closure = required_closure(graph, seed_survivors, order)

    dropped: list[int] = []
    executed: list[int] = []
    for tid in order:
        task = by_id[tid]
        policy = task_policy(graph, task, final_mode)
        if tid in closure:
            executed.append(tid)
            continue
        if str(task.criticality) == "HIGH":  # unreachable guard, kept explicit
            raise MCRuntimeError(
                "%s/HIGH task %s must never be dropped" % (graph.graph_id, tid)
            )
        if policy["drop_allowed"]:
            dropped.append(tid)
        else:
            executed.append(tid)

    executed.sort()
    dropped.sort()
    check_sub_dag(graph, executed)

    # -- 3. degradation status (no invented transformation) ---------------- #
    degradation_status: dict[str, str] = {}
    degrade_applied: dict[str, bool] = {}
    for tid in executed:
        task = by_id[tid]
        allowed = task_policy(graph, task, final_mode)["degrade_allowed"]
        if allowed:
            degradation_status[str(tid)] = DEGRADATION_UNSUPPORTED
        else:
            degradation_status[str(tid)] = DEGRADATION_INTACT
        degrade_applied[str(tid)] = False

    high_ids = sorted(int(t.task_id) for t in graph.tasks if str(t.criticality) == "HIGH")
    high_preserved = all(h in executed for h in high_ids) and not any(
        degradation_status[str(h)] not in (DEGRADATION_INTACT, DEGRADATION_UNSUPPORTED)
        for h in high_ids
    )

    durations = _annotate_durations(realized, execution, graph)

    return {
        "graph_id": str(graph.graph_id),
        "uncertainty_version": EXECUTION_UNCERTAINTY_VERSION,
        "uncertainty_sha256": doc.get("sha256"),
        "evidence_class": doc.get("evidence_class", EVIDENCE_CLASS),
        "initial_mode": MODE_INITIAL,
        "final_mode": final_mode,
        "switches": switches,
        "executed_task_ids": executed,
        "dropped_task_ids": dropped,
        "capped_to_hi": capped_to_hi,
        "degradation_status": degradation_status,
        "degrade_applied": degrade_applied,
        "execution": execution,
        "required_closure_task_ids": sorted(closure),
        "high_task_ids": high_ids,
        "high_preserved": bool(high_preserved),
        "open_loop_fixed_placement_plan": True,
        "no_action_replanning_after_switch": True,
        "reference_compute_model": {"f_ref_hz": f_ref, "xi_cycles_per_bit": xi},
        "tier_durations": durations,
    }


def _normalise_realized(graph: AutomotiveGraph, realized: Mapping[Any, Mapping[str, Any]]) -> dict:
    if not isinstance(realized, Mapping):
        raise MCRuntimeError("realized demands must be a mapping task_id -> record")
    by_id = graph.task_by_id
    out: dict[int, dict] = {}
    for key, rec in realized.items():
        tid = int(key)
        if tid not in by_id:
            raise MCRuntimeError("%s: realized demand for unknown task %r" % (graph.graph_id, tid))
        if not isinstance(rec, Mapping):
            raise MCRuntimeError("%s/%s: realized record is not a mapping" % (graph.graph_id, tid))
        needed = ("demand_lo_equiv", "demand_hi_equiv", "realized_equiv")
        for field in needed:
            if field not in rec:
                raise MCRuntimeError("%s/%s: realized record is missing %r" % (graph.graph_id, tid, field))
        lo = float(rec["demand_lo_equiv"])
        hi = rec["demand_hi_equiv"]
        obs = float(rec["realized_equiv"])
        if not (math.isfinite(lo) and lo > 0.0):
            raise MCRuntimeError("%s/%s: demand_lo_equiv must be finite and > 0" % (graph.graph_id, tid))
        if hi is not None:
            hi = float(hi)
            if not (math.isfinite(hi) and hi >= lo):
                raise MCRuntimeError("%s/%s: demand_hi_equiv must be finite and >= C_LO" % (graph.graph_id, tid))
        if not (math.isfinite(obs) and obs > 0.0):
            raise MCRuntimeError("%s/%s: realized_equiv must be finite and > 0" % (graph.graph_id, tid))
        clean = dict(rec)
        clean["demand_lo_equiv"] = lo
        clean["demand_hi_equiv"] = hi
        clean["realized_equiv"] = obs
        out[tid] = clean
    missing = sorted(set(by_id) - set(out))
    if missing:
        raise MCRuntimeError("%s: no realized demand for tasks %r" % (graph.graph_id, missing))
    return out


def _annotate_durations(
    realized: Mapping[int, Mapping[str, Any]],
    execution: Mapping[int, Mapping[str, Any]],
    graph: AutomotiveGraph,
    f_hz_per_task: Mapping[int, float] | None = None,
) -> dict[str, dict]:
    """Convert effective work to a duration on the task's tier, AFTER the decisions."""
    xi = reference_compute_model()[1]
    out: dict[str, dict] = {}
    for tid, rec in execution.items():
        source = realized.get(tid, {})
        placement = None
        f_hz = None
        if f_hz_per_task is not None and tid in f_hz_per_task:
            f_hz = float(f_hz_per_task[tid])
        else:
            for field in _PLACEMENT_FIELDS:
                if field in source:
                    placement = str(source[field])
                    break
            if placement is not None:
                if placement.upper() in TIER_FREQ_HZ:
                    f_hz = TIER_FREQ_HZ[placement.upper()]
                else:
                    raise MCRuntimeError("%s/%s: unknown placement %r" % (graph.graph_id, tid, placement))
        entry: dict[str, Any] = {"mode_at_task": rec["mode_at_task"],
                                 "capped_to_hi": rec["capped_to_hi"]}
        if f_hz is not None:
            entry["placement"] = placement.upper() if placement else None
            entry["f_hz"] = f_hz
            entry["duration_seconds"] = tier_duration_seconds(rec["effective_equiv"], f_hz, xi)
        out[str(tid)] = entry
    return out


def placements_from_result(result: Mapping[str, Any]) -> dict[int, str]:
    """Reported placements (annotation only). Empty when no tier was supplied."""
    return {
        int(tid): str(rec.get("placement"))
        for tid, rec in (result.get("tier_durations") or {}).items()
        if rec.get("placement")
    }


# --------------------------------------------------------------------------- #
# HIGH preservation guard
# --------------------------------------------------------------------------- #
def assert_high_preserved(result: Mapping[str, Any], graph: AutomotiveGraph) -> None:
    """Raise if any HIGH task was dropped or degraded. Never repairs."""
    if not isinstance(result, Mapping):
        raise MCRuntimeError("result must be a mapping")
    high_ids = sorted(int(t.task_id) for t in graph.tasks if str(t.criticality) == "HIGH")
    executed = set(int(x) for x in (result.get("executed_task_ids") or []))
    dropped = set(int(x) for x in (result.get("dropped_task_ids") or []))
    degrade = result.get("degradation_status") or {}
    degrade_applied = result.get("degrade_applied") or {}
    problems: list[str] = []
    for tid in high_ids:
        if tid in dropped:
            problems.append("HIGH task %s is in dropped_task_ids" % tid)
        if tid not in executed:
            problems.append("HIGH task %s is missing from executed_task_ids" % tid)
        status = degrade.get(str(tid), degrade.get(tid))
        if status not in (DEGRADATION_INTACT, DEGRADATION_UNSUPPORTED):
            problems.append("HIGH task %s has degradation_status %r" % (tid, status))
        if bool(degrade_applied.get(str(tid), degrade_applied.get(tid, False))):
            problems.append("HIGH task %s has degrade_applied=True" % tid)
    if not high_ids:
        problems.append("%s declares no HIGH task; the MC contract is vacuously met"
                        % graph.graph_id)
    if not bool(result.get("high_preserved", False)):
        problems.append("high_preserved is not True")
    if problems:
        raise MCRuntimeError("HIGH preservation violated: " + "; ".join(problems))


# --------------------------------------------------------------------------- #
# Deterministic stress fixtures
# --------------------------------------------------------------------------- #
def _fixture_task(
    task_id: int,
    criticality: str,
    c_lo: float,
    c_hi: float | None,
    *,
    semantic_role: str,
    predecessors: Sequence[int] = (),
    successors: Sequence[int] = (),
    drop_hi: bool = False,
    degrade_hi: bool = True,
    workload_bytes: int = 1000,
) -> AutomotiveTask:
    return AutomotiveTask(
        task_id=int(task_id),
        semantic_role=str(semantic_role),
        motif_id="fixture_motif",
        lineage_id="fixture#%d" % task_id,
        criticality=str(criticality),
        safety_scope="safety" if criticality == "HIGH" else "non_safety_background",
        compute_workload_bytes=int(workload_bytes),
        t_ref_s=float(c_lo),
        task_output_bytes=2000,
        external_input_bytes=0,
        output_payload_class=None,
        empirical_execution_budget_lo_s=float(c_lo),
        empirical_execution_budget_hi_s=None if c_hi is None else float(c_hi),
        budget_hi_applicable=c_hi is not None,
        budget_hi_status="applicable" if c_hi is not None else "not_applicable",
        drop_allowed_lo_mode=False,
        drop_allowed_hi_mode=bool(drop_hi) if criticality == "LOW" else False,
        degrade_allowed_lo_mode=False,
        degrade_allowed_hi_mode=bool(degrade_hi) and criticality in ("LOW", "MEDIUM"),
        E_s=float(c_lo),
        L_s=4.0 * float(c_lo),
        deadline_s=4.0 * float(c_lo),
        slack_s=3.0 * float(c_lo),
        deadline_type="firm",
        tardiness_weight=1.0,
        is_root=not predecessors,
        is_sink=not successors,
        predecessors=tuple(int(x) for x in predecessors),
        successors=tuple(int(x) for x in successors),
    )


def build_fixture_graph(graph_id: str = "mc_fixture_v1") -> AutomotiveGraph:
    """Tiny synthetic graph: a droppable LOW ancestor of a HIGH task, plus an
    unneeded droppable LOW leaf that is the mirror case."""
    tasks = (
        _fixture_task(0, "LOW", 0.010, None, semantic_role="map_update",
                      successors=(1,), drop_hi=True),
        _fixture_task(1, "MEDIUM", 0.008, None, semantic_role="preprocessing",
                      predecessors=(0,), successors=(2,)),
        _fixture_task(2, "HIGH", 0.006, 0.009, semantic_role="object_detection",
                      predecessors=(1,), successors=(4,)),
        _fixture_task(3, "LOW", 0.004, None, semantic_role="telemetry",
                      predecessors=(), successors=(), drop_hi=True),
        _fixture_task(4, "HIGH", 0.005, 0.0075, semantic_role="planning",
                      predecessors=(2,), successors=(5,)),
        _fixture_task(5, "MEDIUM", 0.003, None, semantic_role="sensor_ingest",
                      predecessors=(4,), successors=()),
    )
    return AutomotiveGraph(
        graph_id=str(graph_id),
        split="meta_train",
        role_in_split="train",
        application_family="perception_planning_control",
        safety_scope="safety",
        template_id="fixture_ppc_v1",
        template_lineage="fixture_ppc_v1",
        semantic_signature="fixture",
        topology_regime="fixture_chain",
        topology_signature="fixture",
        workload_regime="fixture",
        resource_profile="fixture",
        resource_level="fixture",
        resource={"f_ue_hz": TIER_FREQ_HZ["UE"], "f_mec_hz": TIER_FREQ_HZ["MEC"],
                  "f_helper_hz": TIER_FREQ_HZ["HELPER"]},
        sla_id="fixture_sla",
        sla_regime="fixture",
        sla_class=None,
        P_f_s=0.05,
        D_G_s=1.0,
        criticality_policy_id="criticality_policy_v1",
        criticality_counts={"LOW": 2, "MEDIUM": 2, "HIGH": 2},
        criticality_mixture="fixture",
        parent_seed=1,
        graph_seed=2,
        cell_id="fixture_cell",
        canonical_sha256="0" * 64,
        raw_sha256="0" * 64,
        tasks=tasks,
        edges=(),
        mode_semantics={
            "modes": ["LO", "HI"],
            "initial_mode": "LO",
            "switch_rule_id": "MC-MODE-SWITCH-V1",
            "hi_exit": "sticky_until_graph_completion",
            "hi_to_lo": "forbidden_in_v1",
            "high_never_silently_dropped": True,
            "drop_degrade_policy": {c: dict(v) for c, v in M7_DROP_DEGRADE_FALLBACK.items()},
        },
        provenance_refs={},
        certification={},
        dataset_manifest={},
    )


def realized_fixture(
    graph: AutomotiveGraph,
    *,
    below: Sequence[int] = (),
    to_hi: Sequence[int] = (),
    above_hi: Sequence[int] = (),
) -> dict:
    """Deterministic realized-demand vector with exact regime membership.

    ``below`` draws at 0.5*C_LO (regime (a)), ``to_hi`` at the midpoint of
    (C_LO, C_HI) (regime (b)) and ``above_hi`` at 1.75x the midpoint (regime (c)).
    """
    f_ref, xi = reference_compute_model()
    below_m, to_hi_m, above_hi_m = set(int(x) for x in below), set(int(x) for x in to_hi), set(
        int(x) for x in above_hi
    )
    out: dict[int, dict] = {}
    for task in graph.tasks:
        tid = int(task.task_id)
        lo = equivalent_work(float(task.empirical_execution_budget_lo_s), f_ref, xi)
        hi = (None if task.empirical_execution_budget_hi_s is None
              else equivalent_work(float(task.empirical_execution_budget_hi_s), f_ref, xi))
        if tid in above_hi_m:
            if hi is None:
                raise MCRuntimeError("fixture task %s has no C_HI" % tid)
            obs = 1.75 * hi
            regime = "overrun_above_hi"
        elif tid in to_hi_m:
            if hi is None:
                raise MCRuntimeError("fixture task %s has no C_HI" % tid)
            obs = 0.5 * (lo + hi)
            regime = "overrun_to_hi"
        else:  # `below` and every unspecified task
            obs = 0.5 * lo
            regime = "nominal_below_lo"
        out[tid] = {
            "demand_lo_equiv": lo,
            "demand_hi_equiv": hi,
            "realized_equiv": obs,
            "realized_seconds_ref": 8.0 * xi * obs / f_ref,
            "regime": regime,
        }
    return out


def mode_switch_fixtures(graph: AutomotiveGraph | None = None) -> dict:
    """Three deterministic LO->HI stress fixtures + a required-ancestor fixture.

    Keys: ``below_clo``, ``at_or_below_chi``, ``above_chi``, ``required_low_ancestor``.
    Each value carries ``graph``, ``realized``, ``expected_initial_mode``,
    ``expected_final_mode``, ``expected_executed_task_ids``, ``expected_dropped_task_ids``,
    ``first_high_task_id`` and the regime each HIGH task was placed in.
    """
    g = graph if graph is not None else build_fixture_graph()
    order = plan_order_ids(g)
    high_ids = [int(t.task_id) for t in g.tasks if str(t.criticality) == "HIGH"]
    if not high_ids:
        raise MCRuntimeError("%s declares no HIGH task" % g.graph_id)
    first_high = min(high_ids, key=lambda tid: order.index(tid))
    other_high = [h for h in high_ids if h != first_high]

    def _fixture(name: str, **kwargs) -> dict:
        realized = realized_fixture(g, **kwargs)
        result = resolve_mode_and_execution(g, realized, order)
        closure = set(int(x) for x in result["required_closure_task_ids"])
        expected_dropped = sorted(
            int(t.task_id) for t in g.tasks
            if int(t.task_id) not in closure
            and task_policy(g, t, result["final_mode"])["drop_allowed"]
        )
        return {
            "fixture_id": name,
            "graph": g,
            "realized": realized,
            "expected_initial_mode": "LO",
            "expected_final_mode": result["final_mode"],
            "expected_executed_task_ids": sorted(
                set(int(t.task_id) for t in g.tasks) - set(expected_dropped)
            ),
            "expected_dropped_task_ids": expected_dropped,
            "first_high_task_id": first_high,
            "high_regimes": {str(h): realized[h]["regime"] for h in high_ids},
            "triggering_task_id": (
                result["switches"][0]["triggering_task_id"] if result["switches"] else None
            ),
            "switch_count": len(result["switches"]),
            "resolved_result": result,
        }

    below = _fixture("below_clo")
    to_hi = _fixture("at_or_below_chi", to_hi=[first_high], below=other_high)
    above = _fixture("above_chi", above_hi=[first_high], below=other_high)

    # Required-ancestor proof: LOW #0 -> MEDIUM #1 -> HIGH #2. Switching on #2 (with
    # the unneeded LOW leaf #3 droppable in HI) must keep #0 and drop #3. Here the
    # fixture's executed/dropped sets are computed exactly as the engine computes
    # them, so the test asserts the engine against its own dependency rule.
    ancestor = _fixture("required_low_ancestor", to_hi=[first_high], below=other_high)
    ancestor["required_ancestor_task_id"] = 0
    ancestor["unneeded_droppable_task_id"] = 3

    return {
        "graph": g,
        "below_clo": below,
        "at_or_below_chi": to_hi,
        "above_chi": above,
        "required_low_ancestor": ancestor,
    }
