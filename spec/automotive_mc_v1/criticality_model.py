#!/usr/bin/env python3
"""M7 mixed-criticality model for MARGO-AUTOMOTIVE-MC-v1.

Three things live here and nowhere else:

1. SEMANTIC CLASS ASSIGNMENT. `criticality_class` is a function of
   (semantic_role, application_family, motif_role, explicit safety policy) only.
   The resolver never reads depth, topology, task id, workload, payload, deadline,
   slack, execution location or resource profile, and the adversarial tests mutate
   exactly those inputs to prove the class does not move.

2. EXECUTION-DEMAND BUDGETS. `empirical_execution_budget_lo_s` / `_hi_s` model
   execution-demand uncertainty (amc-style C_LO/C_HI), NOT safety level and NOT
   WCET. HIGH tasks carry both; LOW/MEDIUM report the HI budget as `not_applicable`.
   No field is ever named WCET because no source in the registry reports one.

3. LO/HI OPERATING MODES. Start in LO. The ONLY LO->HI trigger is a HIGH task whose
   observed execution exceeds its C_LO. HI is sticky until graph completion in v1.
   Every switch is logged with the frozen field set. HIGH is never drop- or
   degrade-allowed in either mode.

Nothing here schedules, searches, mutates a deadline or writes a dataset file.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

CRITICALITY_MODEL_VERSION = "criticality_model_v1"
CLASSES = ("LOW", "MEDIUM", "HIGH")
CLASS_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
MODES = ("LO", "HI")
INITIAL_MODE = "LO"
MODE_SWITCH_RULE_ID = "MC-MODE-SWITCH-V1"
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

FORBIDDEN_INPUTS = (
    "depth", "topological_position", "decoder_position", "task_id", "workload",
    "payload", "deadline", "slack", "execution_location", "resource_profile",
    "edge_count", "in_degree", "out_degree",
)


class CriticalityError(ValueError):
    """Input refused. Never silently repaired."""


def _canon(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def policy_sha256(policy: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canon(policy).encode("utf-8")).hexdigest()


def _scope(policy: Mapping[str, Any], family: str) -> str:
    try:
        return str(policy["safety_scope"][family])
    except KeyError as exc:
        raise CriticalityError(f"unknown application family {family!r}") from exc


def _baseline(policy: Mapping[str, Any], role: str) -> str:
    try:
        cls = str(policy["role_baseline_class"][role])
    except KeyError as exc:
        raise CriticalityError(f"semantic role {role!r} has no baseline class") from exc
    if cls not in CLASSES:
        raise CriticalityError(f"role {role!r} has invalid class {cls!r}")
    return cls


def _motif_floor(policy: Mapping[str, Any], role: str) -> str:
    motif = policy["role_motif"].get(role)
    if motif is None:
        raise CriticalityError(f"semantic role {role!r} has no motif_role")
    try:
        floor = str(policy["motif_semantics"][motif]["baseline_class"])
    except KeyError as exc:
        raise CriticalityError(f"motif {motif!r} has no baseline_class") from exc
    if floor not in CLASSES:
        raise CriticalityError(f"motif {motif!r} has invalid baseline class {floor!r}")
    return floor


def classify(policy: Mapping[str, Any], role: str, family: str) -> dict:
    """Deterministic, total semantic classification of one (role, family) pair."""
    scope = _scope(policy, family)
    baseline = _baseline(policy, role)
    override = (policy.get("family_class_override") or {}).get(family) or {}
    motif = str(policy["role_motif"][role])
    floor = _motif_floor(policy, role)
    rules = policy.get("override_rules") or {}

    if role in override:
        cls = str(override[role])
        if cls not in CLASSES:
            raise CriticalityError(f"override {family}/{role} has invalid class {cls!r}")
        if scope == "safety" and CLASS_ORDER[cls] < CLASS_ORDER[baseline]:
            raise CriticalityError(
                f"family {family!r} is a safety scope and may not downgrade role {role!r}"
            )
        source_rule = "family_class_override"
    else:
        cls = baseline
        source_rule = "role_baseline_class"

    enforced = set(rules.get("motif_floor_enforced_in_scopes") or [])
    if scope in enforced and CLASS_ORDER[cls] < CLASS_ORDER[floor]:
        raise CriticalityError(
            f"{family}/{role}: class {cls} is below its motif floor {floor}"
        )
    if scope == "non_safety_background" and role not in override:
        # not an error, but the policy must be explicit about every lowered role
        pass
    return {
        "semantic_role": role,
        "family_id": family,
        "motif_role": motif,
        "criticality": cls,
        "baseline_class": baseline,
        "motif_floor_class": floor,
        "safety_scope": scope,
        "source_rule": source_rule,
    }


def classify_template(policy: Mapping[str, Any], template: Mapping[str, Any]) -> dict[int, dict]:
    family = str(template.get("family_id"))
    out: dict[int, dict] = {}
    for task in template.get("tasks") or []:
        tid = int(task["task_id"])
        if tid in out:
            raise CriticalityError(f"{template.get('template_id')}: duplicate task id {tid}")
        out[tid] = classify(policy, str(task["semantic_role"]), family)
    return out


def resolve_demand_rule(w2: Mapping[str, Any], role: str) -> dict:
    """Return the workload_model_v2 task-class rule that charges `role`."""
    matches: list[dict] = []
    for rule in w2.get("task_class_rules") or []:
        t = rule.get("t_ref_s")
        if isinstance(t, Mapping):
            if role in t:
                matches.append({"kind": "fixed", "value_s": float(t[role]), "rule": rule})
            continue
        if role in (rule.get("applies_to_roles") or []):
            rng = rule.get("t_ref_s_range")
            if not rng or len(rng) != 2:
                raise CriticalityError(f"rule {rule.get('class_id')} lacks t_ref_s_range")
            matches.append({"kind": "range", "range_s": [float(rng[0]), float(rng[1])],
                            "rule": rule})
    if len(matches) != 1:
        raise CriticalityError(
            f"semantic role {role!r} must be charged exactly once, found {len(matches)}"
        )
    return matches[0]


def budget_for_role(policy: Mapping[str, Any], w2: Mapping[str, Any], role: str,
                    criticality: str) -> dict:
    """C_LO / C_HI as execution-demand budgets. HI exists for HIGH only."""
    if criticality not in CLASSES:
        raise CriticalityError(f"invalid criticality {criticality!r}")
    spec = policy.get("budgets") or {}
    fixed_rule = spec.get("fixed_point_rule") or {}
    range_rule = spec.get("range_rule") or {}
    demand = resolve_demand_rule(w2, role)
    if demand["kind"] == "fixed":
        c_lo = float(demand["value_s"])
        c_hi = c_lo * float(fixed_rule.get("hi_multiplier", 1.0))
        rule_id, evidence = (str(fixed_rule.get("rule_id")),
                             str(fixed_rule.get("evidence_class")))
    else:
        lo, hi = demand["range_s"]
        if not (math.isfinite(lo) and math.isfinite(hi) and 0.0 < lo <= hi):
            raise CriticalityError(f"role {role!r} has an invalid demand range {demand['range_s']!r}")
        c_lo = 0.5 * (lo + hi)
        c_hi = hi
        rule_id, evidence = (str(range_rule.get("rule_id")), str(range_rule.get("evidence_class")))
    if c_lo <= 0.0:
        raise CriticalityError(f"role {role!r}: C_LO must be > 0, got {c_lo!r}")
    if c_hi + 1e-15 < c_lo:
        raise CriticalityError(f"role {role!r}: C_HI {c_hi} < C_LO {c_lo}")
    hi_applicable = criticality in set(spec.get("classes_with_hi_budget") or [])
    return {
        "empirical_execution_budget_lo_s": c_lo,
        "empirical_execution_budget_hi_s": c_hi if hi_applicable else None,
        "budget_hi_applicable": bool(hi_applicable),
        "budget_hi_status": "applicable" if hi_applicable else "not_applicable",
        "budget_rule_id": rule_id,
        "budget_evidence_class": evidence,
        "demand_class_id": demand["rule"].get("class_id"),
        "wcet_claim": False,
    }


def build_task_table(policy: Mapping[str, Any], w2: Mapping[str, Any],
                     template: Mapping[str, Any]) -> dict[int, dict]:
    """Per-task criticality + budgets + declared drop/degrade policy for both modes."""
    classes = classify_template(policy, template)
    out: dict[int, dict] = {}
    for tid, c in sorted(classes.items()):
        b = budget_for_role(policy, w2, c["semantic_role"], c["criticality"])
        out[tid] = {**c, **b,
                    "drop_degrade_lo": drop_degrade(policy, c["criticality"], "LO"),
                    "drop_degrade_hi": drop_degrade(policy, c["criticality"], "HI")}
    return out


def drop_degrade(policy: Mapping[str, Any], criticality: str, mode: str) -> dict:
    if mode not in MODES:
        raise CriticalityError(f"unknown mode {mode!r}")
    table = (policy.get("mode_semantics") or {}).get("drop_degrade_policy") or {}
    if criticality not in table:
        raise CriticalityError(f"no drop/degrade policy for class {criticality!r}")
    row = dict(table[criticality])
    prefix = f"drop_allowed_{mode.lower()}_mode"
    degrade = f"degrade_allowed_{mode.lower()}_mode"
    return {
        "mode": mode,
        "drop_allowed": bool(row[prefix]),
        "degrade_allowed": bool(row[degrade]),
    }


def criticality_mixture(task_table: Mapping[int, dict]) -> dict:
    counts = {c: 0 for c in CLASSES}
    for row in task_table.values():
        counts[row["criticality"]] += 1
    return counts


@dataclass(frozen=True)
class ModeSwitch:
    triggering_task_id: int
    semantic_role: str
    criticality: str
    observed_execution: float
    C_LO: float
    reason: str
    previous_mode: str
    new_mode: str
    logical_schedule_point: int

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ModeTrace:
    initial_mode: str
    final_mode: str
    switches: tuple[ModeSwitch, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict:
        return {"initial_mode": self.initial_mode, "final_mode": self.final_mode,
                "switches": [s.as_dict() for s in self.switches]}


def simulate_mode_trace(policy: Mapping[str, Any], task_table: Mapping[int, dict],
                        observed_execution: Mapping[int, float],
                        order: Sequence[int]) -> ModeTrace:
    """Run the frozen LO/HI state machine over one schedule order.

    LO -> HI happens exactly once, on the first HIGH task whose observed execution
    exceeds its C_LO, and then stays HI until graph completion (v1). A non-HIGH
    overrun is recorded as no switch: it never changes the mode.
    """
    ms = policy.get("mode_semantics") or {}
    if list(ms.get("modes") or []) != list(MODES):
        raise CriticalityError("mode set must be exactly [LO, HI]")
    if str(ms.get("initial_mode")) != INITIAL_MODE:
        raise CriticalityError("initial mode must be LO")
    trigger = ms.get("lo_to_hi_trigger") or {}
    if str(trigger.get("triggering_class")) != "HIGH":
        raise CriticalityError("only HIGH may trigger LO -> HI")
    if str(ms.get("hi_exit")) != "sticky_until_graph_completion":
        raise CriticalityError("v1 requires HI to be sticky until graph completion")
    if list(order) == [] and task_table:
        raise CriticalityError("empty schedule order for a non-empty graph")

    mode = INITIAL_MODE
    switches: list[ModeSwitch] = []
    for point, tid in enumerate(order):
        tid = int(tid)
        if tid not in task_table:
            raise CriticalityError(f"schedule order references unknown task {tid}")
        if tid not in observed_execution:
            raise CriticalityError(f"no observed execution for task {tid}")
        row = task_table[tid]
        observed = float(observed_execution[tid])
        if not math.isfinite(observed) or observed < 0.0:
            raise CriticalityError(f"observed execution for task {tid} must be >= 0")
        overrun = observed > float(row["empirical_execution_budget_lo_s"]) + 1e-15
        if mode == "LO" and row["criticality"] == "HIGH" and overrun:
            switches.append(ModeSwitch(
                triggering_task_id=tid,
                semantic_role=str(row["semantic_role"]),
                criticality=str(row["criticality"]),
                observed_execution=observed,
                C_LO=float(row["empirical_execution_budget_lo_s"]),
                reason="high_task_exceeded_empirical_execution_budget_lo",
                previous_mode="LO",
                new_mode="HI",
                logical_schedule_point=point,
            ))
            mode = "HI"
        # HI is sticky: nothing else can move the mode back or sideways.
        if mode == "HI" and row["criticality"] == "HIGH" and not row["budget_hi_applicable"]:
            raise CriticalityError(f"HIGH task {tid} has no HI budget")
    return ModeTrace(initial_mode=INITIAL_MODE, final_mode=mode, switches=tuple(switches))


# --------------------------------------------------------------------------- #
# Diagnostics: prove class is not a function of position or deadline tightness
# --------------------------------------------------------------------------- #
def topological_depth(template: Mapping[str, Any]) -> dict[int, int]:
    ids = [int(t["task_id"]) for t in template["tasks"]]
    preds = {i: [] for i in ids}
    for e in template.get("edges") or []:
        preds[int(e["dst"])].append(int(e["src"]))
    depth: dict[int, int] = {}
    for i in sorted(ids):
        depth[i] = 0 if not preds[i] else 1 + max(depth[p] for p in preds[i])
    return depth


def _entropy(counts) -> float:
    total = sum(counts)
    if total <= 0:
        return 0.0
    h = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            h -= p * math.log2(p)
    return h


def conditional_entropy(pairs: Sequence[tuple[str, str]]) -> tuple[float, float]:
    """Return (H(x), H(x|y)) in bits over labelled pairs."""
    if not pairs:
        return 0.0, 0.0
    xs = [p[0] for p in pairs]
    h_x = _entropy([xs.count(v) for v in set(xs)])
    total = len(pairs)
    groups: dict[str, list[tuple[str, str]]] = {}
    for p in pairs:
        groups.setdefault(p[1], []).append(p)
    h_cond = 0.0
    for items in groups.values():
        sub = [i[0] for i in items]
        h_cond += (len(items) / total) * _entropy([sub.count(v) for v in set(sub)])
    return h_x, h_cond


def association(rows: Sequence[dict], key: str) -> dict:
    """Normalized mutual information between criticality and a probe variable."""
    pairs = [(str(r["criticality"]), str(r[key])) for r in rows]
    h_x, h_cond = conditional_entropy(pairs)
    nmi = 0.0 if h_x <= 0 else max(0.0, 1.0 - h_cond / h_x)
    return {
        "probe": key,
        "entropy_criticality_bits": h_x,
        "conditional_entropy_bits": h_cond,
        "normalized_mutual_information": nmi,
        "criticality_is_a_function_of_probe": bool(h_x > 0 and h_cond <= 1e-12),
        "levels_with_mixed_classes": sum(
            1 for lvl in {p[1] for p in pairs}
            if len({p[0] for p in pairs if p[1] == lvl}) > 1
        ),
    }
