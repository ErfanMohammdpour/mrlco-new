#!/usr/bin/env python3
"""M6-v2 (REV2): analytic deadline construction with a TRUE lower bound.

Changes vs v1: q_e^LB = 0 whenever legal co-location exists (so the fixed 20 Mbps
scenario is diagnostic only), D_G comes from the frozen family deadline D_f instead
of rho_f * P_f, and the critical path is decomposed into compute and
communication-lower-bound parts. No scheduler/search/witness import.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Mapping

DEADLINE_MODEL_VERSION = "deadline_model_v2"


class DeadlineError(ValueError):
    """Input refused. Never silently repaired."""


def _canon(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sla_config_sha256(sla2: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canon({
        "schema_version": sla2.get("schema_version"), "rule_id": sla2.get("rule_id"),
        "period_registry": sla2.get("period_registry"),
        "deadline_registry": sla2.get("deadline_registry"),
        "subdeadline_rule": sla2.get("subdeadline_rule"),
        "lower_bound_communication": sla2.get("lower_bound_communication"),
    }).encode()).hexdigest()


def workload_config_sha256(w2: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canon({
        "schema_version": w2.get("schema_version"),
        "reference_compute_model": w2.get("reference_compute_model"),
        "motif_aggregates": w2.get("motif_aggregates"),
        "task_class_rules": w2.get("task_class_rules"),
    }).encode()).hexdigest()


def _mid(rng, statistic="mean"):
    if not rng or len(rng) != 2:
        raise DeadlineError(f"invalid range {rng!r}")
    lo, hi = float(rng[0]), float(rng[1])
    if not (math.isfinite(lo) and math.isfinite(hi)) or lo <= 0 or hi < lo:
        raise DeadlineError(f"invalid range {rng!r}")
    return lo if statistic == "min" else (hi if statistic == "max" else 0.5 * (lo + hi))


def _pos(v, name):
    v = float(v)
    if not math.isfinite(v) or v <= 0:
        raise DeadlineError(f"{name} must be positive and finite, got {v!r}")
    return v


def task_reference_durations(template: Mapping[str, Any], w2: Mapping[str, Any], statistic="mean"):
    by_role: dict[str, float] = {}
    for rule in w2.get("task_class_rules") or []:
        t = rule.get("t_ref_s")
        if isinstance(t, Mapping):
            for role, val in t.items():
                by_role[role] = _pos(val, f"t_ref[{role}]")
            continue
        if t is None:
            t = _mid(rule.get("t_ref_s_range"), statistic)
        for role in rule.get("applies_to_roles") or []:
            by_role[role] = _pos(t, f"t_ref[{role}]")
    # motif aggregate invariant
    for name, agg in (w2.get("motif_aggregates") or {}).items():
        weights = agg.get("partition_weights") or {}
        roles = list(weights)
        total = sum(by_role.get(r, 0.0) for r in roles)
        if abs(total - float(agg["aggregate_s"])) > 1e-12:
            raise DeadlineError(f"motif {name} partition sum {total} != aggregate {agg['aggregate_s']}")
    out = {}
    for task in template.get("tasks") or []:
        role = task["semantic_role"]
        if role not in by_role:
            raise DeadlineError(f"role {role!r} has no v2 workload rule")
        out[int(task["task_id"])] = (role, by_role[role])
    return out


def edge_lower_bound_costs(template, w2):
    """q_e^LB: 0 when co-location is legal, else 8B/R_max_allowed. No calendar."""
    payloads = w2.get("payload_models") or {}
    costs, ref_costs = {}, {}
    rate_ref = float((w2.get("reference_scenario") or {}).get("rate_bps", 2.0e7))
    for edge in template.get("edges") or []:
        cls = edge.get("payload_class")
        model = payloads.get(cls)
        if not model:
            raise DeadlineError(f"payload class {cls!r} has no v2 model")
        b = _pos(model.get("nominal_construction_bytes"), f"payload[{cls}]")
        if model.get("allowed_co_location"):
            costs[(int(edge["src"]), int(edge["dst"]))] = 0.0
        else:
            rate = _pos(model.get("r_max_allowed_bps"), f"r_max[{cls}]")
            costs[(int(edge["src"]), int(edge["dst"]))] = 8.0 * b / rate
        ref_costs[(int(edge["src"]), int(edge["dst"]))] = 8.0 * b / rate_ref
    return costs, ref_costs


@dataclass(frozen=True)
class TaskDeadline:
    task_id: int
    role: str
    c_ref: float
    E: float
    L: float
    d: float
    slack: float


@dataclass(frozen=True)
class DeadlineResultV2:
    family_id: str
    P_f: float
    D_f: float
    D_G: float
    CP_ref: float
    CP_compute: float
    CP_communication_lb: float
    CP_reference_scenario: float
    alpha_f: float
    sla_config_sha256: str
    workload_config_sha256: str
    input_fingerprint: str
    output_sha256: str
    status: str
    tasks: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in
                ("family_id", "P_f", "D_f", "D_G", "CP_ref", "CP_compute",
                 "CP_communication_lb", "CP_reference_scenario", "alpha_f",
                 "sla_config_sha256", "workload_config_sha256",
                 "input_fingerprint", "output_sha256", "status")} | {
            "tasks": {str(k): {"role": v.role, "c_ref": v.c_ref, "E": v.E, "L": v.L,
                               "d": v.d, "slack": v.slack}
                      for k, v in sorted(self.tasks.items())}}


def generate_deadlines_v2(template, w2, sla2, family_id):
    periods = (sla2.get("period_registry") or {}).get("families") or {}
    deadlines = (sla2.get("deadline_registry") or {}).get("families") or {}
    if family_id not in deadlines or family_id not in periods:
        raise DeadlineError(f"unknown v2 SLA family {family_id!r}")
    if sla2.get("eta_b_g_branch_active"):
        raise DeadlineError("eta * B_G must stay inactive in v2")
    if family_id != template.get("family_id"):
        raise DeadlineError("template family and SLA family disagree")
    d_spec = deadlines[family_id]
    if d_spec.get("measurement_scope") != "application_e2e" or d_spec.get("semantic_role") != "requirement":
        raise DeadlineError("D_f must be an application_e2e requirement")
    P_f = _pos(periods[family_id]["P_f_s"], "P_f")
    D_f = _mid(d_spec.get("D_f_s_range"))
    D_G = D_f                                  # v2: D_G is the frozen deadline axis
    durations = task_reference_durations(template, w2)
    lb, ref = edge_lower_bound_costs(template, w2)
    ids = sorted(durations)
    succ = {i: [] for i in ids}; pred = {i: [] for i in ids}
    for (s, d) in lb:
        if s not in durations or d not in durations:
            raise DeadlineError(f"edge {s}->{d} is not a task")
        succ[s].append(d); pred[d].append(s)
    indeg = {i: len(pred[i]) for i in ids}; q = [i for i in ids if indeg[i] == 0]; order = []
    while q:
        n = q.pop(0); order.append(n)
        for m in succ[n]:
            indeg[m] -= 1
            if indeg[m] == 0: q.append(m)
    if len(order) != len(ids):
        raise DeadlineError("template contains a cycle")

    def longest(costs):
        ES, EF = {}, {}
        for i in order:
            es = 0.0
            for p in pred[i]:
                es = max(es, EF[p] + costs.get((p, i), 0.0))
            ES[i] = es; EF[i] = es + durations[i][1]
        return ES, EF

    compute_costs = {k: 0.0 for k in lb}
    _ES_c, EF_c = longest(compute_costs)
    ES, EF = longest(lb)
    _ES_r, EF_r = longest(ref)
    CP_compute = max(EF_c.values()); CP_lb = max(EF.values()); CP_ref_scn = max(EF_r.values())
    CP_ref = CP_lb
    CP_comm = CP_lb - CP_compute

    LF, LS = {}, {}
    for i in reversed(order):
        lf = D_G if not succ[i] else min(LS[j] - lb.get((i, j), 0.0) for j in succ[i])
        LF[i] = lf; LS[i] = lf - durations[i][1]

    fingerprint = hashlib.sha256(_canon({
        "deadline_model_version": DEADLINE_MODEL_VERSION,
        "template_id": template.get("template_id"), "family_id": family_id,
        "tasks": [[i, durations[i][0], durations[i][1]] for i in ids],
        "edges_lb": sorted([[s, d, lb[(s, d)]] for (s, d) in lb]),
        "sla_config_sha256": sla_config_sha256(sla2),
        "workload_config_sha256": workload_config_sha256(w2),
    }).encode()).hexdigest()

    alpha = float((sla2.get("subdeadline_rule") or {}).get("alpha_f", 0.5))
    status, tasks = "ok", {}
    if CP_ref > D_G + 1e-12:
        status = "infeasible_reference_lower_bound"
    else:
        for i in ids:
            E, L = EF[i], LF[i]
            if E > L + 1e-12:
                status = "infeasible_reference_lower_bound"; break
            slack = L - E
            if slack < -1e-12:
                raise DeadlineError(f"negative slack on task {i}")
            tasks[i] = TaskDeadline(i, durations[i][0], durations[i][1], E, L,
                                    E + alpha * max(0.0, slack), max(0.0, slack))
    res = DeadlineResultV2(family_id, P_f, D_f, D_G, CP_ref, CP_compute, CP_comm,
                           CP_ref_scn, alpha, sla_config_sha256(sla2),
                           workload_config_sha256(w2), fingerprint, "", status, tasks)
    out = res.as_dict()
    return DeadlineResultV2(**{**res.__dict__,
                               "output_sha256": hashlib.sha256(_canon(out).encode()).hexdigest()})
