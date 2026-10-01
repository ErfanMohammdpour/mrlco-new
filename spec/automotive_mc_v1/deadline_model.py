#!/usr/bin/env python3
"""M6 deadline construction: analytic, deterministic, scheduler-free.

Ordering invariant enforced by construction: the SLA configuration is hashed and
frozen first, D_G is generated from it, task subdeadlines are derived from D_G, and
nothing here imports or accepts a scheduler/search/witness result.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

BASE = Path(__file__).resolve().parent
DEADLINE_MODEL_VERSION = "deadline_model_v1"


class DeadlineError(ValueError):
    """Deadline construction refused an input (never silently repaired)."""


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sla_config_sha256(sla_doc: Mapping[str, Any]) -> str:
    """Hash of the frozen SLA configuration (no witness, no scheduler state)."""
    frozen = {
        "schema_version": sla_doc.get("schema_version"),
        "rule_id": sla_doc.get("rule_id"),
        "branch": sla_doc.get("branch"),
        "eta_b_g_branch_active": sla_doc.get("eta_b_g_branch_active"),
        "deadline_construction_reference": sla_doc.get("deadline_construction_reference"),
        "subdeadline_rule": sla_doc.get("subdeadline_rule"),
        "families": sla_doc.get("families"),
    }
    return hashlib.sha256(_canonical(frozen).encode()).hexdigest()


def _require_positive(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise DeadlineError(f"{name} must be positive and finite, got {value!r}")
    return value


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
class DeadlineResult:
    family_id: str
    D_G: float
    CP_ref: float
    alpha_f: float
    sla_config_sha256: str
    input_fingerprint: str
    output_sha256: str
    status: str
    tasks: dict[int, TaskDeadline] = field(default_factory=dict)
    rho_used: float = 0.0
    period_used: float = 0.0

    def as_dict(self) -> dict:
        return {
            "family_id": self.family_id, "D_G": self.D_G, "CP_ref": self.CP_ref,
            "alpha_f": self.alpha_f, "sla_config_sha256": self.sla_config_sha256,
            "input_fingerprint": self.input_fingerprint,
            "output_sha256": self.output_sha256, "status": self.status,
            "rho_used": self.rho_used, "period_used": self.period_used,
            "tasks": {str(k): {"role": v.role, "c_ref": v.c_ref, "E": v.E,
                               "L": v.L, "d": v.d, "slack": v.slack}
                      for k, v in sorted(self.tasks.items())},
        }


def _quantile_from_range(rng, statistic: str) -> float:
    """Deterministic pick from a frozen range for a declared statistic."""
    if not rng or len(rng) != 2:
        raise DeadlineError(f"invalid frozen range {rng!r}")
    lo, hi = float(rng[0]), float(rng[1])
    if not (math.isfinite(lo) and math.isfinite(hi)) or lo <= 0 or hi < lo:
        raise DeadlineError(f"invalid frozen range {rng!r}")
    return lo if statistic == "min" else (hi if statistic == "max" else 0.5 * (lo + hi))


def task_reference_durations(template: Mapping[str, Any], workload_doc: Mapping[str, Any],
                             statistic: str = "mean") -> dict[int, tuple[str, float]]:
    """c_i^ref per task from the M5 workload model (mean statistic enforced)."""
    ref = workload_doc.get("reference_compute_model") or {}
    f_ref = _require_positive(ref.get("f_ref_hz"), "f_ref_hz")
    xi = _require_positive(ref.get("xi_cycles_per_bit"), "xi")
    if ref.get("reference_timing_statistic") != statistic:
        raise DeadlineError("workload statistic mismatch with the deadline construction model")
    by_role: dict[str, float] = {}
    for rule in workload_doc.get("task_class_rules") or []:
        t = rule.get("t_ref_s")
        if t is None:
            t = _quantile_from_range(rule.get("t_ref_s_range"), statistic)
        else:
            t = _require_positive(t, f"t_ref_s({rule.get('class_id')})")
        for role in rule.get("applies_to_roles") or []:
            by_role[role] = float(t)
    out: dict[int, tuple[str, float]] = {}
    for task in template.get("tasks") or []:
        role = task["semantic_role"]
        if role not in by_role:
            raise DeadlineError(f"role {role!r} has no workload rule")
        out[int(task["task_id"])] = (role, float(by_role[role]))
    return out


def edge_reference_costs(template: Mapping[str, Any], sla_doc: Mapping[str, Any]) -> dict[tuple[int, int], float]:
    """q_e^ref from payload class and the declared reference rate. No calendar."""
    ref = sla_doc.get("deadline_construction_reference") or {}
    if not ref.get("include_communication"):
        return {}
    rate = _require_positive(ref.get("reference_rate_bps"), "reference_rate_bps")
    sizes = ref.get("nominal_payload_bytes") or {}
    out: dict[tuple[int, int], float] = {}
    for edge in template.get("edges") or []:
        cls = edge.get("payload_class")
        if cls not in sizes:
            raise DeadlineError(f"payload class {cls!r} has no nominal construction size")
        out[(int(edge["src"]), int(edge["dst"]))] = 8.0 * float(sizes[cls]) / rate
    return out


def generate_deadlines(template: Mapping[str, Any], workload_doc: Mapping[str, Any],
                       sla_doc: Mapping[str, Any], family_id: str) -> DeadlineResult:
    """D_G, forward/backward bounds, subdeadlines and hashes. Scheduler-free."""
    families = sla_doc.get("families") or {}
    if family_id not in families:
        raise DeadlineError(f"unknown SLA family {family_id!r}")
    fam = families[family_id]
    if fam.get("branch") != "rho_period" or sla_doc.get("eta_b_g_branch_active"):
        raise DeadlineError("only the rho_f * P_f branch is active in v1")
    ref = sla_doc.get("deadline_construction_reference") or {}
    statistic = ref.get("timing_statistic", "mean")
    rho = _quantile_from_range(fam.get("rho_range"), statistic)
    period = _quantile_from_range(fam.get("period_s_range"), statistic)
    D_G = rho * period
    if family_id != template.get("family_id"):
        raise DeadlineError("template family and SLA family_id disagree")

    durations = task_reference_durations(template, workload_doc, statistic)
    for tid, (role, c) in durations.items():
        _require_positive(c, f"c_ref[{tid}]")
    costs = edge_reference_costs(template, sla_doc)
    for key, q in costs.items():
        if not math.isfinite(q) or q < 0.0:
            raise DeadlineError(f"negative reference transfer cost on {key}")

    ids = sorted(durations)
    succ: dict[int, list[int]] = {i: [] for i in ids}
    pred: dict[int, list[int]] = {i: [] for i in ids}
    for (src, dst) in costs:
        if src not in durations or dst not in durations:
            raise DeadlineError(f"edge endpoint {src}->{dst} is not a task")
        succ[src].append(dst)
        pred[dst].append(src)

    indeg = {i: len(pred[i]) for i in ids}
    queue = [i for i in ids if indeg[i] == 0]
    order: list[int] = []
    while queue:
        node = queue.pop(0)
        order.append(node)
        for nxt in succ[node]:
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                queue.append(nxt)
    if len(order) != len(ids):
        raise DeadlineError("template contains a cycle")

    ES: dict[int, float] = {}
    EF: dict[int, float] = {}
    for i in order:
        es = 0.0
        for p in pred[i]:
            es = max(es, EF[p] + costs.get((p, i), 0.0))
        ES[i] = es
        EF[i] = es + durations[i][1]
    CP_ref = max(EF.values()) if EF else 0.0

    sinks = [i for i in ids if not succ[i]]
    LF: dict[int, float] = {}
    LS: dict[int, float] = {}
    for i in reversed(order):
        if not succ[i]:
            lf = D_G
        else:
            lf = min(LS[j] - costs.get((i, j), 0.0) for j in succ[i])
        LF[i] = lf
        LS[i] = lf - durations[i][1]

    fingerprint = hashlib.sha256(_canonical({
        "deadline_model_version": DEADLINE_MODEL_VERSION,
        "template_id": template.get("template_id"),
        "family_id": family_id,
        "tasks": [[i, durations[i][0], durations[i][1]] for i in ids],
        "edges": sorted([[s, d, costs[(s, d)]] for (s, d) in costs]),
        "sla_config_sha256": sla_config_sha256(sla_doc),
    }).encode()).hexdigest()

    status = "ok"
    tasks: dict[int, TaskDeadline] = {}
    if CP_ref > D_G:
        # never repair D_G: report an explicit status and no invented deadlines
        status = "infeasible_reference_lower_bound"
    else:
        alpha = float((sla_doc.get("subdeadline_rule") or {}).get("alpha_f", 0.5))
        for i in ids:
            E, L = EF[i], LF[i]
            if E > L + 1e-12:
                status = "infeasible_reference_lower_bound"
                break
            slack = L - E
            if slack < -1e-12:
                raise DeadlineError(f"negative slack on task {i}")
            d = E + alpha * max(0.0, slack)
            tasks[i] = TaskDeadline(i, durations[i][0], durations[i][1], E, L, d, max(0.0, slack))

    partial = DeadlineResult(family_id=family_id, D_G=D_G, CP_ref=CP_ref,
                             alpha_f=float((sla_doc.get("subdeadline_rule") or {}).get("alpha_f", 0.5)),
                             sla_config_sha256=sla_config_sha256(sla_doc),
                             input_fingerprint=fingerprint, output_sha256="",
                             status=status, tasks=tasks, rho_used=rho, period_used=period)
    payload = dict(partial.as_dict())
    payload["tasks"] = {k: v for k, v in payload["tasks"].items()}
    out_hash = hashlib.sha256(_canonical(payload).encode()).hexdigest()
    return DeadlineResult(**{**partial.__dict__, "output_sha256": out_hash})
