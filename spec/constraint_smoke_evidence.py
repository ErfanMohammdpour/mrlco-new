#!/usr/bin/env python3
"""Part A constraint trainer-integration evidence (CPU, replay-free).

One trajectory, one ScheduleResult, one SYSTEM reference; the three constraint
scenarios are evaluated on that SAME result so no sampling difference can leak
into the comparison:

  A1 total-energy budget absent  -> not_configured, no penalty, Lagrangian off
  A2 system budget big           -> active, raw == system energy, violation 0
  A3 system budget small         -> active, raw == system energy, violation > 0

Lagrangian is OFF everywhere (duals are zero), so the primary latency-only
reward must be identical to the constraints-off reward for the same trajectory.

Usage: python3 -m spec.constraint_smoke_evidence [--json PATH]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from env.mec_offloaing_envs.scheduler.adapter import schedule_via_adapter
from env.mec_offloaing_envs.scheduler.constraints import (
    ConstraintSpec,
    costs_from_metrics,
    measure_metrics,
)
from env.mec_offloaing_envs.scheduler.energy_api import (
    compute_reference_ranges,
    compute_scoped_reference_ranges,
)
from env.mec_offloaing_envs.scheduler.energy_scope import (
    SCOPE_MOBILE,
    SCOPE_REQUESTER,
    SCOPE_SYSTEM,
    EnergyReferenceMismatch,
    energy_scalar,
    require_reference_scope,
)
from env.mec_offloaing_envs.scheduler.objective import (
    LATENCY_REF_ALL_UE,
    ObjectiveSpec,
    evaluate_plan_objective,
)
from env.mec_offloaing_envs.scheduler.primary_config import (
    resolved_primary_scheduler_config,
)
from env.mec_offloaing_envs.scheduler.reward import (
    REWARD_MODE_LATENCY_ONLY,
    telescoping_token_rewards,
)
from env.mec_offloaing_envs.scheduler.resources import resolved_config_sha256

PLAN = [(0, 0), (1, 2), (2, 1)]  # UE, HELPER, MEC
SCENARIOS = {
    "A1_total_absent": ConstraintSpec(mode="lagrangian"),
    "A2_total_big": ConstraintSpec(mode="lagrangian", total_energy_budget_j=1e12),
    "A3_total_small": ConstraintSpec(mode="lagrangian", total_energy_budget_j=1.0),
}


class _FakeTask:
    def __init__(self, proc: int, tx: int):
        self.processing_data_size = proc
        self.transmission_data_size = tx


class _FakeTG:
    def __init__(self):
        self.task_number = 3
        self.task_list = [
            _FakeTask(1_048_576, 458_752),
            _FakeTask(1_048_576, 458_752),
            _FakeTask(1_048_576, 458_752),
        ]
        self.prioritize_sequence = [0, 1, 2]
        self.pre_task_sets = [{}, {0}, {1}]
        self.edge_set = [
            [0, 0, 0, 458_752, 1, 1, 0],
            [1, 1, 0, 458_752, 2, 2, 0],
        ]


def _scenario_row(spec, metrics, refs) -> dict:
    costs = costs_from_metrics(metrics, refs, spec)
    values = dict(zip(costs.names, zip(costs.raw, costs.budgets, costs.signed, costs.violations)))
    row = {
        "enabled": bool(spec.enabled),
        "status": spec.constraint_status()["total_energy"],
        "active_names": list(spec.active_names),
        "total_energy_metric_system_j": float(metrics.total_energy_j),
        "penalty_with_zero_duals": float(costs.penalty([0.0] * len(costs.names))),
        "total_violation": float(costs.total_violation),
    }
    if "total_energy" in values:
        raw, budget, signed, violation = values["total_energy"]
        row.update({
            "total_energy_raw": float(raw),
            "total_energy_budget": float(budget),
            "total_energy_signed": float(signed),
            "total_energy_violation": float(violation),
        })
    else:
        row.update({
            "total_energy_raw": None,
            "total_energy_budget": None,
            "total_energy_signed": None,
            "total_energy_violation": None,
        })
    return row


def run() -> dict:
    tg = _FakeTG()
    res = resolved_primary_scheduler_config()
    refs = compute_scoped_reference_ranges(tg, res, energy_scope=SCOPE_SYSTEM)
    require_reference_scope(refs, expected_scope=SCOPE_SYSTEM)
    result, _, _ = schedule_via_adapter(tg, PLAN, res)
    metrics = measure_metrics(result)

    system_scalar = energy_scalar(result, scope=SCOPE_SYSTEM)
    requester_scalar = energy_scalar(result, scope=SCOPE_REQUESTER)
    mobile_scalar = energy_scalar(result, scope=SCOPE_MOBILE)
    fingerprint = str(resolved_config_sha256(res))

    rows = {name: _scenario_row(spec, metrics, refs) for name, spec in SCENARIOS.items()}

    def reward(spec, duals):
        return telescoping_token_rewards(
            tg, PLAN, res, reward_mode=REWARD_MODE_LATENCY_ONLY, refs=refs,
            constraints=spec, duals=duals,
        )

    off = reward(None, None)
    a1 = reward(SCENARIOS["A1_total_absent"], [])
    a2 = reward(SCENARIOS["A2_total_big"], [0.0])
    a3 = reward(SCENARIOS["A3_total_small"], [0.0])

    # C_UE_ENERGY keeps the requester boundary
    ue_spec = ConstraintSpec(mode="lagrangian", ue_energy_budget_j=1.0)
    ue_costs = costs_from_metrics(metrics, refs, ue_spec)
    ue_raw = dict(zip(ue_costs.names, ue_costs.raw))["ue_energy"]

    # mobile reference must be rejected by a system consumer
    mobile_refs = compute_reference_ranges(tg, res)
    objective_rejected = False
    try:
        evaluate_plan_objective(
            result, mobile_refs,
            ObjectiveSpec(latency_ref=LATENCY_REF_ALL_UE, energy_budget_j=1e12),
        )
    except EnergyReferenceMismatch:
        objective_rejected = True
    constraint_rejected = False
    try:
        costs_from_metrics(metrics, mobile_refs, SCENARIOS["A2_total_big"])
    except EnergyReferenceMismatch:
        constraint_rejected = True

    # Lagrangian OFF: a zero-learning-rate controller observes the costs but
    # never moves lambda, so no penalty can be applied.
    from env.mec_offloaing_envs.scheduler.constraints import ConstraintController

    ctrl = ConstraintController(spec=SCENARIOS["A3_total_small"], dual_lr=0.0)
    ctrl.observe(costs_from_metrics(metrics, refs, SCENARIOS["A3_total_small"]))
    ctrl.dual_step()

    checks = {
        "A1_status_not_configured": rows["A1_total_absent"]["status"] == "not_configured",
        "A1_controller_off": rows["A1_total_absent"]["enabled"] is False,
        "A1_no_active_constraint": rows["A1_total_absent"]["active_names"] == [],
        "A1_penalty_zero": rows["A1_total_absent"]["penalty_with_zero_duals"] == 0.0,
        "A1_raw_equals_system": rows["A1_total_absent"]["total_energy_metric_system_j"] == system_scalar,
        "A2_status_active": rows["A2_total_big"]["status"] == "active",
        "A2_raw_equals_system": rows["A2_total_big"]["total_energy_raw"] == system_scalar,
        "A2_under_budget": rows["A2_total_big"]["total_energy_signed"] < 0.0,
        "A2_violation_zero": rows["A2_total_big"]["total_energy_violation"] == 0.0,
        "A3_status_active": rows["A3_total_small"]["status"] == "active",
        "A3_raw_equals_system": rows["A3_total_small"]["total_energy_raw"] == system_scalar,
        "A3_violation_positive": rows["A3_total_small"]["total_energy_violation"] > 0.0,
        "A3_signed_positive": rows["A3_total_small"]["total_energy_signed"] > 0.0,
        "A3_penalty_zero": rows["A3_total_small"]["penalty_with_zero_duals"] == 0.0,
        "ue_raw_equals_requester": ue_raw == requester_scalar,
        "reward_A1_equals_off": a1.rewards == off.rewards,
        "reward_A2_equals_off": a2.rewards == off.rewards,
        "reward_A3_equals_off": a3.rewards == off.rewards,
        "reward_is_latency_only": a3.final_energy == energy_scalar(a3.final_result, scope=SCOPE_MOBILE),
        "mobile_ref_rejected_by_objective": objective_rejected,
        "mobile_ref_rejected_by_constraint": constraint_rejected,
        "lagrangian_off_lambda_stays_zero": ctrl.lambdas == [0.0],
        "lagrangian_off_penalty_zero": ctrl.penalty(
            costs_from_metrics(metrics, refs, SCENARIOS["A3_total_small"])
        ) == 0.0,
    }
    evidence = {
        "schema": "constraint_smoke_evidence_v1",
        "scheduler_config_sha256": fingerprint,
        "reference_scope": refs.energy_scope,
        "plan": [[int(t), int(a)] for t, a in PLAN],
        "system_joules": system_scalar,
        "mobile_joules": mobile_scalar,
        "requester_joules": requester_scalar,
        "scenarios": rows,
        "ue_energy_raw": ue_raw,
        "reward_latency_only_first_last": [a3.rewards[0], a3.rewards[-1]],
        "checks": checks,
        "all_checks_pass": all(checks.values()),
    }
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser()
    default = (
        Path(__file__).resolve().parents[1]
        / "reports" / "v0.3-audit" / "constraint_smoke" / "constraint_smoke_evidence.json"
    )
    parser.add_argument("--json", default=str(default))
    args = parser.parse_args()
    evidence = run()
    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    print(json.dumps(evidence["checks"], indent=1, sort_keys=True))
    print("all_checks_pass: %s" % evidence["all_checks_pass"])
    print("wrote %s" % out)
    return 0 if evidence["all_checks_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
