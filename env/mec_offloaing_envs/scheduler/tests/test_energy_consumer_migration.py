#!/usr/bin/env python3
"""E2.1: the consumer migration is a semantic no-op.

Every migrated path must keep the number it had before: reward energy, per-task
mobile attribution, mobile ReferenceRanges, j_report, constraint metrics and the
objective's system numerator. The objective/reference scope mismatch that exists
TODAY is characterised, not hidden.
"""

from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
for _name in ("gym", "gym.core"):
    sys.modules.setdefault(_name, types.ModuleType(_name))
sys.modules["gym.core"].Env = type("Env", (), {})


from env.mec_offloaing_envs.scheduler.adapter import schedule_via_adapter  # noqa: E402
from env.mec_offloaing_envs.scheduler.constraints import (  # noqa: E402
    C_TOTAL_ENERGY,
    C_UE_ENERGY,
    ConstraintSpec,
    costs_from_metrics,
    measure_metrics,
)
from env.mec_offloaing_envs.scheduler.energy_api import (  # noqa: E402
    attribute_mobile_energy_by_task,
    attribute_scoped_energy_by_task,
    compute_reference_ranges,
    j_report,
)
from env.mec_offloaing_envs.scheduler.energy_scope import (  # noqa: E402
    SCOPE_MOBILE,
    SCOPE_REQUESTER,
    SCOPE_SYSTEM,
    energy_scalar,
    require_reference_scope,
)
from env.mec_offloaing_envs.scheduler.objective import (  # noqa: E402
    LATENCY_REF_ALL_UE,
    ObjectiveSpec,
    evaluate_plan_objective,
)
from env.mec_offloaing_envs.scheduler.primary_config import (  # noqa: E402
    resolved_primary_scheduler_config,
)
from env.mec_offloaing_envs.scheduler.reward import (  # noqa: E402
    telescoping_token_rewards,
)


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


class ConsumerCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tg = _FakeTG()
        cls.res = resolved_primary_scheduler_config()
        cls.plan = [(0, 0), (1, 2), (2, 1)]  # one UE, one HELPER, one MEC
        cls.result, _, _ = schedule_via_adapter(cls.tg, cls.plan, cls.res)
        cls.refs = compute_reference_ranges(cls.tg, cls.res)
        require_reference_scope(cls.refs, expected_scope=SCOPE_MOBILE)


class TestBoundaryIdentities(ConsumerCase):
    def test_mobile_and_requester_are_the_old_numbers(self):
        self.assertAlmostEqual(
            energy_scalar(self.result, scope=SCOPE_MOBILE),
            self.result.energy.total_mobile_joules,
            places=12,
        )
        self.assertAlmostEqual(
            energy_scalar(self.result, scope=SCOPE_REQUESTER),
            self.result.energy.total_ue_joules,
            places=12,
        )
        self.assertAlmostEqual(
            energy_scalar(self.result, scope=SCOPE_SYSTEM),
            self.result.energy.total_system_joules,
            places=12,
        )

    def test_physical_plan_has_a_strictly_larger_system_boundary(self):
        # guarantees the objective/constraint migration is tested against a plan
        # where system != mobile
        self.assertGreater(
            energy_scalar(self.result, scope=SCOPE_SYSTEM),
            energy_scalar(self.result, scope=SCOPE_MOBILE),
        )


class TestReward(ConsumerCase):
    def test_reward_energies_are_mobile_scalars(self):
        out = telescoping_token_rewards(
            self.tg, self.plan, self.res, refs=self.refs
        )
        self.assertAlmostEqual(out.energies[0], self.refs.E_ue, places=12)
        self.assertAlmostEqual(
            out.energies[-1],
            energy_scalar(self.result, scope=SCOPE_MOBILE),
            places=12,
        )
        for t, energy in enumerate(out.energies):
            self.assertGreaterEqual(energy, 0.0)
            self.assertLessEqual(t, len(out.energies) - 1)

    def test_final_per_task_energy_is_mobile_attribution(self):
        out = telescoping_token_rewards(
            self.tg, self.plan, self.res, refs=self.refs
        )
        scoped = attribute_scoped_energy_by_task(
            out.final_result, self.res, scope=SCOPE_MOBILE
        )
        legacy = attribute_mobile_energy_by_task(out.final_result, self.res)
        for tid, value in scoped.items():
            self.assertAlmostEqual(value, legacy[tid], places=12)
        self.assertAlmostEqual(
            sum(out.final_per_task_energy),
            energy_scalar(out.final_result, scope=SCOPE_MOBILE),
            places=9,
        )

    def test_j_report_uses_the_mobile_scalar(self):
        old = j_report(self.result.makespan_seconds, self.result.total_mobile_joules, self.refs)
        new = j_report(
            self.result.makespan_seconds,
            energy_scalar(self.result, scope=SCOPE_MOBILE),
            self.refs,
        )
        self.assertAlmostEqual(old, new, places=12)


class TestConstraints(ConsumerCase):
    def test_metrics_are_the_old_boundaries(self):
        m = measure_metrics(self.result)
        self.assertAlmostEqual(m.total_energy_j, self.result.energy.total_mobile_joules, places=12)
        self.assertAlmostEqual(m.ue_energy_j, self.result.energy.total_ue_joules, places=12)
        self.assertAlmostEqual(m.helper_energy_j, self.result.energy.total_helper_joules, places=9)

    def test_ue_constraint_is_requester_and_total_constraint_is_mobile(self):
        m = measure_metrics(self.result)
        spec = ConstraintSpec(
            mode="lagrangian",
            ue_energy_budget_j=1.0,
            total_energy_budget_j=1.0,
        )
        costs = costs_from_metrics(m, self.refs, spec)
        raw = dict(zip(costs.names, costs.raw))
        self.assertIn(C_UE_ENERGY, raw)
        self.assertIn(C_TOTAL_ENERGY, raw)
        self.assertAlmostEqual(raw[C_UE_ENERGY], energy_scalar(self.result, scope=SCOPE_REQUESTER), places=12)
        self.assertAlmostEqual(raw[C_TOTAL_ENERGY], energy_scalar(self.result, scope=SCOPE_MOBILE), places=12)


class TestObjective(ConsumerCase):
    def test_objective_numerator_is_the_system_boundary(self):
        spec = ObjectiveSpec(
            latency_ref=LATENCY_REF_ALL_UE,
            energy_budget_frac_of_all_ue=2.0,
        )
        obj = evaluate_plan_objective(self.result, self.refs, spec)
        self.assertAlmostEqual(
            obj.energy_system_j,
            energy_scalar(self.result, scope=SCOPE_SYSTEM),
            places=12,
        )
        self.assertAlmostEqual(obj.energy_system_j, self.result.energy.total_system_joules, places=12)
        self.assertAlmostEqual(obj.energy_budget_j, 2.0 * self.refs.E_ue, places=9)

    def test_objective_reference_mismatch_is_real_and_recorded(self):
        """E3 fixes this; E2.1 must not hide it."""
        spec = ObjectiveSpec(
            latency_ref=LATENCY_REF_ALL_UE,
            energy_budget_frac_of_all_ue=2.0,
        )
        obj = evaluate_plan_objective(self.result, self.refs, spec)
        # the reference/budget were built at the MOBILE boundary ...
        self.assertEqual(self.refs.energy_scope, SCOPE_MOBILE)
        # ... while the numerator is measured at SYSTEM
        self.assertNotAlmostEqual(
            obj.energy_system_j,
            energy_scalar(self.result, scope=SCOPE_MOBILE),
            places=6,
        )
        self.assertAlmostEqual(
            obj.energy_budget_j, spec.energy_budget_frac_of_all_ue * self.refs.E_ue, places=9
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
