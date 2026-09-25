#!/usr/bin/env python3
"""Constraint layer tests (pure numpy — no TF, runs on the laptop).

Covers:
  1. spec parsing + off-by-default no-op guarantees
  2. metric measurement on pure-location plans
  3. fractional budgets resolve against episode-local references
  4. terminal vs telescoped attribution sum identities
  5. Lagrangian dual ascent behaviour
  6. bit-exactness when constraints are off
"""

from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _stub_optional(name: str) -> None:
    if name in sys.modules:
        return
    try:
        __import__(name)
    except Exception:
        sys.modules[name] = types.ModuleType(name)


for _name in ("gym", "gym.core", "graphviz", "pydotplus", "pydotplus.graphviz"):
    _stub_optional(_name)
if not hasattr(sys.modules.get("gym.core", types.ModuleType("gym.core")), "Env"):
    sys.modules.setdefault("gym", types.ModuleType("gym"))
    sys.modules.setdefault("gym.core", types.ModuleType("gym.core"))
    sys.modules["gym.core"].Env = type("Env", (), {})
if not hasattr(sys.modules.get("graphviz", types.ModuleType("graphviz")), "Digraph"):
    sys.modules.setdefault("graphviz", types.ModuleType("graphviz"))
    sys.modules["graphviz"].Digraph = type("Digraph", (), {})

from env.mec_offloaing_envs.scheduler import (  # noqa: E402
    ALL_CONSTRAINTS,
    ATTRIBUTION_TELESCOPED,
    ATTRIBUTION_TERMINAL,
    CONSTRAINT_MODE_LAGRANGIAN,
    CONSTRAINT_MODE_OFF,
    ConstraintController,
    ConstraintSpec,
    ResourceConfig,
    compute_reference_ranges,
    compute_scoped_reference_ranges,
    costs_from_metrics,
    measure_metrics,
    pure_location_plan,
    schedule_via_adapter,
    telescoping_token_rewards,
)
from env.mec_offloaing_envs.scheduler.energy_scope import SCOPE_SYSTEM  # noqa: E402

N = 5


class _FakeTask:
    def __init__(self, proc: int, tx: int):
        self.processing_data_size = proc
        self.transmission_data_size = tx


class _FakeTG:
    """Chain graph with one sink so the terminal return hop is exercised."""

    def __init__(self, n: int = N):
        self.task_number = n
        self.task_list = [_FakeTask(2_097_152, 1_048_576) for _ in range(n)]
        self.prioritize_sequence = list(range(n))
        self.pre_task_sets = [set() if i == 0 else {i - 1} for i in range(n)]
        self.succ_task_sets = [set() if i == n - 1 else {i + 1} for i in range(n)]
        self.edge_set = [
            [i, i, 2_097_152, 1_048_576, i + 1, i + 1, 2_097_152] for i in range(n - 1)
        ]


def _resources() -> ResourceConfig:
    return ResourceConfig.from_frozen_yaml()


def _plan(tg, action: int):
    return pure_location_plan([int(t) for t in tg.prioritize_sequence], action)


class TestSpec(unittest.TestCase):
    def test_off_by_default(self):
        self.assertFalse(ConstraintSpec().enabled)
        self.assertEqual(ConstraintSpec().active_names, ())
        self.assertFalse(ConstraintSpec.from_config({}).enabled)
        self.assertFalse(ConstraintSpec.from_config({"constraints": None}).enabled)

    def test_active_names_order_is_stable(self):
        spec = ConstraintSpec(
            mode=CONSTRAINT_MODE_LAGRANGIAN,
            helper_energy_frac_of_all_ue=0.3,
            ue_energy_frac_of_all_ue=0.6,
            v2v_task_fraction_max=0.1,
        )
        self.assertTrue(spec.enabled)
        # canonical order, not declaration order
        self.assertEqual(spec.active_names, ("ue_energy", "helper_energy", "v2v_task_fraction"))

    def test_to_config_dict_round_trips_through_from_dict(self):
        spec = ConstraintSpec(
            mode=CONSTRAINT_MODE_LAGRANGIAN,
            attribution=ATTRIBUTION_TELESCOPED,
            total_energy_budget_j=1e12,
            ue_energy_frac_of_all_ue=0.5,
            v2v_airtime_budget_s=2.0,
        )
        cfg = spec.to_config_dict()
        # as_dict carries logging-only keys that from_dict rejects
        self.assertIn("active", spec.as_dict())
        self.assertIn("constraint_status", spec.as_dict())
        self.assertNotIn("active", cfg)
        self.assertNotIn("constraint_status", cfg)
        self.assertEqual(ConstraintSpec.from_dict(cfg), spec)
        # and from_config accepts it (the env path)
        self.assertEqual(ConstraintSpec.from_config({"constraints": cfg}), spec)

    def test_to_config_dict_of_a_disabled_spec_keeps_the_scenario(self):
        spec = ConstraintSpec(mode=CONSTRAINT_MODE_LAGRANGIAN)
        parsed = ConstraintSpec.from_config({"constraints": spec.to_config_dict()})
        self.assertEqual(parsed.mode, CONSTRAINT_MODE_LAGRANGIAN)
        self.assertFalse(parsed.enabled)
        self.assertEqual(parsed.constraint_status()["total_energy"], "not_configured")

    def test_mode_off_with_budgets_is_still_off(self):
        spec = ConstraintSpec(mode=CONSTRAINT_MODE_OFF, ue_energy_frac_of_all_ue=0.5)
        self.assertFalse(spec.enabled)

    def test_constraint_status_marks_unconfigured_constraints(self):
        from env.mec_offloaing_envs.scheduler.constraints import (
            C_TOTAL_ENERGY,
            C_UE_ENERGY,
        )

        spec = ConstraintSpec(mode=CONSTRAINT_MODE_LAGRANGIAN, ue_energy_frac_of_all_ue=0.5)
        status = spec.constraint_status()
        self.assertEqual(status[C_UE_ENERGY], "active")
        # E3.2: no budget -> explicit not_configured, not silently satisfied
        self.assertEqual(status[C_TOTAL_ENERGY], "not_configured")
        self.assertNotIn(C_TOTAL_ENERGY, spec.active_names)
        self.assertIn(C_TOTAL_ENERGY, ALL_CONSTRAINTS)
        off = ConstraintSpec()
        self.assertTrue(all(v == "not_configured" for v in off.constraint_status().values()))

    def test_unknown_key_rejected(self):
        with self.assertRaises(ValueError):
            ConstraintSpec.from_dict({"ue_energy_budget_j": 1.0, "nope": 2})

    def test_negative_budget_rejected(self):
        with self.assertRaises(ValueError):
            ConstraintSpec(ue_energy_budget_j=-1.0)

    def test_bad_attribution_rejected(self):
        with self.assertRaises(ValueError):
            ConstraintSpec(attribution="bogus")


class TestMetrics(unittest.TestCase):
    def setUp(self):
        self.tg = _FakeTG()
        self.res = _resources()
        self.refs = compute_scoped_reference_ranges(self.tg, self.res, energy_scope=SCOPE_SYSTEM)

    def test_all_ue_plan_energy_partition(self):
        result, _, _ = schedule_via_adapter(self.tg, _plan(self.tg, 0), self.res)
        m = measure_metrics(result)
        self.assertAlmostEqual(m.helper_energy_j, 0.0, places=9)
        self.assertAlmostEqual(m.ue_energy_j, self.refs.E_ue, places=6)
        self.assertAlmostEqual(m.total_energy_j, self.refs.E_ue, places=6)
        self.assertAlmostEqual(m.v2v_airtime_s, 0.0, places=9)
        self.assertEqual(m.n_helper_tasks, 0)

    def test_all_mec_plan_has_no_mobile_compute(self):
        result, _, _ = schedule_via_adapter(self.tg, _plan(self.tg, 1), self.res)
        m = measure_metrics(result)
        self.assertAlmostEqual(m.ue_energy_j, self.refs.E_mec, places=6)
        self.assertAlmostEqual(m.helper_energy_j, 0.0, places=9)
        self.assertEqual(m.n_helper_tasks, 0)

    def test_all_helper_plan_uses_helper_and_v2v(self):
        result, _, _ = schedule_via_adapter(self.tg, _plan(self.tg, 2), self.res)
        m = measure_metrics(result)
        self.assertEqual(m.n_helper_tasks, N)
        self.assertAlmostEqual(m.v2v_task_fraction, 1.0, places=9)
        self.assertGreater(m.v2v_airtime_s, 0.0)
        self.assertGreater(m.helper_energy_j, 0.0)
        self.assertAlmostEqual(m.total_energy_j, self.refs.E_helper, places=6)
        # helper cpu is charged to the helper (0.7 W) and its own tx/rx too
        self.assertGreater(m.helper_compute_j, 0.0)

    def test_partition_keeps_total_mobile_energy(self):
        for action in (0, 1, 2):
            result, _, _ = schedule_via_adapter(self.tg, _plan(self.tg, action), self.res)
            m = measure_metrics(result)
            self.assertAlmostEqual(
                m.ue_energy_j + m.helper_energy_j, result.total_mobile_joules, places=9
            )


class TestBudgets(unittest.TestCase):
    def setUp(self):
        self.tg = _FakeTG()
        self.res = _resources()
        self.refs = compute_scoped_reference_ranges(self.tg, self.res, energy_scope=SCOPE_SYSTEM)
        self.helper_plan = _plan(self.tg, 2)
        self.mec_plan = _plan(self.tg, 1)

    def _costs(self, plan, spec):
        result, _, _ = schedule_via_adapter(self.tg, plan, self.res)
        return costs_from_metrics(measure_metrics(result), self.refs, spec)

    def test_fractional_budget_is_relative_to_all_ue(self):
        spec = ConstraintSpec(
            mode=CONSTRAINT_MODE_LAGRANGIAN, helper_energy_frac_of_all_ue=0.25
        )
        costs = self._costs(self.helper_plan, spec)
        self.assertEqual(costs.names, ("helper_energy",))
        self.assertAlmostEqual(costs.budgets[0], 0.25 * self.refs.E_ue, places=6)
        self.assertGreater(costs.violations[0], 0.0)  # helper plan is expensive

    def test_mec_plan_respects_ue_energy_budget(self):
        spec = ConstraintSpec(mode=CONSTRAINT_MODE_LAGRANGIAN, ue_energy_frac_of_all_ue=0.5)
        costs = self._costs(self.mec_plan, spec)
        self.assertEqual(costs.violations[0], 0.0)
        self.assertLess(costs.signed[0], 0.0)

    def test_v2v_task_fraction_is_unit_scale(self):
        spec = ConstraintSpec(mode=CONSTRAINT_MODE_LAGRANGIAN, v2v_task_fraction_max=0.5)
        costs = self._costs(self.helper_plan, spec)
        self.assertAlmostEqual(costs.raw[0], 1.0, places=9)
        self.assertAlmostEqual(costs.signed[0], 0.5, places=9)

    def test_mec_task_fraction_detects_the_mec_shortcut(self):
        """ADR-001 makes all-MEC the argmin of L and E; the MEC cap must bite."""
        spec = ConstraintSpec(
            mode=CONSTRAINT_MODE_LAGRANGIAN,
            mec_task_fraction_max=0.5,
            helper_energy_frac_of_all_ue=0.35,
            v2v_task_fraction_max=0.15,
        )
        mec_costs = self._costs(self.mec_plan, spec)
        names = list(mec_costs.names)
        mec_idx = names.index("mec_task_fraction")
        self.assertAlmostEqual(mec_costs.raw[mec_idx], 1.0, places=9)
        self.assertAlmostEqual(mec_costs.signed[mec_idx], 0.5, places=9)
        # energy/V2V-only budgets are NOT violated by the all-MEC shortcut
        for name, violation in zip(mec_costs.names, mec_costs.violations):
            if name != "mec_task_fraction":
                self.assertEqual(violation, 0.0, msg=f"{name} unexpectedly violated by all-MEC")

    def test_mec_task_fraction_counts_executor_not_route(self):
        from env.mec_offloaing_envs.scheduler import measure_metrics

        result, _, _ = schedule_via_adapter(self.tg, self.mec_plan, self.res)
        m = measure_metrics(result)
        self.assertAlmostEqual(m.mec_task_fraction, 1.0, places=9)
        result_ue, _, _ = schedule_via_adapter(self.tg, _plan(self.tg, 0), self.res)
        self.assertAlmostEqual(measure_metrics(result_ue).mec_task_fraction, 0.0, places=9)

    def test_absolute_budget_wins_over_fraction(self):
        spec = ConstraintSpec(
            mode=CONSTRAINT_MODE_LAGRANGIAN,
            ue_energy_budget_j=1.0,
            ue_energy_frac_of_all_ue=0.9,
        )
        costs = self._costs(self.mec_plan, spec)
        self.assertAlmostEqual(costs.budgets[0], 1.0, places=9)


class TestRewardIntegration(unittest.TestCase):
    def setUp(self):
        self.tg = _FakeTG()
        self.res = _resources()
        self.plan = _plan(self.tg, 2)  # helper-heavy → violates helper/airtime budgets

    def test_off_is_bit_exact(self):
        base = telescoping_token_rewards(self.tg, self.plan, self.res)
        off_spec = ConstraintSpec(mode=CONSTRAINT_MODE_OFF, helper_energy_frac_of_all_ue=0.1)
        with_off = telescoping_token_rewards(self.tg, self.plan, self.res, constraints=off_spec)
        self.assertEqual(base.rewards, with_off.rewards)
        self.assertFalse(base.constraint_costs.active)
        self.assertEqual(base.constraint_penalty, 0.0)

    def test_terminal_attribution_lands_on_last_token(self):
        spec = ConstraintSpec(
            mode=CONSTRAINT_MODE_LAGRANGIAN,
            attribution=ATTRIBUTION_TERMINAL,
            helper_energy_frac_of_all_ue=0.1,
        )
        base = telescoping_token_rewards(self.tg, self.plan, self.res)
        out = telescoping_token_rewards(
            self.tg, self.plan, self.res, constraints=spec, duals=[2.0]
        )
        expected = 2.0 * out.constraint_costs.violations[0]
        self.assertAlmostEqual(out.constraint_penalty, expected, places=9)
        for i in range(len(out.rewards) - 1):
            self.assertAlmostEqual(out.rewards[i], base.rewards[i], places=12)
        self.assertAlmostEqual(
            out.rewards[-1], base.rewards[-1] - expected, places=9
        )

    def test_telescoped_attribution_matches_signed_delta_identity(self):
        spec = ConstraintSpec(
            mode=CONSTRAINT_MODE_LAGRANGIAN,
            attribution=ATTRIBUTION_TELESCOPED,
            helper_energy_frac_of_all_ue=0.1,
        )
        base = telescoping_token_rewards(self.tg, self.plan, self.res)
        out = telescoping_token_rewards(
            self.tg, self.plan, self.res, constraints=spec, duals=[1.5]
        )
        from env.mec_offloaing_envs.scheduler.constraints import (
            ConstraintMetrics,
            costs_from_metrics,
        )

        p0 = ConstraintMetrics(
            ue_energy_j=out.refs.E_ue,
            helper_energy_j=0.0,
            total_energy_j=out.refs.E_ue,
            helper_compute_j=0.0,
            v2v_airtime_s=0.0,
            v2v_task_fraction=0.0,
            makespan_s=out.refs.L_ue,
            n_tasks=N,
            n_helper_tasks=0,
        )
        base_signed = costs_from_metrics(p0, out.refs, spec).signed[0]
        identity = 1.5 * (out.constraint_costs.signed[0] - base_signed)
        self.assertAlmostEqual(out.constraint_penalty, identity, places=9)
        self.assertAlmostEqual(
            sum(out.rewards), sum(base.rewards) - identity, places=9
        )

    def test_duals_length_mismatch_raises(self):
        spec = ConstraintSpec(mode=CONSTRAINT_MODE_LAGRANGIAN, helper_energy_frac_of_all_ue=0.1)
        with self.assertRaises(ValueError):
            telescoping_token_rewards(self.tg, self.plan, self.res, constraints=spec, duals=[])


class TestController(unittest.TestCase):
    def _costs(self, signed):
        from env.mec_offloaing_envs.scheduler.constraints import ConstraintCosts

        return ConstraintCosts(
            names=("helper_energy",),
            raw=(signed + 1.0,),
            budgets=(1.0,),
            scales=(1.0,),
            signed=(signed,),
            violations=(max(0.0, signed),),
        )

    def test_dual_ascent_rises_on_violation(self):
        spec = ConstraintSpec(mode=CONSTRAINT_MODE_LAGRANGIAN, helper_energy_frac_of_all_ue=0.1)
        ctrl = ConstraintController(spec=spec, dual_lr=0.5)
        self.assertEqual(ctrl.lambdas, [0.0])
        ctrl.observe(self._costs(+2.0))
        ctrl.dual_step()
        self.assertAlmostEqual(ctrl.lambdas[0], 1.0, places=9)

    def test_dual_descent_when_satisfied_and_never_negative(self):
        spec = ConstraintSpec(mode=CONSTRAINT_MODE_LAGRANGIAN, helper_energy_frac_of_all_ue=0.1)
        ctrl = ConstraintController(spec=spec, dual_lr=0.5)
        ctrl.observe(self._costs(+1.0))
        ctrl.dual_step()
        self.assertAlmostEqual(ctrl.lambdas[0], 0.5, places=9)
        for _ in range(10):
            ctrl.observe(self._costs(-1.0))
            ctrl.dual_step()
        self.assertEqual(ctrl.lambdas[0], 0.0)

    def test_max_lambda_clip(self):
        spec = ConstraintSpec(mode=CONSTRAINT_MODE_LAGRANGIAN, helper_energy_frac_of_all_ue=0.1)
        ctrl = ConstraintController(spec=spec, dual_lr=1e6, max_lambda=10.0)
        ctrl.observe(self._costs(+5.0))
        ctrl.dual_step()
        self.assertEqual(ctrl.lambdas[0], 10.0)

    def test_penalty_uses_current_lambdas(self):
        spec = ConstraintSpec(mode=CONSTRAINT_MODE_LAGRANGIAN, helper_energy_frac_of_all_ue=0.1)
        ctrl = ConstraintController(spec=spec, dual_lr=1.0)
        ctrl.observe(self._costs(+1.0))
        ctrl.dual_step()
        self.assertAlmostEqual(ctrl.penalty(self._costs(+1.0)), 1.0, places=9)
        self.assertAlmostEqual(ctrl.penalty(self._costs(-1.0)), 0.0, places=9)

    def test_state_serializable(self):
        spec = ConstraintSpec(
            mode=CONSTRAINT_MODE_LAGRANGIAN,
            ue_energy_frac_of_all_ue=0.6,
            v2v_task_fraction_max=0.1,
        )
        ctrl = ConstraintController(spec=spec, dual_lr=0.05)
        state = ctrl.state()
        self.assertEqual(state["lambdas"], {"ue_energy": 0.0, "v2v_task_fraction": 0.0})
        self.assertEqual(state["updates"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
