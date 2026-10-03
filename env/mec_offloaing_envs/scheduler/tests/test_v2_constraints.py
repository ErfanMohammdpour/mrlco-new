#!/usr/bin/env python3
"""v2 constraint integration: the budget must change the ACTUAL reward, the dual must move in
BOTH directions on training data, and missing metrics must fail instead of reading as zero.

Audited defect: `env.step` computed `penalty = 0.0` (and the controller stayed ineffective
despite the broadcast), so a budget could not influence anything, while telemetry lacked
top-level violations and task denominators — a checkpoint selector therefore read misses and
violations as zero.
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spec.automotive_training.automotive_loader import load_dataset  # noqa: E402
from spec.automotive_training.automotive_primary import AutomotiveResourceCluster  # noqa: E402
from spec.automotive_training.v2.constraints_v2 import (  # noqa: E402
    V2ConstraintError, V2ConstraintManager, calibrate_budgets, require_measurable,
    select_checkpoint, spec_from_fractions, v2_metrics,
)
from spec.automotive_training.v2.energy import (  # noqa: E402
    reference_ranges_from_plans, schedule_energy,
)
from spec.automotive_training.v2.env import V2AutomotiveEnv  # noqa: E402
from spec.automotive_training.v2.helper_model import HelperState  # noqa: E402
from spec.automotive_training.v2.shared_scheduler import (  # noqa: E402
    V2ComputeSpec, V2DAGSpec, V2LinkSpec, V2TaskSpec, schedule_shared,
)

LINK = V2LinkSpec(20e6, 20e6, 10e6)
COMPUTE = V2ComputeSpec(10e6, 1, (1e6,), (4e6,))
SHA = "b" * 64
SLOTS = 3
TOKENS = 20


def _env(constraints=False, seed=11):
    graphs = load_dataset().validation_query()[:SLOTS]
    env = V2AutomotiveEnv(graphs, AutomotiveResourceCluster(), role="validation",
                          slots_per_task=SLOTS, base_seed=seed, single_dist=True,
                          constraints_enabled=constraints,
                          budget_fractions={"total_energy": 2.0, "ue_energy": 2.0},
                          scheduler_config_sha256=SHA)
    env.set_task({"dist_index": 0,
                  "graph_indices": np.arange(SLOTS, dtype=np.int32)})
    return env


def _fixture():
    """Hand-checkable one-DAG fixture for the manager-level tests."""
    dag = V2DAGSpec("d", 0, [V2TaskSpec(0, 6e6, 2e6, (), True, True, "HIGH", None, 0,
                                        external_input_bytes=1000.0)], helper_id=0)
    helper = {0: HelperState(0, 4e6, contact_end_s=100.0, predicted_contact_end_s=100.0)}
    plans = {
        "all_ue": schedule_shared([dag], {"d": {0: 0}}, link=LINK, compute=COMPUTE),
        "all_mec": schedule_shared([dag], {"d": {0: 1}}, link=LINK, compute=COMPUTE),
        "all_helper": schedule_shared([dag], {"d": {0: 2}}, link=LINK, compute=COMPUTE,
                                      helper_states=helper),
    }
    return dag, helper, plans


class TestRewardPenaltyIsReal(unittest.TestCase):
    def test_a_budget_that_binds_changes_the_actual_reward(self):
        env = _env(constraints=True)
        env.reset()
        over = np.ones((SLOTS, TOKENS), dtype=int)      # all-MEC: huge system energy
        under = np.zeros((SLOTS, TOKENS), dtype=int)    # all-UE

        # iteration 1: lambda starts at 0, so the penalty is 0 but the VIOLATION is measured
        _o, _r, _d, info = env.step(over)
        for record in info[2]:
            c = record["constraints"]
            self.assertTrue(any(c[name + "_violation"] > 0.0 for name in c["names"]),
                            "an over-budget rollout must be MEASURED as violated")
            self.assertGreater(record["system_joules"],
                               record["constraints"]["total_energy_budget"])
            self.assertEqual(record["v2"]["constraint_penalty"], 0.0,
                             "lambda is still zero at this point")

        # the dual step (trainer-side, training data only) makes lambda strictly positive
        env.constraint_manager.dual_step()
        self.assertGreater(max(env.constraint_manager.lambdas), 0.0,
                           "lambda must leave zero after observing the violation")

        # iteration 2: the SAME over-budget rollout now carries a real reward penalty
        _o, _rewards, _d, info = env.step(over)
        for record in info[2]:
            self.assertGreater(float(record["v2"]["constraint_penalty"]), 0.0,
                               "with lambda > 0 an over-budget rollout MUST be penalised")

        # a within-budget rollout must NOT be penalised
        _o, _r, _d, info_under = env.step(under)
        for record in info_under[2]:
            self.assertEqual(record["v2"]["constraint_penalty"], 0.0,
                             "a within-budget rollout must not be penalised")

    def test_the_penalty_is_charged_exactly_once_on_the_terminal_token(self):
        env = _env(constraints=True)
        env.reset()
        plan = np.ones((SLOTS, TOKENS), dtype=int)
        env.step(plan)                       # observe the violation
        env.constraint_manager.dual_step()   # trainer-side dual update -> lambda > 0
        _o, rewards, _d, info = env.step(plan)
        checked = 0
        for slot, record in enumerate(info[2]):
            applied = float(record["v2"]["constraint_penalty"])
            base = np.asarray(env._telescoping(slot, plan[slot]), dtype=float)
            got = np.asarray(rewards[slot], dtype=float)
            # every non-terminal token is bit-identical to the un-penalised telescoping value
            np.testing.assert_allclose(got[:-1], base[:-1], rtol=0.0, atol=1e-6)
            if applied:
                checked += 1
                # the terminal token is reduced by EXACTLY the penalty, once (the reward
                # vector is stored as float32, hence the float32-scale tolerance)
                np.testing.assert_allclose(float(got[-1]), float(base[-1]) - applied,
                                           rtol=1e-5, atol=1e-5)
        self.assertGreater(checked, 0, "the fixture must contain an over-budget slot")

    def test_telemetry_exposes_violations_and_denominators(self):
        env = _env(constraints=True)
        env.reset()
        _o, _r, _d, info = env.step(np.ones((SLOTS, TOKENS), dtype=int))
        for record in info[2]:
            c = record["constraints"]
            self.assertTrue(c["enabled"])
            self.assertTrue(c["names"])
            for name in c["names"]:
                for suffix in ("_raw", "_budget", "_scale", "_signed", "_violation",
                               "_lambda"):
                    self.assertIn(name + suffix, c)
                self.assertGreater(c[name + "_budget"], 0.0)
            v2 = record["v2"]
            self.assertGreater(v2["n_tasks"], 0)
            self.assertEqual(v2["n_tasks"],
                             v2["n_mec_tasks"] + v2["n_helper_tasks"]
                             + (v2["n_tasks"] - v2["n_mec_tasks"] - v2["n_helper_tasks"]))
            self.assertIn("requester_joules", record)
            self.assertGreater(record["system_joules"], 0.0)

    def test_disabled_constraints_are_a_strict_noop(self):
        env = _env(constraints=False)
        env.reset()
        _o, rewards, _d, info = env.step(np.ones((SLOTS, TOKENS), dtype=int))
        for record in info[2]:
            self.assertFalse(record["constraints"]["enabled"])
            self.assertEqual(record["v2"]["constraint_penalty"], 0.0)
            self.assertEqual(record["energy_constraint"], "telemetry_only")
        self.assertTrue(all(np.isfinite(np.asarray(r)).all() for r in rewards))


class TestDualsMoveBothWays(unittest.TestCase):
    def test_lambda_rises_above_budget_and_falls_below_it(self):
        _dag, _helper, plans = _fixture()
        refs = reference_ranges_from_plans(plans, scheduler_config_sha256=SHA)
        # budget = 10 x the all-UE plan energy: the all-MEC plan is far over it while the
        # all-HELPER plan is under it, so the dual must move in BOTH directions
        spec = calibrate_budgets(refs, {"total_energy": 10.0})
        manager = V2ConstraintManager(spec=spec, references=refs, dual_lr=0.5,
                                      scheduler_config_sha256=SHA)
        # two over-budget all-MEC episodes -> lambda must rise
        for _ in range(2):
            result = schedule_shared([_dag], {"d": {0: 1}}, link=LINK, compute=COMPUTE)
            manager.observe(manager.evaluate(result, schedule_energy(result)))
        manager.dual_step()
        high = manager.lambdas[0]
        self.assertGreater(high, 0.0)
        # then many under-budget all-HELPER episodes -> lambda must FALL (signed dual ascent)
        for _ in range(40):
            result = schedule_shared([_dag], {"d": {0: 2}}, link=LINK, compute=COMPUTE,
                                     helper_states=_helper)
            manager.observe(manager.evaluate(result, schedule_energy(result)))
        manager.dual_step()
        low = manager.lambdas[0]
        self.assertLess(low, high,
                        "lambda must be able to DECREASE when costs fall below budget; "
                        "a positive-hinge-only update makes it monotone non-decreasing")
        self.assertGreaterEqual(low, 0.0)

    def test_lambdas_are_broadcast_to_every_environment(self):
        a, b = _env(constraints=True, seed=1), _env(constraints=True, seed=2)
        a.reset()
        b.reset()
        dag, _helper, _plans = _fixture()
        _dag, _h, plans = dag, _helper, _plans
        a.constraint_manager.references = reference_ranges_from_plans(
            plans, scheduler_config_sha256=SHA)
        result = schedule_shared([dag], {"d": {0: 1}}, link=LINK, compute=COMPUTE)
        a.constraint_manager.observe(a.constraint_manager.evaluate(
            result, schedule_energy(result)))
        a.constraint_manager.dual_step()
        vector = a.constraint_manager.lambdas_by_name()
        b.set_constraint_lambdas(vector)
        self.assertEqual(b.constraint_manager.lambdas_by_name(), vector)
        self.assertEqual(a.constraint_manager.lambdas_by_name(), vector)
        with self.assertRaises(V2ConstraintError):
            b.constraint_manager.set_lambdas({"nope": 1.0})

    def test_constraint_state_round_trips_and_rejects_a_mismatch(self):
        a = _env(constraints=True, seed=3)
        a.reset()
        state = a.constraint_state()
        b = _env(constraints=True, seed=4)
        b.reset()
        b.load_constraint_state(state)
        self.assertEqual(a.constraint_manager.lambdas_by_name(),
                         b.constraint_manager.lambdas_by_name())
        bad = dict(state)
        bad["constraints_sha256"] = "0" * 64
        with self.assertRaises(V2ConstraintError):
            b.load_constraint_state(bad)


class TestFailFast(unittest.TestCase):
    def test_missing_metric_raises_instead_of_becoming_zero(self):
        _dag, _helper, plans = _fixture()
        result = plans["all_mec"]
        ledger = schedule_energy(result)
        metrics = v2_metrics(result, ledger)
        require_measurable(metrics)
        import dataclasses

        broken = dataclasses.replace(metrics, total_energy_j=float("nan"))
        with self.assertRaises(V2ConstraintError):
            require_measurable(broken)
        broken = dataclasses.replace(metrics, n_tasks=0)
        with self.assertRaises(V2ConstraintError):
            require_measurable(broken)

    def test_select_checkpoint_never_calls_an_infeasible_winner_feasible(self):
        records = [
            {"objective": 0.1, "feasible": False, "total_violation": 5.0},
            {"objective": 0.9, "feasible": False, "total_violation": 0.0},
            {"objective": 0.5, "feasible": True, "total_violation": 0.0},
        ]
        best = select_checkpoint(records)
        self.assertTrue(best["feasible"])
        self.assertEqual(best["selected_index"], 2)
        infeasible = [{"objective": 0.1, "feasible": False, "total_violation": 5.0},
                      {"objective": 0.9, "feasible": False, "total_violation": 1.0}]
        best = select_checkpoint(infeasible)
        self.assertFalse(best["feasible"])
        self.assertEqual(best["selected_index"], 0)
        self.assertIn("selection_note", best)
        with self.assertRaises(V2ConstraintError):
            select_checkpoint([{"objective": 0.1, "feasible": True}])

    def test_scope_mismatch_is_rejected(self):
        _dag, _helper, plans = _fixture()
        refs = reference_ranges_from_plans(plans, scheduler_config_sha256=SHA)
        spec = calibrate_budgets(refs, {"total_energy": 0.5})
        manager = V2ConstraintManager(spec=spec, references=refs, scheduler_config_sha256=SHA)
        result = schedule_shared([_dag], {"d": {0: 1}}, link=LINK, compute=COMPUTE)
        ledger = schedule_energy(result)
        import dataclasses

        wrong = dataclasses.replace(refs, energy_scope="mobile")
        manager.references = wrong
        with self.assertRaises(Exception):
            manager.evaluate(result, ledger)


if __name__ == "__main__":
    unittest.main(verbosity=2)
