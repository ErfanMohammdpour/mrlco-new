#!/usr/bin/env python3
"""objective_contract_v1: one criterion for logging, selection and evaluation.

The contract is only meaningful if three things agree:
  * what PPO optimises (the discounted return of the token rewards),
  * what the trainer logs and uses to pick ``meta_model_best_val.ckpt``,
  * what the held-out evaluator reports.
These tests pin the maths and then pin the wiring by name.
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

import numpy as np  # noqa: E402

from spec.objective_contract import (  # noqa: E402
    METRIC_LEGACY,
    METRIC_PRIMARY,
    SAMPLES_DATA_KEYS,
    SCHEMA,
    SELECTION_METRIC_DEFAULT,
    affine_relation,
    contract_log_kvs,
    discounted_return,
    higher_is_better,
    legacy_sum_from_potentials,
    legacy_undiscounted_sum,
    ranking_agrees,
    selection_value,
    telescoped_from_potentials,
    terminal_objective_from_return,
)

GAMMA = 0.99


def _rewards_from_potentials(j, discount=GAMMA):
    j = np.asarray(j, dtype=np.float64)
    return np.asarray(
        [j[t - 1] - discount * j[t] for t in range(1, len(j))], dtype=np.float64
    )


class TestTelescopingMaths(unittest.TestCase):
    def test_discounted_return_equals_the_telescoped_objective(self):
        j = np.array([1.0, 0.93, 0.81, 0.6, 0.55, 0.4])
        rewards = _rewards_from_potentials(j)
        self.assertAlmostEqual(
            discounted_return(rewards.reshape(1, -1), GAMMA),
            telescoped_from_potentials(j, GAMMA),
            places=12,
        )

    def test_terminal_objective_inverts_the_identity(self):
        j = np.array([1.0, 0.93, 0.81, 0.6, 0.55, 0.4])
        rewards = _rewards_from_potentials(j)
        r = discounted_return(rewards.reshape(1, -1), GAMMA)
        n = len(j) - 1
        self.assertAlmostEqual(
            terminal_objective_from_return(r, j[0], GAMMA, n), j[-1], places=12
        )

    def test_legacy_sum_is_a_different_functional(self):
        j = np.array([1.0, 0.93, 0.81, 0.6, 0.55, 0.4])
        rewards = _rewards_from_potentials(j)
        legacy = legacy_undiscounted_sum(rewards.reshape(1, -1))
        self.assertAlmostEqual(legacy, legacy_sum_from_potentials(j, GAMMA), places=12)
        # analytic form J_0 + (1-g) sum_{t<N} J_t - g J_N
        expected = j[0] + (1 - GAMMA) * float(np.sum(j[1:-1])) - GAMMA * j[-1]
        self.assertAlmostEqual(legacy, expected, places=12)
        # and it is NOT the optimised objective
        self.assertNotAlmostEqual(
            legacy, telescoped_from_potentials(j, GAMMA), places=6
        )

    def test_affine_relation_between_return_and_terminal_objective(self):
        terminals = [0.80, 0.62, 0.55, 0.71]
        j0 = 1.0
        n = 20
        gamma_n = GAMMA ** n
        returns = [j0 - gamma_n * t for t in terminals]
        rel = affine_relation(returns, terminals, GAMMA, n)
        self.assertTrue(rel["exact"], rel)
        self.assertAlmostEqual(rel["implied_mean_j0"], j0, places=12)


class TestRankingJustification(unittest.TestCase):
    """The counterexample that motivated replacing the selection scalar.

    Both checkpoints are evaluated on the same graph (J_0 = 1.0, N = 2,
    gamma = 0.99). A has the nicer first token, B the slightly better terminal
    plan. The contract (discounted return / terminal objective) picks B; the old
    undiscounted sum picks A, because its prefix weight is 1-gamma = 0.01 while
    the terminal difference it must overcome is only ~0.001.
    """

    gamma = 0.99
    potentials = {"A": [1.0, 0.50, 0.701], "B": [1.0, 0.00, 0.700]}

    def _metrics(self, label):
        j = self.potentials[label]
        rewards = _rewards_from_potentials(j, self.gamma).reshape(1, -1)
        return (
            discounted_return(rewards, self.gamma),
            legacy_undiscounted_sum(rewards),
            float(j[-1]),
        )

    def test_primary_prefers_the_better_terminal_plan(self):
        objective_a, legacy_a, terminal_a = self._metrics("A")
        objective_b, legacy_b, terminal_b = self._metrics("B")
        # contract: B wins on the objective and on the terminal plan
        self.assertGreater(objective_b, objective_a)
        self.assertLess(terminal_b, terminal_a)
        # legacy scalar: A wins on a prefix difference the contract ignores
        self.assertGreater(legacy_a, legacy_b)
        # auditable: the objective is the telescoped terminal objective
        self.assertAlmostEqual(objective_a, 1 - self.gamma ** 2 * terminal_a, places=12)
        self.assertAlmostEqual(objective_b, 1 - self.gamma ** 2 * terminal_b, places=12)

    def test_primary_ranking_matches_the_terminal_objective_ranking(self):
        rows = [self._metrics(label) for label in ("A", "B")]
        returns = [row[0] for row in rows]
        legacy = [row[1] for row in rows]
        terminals = [row[2] for row in rows]
        self.assertTrue(ranking_agrees(returns, terminals, higher_is_better_b=False))
        self.assertFalse(ranking_agrees(returns, legacy))


class TestSelectionContract(unittest.TestCase):
    def test_selection_value_reads_the_declared_key(self):
        metrics = {
            "query_discounted_return": -0.25,
            "query_legacy_undiscounted_sum": -0.15,
        }
        value = selection_value(METRIC_PRIMARY, metrics)
        self.assertAlmostEqual(value, -0.25)
        self.assertAlmostEqual(selection_value(METRIC_LEGACY, metrics), -0.15)
        self.assertTrue(higher_is_better(METRIC_PRIMARY))

    def test_selection_value_rejects_unknown_and_non_finite(self):
        with self.assertRaises(ValueError):
            selection_value("not_a_metric", {})
        with self.assertRaises(ValueError):
            selection_value(METRIC_PRIMARY, {"query_discounted_return": float("nan")})
        with self.assertRaises(KeyError):
            selection_value(METRIC_PRIMARY, {})

    def test_default_selection_metric_is_the_primary(self):
        self.assertEqual(SELECTION_METRIC_DEFAULT, METRIC_PRIMARY)


class TestWiring(unittest.TestCase):
    """Name-level checks: the contract must be the one actually used."""

    def test_contract_log_kvs_are_self_describing(self):
        kvs = contract_log_kvs()
        self.assertEqual(kvs["objective_contract/schema"], SCHEMA)
        self.assertEqual(kvs["objective_contract/selection_metric"], SELECTION_METRIC_DEFAULT)
        self.assertEqual(kvs["objective_contract/reward_mode"], "latency_only")
        self.assertAlmostEqual(kvs["objective_contract/discount"], GAMMA)

    def test_trainer_logs_and_selects_on_the_same_scalar(self):
        src = (ROOT / "meta_trainer.py").read_text()
        self.assertIn("selection_value(selection_metric, k3)", src)
        self.assertIn('"checkpoint_selection_metric", "validation/objective_discounted_return"', src)
        # the saved best-val model must be chosen by the contract objective
        self.assertIn("self.best_val_objective is None or composite > self.best_val_objective", src)
        # ... and the choice must be recorded next to the checkpoint
        self.assertIn("meta_model_best_val.metric.json", src)
        self.assertIn('"contract": contract_schema', src)

    def test_evaluator_reports_the_contract_keys(self):
        protocol = (ROOT / "spec" / "eval_protocol.py").read_text()
        for key in SAMPLES_DATA_KEYS.values():
            self.assertIn(key, protocol)
        self.assertIn("discounted_return(rewards, gamma)", protocol)

    def test_declared_metric_matches_the_code(self):
        for rel in ("spec/frozen_experiment.yaml", "spec/phase4_campaign.yaml"):
            text = (ROOT / rel).read_text()
            self.assertIn(
                "checkpoint_selection_metric: validation/objective_discounted_return", text
            )
            self.assertIn("checkpoint_selection_contract: " + SCHEMA, text)


class TestSchedulerRewardAgrees(unittest.TestCase):
    """End-to-end on the real reward function, not just on the helpers."""

    def test_latency_only_reward_reproduces_the_contract(self):
        from env.mec_offloaing_envs.scheduler import (
            ResourceConfig,
            telescoping_token_rewards,
        )

        class _FakeTask:
            def __init__(self, proc, tx):
                self.processing_data_size = proc
                self.transmission_data_size = tx

        class _FakeTG:
            def __init__(self, n=6):
                self.task_number = n
                self.task_list = [_FakeTask(1_572_864, 786_432) for _ in range(n)]
                self.prioritize_sequence = list(range(n))
                self.pre_task_sets = [set() if i == 0 else {i - 1} for i in range(n)]
                self.succ_task_sets = [set() if i == n - 1 else {i + 1} for i in range(n)]
                self.edge_set = [
                    [i, i, 1_572_864, 786_432, i + 1, i + 1, 1_572_864]
                    for i in range(n - 1)
                ]

        tg = _FakeTG()
        res = ResourceConfig.from_frozen_yaml()
        plan = [(tid, k % 3) for k, tid in enumerate(tg.prioritize_sequence)]
        out = telescoping_token_rewards(
            tg, plan, res, reward_mode="latency_only", discount=GAMMA
        )
        j = [float(m) / float(out.refs.L_scale) for m in out.makespans]
        got = discounted_return(np.asarray(out.rewards, dtype=np.float64).reshape(1, -1), GAMMA)
        self.assertAlmostEqual(got, telescoped_from_potentials(j, GAMMA), places=9)
        self.assertAlmostEqual(
            terminal_objective_from_return(got, j[0], GAMMA, len(j) - 1), j[-1], places=9
        )
        # the reward's own potential for the final plan IS the terminal objective
        self.assertAlmostEqual(j[-1], float(out.final_makespan) / float(out.refs.L_scale), places=12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
