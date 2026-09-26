#!/usr/bin/env python3
"""Phase P1: pure pilot-metric helpers."""

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


from env.mec_offloaing_envs.scheduler import (  # noqa: E402
    CanonicalDAG,
    CanonicalTask,
    schedule,
)
from env.mec_offloaing_envs.scheduler.calendar import RESOURCE_NAMES  # noqa: E402
from env.mec_offloaing_envs.scheduler.primary_config import (  # noqa: E402
    resolved_primary_scheduler_config,
)
from spec.pilot_metrics import (  # noqa: E402
    action_fractions,
    collapse_flag,
    flatten_actions,
    instability_reason,
    plan_summary,
    update_metric_kvs,
    validation_gaps,
)


def _dag():
    tasks = [
        CanonicalTask(0, 1_048_576, 524_288, 1_048_576),
        CanonicalTask(1, 1_048_576, 524_288, 0),
        CanonicalTask(2, 1_048_576, 524_288, 0),
    ]
    return CanonicalDAG.from_records(tasks, [(0, 1, 524_288), (0, 2, 524_288)])


class TestActionMix(unittest.TestCase):
    def test_fractions_sum_to_one(self):
        fracs = action_fractions([0, 0, 1, 2])
        self.assertAlmostEqual(fracs["local"], 0.5)
        self.assertAlmostEqual(fracs["mec"], 0.25)
        self.assertAlmostEqual(fracs["v2v"], 0.25)
        self.assertAlmostEqual(sum(fracs.values()), 1.0)

    def test_empty_actions_are_zero(self):
        self.assertEqual(action_fractions([]), {"local": 0.0, "mec": 0.0, "v2v": 0.0})

    def test_flatten_actions_walks_samples(self):
        class S:
            def __init__(self, actions):
                self._a = actions

            def get(self, key):
                return self._a if key == "actions" else None

        actions = flatten_actions([S([[0, 1], [2, 0]]), S([[1, 1]])])
        self.assertEqual(actions, [0, 1, 2, 0, 1, 1])


class TestPlanSummary(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dag = _dag()
        cls.res = resolved_primary_scheduler_config()
        cls.order = sorted(cls.dag.tasks)

    def test_counts_and_colocation(self):
        actions = [0, 1, 1]  # root local, two children MEC -> mixed
        result = schedule(self.dag, self.order, actions, self.res)
        edges = [[int(e.src_task_id), int(e.dst_task_id), int(e.edge_output_bytes)]
                 for e in self.dag.edges]
        summary = plan_summary(result, edges, len(self.order))
        self.assertAlmostEqual(summary["task_fraction/local"], 1 / 3)
        self.assertAlmostEqual(summary["task_fraction/mec"], 2 / 3)
        self.assertAlmostEqual(summary["task_fraction/v2v"], 0.0)
        self.assertEqual(summary["total_edges"], 2.0)
        self.assertEqual(summary["cross_location_edges"], 2.0)
        self.assertAlmostEqual(summary["co_location_rate"], 0.0)
        self.assertGreaterEqual(summary["utilization_mean"], 0.0)
        self.assertLessEqual(summary["utilization_max"], 1.0 + 1e-9)

    def test_all_colocated_when_single_location(self):
        result = schedule(self.dag, self.order, [1, 1, 1], self.res)
        edges = [[int(e.src_task_id), int(e.dst_task_id), int(e.edge_output_bytes)]
                 for e in self.dag.edges]
        summary = plan_summary(result, edges, len(self.order))
        self.assertAlmostEqual(summary["co_location_rate"], 1.0)
        self.assertEqual(summary["cross_location_edges"], 0.0)

    def test_utilization_covers_all_calendars(self):
        result = schedule(self.dag, self.order, [0, 2, 2], self.res)
        self.assertEqual(len(RESOURCE_NAMES), 6)
        summary = plan_summary(result, [], 3)
        self.assertGreaterEqual(summary["utilization_mean"], 0.0)


class TestGapsAndCollapse(unittest.TestCase):
    def test_gaps(self):
        gaps = validation_gaps(10.0, 12.0, 9.0)
        self.assertAlmostEqual(gaps["validation_gap_to_all_mec"], -2.0)
        self.assertAlmostEqual(gaps["validation_gap_to_greedy"], 1.0)
        self.assertAlmostEqual(gaps["validation_all_mec_latency"], 12.0)
        self.assertAlmostEqual(gaps["validation_greedy_latency"], 9.0)

    def test_non_finite_gap_rejected(self):
        with self.assertRaises(ValueError):
            validation_gaps(float("nan"), 1.0, 1.0)

    def test_collapse_needs_five_iterations(self):
        self.assertFalse(collapse_flag([0.99] * 4, 1.0, 1.0))

    def test_collapse_when_not_beating_baselines(self):
        self.assertTrue(collapse_flag([0.99] * 5, 0.0, 1.0))
        self.assertTrue(collapse_flag([0.99] * 5, 1.0, 0.0))

    def test_no_collapse_when_beating_both(self):
        self.assertFalse(collapse_flag([0.99] * 5, -1.0, -1.0))

    def test_no_collapse_when_share_drops(self):
        self.assertFalse(collapse_flag([0.99, 0.99, 0.99, 0.99, 0.5], 1.0, 1.0))

    def test_missing_gaps_do_not_collapse(self):
        self.assertFalse(collapse_flag([0.99] * 5, None, 1.0))


class TestIterationKvs(unittest.TestCase):
    def test_optional_metrics_are_omitted_when_absent(self):
        kvs = update_metric_kvs(actions=[0, 1])
        self.assertIn("action_fraction/mec", kvs)
        self.assertNotIn("policy/approx_kl", kvs)

    def test_present_metrics_are_finite_checked(self):
        kvs = update_metric_kvs(
            actions=[1], policy_loss_mean=0.5, value_loss_mean=0.25,
            approx_kl=0.01, clip_fraction=0.1, grad_norm=0.7,
        )
        for key in (
            "policy/policy_loss_mean", "policy/value_loss_mean",
            "policy/approx_kl", "policy/clip_fraction", "policy/grad_norm",
        ):
            self.assertIn(key, kvs)
        with self.assertRaises(ValueError):
            update_metric_kvs(actions=[0], grad_norm=float("inf"))


class TestInstabilityReason(unittest.TestCase):
    """Inline watchdog predicate shared by Trainer.pilot_watchdog."""

    def test_healthy_iteration_has_no_reason(self):
        self.assertIsNone(instability_reason(
            {"policy/policy_loss_mean": 0.5, "policy/grad_norm": 1.0}, 3.0))

    def test_nan_and_inf_are_rejected(self):
        for bad in (float("nan"), float("inf")):
            reason = instability_reason({"policy/approx_kl": bad}, 1.0)
            self.assertIsNotNone(reason)
            self.assertIn("policy/approx_kl", reason)

    def test_non_numeric_is_rejected(self):
        self.assertIsNotNone(instability_reason({"policy/grad_norm": None}))

    def test_value_abs_max_limit_is_inclusive(self):
        self.assertIsNone(instability_reason({}, 999.99, limit=1e3))
        self.assertIsNotNone(instability_reason({}, 1e3, limit=1e3))

    def test_missing_value_abs_max_is_not_a_stop(self):
        self.assertIsNone(instability_reason({"action_fraction/mec": 0.5}, None))

    def test_trainer_wires_the_inline_watchdog(self):
        source = (Path(__file__).resolve().parents[4] / "meta_trainer.py").read_text()
        self.assertIn("class PilotInstabilityError(RuntimeError)", source)
        self.assertIn("instability_reason(", source)
        self.assertIn("self.pilot_watchdog", source)
        # inline (inside the training loop), not a post-hoc scan of the CSV
        self.assertIn("raise PilotInstabilityError(", source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
