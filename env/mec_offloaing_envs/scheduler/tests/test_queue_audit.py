#!/usr/bin/env python3
"""Part B: queue/parallelism instrumentation + counterfactual prefix audit."""

from __future__ import annotations

import json
import math
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


from env.mec_offloaing_envs.scheduler.calendar import RESOURCE_NAMES  # noqa: E402
from spec.queue_audit import run  # noqa: E402


class TestQueueAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evidence = run()

    def test_all_checks_pass(self):
        self.assertTrue(self.evidence["all_checks_pass"], self.evidence["checks"])

    def test_five_graphs_and_five_plans(self):
        self.assertEqual(len(self.evidence["per_graph"]), 5)
        for entry in self.evidence["per_graph"]:
            self.assertEqual(
                set(entry["plans"]),
                {"all_UE", "all_MEC", "all_HELPER", "greedy_mixed", "two_opt_mixed"},
            )

    def test_every_resource_is_reported_with_sane_metrics(self):
        for entry in self.evidence["per_graph"]:
            for plan in entry["plans"].values():
                self.assertEqual(set(plan["resources"]), set(RESOURCE_NAMES))
                for name, row in plan["resources"].items():
                    self.assertGreaterEqual(row["busy_s"], 0.0)
                    self.assertGreaterEqual(row["idle_s"], 0.0)
                    self.assertGreaterEqual(row["utilization"], 0.0)
                    self.assertLessEqual(row["utilization"], 1.0 + 1e-9)
                    self.assertGreaterEqual(row["max_queue_delay_s"], 0.0)
                    self.assertGreaterEqual(row["p95_wait_s"], row["p50_wait_s"] - 1e-12)

    def test_per_task_fields_present(self):
        required = {
            "decoder_position", "task_id", "indegree", "outdegree", "depth",
            "ready_s", "compute_start_s", "compute_finish_s", "queue_delay_s",
            "all_consumers_ready_s", "energy_components",
        }
        for entry in self.evidence["per_graph"]:
            rows = entry["plans"]["two_opt_mixed"]["tasks"]
            self.assertEqual(len(rows), entry["n_tasks"])
            for row in rows:
                self.assertTrue(required <= set(row), required - set(row))
                self.assertGreaterEqual(row["queue_delay_s"], -1e-9)

    def test_counterfactual_is_prefix_only_with_base_suffix(self):
        for entry in self.evidence["per_graph"]:
            cf = entry["counterfactual"]
            rows = entry["counterfactual_rows"]
            self.assertEqual(len(rows), 3 * entry["n_tasks"])
            for row in rows:
                self.assertEqual(row["suffix_contract"], "base_plan_suffix")
                self.assertTrue(math.isfinite(row["actual_makespan_s"]))
                self.assertTrue(math.isfinite(row["static_bound_s"]))
                self.assertLessEqual(row["outdegree"], entry["n_tasks"])
            self.assertGreaterEqual(cf["static_vs_actual_spearman"], -1.0)
            self.assertLessEqual(cf["static_vs_actual_spearman"], 1.0)
            self.assertGreaterEqual(cf["mean_abs_static_error_s"], 0.0)

    def test_bottleneck_is_a_real_resource(self):
        self.assertIn(
            self.evidence["bottleneck_resource_by_critical_contribution"], RESOURCE_NAMES
        )

    def test_json_serializable(self):
        json.dumps(self.evidence)


if __name__ == "__main__":
    unittest.main(verbosity=2)
