#!/usr/bin/env python3
"""E4.1: the pure-plan evidence runner produces valid, gated evidence."""

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


from spec.pure_plan_evidence import PLANS, run  # noqa: E402


class TestPurePlanEvidence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evidence = run()

    def test_all_gates_pass(self):
        self.assertTrue(self.evidence["all_gates_pass"], self.evidence["gates"])
        for gate, ok in self.evidence["gates"].items():
            self.assertTrue(ok, gate)

    def test_four_plans_and_required_columns(self):
        plans = {row["plan"] for row in self.evidence["rows"]}
        self.assertEqual(plans, set(PLANS))
        required = {
            "latency_s", "requester_joules", "mobile_joules", "system_joules",
            "mec_compute_joules", "mec_tx_joules", "primary_scope",
            "primary_joules", "scheduler_config_sha256", "constraint_raw_system_j",
            "total_energy_with_budget", "total_energy_without_budget",
            "without_budget_spec_enabled", "without_budget_active_names",
            "without_budget_penalty", "with_budget_spec_enabled",
            "with_budget_active_names",
        }
        for row in self.evidence["rows"]:
            self.assertTrue(required <= set(row), required - set(row))
            self.assertEqual(row["primary_scope"], "system")
            self.assertEqual(row["primary_joules"], row["system_joules"])
            self.assertEqual(len(row["scheduler_config_sha256"]), 64)
            for key in ("latency_s", "requester_joules", "mobile_joules", "system_joules"):
                self.assertTrue(math.isfinite(row[key]))
                self.assertGreaterEqual(row[key], 0.0)
            self.assertLessEqual(row["requester_joules"], row["mobile_joules"] + 1e-9)
            self.assertLessEqual(row["mobile_joules"], row["system_joules"] + 1e-9)

    def test_mec_plans_actually_exercise_mec_energy(self):
        mec_rows = [r for r in self.evidence["rows"] if r["plan"] in ("all_MEC", "mixed")]
        for row in mec_rows:
            self.assertGreater(row["mec_compute_joules"], 0.0)
            self.assertGreater(row["mec_tx_joules"], 0.0)
            self.assertGreater(row["system_joules"], row["mobile_joules"])

    def test_not_configured_budget_is_reported_without_penalty(self):
        """Without a total-energy budget: not_configured, Lagrangian off, penalty 0.

        With a budget it is active. The two statuses come from two separate
        ConstraintSpec invocations, which the field names make explicit.
        """
        for row in self.evidence["rows"]:
            self.assertEqual(row["total_energy_without_budget"], "not_configured")
            self.assertFalse(row["without_budget_spec_enabled"])
            self.assertEqual(row["without_budget_active_names"], [])
            self.assertEqual(row["without_budget_penalty"], 0.0)
            self.assertEqual(row["total_energy_with_budget"], "active")
            self.assertTrue(row["with_budget_spec_enabled"])
            self.assertIn("total_energy", row["with_budget_active_names"])
            self.assertEqual(row["constraint_raw_system_j"], row["system_joules"])

    def test_schema_is_v2(self):
        self.assertEqual(self.evidence["schema"], "pure_plan_evidence_v2")

    def test_evidence_is_json_serializable(self):
        json.dumps(self.evidence)


if __name__ == "__main__":
    unittest.main(verbosity=2)
