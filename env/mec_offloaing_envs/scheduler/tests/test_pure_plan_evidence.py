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
            "primary_joules", "scheduler_config_sha256", "total_energy_raw_system_j",
            "total_energy_status", "budget_status",
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
        for row in self.evidence["rows"]:
            self.assertEqual(row["budget_status"], "not_configured")
            self.assertEqual(row["total_energy_status"], "active")

    def test_evidence_is_json_serializable(self):
        json.dumps(self.evidence)


if __name__ == "__main__":
    unittest.main(verbosity=2)
