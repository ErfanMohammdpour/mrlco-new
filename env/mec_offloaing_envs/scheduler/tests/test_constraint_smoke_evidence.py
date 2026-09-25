#!/usr/bin/env python3
"""Part A: constraint trainer-integration evidence and CSV contract."""

from __future__ import annotations

import csv
import sys
import tempfile
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
for _name in ("gym", "gym.core"):
    sys.modules.setdefault(_name, types.ModuleType(_name))
sys.modules["gym.core"].Env = type("Env", (), {})


from spec.constraint_smoke_evidence import SCENARIOS, run  # noqa: E402
from spec.mask_sanity import METRIC_KEYS, TELEMETRY_KEYS, validate_progress_csv  # noqa: E402


class TestConstraintFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evidence = run()

    def test_all_checks_pass(self):
        self.assertTrue(self.evidence["all_checks_pass"], self.evidence["checks"])

    def test_scenarios_are_distinct_specs(self):
        rows = self.evidence["scenarios"]
        self.assertEqual(set(rows), set(SCENARIOS))
        self.assertEqual(rows["A1_total_absent"]["status"], "not_configured")
        self.assertEqual(rows["A2_total_big"]["status"], "active")
        self.assertEqual(rows["A3_total_small"]["status"], "active")

    def test_raw_is_the_system_boundary_in_every_scenario(self):
        system = self.evidence["system_joules"]
        for name, row in self.evidence["scenarios"].items():
            self.assertEqual(row["total_energy_metric_system_j"], system, name)
        self.assertEqual(self.evidence["scenarios"]["A2_total_big"]["total_energy_raw"], system)
        self.assertEqual(self.evidence["scenarios"]["A3_total_small"]["total_energy_raw"], system)

    def test_ue_constraint_is_requester(self):
        self.assertEqual(self.evidence["ue_energy_raw"], self.evidence["requester_joules"])

    def test_mobile_reference_rejected(self):
        self.assertTrue(self.evidence["checks"]["mobile_ref_rejected_by_objective"])
        self.assertTrue(self.evidence["checks"]["mobile_ref_rejected_by_constraint"])

    def test_json_serializable(self):
        import json

        json.dumps(self.evidence)


def _row(**overrides):
    row = {key: "0.0" for key in METRIC_KEYS}
    row.update({
        "energy/requester_joules": "1.0",
        "energy/mobile_joules": "2.0",
        "energy/system_joules": "3.0",
        "energy/primary_joules": "3.0",
        "energy/primary_scope": "system",
        "constraint_status/total_energy": "not_configured",
        "constraint/penalty_applied": "0.0",
    })
    row.update(overrides)
    return row


class TestConstraintCsvContract(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="constraint_csv_"))
        self.logs = self.tmp / "logs"
        self.logs.mkdir(parents=True)

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmp, ignore_errors=True)

    def _write(self, rows):
        keys = sorted({k for row in rows for k in row})
        with (self.logs / "progress.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def test_absent_scenario_requires_not_configured_and_zero_penalty(self):
        self._write([_row()])
        out = validate_progress_csv(self.tmp, 1, constraints_scenario="total_absent")
        self.assertEqual(out["failures"], [], out["failures"])

    def test_active_scenario_requires_status_and_values(self):
        self._write([_row(**{
            "constraint_status/total_energy": "active",
            "constraint/total_energy_raw": "3.0",
            "constraint/total_energy_budget": "1e12",
            "constraint/total_energy_signed": "-1.0",
            "constraint/lambda_total_energy": "0.0",
        })])
        out = validate_progress_csv(self.tmp, 1, constraints_scenario="total_big")
        self.assertEqual(out["failures"], [], out["failures"])

    def test_small_scenario_requires_positive_violation(self):
        self._write([_row(**{
            "constraint_status/total_energy": "active",
            "constraint/total_energy_raw": "3.0",
            "constraint/total_energy_budget": "1.0",
            "constraint/total_energy_signed": "2.0",
            "constraint/lambda_total_energy": "0.0",
        })])
        out = validate_progress_csv(self.tmp, 1, constraints_scenario="total_small")
        self.assertEqual(out["failures"], [], out["failures"])

    def test_nonzero_penalty_or_lambda_fails(self):
        self._write([_row(**{
            "constraint_status/total_energy": "active",
            "constraint/total_energy_raw": "3.0",
            "constraint/total_energy_budget": "1.0",
            "constraint/total_energy_signed": "2.0",
            "constraint/lambda_total_energy": "0.5",
            "constraint/penalty_applied": "0.5",
        })])
        out = validate_progress_csv(self.tmp, 1, constraints_scenario="total_small")
        checks = {f["check"] for f in out["failures"]}
        self.assertIn("constraint_penalty_zero", checks)
        self.assertIn("constraint_lagrangian_off", checks)

    def test_missing_constraint_columns_fail(self):
        row = _row()
        row.pop("constraint_status/total_energy")
        row.pop("constraint/penalty_applied")
        self._write([row])
        out = validate_progress_csv(self.tmp, 1, constraints_scenario="total_absent")
        checks = {f["check"] for f in out["failures"]}
        self.assertIn("constraint_column_present", checks)


if __name__ == "__main__":
    unittest.main(verbosity=2)
