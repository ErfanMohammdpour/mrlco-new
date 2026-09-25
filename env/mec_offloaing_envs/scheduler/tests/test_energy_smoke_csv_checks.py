#!/usr/bin/env python3
"""E4.1/E4.2: the one-iteration smoke CSV validator enforces the new contract."""

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

from spec.mask_sanity import (  # noqa: E402
    ENERGY_JOULES_KEYS,
    METRIC_KEYS,
    TELEMETRY_KEYS,
    validate_progress_csv,
)


def _row(**overrides):
    row = {key: "0.0" for key in METRIC_KEYS}
    row.update(
        {
            "energy/requester_joules": "1.0",
            "energy/mobile_joules": "2.0",
            "energy/system_joules": "3.0",
            "energy/primary_joules": "3.0",
            "energy/primary_scope": "system",
            "Average energy,": "2.0",
        }
    )
    row.update(overrides)
    return row


class TestSmokeCsvContract(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="mask_sanity_csv_"))
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
        return keys

    def test_valid_row_passes(self):
        self._write([_row(**{"objective/J": "0.5"})])
        out = validate_progress_csv(self.tmp, 1, objective_mode="log_only")
        self.assertEqual(out["failures"], [], out["failures"])

    def test_missing_energy_column_fails(self):
        row = _row()
        row.pop("energy/system_joules")
        self._write([row])
        out = validate_progress_csv(self.tmp, 1)
        checks = {f["check"] for f in out["failures"]}
        self.assertIn("energy_column_present", checks)

    def test_objective_unavailable_fails(self):
        self._write([_row(**{"objective/J": "0.5", "objective/unavailable": "1.0"})])
        out = validate_progress_csv(self.tmp, 1, objective_mode="log_only")
        checks = {f["check"] for f in out["failures"]}
        self.assertIn("objective_not_unavailable", checks)

    def test_objective_columns_required_when_enabled(self):
        self._write([_row()])
        out = validate_progress_csv(self.tmp, 1, objective_mode="log_only")
        checks = {f["check"] for f in out["failures"]}
        self.assertIn("objective_columns_present", checks)

    def test_wrong_primary_scope_fails(self):
        self._write([_row(**{"energy/primary_scope": "mobile"})])
        out = validate_progress_csv(self.tmp, 1)
        checks = {f["check"] for f in out["failures"]}
        self.assertIn("energy_primary_scope_system", checks)

    def test_primary_not_system_fails(self):
        self._write([_row(**{"energy/primary_joules": "2.5"})])
        out = validate_progress_csv(self.tmp, 1)
        checks = {f["check"] for f in out["failures"]}
        self.assertIn("energy_primary_equals_system", checks)

    def test_boundary_order_violation_fails(self):
        self._write([_row(**{"energy/requester_joules": "5.0"})])
        out = validate_progress_csv(self.tmp, 1)
        checks = {f["check"] for f in out["failures"]}
        self.assertIn("energy_boundary_order", checks)

    def test_energy_keys_are_the_contract(self):
        self.assertEqual(len(ENERGY_JOULES_KEYS), 4)
        self.assertEqual(len(TELEMETRY_KEYS), 5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
