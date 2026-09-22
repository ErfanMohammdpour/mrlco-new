#!/usr/bin/env python3
"""Tests for the ⑥b stop-rule watchdog (spec/mask_run_watchdog.py).

The CSV snapshot check and the stall decision are pure functions, so the stop
rules can be verified without docker or a GPU. The kill path itself is exercised
only in --dry-run on the server.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spec.mask_run_watchdog import (  # noqa: E402
    check_rows,
    disappeared_violation,
    stall_violation,
)
from spec.mask_sanity import METRIC_KEYS  # noqa: E402

HEADER = ["Itr"] + list(METRIC_KEYS) + ["mask/late_key"]


def row(**overrides):
    values = {"Itr": "0"}
    for key in METRIC_KEYS:
        values[key] = "0.0"
    values["policy/entropy_valid"] = "1.05"
    values["critic/value_abs_max"] = "0.3"
    values["mask/late_key"] = "0.0"
    values.update(overrides)
    return [values.get(k, "") for k in HEADER]


class TestCsvSnapshotChecks(unittest.TestCase):
    def test_clean_snapshot(self):
        self.assertEqual(check_rows(HEADER, [row(), row()]), [])

    def test_missing_metric_column(self):
        header = [k for k in HEADER if k != "mask/forced_rate"]
        found = check_rows(header, [[v for k, v in zip(HEADER, row()) if k != "mask/forced_rate"]])
        self.assertTrue(any(v["check"] == "metric_column_missing" for v in found))

    def test_shifted_row_is_detected(self):
        broken = row()[:-1]                     # the failure the CSV logger had
        found = check_rows(HEADER, [broken])
        self.assertTrue(any(v["check"] == "row_field_count" for v in found))

    def test_nonzero_control_rate(self):
        found = check_rows(HEADER, [row(**{"mask/active_rate": "0.12"})])
        self.assertTrue(any(v["check"] == "control_rate_nonzero" for v in found))

    def test_all_five_control_rates_are_enforced(self):
        for key in METRIC_KEYS[:5]:
            found = check_rows(HEADER, [row(**{key: "0.5"})])
            self.assertTrue(
                any(v["check"] == "control_rate_nonzero" for v in found), key
            )

    def test_non_finite_metric(self):
        for bad in ("nan", "inf", "", "abc"):
            found = check_rows(HEADER, [row(**{"policy/entropy_valid": bad})])
            self.assertTrue(
                any(v["check"] == "metric_not_finite" for v in found), bad
            )

    def test_value_limit(self):
        found = check_rows(HEADER, [row(**{"critic/value_abs_max": "1000"})])
        self.assertTrue(any(v["check"] == "value_abs_max_too_large" for v in found))
        found = check_rows(HEADER, [row(**{"critic/value_abs_max": "999"})])
        self.assertEqual(found, [])

    def test_duplicate_and_empty_keys(self):
        self.assertTrue(
            any(v["check"] == "csv_header_duplicate_keys"
                for v in check_rows(["a", "a"] + HEADER, []))
        )
        self.assertTrue(
            any(v["check"] == "csv_header_empty_key"
                for v in check_rows(["a", ""] + HEADER, []))
        )

    def test_empty_snapshot_is_not_a_violation_by_itself(self):
        # startup produces no rows for ~25 minutes; only a stall rule may fire
        self.assertEqual(check_rows(HEADER, []), [])


class TestDisappearanceRule(unittest.TestCase):
    NOW = 1_000_000.0

    def test_running_container_is_fine(self):
        self.assertIsNone(
            disappeared_violation(0, ["abc"], False, self.NOW - 3600, self.NOW)
        )

    def test_finished_run_is_fine(self):
        self.assertIsNone(disappeared_violation(3, [], True, self.NOW - 3600, self.NOW))

    def test_startup_grace(self):
        # no rows, no container, started 2 minutes ago -> still starting
        self.assertIsNone(
            disappeared_violation(0, [], False, self.NOW - 120, self.NOW)
        )

    def test_gone_after_grace(self):
        found = disappeared_violation(0, [], False, self.NOW - 20 * 60, self.NOW)
        self.assertIsNotNone(found)
        self.assertEqual(found["check"], "run_disappeared")

    def test_rows_then_gone_is_immediate(self):
        found = disappeared_violation(7, [], False, self.NOW - 60, self.NOW)
        self.assertIsNotNone(found)

    def test_unknown_start_time_is_not_flagged(self):
        self.assertIsNone(disappeared_violation(0, [], False, None, self.NOW))


class TestStallRule(unittest.TestCase):
    def test_startup_is_exempt(self):
        self.assertIsNone(stall_violation(0, 99999.0, 30.0))

    def test_quiet_run_after_rows_is_flagged(self):
        found = stall_violation(5, 31 * 60.0, 30.0)
        self.assertIsNotNone(found)
        self.assertEqual(found["check"], "stalled")

    def test_recent_row_is_fine(self):
        self.assertIsNone(stall_violation(5, 60.0, 30.0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
