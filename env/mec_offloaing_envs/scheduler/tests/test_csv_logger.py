#!/usr/bin/env python3
"""Regression tests for the CSV logger used by every training run.

The bug these guard against: keys are added incrementally (stages log new
metrics as they appear), and the previous implementation extended the header
from a SET difference, then rewrote the file without truncating it. Column order
became hash-dependent and stale bytes shifted every later column, so a metric
could read a neighbouring column's value -- or a string.

The trainer-level smoke surfaced it as
`policy/invalid_action_rate = 1.0961478823802395` (a rate above 1) and
`policy/argmax_masked_rate = validation_query_composite_objective`.
"""

from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import types  # noqa: E402

# utils.logger imports tensorflow at module level for its TensorBoard writer;
# the CSV path needs none of it, so stub it the way the other tests stub gym.
if "tensorflow" not in sys.modules:
    sys.modules["tensorflow"] = types.ModuleType("tensorflow")
if "joblib" not in sys.modules:
    # only used by the snapshot helpers, not by the CSV writer
    sys.modules["joblib"] = types.ModuleType("joblib")

from utils.logger import CSVOutputFormat  # noqa: E402


class TestCsvAlignment(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="csv_logger_"))
        self.path = self.tmp / "progress.csv"

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmp, ignore_errors=True)

    def _read(self):
        with self.path.open() as handle:
            rows = list(csv.reader(handle))
        return rows[0], rows[1:]

    def test_incremental_keys_keep_header_and_values_aligned(self):
        writer = CSVOutputFormat(str(self.path))
        writer.writekvs({"loss": 1.0})
        writer.writekvs({"loss": 0.9, "mask/active_rate": 0.25})
        writer.writekvs({"loss": 0.8, "mask/active_rate": 0.0, "critic/value_abs_max": 80.0})
        writer.close()

        header, rows = self._read()
        self.assertEqual(len(set(header)), len(header), header)
        self.assertNotIn("", header)
        self.assertEqual(len(rows), 3)
        for row in rows:
            self.assertEqual(len(row), len(header), row)
        by_row = [dict(zip(header, row)) for row in rows]
        self.assertEqual(by_row[0]["mask/active_rate"], "")       # not logged yet
        self.assertEqual(by_row[1]["mask/active_rate"], "0.25")
        self.assertEqual(by_row[2]["mask/active_rate"], "0.0")
        self.assertEqual(by_row[2]["critic/value_abs_max"], "80.0")
        self.assertEqual(by_row[2]["loss"], "0.8")

    def test_late_key_does_not_shift_earlier_columns(self):
        writer = CSVOutputFormat(str(self.path))
        writer.writekvs({"a": 1, "b": 2})
        writer.writekvs({"a": 3, "b": 4, "a_very_long_new_metric_name_that_is_longer": 5})
        writer.close()

        header, rows = self._read()
        for row in rows:
            self.assertEqual(len(row), len(header))
        by_row = [dict(zip(header, row)) for row in rows]
        self.assertEqual((by_row[0]["a"], by_row[0]["b"]), ("1", "2"))
        self.assertEqual((by_row[1]["a"], by_row[1]["b"]), ("3", "4"))
        self.assertEqual(by_row[0]["a_very_long_new_metric_name_that_is_longer"], "")

    def test_column_order_is_deterministic(self):
        orders = []
        for _ in range(5):
            path = self.tmp / ("p%d.csv" % len(orders))
            writer = CSVOutputFormat(str(path))
            writer.writekvs({"z": 1})
            writer.writekvs({"z": 2, "a": 3, "m": 4, "b": 5})
            writer.close()
            with path.open() as handle:
                orders.append(handle.readline().strip())
        self.assertEqual(len(set(orders)), 1, orders)

    def test_comma_in_a_key_does_not_split_the_header(self):
        """The real trainer logs a key that ends with a comma."""
        writer = CSVOutputFormat(str(self.path))
        writer.writekvs({"Average greedy latency,": 1.0, "a": 2.0})
        writer.writekvs({"Average greedy latency,": 3.0, "a": 4.0, "mask/active_rate": 0.0})
        writer.close()

        with self.path.open() as handle:
            raw = list(csv.reader(handle))
        header, rows = raw[0], raw[1:]
        self.assertEqual(len(header), 3, header)
        self.assertIn("Average greedy latency,", header)
        for row in rows:
            self.assertEqual(len(row), len(header), row)
        by_row = [dict(zip(header, row)) for row in rows]
        self.assertEqual(by_row[0]["Average greedy latency,"], "1.0")
        self.assertEqual(by_row[1]["mask/active_rate"], "0.0")

    def test_commas_in_values_do_not_break_columns(self):
        writer = CSVOutputFormat(str(self.path))
        writer.writekvs({"note": "a,b", "x": 1})
        writer.close()
        header, rows = self._read()
        self.assertEqual(len(rows[0]), len(header))
        self.assertEqual(dict(zip(header, rows[0]))["note"], "a,b")


if __name__ == "__main__":
    unittest.main(verbosity=2)
