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

import importlib.util  # noqa: E402
import types  # noqa: E402

# utils.logger imports tensorflow at module level for its TensorBoard writer;
# the CSV path needs none of it. Stub it ONLY when TF is genuinely absent: under
# a single-process unittest run a stub here would shadow the real TF that the
# Phase 2/3 smoke tests import later.
try:
    _TF_INSTALLED = importlib.util.find_spec("tensorflow") is not None
except (ImportError, ValueError):
    _TF_INSTALLED = False
if not _TF_INSTALLED:
    sys.modules.setdefault("tensorflow", types.ModuleType("tensorflow"))
if "joblib" not in sys.modules:
    # only used by the snapshot helpers, not by the CSV writer
    sys.modules["joblib"] = types.ModuleType("joblib")

from utils.logger import CSVOutputFormat  # noqa: E402


# The real trainer key set, taken from the failing off-mode run: one key ends
# with a comma and another is a whitespace spacer. Before the fix the header had
# 47 fields against 43 in the row.
REAL_KEYS = (
    "Average greedy latency,", "mask/active_rate", "mask/forced_rate",
    "mask/all_invalid_rate", "policy/invalid_action_rate",
    "policy/argmax_masked_rate", "policy/entropy_valid", "critic/value_abs_max",
    "inner_update_mode", "PolicyExecTime", "EnvExecTime", "AverageDiscountedReturn",
    "AverageReturn", "NumTrajs", "StdReturn", "MaxReturn", "MinReturn",
    "Average energy", " ", "Itr", "Average reward", "Average latency", "split_role",
    "seed", "split_version", "support_graphs_per_meta_task",
    "support_trajectories_per_meta_task", "query_graph_count", "k_steps",
    "outer_update_count", "entropy_coefficient", "value_clip_epsilon",
    "ppo_batch_size_trajectories", "meta_batch_size_distributions",
    "inner_learning_rate", "outer_learning_rate", "outer_update_method",
    "hyperparameter_provenance.policy", "validation_query_composite_objective_k0",
    "validation_query_composite_objective", "validation_query_mean_latency_k0",
    "validation_query_mean_latency_k3", "checkpoint_selection_metric",
    "checkpoint_is_best_val", "mask/late_arrival",
)


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

    def test_trainer_shaped_keys_stay_aligned(self):
        """Replay of the real key set, with a late-arriving key added mid-run."""
        writer = CSVOutputFormat(str(self.path))
        first_row = {k: ("publication" if k == "inner_update_mode" else 1.0) for k in REAL_KEYS}
        writer.writekvs({k: v for k, v in first_row.items() if k != "mask/late_arrival"})
        second_row = dict(first_row)
        second_row["mask/active_rate"] = 0.0
        second_row["policy/entropy_valid"] = 0.31908559799194336
        second_row["critic/value_abs_max"] = 80.0
        writer.writekvs(second_row)
        writer.close()

        with self.path.open() as handle:
            raw = list(csv.reader(handle))
        header, rows = raw[0], raw[1:]
        self.assertEqual(len(header), len(REAL_KEYS), header)
        self.assertIn("Average greedy latency,", header)
        self.assertNotIn("", header)
        for row in rows:
            self.assertEqual(len(row), len(header), row)
        by_row = [dict(zip(header, row)) for row in rows]
        self.assertEqual(by_row[0]["mask/active_rate"], "1.0")
        self.assertEqual(by_row[1]["mask/active_rate"], "0.0")
        self.assertEqual(by_row[1]["policy/entropy_valid"], "0.31908559799194336")
        self.assertEqual(by_row[1]["critic/value_abs_max"], "80.0")
        self.assertEqual(by_row[1]["inner_update_mode"], "publication")
        self.assertEqual(by_row[1]["mask/late_arrival"], "1.0")

    def test_commas_in_values_do_not_break_columns(self):
        writer = CSVOutputFormat(str(self.path))
        writer.writekvs({"note": "a,b", "x": 1})
        writer.close()
        header, rows = self._read()
        self.assertEqual(len(rows[0]), len(header))
        self.assertEqual(dict(zip(header, rows[0]))["note"], "a,b")

    def test_two_writers_sharing_one_path_never_desync(self):
        """The 2N-field signature seen on Kish: header N, first row 2N.

        Two CSV writers open the same path before the first dump. Both write the
        same keys; the second one padded the first writer's row while keeping the
        shorter header. With the buffered rewrite every write leaves a complete
        file, so no row can be longer than the header.
        """
        w1 = CSVOutputFormat(str(self.path))
        w2 = CSVOutputFormat(str(self.path))
        kvs = {"a": 1, "b": 2, "c": 3}
        w1.writekvs(kvs)
        w2.writekvs(kvs)
        w1.close()
        w2.close()
        header, rows = self._read()
        self.assertEqual(len(header), 3, header)
        for row in rows:
            self.assertEqual(len(row), len(header), (header, rows))

    def test_repeated_writes_keep_every_row_aligned(self):
        writer = CSVOutputFormat(str(self.path))
        writer.writekvs({"a": 1})
        writer.writekvs({"a": 2, "b": 3})
        writer.writekvs({"a": 4, "b": 5, "c": 6})
        writer.close()
        header, rows = self._read()
        self.assertEqual(header, ["a", "b", "c"])
        self.assertEqual(len(rows), 3)
        for row in rows:
            self.assertEqual(len(row), len(header), row)
        by_row = [dict(zip(header, row)) for row in rows]
        self.assertEqual(by_row[0]["b"], "")
        self.assertEqual(by_row[2]["c"], "6")


if __name__ == "__main__":
    unittest.main(verbosity=2)
