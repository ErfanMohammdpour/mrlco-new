#!/usr/bin/env python3
"""TF-gated blocking tests: the validation evaluator must measure the TRAINED core.

Gate B of the correctness repair. Before the fix, `AutomotiveHeldOutEvaluator`
guarded the weight copy with `getattr(self.policy, "assign_trainable", None)`, but
`Seq2SeqPolicy` has no such method, so k=0 evaluated a stale scratch policy and the
k=3 adaptation accumulated across validations. The whole 5x500 pilot table was
therefore not a measurement of the trained model.
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

try:
    import tensorflow as tf

    HAS_TF = True
except Exception:  # pragma: no cover
    HAS_TF = False


@unittest.skipUnless(HAS_TF, "requires TensorFlow (run in the TF1.15 image)")
class TestValidationMeasuresTheTrainedCore(unittest.TestCase):
    def _stack(self, tmp):
        from spec.automotive_training.automotive_primary import (
            build_automotive_primary_stack,
        )

        return build_automotive_primary_stack(seed=0, n_itr=1, ckpt_dir=tmp)

    def test_sync_from_core_and_core_preservation(self):
        from meta_algos.variable_io import snapshot_trainable

        with tempfile.TemporaryDirectory() as tmp:
            trainer, algo = self._stack(tmp)
            evaluator = trainer.held_out_evaluator
            with tf.compat.v1.Session() as sess:
                sess.run(tf.compat.v1.global_variables_initializer())
                algo.sync_task_policies_from_core()
                core_before = snapshot_trainable(evaluator.source_policy, sess=sess)

                evaluator.evaluate_all(k_steps=0, sess=sess)
                self.assertEqual(evaluator.sync_count, 1, "k=0 must sync from the core")
                self.assertLessEqual(evaluator.last_sync_max_abs_diff, 1e-6)

                evaluator.evaluate_all(k_steps=3, sess=sess)
                self.assertEqual(evaluator.sync_count, 2,
                                 "every evaluation must start from the core")
                self.assertTrue(evaluator.core_unchanged_after_adaptation,
                                "k-step adaptation must not mutate the core")
                core_after = snapshot_trainable(evaluator.source_policy, sess=sess)
                for before, after in zip(core_before, core_after):
                    self.assertTrue((before == after).all(),
                                    "core weights changed during validation")

                # a second k=3 must not accumulate on top of the previous adaptation:
                # the sync resets the scratch to the core every time
                evaluator.evaluate_all(k_steps=3, sess=sess)
                self.assertEqual(evaluator.sync_count, 3)
                self.assertEqual(evaluator.adaptation_ppo_constructions, 1,
                                 "the adaptation PPO must be built exactly once")

    def test_scratch_equals_core_immediately_after_sync(self):
        from meta_algos.variable_io import snapshot_trainable

        with tempfile.TemporaryDirectory() as tmp:
            trainer, algo = self._stack(tmp)
            evaluator = trainer.held_out_evaluator
            with tf.compat.v1.Session() as sess:
                sess.run(tf.compat.v1.global_variables_initializer())
                algo.sync_task_policies_from_core()
                evaluator._sync_from_core(sess)
                core = snapshot_trainable(evaluator.source_policy, sess=sess)
                scratch = snapshot_trainable(evaluator.policy, sess=sess)
                self.assertEqual(len(core), len(scratch))
                for a, b in zip(core, scratch):
                    self.assertTrue((a == b).all(), "scratch must equal the core")

    def test_training_iteration_logs_correctness_columns(self):
        from utils import logger

        with tempfile.TemporaryDirectory() as tmp:
            trainer, algo = self._stack(tmp)
            log_dir = Path(tmp) / "logs"
            logger.configure(dir=str(log_dir), format_strs=["csv"])
            trainer.train()
            csv_path = log_dir / "progress.csv"
            self.assertTrue(csv_path.exists(), "progress.csv must be written")
            rows = list(csv.DictReader(open(csv_path)))
            self.assertTrue(rows)
            row = rows[0]
            wanted = [
                "validation/query_mean_latency_seconds_k0",
                "validation/query_mean_latency_seconds_k3",
                "validation/high_task_tardiness_task_rate_k3",
                "validation/graph_high_tardiness_incidence_rate_k3",
                "validation/firm_task_miss_rate_k3",
                "correctness/lambda_broadcast_targets",
                "correctness/core_unchanged_after_adaptation",
            ]
            for key in wanted:
                self.assertIn(key, row, "missing logged column %s" % key)
            for name in ("C_GRAPH_HARD_DEADLINE", "C_HI_TASK_TARDINESS"):
                key = "constraint/batch_size_after_reset_%s" % name
                self.assertIn(key, row, "missing dual-reset column %s" % key)
                self.assertEqual(float(row[key]), 0.0,
                                 "the dual batch must be empty after each iteration")

    def test_report_contract(self):
        from utils import logger

        with tempfile.TemporaryDirectory() as tmp:
            trainer, algo = self._stack(tmp)
            logger.configure(dir=str(Path(tmp) / "logs"), format_strs=["csv"])
            report = trainer.train()
            n_itr = 1
            counters = report["sampler_counters"]
            self.assertEqual(counters["support_calls"], n_itr)
            self.assertEqual(counters["query_calls"], n_itr)
            self.assertTrue(counters["counter_role_contract_ok"])
            self.assertEqual(report["sampler_counter_contract"]["balanced"], True)
            self.assertGreater(report["lambda_broadcast_targets"], 0)
            self.assertEqual(report["penalty_feedback"]["episodes_with_nonzero_penalty"],
                             report["penalty_feedback"]["episodes"])
            self.assertEqual(report["query_graph_count"], 40)
            self.assertEqual(report["method_id"], "margo_automotive_mc_v1_primary")
            self.assertNotEqual(report["training_fingerprint_parts"]["git_sha"], "unknown")
            self.assertIn("code_dirty", report["training_fingerprint_parts"])
            for name, size in report["dual_batch_size_at_end"].items():
                self.assertEqual(size, 0, "dual batch not reset: %s" % name)


if __name__ == "__main__":
    unittest.main(verbosity=2)
