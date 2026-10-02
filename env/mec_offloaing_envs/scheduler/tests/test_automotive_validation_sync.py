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
    """One frozen stack per process: the TF graph cannot be built twice."""

    @classmethod
    def setUpClass(cls):
        import shutil

        from spec.automotive_training.automotive_primary import (
            build_automotive_primary_stack,
        )

        cls.tmp = tempfile.mkdtemp(prefix="automotive_sync_test_")
        cls.trainer, cls.algo = build_automotive_primary_stack(
            seed=0, n_itr=1, ckpt_dir=cls.tmp)
        cls.evaluator = cls.trainer.held_out_evaluator
        cls.sess = tf.compat.v1.Session()
        cls.sess.__enter__()
        cls.sess.run(tf.compat.v1.global_variables_initializer())
        cls._shutil = shutil

    @classmethod
    def tearDownClass(cls):
        cls.sess.__exit__(None, None, None)
        cls._shutil.rmtree(cls.tmp, ignore_errors=True)

    def setUp(self):
        self.sess.run(tf.compat.v1.global_variables_initializer())
        self.algo.sync_task_policies_from_core()
        self.evaluator.sync_count = 0

    def test_sync_from_core_and_core_preservation(self):
        from meta_algos.variable_io import snapshot_trainable

        ev = self.evaluator
        core_before = snapshot_trainable(ev.source_policy, sess=self.sess)

        ev.evaluate_all(k_steps=0, sess=self.sess)
        self.assertEqual(ev.sync_count, 1, "k=0 must sync from the core")
        self.assertLessEqual(ev.last_sync_max_abs_diff, 1e-6)

        ev.evaluate_all(k_steps=3, sess=self.sess)
        self.assertEqual(ev.sync_count, 2, "every evaluation must start from the core")
        self.assertTrue(ev.core_unchanged_after_adaptation,
                        "k-step adaptation must not mutate the core")
        core_after = snapshot_trainable(ev.source_policy, sess=self.sess)
        for before, after in zip(core_before, core_after):
            self.assertTrue((before == after).all(),
                            "core weights changed during validation")

        # a second k=3 must not accumulate on the previous adaptation: the sync
        # resets the scratch to the core every time, and the PPO graph is reused
        ppo_before = id(ev._adaptation_ppo)
        ev.evaluate_all(k_steps=3, sess=self.sess)
        self.assertEqual(ev.sync_count, 3)
        self.assertEqual(id(ev._adaptation_ppo), ppo_before,
                         "the adaptation PPO must be constructed once")

    def test_scratch_equals_core_immediately_after_sync(self):
        from meta_algos.variable_io import snapshot_trainable

        ev = self.evaluator
        ev._sync_from_core(self.sess)
        core = snapshot_trainable(ev.source_policy, sess=self.sess)
        scratch = snapshot_trainable(ev.policy, sess=self.sess)
        self.assertEqual(len(core), len(scratch))
        for a, b in zip(core, scratch):
            self.assertTrue((a == b).all(), "scratch must equal the core")

    def test_report_contract_and_logged_columns(self):
        from utils import logger

        trainer = self.trainer
        log_dir = Path(trainer.auto_run_dir) / "logs_gatef"
        logger.configure(dir=str(log_dir), format_strs=["csv"])
        report = trainer.train()

        rows = list(csv.DictReader(open(log_dir / "progress.csv")))
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
            "constraint/batch_size_after_reset_C_GRAPH_HARD_DEADLINE",
            "constraint/batch_size_after_reset_C_HI_TASK_TARDINESS",
        ]
        for key in wanted:
            self.assertIn(key, row, "missing logged column %s" % key)
        self.assertEqual(float(row["constraint/batch_size_after_reset_C_HI_TASK_TARDINESS"]), 0.0,
                         "the dual batch must be empty after each iteration")
        self.assertGreater(float(row["correctness/lambda_broadcast_targets"]), 0.0)
        self.assertEqual(float(row["correctness/core_unchanged_after_adaptation"]), 1.0)

        counters = report["sampler_counters"]
        self.assertEqual(counters["support_calls"], 1)
        self.assertEqual(counters["query_calls"], 1)
        self.assertTrue(counters["counter_role_contract_ok"])
        self.assertTrue(report["sampler_counter_contract"]["balanced"])
        self.assertGreater(report["lambda_broadcast_targets"], 0)
        self.assertEqual(report["penalty_feedback"]["episodes_with_nonzero_penalty"],
                         report["penalty_feedback"]["episodes"])
        self.assertGreater(report["penalty_feedback"]["penalty_sum"], 0.0,
                           "the Lagrangian penalty must reach the rollout rewards")
        self.assertEqual(report["query_graph_count"], 40)
        self.assertEqual(report["method_id"], "margo_automotive_mc_v1_primary")
        self.assertNotEqual(report["training_fingerprint_parts"]["git_sha"], "unknown")
        self.assertIn("code_dirty", report["training_fingerprint_parts"])
        for name, size in report["dual_batch_size_at_end"].items():
            self.assertEqual(size, 0, "dual batch not reset: %s" % name)
        self.assertEqual(report["meta_test_access_count"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
