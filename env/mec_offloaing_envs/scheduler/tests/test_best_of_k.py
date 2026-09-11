#!/usr/bin/env python3
"""Phase 2 best-of-k tests. Numpy always. TF skipped if missing."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env.mec_offloaing_envs.scheduler.resources import ResourceConfig  # noqa: E402
from spec.best_of_k import (  # noqa: E402
    K_SWEEP,
    TOY_ORACLE_FILES,
    assert_monotone_k,
    duplicates_ratio,
    load_toy_oracle,
    sample_plans,
    score_plans,
    summarize_k,
)


def _tf_or_skip():
    try:
        import tensorflow as tf
    except ImportError:
        raise unittest.SkipTest("TensorFlow not installed")
    if not hasattr(tf, "contrib"):
        raise unittest.SkipTest("need tf.contrib (TensorFlow 1.15)")
    return tf


class TestBestOfKNumpy(unittest.TestCase):
    def test_prefix_best_monotone_in_k(self):
        scores = np.array(
            [
                [5.0, 4.0, 6.0, 3.5, 3.5, 2.0],
                [9.0, 8.5, 8.5, 8.0, 7.0, 7.0],
            ],
            dtype=np.float64,
        )
        means = assert_monotone_k(scores, ks=(1, 2, 3, 4, 5, 6))
        vals = [m for _, m in means]
        for a, b in zip(vals, vals[1:]):
            self.assertLessEqual(b, a + 1e-12)

    def test_k_sweep_tuple(self):
        self.assertEqual(K_SWEEP, (1, 4, 8, 16, 32, 64))

    def test_duplicates_ratio(self):
        plans = np.array(
            [
                [[0, 1], [0, 1], [1, 1], [1, 1]],
                [[0, 0], [0, 0], [0, 0], [0, 0]],
            ],
            dtype=np.int32,
        )
        self.assertAlmostEqual(duplicates_ratio(plans), 0.625)

    def test_toy_oracle_makespans(self):
        cfg = ResourceConfig.from_frozen_yaml()
        for name in TOY_ORACLE_FILES:
            tg, acts, want, _doc = load_toy_oracle(name)
            got = score_plans([tg], acts[None, :], cfg)
            self.assertEqual(got.shape, (1, 1), name)
            self.assertAlmostEqual(float(got[0, 0]), want, places=9, msg=name)

    def test_summarize_sample0_is_greedy(self):
        plans = np.zeros((3, 4, 20), dtype=np.int32)
        scores = np.array(
            [[4.0, 5.0, 3.0, 6.0], [8.0, 7.0, 9.0, 7.5], [2.0, 2.0, 2.0, 1.5]],
            dtype=np.float64,
        )
        row = summarize_k(plans, scores, k=1)
        self.assertAlmostEqual(row["T_best_k"], row["T_greedy"])
        self.assertAlmostEqual(row["frac_best_equals_greedy"], 1.0)
        row4 = summarize_k(plans, scores, k=4)
        self.assertLessEqual(row4["T_best_k"], row["T_best_k"] + 1e-12)


class TestBestOfKTF(unittest.TestCase):
    def test_sample0_equals_greedy(self):
        tf = _tf_or_skip()
        from policies.meta_seq2seq_policy import Seq2SeqPolicy

        tf.compat.v1.reset_default_graph()
        tf.compat.v1.set_random_seed(0)
        obs = np.zeros((2, 20, 50), dtype=np.float32)
        obs[:, :, 0] = 1.0
        policy = Seq2SeqPolicy(
            obs_dim=50,
            encoder_units=128,
            decoder_units=128,
            vocab_size=3,
            name="pi",
            encoder_type="meanagg",
            readout_type="triple",
        )
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            from spec.best_of_k import greedy_plan

            greedy = greedy_plan(sess, policy, obs)
            plans = sample_plans(sess, policy, obs, k=4, temperature=1.0)
        np.testing.assert_array_equal(plans[:, 0], greedy)
        self.assertEqual(plans.shape, (2, 4, 20))
        self.assertLessEqual(int(plans.max()), 2)


if __name__ == "__main__":
    unittest.main()
