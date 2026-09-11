#!/usr/bin/env python3
"""All-MEC rewrite diagnostic tests. Numpy/stdlib only. No GPU. No TensorFlow."""

from __future__ import annotations

import unittest

import numpy as np

from spec.pair_head import joint_id
from spec.rewrite_mec import (
    REWRITE_A_TEST_T,
    REWRITE_A_VAL_T,
    REWRITE_EPOCHS,
    REWRITE_K,
    REWRITE_LAMBDA,
    apply_k_neural,
    classify_rewrite_verdict,
    log_softmax,
    pick_neural_joint,
)


class TestRewriteMec(unittest.TestCase):
    def test_verdict_and_k(self):
        self.assertEqual(REWRITE_K, (1, 2, 3))
        self.assertEqual(REWRITE_EPOCHS, 40)
        self.assertAlmostEqual(REWRITE_LAMBDA, 0.5)
        self.assertAlmostEqual(REWRITE_A_VAL_T, 575.0)
        self.assertAlmostEqual(REWRITE_A_TEST_T, 557.0)
        self.assertEqual(classify_rewrite_verdict(575.0, 0.20), "rewrite_no_gain")
        self.assertEqual(classify_rewrite_verdict(560.0, 0.20), "rewrite_helps")
        self.assertEqual(classify_rewrite_verdict(568.0, 0.20), "rewrite_weak")
        self.assertEqual(classify_rewrite_verdict(560.0, 0.36), "rewrite_hurts")
        self.assertEqual(classify_rewrite_verdict(600.0, 0.20), "rewrite_hurts")

    def test_pick_neural_joint_no_schedule(self):
        plan = [1, 1, 1]
        pairs = [{"i": 0, "j": 1, "direct": 1, "sibling": 0, "join": 0}]
        logp = np.zeros((3, 3), dtype=np.float64)
        logp[0, 0] = 4.0
        logp[1, 2] = 4.0
        logp[0, 1] = 0.0
        logp[1, 1] = 0.0
        picked = pick_neural_joint(plan, pairs, logp)
        self.assertIsNotNone(picked)
        self.assertEqual(picked["ai"], 0)
        self.assertEqual(picked["aj"], 2)
        self.assertEqual(picked["jid"], joint_id(0, 2))
        self.assertGreater(picked["gain"], 0.0)

    def test_no_gain_stays(self):
        plan = [1, 1]
        pairs = [{"i": 0, "j": 1}]
        logp = np.zeros((2, 3), dtype=np.float64)
        logp[0, 1] = 5.0
        logp[1, 1] = 5.0
        self.assertIsNone(pick_neural_joint(plan, pairs, logp))

    def test_apply_k_sequential(self):
        plan = [1, 1, 1]
        pairs = [{"i": 0, "j": 1}, {"i": 1, "j": 2}]

        def logp_fn(cur):
            lp = np.zeros((3, 3), dtype=np.float64)
            if int(cur[0]) == 1 and int(cur[1]) == 1:
                lp[0, 0] = 3.0
                lp[1, 0] = 3.0
            else:
                lp[1, 2] = 3.0
                lp[2, 2] = 3.0
            return lp

        out, applied = apply_k_neural(plan, pairs, logp_fn, 2)
        self.assertEqual(len(applied), 2)
        self.assertEqual(out[0], 0)
        self.assertEqual(out[1], 2)
        self.assertEqual(out[2], 2)

    def test_log_softmax_row(self):
        p = np.exp(log_softmax([0.0, 0.0, 0.0]))
        self.assertTrue(np.allclose(p, 1.0 / 3.0))


if __name__ == "__main__":
    unittest.main()
