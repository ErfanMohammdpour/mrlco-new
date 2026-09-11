#!/usr/bin/env python3
"""Pair-sup diagnostic tests. Numpy/stdlib only. No GPU. No TensorFlow."""

from __future__ import annotations

import unittest

from spec.pair_head import joint_id
from spec.pair_sup import (
    PAIRSUP_A_TEST_T,
    PAIRSUP_A_VAL_T,
    PAIRSUP_EPOCHS,
    PAIRSUP_LAMBDA,
    _batch_pairs,
    classify_pairsup_verdict,
    pick_improving_joint,
)


class TestPairSup(unittest.TestCase):
    def test_pick_improving_joint(self):
        plan = [1, 1, 1]

        def score_fn(acts):
            if acts[0] == 0 and acts[1] == 2:
                return 10.0
            return 20.0

        picked, n_eval = pick_improving_joint(plan, 0, 1, 20.0, score_fn)
        self.assertEqual(n_eval, 8)
        self.assertIsNotNone(picked)
        self.assertEqual(picked["ai"], 0)
        self.assertEqual(picked["aj"], 2)
        self.assertEqual(picked["jid"], joint_id(0, 2))
        self.assertAlmostEqual(picked["delta"], 10.0)

    def test_no_gain_returns_none(self):
        plan = [1, 1]

        def score_fn(acts):
            return 20.0 + float(acts[0])

        picked, n_eval = pick_improving_joint(plan, 0, 1, 20.0, score_fn)
        self.assertIsNone(picked)
        self.assertEqual(n_eval, 8)

    def test_verdict(self):
        self.assertEqual(classify_pairsup_verdict(575.0, 0.20), "pairsup_no_gain")
        self.assertEqual(classify_pairsup_verdict(560.0, 0.20), "pairsup_helps")
        self.assertEqual(classify_pairsup_verdict(568.0, 0.20), "pairsup_weak")
        self.assertEqual(classify_pairsup_verdict(560.0, 0.36), "pairsup_hurts")
        self.assertEqual(classify_pairsup_verdict(600.0, 0.20), "pairsup_hurts")

    def test_a_baseline_is_bc_unseen_not_2opt(self):
        self.assertAlmostEqual(PAIRSUP_A_VAL_T, 575.0)
        self.assertAlmostEqual(PAIRSUP_A_TEST_T, 557.0)
        self.assertEqual(PAIRSUP_EPOCHS, 40)
        self.assertAlmostEqual(PAIRSUP_LAMBDA, 0.5)

    def test_batch_pairs_remap_and_empty(self):
        labels = {5: [{"i": 1, "j": 3, "ai": 0, "aj": 2}]}
        packed = _batch_pairs([5, 9], labels)
        self.assertEqual(int(packed["b"][0]), 0)
        self.assertEqual(int(packed["i"][0]), 1)
        self.assertEqual(float(packed["w"][0]), 1.0)
        empty = _batch_pairs([9], labels)
        self.assertEqual(empty["w"].shape[0], 1)
        self.assertEqual(float(empty["w"][0]), 0.0)


if __name__ == "__main__":
    unittest.main()
