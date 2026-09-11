#!/usr/bin/env python3
"""All-MEC pair-frac diagnostic tests. Numpy/stdlib only. No GPU. No TensorFlow."""

from __future__ import annotations

import unittest

from spec.pair_frac import (
    PAIRFRAC_CLONE_FRAC,
    PAIRFRAC_HOLD,
    PAIRFRAC_N_GRAPHS,
    PAIRFRAC_OK,
    classify_pairfrac,
    proceed_rewrite,
)


class TestPairFrac(unittest.TestCase):
    def test_verdict_and_gate(self):
        self.assertEqual(classify_pairfrac(0.07), "pairfrac_mec_clone_trap")
        self.assertEqual(classify_pairfrac(PAIRFRAC_CLONE_FRAC), "pairfrac_mec_clone_trap")
        self.assertEqual(classify_pairfrac(0.15), "pairfrac_mec_weak")
        self.assertEqual(classify_pairfrac(0.25), "pairfrac_mec_ok")
        self.assertEqual(classify_pairfrac(0.50), "pairfrac_mec_holdout_like")
        self.assertFalse(proceed_rewrite(0.07))
        self.assertFalse(proceed_rewrite(0.19))
        self.assertTrue(proceed_rewrite(PAIRFRAC_OK))
        self.assertTrue(proceed_rewrite(PAIRFRAC_HOLD))

    def test_n_graphs_matches_h2(self):
        from spec.hamming2_probe import H2_N_GRAPHS

        self.assertEqual(PAIRFRAC_N_GRAPHS, 200)
        self.assertEqual(PAIRFRAC_N_GRAPHS, H2_N_GRAPHS)


if __name__ == "__main__":
    unittest.main()
