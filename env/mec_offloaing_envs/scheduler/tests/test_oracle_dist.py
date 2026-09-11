#!/usr/bin/env python3
"""Oracle dist_id diagnostic tests. Numpy/stdlib only. No GPU. No TensorFlow."""

from __future__ import annotations

import unittest

from spec.oracle_dist import (
    ORACLE_A_TEST_T,
    ORACLE_A_VAL_T,
    ORACLE_EPOCHS,
    ORACLE_IDENTITY_T_VAL_HI,
    ORACLE_IDENTITY_T_VAL_LO,
    ORACLE_N_DIST,
    ORACLE_TRAIN_REF_T,
    ORACLE_Z_DIM,
    classify_oracle_verdict,
    dist_ids_for_env,
    is_oracle_dist_var_name,
)


class _FakeEnv(object):
    def __init__(self):
        self.distribution_ids = [1, 3]
        self.task_graphs_batchs = [[object(), object()], [object()]]


class TestOracleDist(unittest.TestCase):
    def test_constants_and_verdict(self):
        self.assertEqual(ORACLE_N_DIST, 26)
        self.assertEqual(ORACLE_Z_DIM, 32)
        self.assertEqual(ORACLE_EPOCHS, 40)
        self.assertAlmostEqual(ORACLE_A_VAL_T, 575.0)
        self.assertAlmostEqual(ORACLE_A_TEST_T, 557.0)
        self.assertAlmostEqual(ORACLE_TRAIN_REF_T, 464.0)
        self.assertAlmostEqual(ORACLE_IDENTITY_T_VAL_LO, 568.0)
        self.assertAlmostEqual(ORACLE_IDENTITY_T_VAL_HI, 582.0)
        self.assertEqual(classify_oracle_verdict(575.0, 0.20), "oracle_no_gain")
        self.assertEqual(classify_oracle_verdict(560.0, 0.20), "oracle_helps")
        self.assertEqual(classify_oracle_verdict(568.0, 0.20), "oracle_weak")
        self.assertEqual(classify_oracle_verdict(560.0, 0.36), "oracle_hurts")
        self.assertEqual(classify_oracle_verdict(600.0, 0.20), "oracle_hurts")
        self.assertEqual(
            classify_oracle_verdict(575.0, 0.20, train_t=440.0),
            "oracle_in_dist_only",
        )

    def test_var_name_and_env_ids(self):
        self.assertTrue(is_oracle_dist_var_name("oracle_policy/oracle_dist/table:0"))
        self.assertTrue(is_oracle_dist_var_name("oracle_policy/oracle_dist/delta/kernel:0"))
        self.assertFalse(is_oracle_dist_var_name("oracle_policy/decoder/lstm_cell/kernel:0"))
        did = dist_ids_for_env(_FakeEnv())
        self.assertEqual(did.tolist(), [1, 1, 3])


if __name__ == "__main__":
    unittest.main()
