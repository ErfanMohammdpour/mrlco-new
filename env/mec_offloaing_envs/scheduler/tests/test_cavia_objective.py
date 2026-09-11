#!/usr/bin/env python3
"""CAVIA objective tests. Stdlib/numpy. No GPU. No TensorFlow."""

from __future__ import annotations

import unittest

from env.mec_offloaing_envs.scheduler.energy_api import ReferenceRanges, j_report
from spec.cavia_objective import (
    CAVIA_Z_DIM,
    CaviaObjective,
    centered_advantages,
    classify_cavia_verdict,
    is_cavia_var_name,
)


def _refs():
    return ReferenceRanges(
        L_ue=10.0,
        L_mec=4.0,
        L_helper=8.0,
        E_ue=1.0,
        E_mec=5.0,
        E_helper=3.0,
    )


class TestCaviaObjective(unittest.TestCase):
    def test_energy_off_cost_is_makespan(self):
        obj = CaviaObjective(use_energy=False)
        cost, t, e = obj.cost(12.5, 99.0, refs=_refs())
        self.assertEqual(cost, 12.5)
        self.assertEqual(t, 12.5)
        self.assertEqual(e, 99.0)
        self.assertEqual(obj.method_suffix, "frozen")

    def test_energy_on_uses_j_report(self):
        obj = CaviaObjective(use_energy=True)
        refs = _refs()
        cost, t, e = obj.cost(7.0, 3.0, refs=refs)
        self.assertAlmostEqual(cost, j_report(7.0, 3.0, refs))
        self.assertEqual(t, 7.0)
        self.assertEqual(e, 3.0)
        self.assertEqual(obj.method_suffix, "energy")

    def test_energy_on_requires_refs(self):
        obj = CaviaObjective(use_energy=True)
        with self.assertRaises(ValueError):
            obj.cost(1.0, 1.0, refs=None)

    def test_energy_on_weights_must_sum_one(self):
        with self.assertRaises(ValueError):
            CaviaObjective(use_energy=True, latency_weight=0.7, energy_weight=0.7)

    def test_energy_off_allows_unused_weights(self):
        obj = CaviaObjective(use_energy=False, latency_weight=1.0, energy_weight=0.0)
        cost, _, _ = obj.cost(3.0, 8.0)
        self.assertEqual(cost, 3.0)

    def test_centered_advantages(self):
        a = centered_advantages([10.0, 20.0, 30.0])
        self.assertAlmostEqual(float(a.sum()), 0.0, places=5)
        self.assertAlmostEqual(float(a[1]), 0.0, places=5)

    def test_verdict_identity_and_help(self):
        self.assertEqual(classify_cavia_verdict(581.8, 581.8, 0.20, identity_t=500.0), "identity_fail")
        self.assertEqual(classify_cavia_verdict(581.8, 500.0, 0.20, identity_t=581.8), "cavia_helps")
        self.assertEqual(classify_cavia_verdict(580.0, 657.0, 0.30, identity_t=580.0), "cavia_hurts")
        self.assertEqual(classify_cavia_verdict(580.0, 574.0, 0.20, identity_t=580.0), "cavia_weak")
        self.assertEqual(classify_cavia_verdict(580.0, 578.0, 0.20, identity_t=580.0), "cavia_no_gain")
        self.assertEqual(classify_cavia_verdict(575.0, 560.0, 0.20, identity_t=575.0, identity_lo=568.0, identity_hi=582.0), "cavia_helps")
        self.assertEqual(classify_cavia_verdict(575.0, 575.0, 0.20, identity_t=560.0, identity_lo=568.0, identity_hi=582.0), "identity_fail")

    def test_cavia_var_names(self):
        self.assertTrue(is_cavia_var_name("core_policy/cavia_z:0"))
        self.assertTrue(is_cavia_var_name("core_policy/decoder/cavia_film/delta/kernel:0"))
        self.assertFalse(is_cavia_var_name("core_policy/encoder/node_feature_embed/kernel:0"))
        self.assertEqual(CAVIA_Z_DIM, 32)

    def test_mix_helper(self):
        import numpy as np
        from spec.cavia_loop import _mix_and_nnon

        mix = _mix_and_nnon(np.asarray([[0, 1, 1, 2], [1, 1, 1, 1]], dtype=np.int32))
        self.assertAlmostEqual(mix["local_frac"], 1.0 / 8.0)
        self.assertAlmostEqual(mix["mec_frac"], 6.0 / 8.0)
        self.assertAlmostEqual(mix["v2v_frac"], 1.0 / 8.0)


if __name__ == "__main__":
    unittest.main()
