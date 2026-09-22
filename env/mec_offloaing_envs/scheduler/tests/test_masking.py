#!/usr/bin/env python3
"""⑥b acceptance tests for the masked action distribution (pure numpy).

These are the semantics the TF graph must mirror. Test names follow the agreed
list; the PPO-side checks are expressed as log-probability identities so they can
be verified without TensorFlow.
"""

from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
for _name in ("gym", "gym.core"):
    if _name not in sys.modules:
        sys.modules[_name] = types.ModuleType(_name)
if not hasattr(sys.modules.get("gym.core", types.ModuleType("gym.core")), "Env"):
    sys.modules["gym.core"].Env = type("Env", (), {})

import numpy as np  # noqa: E402

from env.mec_offloaing_envs.scheduler import encoder_obs as eo  # noqa: E402
from env.mec_offloaing_envs.scheduler.masking import (  # noqa: E402
    NEG_LARGE,
    apply_mask,
    dead_end_guard,
    distribution,
    entropy_valid,
    likelihood_ratio,
    mask_from_observation,
    masked_log_softmax,
    masked_softmax,
    sample,
)

LOGITS = np.array([[2.1, 3.5, 1.2], [0.5, 0.4, 3.0]])
ALL_VALID = np.ones((2, 3), dtype=bool)


class TestTest1FixedMaskRatioIsOne(unittest.TestCase):
    """Test 1: with the SAME logits and the same mask, ratio == 1 exactly.

    PPO's first update must reproduce the rollout distribution; any deviation
    means the mask or the log-prob path is inconsistent between rollout and
    update.
    """

    def test_ratio_one_when_logits_unchanged(self):
        feasible = np.array([[True, False, True], [True, True, False]])
        logits = np.array([[2.1, 3.5, 1.2], [0.5, 0.4, 3.0]])
        for action in (0, 1, 2):
            act = np.array([action, action])
            ratio = likelihood_ratio(act, logits, logits, feasible)
            np.testing.assert_allclose(ratio, 1.0, rtol=0, atol=1e-12)

    def test_log_probs_identical_for_repeated_calls(self):
        feasible = np.array([[True, False, True]])
        logits = np.array([[1.0, 2.0, 3.0]])
        np.testing.assert_allclose(
            masked_log_softmax(logits, feasible), masked_log_softmax(logits, feasible),
            rtol=0, atol=1e-12,
        )

    def test_ratio_is_finite_for_different_logits(self):
        feasible = np.array([[True, False, True], [False, True, True]])
        old = np.array([[2.1, 3.5, 1.2], [0.5, 0.4, 3.0]])
        new = np.array([[-1.0, 9.9, 0.3], [7.7, -2.0, 1.0]])
        act = np.array([0, 1])
        ratio = likelihood_ratio(act, old, new, feasible)
        self.assertTrue(np.all(np.isfinite(ratio)))
        self.assertTrue(np.all(ratio > 0.0))

    def test_mask_must_not_change_between_rollout_and_update(self):
        """Documents the hazard: recomputing a DIFFERENT mask at update time
        collapses the importance ratio. This is why the runtime (prefix-dependent)
        mask must be STORED with the batch instead of recomputed, and why the
        static mask is read from the stored observation.
        """
        logits = np.array([[2.1, 3.5, 1.2]])
        act = np.array([1])
        m_keep = np.array([[True, True, True]])
        m_drop = np.array([[True, False, True]])
        # same logits, same mask -> exactly 1
        self.assertAlmostEqual(float(likelihood_ratio(act, logits, logits, m_keep)[0]), 1.0, places=12)
        # same logits, but the mask CHANGED between rollout and update under the
        # chosen action -> ratio collapses (the hazard this design avoids)
        self.assertLess(
            float(likelihood_ratio(act, logits, logits, m_keep, new_feasible=m_drop)[0]),
            1e-6,
        )

    def test_ratio_one_with_a_restrictive_mask(self):
        feasible = np.array([[True, False, False], [False, True, True]])
        logits = np.array([[2.1, 3.5, 1.2], [0.5, 0.4, 3.0]])
        act = np.array([0, 2])
        np.testing.assert_allclose(
            likelihood_ratio(act, logits, logits, feasible), 1.0, rtol=0, atol=1e-12
        )


class TestTest2MaskedActionHasZeroProbability(unittest.TestCase):
    def test_probability_exactly_zero(self):
        feasible = np.array([[True, False, True]])
        p = masked_softmax(LOGITS[:1], feasible)
        self.assertAlmostEqual(float(p[0, 1]), 0.0, places=15)
        self.assertAlmostEqual(float(p.sum()), 1.0, places=12)

    def test_log_probability_is_neg_large(self):
        feasible = np.array([[True, False, True]])
        logp = masked_log_softmax(LOGITS[:1], feasible)
        self.assertLess(float(logp[0, 1]), -1e8)

    def test_two_valid_actions_renormalise(self):
        feasible = np.array([[True, False, True]])
        p = masked_softmax(LOGITS[:1], feasible)
        expected = np.exp([2.1, 1.2]) / np.exp([2.1, 1.2]).sum()
        self.assertAlmostEqual(float(p[0, 0]), float(expected[0]), places=12)
        self.assertAlmostEqual(float(p[0, 2]), float(expected[1]), places=12)

    def test_entropy_uses_valid_actions_only(self):
        feasible = np.array([[True, False, True]])
        h_valid = float(entropy_valid(LOGITS[:1], feasible)[0])
        self.assertLessEqual(h_valid, np.log(2.0) + 1e-12)
        h_all = float(entropy_valid(LOGITS[:1], ALL_VALID[:1])[0])
        self.assertGreater(h_all, h_valid)


class TestTest3DeadEndGuard(unittest.TestCase):
    def test_guard_fires_and_unmasks(self):
        feasible = np.zeros((1, 3), dtype=bool)
        guarded, all_invalid = dead_end_guard(feasible)
        self.assertTrue(bool(all_invalid[0]))
        self.assertTrue(bool(guarded.all()))

    def test_distribution_is_finite(self):
        feasible = np.zeros((1, 3), dtype=bool)
        d = distribution(LOGITS[:1], feasible)
        self.assertTrue(np.all(np.isfinite(d.probabilities)))
        self.assertAlmostEqual(float(d.probabilities.sum()), 1.0, places=12)
        self.assertEqual(int(d.valid_count[0]), 3)     # guard restored validity
        self.assertTrue(d.any_all_invalid)

    def test_guard_does_not_touch_normal_rows(self):
        feasible = np.array([[True, False, True], [False, False, False]])
        guarded, all_invalid = dead_end_guard(feasible)
        np.testing.assert_array_equal(guarded[0], feasible[0])
        self.assertFalse(bool(all_invalid[0]))
        self.assertTrue(bool(all_invalid[1]))


class TestTest4SamplingNeverPicksInvalid(unittest.TestCase):
    def test_thousand_samples(self):
        rng = np.random.RandomState(0)
        # one mask row per logits row (row 0: only action 0; row 1: actions 1 and 2)
        feasible = np.array([[True, False, False], [False, True, True]])
        for _ in range(1000):
            a = sample(LOGITS, feasible, rng)
            for row, mask in zip(a, feasible):
                self.assertTrue(bool(mask[row]), "sampled infeasible action")

    def test_masked_action_probability_mass_never_leaks(self):
        rng = np.random.RandomState(1)
        feasible = np.array([[True, False, False]])
        picks = [int(sample(LOGITS[:1], feasible, rng)[0]) for _ in range(500)]
        self.assertEqual(set(picks), {0})

    def test_apply_mask_uses_neg_large_not_inf(self):
        masked = apply_mask(LOGITS[:1], np.array([[True, False, True]]))
        self.assertEqual(float(masked[0, 1]), NEG_LARGE)
        self.assertTrue(np.all(np.isfinite(masked)))


class TestTest5MaskIsDeterministicFromState(unittest.TestCase):
    def setUp(self):
        # obs version is module-global: never leak v3 into other test modules
        self._saved = eo.OBS_VERSION

    def tearDown(self):
        eo.set_obs_version(self._saved)

    def test_same_observation_gives_same_mask(self):
        eo.set_obs_version("v3")
        idx = eo.feasibility_channel_indices()
        packed = np.random.RandomState(2).normal(size=(4, 20, eo.PACKED_DIM))
        packed[..., idx[1]] = 0.0        # MEC infeasible everywhere
        packed[..., idx[0]] = 1.0
        packed[..., idx[2]] = 1.0
        m1 = mask_from_observation(packed, idx)
        m2 = mask_from_observation(packed, idx)
        np.testing.assert_array_equal(m1, m2)
        self.assertFalse(bool(m1[..., 1].any()))
        self.assertTrue(bool(m1[..., 0].all()))

    def test_indices_match_the_schema(self):
        eo.set_obs_version("v3")
        idx = eo.feasibility_channel_indices()
        names = [eo.FEATURE_NAMES[i] for i in idx]
        self.assertEqual(names, ["feasible_ue", "feasible_mec", "feasible_helper"])

    def test_v1_v2_have_no_feasibility_channels(self):
        for version in ("v1", "v2"):
            eo.set_obs_version(version)
            with self.assertRaises(eo.EncoderGraphError):
                eo.feasibility_channel_indices()
        eo.set_obs_version("v3")

    def test_batch_shape_preserved(self):
        eo.set_obs_version("v3")
        idx = eo.feasibility_channel_indices()
        packed = np.ones((2, 20, eo.PACKED_DIM))
        mask = mask_from_observation(packed, idx)
        self.assertEqual(mask.shape, (2, 20, 3))


if __name__ == "__main__":
    unittest.main(verbosity=2)
