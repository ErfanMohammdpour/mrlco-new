#!/usr/bin/env python3
"""⑥b acceptance tests for the masked action distribution (pure numpy).

These are the semantics the TF graph must mirror. Test names follow the agreed
list; the PPO-side checks are expressed as log-probability identities so they can
be verified without TensorFlow.
"""

from __future__ import annotations

import os
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
    MASK_MODE_OFF,
    MASK_MODE_RUNTIME,
    MASK_MODE_STATIC,
    NEG_LARGE,
    apply_mask,
    dead_end_guard,
    distribution,
    entropy_valid,
    intersect_masks,
    likelihood_ratio,
    mask_from_observation,
    mask_mode_active,
    masked_log_softmax,
    masked_softmax,
    observation_mask,
    resolve_mask_mode,
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


class TestTest6MaskModeAndFeed(unittest.TestCase):
    """Mode resolution and the feed-level contract used by the TF plumbing."""

    def setUp(self):
        self._saved = eo.OBS_VERSION
        self._env_saved = os.environ.pop("MARGO_MASK_MODE", None)

    def tearDown(self):
        eo.set_obs_version(self._saved)
        if self._env_saved is not None:
            os.environ["MARGO_MASK_MODE"] = self._env_saved
        else:
            os.environ.pop("MARGO_MASK_MODE", None)

    def test_default_is_off(self):
        self.assertEqual(resolve_mask_mode(), MASK_MODE_OFF)
        self.assertFalse(mask_mode_active())

    def test_explicit_and_env_resolution(self):
        self.assertEqual(resolve_mask_mode("STATIC "), MASK_MODE_STATIC)
        os.environ["MARGO_MASK_MODE"] = "runtime"
        self.assertEqual(resolve_mask_mode(), MASK_MODE_RUNTIME)
        self.assertTrue(mask_mode_active())
        # an explicit value always wins over the environment
        self.assertEqual(resolve_mask_mode("off"), MASK_MODE_OFF)

    def test_unknown_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            resolve_mask_mode("sometimes")

    def test_off_mode_returns_no_mask(self):
        eo.set_obs_version("v3")
        packed = np.ones((2, 20, eo.PACKED_DIM))
        self.assertIsNone(observation_mask(packed, mode="off"))

    def test_active_mode_without_v3_fails_loudly(self):
        eo.set_obs_version("v1")
        packed = np.ones((2, 20, eo.PACKED_DIM))
        with self.assertRaises(eo.EncoderGraphError):
            observation_mask(packed, mode="static")

    def _hard_obs(self, *, hard=True, feasible=(True, True, True)):
        """Packed obs with proof-feasibility and the hard-deadline flag set."""
        eo.set_obs_version("v3")
        packed = np.zeros((1, 2, eo.PACKED_DIM))
        idx = eo.feasibility_channel_indices()
        for a, ch in enumerate(idx):
            packed[..., ch] = 1.0 if feasible[a] else 0.0
        packed[..., eo.hard_deadline_channel_index()] = 1.0 if hard else 0.0
        return packed

    def test_static_mask_matches_action_order(self):
        packed = self._hard_obs(hard=True, feasible=(False, False, True))
        mask = observation_mask(packed, mode="static")
        np.testing.assert_array_equal(mask, np.array([[[False, False, True]] * 2]))

    def test_soft_deadline_is_not_shielded(self):
        # proof says MEC is unreachable in time, but a soft deadline only prices
        # that lateness -- masking it would change the optimisation problem
        packed = self._hard_obs(hard=False, feasible=(True, False, False))
        mask = observation_mask(packed, mode="static")
        np.testing.assert_array_equal(mask, np.ones((1, 2, 3), dtype=bool))

    def test_hard_deadline_is_shielded(self):
        packed = self._hard_obs(hard=True, feasible=(True, False, False))
        mask = observation_mask(packed, mode="static")
        np.testing.assert_array_equal(
            mask, np.array([[[True, False, False]] * 2])
        )

    def test_static_base_mask_ignores_deadline_type(self):
        # the featurised signal stays available for soft/firm; only the SHIELD
        # is gated, so the policy can still learn to price lateness
        packed = self._hard_obs(hard=False, feasible=(True, False, True))
        from env.mec_offloaing_envs.scheduler.masking import static_base_mask

        base = static_base_mask(packed)
        np.testing.assert_array_equal(
            base, np.array([[[True, False, True]] * 2])
        )

    def test_runtime_mode_refuses_to_derive_a_mask(self):
        packed = self._hard_obs(hard=True, feasible=(True, True, True))
        with self.assertRaises(ValueError):
            observation_mask(packed, mode="runtime")

    def test_runtime_shield_is_intersected_explicitly(self):
        # what a future token-by-token runtime implementation must do: the
        # explicit env shield ANDs with the static base, never replaces it
        packed = self._hard_obs(hard=True, feasible=(True, True, False))
        from env.mec_offloaing_envs.scheduler.masking import static_base_mask

        base = static_base_mask(packed)
        shield = np.ones((1, 2, 3), dtype=bool)
        shield[..., 0] = False                      # env says UE not allowed
        merged = intersect_masks(base, shield)
        np.testing.assert_array_equal(
            merged, np.array([[[False, True, False]] * 2])
        )

    def test_intersect_rejects_shape_mismatch(self):
        with self.assertRaises(ValueError):
            intersect_masks(np.ones((1, 2, 3), dtype=bool), np.ones((2, 3), dtype=bool))

    def test_tf_wiring_covers_every_decoder(self):
        """Wiring completeness guard: no decoder may bypass the mask.

        TensorFlow is unavailable locally, so instead of building the graph we
        assert the masking helper is applied at every decoder output site and
        that the sampler stores the mask it actually used.
        """
        repo = Path(__file__).resolve().parents[4]
        policy_src = (repo / "policies" / "meta_seq2seq_policy.py").read_text()
        self.assertGreaterEqual(policy_src.count("mask_logits_tf("), 4)
        for attr in ("decoder_logits_raw", "sample_decoder_logits_raw", "greedy_decoder_logits_raw"):
            self.assertIn(attr, policy_src)
        self.assertIn("self.last_feasible_mask = feasible_mask", policy_src)
        self.assertIn("feasibility_feed", policy_src)

        sampler_src = (repo / "samplers" / "seq2seq_meta_sampler.py").read_text()
        self.assertIn('running_paths[idx]["feasible"]', sampler_src)
        self.assertIn('path_dict["feasible"]', sampler_src)

        # critic must read the RAW logits: a -1e9 entry inside the dense layer
        # contaminates vf for the valid actions too (audit blocker 1)
        self.assertEqual(policy_src.count("dense(self.decoder_logits_raw"), 1)
        self.assertEqual(policy_src.count("dense(self.sample_decoder_logits_raw"), 1)
        self.assertEqual(policy_src.count("dense(self.greedy_decoder_logits_raw"), 1)
        self.assertNotIn("dense(self.decoder_logits,", policy_src)
        self.assertNotIn("dense(self.sample_decoder_logits,", policy_src)
        self.assertNotIn("dense(self.greedy_decoder_logits,", policy_src)
        # teacher-forced prediction must follow the shield, not the raw argmax
        self.assertIn("tf.argmax(self.decoder_logits, axis=-1", policy_src)

        ppo_src = (repo / "meta_algos" / "ppo_offloading.py").read_text()
        mrclo_src = (repo / "meta_algos" / "MRLCO.py").read_text()
        for src, label in ((ppo_src, "ppo_offloading"), (mrclo_src, "MRLCO")):
            self.assertIn("feasible", src, label)
            self.assertIn("never recomputed at update time", src, label)


if __name__ == "__main__":
    unittest.main(verbosity=2)
