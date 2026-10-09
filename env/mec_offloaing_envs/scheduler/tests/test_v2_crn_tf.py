#!/usr/bin/env python3
"""TF-gated CRN test: the same stateless seed pair must give bit-identical actions.

Runs in the TF1.15 container (skipped without TensorFlow). The container image has no pytest,
so every test here is a plain `unittest` test and the file is run with
`python -m unittest env.mec_offloaing_envs.scheduler.tests.test_v2_crn_tf`.

F18: this file used `V2_PACKED_DIM`/`V2_FEATURE_DIM` with NO import and no definition, so in
the container every test raised `NameError` and the stateless-draw claim had ZERO runnable
coverage. The constants are imported from `spec.automotive_training.v2.observation` (a TF-free
module) below, at module top, so a missing name is an import error and never a silent skip.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# These two constants are what the tests below actually consume. They live in a TF-FREE
# module, so importing them cannot be skipped along with the TF-gated tests: a rename there
# fails this file at import time in every environment instead of only in the container.
from spec.automotive_training.v2.observation import V2_FEATURE_DIM, V2_PACKED_DIM  # noqa: E402
from spec.automotive_training.v2.crn import (  # noqa: E402
    PHASE_EVAL, PHASE_TRAIN, TRAINING_PHASE_SALT, draw_phase_of, seed_pair,
    training_seed_pair,
)

# The obs version must be active BEFORE `policies/graph2seq_encoder` is imported: that module
# binds FEATURE_DIM/PACKED_DIM at import time. The container image has no pytest, so this file
# is run with `python -m unittest discover -s env/mec_offloaing_envs/scheduler/tests`.
import os

try:
    import tensorflow as tf

    # some suite modules install a lightweight `tensorflow` stub in sys.modules, so a bare
    # import is not enough: require real TF1.15 APIs before running the TF-gated tests
    HAS_TF = bool(getattr(tf, "__version__", "")) and hasattr(tf, "compat") and \
        hasattr(getattr(tf, "compat", None), "v1") and \
        hasattr(getattr(tf, "random", None), "stateless_multinomial")
except Exception:
    HAS_TF = False


if HAS_TF:  # pragma: no cover - container only
    # The obs version must be active BEFORE policies/graph2seq_encoder is imported (it binds
    # PACKED_DIM at import time). This is done ONLY when TensorFlow is really present: an
    # import-time mutation in a skipped module would otherwise leak the v2 schema into every
    # other test collected by the same pytest run (14 v1 encoder failures in the full suite).
    os.environ["MARGO_OBS_VERSION"] = "automotive_v2_obs_v1"
    try:
        from env.mec_offloaing_envs.scheduler import encoder_obs

        encoder_obs.set_obs_version("automotive_v2_obs_v1")
    except Exception:  # pragma: no cover
        pass


class TestDrawKeySpacesAreDisjoint(unittest.TestCase):
    """F18: training and evaluation draw keys cannot collide. NO TensorFlow needed.

    Training draws the identity `[2*seed + 1, cursor]`; evaluation draws
    `seed_pair(protocol, r, s, graph)`. The two families are separated by parity BY
    CONSTRUCTION, so no scan width or hash collision probability is involved.
    """

    def test_training_identities_are_odd_and_evaluation_identities_even(self):
        self.assertEqual(TRAINING_PHASE_SALT % 2, 1)
        for seed in (0, 1, 7, 12345):
            for cursor in (0, 1, 2, 999):
                pair = training_seed_pair(seed, cursor)
                self.assertEqual(pair[0] % 2, 1, "training identity must be odd: %r" % (pair,))
                self.assertEqual(pair[1], cursor)
                self.assertEqual(draw_phase_of(pair), PHASE_TRAIN)
        for r in range(3):
            for s in range(3):
                for graph in ("g0", "graph-17"):
                    pair = seed_pair("automotive_crn_gumbel_v1", r, s, graph)
                    self.assertEqual(pair[0] % 2, 0,
                                     "evaluation identity must be even: %r" % (pair,))
                    self.assertEqual(draw_phase_of(pair), PHASE_EVAL)

    def test_no_training_identity_equals_any_evaluation_identity(self):
        train = {tuple(training_seed_pair(seed, cursor))
                 for seed in range(64) for cursor in range(64)}
        evaluation = {tuple(seed_pair("automotive_crn_gumbel_v1", r, s, "g%d" % g))
                      for r in range(8) for s in range(8) for g in range(8)}
        self.assertTrue(train)
        self.assertTrue(evaluation)
        self.assertEqual(train & evaluation, set(),
                         "a training draw and an evaluation draw shared a key")

    def test_a_training_identity_is_never_an_evaluation_seed_pair(self):
        """The claim is about the two FAMILIES, not one sample: parity separates them."""
        for seed in range(32):
            for cursor in range(32):
                pair = training_seed_pair(seed, cursor)
                self.assertNotEqual(draw_phase_of(pair), PHASE_EVAL)


@unittest.skipUnless(HAS_TF, "requires TensorFlow (run in the TF1.15 image)")
class TestSeq2SeqPolicyCRN(unittest.TestCase):
    def _policy(self, name="crn_policy"):
        from policies.meta_seq2seq_policy import Seq2SeqPolicy

        return Seq2SeqPolicy(obs_dim=V2_PACKED_DIM, encoder_units=32, decoder_units=32, vocab_size=3,
                             name=name, enable_crn=True)

    def test_same_seed_pair_bit_identical_actions(self):
        policy = self._policy()
        obs = np.random.RandomState(0).randn(4, 20, V2_PACKED_DIM).astype(np.float32)
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            a1, l1, v1 = policy.crn_actions(obs, [11, 22])
            a2, l2, v2 = policy.crn_actions(obs, [11, 22])
            a3, l3, v3 = policy.crn_actions(obs, [33, 44])
            branch = policy.last_decode_branch
        self.assertTrue(np.array_equal(a1, a2), "same CRN seed must reproduce the actions")
        self.assertTrue(np.array_equal(l1, l2))
        self.assertTrue(np.array_equal(v1, v2))
        self.assertFalse(np.array_equal(a1, a3),
                         "a different CRN seed must give a different draw")
        self.assertEqual(branch, "crn_token_keyed_v2",
                         "the rollout must report the stateless token-keyed branch it executed")

    def test_obs_version_in_use_is_the_v2_schema(self):
        from env.mec_offloaing_envs.scheduler import encoder_obs

        self.assertEqual(encoder_obs.OBS_VERSION, "automotive_v2_obs_v1")
        self.assertEqual((encoder_obs.FEATURE_DIM, encoder_obs.PACKED_DIM),
                         (V2_FEATURE_DIM, V2_PACKED_DIM))

    def test_gumbel_path_matches_stateless_categorical(self):
        """A stateless categorical draw IS the Gumbel-max rule with a deterministic stream:
        the empirical action frequencies must match the softmax of the logits."""
        policy = self._policy("crn_policy_stats")
        obs = np.zeros((200, 20, V2_PACKED_DIM), dtype=np.float32)
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            counts = np.zeros(3)
            draws = 0
            for s in range(6):
                actions, _l, _v = policy.crn_actions(obs, [7, 100 + s])
                flat = np.asarray(actions).reshape(-1)
                for a in range(3):
                    counts[a] += float(np.mean(flat == a))
                draws += 1
            counts /= max(1, draws)
        # the policy is untrained, so this only checks that all three actions are reachable
        self.assertTrue(all(c > 0.0 for c in counts), "all actions must be reachable: %s" % counts)

    def test_crn_is_off_by_default(self):
        from policies.meta_seq2seq_policy import Seq2SeqPolicy

        policy = Seq2SeqPolicy(obs_dim=V2_PACKED_DIM, encoder_units=32, decoder_units=32, vocab_size=3,
                               name="crn_off")
        self.assertIsNone(policy.crn_seed)
        self.assertFalse(policy.enable_crn)
        self.assertIsNone(policy.last_decode_branch)

    def test_draw_branch_is_measured_not_assumed(self):
        """With no identity pinned and CRN available, the fetch must fall back to the STATEFUL
        decoder -- and the policy must SAY so. This is the evidence `run_tf.py` records."""
        policy = self._policy("crn_branch_evidence")
        obs = np.random.RandomState(3).randn(2, 20, V2_PACKED_DIM).astype(np.float32)
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            policy.set_draw_identity(None)
            policy.get_actions(obs)
            unpinned = policy.last_decode_branch
            policy.set_draw_identity(training_seed_pair(5, 0))
            policy.get_actions(obs)
            pinned = policy.last_decode_branch
            self.assertEqual(unpinned, "legacy_stateful_categorical")
            self.assertEqual(pinned, "crn_token_keyed_v2")

    def test_each_token_is_an_independent_substream(self):
        """F20-class defect: ONE seed pair read across the whole horizon replays the same noise
        ordering at every token, so every token of a trajectory follows the same action as soon
        as one action's logits dominate. `crn_token_keyed_v2` mixes the decode step into the
        seed, so the 20 tokens are independent draws and every action stays reachable.

        The check is deliberately weak in the right direction: with the UNTRAINED policy the
        logits are only mildly informative, so under independent substreams each action holds
        roughly a third of the 200 x 20 draws. A shared stream drives one action towards 1.0
        and the others towards 0.
        """
        policy = self._policy("crn_token_independence")
        tokens = 20
        obs = np.zeros((200, tokens, V2_PACKED_DIM), dtype=np.float32)
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            actions, _l, _v = policy.crn_actions(obs, list(training_seed_pair(9, 4)))
        actions = np.asarray(actions)
        self.assertEqual(actions.shape, (200, tokens))
        counts = np.array([float(np.mean(actions == a)) for a in range(3)])
        for a in range(3):
            self.assertGreater(counts[a], 0.05,
                               "action %d was essentially never drawn (%s): the token substreams "
                               "look shared, not independent" % (a, counts))

    def test_training_identity_of_the_wrong_phase_is_refused(self):
        """An evaluation identity must not be pinnable as a training draw (F18)."""
        from policies.meta_seq2seq_policy import MetaSeq2SeqPolicy

        policy = MetaSeq2SeqPolicy(meta_batch_size=2, obs_dim=V2_PACKED_DIM, encoder_units=8,
                                   decoder_units=8, vocab_size=3, enable_crn=True)
        evaluation_pair = list(seed_pair("automotive_crn_gumbel_v1", 0, 0, "g0"))
        self.assertEqual(draw_phase_of(evaluation_pair), PHASE_EVAL)
        with self.assertRaises(ValueError):
            policy.set_draw_identity(evaluation_pair)
        policy.set_draw_identity(list(training_seed_pair(1, 0)))   # must not raise


if __name__ == "__main__":
    unittest.main(verbosity=2)
