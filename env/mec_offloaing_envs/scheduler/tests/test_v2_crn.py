#!/usr/bin/env python3
"""v2 stage-7 tests: CRN protocol (Gumbel common random numbers) - no TF needed."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spec.automotive_training.v2.crn import (  # noqa: E402
    CRNError, DEFAULT_R_SELECT, DEFAULT_S_SELECT, EVALUATION_PHASE_SALT, PHASE_EVAL,
    PHASE_TRAIN, PROTOCOL_ID, TRAINING_PHASE_SALT,
    categorical_probabilities, draw_phase_of, frozen_protocol, gumbel_argmax, gumbel_noise,
    nested_aggregate, paired_delta, protocol_sha, seed_pair, training_seed_pair,
)


class TestSeedAndNoise(unittest.TestCase):
    def test_seed_pair_deterministic_and_key_sensitive(self):
        a = seed_pair(PROTOCOL_ID, 0, 0, "g1")
        self.assertEqual(a, seed_pair(PROTOCOL_ID, 0, 0, "g1"))
        self.assertNotEqual(a, seed_pair(PROTOCOL_ID, 0, 1, "g1"))
        self.assertNotEqual(a, seed_pair(PROTOCOL_ID, 1, 0, "g1"))
        self.assertNotEqual(a, seed_pair(PROTOCOL_ID, 0, 0, "g2"))
        self.assertNotEqual(a, seed_pair("other_protocol", 0, 0, "g1"))

    def test_noise_is_reproducible_and_independent_per_sample(self):
        n1 = gumbel_noise(PROTOCOL_ID, 0, 0, "g", 20, 3)
        n2 = gumbel_noise(PROTOCOL_ID, 0, 0, "g", 20, 3)
        n3 = gumbel_noise(PROTOCOL_ID, 0, 1, "g", 20, 3)
        self.assertTrue(np.array_equal(n1, n2), "same CRN key must give identical noise")
        self.assertFalse(np.array_equal(n1, n3), "different sample must give different noise")
        self.assertEqual(n1.shape, (20, 3))

    def test_gumbel_noise_moments(self):
        noise = gumbel_noise(PROTOCOL_ID, 0, 0, "g", 20000, 3)
        self.assertAlmostEqual(float(np.mean(noise)), 0.5772, delta=0.05)
        self.assertAlmostEqual(float(np.std(noise)), 1.2825, delta=0.05)


class TestGumbelArgmax(unittest.TestCase):
    def test_same_noise_same_actions_and_different_noise_differs(self):
        logits = np.tile(np.array([0.2, 0.5, 0.3]), (50, 1))
        a1 = gumbel_argmax(logits, gumbel_noise(PROTOCOL_ID, 0, 0, "g", 50, 3))
        a2 = gumbel_argmax(logits, gumbel_noise(PROTOCOL_ID, 0, 0, "g", 50, 3))
        a3 = gumbel_argmax(logits, gumbel_noise(PROTOCOL_ID, 0, 1, "g", 50, 3))
        self.assertTrue(np.array_equal(a1, a2))
        self.assertFalse(np.array_equal(a1, a3))

    def test_argmax_matches_categorical_distribution(self):
        # Gumbel-max must reproduce the categorical law: empirical frequencies ~ softmax
        logits = np.tile(np.array([0.0, 1.0, 2.0]), (40000, 1))
        counts = np.zeros(3)
        for s in range(8):
            actions = gumbel_argmax(logits, gumbel_noise(PROTOCOL_ID, 0, s, "g", 40000, 3))
            for a in range(3):
                counts[a] += float(np.mean(actions == a))
        counts /= 8.0
        expected = categorical_probabilities(np.array([[0.0, 1.0, 2.0]]))[0]
        for a in range(3):
            self.assertAlmostEqual(counts[a], expected[a], delta=0.02)

    def test_shape_mismatch_rejected(self):
        with self.assertRaises(CRNError):
            gumbel_argmax(np.zeros((4, 3)), np.zeros((5, 3)))

    def test_masking_removes_actions(self):
        logits = np.zeros((10, 3))
        mask = np.tile(np.array([1.0, 0.0, 1.0]), (10, 1))
        actions = gumbel_argmax(logits, gumbel_noise(PROTOCOL_ID, 0, 0, "g", 10, 3), mask=mask)
        self.assertTrue(np.all(actions != 1))


class TestAggregation(unittest.TestCase):
    def test_nested_mean_and_stderr(self):
        agg = nested_aggregate([[1.0, 1.2], [2.0, 2.2], [3.0, 3.2]])
        self.assertAlmostEqual(agg.mean, 2.1)
        self.assertAlmostEqual(agg.per_replicate_means[0], 1.1)
        self.assertAlmostEqual(agg.per_sample_means[0], 2.0)
        self.assertGreater(agg.stderr, 0.0)
        self.assertEqual(agg.replicates, 3)
        self.assertEqual(agg.samples_per_replicate, 2)

    def test_ragged_grid_rejected(self):
        with self.assertRaises(CRNError):
            nested_aggregate([[1.0], [1.0, 2.0]])
        with self.assertRaises(CRNError):
            nested_aggregate([])

    def test_paired_delta(self):
        out = paired_delta([1.0, 2.0, 3.0], [0.5, 1.5, 2.5])
        self.assertAlmostEqual(out["mean"], 0.5)
        self.assertEqual(out["n"], 3)
        with self.assertRaises(CRNError):
            paired_delta([1.0], [1.0, 2.0])

    def test_protocol_freeze(self):
        doc = frozen_protocol()
        self.assertEqual(doc["protocol_id"], PROTOCOL_ID)
        self.assertEqual(doc["r_select"], DEFAULT_R_SELECT)
        self.assertEqual(doc["s_select"], DEFAULT_S_SELECT)
        self.assertIn("Gumbel", doc["action_rule"])
        self.assertEqual(protocol_sha(), protocol_sha())

    def test_advertised_key_matches_the_implementation(self):
        """F18: the frozen document must not advertise a key the code does not use.

        `key` used to include `token_index`/`action_index` while `seed_pair` keyed only on
        (protocol, r, s, graph). The document now separates the seed key from the full draw key,
        and `seed_pair` is required to be insensitive to the token/action axes.
        """
        doc = frozen_protocol()
        self.assertEqual(doc["key"], "(protocol_id, replicate, sample, graph_id)")
        self.assertIn("token_index", doc["draw_key"])
        self.assertIn("action_index", doc["draw_key"])
        a = seed_pair(PROTOCOL_ID, 1, 2, "g")
        self.assertEqual(a, seed_pair(PROTOCOL_ID, 1, 2, "g"))
        # no token/action argument exists at all: the seed cannot depend on them
        import inspect
        self.assertEqual(list(inspect.signature(seed_pair).parameters),
                         ["protocol_id", "replicate", "sample", "graph_id"])
        self.assertIn("training", doc["phases"])
        self.assertIn("evaluation", doc["phases"])

    def test_per_token_substreams_are_independent(self):
        """A per-token substream means one token's noise does not move another token's draw:
        perturbing token 0's logits leaves every other token's action unchanged."""
        tokens, vocab = 24, 3
        logits = np.zeros((tokens, vocab))
        noise = gumbel_noise(PROTOCOL_ID, 0, 0, "g", tokens, vocab)
        base = gumbel_argmax(logits, noise)
        perturbed_logits = logits.copy()
        perturbed_logits[0] = [10.0, 0.0, 0.0]
        moved = gumbel_argmax(perturbed_logits, noise)
        self.assertEqual(int(moved[0]), 0)
        self.assertTrue(np.array_equal(base[1:], moved[1:]),
                        "only token 0 may move; a shared stream would move every token")
        # and the substreams are genuinely different draws, not the same row repeated
        self.assertFalse(np.array_equal(noise[0], noise[1]))
        self.assertFalse(np.array_equal(noise[1], noise[2]))

    def test_per_token_substream_seed_is_distinct_and_stable(self):
        """The per-token substream seed (the same derivation the TF helper uses) is a function of
        the token index: no two tokens can be handed the same seed, and it is reproducible."""
        from spec.automotive_training.v2.crn import _token_seed

        base = seed_pair(PROTOCOL_ID, 2, 3, "g")
        seeds = [_token_seed(base, t) for t in range(64)]
        self.assertEqual(len(set(seeds)), 64, "two tokens share a substream seed")
        self.assertEqual(seeds, [_token_seed(base, t) for t in range(64)])
        self.assertNotEqual(_token_seed(base, 0), _token_seed(seed_pair(PROTOCOL_ID, 2, 4, "g"), 0))
        for s in seeds:
            self.assertGreaterEqual(s, 0)
            self.assertLess(s, 2 ** 32)

    def test_evaluation_seed_is_stable_across_calls(self):
        """The evaluation key is a pure function: the same key gives the same pair FOREVER."""
        first = [seed_pair(PROTOCOL_ID, r, s, "g%d" % g)
                 for r in range(3) for s in range(3) for g in range(3)]
        second = [seed_pair(PROTOCOL_ID, r, s, "g%d" % g)
                  for r in range(3) for s in range(3) for g in range(3)]
        self.assertEqual(first, second)


class TestDrawPhasesAreDisjoint(unittest.TestCase):
    """F18: training and evaluation draw spaces cannot collide (pure parity argument)."""

    def test_phase_constants_have_opposite_parity(self):
        self.assertEqual(TRAINING_PHASE_SALT % 2, 1)
        self.assertEqual(EVALUATION_PHASE_SALT % 2, 0)
        self.assertEqual(draw_phase_of(training_seed_pair(0, 0)), PHASE_TRAIN)
        self.assertEqual(draw_phase_of(seed_pair(PROTOCOL_ID, 0, 0, "g")), PHASE_EVAL)

    def test_training_pair_encodes_seed_and_cursor(self):
        self.assertEqual(training_seed_pair(7, 3), (15, 3))
        self.assertEqual(training_seed_pair(0, 0), (1, 0))
        with self.assertRaises(CRNError):
            training_seed_pair(-1, 0)
        with self.assertRaises(CRNError):
            training_seed_pair(2 ** 31, 0)

    def test_no_training_key_equals_any_evaluation_key(self):
        train = {training_seed_pair(seed, cursor)
                 for seed in range(48) for cursor in range(48)}
        evaluation = {seed_pair(PROTOCOL_ID, r, s, "g%d" % g)
                      for r in range(8) for s in range(8) for g in range(8)}
        self.assertEqual(train & evaluation, set())
        self.assertFalse(any(draw_phase_of(p) == PHASE_EVAL for p in train))
        self.assertFalse(any(draw_phase_of(p) == PHASE_TRAIN for p in evaluation))

    def test_draw_phase_rejects_a_non_pair(self):
        with self.assertRaises(CRNError):
            draw_phase_of([1])
        with self.assertRaises(CRNError):
            draw_phase_of(None)


if __name__ == "__main__":
    unittest.main(verbosity=2)
