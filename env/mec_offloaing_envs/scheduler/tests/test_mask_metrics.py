#!/usr/bin/env python3
"""⑥b metric definitions: deterministic numpy tests (no TensorFlow).

Covers the seven agreed behaviours: all-valid, one-closed, forced, all-invalid,
fail-loud on a missing/mis-shaped mask, token-weighted aggregation across paths
of different lengths, and off-mode zeros.
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

from env.mec_offloaing_envs.scheduler.mask_metrics import (  # noqa: E402
    METRIC_KEYS,
    RATE_KEYS,
    accumulate,
    merge,
    rates,
)
from env.mec_offloaing_envs.scheduler.masking import N_ACTIONS  # noqa: E402

ALL = np.ones((1, N_ACTIONS), dtype=bool)


def logits_with_argmax(action: int, boost: float = 5.0) -> np.ndarray:
    """Logits whose raw argmax is `action` (for deterministic argmax tests)."""
    row = np.zeros((1, N_ACTIONS), dtype=np.float64)
    row[0, action] = boost
    return row


class TestAllValid(unittest.TestCase):
    def test_every_rate_is_zero(self):
        acc = accumulate(
            pre_guard=ALL,
            actions=np.array([0]),
            raw_logits=logits_with_argmax(2),
            values=np.array([0.25]),
            require_mask=True,
        )
        out = rates(acc)
        for key in RATE_KEYS:
            self.assertEqual(out[key], 0.0, key)
        self.assertGreater(out["policy/entropy_valid"], 0.0)
        self.assertAlmostEqual(out["critic/value_abs_max"], 0.25)


class TestOneActionClosed(unittest.TestCase):
    def setUp(self):
        self.guard = np.array([[True, False, True]])

    def test_active_and_valid_sample(self):
        acc = accumulate(
            pre_guard=self.guard,
            actions=np.array([2]),
            raw_logits=logits_with_argmax(2),
            values=np.array([0.1]),
            require_mask=True,
        )
        out = rates(acc)
        self.assertEqual(out["mask/active_rate"], 1.0)
        self.assertEqual(out["mask/forced_rate"], 0.0)
        self.assertEqual(out["mask/all_invalid_rate"], 0.0)
        self.assertEqual(out["policy/invalid_action_rate"], 0.0)
        self.assertEqual(out["policy/argmax_masked_rate"], 0.0)

    def test_closed_raw_argmax_counts(self):
        # raw argmax sits on the closed action -> the metric must see it, which
        # is exactly what a masked-logits argmax could not report
        acc = accumulate(
            pre_guard=self.guard,
            actions=np.array([0]),
            raw_logits=logits_with_argmax(1),
            values=np.array([0.1]),
            require_mask=True,
        )
        out = rates(acc)
        self.assertEqual(out["policy/argmax_masked_rate"], 1.0)
        self.assertEqual(out["policy/invalid_action_rate"], 0.0)

    def test_entropy_is_over_the_open_actions_only(self):
        # two open actions with equal logits -> log(2) nats, never log(3)
        guard = np.array([[True, False, True]])
        raw = np.array([[1.0, 9.0, 1.0]])
        acc = accumulate(
            pre_guard=guard,
            actions=np.array([0]),
            raw_logits=raw,
            values=np.array([0.0]),
            require_mask=True,
        )
        self.assertAlmostEqual(rates(acc)["policy/entropy_valid"], np.log(2.0), places=9)


class TestForcedRow(unittest.TestCase):
    def test_valid_count_one(self):
        guard = np.array([[False, True, False]])
        # raw argmax on the OPEN action -> nothing to report
        acc = accumulate(
            pre_guard=guard,
            actions=np.array([1]),
            raw_logits=np.array([[0.1, 0.9, 0.2]]),
            values=np.array([0.0]),
            require_mask=True,
        )
        out = rates(acc)
        self.assertEqual(out["mask/forced_rate"], 1.0)
        self.assertEqual(out["mask/active_rate"], 1.0)
        self.assertEqual(out["policy/entropy_valid"], 0.0)   # one action, no doubt
        self.assertEqual(out["policy/argmax_masked_rate"], 0.0)
        self.assertEqual(out["policy/invalid_action_rate"], 0.0)

    def test_forced_row_with_raw_argmax_on_the_closed_action(self):
        # a forced row is exactly where an unmasked policy would have failed;
        # the metric must still flag its raw argmax
        acc = accumulate(
            pre_guard=np.array([[False, True, False]]),
            actions=np.array([1]),
            raw_logits=np.array([[0.3, 0.2, 0.1]]),
            values=np.array([0.0]),
            require_mask=True,
        )
        out = rates(acc)
        self.assertEqual(out["mask/forced_rate"], 1.0)
        self.assertEqual(out["policy/argmax_masked_rate"], 1.0)
        self.assertEqual(out["policy/invalid_action_rate"], 0.0)


class TestAllInvalidRow(unittest.TestCase):
    def setUp(self):
        self.guard = np.array([[False, False, False], [True, True, True]])

    def test_counted_before_the_guard(self):
        acc = accumulate(
            pre_guard=self.guard,
            actions=np.array([0, 1]),
            raw_logits=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            values=np.array([0.5, 0.5]),
            require_mask=True,
        )
        out = rates(acc)
        self.assertEqual(out["mask/all_invalid_rate"], 0.5)
        self.assertEqual(out["mask/active_rate"], 0.5)
        # the guard frees the dead-end row, so nothing is an invalid action
        self.assertEqual(out["policy/invalid_action_rate"], 0.0)
        self.assertTrue(np.isfinite(out["policy/entropy_valid"]))
        self.assertGreater(out["policy/entropy_valid"], 0.0)

    def test_dead_end_excluded_from_argmax_numerator(self):
        # even with an all-closed row the numerator must stay 0 there while the
        # denominator still counts both tokens
        acc = accumulate(
            pre_guard=np.array([[False, False, False], [True, False, True]]),
            actions=np.array([0, 1]),
            # row 0: dead end, argmax on a closed action -> excluded from numerator
            # row 1: argmax on the closed middle action -> counted
            raw_logits=np.array([[9.0, 0.0, 0.0], [0.0, 9.0, 0.0]]),
            values=np.array([0.0, 0.0]),
            require_mask=True,
        )
        self.assertEqual(rates(acc)["policy/argmax_masked_rate"], 0.5)


class TestFailLoud(unittest.TestCase):
    def test_missing_mask_in_active_mode(self):
        with self.assertRaises(ValueError):
            accumulate(
                pre_guard=None,
                actions=np.array([0]),
                raw_logits=logits_with_argmax(0),
                values=np.array([0.0]),
                require_mask=True,
            )

    def test_shape_mismatch(self):
        with self.assertRaises(ValueError):
            accumulate(
                pre_guard=np.ones((2, N_ACTIONS), dtype=bool),
                actions=np.array([0, 0, 0]),
                raw_logits=np.zeros((3, N_ACTIONS)),
                values=np.zeros(3),
                require_mask=True,
            )

    def test_action_shape_must_drop_the_action_axis(self):
        with self.assertRaises(ValueError):
            accumulate(
                pre_guard=ALL,
                actions=np.zeros((1, 1), dtype=np.int64),
                raw_logits=np.zeros((1, N_ACTIONS)),
                values=np.zeros(1),
                require_mask=True,
            )

    def test_non_finite_values(self):
        with self.assertRaises(ValueError):
            accumulate(
                pre_guard=ALL,
                actions=np.array([0]),
                raw_logits=logits_with_argmax(0),
                values=np.array([np.nan]),
                require_mask=True,
            )

    def test_out_of_range_action(self):
        with self.assertRaises(ValueError):
            accumulate(
                pre_guard=ALL,
                actions=np.array([3]),
                raw_logits=logits_with_argmax(0),
                values=np.array([0.0]),
                require_mask=True,
            )

    def test_non_finite_mask_is_rejected(self):
        # NaN would silently cast to True and produce a fake but plausible report
        for bad in (np.nan, np.inf, -np.inf):
            with self.assertRaises(ValueError):
                accumulate(
                    pre_guard=np.array([[bad, 1.0, 1.0]]),
                    actions=np.array([0]),
                    raw_logits=logits_with_argmax(0),
                    values=np.array([0.0]),
                    require_mask=True,
                )

    def test_non_binary_mask_is_rejected(self):
        for bad in (0.5, -1.0, 2.0):
            with self.assertRaises(ValueError):
                accumulate(
                    pre_guard=np.array([[bad, 1.0, 1.0]]),
                    actions=np.array([0]),
                    raw_logits=logits_with_argmax(0),
                    values=np.array([0.0]),
                    require_mask=True,
                )

    def test_binary_forms_are_accepted(self):
        for good in (
            np.array([[True, False, True]]),
            np.array([[1.0, 0.0, 1.0]]),
            np.array([[1, 0, 1]]),
        ):
            acc = accumulate(
                pre_guard=good,
                actions=np.array([0]),
                raw_logits=logits_with_argmax(0),
                values=np.array([0.0]),
                require_mask=True,
            )
            self.assertEqual(rates(acc)["mask/active_rate"], 1.0)

    def test_empty_batch(self):
        with self.assertRaises(ValueError):
            accumulate(
                pre_guard=np.zeros((0, N_ACTIONS), dtype=bool),
                actions=np.zeros(0, dtype=np.int64),
                raw_logits=np.zeros((0, N_ACTIONS)),
                values=np.zeros(0),
                require_mask=True,
            )


class TestTokenWeightedAggregation(unittest.TestCase):
    def test_short_and_long_paths(self):
        # path A: 1 token, fully masked. path B: 3 tokens, nothing masked.
        # unweighted averaging of paths would give 0.5; tokens give 0.25
        a = accumulate(
            pre_guard=np.array([[True, False, True]]),
            actions=np.array([0]),
            raw_logits=np.array([[1.0, 0.0, 0.0]]),
            values=np.array([0.0]),
            require_mask=True,
        )
        b = accumulate(
            pre_guard=np.ones((3, N_ACTIONS), dtype=bool),
            actions=np.array([0, 1, 2]),
            raw_logits=np.zeros((3, N_ACTIONS)),
            values=np.array([0.0, 0.0, 0.5]),
            require_mask=True,
        )
        out = rates(merge([a, b]))
        self.assertAlmostEqual(out["mask/active_rate"], 0.25)
        self.assertAlmostEqual(out["critic/value_abs_max"], 0.5)   # global max
        self.assertEqual(out["policy/invalid_action_rate"], 0.0)

    def test_merge_is_max_for_values_and_sum_for_counts(self):
        accs = [
            accumulate(
                pre_guard=ALL,
                actions=np.array([0]),
                raw_logits=np.zeros((1, N_ACTIONS)),
                values=np.array([v]),
                require_mask=True,
            )
            for v in (0.2, -0.7, 0.3)
        ]
        out = rates(merge(accs))
        self.assertAlmostEqual(out["critic/value_abs_max"], 0.7)
        self.assertEqual(out["mask/active_rate"], 0.0)

    def test_merge_of_many_fits_in_a_float32_log(self):
        # token counts are summed as ints, so 100k tokens do not lose precision
        accs = [
            accumulate(
                pre_guard=np.array([[True, False, True]]),
                actions=np.array([0]),
                raw_logits=np.zeros((1, N_ACTIONS)),
                values=np.array([0.0]),
                require_mask=True,
            )
            for _ in range(1000)
        ]
        out = rates(merge(accs))
        self.assertAlmostEqual(out["mask/active_rate"], 1.0, places=12)


class TestMetricWiring(unittest.TestCase):
    """Static wiring guards: TF is unavailable locally, so assert the plumbing.

    The metrics must come from the rollout sess.run, travel with the batch, and be
    logged per iteration; a half-applied edit would silently drop them.
    """

    def setUp(self):
        self.repo = Path(__file__).resolve().parents[4]

    def _src(self, rel):
        return (self.repo / rel).read_text()

    def test_policy_captures_raw_logits_in_the_rollout_run(self):
        src = self._src("policies/meta_seq2seq_policy.py")
        self.assertIn("self.network.sample_decoder_logits_raw,", src)
        self.assertIn("self.last_raw_logits = np.asarray(raw_logits)", src)
        # the public 3-tuple return must survive
        self.assertIn("return actions, logits, v_value", src)

    def test_off_mode_builds_no_mask_node(self):
        src = self._src("policies/meta_seq2seq_policy.py")
        self.assertIn("return logits, None", src)
        self.assertIn("self.dead_end_rows = None", src)

    def test_samplers_store_raw_logits(self):
        for rel in ("samplers/seq2seq_meta_sampler.py", "samplers/seq2seq_sampler.py"):
            self.assertIn('path_dict["raw_logits"]', self._src(rel), rel)

    def test_processors_compute_the_accumulator(self):
        for rel in (
            "samplers/seq2seq_meta_sampler_process.py",
            "samplers/seq2seq_sampler_process.py",
        ):
            src = self._src(rel)
            self.assertIn("samples_data['mask_accumulator'] = accumulate(", src, rel)
            self.assertIn("require_mask=self.resolved_mask_mode()", src, rel)
            self.assertIn('"raw_logits": raw_logits', src, rel)

    def test_trainer_logs_every_metric_each_iteration(self):
        src = self._src("meta_trainer.py")
        self.assertIn("mask_metric_rates(merge_mask_metrics(accumulators))", src)
        self.assertIn("logger.logkv(metric_name, float(metric_value))", src)
        # the update must REUSE the rollout mask, never rebuild it
        self.assertNotIn("observation_mask", src)
        self.assertIn("mask_metric_rates(merge_mask_metrics(accumulators))", src)

    def test_metric_names_are_frozen(self):
        self.assertEqual(
            METRIC_KEYS,
            (
                "mask/active_rate",
                "mask/forced_rate",
                "mask/all_invalid_rate",
                "policy/invalid_action_rate",
                "policy/argmax_masked_rate",
                "policy/entropy_valid",
                "critic/value_abs_max",
            ),
        )
        self.assertEqual(RATE_KEYS, METRIC_KEYS[:5])


class TestOffMode(unittest.TestCase):
    def test_five_rates_zero_and_real_entropy_value(self):
        acc = accumulate(
            pre_guard=None,
            actions=np.array([0, 1]),
            raw_logits=np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 2.0]]),
            values=np.array([-0.4, 0.2]),
            require_mask=False,
        )
        out = rates(acc)
        for key in RATE_KEYS:
            self.assertEqual(out[key], 0.0, key)
        self.assertGreater(out["policy/entropy_valid"], 0.0)
        self.assertAlmostEqual(out["critic/value_abs_max"], 0.4)
        self.assertEqual(set(out), set(METRIC_KEYS))

    def test_off_mode_still_ignores_a_stray_mask(self):
        # explicit mask with require_mask=False is respected, not discarded
        acc = accumulate(
            pre_guard=np.array([[True, False, True]]),
            actions=np.array([0]),
            raw_logits=np.zeros((1, N_ACTIONS)),
            values=np.array([0.0]),
            require_mask=False,
        )
        self.assertEqual(rates(acc)["mask/active_rate"], 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
