#!/usr/bin/env python3
"""Phase 3 learning-contract tests. Numpy only. No GPU. No training."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spec.learning_ops import clipped_value_prediction, mean_pseudogradient
from spec.split_loader import (
    assert_train_prefixes,
    graph_indices_for_role,
    meta_test_distribution_ids,
    meta_train_distribution_ids,
    meta_train_graph_prefixes,
    validation_distribution_ids,
)


class TestValueClip(unittest.TestCase):
    def test_clip_uses_v_old_plus_clipped_delta(self):
        v_old = np.array([0.0, 1.0, -1.0])
        v_new = np.array([5.0, 1.1, -1.5])
        out = clipped_value_prediction(v_old, v_new, 0.2)
        expected = np.array([0.2, 1.1, -1.2])
        np.testing.assert_allclose(out, expected)

    def test_wrong_legacy_formula_is_different(self):
        v_old = np.array([0.0])
        v_new = np.array([5.0])
        correct = clipped_value_prediction(v_old, v_new, 0.2)
        legacy = v_new + np.clip(v_new - v_old, -0.2, 0.2)
        self.assertFalse(np.allclose(correct, legacy))


class TestMeanPseudogradient(unittest.TestCase):
    def test_order_invariant(self):
        theta0 = [np.array([1.0, 2.0])]
        a = [np.array([1.1, 2.2])]
        b = [np.array([0.5, 1.5])]
        g_ab = mean_pseudogradient(theta0, [a, b], alpha=5e-4, k_steps=3)
        g_ba = mean_pseudogradient(theta0, [b, a], alpha=5e-4, k_steps=3)
        np.testing.assert_allclose(g_ab[0], g_ba[0])

    def test_denominator_is_alpha_times_k_not_batch(self):
        theta0 = [np.array([1.0])]
        adapted = [[np.array([0.0])]]
        g = mean_pseudogradient(theta0, adapted, alpha=0.5, k_steps=2)
        np.testing.assert_allclose(g[0], np.array([1.0]))


class TestSplitWiring(unittest.TestCase):
    def test_meta_train_is_15_and_disjoint(self):
        train = set(meta_train_distribution_ids())
        val = set(validation_distribution_ids())
        test = set(meta_test_distribution_ids())
        self.assertEqual(len(train), 15)
        self.assertEqual(len(val), 5)
        self.assertEqual(len(test), 5)
        self.assertFalse(train & val)
        self.assertFalse(train & test)
        self.assertFalse(val & test)

    def test_legacy_19_path_list_is_rejected(self):
        leaked = meta_train_graph_prefixes() + [
            "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_2/random.20.",
            "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_7/random.20.",
            "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_12/random.20.",
            "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_23/random.20.",
        ]
        with self.assertRaises(ValueError):
            assert_train_prefixes(leaked)

    def test_train_prefixes_pass(self):
        assert_train_prefixes(meta_train_graph_prefixes())

    def test_held_out_support_query_counts(self):
        dist_id = meta_test_distribution_ids()[0]
        support = graph_indices_for_role(dist_id, "meta_test_support")
        query = graph_indices_for_role(dist_id, "meta_test_query")
        self.assertEqual(len(support), 20)
        self.assertEqual(len(query), 80)
        self.assertFalse(set(support) & set(query))


class TestMRLCOSourceContract(unittest.TestCase):
    def test_value_clip_source(self):
        text = (ROOT / "meta_algos" / "MRLCO.py").read_text()
        self.assertIn("self.old_v[i] + tf.clip_by_value", text)
        self.assertNotIn("self.vpred[i] + tf.clip_by_value(self.vpred[i] - self.old_v[i]", text)

    def test_outer_update_is_single_apply(self):
        text = (ROOT / "meta_algos" / "MRLCO.py").read_text()
        self.assertIn("mean_pseudogradient(", text)
        self.assertIn("sess.run(self._outer_train", text)
        # sequential per-task outer apply used to live inside `for i in range(self.meta_batch_size)`
        self.assertNotIn("/ self.meta_batch_size / self.update_numbers", text)

    def test_fresh_inner_optimizer(self):
        text = (ROOT / "meta_algos" / "MRLCO.py").read_text()
        self.assertIn("def reset_inner_optimizer", text)
        self.assertIn("inner_adam_task_", text)

    def test_trainer_uses_frozen_split_and_k_steps(self):
        text = (ROOT / "meta_trainer.py").read_text()
        self.assertIn("meta_train_graph_prefixes", text)
        self.assertIn("num_inner_grad_steps=K_STEPS", text)
        self.assertIn("K_STEPS = 3", text)
        self.assertIn("PPO_BATCH_SIZE = 20", text)
        self.assertNotIn("num_inner_grad_steps=1", text)
        self.assertNotIn("inner_batch_size=10", text)
        self.assertIn("CUDA_VISIBLE_DEVICES", text)


if __name__ == "__main__":
    unittest.main()
