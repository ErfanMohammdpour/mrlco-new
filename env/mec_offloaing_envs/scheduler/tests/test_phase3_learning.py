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

from spec.learning_ops import (
    clipped_value_prediction,
    composite_query_objective,
    expected_adam_apply_count,
    mean_pseudogradient,
    select_support_rows,
    shuffled_minibatch_slices,
)
from spec.eval_protocol import REQUIRED_LOG_FIELDS, protocol_log_kvs, require_sliced_task
from spec.split_loader import (
    assert_held_out_prefixes,
    assert_train_prefixes,
    graph_indices_for_role,
    meta_test_distribution_ids,
    meta_test_graph_prefixes,
    meta_train_distribution_ids,
    meta_train_graph_prefixes,
    split_version,
    support_query_tasks,
    validation_distribution_ids,
    validation_graph_prefixes,
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

    def test_validation_prefixes_are_held_out(self):
        assert_held_out_prefixes(validation_graph_prefixes(), "validation")
        assert_held_out_prefixes(meta_test_graph_prefixes(), "meta_test")

    def test_support_query_tasks_reject_train_dist(self):
        with self.assertRaises(ValueError):
            support_query_tasks(0, meta_train_distribution_ids()[0])

    def test_meta_test_support_query_tasks(self):
        support, query = support_query_tasks(0, 12)
        self.assertEqual(support["dist_index"], 0)
        self.assertEqual(len(support["graph_indices"]), 20)
        self.assertEqual(len(query["graph_indices"]), 80)
        self.assertFalse(set(support["graph_indices"]) & set(query["graph_indices"]))


class TestShuffleAndObjective(unittest.TestCase):
    def test_k_steps_apply_count_full_batch(self):
        self.assertEqual(expected_adam_apply_count(20, 20, 3), 3)
        self.assertEqual(expected_adam_apply_count(20, 20, 0), 0)
        self.assertEqual(expected_adam_apply_count(20, 10, 3), 6)

    def test_shuffled_epoch_covers_all(self):
        rng = np.random.RandomState(0)
        seen = []
        for idx in shuffled_minibatch_slices(20, 20, rng):
            seen.extend(idx.tolist())
        self.assertEqual(sorted(seen), list(range(20)))

    def test_select_support_rows_rejects_short(self):
        with self.assertRaises(ValueError):
            select_support_rows(10, 20, np.random.RandomState(0))

    def test_composite_is_mean_trajectory_return(self):
        rewards = np.array([[1.0, 2.0], [3.0, 0.0]])
        self.assertAlmostEqual(composite_query_objective(rewards), 3.0)


class TestProtocolLogs(unittest.TestCase):
    def test_required_fields_present(self):
        kvs = protocol_log_kvs(seed=0, k_steps=3, outer_update_count=1)
        for field in REQUIRED_LOG_FIELDS:
            self.assertIn(field, kvs)
        self.assertEqual(kvs["split_version"], split_version())
        self.assertEqual(kvs["entropy_coefficient"], 0.0)
        self.assertEqual(kvs["k_steps"], 3)
        self.assertEqual(kvs["outer_update_method"], "mrlco_first_order_mean_pseudogradient")
        self.assertEqual(kvs["hyperparameter_provenance.policy"], "fixed_literature_derived_defaults")

    def test_sliced_task_rejects_integer_leak(self):
        with self.assertRaises(ValueError):
            require_sliced_task(0)
        with self.assertRaises(ValueError):
            require_sliced_task({"dist_index": 0})
        require_sliced_task({"dist_index": 0, "graph_indices": list(range(20))})


class TestHeldOutScenarios(unittest.TestCase):
    def test_every_validation_dist_is_20_80_disjoint(self):
        for dist_id in validation_distribution_ids():
            support, query = support_query_tasks(0, dist_id)
            self.assertEqual(len(support["graph_indices"]), 20, msg="val dist %s" % dist_id)
            self.assertEqual(len(query["graph_indices"]), 80, msg="val dist %s" % dist_id)
            self.assertFalse(set(support["graph_indices"]) & set(query["graph_indices"]))

    def test_every_meta_test_dist_is_20_80_disjoint(self):
        for dist_id in meta_test_distribution_ids():
            support, query = support_query_tasks(0, dist_id)
            self.assertEqual(len(support["graph_indices"]), 20, msg="test dist %s" % dist_id)
            self.assertEqual(len(query["graph_indices"]), 80, msg="test dist %s" % dist_id)
            self.assertFalse(set(support["graph_indices"]) & set(query["graph_indices"]))

    def test_latin_grid_ids_match_policy(self):
        self.assertEqual(validation_distribution_ids(), [2, 6, 10, 16, 17])
        self.assertEqual(meta_test_distribution_ids(), [7, 12, 14, 20, 23])

    def test_zero_shot_log_fields(self):
        kvs = protocol_log_kvs(seed=7, k_steps=0, outer_update_count=0)
        self.assertEqual(kvs["k_steps"], 0)
        self.assertEqual(kvs["outer_update_count"], 0)
        self.assertEqual(kvs["query_graph_count"], 80)

    def test_query_metrics_energy_and_latency(self):
        from spec.eval_protocol import query_metrics_from_samples

        samples = {
            "rewards": np.array([[-1.0, -1.0], [-2.0, 0.0]]),
            "finish_time": np.array([4.0, 6.0]),
            "energy": np.array([[0.5, 0.5], [1.0, 1.0]]),
        }
        metrics = query_metrics_from_samples(samples)
        self.assertAlmostEqual(metrics["query_mean_return"], -2.0)
        self.assertAlmostEqual(metrics["query_mean_latency"], 5.0)
        self.assertAlmostEqual(metrics["query_mean_energy"], 1.5)
        self.assertAlmostEqual(metrics["validation_query_composite_objective"], -2.0)


class TestShuffleLeftoverAndRejects(unittest.TestCase):
    def test_two_minibatches_cover_unique_indices(self):
        rng = np.random.RandomState(1)
        seen = []
        for idx in shuffled_minibatch_slices(20, 10, rng):
            seen.extend(idx.tolist())
        self.assertEqual(sorted(seen), list(range(20)))
        self.assertEqual(len(seen), 20)

    def test_select_support_exact_n_is_identity(self):
        idx = select_support_rows(20, 20, np.random.RandomState(0))
        np.testing.assert_array_equal(idx, np.arange(20))

    def test_mean_pg_rejects_k_zero(self):
        with self.assertRaises(ValueError):
            mean_pseudogradient([np.array([1.0])], [[np.array([0.0])]], alpha=5e-4, k_steps=0)

    def test_held_out_train_prefix_rejected(self):
        with self.assertRaises(ValueError):
            assert_held_out_prefixes(meta_train_graph_prefixes()[:5], "validation")


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
        self.assertNotIn("inner_batch_size = 500", text)
        self.assertNotIn("inner_batch_size=500", text)
        self.assertIn("CUDA_VISIBLE_DEVICES", text)
        self.assertIn("sync_task_policies_from_core", text)
        self.assertIn("validation_interval", text)
        self.assertIn("held_out_evaluator", text)
        self.assertIn("protocol_log_kvs", text)
        self.assertIn("HeldOutQueryEvaluator", text)
        self.assertIn("validation_graph_prefixes", text)

    def test_sources_compile(self):
        import py_compile

        for rel in (
            "meta_trainer.py",
            "meta_evaluator.py",
            "meta_algos/ppo_offloading.py",
            "meta_algos/MRLCO.py",
            "spec/split_loader.py",
            "spec/eval_protocol.py",
            "spec/learning_ops.py",
        ):
            py_compile.compile(str(ROOT / rel), doraise=True)

    def test_evaluator_no_query_leak(self):
        text = (ROOT / "meta_evaluator.py").read_text()
        self.assertNotIn("set_task(0)", text)
        self.assertIn("greedy_solution_for_current_task", text)
        self.assertIn("k_steps=0", text)
        self.assertIn("restore_trainable", text)
        self.assertIn("require_sliced_task", text)
        self.assertNotIn("repilte", text)

    def test_ppo_matches_inner_contract(self):
        text = (ROOT / "meta_algos" / "ppo_offloading.py").read_text()
        self.assertNotIn("mpi4py", text)
        self.assertNotIn("MpiAdamOptimizer", text)
        self.assertNotIn("lr=1e-4", text)
        self.assertNotIn("epsilon=1e-5", text)
        self.assertNotIn("num_inner_grad_steps=4", text)
        self.assertNotIn("batch_size=50", text)
        self.assertIn("lr=5e-4", text)
        self.assertIn("def reset_inner_optimizer", text)
        self.assertIn("shuffled_minibatch_slices", text)
        self.assertIn("old_v + tf.clip_by_value", text)


if __name__ == "__main__":
    unittest.main()
