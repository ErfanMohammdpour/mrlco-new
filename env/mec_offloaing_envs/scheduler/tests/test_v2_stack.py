#!/usr/bin/env python3
"""v2 stack wiring tests (no TF): routing, guards, delegating env surface."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spec.automotive_training.v2.env import V2AutomotiveEnv  # noqa: E402
from spec.automotive_training.v2.observation import V2_OBS_VERSION  # noqa: E402
from spec.automotive_training.v2.stack import (  # noqa: E402
    V2_ALLOWED_LINK_REGIMES, build_automotive_v2_stack,
)


class TestRouting(unittest.TestCase):
    def test_meta_trainer_routes_automotive_mc_v2(self):
        src = (ROOT / "meta_trainer.py").read_text()
        self.assertIn('str(dataset) == "automotive_mc_v2"', src)
        self.assertIn("build_automotive_v2_stack", src)

    def test_builder_guards_run_before_any_tf_import(self):
        with self.assertRaises(Exception):
            build_automotive_v2_stack(seed=0, n_itr=1, ckpt_dir="/tmp/x",
                                      link_regime="nonsense")
        with self.assertRaises(Exception):
            build_automotive_v2_stack(seed=0, n_itr=1, ckpt_dir="/tmp/x", mec_workers=0)
        with self.assertRaises(Exception):
            build_automotive_v2_stack(seed=0, n_itr=1, ckpt_dir="/tmp/x", meta_batch_size=4)

    def test_regime_list_and_obs_version(self):
        self.assertEqual(V2_ALLOWED_LINK_REGIMES, ("stable", "moderate", "degraded"))
        self.assertEqual(V2_OBS_VERSION, "automotive_v2_obs_v1")


class TestDelegation(unittest.TestCase):
    def _env(self):
        from spec.automotive_training.automotive_loader import load_dataset
        from spec.automotive_training.automotive_primary import AutomotiveResourceCluster

        graphs = load_dataset().validation_query()[:2]
        return V2AutomotiveEnv(graphs, AutomotiveResourceCluster(), role="validation",
                               slots_per_task=2, base_seed=303)

    def test_v1_attributes_are_delegated(self):
        from spec.automotive_training.automotive_env import AutomotiveEnv

        env = self._env()
        self.assertIsInstance(env.base, AutomotiveEnv)
        for attr in ("configs", "graph_objects", "orders", "dags", "graph_indices",
                     "encoder_batchs"):
            self.assertTrue(hasattr(env, attr), attr)
        self.assertEqual(env.input_dim, 79)
        self.assertEqual(len(env.configs), 2)

    def test_unknown_attribute_still_raises(self):
        env = self._env()
        with self.assertRaises(AttributeError):
            _ = env.definitely_not_a_real_attribute
        with self.assertRaises(AttributeError):
            _ = env.__deepcopy__


class TestEvaluatorEnvFactory(unittest.TestCase):
    """The held-out evaluator must build v2 envs when a factory is injected - otherwise
    validation would measure v1 dynamics under a v2 label."""

    def _evaluator(self, factory):
        from spec.automotive_training.automotive_loader import load_dataset
        from spec.automotive_training.automotive_primary import AutomotiveHeldOutEvaluator

        val = load_dataset().validation_query()[:4]
        return AutomotiveHeldOutEvaluator(support_graphs=val[:2], query_graphs=val,
                                         policy=None, source_policy=None,
                                         env_factory=factory), val

    def test_factory_is_honoured_for_both_layouts(self):
        from spec.automotive_training.automotive_primary import AutomotiveResourceCluster

        calls = []

        def factory(graphs, slots, seed, single_dist):
            calls.append((len(graphs), int(slots), int(seed), bool(single_dist)))
            env = V2AutomotiveEnv(list(graphs), AutomotiveResourceCluster(),
                                  role="validation", slots_per_task=int(slots),
                                  base_seed=int(seed), link_regime="degraded")
            if single_dist:
                env.set_task({"dist_index": 0,
                              "graph_indices": np.arange(len(graphs), dtype=np.int32)})
            return env

        evaluator, val = self._evaluator(factory)
        paired = evaluator._make_env(val, 4, 7, single_dist=True)
        self.assertEqual(calls, [(4, 4, 7, True)])
        self.assertIsInstance(paired, V2AutomotiveEnv)
        self.assertTrue(paired.single_dist)
        self.assertIsNotNone(paired.graph_indices)

    def test_default_path_still_builds_the_frozen_v1_env(self):
        from spec.automotive_training.automotive_env import AutomotiveEnv

        evaluator, val = self._evaluator(None)
        env = evaluator._env(val[:2], 2, 303)
        self.assertIsInstance(env, AutomotiveEnv)
        self.assertNotIsInstance(env, V2AutomotiveEnv)
        self.assertEqual(env.input_dim, 79)

    def test_v2_stack_wires_the_factory(self):
        src = (ROOT / "spec/automotive_training/v2/stack.py").read_text()
        self.assertIn("env_factory=v2_env_factory", src)
        self.assertIn("def v2_env_factory(", src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
