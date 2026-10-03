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
                               slots_per_task=2, base_seed=303, single_dist=True)

    def test_v1_attributes_are_delegated(self):
        from spec.automotive_training.automotive_env import AutomotiveEnv

        env = self._env()
        self.assertIsInstance(env.base, AutomotiveEnv)
        for attr in ("configs", "graph_objects", "orders", "dags", "graph_indices",
                     "encoder_batchs"):
            self.assertTrue(hasattr(env, attr), attr)
        self.assertEqual(env.input_dim, 91)   # the v2 env always emits the v2 schema
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
                                  base_seed=int(seed), single_dist=bool(single_dist),
                                  link_regime="degraded")
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


class TestMCSinkSemantics(unittest.TestCase):
    """A task whose successors were dropped by the MC realization is an endpoint: its output
    must be returned to the vehicle, exactly as the v1 engine does."""

    def test_sink_set_is_mc_aware(self):
        import numpy as np
        from spec.automotive_training.automotive_env import AutomotiveEnv
        from spec.automotive_training.automotive_loader import load_dataset
        from spec.automotive_training.automotive_primary import AutomotiveResourceCluster
        from spec.automotive_training.v2.adapters import (
            compute_spec, dag_spec_from_graph, link_spec, plan_map_from_actions,
        )
        from spec.automotive_training.v2.helper_model import HelperState
        from spec.automotive_training.v2.shared_scheduler import schedule_shared

        graph = load_dataset().validation_query()[0]
        env = AutomotiveEnv([graph], AutomotiveResourceCluster(), role="validation",
                            slots_per_task=1, base_seed=303, single_dist=True)
        env.set_task({"dist_index": 0, "graph_indices": np.arange(1, dtype=np.int32)})
        env.reset()
        mc = env._slot_mc[0]
        v1, _e, _m = env._schedule(0, [1] * 20, mc)
        v1_bytes = sum(float(t.bytes) for t in (getattr(v1, "transfers", []) or []))
        dag = dag_spec_from_graph(graph, dag_id="g", owner=0, mc=mc, helper_id=0)
        compute = compute_spec([graph])
        hs = {0: HelperState(0, compute.helper_cpu_bytes_per_s[0],
                             contact_end_s=1e18, predicted_contact_end_s=1e18)}
        res = schedule_shared([dag], {"g": plan_map_from_actions(graph, [1] * 20)},
                              link=link_spec(graph), compute=compute, helper_states=hs)
        self.assertAlmostEqual(float(res.mechanics["radio_bytes"]), v1_bytes, places=6)
        self.assertEqual(res.mechanics["radio_events"],
                         len(getattr(v1, "transfers", []) or []))
        # at least one sink must be a task that is NOT a static sink in the full DAG
        self.assertGreater(len([t for t in dag.tasks if t.is_sink]), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
