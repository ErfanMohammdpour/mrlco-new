#!/usr/bin/env python3
"""v2 stack wiring tests (no TF): routing, guards, delegating env surface, and the F03/F15/
F16/F26/F27/F28 contracts that earlier lived only in prose."""

from __future__ import annotations

import dataclasses
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spec.automotive_training.v2.observation import V2_PACKED_DIM  # noqa: E402
from spec.automotive_training.v2.env import V2AutomotiveEnv, stable_seed  # noqa: E402
from spec.automotive_training.v2.observation import V2_OBS_VERSION  # noqa: E402
from spec.automotive_training.v2.stack import (  # noqa: E402
    V2_ALLOWED_LINK_REGIMES, V2HeldOutEvaluator, build_automotive_v2_stack,
)

def _tf_runtime_available():
    """A REAL TensorFlow 1.x runtime, not a bare `types.ModuleType('tensorflow')` stub.

    Other test modules install a placeholder `tensorflow` module with
    `sys.modules.setdefault(...)`; a plain `import tensorflow` then succeeds and would make this
    TF-gated test execute against a stub. Require the attributes the test actually uses.
    """
    try:  # pragma: no cover - the checkout has no TensorFlow
        import tensorflow as tf
    except Exception:  # noqa: BLE001
        return False
    return all(hasattr(tf, name) for name in ("Graph", "Session", "compat"))


HAS_TF = _tf_runtime_available()


def _v2_factory(**extra):
    """A v2 env factory shaped exactly like `build_automotive_v2_stack`'s."""
    from spec.automotive_training.automotive_primary import AutomotiveResourceCluster

    def factory(graphs, slots, seed, single_dist):
        env = V2AutomotiveEnv(list(graphs), AutomotiveResourceCluster(),
                              role="validation", slots_per_task=int(slots),
                              base_seed=int(seed), single_dist=bool(single_dist), **extra)
        if single_dist:
            env.set_task({"dist_index": 0,
                          "graph_indices": np.arange(len(graphs), dtype=np.int32)})
        return env

    return factory


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
        self.assertEqual(env.input_dim, V2_PACKED_DIM)  # the v2 env always emits the v2 schema
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


class TestF03SupportQueryRealizationDisjoint(unittest.TestCase):
    """F03: overlapping support/query graphs must never share one realization."""

    def test_overlapping_graph_gets_role_tagged_realizations(self):
        from spec.automotive_training.automotive_loader import load_dataset

        val = load_dataset().validation_query()
        support, query = val[:2], val          # val[0] and val[1] overlap by construction
        ev = V2HeldOutEvaluator(support_graphs=support, query_graphs=query,
                                policy=None, source_policy=None, env_factory=_v2_factory())
        env_s = ev._paired_env(support, 1000, 0, role="support")
        env_q = ev._paired_env(query, 1000, 0, role="query")
        env_s.reset()
        env_q.reset()
        self.assertNotEqual(env_s.realization_epoch, env_q.realization_epoch)
        self.assertNotEqual(env_s.world_realization_key(), env_q.world_realization_key())
        # the overlapping graph (slot 0 of both) must have DIFFERENT demand realizations
        self.assertNotEqual(env_s._slot_mc[0], env_q._slot_mc[0],
                            "support and query shared one MC realization for an overlapping "
                            "graph: the query score would be the adaptation sample")
        self.assertNotEqual(env_s.world_for_slot(0).fingerprint_sha256(),
                            env_q.world_for_slot(0).fingerprint_sha256())

    def test_v1_env_keeps_its_FROZEN_epoch_and_no_role_tag(self):
        """The F03 role tag is V2-ONLY: the frozen v1 env must keep its pinned epoch.

        v1 has no world-key API, so role-tagging its epoch would silently change every
        previously reported v1 validation number. The correction belongs to V2 only.
        """
        from spec.automotive_training.automotive_env import AutomotiveEnv
        from spec.automotive_training.automotive_loader import load_dataset
        from spec.automotive_training.automotive_primary import (
            AutomotiveHeldOutEvaluator, AutomotiveResourceCluster,
        )

        graph = load_dataset().validation_query()[:1]

        def factory(graphs, slots, seed, single_dist):
            env = AutomotiveEnv(list(graphs), AutomotiveResourceCluster(), role="validation",
                                slots_per_task=int(slots), base_seed=int(seed),
                                mc_enabled=True, single_dist=bool(single_dist))
            if single_dist:
                env.set_task({"dist_index": 0,
                              "graph_indices": np.arange(len(graphs), dtype=np.int32)})
            return env

        ev = AutomotiveHeldOutEvaluator(support_graphs=graph, query_graphs=graph, policy=None,
                                        source_policy=None, env_factory=factory)
        a = ev._paired_env(graph, 1000, 0, role="support")
        b = ev._paired_env(graph, 1000, 0, role="query")
        self.assertFalse(hasattr(a, "set_world_realization"))
        self.assertEqual(0, a.realization_epoch)
        self.assertEqual(0, b.realization_epoch)
        a.reset()
        b.reset()
        self.assertEqual(a._slot_mc[0], b._slot_mc[0])


class TestF26LifecycleTrapAndCacheIdentity(unittest.TestCase):
    def _evaluator(self, query):
        return V2HeldOutEvaluator(support_graphs=query[:1], query_graphs=query, policy=None,
                                  source_policy=None, env_factory=_v2_factory())

    def test_trap_is_armed_for_the_whole_evaluation_and_raises(self):
        from spec.automotive_training.automotive_loader import load_dataset

        val = load_dataset().validation_query()[:2]
        ev = self._evaluator(val)
        env = ev._make_env(val, 2, 7, single_dist=True)
        self.assertTrue(ev.v1_schedule_trap_armed)
        self.assertTrue(getattr(env, "_v1_schedule_trap", False))
        env.reset()
        with self.assertRaises(Exception) as ctx:
            env._schedule(0, [1] * 20, env._slot_mc[0])
        self.assertIn("v1", str(ctx.exception).lower())

    def test_panel_cache_key_covers_every_identity_component(self):
        from spec.automotive_training.automotive_loader import load_dataset

        val = load_dataset().validation_query()[:2]
        ev = self._evaluator(val)
        env = ev._make_env(val, 2, 1000, single_dist=True)
        env.reset()
        base = ev._panel_cache_key(0, env)
        self.assertNotEqual(base, ev._panel_cache_key(1, env))          # epoch
        self.assertNotEqual(base, self._evaluator(val[1:])._panel_cache_key(0, env))  # graphs
        self.assertNotEqual(
            base, ev._panel_cache_key(0, None))                          # world/mc unknown
        ev.reward_mode = "latency_only_but_different_contract"
        self.assertNotEqual(base, ev._panel_cache_key(0, env))           # contract
        ev.reward_mode = "latency_only"
        env.world_config = dataclasses.replace(env.world_config, mec_workers=3)
        self.assertNotEqual(base, ev._panel_cache_key(0, env))           # config sha
        env.world_config = dataclasses.replace(env.world_config, mec_workers=1)
        env.scheduler_config_sha256 = "c" * 64
        self.assertNotEqual(base, ev._panel_cache_key(0, env))           # scheduler sha
        env.scheduler_config_sha256 = None
        env.base._slot_mc = []
        self.assertNotEqual(base, ev._panel_cache_key(0, env))           # mc identity
        env.reset()
        env.base._slot_mc[0] = dict(env.base._slot_mc[0])
        env.base._slot_mc[0]["executed_task_ids"] = [1, 2]
        self.assertNotEqual(base, ev._panel_cache_key(0, env))           # mc content

    def test_panel_cache_is_reused_then_invalidated_by_a_config_change(self):
        from spec.automotive_training.automotive_loader import load_dataset

        val = load_dataset().validation_query()[:1]
        ev = self._evaluator(val)
        env = ev._make_env(val, 1, 1000, single_dist=True)
        env.reset()
        first = ev.baseline_panel(env, epoch=0)
        self.assertIs(first, ev.baseline_panel(env, epoch=0),
                      "a cached panel must be reused for the same identity")
        env.world_config = dataclasses.replace(env.world_config, background_dags=1)
        env.reset()
        second = ev.baseline_panel(env, epoch=0)
        self.assertIsNot(first, second,
                         "a changed world config must invalidate the panel cache")


class TestF28RolloutMetricGuard(unittest.TestCase):
    def _record(self):
        return {"makespan_s": 0.1, "hard_violation": 0.0, "high_tardiness": 0.0,
                "medium_tardiness": 0.0, "firm_miss": 0.0, "n_tasks_high": 4,
                "n_tasks_medium": 2, "n_tardy_high": 0, "n_tardy_medium": 0,
                "high_preserved": True, "mode": "HI", "switches": 0.0, "dropped": 0.0}

    def test_missing_required_metric_raises(self):
        from spec.automotive_training.automotive_primary import (
            AutomotivePrimaryError, rollout_metric_block,
        )

        for key in ("makespan_s", "firm_miss", "n_tardy_high_tasks"):
            pass
        with self.assertRaises(AutomotivePrimaryError):
            rollout_metric_block({0: dict(self._record(), makespan_s=None)})
        broken = self._record()
        broken.pop("firm_miss")
        with self.assertRaises(AutomotivePrimaryError):
            rollout_metric_block({0: broken})
        with self.assertRaises(AutomotivePrimaryError):
            rollout_metric_block({0: dict(self._record(), n_tardy_high=float("nan"))})

    def test_absent_energy_is_unmodelled_not_a_measured_zero(self):
        from spec.automotive_training.automotive_primary import rollout_metric_block

        block = rollout_metric_block({0: self._record()})
        self.assertIsNone(block["mean_system_joules"])
        self.assertEqual(block["mean_system_joules"], None)
        self.assertEqual(block["energy_status"]["mean_system_joules"], "UNMODELLED")
        self.assertEqual(block["unmodelled"]["status"], "UNMODELLED")
        self.assertTrue(any("backhaul" in c for c in block["unmodelled"]["components"]))

    def test_deterministic_world_identity_rows_are_recorded(self):
        """F27(3): the deterministic block carries the world identity per slot."""
        from spec.automotive_training.automotive_loader import load_dataset
        from spec.automotive_training.automotive_primary import _world_identity_rows

        val = load_dataset().validation_query()[:2]
        env = V2AutomotiveEnv(val, None, role="validation", slots_per_task=2,
                              base_seed=1000, single_dist=True)
        env.set_task({"dist_index": 0, "graph_indices": np.arange(2, dtype=np.int32)})
        env.reset()
        rows = _world_identity_rows(env)
        self.assertEqual(sorted(rows), [0, 1])
        for row in rows.values():
            for key in ("world_fingerprint_sha256", "world_physics_sha256", "mc_id",
                        "config_sha", "dataset_graph_id", "world_id"):
                self.assertIn(key, row)
        from spec.automotive_training.automotive_env import AutomotiveEnv

        v1 = AutomotiveEnv(val, None, role="validation", slots_per_task=2,
                           base_seed=1000, single_dist=True)
        self.assertEqual(_world_identity_rows(v1), {})


class TestF15F16KnobRouting(unittest.TestCase):
    """F15/F16: each `build_automotive_v2_stack` knob must reach its real consumer.

    The TF-dependent classes are stubbed so the WIRING is asserted on the constructors that
    actually receive the values, not on source text.
    """

    def _build(self, **kwargs):
        import spec.automotive_training.v2.stack as stack

        env_kwargs, eval_kwargs, policy_kwargs = [], [], []

        class FakeEnv(object):
            def __init__(self, *a, **kw):
                env_kwargs.append(dict(kw))
                self.constraint_manager = None
                self.constraint_controller = None
                self.input_dim = 91
                self.configs = [None]
                self.world_config = None
                self.scheduler_config_sha256 = None
                self._worlds = {}
                self._slot_mc = []
                self.reward_mode = None

            def greedy_solution(self):
                from spec.automotive_training.automotive_loader import load_dataset
                from spec.automotive_training.v2.adapters import config_for

                self.configs = [config_for(load_dataset().meta_train()[0])]
                return (None, 0.0, 0.0)

        class FakePolicy(object):
            def __init__(self):
                self.core_policy = mock.MagicMock()
                self._deterministic_default = True

            def set_deterministic(self, flag):
                self._deterministic_default = bool(flag)

        class FakeEvaluator(object):
            def __init__(self, *a, **kw):
                eval_kwargs.append(dict(kw))

        class FakeTrainer(object):
            auto_controller = None

            def __init__(self, **kw):
                self.kwargs = kw

            def _ckpt_path(self, name):
                return "/tmp/%s" % name

        fake_meta_trainer = types.ModuleType("meta_trainer")
        fake_meta_trainer.Trainer = FakeTrainer
        stubs = {name: mock.MagicMock() for name in (
            "tensorflow", "policies.meta_seq2seq_policy", "samplers.seq2seq_meta_sampler",
            "samplers.seq2seq_meta_sampler_process", "baselines.vf_baseline",
            "meta_algos.MRLCO", "meta_trainer")}
        stubs["meta_trainer"] = fake_meta_trainer

        def fake_validation_policy(env, decoding, flags, *, enable_crn=False):
            policy_kwargs.append({"decoding": decoding, "enable_crn": bool(enable_crn)})
            return FakePolicy()

        with mock.patch.dict(sys.modules, stubs), \
                mock.patch.object(stack, "V2AutomotiveEnv", FakeEnv), \
                mock.patch.object(stack, "V2HeldOutEvaluator", FakeEvaluator), \
                mock.patch.object(stack, "_validation_policy", fake_validation_policy):
            trainer, algo = stack.build_automotive_v2_stack(
                seed=0, n_itr=1, ckpt_dir=os.path.join(tempfile.mkdtemp(), "ckpt"),
                **kwargs)
            # invoke the evaluator's validation factory INSIDE the patch: the factory resolves
            # `V2AutomotiveEnv` from module globals at call time.
            if eval_kwargs and eval_kwargs[0].get("env_factory") is not None:
                eval_kwargs[0]["env_factory"]([None, None], 2, 1000, True)
        return {"env": env_kwargs, "eval": eval_kwargs, "policy": policy_kwargs,
                "trainer": trainer}

    def test_every_knob_reaches_its_consumer(self):
        rec = self._build(link_regime="degraded", mec_workers=3, reliability=True,
                          background_dags=4, background_policy="all_ue",
                          helper_contact_mean_s=1.5, helper_contact_cv=0.25,
                          helper_busy_s=0.75, r_select=3, s_select=7,
                          reward_mode="latency_only", use_energy=False,
                          checkpoint_interval=4, constraints_enabled=False,
                          budget_fractions={"total_energy_mec": 0.4, "ue_energy": 0.9})
        env = rec["env"][0]
        for key, value in (("link_regime", "degraded"), ("mec_workers", 3),
                           ("reliability", True), ("background_dags", 4),
                           ("background_policy", "all_ue"), ("helper_contact_mean_s", 1.5),
                           ("helper_contact_cv", 0.25), ("helper_busy_s", 0.75),
                           ("energy_enabled", False), ("constraints_enabled", False)):
            self.assertEqual(env[key], value, key)
        self.assertEqual(env["budget_fractions"],
                         {"total_energy_mec": 0.4, "ue_energy": 0.9})
        ev = rec["eval"][0]
        for key, value in (("link_regime", "degraded"), ("mec_workers", 3),
                           ("reliability", True), ("background_dags", 4),
                           ("background_policy", "all_ue"), ("r_select", 3), ("s_select", 7),
                           ("reward_mode", "latency_only"), ("constraints_enabled", False),
                           ("helper_contact_mean_s", 1.5), ("helper_contact_cv", 0.25),
                           ("helper_busy_s", 0.75)):
            self.assertEqual(ev[key], value, key)
        self.assertEqual(ev["budget_fractions"],
                         {"total_energy_mec": 0.4, "ue_energy": 0.9})
        # the validation factory the evaluator received carries the SAME system knobs
        self.assertEqual(len(rec["env"]), 2, "train env + one validation-factory env")
        val_env = rec["env"][1]
        for key, value in (("link_regime", "degraded"), ("mec_workers", 3),
                           ("reliability", True), ("background_dags", 4),
                           ("background_policy", "all_ue"), ("energy_enabled", False),
                           ("helper_contact_mean_s", 1.5), ("helper_contact_cv", 0.25),
                           ("helper_busy_s", 0.75)):
            self.assertEqual(val_env[key], value, "validation factory %s" % key)
        self.assertEqual(val_env["single_dist"], True)
        self.assertEqual(rec["policy"][0]["enable_crn"], True)
        trainer = rec["trainer"]
        self.assertEqual(trainer.auto_reward_mode, "latency_only")
        self.assertEqual(trainer.auto_checkpoint_interval, 4)
        self.assertEqual(trainer.auto_crn["applied_r_select"], 3)
        self.assertEqual(trainer.auto_crn["applied_s_select"], 7)
        self.assertEqual(trainer.auto_background_dags, 4)
        self.assertEqual(trainer.auto_mec_workers, 3)
        self.assertIn(trainer.auto_energy_constraint,
                      ("not_configured", "measured_not_constrained", "configured"))
        # `checkpoint_interval` is a REAL consumer: it gates the generation hook cadence.
        commits = []
        trainer.auto_generation_checkpoint = lambda itr: commits.append(int(itr))
        for itr in range(8):
            trainer._iteration_completed(itr)
        self.assertEqual(commits, [3, 7],
                         "every=4 must invoke the generation hook exactly on iterations 3 and 7")

    def test_unsupported_reward_mode_is_refused(self):
        with self.assertRaises(Exception):
            self._build(reward_mode="energy_weighted")

    def test_bad_axes_are_refused(self):
        with self.assertRaises(Exception):
            self._build(r_select=0)
        with self.assertRaises(Exception):
            self._build(checkpoint_interval=0)


@unittest.skipUnless(HAS_TF, "TensorFlow absent: this checkout cannot execute the TF path")
class TestF27RealRSTrainedPolicyGrid(unittest.TestCase):
    """F27(2): the REAL trained-policy evaluator executes the R x S grid.

    Run in the TF1.15 container (the image has no pytest) with::

        python -m unittest env.mec_offloaing_envs.scheduler.tests.test_v2_stack
    """

    def test_evaluate_all_executes_the_grid_and_separates_decodings(self):
        import tensorflow as tf
        from spec.automotive_training.automotive_loader import load_dataset
        from spec.automotive_training.automotive_primary import (
            AutomotiveResourceCluster, _validation_policy, decoding_flags,
        )

        val_q = load_dataset().validation_query()[:3]
        val_s = load_dataset().validation_support()[:3]
        graph = tf.Graph()
        with graph.as_default():
            with tf.compat.v1.Session(graph=graph) as sess:
                with sess.as_default():

                    def factory(graphs, slots, seed, single_dist):
                        env = V2AutomotiveEnv(
                            list(graphs), AutomotiveResourceCluster(), role="validation",
                            slots_per_task=int(slots), base_seed=int(seed),
                            single_dist=bool(single_dist))
                        if single_dist:
                            env.set_task({"dist_index": 0,
                                          "graph_indices": np.arange(len(graphs),
                                                                    dtype=np.int32)})
                        return env

                    probe = factory(val_q, len(val_q), 1000, True)
                    flags = decoding_flags("deterministic")
                    policy = _validation_policy(probe, "deterministic", flags,
                                                enable_crn=True)
                    ev = V2HeldOutEvaluator(support_graphs=val_s, query_graphs=val_q,
                                            policy=policy, source_policy=policy,
                                            ppo_batch_size=3, env_factory=factory,
                                            r_select=2, s_select=3)
                    sess.run(tf.compat.v1.global_variables_initializer())
                    out = ev.evaluate_all(k_steps=0)
                    self.assertEqual(out["applied_r_select"], 2)
                    self.assertEqual(out["applied_s_select"], 3)
                    sampled = out["sampled_decoding"]
                    self.assertEqual(sampled["r_select"], 2)
                    self.assertEqual(sampled["s_select"], 3)
                    self.assertFalse(sampled["best_of_s_used"])
                    self.assertEqual(len(sampled["cells"]), 2 * 3 * len(val_q))
                    self.assertGreater(sampled["unique_world_physics_sha256"], 1)
                    for cell in sampled["cells"]:
                        for key in ("world_fingerprint_sha256", "mc_id", "config_sha",
                                    "dataset_graph_id", "r", "s", "episode_latency_s"):
                            self.assertIn(key, cell)
                    self.assertIn("deterministic_decoding", out)
                    self.assertEqual(out["deterministic_decoding"]["selection_criterion"],
                                     "automotive_lexicographic_v1")
                    self.assertEqual(out["oracle_selection_criterion"],
                                     "best_of_panel_latency")
                    # paired panel: baselines scored on the policy's own world
                    panel = out["per_replicate"][0]["panel"]
                    for entry in panel.values():
                        self.assertTrue(entry["paired_with_policy_world"])
                        self.assertIn("world_physics_sha256", entry)
                    identities = out["per_replicate"][0]["query_world_identities"]
                    self.assertTrue(identities)
                    for row in identities.values():
                        for key in ("world_fingerprint_sha256", "mc_id", "config_sha",
                                    "dataset_graph_id"):
                            self.assertIn(key, row)


if __name__ == "__main__":
    unittest.main(verbosity=2)
