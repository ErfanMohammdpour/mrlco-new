#!/usr/bin/env python3
"""F33: the resume bundle must carry the ENVIRONMENT IDENTITY, not only the counters.

The v1 bundle persisted the iteration ledger, duals, sampler counters and fingerprints, but not
the environment episode/reset counters, the world key, the realization epoch, or the RNG streams.
A resumed process therefore rebuilt the SAME iteration NUMBER over a DIFFERENT world sequence
(measured: control `ep11_slot19` vs resume `ep3_slot19`). These tests pin the bundle contract.
"""
import json
import sys
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spec.automotive_training.v2 import run_tf  # noqa: E402


class _FakeRng:
    def __init__(self, seed=3):
        self._r = np.random.RandomState(seed)

    def get_state(self):
        return self._r.get_state()

    def set_state(self, s):
        self._r.set_state(s)


class _FakeBase:
    def __init__(self):
        self.reset_count = 7
        self.realization_epoch = None


class _FakeEnv:
    def __init__(self):
        self.episodes = 11
        self._world_key = None
        self.base_seed = 0
        self.slots_per_task = 20
        self.link_regime = "moderate"
        self.background_dags = 2
        self.base = _FakeBase()

    def set_world_key(self, key):
        self._world_key = key

    def set_realization_epoch(self, epoch):
        self.base.realization_epoch = epoch


class _FakeTrainer:
    def __init__(self, env, clones):
        self.env = env
        self.sampler = type("S", (), {"vec_env": type("V", (), {"envs": clones})()})()
        self.algo = type("A", (), {"rng": _FakeRng()})()


class ResumeIdentityTests(unittest.TestCase):
    def test_bundle_is_v2_and_carries_environment_and_rng(self):
        env = _FakeEnv()
        clones = [_FakeEnv() for _ in range(3)]
        for i, c in enumerate(clones):
            c.episodes = 5 + i
        tr = _FakeTrainer(env, clones)
        doc = {"environment": run_tf._env_identity(env, tr),
               "rng": run_tf._rng_state(tr)}
        self.assertEqual(run_tf.RESUME_SCHEMA, "v2_resume_state_v2")
        self.assertEqual(doc["environment"]["episodes"], 11)
        self.assertEqual(doc["environment"]["clone_episodes"], [5, 6, 7])
        self.assertIn("algo", doc["rng"])
        self.assertIn("numpy_global", doc["rng"])

    def test_env_identity_round_trips_exactly(self):
        env = _FakeEnv()
        clones = [_FakeEnv() for _ in range(3)]
        for i, c in enumerate(clones):
            c.episodes = 5 + i
            c.base.reset_count = 20 + i
        tr = _FakeTrainer(env, clones)
        doc = run_tf._env_identity(env, tr)
        # a FRESH process: counters start at zero, then the bundle is applied
        fresh = _FakeEnv()
        fresh.episodes = 0
        fresh.base.reset_count = 0
        fresh_clones = [_FakeEnv() for _ in range(3)]
        for c in fresh_clones:
            c.episodes = 0
            c.base.reset_count = 0
        ntr = _FakeTrainer(fresh, fresh_clones)
        run_tf._restore_env_identity(ntr, doc)
        self.assertEqual(fresh.episodes, 11)
        self.assertEqual(fresh.base.reset_count, 7)
        self.assertEqual([c.episodes for c in fresh_clones], [5, 6, 7])
        self.assertEqual([c.base.reset_count for c in fresh_clones], [20, 21, 22])

    def test_clone_count_mismatch_is_refused(self):
        env = _FakeEnv()
        tr = _FakeTrainer(env, [_FakeEnv() for _ in range(3)])
        doc = run_tf._env_identity(env, tr)
        other = _FakeTrainer(_FakeEnv(), [_FakeEnv() for _ in range(2)])
        with self.assertRaises(run_tf.V2RunError):
            run_tf._restore_env_identity(other, doc)

    def test_rng_state_round_trips_and_advances(self):
        tr = _FakeTrainer(_FakeEnv(), [])
        doc = run_tf._rng_state(tr)
        tr.algo.rng._r = np.random.RandomState(3)
        drawn_before = tr.algo.rng._r.randint(0, 2 ** 31)
        # a fresh stream that consumes the SAME state must produce the SAME next draw
        fresh = _FakeTrainer(_FakeEnv(), [])
        run_tf._restore_rng_state(fresh, doc)
        self.assertEqual(fresh.algo.rng._r.randint(0, 2 ** 31), drawn_before)

    def test_legacy_v1_bundle_is_refused_with_an_explanation(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            ckpt = Path(d) / "v2_run.ckpt"
            (Path(d) / "v2_resume_state.json").write_text(
                json.dumps({"schema": run_tf.RESUME_SCHEMA_V1}))
            args = type("A", (), {"seed": 0, "background": 2, "link_regime": "moderate",
                                  "background_policy": "all_mec", "mec_workers": 1,
                                  "ckpt_dir": d})()
            tr = _FakeTrainer(_FakeEnv(), [])
            with self.assertRaises(run_tf.V2RunError) as ctx:
                run_tf._restore_resume_state(tr, Path(d) / "v2_resume_state.json", args)
            self.assertIn("LEGACY", str(ctx.exception))
            self.assertIn("warm start", str(ctx.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)


class RngAndCounterRestoreTests(unittest.TestCase):
    """F33: the boundary must carry the stdlib random state and the sampler counters."""

    def test_python_random_state_is_captured_and_restored(self):
        import random as _random
        tr = _FakeTrainer(_FakeEnv(), [])
        _random.seed(1234)
        doc = run_tf._rng_state(tr)
        self.assertIn("python_random", doc)
        expected = _random.random()
        _random.seed(999)                      # a fresh process reseeds during stack build
        run_tf._restore_rng_state(tr, doc)
        self.assertEqual(_random.random(), expected,
                         "the stdlib random stream was not restored")

    def test_sampler_counters_are_applied(self):
        class _S:
            def __init__(self):
                self.counters = {"support_calls": 0, "query_calls": 0,
                                 "unique_rollout_seeds": 0}
        tr = _FakeTrainer(_FakeEnv(), [])
        tr.sampler = _S()
        run_tf._restore_sampler_counters(tr, {"support_calls": 2, "query_calls": 2,
                                              "unique_rollout_seeds": 1})
        self.assertEqual(tr.sampler.counters["support_calls"], 2)
        self.assertEqual(tr.sampler.counters["query_calls"], 2)
        self.assertEqual(tr.sampler.counters["unique_rollout_seeds"], 1)

    def test_per_clone_task_identity_round_trips(self):
        env = _FakeEnv()
        clones = [_FakeEnv() for _ in range(3)]
        for i, c in enumerate(clones):
            c.episodes = 4 + i
            c.base.task_id = 100 + i
            c.base.graph_indices = np.zeros(5, dtype=np.int32)
        tr = _FakeTrainer(env, clones)
        doc = run_tf._env_identity(env, tr)
        self.assertEqual(doc["clone_task_ids"], [100, 101, 102])
        fresh = _FakeEnv()
        fclones = [_FakeEnv() for _ in range(3)]
        for c in fclones:
            c.base.task_id = -1
        ntr = _FakeTrainer(fresh, fclones)
        run_tf._restore_env_identity(ntr, doc)
        self.assertEqual([c.base.task_id for c in fclones], [100, 101, 102])
        self.assertTrue(all(len(c.base.graph_indices) == 5 for c in fclones))
