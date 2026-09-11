#!/usr/bin/env python3
"""Serial vs spawn-parallel executor equality. No GPU. No TensorFlow."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from samplers.dummy_step_env import DummyStepEnv
from samplers.vectorized_env_executor import MetaIterativeEnvExecutor, MetaParallelEnvExecutor


def _tasks():
    return [
        {"dist_index": 3, "graph_indices": np.asarray([0, 1], dtype=np.int32)},
        {"dist_index": 7, "graph_indices": np.asarray([0, 1], dtype=np.int32)},
        {"dist_index": 11, "graph_indices": np.asarray([0, 1], dtype=np.int32)},
    ]


def _actions():
    return [
        np.asarray([[0, 1], [2, 0]], dtype=np.int32),
        np.asarray([[1, 1], [1, 2]], dtype=np.int32),
        np.asarray([[2, 2], [0, 0]], dtype=np.int32),
    ]


class TestParallelEnvEquality(unittest.TestCase):
    def test_step_matches_iterative(self):
        tasks = _tasks()
        actions = _actions()
        serial = MetaIterativeEnvExecutor(DummyStepEnv(), 3, 1, 20000)
        parallel = MetaParallelEnvExecutor(DummyStepEnv(), 3, 1, 20000)
        try:
            serial.set_tasks(tasks)
            parallel.set_tasks(tasks)
            s_obs = serial.reset()
            p_obs = parallel.reset()
            np.testing.assert_array_equal(np.asarray(s_obs), np.asarray(p_obs))
            s_obs, s_rew, s_done, s_info = serial.step(actions)
            p_obs, p_rew, p_done, p_info = parallel.step(actions)
            np.testing.assert_array_equal(np.asarray(s_rew), np.asarray(p_rew))
            np.testing.assert_array_equal(np.asarray(s_done), np.asarray(p_done))
            np.testing.assert_array_equal(np.asarray(s_obs), np.asarray(p_obs))
            self.assertEqual([row["task_id"] for row in s_info], [row["task_id"] for row in p_info])
            self.assertEqual([row["action_sum"] for row in s_info], [row["action_sum"] for row in p_info])
        finally:
            parallel.close()

    def test_parallel_ctor_does_not_advance_global_numpy(self):
        np.random.seed(123)
        expect = np.random.randint(0, 10**6, size=8)
        np.random.seed(123)
        parallel = MetaParallelEnvExecutor(DummyStepEnv(), 3, 1, 20000)
        try:
            got = np.random.randint(0, 10**6, size=8)
        finally:
            parallel.close()
        np.testing.assert_array_equal(got, expect)


if __name__ == "__main__":
    unittest.main()
