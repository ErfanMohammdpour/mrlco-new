"""Picklable dummy env for serial-vs-parallel executor equality. No TensorFlow."""

from __future__ import annotations

import numpy as np


class DummyStepEnv(object):
    """Deterministic step: reward depends only on task_id and actions."""

    def __init__(self):
        self.task_id = 0
        self.graph_indices = np.asarray([0, 1], dtype=np.int32)

    def set_task(self, task):
        if isinstance(task, dict):
            self.task_id = int(task["dist_index"])
            self.graph_indices = np.asarray(task["graph_indices"], dtype=np.int32)
        else:
            self.task_id = int(task)
            self.graph_indices = np.asarray([0, 1], dtype=np.int32)

    def reset(self):
        n = int(self.graph_indices.size)
        return np.full((n, 4), float(self.task_id), dtype=np.float32)

    def step(self, actions):
        actions = np.asarray(actions, dtype=np.int32)
        if actions.ndim == 1:
            actions = actions.reshape(1, -1)
        n = int(actions.shape[0])
        scale = float(self.task_id + 1)
        rewards = np.ones((n,), dtype=np.float64) * scale + actions.sum(axis=1).astype(np.float64)
        info = {
            "task_id": int(self.task_id),
            "action_sum": int(actions.sum()),
        }
        return self.reset(), rewards, True, info
