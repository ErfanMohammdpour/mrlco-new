#!/usr/bin/env python3
"""Audit dump helpers. Numpy/stdlib only. No GPU. No training."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spec.train_audit import (
    TrainAuditWriter,
    action_histogram,
    compact_paths,
    health_verdict,
    jsonable,
    mean_token_entropy,
    summarize_paths,
)


class DummyTask:
    def __init__(self):
        self.id_name = "1"
        self.processing_data_size = 10.0
        self.transmission_data_size = 1.0
        self.depth = 0
        self.heft_score = 0.0


class DummyGraph:
    def __init__(self):
        self.task_number = 2
        t = DummyTask()
        self.task_list = [t, t]
        self.prioritize_sequence = [0, 1]
        self.dependency = np.eye(2)
        self.pre_task_sets = [set(), {0}]
        self.succ_task_sets = [{1}, set()]


class TestAuditHelpers(unittest.TestCase):
    def test_jsonable_numpy(self):
        self.assertEqual(jsonable(np.int32(3)), 3)
        self.assertEqual(jsonable(np.array([1.0, 2.0])), [1.0, 2.0])

    def test_entropy_uniform_logits(self):
        logits = np.zeros((4, 3), dtype=np.float64)
        ent = mean_token_entropy(logits)
        self.assertGreater(ent, 1.0)
        self.assertLess(ent, 1.2)

    def test_collapse_flag(self):
        rows = [
            {
                "actions": [1, 1, 1, 1],
                "rewards": [0.1, 0.1, 0.1, 0.1],
                "return_sum": 0.4,
                "finish_time": 10.0,
                "energy_sum": 1.0,
                "entropy": 0.01,
                "distribution_id": 3,
            }
        ]
        summary = summarize_paths(rows)
        health = health_verdict(summary, inner_policy_losses=[[0.1, 0.1, 0.1]], k_steps=3)
        self.assertFalse(health["ok"])
        self.assertTrue(any(f.startswith("action_collapse_MEC") for f in health["flags"]))

    def test_healthy_mixed_actions(self):
        rows = [
            {
                "actions": [0, 1, 2, 1],
                "rewards": [0.1, -0.2, 0.0, 0.3],
                "return_sum": 0.2,
                "finish_time": 8.0,
                "energy_sum": 1.0,
                "entropy": 0.9,
                "distribution_id": 1,
            }
        ]
        summary = summarize_paths(rows)
        health = health_verdict(
            summary,
            inner_policy_losses=[[0.01, 0.02, 0.015]],
            inner_value_losses=[[0.1, 0.1, 0.1]],
            k_steps=3,
        )
        self.assertTrue(health["ok"])
        self.assertEqual(health["flags"], [])

    def test_wrong_k_steps_flag(self):
        summary = {"n_traj": 1, "action_hist": action_histogram([0, 1, 2]), "has_nan_reward": False}
        health = health_verdict(summary, inner_policy_losses=[[0.1]], k_steps=3)
        self.assertFalse(health["inner_apply_ok"])

    def test_compact_paths_graph_index_cycles(self):
        paths = {
            0: [
                {"actions": [0, 1], "rewards": [0.1, 0.2], "finish_time": 5.0, "logits": np.zeros((2, 3))},
                {"actions": [2, 2], "rewards": [0.0, 0.0], "finish_time": 6.0, "logits": np.zeros((2, 3))},
                {"actions": [1, 0], "rewards": [0.3, 0.1], "finish_time": 4.0, "logits": np.zeros((2, 3))},
            ]
        }
        specs = [{"dist_index": 0, "graph_indices": np.array([7, 9])}]

        class Env:
            distribution_ids = [11]

        rows = compact_paths(paths, specs, Env())
        self.assertEqual(rows[0]["graph_index"], 7)
        self.assertEqual(rows[1]["graph_index"], 9)
        self.assertEqual(rows[2]["graph_index"], 7)
        self.assertEqual(rows[2]["repeat"], 1)
        self.assertEqual(rows[0]["distribution_id"], 11)

    def test_writer_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            writer = TrainAuditWriter(tmp)
            writer.begin_iter(0)
            writer.stage(0, "sample_tasks", {"n": 1})
            writer.write_jsonl(0, "trajs_ppo.jsonl", [{"return_sum": 0.1}])
            writer.finish_iter(0, {"health": {"ok": True, "flags": []}})
            stages = (Path(tmp) / "audit" / "iter_0000" / "stages.jsonl").read_text().strip().splitlines()
            self.assertGreaterEqual(len(stages), 3)
            first = json.loads(stages[0])
            self.assertEqual(first["stage"], "begin_iter")
            health = json.loads((Path(tmp) / "audit" / "health.jsonl").read_text().strip())
            self.assertTrue(health["ok"])


if __name__ == "__main__":
    unittest.main()
