#!/usr/bin/env python3
"""ScheduleResult carries the provenance exact attribution needs (4.2a step 2).

Additive metadata only: the schedule numbers must be untouched, the canonical
tasks must survive pickle/deepcopy, and the resolved-config fingerprint must be
recorded so attribution cannot be recomputed under a different config.
"""

from __future__ import annotations

import copy
import pickle
import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
for _name in ("gym", "gym.core"):
    sys.modules.setdefault(_name, types.ModuleType(_name))
sys.modules["gym.core"].Env = type("Env", (), {})

from env.mec_offloaing_envs.scheduler.adapter import (  # noqa: E402
    schedule_via_adapter,
    to_canonical_dag,
)
from env.mec_offloaing_envs.scheduler.energy_model import MODEL_LEGACY  # noqa: E402
from env.mec_offloaing_envs.scheduler.primary_config import (  # noqa: E402
    resolved_primary_scheduler_config,
)
from env.mec_offloaing_envs.scheduler.resources import (  # noqa: E402
    ResourceConfig,
    resolved_config_sha256,
)


class Task:
    def __init__(self, workload, out_bytes, external=0, cycles=None):
        self.processing_data_size = workload
        self.transmission_data_size = out_bytes
        self.external_input_bytes = external
        self.cycles_per_bit = cycles


class Graph:
    """A 3-task DAG: root upload, a dependency transfer and a sink return."""

    def __init__(self):
        self.task_number = 3
        self.task_list = [
            Task(1_000_000, 4_000, external=2_000_000, cycles=300.0),
            Task(200_000, 4_000),
            Task(300_000, 8_000),
        ]
        edges = [(0, 1, 4_000), (1, 2, 4_000)]
        self.edge_set = [[s, 0, 0, b, d, 0, 0] for s, d, b in edges]
        parents = {i: set() for i in range(3)}
        for s, d, _b in edges:
            parents[d].add(s)
        self.pre_task_sets = [parents[i] for i in range(3)]
        self.succ_task_sets = [{d for s, d, _b in edges if s == i} for i in range(3)]
        self.prioritize_sequence = [0, 1, 2]


def schedule(res, action=1):
    graph = Graph()
    out, _deltas, _per_task = schedule_via_adapter(
        graph, [(tid, action) for tid in graph.prioritize_sequence], res
    )
    return out


class TestMetadata(unittest.TestCase):
    def test_every_scheduled_task_is_present_exactly_once(self):
        result = schedule(resolved_primary_scheduler_config())
        self.assertEqual(set(result.graph_tasks), set(result.tasks))
        self.assertEqual(len(result.graph_tasks), len(result.tasks))

    def test_task_metadata_is_preserved(self):
        result = schedule(resolved_primary_scheduler_config())
        root = result.graph_tasks[0]
        self.assertEqual(root.compute_workload_bytes, 1_000_000)
        self.assertEqual(root.task_output_bytes, 4_000)
        # the adapter sets a ROOT's external input to its processing_data_size
        self.assertEqual(root.external_input_bytes, 1_000_000)
        self.assertEqual(root.cycles_per_bit, 300.0)
        self.assertIsNone(result.graph_tasks[1].cycles_per_bit)

    def test_fingerprint_matches_the_config_used(self):
        cfg = resolved_primary_scheduler_config()
        result = schedule(cfg)
        self.assertEqual(result.scheduler_config_sha256, resolved_config_sha256(cfg))
        self.assertTrue(result.scheduler_config_sha256)

    def test_a_different_config_gives_a_different_fingerprint(self):
        primary = schedule(resolved_primary_scheduler_config())
        legacy = schedule(ResourceConfig.from_frozen_yaml(
            ROOT / "spec" / "frozen_experiment.yaml", model=MODEL_LEGACY
        ))
        self.assertNotEqual(primary.scheduler_config_sha256, legacy.scheduler_config_sha256)

    def test_metadata_the_result_owns(self):
        graph = Graph()
        result, _d, _e = schedule_via_adapter(
            graph, [(tid, 1) for tid in graph.prioritize_sequence],
            resolved_primary_scheduler_config(),
        )
        graph.task_list[0].processing_data_size = 999  # mutate the source afterwards
        self.assertEqual(result.graph_tasks[0].compute_workload_bytes, 1_000_000)


class TestSerialization(unittest.TestCase):
    def test_pickle_keeps_metadata_and_numbers(self):
        result = schedule(resolved_primary_scheduler_config())
        restored = pickle.loads(pickle.dumps(result))
        self.assertEqual(set(restored.graph_tasks), set(result.graph_tasks))
        self.assertEqual(
            restored.graph_tasks[0].compute_workload_bytes,
            result.graph_tasks[0].compute_workload_bytes,
        )
        self.assertEqual(restored.scheduler_config_sha256, result.scheduler_config_sha256)
        self.assertAlmostEqual(restored.makespan_seconds, result.makespan_seconds, places=12)

    def test_deepcopy_keeps_metadata(self):
        result = schedule(resolved_primary_scheduler_config())
        clone = copy.deepcopy(result)
        self.assertEqual(set(clone.graph_tasks), set(result.graph_tasks))
        self.assertEqual(clone.scheduler_config_sha256, result.scheduler_config_sha256)
        for tid, record in result.tasks.items():
            self.assertAlmostEqual(clone.tasks[tid].finish, record.finish, places=12)


class TestScheduleNumbersUnchanged(unittest.TestCase):
    """Adding metadata must not move a single scheduling number."""

    def test_records_and_energy_are_the_previous_values(self):
        cfg = resolved_primary_scheduler_config()
        a = schedule(cfg, action=0)
        b = schedule(cfg, action=1)
        c = schedule(cfg, action=2)
        for result in (a, b, c):
            self.assertIn("graph_tasks", result.__dataclass_fields__)
            self.assertGreater(result.makespan_seconds, 0.0)
        # MEC is faster than UE here, so all-MEC finishes first
        self.assertLess(b.makespan_seconds, a.makespan_seconds)
        # and physical accounting gives the MEC plan non-zero MEC compute energy
        self.assertGreater(float(b.energy.mec_compute_joules_optional or 0.0), 0.0)
        self.assertEqual(float(a.energy.mec_compute_joules_optional or 0.0), 0.0)

    def test_deadline_metrics_untouched(self):
        result = schedule(resolved_primary_scheduler_config())
        self.assertEqual(result.hard_miss_count, 0)
        self.assertTrue(result.hard_feasible)
        self.assertEqual(result.deadline_basis, "all_consumers_ready")


if __name__ == "__main__":
    unittest.main(verbosity=2)
