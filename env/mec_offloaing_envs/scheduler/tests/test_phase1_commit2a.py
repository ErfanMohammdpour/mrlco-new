#!/usr/bin/env python3
"""Commit 2A: adapter validation + env wrapper == direct engine."""

from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def _stub_optional(name: str) -> None:
    if name in sys.modules:
        return
    try:
        __import__(name)
    except Exception:
        sys.modules[name] = types.ModuleType(name)


for name in ("gym", "gym.core", "graphviz", "pydotplus", "pydotplus.graphviz"):
    _stub_optional(name)
if not hasattr(sys.modules.get("gym.core", types.ModuleType("gym.core")), "Env"):
    sys.modules.setdefault("gym", types.ModuleType("gym"))
    sys.modules.setdefault("gym.core", types.ModuleType("gym.core"))
    sys.modules["gym.core"].Env = type("Env", (), {})
if not hasattr(sys.modules.get("graphviz", types.ModuleType("graphviz")), "Digraph"):
    sys.modules.setdefault("graphviz", types.ModuleType("graphviz"))
    sys.modules["graphviz"].Digraph = type("Digraph", (), {})


class _FakeTask:
    def __init__(self, proc: int, tx: int):
        self.processing_data_size = proc
        self.transmission_data_size = tx


class _FakeTG:
    """Minimal OffloadingTaskGraph stand-in with edge_set records."""

    def __init__(self):
        proc, tx = 1048576, 458752
        self.task_number = 3
        self.task_list = [_FakeTask(proc, tx) for _ in range(3)]
        self.pre_task_sets = [set(), {0}, {1}]
        self.succ_task_sets = [{1}, {2}, set()]
        # edge = [src, src_depth, src_proc, transmission_cost, dst, dst_depth, dst_proc]
        self.edge_set = [
            [0, 0, proc, tx, 1, 1, proc],
            [1, 1, proc, tx, 2, 2, proc],
            [1, 1, proc, tx, 2, 2, proc],  # exact duplicate
        ]
        self.prioritize_sequence = [0, 1, 2]


from env.mec_offloaing_envs.scheduler import (  # noqa: E402
    AdapterValidationError,
    ResourceConfig,
    schedule,
    schedule_via_adapter,
    to_canonical_dag,
    validate_plan,
)


class TestAdapter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tg = _FakeTG()
        cls.cfg = ResourceConfig.from_frozen_yaml()

    def test_duplicate_collapse_from_edge_set(self):
        dag = to_canonical_dag(self.tg)
        self.assertEqual(dag.edge_record_count, 3)
        self.assertEqual(dag.unique_edge_count, 2)

    def test_mapping_and_schedule(self):
        plan = [(0, 0), (1, 0), (2, 0)]
        validate_plan(self.tg, plan)
        result, deltas, energy = schedule_via_adapter(self.tg, plan, self.cfg)
        self.assertEqual(len(deltas), 3)
        self.assertAlmostEqual(sum(deltas), result.makespan_seconds, places=9)
        self.assertAlmostEqual(sum(energy), result.total_mobile_joules, places=6)
        self.assertAlmostEqual(result.makespan_seconds, 3.0, places=9)

    def test_wrapper_matches_direct_engine(self):
        # Emulate OffloadingEnvironment.get_scheduling_cost_step_by_step body
        plan = [(0, 0), (1, 1), (2, 0)]
        result, lat, energy = schedule_via_adapter(self.tg, plan, self.cfg)
        dag = to_canonical_dag(self.tg)
        direct = schedule(dag, [0, 1, 2], [0, 1, 0], self.cfg)
        self.assertAlmostEqual(result.makespan_seconds, direct.makespan_seconds, places=9)
        self.assertAlmostEqual(lat and sum(lat), direct.makespan_seconds, places=9)
        self.assertEqual(result.topo_order, direct.topo_order)

    def test_bad_action_rejected(self):
        with self.assertRaises(AdapterValidationError):
            validate_plan(self.tg, [(0, 9), (1, 0), (2, 0)])

    def test_nontopo_rejected(self):
        with self.assertRaises(AdapterValidationError):
            validate_plan(self.tg, [(2, 0), (1, 0), (0, 0)])


if __name__ == "__main__":
    unittest.main()
