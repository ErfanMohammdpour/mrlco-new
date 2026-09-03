#!/usr/bin/env python3
"""Deterministic Phase 1 property checks for closure gate."""

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


from env.mec_offloaing_envs.scheduler import (  # noqa: E402
    CanonicalDAG,
    CanonicalTask,
    ResourceConfig,
    schedule,
)
from env.mec_offloaing_envs.scheduler.reward import (  # noqa: E402
    expected_episode_return,
    telescoping_token_rewards,
)


class _FakeTask:
    def __init__(self, proc: int, tx: int):
        self.processing_data_size = proc
        self.transmission_data_size = tx


class _FakeTG:
    def __init__(self, n: int = 4, seed: int = 0):
        self.task_number = n
        self.task_list = [_FakeTask(1_048_576, 458_752) for _ in range(n)]
        self.prioritize_sequence = list(range(n))
        self.pre_task_sets = [set() if i == 0 else {i - 1} for i in range(n)]
        self.edge_set = [
            [i, 0, 0, 458_752, i + 1, 0, 0] for i in range(n - 1)
        ]


def _intervals_overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> bool:
    # Zero-duration intervals do not occupy capacity.
    if a_end <= a_start or b_end <= b_start:
        return False
    return a_start < b_end and b_start < a_end


class TestPhase1Properties(unittest.TestCase):
    def setUp(self):
        self.cfg = ResourceConfig.from_frozen_yaml()
        self.dag = CanonicalDAG.from_records(
            [
                CanonicalTask(0, 1_048_576, 458_752, 1_048_576),
                CanonicalTask(1, 1_048_576, 458_752, 0),
                CanonicalTask(2, 1_048_576, 458_752, 0),
            ],
            [(0, 1, 458_752), (1, 2, 458_752)],
        )
        self.plans = [
            ([0, 1, 2], [0, 0, 0]),
            ([0, 1, 2], [1, 1, 1]),
            ([0, 1, 2], [2, 2, 2]),
            ([0, 1, 2], [0, 1, 2]),
            ([0, 1, 2], [2, 1, 0]),
        ]

    def test_no_resource_overlap(self):
        for order, actions in self.plans:
            result = schedule(self.dag, order, actions, self.cfg)
            by_res: dict[str, list] = {}
            for iv in result.resource_intervals:
                by_res.setdefault(iv.resource, []).append(iv)
            for res, ivs in by_res.items():
                for i in range(len(ivs)):
                    for j in range(i + 1, len(ivs)):
                        self.assertFalse(
                            _intervals_overlap(ivs[i].start, ivs[i].end, ivs[j].start, ivs[j].end),
                            f"overlap on {res} for actions={actions}",
                        )

    def test_precedence(self):
        for order, actions in self.plans:
            result = schedule(self.dag, order, actions, self.cfg)
            for e in self.dag.edges:
                src = result.tasks[e.src_task_id]
                dst = result.tasks[e.dst_task_id]
                hop_ends = [
                    t.end
                    for t in result.transfers
                    if t.src_task_id == e.src_task_id and t.dst_task_id == e.dst_task_id
                ]
                # Same-location / zero-byte: payload already at dst after src.finish.
                arrival = max([src.finish] + hop_ends)
                self.assertLessEqual(
                    arrival,
                    dst.start + 1e-9,
                    f"edge {e.src_task_id}->{e.dst_task_id} actions={actions}: "
                    f"arrival={arrival} dst.start={dst.start}",
                )

    def test_terminal_return_in_makespan(self):
        for order, actions in self.plans:
            result = schedule(self.dag, order, actions, self.cfg)
            self.assertAlmostEqual(result.makespan_seconds, result.terminal_return_time)
            for tid in self.dag.sinks():
                self.assertLessEqual(result.tasks[tid].finish, result.makespan_seconds + 1e-9)

    def test_energy_conservation(self):
        for order, actions in self.plans:
            result = schedule(self.dag, order, actions, self.cfg)
            e = result.energy
            self.assertAlmostEqual(e.total_mobile_joules, e.total_ue_joules + e.total_helper_joules)
            self.assertGreaterEqual(e.total_mobile_joules, 0.0)

    def test_determinism(self):
        for order, actions in self.plans:
            a = schedule(self.dag, order, actions, self.cfg)
            b = schedule(self.dag, order, actions, self.cfg)
            self.assertAlmostEqual(a.makespan_seconds, b.makespan_seconds)
            self.assertAlmostEqual(a.total_mobile_joules, b.total_mobile_joules)
            self.assertEqual(a.topo_order, b.topo_order)

    def test_telescoping_identity(self):
        tg = _FakeTG(4)
        for actions in ([0, 0, 0, 0], [1, 1, 1, 1], [2, 1, 0, 2], [0, 1, 2, 1]):
            plan = list(zip(range(4), actions))
            out = telescoping_token_rewards(tg, plan, self.cfg)
            expected = expected_episode_return(out.makespans, out.energies, out.refs)
            self.assertAlmostEqual(sum(out.rewards), expected, places=9)
            self.assertTrue(all(abs(r) < 1e100 and r == r for r in out.rewards))  # finite


if __name__ == "__main__":
    unittest.main()
