#!/usr/bin/env python3
"""Energy API hardenings + post-hoc telescoping reward tests."""

from __future__ import annotations

import math
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
    EnergyBreakdown,
    ResourceConfig,
    schedule,
    schedule_via_adapter,
)
from env.mec_offloaing_envs.scheduler.energy_api import (  # noqa: E402
    ReferenceRanges,
    attribute_energy_components_by_task,
    normalize,
)
from env.mec_offloaing_envs.scheduler.reward import (  # noqa: E402
    expected_episode_return,
    provisional_plan,
    telescoping_token_rewards,
)


class _FakeTask:
    def __init__(self, proc: int, tx: int):
        self.processing_data_size = proc
        self.transmission_data_size = tx


class _FakeTG:
    def __init__(self):
        self.task_number = 3
        self.task_list = [
            _FakeTask(1_048_576, 458_752),
            _FakeTask(1_048_576, 458_752),
            _FakeTask(1_048_576, 458_752),
        ]
        self.prioritize_sequence = [0, 1, 2]
        self.pre_task_sets = [{}, {0}, {1}]
        self.edge_set = [
            [0, 0, 0, 458_752, 1, 1, 0],
            [1, 1, 0, 458_752, 2, 2, 0],
        ]


class TestNormalizeHardening(unittest.TestCase):
    def test_nan_rejected(self):
        with self.assertRaises(ValueError):
            normalize(float("nan"), 0.0, 1.0)
        with self.assertRaises(ValueError):
            normalize(0.5, float("nan"), 1.0)
        with self.assertRaises(ValueError):
            normalize(0.5, 0.0, float("inf"))

    def test_inverted_range_rejected(self):
        with self.assertRaises(ValueError):
            normalize(0.5, 1.0, 0.0)

    def test_nan_refs_rejected(self):
        with self.assertRaises(ValueError):
            ReferenceRanges(
                L_ue=float("nan"),
                L_mec=1.0,
                L_helper=2.0,
                E_ue=1.0,
                E_mec=1.0,
                E_helper=1.0,
            )


class TestComponentAttribution(unittest.TestCase):
    def test_components_sum_to_episode(self):
        cfg = ResourceConfig.from_frozen_yaml()
        dag = CanonicalDAG.from_records(
            [
                CanonicalTask(0, 1_048_576, 458_752, 1_048_576),
                CanonicalTask(1, 1_048_576, 458_752, 0),
            ],
            [(0, 1, 458_752)],
        )
        for a0, a1 in ((0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 1), (2, 2)):
            result = schedule(dag, [0, 1], [a0, a1], cfg)
            by_task = attribute_energy_components_by_task(result, cfg)
            summed = EnergyBreakdown.sum_many(by_task.values())
            for name in EnergyBreakdown.COMPONENT_FIELDS:
                if name == "mec_compute_joules_optional":
                    continue
                self.assertAlmostEqual(
                    getattr(summed, name),
                    getattr(result.energy, name),
                    places=9,
                    msg=f"actions=({a0},{a1}) field={name}",
                )
            self.assertAlmostEqual(summed.total_mobile_joules, result.total_mobile_joules, places=9)


class TestTelescopingReward(unittest.TestCase):
    def setUp(self):
        self.cfg = ResourceConfig.from_frozen_yaml()
        self.tg = _FakeTG()

    def test_provisional_all_ue_suffix(self):
        plan = provisional_plan([0, 1, 2], [1], fill=0)
        self.assertEqual(plan, [(0, 1), (1, 0), (2, 0)])

    def test_sum_rewards_equals_closed_form(self):
        plan = [(0, 1), (1, 2), (2, 0)]
        out = telescoping_token_rewards(self.tg, plan, self.cfg, include_energy=True)
        expected = expected_episode_return(
            out.makespans, out.energies, out.refs, include_energy=True
        )
        self.assertAlmostEqual(sum(out.rewards), expected, places=9)
        closed = -(
            0.5 * (out.makespans[-1] - out.makespans[0]) / out.refs.L_scale
            + 0.5 * (out.energies[-1] - out.energies[0]) / out.refs.E_scale
        )
        self.assertAlmostEqual(sum(out.rewards), closed, places=9)

    def test_all_ue_plan_zero_return(self):
        plan = [(0, 0), (1, 0), (2, 0)]
        out = telescoping_token_rewards(self.tg, plan, self.cfg)
        self.assertAlmostEqual(sum(out.rewards), 0.0, places=9)
        self.assertTrue(all(abs(r) < 1e-12 for r in out.rewards))

    def test_rewards_may_be_signed(self):
        # Mixed plan vs all_UE fill can improve or worsen provisional metrics.
        plan = [(0, 1), (1, 1), (2, 1)]
        out = telescoping_token_rewards(self.tg, plan, self.cfg)
        self.assertEqual(len(out.rewards), 3)
        # At least one nonzero delta expected for all-MEC vs all-UE baseline.
        self.assertTrue(any(abs(r) > 1e-12 for r in out.rewards))

    def test_j_report_separate_from_training_sum(self):
        plan = [(0, 1), (1, 2), (2, 1)]
        out = telescoping_token_rewards(self.tg, plan, self.cfg)
        # J_report is clipped [0,1] scientific metric — not equal to -sum(r) in general.
        self.assertGreaterEqual(out.j_report_value, 0.0)
        self.assertLessEqual(out.j_report_value, 1.0)
        self.assertNotAlmostEqual(out.j_report_value, -sum(out.rewards), places=6)

    def test_final_matches_direct_schedule(self):
        plan = [(0, 2), (1, 1), (2, 0)]
        out = telescoping_token_rewards(self.tg, plan, self.cfg)
        direct, _, energy = schedule_via_adapter(self.tg, plan, self.cfg)
        self.assertAlmostEqual(out.final_makespan, direct.makespan_seconds, places=9)
        self.assertAlmostEqual(out.final_energy, direct.total_mobile_joules, places=9)
        self.assertAlmostEqual(sum(out.final_per_task_energy), sum(energy), places=9)


if __name__ == "__main__":
    unittest.main()
