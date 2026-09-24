#!/usr/bin/env python3
"""Timing x accounting separation: the lock the energy-constraint work depends on.

The mode the experiment needs is timing=legacy_frozen_rates with
energy_model=physical_v1. Under it, schedule, bounds, masks and observation v3 must
be BIT-IDENTICAL to the all-legacy run, and only the energy breakdown may move
(with MEC compute > 0, which legacy zeroes).

Also pins the full cross-product for CPU and radio, and the fail-loud rules for
physical timing sources: an accounting model is never a rate source.
"""

from __future__ import annotations

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

from env.mec_offloaing_envs.scheduler import encoder_obs as eo  # noqa: E402
from env.mec_offloaing_envs.scheduler.adapter import (  # noqa: E402
    schedule_via_adapter,
    to_canonical_dag,
)
from env.mec_offloaing_envs.scheduler.energy_model import (  # noqa: E402
    MODEL_LEGACY,
    MODEL_PHYSICAL,
    SCOPE_SYSTEM,
)
from env.mec_offloaing_envs.scheduler.model import CanonicalDAG, CanonicalTask  # noqa: E402
from env.mec_offloaing_envs.scheduler.resources import (  # noqa: E402
    TIMING_LEGACY,
    TIMING_PHYSICAL,
    ResourceConfig,
)
from env.mec_offloaing_envs.scheduler.static_bounds import (  # noqa: E402
    static_action_bounds,
)

YAML = Path(__file__).resolve().parents[4] / "spec" / "frozen_experiment.yaml"
XI = 300.0


def config(timing=TIMING_LEGACY, accounting=MODEL_LEGACY, scope=SCOPE_SYSTEM,
           radio_timing=TIMING_LEGACY, radio_accounting=MODEL_LEGACY):
    return ResourceConfig.from_frozen_yaml(
        YAML,
        energy_model=accounting,
        radio_model=radio_accounting,
        timing_model=timing,
        radio_timing_model=radio_timing,
        energy_scope=scope,
    )


def dag(n=4):
    tasks = [
        CanonicalTask(
            task_id=i,
            compute_workload_bytes=4_000_000 if i == 0 else 200_000,
            task_output_bytes=50_000,
            external_input_bytes=4_000_000 if i == 0 else 0,
        )
        for i in range(n)
    ]
    return CanonicalDAG.from_records(tasks, [(i, i + 1, 50_000) for i in range(n - 1)])


def plan(order, action):
    return [(tid, action) for tid in order]


class TestCrossProductCpu(unittest.TestCase):
    """Same timing must give the same rates whatever the accounting model is."""

    def test_rates_follow_timing_only(self):
        legacy_legacy = config()
        legacy_physical = config(accounting=MODEL_PHYSICAL)
        physical_legacy = config(timing=TIMING_PHYSICAL)
        physical_physical = config(timing=TIMING_PHYSICAL, accounting=MODEL_PHYSICAL)
        for loc in ("UE", "MEC", "HELPER"):
            from env.mec_offloaing_envs.scheduler.model import Location

            a = legacy_legacy.cpu_rate_bytes_per_second(Location[loc], XI)
            b = legacy_physical.cpu_rate_bytes_per_second(Location[loc], XI)
            c = physical_legacy.cpu_rate_bytes_per_second(Location[loc], XI)
            d = physical_physical.cpu_rate_bytes_per_second(Location[loc], XI)
            self.assertAlmostEqual(a, b, places=12, msg="legacy timing moved by accounting")
            self.assertAlmostEqual(c, d, places=12, msg="physical timing moved by accounting")

    def test_physical_timing_source_is_independent_of_accounting(self):
        # accounting legacy + physical timing must still resolve a PHYSICAL source
        res = config(timing=TIMING_PHYSICAL, accounting=MODEL_LEGACY)
        self.assertTrue(res.timing_is_physical)
        self.assertFalse(res.physical)          # accounting is legacy
        self.assertTrue(res.timing_tiers.is_physical)

    def test_radio_cross_product(self):
        for rt in (TIMING_LEGACY, TIMING_PHYSICAL):
            legacy = config(radio_timing=rt, radio_accounting=MODEL_LEGACY)
            physical = config(radio_timing=rt, radio_accounting=MODEL_PHYSICAL)
            for hop in ("MEC_UL", "MEC_DL", "V2V"):
                self.assertAlmostEqual(
                    legacy.hop_rate(hop), physical.hop_rate(hop), places=12,
                    msg="hop rate moved when only radio ACCOUNTING changed",
                )


class TestFailLoudSources(unittest.TestCase):
    def test_physical_timing_needs_a_physical_source(self):
        res = config(timing=TIMING_PHYSICAL)
        with self.assertRaises(ValueError):
            ResourceConfig(
                **{**res.__dict__, "timing_tiers": None},
            )

    def test_legacy_accounting_object_is_not_a_rate_source(self):
        res = config(timing=TIMING_PHYSICAL, accounting=MODEL_PHYSICAL)
        legacy_spec = ResourceConfig.from_frozen_yaml(YAML, model=MODEL_LEGACY).energy_model
        with self.assertRaises(ValueError):
            ResourceConfig(**{**res.__dict__, "timing_tiers": legacy_spec})

    def test_physical_radio_timing_needs_a_spec(self):
        res = config(radio_timing=TIMING_PHYSICAL)
        with self.assertRaises(ValueError):
            ResourceConfig(**{**res.__dict__, "radio_timing_spec": None})


class TestEndToEndInvariance(unittest.TestCase):
    """A vs B: only the accounting model differs, nothing else may move."""

    def setUp(self):
        self.graph = dag()
        self.order = [0, 1, 2, 3]
        self.a = config()                                  # legacy / legacy
        self.b = config(accounting=MODEL_PHYSICAL)         # legacy timing, physical energy

    def _schedule(self, res, action):
        out, _, per_task = schedule_via_adapter(
            _FakeGraph(self.graph), plan(self.order, action), res
        )
        return out

    def test_schedule_is_bit_identical(self):
        for action in (0, 1, 2):
            a = self._schedule(self.a, action)
            b = self._schedule(self.b, action)
            self.assertAlmostEqual(a.makespan_seconds, b.makespan_seconds, places=12)
            for tid, rec in a.tasks.items():
                other = b.tasks[tid]
                self.assertAlmostEqual(rec.start, other.start, places=12)
                self.assertAlmostEqual(rec.finish, other.finish, places=12)
                self.assertAlmostEqual(
                    rec.all_consumers_ready, other.all_consumers_ready, places=12
                )
                self.assertEqual(rec.location, other.location)

    def test_bounds_and_masks_are_identical(self):
        ba = static_action_bounds(self.graph, self.order, self.a, cycles_per_bit=XI)
        bb = static_action_bounds(self.graph, self.order, self.b, cycles_per_bit=XI)
        self.assertEqual(ba.finish_lb, bb.finish_lb)
        self.assertEqual(ba.ready_lb, bb.ready_lb)
        deadlines = [0.5 * min(row) for row in ba.ready_lb]
        self.assertEqual(
            ba.feasible_by_deadline(deadlines), bb.feasible_by_deadline(deadlines)
        )

    def test_observation_v3_is_identical(self):
        eo.set_obs_version("v3")
        self.addCleanup(eo.set_obs_version, "v1")
        packed_a = eo.encode_canonical_dag(
            self.graph, self.order, resources=self.a, cycles_per_bit=XI
        )
        packed_b = eo.encode_canonical_dag(
            self.graph, self.order, resources=self.b, cycles_per_bit=XI
        )
        self.assertTrue((packed_a == packed_b).all())

    def test_energy_is_the_only_difference_and_mec_is_positive(self):
        a = self._schedule(self.a, 1)
        b = self._schedule(self.b, 1)
        self.assertNotAlmostEqual(
            a.energy.total_mobile_joules, b.energy.total_mobile_joules, places=6
        )
        self.assertGreater(float(b.energy.mec_compute_joules_optional or 0.0), 0.0)
        self.assertEqual(float(a.energy.mec_compute_joules_optional or 0.0), 0.0)
        self.assertGreater(b.energy.total_system_joules, b.energy.total_mobile_joules)

    def test_legacy_plan_is_byte_exact(self):
        # model="legacy" must stay exactly what it was: both axes legacy
        legacy = ResourceConfig.from_frozen_yaml(YAML, model=MODEL_LEGACY)
        self.assertEqual(legacy.timing_model, TIMING_LEGACY)
        self.assertEqual(legacy.radio_timing_model, TIMING_LEGACY)
        self.assertFalse(legacy.physical)
        a = self._schedule(legacy, 0)
        b = self._schedule(self.a, 0)
        self.assertAlmostEqual(a.makespan_seconds, b.makespan_seconds, places=12)


class _Task:
    def __init__(self, task):
        self.processing_data_size = task.compute_workload_bytes
        self.transmission_data_size = task.task_output_bytes


class _FakeGraph:
    """The adapter surface, built from the canonical DAG for a real schedule."""

    def __init__(self, graph: CanonicalDAG):
        self.task_number = len(graph.tasks)
        self.task_list = [_Task(graph.tasks[i]) for i in sorted(graph.tasks)]
        parents = {i: set() for i in range(self.task_number)}
        edges = []
        for edge in graph.edges:
            src, dst = int(edge.src_task_id), int(edge.dst_task_id)
            parents[dst].add(src)
            edges.append((src, dst, int(edge.edge_output_bytes)))
        self.edge_set = [[s, 0, 0, b, d, 0, 0] for s, d, b in edges]
        self.pre_task_sets = [parents[i] for i in range(self.task_number)]
        self.succ_task_sets = [
            {d for s, d, _b in edges if s == i} for i in range(self.task_number)
        ]
        self.prioritize_sequence = list(range(self.task_number))


if __name__ == "__main__":
    unittest.main(verbosity=2)
