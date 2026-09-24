#!/usr/bin/env python3
"""Exact scoped per-task attribution (4.2a part A / E1.1)."""

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
)
from env.mec_offloaing_envs.scheduler.energy_api import (  # noqa: E402
    EnergyAttributionError,
    attribute_energy_by_task,
    attribute_energy_components_by_task,
    attribute_mobile_energy_by_task,
    attribute_scoped_energy_by_task,
)
from env.mec_offloaing_envs.scheduler.energy_model import MODEL_LEGACY  # noqa: E402
from env.mec_offloaing_envs.scheduler.energy_scope import (  # noqa: E402
    SCOPE_MOBILE,
    SCOPE_REQUESTER,
    SCOPE_SYSTEM,
)
from env.mec_offloaing_envs.scheduler.model import EnergyBreakdown  # noqa: E402
from env.mec_offloaing_envs.scheduler.primary_config import (  # noqa: E402
    resolved_primary_scheduler_config,
)
from env.mec_offloaing_envs.scheduler.resources import ResourceConfig  # noqa: E402

YAML = ROOT / "spec" / "frozen_experiment.yaml"


class Task:
    def __init__(self, workload, out_bytes, cycles=None):
        self.processing_data_size = workload
        self.transmission_data_size = out_bytes
        self.cycles_per_bit = cycles


class Graph:
    """root --(dep)--> middle --(dep)--> sink; the root also uploads."""

    def __init__(self, workloads=(1_000_000, 300_000, 400_000), out=8_000, cycles=None):
        n = len(workloads)
        self.task_number = n
        self.task_list = [
            Task(workloads[i], out, cycles[i] if cycles else None) for i in range(n)
        ]
        edges = [(i, i + 1, out) for i in range(n - 1)]
        self.edge_set = [[s, 0, 0, b, d, 0, 0] for s, d, b in edges]
        parents = {i: set() for i in range(n)}
        for s, d, _b in edges:
            parents[d].add(s)
        self.pre_task_sets = [parents[i] for i in range(n)]
        self.succ_task_sets = [{d for s, d, _b in edges if s == i} for i in range(n)]
        self.prioritize_sequence = list(range(n))


def run(res, actions, graph=None):
    graph = graph or Graph()
    plan = [(tid, a) for tid, a in zip(graph.prioritize_sequence, actions)]
    result, _d, _e = schedule_via_adapter(graph, plan, res)
    return result, graph


PHYS = resolved_primary_scheduler_config()
LEGACY = ResourceConfig.from_frozen_yaml(YAML, model=MODEL_LEGACY)


class TestOwnership(unittest.TestCase):
    """Characterization: owner = dst if present else src."""

    def test_three_transfer_kinds_are_attributed_to_the_expected_task(self):
        result, _g = run(PHYS, [0, 1, 1])           # root on UE, rest on MEC
        comps = attribute_energy_components_by_task(result, PHYS)
        kinds = {(t.src_task_id, t.dst_task_id) for t in result.transfers}
        self.assertTrue(any(d is not None for _s, d in kinds), "no dependency transfer")
        self.assertTrue(any(s is not None and d is None for s, d in kinds), "no sink return")
        # every transfer owner is a scheduled task and owns its hop components
        for t in result.transfers:
            owner = t.dst_task_id if t.dst_task_id is not None else t.src_task_id
            self.assertIn(owner, comps)

    def test_all_tasks_get_a_breakdown_even_at_zero(self):
        result, _g = run(PHYS, [1, 1, 1])
        comps = attribute_energy_components_by_task(result, PHYS)
        self.assertEqual(set(comps), set(result.tasks))
        for bd in comps.values():
            self.assertIsInstance(bd, EnergyBreakdown)


class TestIdentities(unittest.TestCase):
    def _check(self, res, actions, graph=None):
        result, _g = run(res, actions, graph)
        comps = attribute_energy_components_by_task(result, res)
        for field_name in EnergyBreakdown.COMPONENT_FIELDS:
            attributed = sum(float(getattr(bd, field_name)) for bd in comps.values())
            self.assertAlmostEqual(
                attributed, float(getattr(result.energy, field_name)), places=6,
                msg=field_name,
            )
        for scope in (SCOPE_REQUESTER, SCOPE_MOBILE, SCOPE_SYSTEM):
            per_task = attribute_scoped_energy_by_task(result, res, scope=scope)
            self.assertEqual(set(per_task), set(result.tasks))
            self.assertAlmostEqual(
                sum(per_task.values()),
                __import__("env.mec_offloaing_envs.scheduler.energy_scope",
                           fromlist=["energy_scalar"]).energy_scalar(result, scope=scope),
                places=6, msg=scope,
            )
            for tid, value in per_task.items():
                self.assertGreaterEqual(value, 0.0)
        req = attribute_scoped_energy_by_task(result, res, scope=SCOPE_REQUESTER)
        mob = attribute_scoped_energy_by_task(result, res, scope=SCOPE_MOBILE)
        sysm = attribute_scoped_energy_by_task(result, res, scope=SCOPE_SYSTEM)
        for tid in result.tasks:
            self.assertLessEqual(req[tid], mob[tid] + 1e-9)
            self.assertLessEqual(mob[tid], sysm[tid] + 1e-9)

    def test_pure_and_mixed_plans(self):
        for actions in ([0, 0, 0], [1, 1, 1], [2, 2, 2], [0, 1, 2], [1, 0, 1], [2, 1, 0]):
            self._check(PHYS, actions)

    def test_same_location_chain_has_no_dependency_transfer_energy(self):
        result, _g = run(PHYS, [1, 1, 1])
        comps = attribute_energy_components_by_task(result, PHYS)
        # all-MEC: residency makes every dependency transfer zero, so only the
        # sink return can carry hop energy
        hops = [t for t in result.transfers if float(t.end) - float(t.start) > 0.0]
        # only the root's external upload (src None) and the sink return (dst None)
        # carry hop energy; every dependency transfer is co-located and zero
        self.assertTrue(all(t.src_task_id is None or t.dst_task_id is None for t in hops), hops)
        self.assertTrue(any(t.src_task_id is None for t in hops), hops)
        self.assertTrue(any(t.dst_task_id is None for t in hops), hops)
        self.assertGreater(
            sum(float(bd.mec_compute_joules_optional or 0.0) for bd in comps.values()),
            0.0,
        )


class TestCyclesPerBit(unittest.TestCase):
    def test_per_task_intensity_reaches_the_energy(self):
        fast, _g = run(PHYS, [1, 1, 1], Graph(cycles=(300.0, 300.0, 300.0)))
        slow, _g2 = run(PHYS, [1, 1, 1], Graph(cycles=(600.0, 300.0, 300.0)))
        a = attribute_energy_components_by_task(fast, PHYS)
        b = attribute_energy_components_by_task(slow, PHYS)
        self.assertGreater(
            float(b[0].mec_compute_joules_optional or 0.0),
            float(a[0].mec_compute_joules_optional or 0.0),
        )
        # and the attribution still matches the engine's own breakdown
        for field_name in EnergyBreakdown.COMPONENT_FIELDS:
            self.assertAlmostEqual(
                sum(float(getattr(bd, field_name)) for bd in b.values()),
                float(getattr(slow.energy, field_name)),
                places=6,
                msg=field_name,
            )


class TestPhysicalEvidence(unittest.TestCase):
    def test_mec_compute_and_tx_are_positive_and_system_exceeds_mobile(self):
        result, _g = run(PHYS, [1, 1, 1])
        comps = attribute_energy_components_by_task(result, PHYS)
        self.assertGreater(
            sum(float(bd.mec_compute_joules_optional or 0.0) for bd in comps.values()), 0.0
        )
        self.assertGreater(
            sum(float(bd.mec_tx_joules_optional or 0.0) for bd in comps.values()), 0.0
        )
        self.assertGreater(
            sum(attribute_scoped_energy_by_task(result, PHYS, scope=SCOPE_SYSTEM).values()),
            sum(attribute_scoped_energy_by_task(result, PHYS, scope=SCOPE_MOBILE).values()),
        )


class TestLegacyParity(unittest.TestCase):
    def test_legacy_mobile_is_zero_mec_and_matches_manual_sum(self):
        result, _g = run(LEGACY, [1, 1, 1])
        comps = attribute_energy_components_by_task(result, LEGACY)
        self.assertEqual(
            sum(float(bd.mec_compute_joules_optional or 0.0) for bd in comps.values()), 0.0
        )
        mobile = attribute_mobile_energy_by_task(result, LEGACY)
        legacy = attribute_energy_by_task(result, LEGACY)
        self.assertEqual(mobile, legacy)
        manual = {
            tid: float(bd.total_mobile_joules) for tid, bd in comps.items()
        }
        self.assertEqual(mobile, manual)


class TestFailLoud(unittest.TestCase):
    def test_fingerprint_mismatch_raises(self):
        result, _g = run(PHYS, [1, 1, 1])
        other = resolved_primary_scheduler_config(overrides={"timing_model": "physical_rates"})
        with self.assertRaises(EnergyAttributionError):
            attribute_energy_components_by_task(result, other)

    def test_physical_without_fingerprint_raises(self):
        result, _g = run(PHYS, [1, 1, 1])
        stripped = copy.copy(result)
        object.__setattr__(stripped, "scheduler_config_sha256", "")
        with self.assertRaises(EnergyAttributionError):
            attribute_energy_components_by_task(stripped, PHYS)

    def test_bad_scope_raises(self):
        result, _g = run(PHYS, [1, 1, 1])
        for bad in ("", "solar", None):
            with self.assertRaises(ValueError):
                attribute_scoped_energy_by_task(result, PHYS, scope=bad)

    def test_missing_task_metadata_raises_in_physical(self):
        result, _g = run(PHYS, [1, 1, 1])
        naked = copy.copy(result)
        object.__setattr__(naked, "graph_tasks", {})
        with self.assertRaises(EnergyAttributionError):
            attribute_energy_components_by_task(naked, PHYS)

    def test_task_set_mismatch_raises(self):
        result, _g = run(PHYS, [1, 1, 1])
        subset = {k: v for k, v in result.graph_tasks.items() if k != 0}
        broken = copy.copy(result)
        object.__setattr__(broken, "graph_tasks", subset)
        with self.assertRaises(EnergyAttributionError):
            attribute_energy_components_by_task(broken, PHYS)


class TestSerialization(unittest.TestCase):
    def test_attribution_survives_pickle_and_deepcopy(self):
        result, _g = run(PHYS, [1, 1, 1])
        expected = attribute_scoped_energy_by_task(result, PHYS, scope=SCOPE_SYSTEM)
        for clone in (pickle.loads(pickle.dumps(result)), copy.deepcopy(result)):
            self.assertEqual(
                attribute_scoped_energy_by_task(clone, PHYS, scope=SCOPE_SYSTEM), expected
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
