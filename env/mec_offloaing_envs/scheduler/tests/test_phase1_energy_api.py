#!/usr/bin/env python3
"""Phase 1 Energy API: component totals, attribution, pure-location refs."""

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
    Location,
    schedule,
    schedule_via_adapter,
)
from env.mec_offloaing_envs.scheduler.energy_api import (  # noqa: E402
    attribute_energy_by_task,
    compute_reference_ranges,
    j_report,
    normalize,
    pure_location_plan,
    split_v2v_times,
    transfers_for_task,
)


class _FakeTask:
    def __init__(self, proc: int, tx: int):
        self.processing_data_size = proc
        self.transmission_data_size = tx


class _FakeTG:
    def __init__(self):
        # Chain 0→1→2; mixed actions useful for V2V direction.
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


class TestEnergyComponents(unittest.TestCase):
    def setUp(self):
        self.cfg = ResourceConfig.from_frozen_yaml()
        self.dag = CanonicalDAG.from_records(
            [
                CanonicalTask(0, 1_048_576, 458_752, 1_048_576),
                CanonicalTask(1, 1_048_576, 458_752, 0),
            ],
            [(0, 1, 458_752)],
        )

    def test_mobile_equals_ue_plus_helper(self):
        result = schedule(self.dag, [0, 1], [0, 2], self.cfg)
        e = result.energy
        self.assertAlmostEqual(e.total_mobile_joules, e.total_ue_joules + e.total_helper_joules)
        self.assertAlmostEqual(
            e.total_ue_joules,
            e.ue_local_cpu_joules
            + e.ue_mec_uplink_joules
            + e.ue_mec_downlink_joules
            + e.ue_v2v_tx_joules
            + e.ue_v2v_rx_joules,
        )
        self.assertAlmostEqual(
            e.total_helper_joules,
            e.helper_compute_joules + e.helper_v2v_tx_joules + e.helper_v2v_rx_joules,
        )

    def test_mec_optional_excluded_from_mobile(self):
        result = schedule(self.dag, [0, 1], [1, 1], self.cfg)
        self.assertEqual(result.energy.mec_compute_joules_optional, 0.0)
        self.assertAlmostEqual(
            result.energy.total_system_joules_optional, result.total_mobile_joules
        )
        self.assertNotIn(
            "mec_compute",
            str(result.energy.total_mobile_joules),
        )
        # All-MEC still only radio (UE) in mobile scope.
        self.assertGreater(result.total_mobile_joules, 0.0)

    def test_attribution_sums_to_mobile(self):
        result = schedule(self.dag, [0, 1], [0, 2], self.cfg)
        by_task = attribute_energy_by_task(result, self.cfg)
        self.assertAlmostEqual(sum(by_task.values()), result.total_mobile_joules, places=9)

    def test_all_mec_internal_task_present(self):
        dag = CanonicalDAG.from_records(
            [
                CanonicalTask(0, 1_048_576, 458_752, 1_048_576),
                CanonicalTask(1, 1_048_576, 458_752, 0),
                CanonicalTask(2, 1_048_576, 458_752, 0),
            ],
            [(0, 1, 458_752), (1, 2, 458_752)],
        )
        result = schedule(dag, [0, 1, 2], [1, 1, 1], self.cfg)
        by_task = attribute_energy_by_task(result, self.cfg)
        self.assertEqual(set(by_task), set(result.tasks))
        self.assertEqual(by_task[1], 0.0)
        self.assertAlmostEqual(sum(by_task.values()), result.total_mobile_joules, places=9)


class TestReferenceRanges(unittest.TestCase):
    def setUp(self):
        self.cfg = ResourceConfig.from_frozen_yaml()
        self.tg = _FakeTG()

    def test_pure_location_refs(self):
        refs = compute_reference_ranges(self.tg, self.cfg)
        self.assertEqual(refs.source, "pure_location_reference_range")
        self.assertLessEqual(refs.L_ref_min, refs.L_ref_max)
        self.assertLessEqual(refs.E_ref_min, refs.E_ref_max)
        self.assertGreater(refs.L_scale, 0.0)
        self.assertGreater(refs.E_scale, 0.0)
        # all_UE energy should be one of the three pure plans
        order = self.tg.prioritize_sequence
        ue_result, _, _ = schedule_via_adapter(
            self.tg, pure_location_plan(order, 0), self.cfg
        )
        self.assertAlmostEqual(refs.E_ue, ue_result.total_mobile_joules)
        self.assertAlmostEqual(refs.L_ue, ue_result.makespan_seconds)

    def test_j_report_in_unit_interval(self):
        refs = compute_reference_ranges(self.tg, self.cfg)
        j = j_report(refs.L_ue, refs.E_ue, refs)
        self.assertGreaterEqual(j, 0.0)
        self.assertLessEqual(j, 1.0)
        # Midpoint of both norms → 0.5 if scales nonzero
        mid_l = 0.5 * (refs.L_ref_min + refs.L_ref_max)
        mid_e = 0.5 * (refs.E_ref_min + refs.E_ref_max)
        self.assertAlmostEqual(j_report(mid_l, mid_e, refs), 0.5, places=6)

    def test_normalize_clips(self):
        with self.assertLogs(level="WARNING") as cm:
            v = normalize(10.0, 0.0, 1.0, name="L")
        self.assertEqual(v, 1.0)
        self.assertTrue(any("clip_and_log" in m for m in cm.output))


class TestPerStepAttribution(unittest.TestCase):
    def test_adapter_energy_list_sums(self):
        cfg = ResourceConfig.from_frozen_yaml()
        tg = _FakeTG()
        plan = [(0, 0), (1, 1), (2, 2)]
        result, deltas, energy = schedule_via_adapter(tg, plan, cfg)
        self.assertAlmostEqual(sum(energy), result.total_mobile_joules, places=9)
        self.assertAlmostEqual(sum(deltas), result.makespan_seconds, places=9)


class TestV2VDirectionSplit(unittest.TestCase):
    def test_helper_to_ue_return_is_downlink(self):
        cfg = ResourceConfig.from_frozen_yaml()
        # Single helper sink: external UE→HELPER (up) + terminal HELPER→UE (down).
        dag = CanonicalDAG.from_records(
            [CanonicalTask(0, 1_048_576, 458_752, 1_048_576)],
            [],
        )
        result = schedule(dag, [0], [2], cfg)
        hops = transfers_for_task(result, 0)
        up, down = split_v2v_times(hops)
        self.assertGreater(up, 0.0)
        self.assertGreater(down, 0.0)
        returns = [t for t in hops if t.hop == "V2V" and t.dst_task_id is None]
        self.assertEqual(len(returns), 1)
        self.assertEqual(returns[0].src_location, Location.HELPER)
        self.assertEqual(returns[0].dst_location, Location.UE)
        self.assertAlmostEqual(returns[0].end - returns[0].start, down)

if __name__ == "__main__":
    unittest.main()
