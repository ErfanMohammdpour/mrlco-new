#!/usr/bin/env python3
"""Reference-range / normalization regression tests on REAL graphs.

Motivation (measured, `spec/energy_reward_audit.py`): with the MARGO-SPEC-v0.1
"pure_location" reference box, a mixed plan beats all-MEC on makespan in 100% of
graphs, so the latency term of the clipped composite `j_report` saturates at 0
and the composite ranks the faster plan WORSE than all-MEC (40/40 inversions
across four distributions).

`mode="candidate_panel"` additionally bounds the reference range by
`greedy_from_mec`, which removes the inversions.
"""

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


for _name in ("gym", "gym.core", "graphviz", "pydotplus", "pydotplus.graphviz"):
    _stub_optional(_name)
if not hasattr(sys.modules.get("gym.core", types.ModuleType("gym.core")), "Env"):
    sys.modules.setdefault("gym", types.ModuleType("gym"))
    sys.modules.setdefault("gym.core", types.ModuleType("gym.core"))
    sys.modules["gym.core"].Env = type("Env", (), {})
if not hasattr(sys.modules.get("graphviz", types.ModuleType("graphviz")), "Digraph"):
    sys.modules.setdefault("graphviz", types.ModuleType("graphviz"))
    sys.modules["graphviz"].Digraph = type("Digraph", (), {})

from env.mec_offloaing_envs.offloading_task_graph import OffloadingTaskGraph  # noqa: E402
from env.mec_offloaing_envs.scheduler import (  # noqa: E402
    REFERENCE_MODE_PANEL,
    REFERENCE_MODE_PURE,
    ResourceConfig,
    compute_reference_ranges,
    greedy_from_mec_plan,
    j_report,
    pure_location_plan,
    schedule_via_adapter,
)

DATA = (
    ROOT
    / "env"
    / "mec_offloaing_envs"
    / "data"
    / "meta_offloading_20"
    / "offload_random20_1"
)
MBPS_TO_BPS = 1024.0 * 1024.0 / 8.0


class _FrozenCluster:
    def __init__(self):
        self.mobile_process_capable = 1.0 * 1024 * 1024
        self.mec_process_capable = 10.0 * 1024 * 1024
        self.v2v_process_capable = 1.0 * 1024 * 1024
        self.bandwidth_up = 7.0
        self.bandwidth_dl = 7.0
        self.v2v_bandwidth = 5.0

    def up_transmission_cost(self, data):
        return data / (self.bandwidth_up * MBPS_TO_BPS)

    def dl_transmission_cost(self, data):
        return data / (self.bandwidth_dl * MBPS_TO_BPS)

    def v2v_transmission_cost(self, data):
        return data / (self.v2v_bandwidth * MBPS_TO_BPS)


@unittest.skipUnless(DATA.exists(), "dataset not present")
class TestReferencePanel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.res = ResourceConfig.from_frozen_yaml()
        cls.cluster = _FrozenCluster()
        cls.files = sorted(DATA.glob("random.20.*.gv"))[:6]
        cls.graphs = []
        for path in cls.files:
            tg = OffloadingTaskGraph(str(path))
            tg.prioritize_tasks(cls.cluster)
            order = [int(t) for t in tg.prioritize_sequence]
            greedy, _ = greedy_from_mec_plan(tg, cls.res, max_passes=2)
            mec_res, _, _ = schedule_via_adapter(tg, pure_location_plan(order, 1), cls.res)
            gre_res, _, _ = schedule_via_adapter(tg, greedy, cls.res)
            pure = compute_reference_ranges(tg, cls.res, mode=REFERENCE_MODE_PURE)
            panel = compute_reference_ranges(tg, cls.res, mode=REFERENCE_MODE_PANEL)
            cls.graphs.append((path.name, order, pure, panel, mec_res, gre_res, greedy))

    def test_panel_bounds_are_wider_or_equal(self):
        for name, _order, pure, panel, *_ in self.graphs:
            self.assertLessEqual(panel.L_min, pure.L_ref_min + 1e-9, msg=name)
            self.assertLessEqual(panel.E_min, pure.E_ref_min + 1e-9, msg=name)
            self.assertGreaterEqual(panel.L_max, pure.L_ref_max - 1e-9, msg=name)
            self.assertGreaterEqual(panel.E_max, pure.E_ref_max - 1e-9, msg=name)

    def test_panel_range_scales_are_positive(self):
        for name, _order, _pure, panel, *_ in self.graphs:
            self.assertGreater(panel.L_scale, 0.0, msg=name)
            self.assertGreater(panel.E_scale, 0.0, msg=name)

    def test_pure_mode_still_reports_the_old_box(self):
        for name, _order, pure, _panel, *_ in self.graphs:
            self.assertEqual(pure.reference_mode, REFERENCE_MODE_PURE)
            self.assertEqual(
                pure.L_ref_min, min(pure.L_ue, pure.L_mec, pure.L_helper), msg=name
            )
            self.assertEqual(pure.L_scale, pure.L_ref_max - pure.L_ref_min, msg=name)

    def test_panel_mode_removes_clipping_saturation(self):
        """The defect is saturation, not a missing preference for speed.

        With the pure-location box every realistic mixed plan falls below
        `L_ref_min`, so its latency term clips to 0 and even all-MEC sits exactly
        on the floor 0.000 — the composite loses all resolution in the region of
        interest (measured: mean +0.097 clipped vs -0.860 unclipped).  With the
        candidate panel every plan of the panel lies inside the box, so no
        clipping occurs for any of them.
        """
        fast_plans_checked = 0
        for name, _order, pure, panel, mec_res, gre_res, _greedy in self.graphs:
            t_mec, e_mec = mec_res.makespan_seconds, mec_res.total_mobile_joules
            t_gre, e_gre = gre_res.makespan_seconds, gre_res.total_mobile_joules

            # 1) documented defect: the pure box is saturated at the floor
            self.assertAlmostEqual(j_report(t_mec, e_mec, pure), 0.0, places=9, msg=name)

            # 2) the fix: with the panel box every panel member is inside [0,1]
            for t, e in ((t_mec, e_mec), (t_gre, e_gre)):
                value = j_report(t, e, panel)
                self.assertGreaterEqual(value, 0.0, msg=name)
                self.assertLessEqual(value, 1.0, msg=name)

            # 3) and the panel actually bounds that plan (no normalization clip)
            if t_gre < t_mec:
                fast_plans_checked += 1
                self.assertGreaterEqual(t_gre, panel.L_min - 1e-9, msg=name)
                self.assertLessEqual(t_gre, panel.L_max + 1e-9, msg=name)
                self.assertGreaterEqual(e_gre, panel.E_min - 1e-9, msg=name)
                self.assertLessEqual(e_gre, panel.E_max + 1e-9, msg=name)

        self.assertGreater(fast_plans_checked, 0, "no graph had a faster mixed plan")

    def test_substantive_preference_is_pinned(self):
        """Under unclipped 0.5/0.5 the all-MEC plan is *legitimately* better.

        Pinning this keeps the record straight: the panel fixes the *metric*, it
        does not change the finding that the scalarized objective prefers all-MEC
        (energy dominates).  Changing that requires the constraint formulation /
        energy-scope fix, not a normalization change.
        """
        for name, _order, _pure, panel, mec_res, gre_res, _greedy in self.graphs:
            mec = j_report(mec_res.makespan_seconds, mec_res.total_mobile_joules, panel)
            gre = j_report(gre_res.makespan_seconds, gre_res.total_mobile_joules, panel)
            self.assertLess(mec, gre, msg=name)

    def test_panel_min_is_an_achievable_plan(self):
        for name, _order, _pure, panel, _mec, gre_res, _greedy in self.graphs:
            self.assertAlmostEqual(
                panel.L_panel_min, gre_res.makespan_seconds, places=6, msg=name
            )

    def test_invalid_mode_rejected(self):
        tg = OffloadingTaskGraph(str(self.files[0]))
        tg.prioritize_tasks(self.cluster)
        with self.assertRaises(ValueError):
            compute_reference_ranges(tg, self.res, mode="bogus")


if __name__ == "__main__":
    unittest.main(verbosity=2)
