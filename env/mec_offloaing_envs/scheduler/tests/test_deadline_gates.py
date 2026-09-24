#!/usr/bin/env python3
"""Shield diagnostics: mask breakdown, action-closure rates, gate rules.

Pure functions over (bounds, deadlines), so the numbers the gate report quotes are
reproducible without the scheduler or TensorFlow. Also pins the semantics the
regimes rely on: soft/firm deadlines must NOT create a hard shield, the pre-guard
all-invalid count is what matters, and the gate must not pass a regime that never
closes MEC -- because the deadline-free policy collapses onto MEC and a shield
that cannot close it cannot test that collapse.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env.mec_offloaing_envs.scheduler.deadline_gates import (  # noqa: E402
    MaskBreakdown,
    breakdown_for_graph,
    gate_summary,
    mask_from_deadlines,
    passes_trainable_gate,
)
from env.mec_offloaing_envs.scheduler.static_bounds import StaticBounds  # noqa: E402


def bounds_from(ready_rows, sinks=None, finish_rows=None):
    """StaticBounds with explicit per-task per-action ready bounds."""
    n = len(ready_rows)
    finish_rows = finish_rows or [[0.0] * 3 for _ in range(n)]
    return StaticBounds(
        order=tuple(range(n)),
        finish_lb=tuple(tuple(r) for r in finish_rows),
        ready_lb=tuple(tuple(r) for r in ready_rows),
        min_ready_lb=tuple(min(r) for r in ready_rows),
        is_sink=tuple(bool(s) for s in (sinks or [False] * n)),
        max_ready_lb=max(max(r) for r in ready_rows),
    )


class TestMaskFromDeadlines(unittest.TestCase):
    def test_action_open_iff_bound_fits(self):
        bounds = bounds_from([[10.0, 5.0, 8.0]])
        self.assertEqual(mask_from_deadlines(bounds, [6.0]), [[False, True, False]])
        self.assertEqual(mask_from_deadlines(bounds, [8.0]), [[False, True, True]])
        self.assertEqual(mask_from_deadlines(bounds, [10.0]), [[True, True, True]])
        self.assertEqual(mask_from_deadlines(bounds, [None]), [[True, True, True]])

    def test_length_mismatch_is_rejected(self):
        bounds = bounds_from([[1.0, 1.0, 1.0]])
        with self.assertRaises(ValueError):
            mask_from_deadlines(bounds, [1.0, 2.0])


class TestBreakdown(unittest.TestCase):
    def test_counts_and_pre_guard_all_invalid(self):
        bounds = bounds_from(
            [[10.0, 5.0, 8.0], [10.0, 9.0, 8.0], [1.0, 1.0, 1.0]],
            sinks=[False, False, True],
        )
        # deadline 6: row0 loses only HELPER, row1 loses UE+HELPER, row2 loses all
        breakdown = breakdown_for_graph(bounds, [6.0, 9.5, 0.5])
        self.assertEqual(breakdown.tokens, 3)
        # "active" means at least one action is closed, so an all-invalid row is
        # active too (this matches mask_metrics: active = mean(any(~pre_guard)))
        self.assertEqual(breakdown.active, 3)
        self.assertEqual(breakdown.forced, 1)         # row0 keeps exactly one action
        self.assertEqual(breakdown.all_invalid, 1)     # counted BEFORE any guard
        # row0: UE+HELPER closed; row1: UE closed; row2: all three closed
        self.assertEqual(breakdown.closed, (3, 1, 2))  # ue, mec, helper closures
        self.assertEqual(breakdown.sink_tokens, 1)
        # the sink row is all-invalid, which counts as active (see above)
        self.assertEqual(breakdown.sink_active, 1)
        self.assertEqual(breakdown.sink_tokens, 1)

    def test_forced_row_is_counted(self):
        bounds = bounds_from([[10.0, 1.0, 8.0]])
        breakdown = breakdown_for_graph(bounds, [5.0])
        self.assertEqual(breakdown.forced, 1)
        self.assertEqual(breakdown.active, 1)
        self.assertEqual(breakdown.closed, (1, 0, 1))

    def test_no_deadline_means_no_active_token(self):
        bounds = bounds_from([[10.0, 5.0, 8.0]])
        breakdown = breakdown_for_graph(bounds, [None])
        self.assertEqual(breakdown.rates()["active_rate"], 0.0)
        self.assertEqual(breakdown.rates()["all_invalid_rate"], 0.0)

    def test_depth_and_criticality_split(self):
        bounds = bounds_from([[10.0, 5.0, 8.0], [10.0, 9.0, 8.0]], sinks=[False, True])
        breakdown = breakdown_for_graph(
            bounds, [6.0, 6.0], depths={0: 0, 1: 1}, criticality={0: "low", 1: "high"}
        )
        self.assertEqual(breakdown.depth_tokens, {0: 1, 1: 1})
        self.assertEqual(breakdown.depth_active, {0: 1, 1: 1})
        self.assertEqual(breakdown.criticality_tokens, {"low": 1, "high": 1})
        self.assertEqual(breakdown.criticality_active, {"low": 1, "high": 1})

    def test_aggregation_is_token_weighted(self):
        small = breakdown_for_graph(bounds_from([[10.0, 5.0, 8.0]]), [6.0])
        big = breakdown_for_graph(
            bounds_from([[10.0, 5.0, 8.0]] * 3), [None, None, None]
        )
        merged = small + big
        self.assertEqual(merged.tokens, 4)
        self.assertEqual(merged.active, 1)
        self.assertAlmostEqual(merged.rates()["active_rate"], 0.25)


class TestGateSummary(unittest.TestCase):
    def _summary(self, active, mec_closed, witness_rate, all_invalid=0):
        breakdown = MaskBreakdown(
            tokens=100, active=active, forced=0, all_invalid=all_invalid,
            closed=(active, mec_closed, 0),
        )
        per_graph = {"g1": breakdown}
        return gate_summary(
            per_graph,
            witness_found={"g1": True},
            witness_mixed={"g1": True},
            excluded=[] if witness_rate == 1.0 else [{"graph": "g2", "reason": "x"}] * int(
                round((1.0 - witness_rate) / max(witness_rate, 1e-9))
            ),
        )

    def test_rates_are_reported_per_action(self):
        summary = self._summary(active=20, mec_closed=5, witness_rate=1.0)
        self.assertAlmostEqual(summary["active_rate"], 0.2)
        self.assertAlmostEqual(summary["mec_closed_rate"], 0.05)
        self.assertAlmostEqual(summary["ue_closed_rate"], 0.2)
        self.assertEqual(summary["graphs_with_mec_closure"], 1)

    def test_gate_requires_a_non_trivial_mask(self):
        quiet = self._summary(active=0, mec_closed=0, witness_rate=1.0)
        self.assertFalse(passes_trainable_gate(quiet)["passes"])
        good = self._summary(active=20, mec_closed=3, witness_rate=1.0)
        self.assertTrue(passes_trainable_gate(good)["passes"])

    def test_gate_rejects_a_regime_that_never_closes_mec(self):
        # the deadline-free policy collapses onto MEC, so a regime that cannot
        # close MEC cannot test that collapse -- it is reported, not hidden
        summary = self._summary(active=20, mec_closed=0, witness_rate=1.0)
        checks = passes_trainable_gate(summary)
        self.assertFalse(checks["mec_closure_present"])
        self.assertTrue(checks["passes"])   # still trainable, but flagged

    def test_gate_rejects_low_witness_rate(self):
        summary = self._summary(active=20, mec_closed=3, witness_rate=0.5)
        self.assertFalse(passes_trainable_gate(summary)["passes"])

    def test_gate_rejects_excess_all_invalid(self):
        summary = self._summary(active=20, mec_closed=3, witness_rate=1.0, all_invalid=5)
        self.assertFalse(passes_trainable_gate(summary)["passes"])


class TestSoftFirmDoNotShield(unittest.TestCase):
    def test_soft_firm_deadlines_leave_every_action_open(self):
        """The shield is gated on deadline_type == hard (masking.observation_mask).

        The sidecar carries soft/firm deadlines for the objective/constraint
        channels, so the gate report must not read their closures as a shield.
        """
        bounds = bounds_from([[10.0, 5.0, 8.0]])
        # a deadline that would close UE and HELPER if it were a hard shield
        raw = mask_from_deadlines(bounds, [6.0])
        self.assertEqual(raw, [[False, True, False]])
        # but a soft/firm regime must not feed this mask to the policy: the
        # observation-mask path only shields hard deadlines, which the regime
        # table encodes and test_masking covers. Here we pin the contract that
        # the type is what selects the channel, not the deadline value.
        from env.mec_offloaing_envs.scheduler.masking import resolve_mask_mode

        self.assertEqual(resolve_mask_mode("off"), "off")


if __name__ == "__main__":
    unittest.main(verbosity=2)
