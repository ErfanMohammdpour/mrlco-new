#!/usr/bin/env python3
"""Runtime mixed-criticality semantics: tier invariance, regimes, mode machine,
dependency-safe dropping, HI capping, HIGH preservation.

Everything here is numpy/stdlib only. The synthetic graph is built locally so the
suite passes even when the frozen dataset is absent; the dataset integration check
skips itself when `graphs.jsonl` is missing.
"""

from __future__ import annotations

import copy
import json
import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spec.automotive_training import mc_runtime as mc  # noqa: E402

DATASET_DIR = ROOT / "env" / "mec_offloaing_envs" / "data" / "automotive_mc_v1"
GRAPHS_PATH = DATASET_DIR / "graphs.jsonl"

TIERS = {"UE": mc.TIER_FREQ_HZ["UE"], "MEC": mc.TIER_FREQ_HZ["MEC"],
         "HELPER": mc.TIER_FREQ_HZ["HELPER"]}


def _tag_placement(realized: dict, placement: str) -> dict:
    return {tid: dict(rec, placement=placement) for tid, rec in realized.items()}


def _assert_finite(test: unittest.TestCase, obj, path: str = "root") -> None:
    if isinstance(obj, dict):
        for key, val in obj.items():
            _assert_finite(test, val, "%s.%s" % (path, key))
    elif isinstance(obj, (list, tuple)):
        for i, val in enumerate(obj):
            _assert_finite(test, val, "%s[%d]" % (path, i))
    elif isinstance(obj, float):
        test.assertTrue(math.isfinite(obj), "%s is not finite: %r" % (path, obj))
    elif isinstance(obj, bool) or obj is None:
        return
    elif isinstance(obj, int):
        return


class UncertaintyModelTest(unittest.TestCase):
    def test_yaml_is_self_describing_and_frozen(self):
        doc = mc.load_uncertainty()
        self.assertEqual(doc["schema_version"], mc.EXECUTION_UNCERTAINTY_VERSION)
        self.assertEqual(doc["model_id"], "execution_uncertainty_v1")
        self.assertEqual(doc["evidence_class"], "source_calibrated_synthetic")
        self.assertIs(doc["frozen_before_smoke"], True)
        self.assertEqual(set(doc["forbidden"]), {"tuning_after_smoke", "measured_claim"})
        self.assertEqual(len(doc["sha256"]), 64)
        regimes = doc["regimes"]
        self.assertGreaterEqual(len(regimes), 3)
        self.assertAlmostEqual(sum(float(r["probability"]) for r in regimes), 1.0, places=12)
        for r in regimes:
            for field in ("regime_id", "rule_id", "demand_band", "probability", "demand_scale"):
                self.assertIn(field, r, "regime missing %s" % field)
        bands = {r["regime_id"]: r["demand_band"] for r in regimes}
        self.assertIn("demand_lo_equiv", bands["nominal_below_lo"])
        self.assertIn("demand_hi_equiv", bands["overrun_to_hi"])
        self.assertIn("demand_hi_equiv", bands["overrun_above_hi"])
        seed = doc["seed_derivation"]
        self.assertIn("margo-mc|", seed["material_template"])
        self.assertIn("hexdigest()[:15]", seed["expression"])
        self.assertEqual(seed["prng"]["kind"], "sha256_counter")
        self.assertEqual(doc["declared_uncertainties"] and True, True)

    def test_load_uncertainty_sha_matches_file(self):
        import hashlib

        doc = mc.load_uncertainty()
        raw = mc.EXECUTION_UNCERTAINTY_PATH.read_bytes()
        self.assertEqual(doc["sha256"], hashlib.sha256(raw).hexdigest())

    def test_reference_model_matches_frozen_m5_v2(self):
        f_ref, xi = mc.reference_compute_model()
        self.assertAlmostEqual(xi, 300.0, places=9)
        self.assertAlmostEqual(f_ref, 2.30e9, places=3)

    def test_seed_derivation_matches_declared_expression(self):
        import hashlib

        for gid, tid, roll in (("g0", 0, 0), ("mcv1_x", 17, 424242), ("g", 19, 2 ** 40 + 3)):
            material = "margo-mc|%s|%s|%s" % (gid, tid, roll)
            expect = int(hashlib.sha256(material.encode()).hexdigest()[:15], 16)
            self.assertEqual(mc.demand_seed(gid, tid, roll), expect)


class TierInvarianceTest(unittest.TestCase):
    """ACCEPTANCE: the LO->HI trigger is a property of the demand vector alone."""

    def setUp(self):
        self.graph = mc.build_fixture_graph()
        self.realized = mc.draw_realized_demands(self.graph, rollout_seed=987654321)

    def test_same_demand_same_trigger_on_ue_mec_helper(self):
        decisions = {}
        durations = {}
        base = {tid: {"mode_at_task": None} for tid in self.realized}
        for tier, f_hz in TIERS.items():
            tagged = _tag_placement(self.realized, tier)
            # the demand vector itself must be identical under every placement
            for tid, rec in tagged.items():
                for field in ("demand_lo_equiv", "demand_hi_equiv", "realized_equiv",
                              "realized_seconds_ref", "regime"):
                    self.assertEqual(rec[field], self.realized[tid][field])
            result = mc.resolve_mode_and_execution(self.graph, tagged)
            decisions[tier] = result
            durations[tier] = {
                int(tid): rec["duration_seconds"]
                for tid, rec in result["tier_durations"].items()
                if "duration_seconds" in rec
            }
        reference = decisions["UE"]
        for tier in ("MEC", "HELPER"):
            self.assertEqual(decisions[tier]["switches"], reference["switches"],
                             "%s changed the switch set" % tier)
            self.assertEqual(decisions[tier]["final_mode"], reference["final_mode"])
            self.assertEqual(decisions[tier]["executed_task_ids"], reference["executed_task_ids"])
            self.assertEqual(decisions[tier]["dropped_task_ids"], reference["dropped_task_ids"])
            self.assertEqual(decisions[tier]["capped_to_hi"], reference["capped_to_hi"])
        # the same work really does take different wall-clock time per tier
        self.assertGreater(durations["UE"][2], durations["MEC"][2])
        self.assertGreater(durations["UE"][2], durations["HELPER"][2])
        ratio = durations["UE"][2] / durations["MEC"][2]
        self.assertAlmostEqual(ratio, TIERS["MEC"] / TIERS["UE"], places=9)

    def test_empty_or_none_placement_gives_identical_switches(self):
        plain = mc.resolve_mode_and_execution(self.graph, self.realized)
        for tier in TIERS:
            tagged = mc.resolve_mode_and_execution(self.graph, _tag_placement(self.realized, tier))
            self.assertEqual(tagged["switches"], plain["switches"])
        # without a placement the duration annotation is absent but the decisions are not
        for tid in plain["executed_task_ids"]:
            self.assertNotIn("duration_seconds", plain["tier_durations"][str(tid)])
        tagged = mc.resolve_mode_and_execution(self.graph, _tag_placement(self.realized, "MEC"))
        for tid in tagged["executed_task_ids"]:
            self.assertIn("duration_seconds", tagged["tier_durations"][str(tid)])
        self.assertEqual(
            mc.placements_from_result(tagged),
            {tid: "MEC" for tid in tagged["executed_task_ids"] + tagged["dropped_task_ids"]})
        self.assertEqual(mc.placements_from_result(plain), {})

    def test_trigger_signature_has_no_placement_input(self):
        import inspect

        params = list(inspect.signature(mc.lo_overrun_trigger).parameters)
        self.assertEqual(params, ["criticality", "realized_equiv", "demand_lo_equiv"])
        for token in ("placement", "tier", "location", "f_hz", "duration", "seconds"):
            self.assertNotIn(token, params)
        sig = inspect.signature(mc.resolve_mode_and_execution).parameters
        self.assertNotIn("tiers", sig)
        self.assertNotIn("placement", sig)

    def test_tier_duration_formula(self):
        f_ref, xi = mc.reference_compute_model()
        equiv = 12345.678
        for f_hz in TIERS.values():
            expect = 8.0 * xi * equiv / f_hz
            self.assertAlmostEqual(mc.tier_duration_seconds(equiv, f_hz, xi), expect, places=12)
        seconds = 0.00775
        self.assertAlmostEqual(
            mc.tier_duration_seconds(mc.equivalent_work(seconds, f_ref, xi), f_ref, xi),
            seconds, places=12)


class DrawDeterminismAndRegimeTest(unittest.TestCase):
    def setUp(self):
        self.graph = mc.build_fixture_graph()
        self.result = mc.resolve_mode_and_execution(
            self.graph, mc.draw_realized_demands(self.graph, rollout_seed=7))

    def test_same_seeds_identical_draws(self):
        a = mc.draw_realized_demands(self.graph, rollout_seed=31415)
        b = mc.draw_realized_demands(self.graph, rollout_seed=31415)
        self.assertEqual(a, b)
        # a fresh, uncached uncertainty document must reproduce the draws exactly
        c = mc.draw_realized_demands(self.graph, rollout_seed=31415,
                                     uncertainty=mc.load_uncertainty())
        self.assertEqual(a, c)

    def test_different_rollout_seed_different_draws(self):
        a = mc.draw_realized_demands(self.graph, rollout_seed=1)
        b = mc.draw_realized_demands(self.graph, rollout_seed=2)
        self.assertNotEqual(a, b)
        differing = [tid for tid in a if a[tid]["realized_equiv"] != b[tid]["realized_equiv"]]
        self.assertTrue(differing)

    def test_all_three_regimes_are_reachable(self):
        seen = set()
        for seed in range(64):
            for rec in mc.draw_realized_demands(self.graph, rollout_seed=seed).values():
                seen.add(rec["regime"])
        self.assertEqual(
            seen, {"nominal_below_lo", "overrun_to_hi", "overrun_above_hi"},
            "fixture population must exercise all three regimes: %r" % sorted(seen))

    def test_regime_bands_are_respected(self):
        for seed in range(24):
            for tid, rec in mc.draw_realized_demands(self.graph, rollout_seed=seed).items():
                lo = rec["demand_lo_equiv"]
                hi = rec["demand_hi_equiv"]
                obs = rec["realized_equiv"]
                regime = rec["regime"]
                self.assertGreater(obs, 0.0)
                if regime == "nominal_below_lo":
                    self.assertLessEqual(obs, lo)
                elif regime == "overrun_to_hi":
                    self.assertGreater(obs, lo)
                    self.assertIsNotNone(hi, "C_HI-less task landed in the to-C_HI regime")
                    self.assertLessEqual(obs, hi)
                else:
                    self.assertIsNotNone(hi)
                    self.assertGreater(obs, hi)
                    self.assertEqual(self.graph.task_by_id[tid].criticality, "HIGH")

    def test_non_high_never_enters_the_above_hi_regime(self):
        for seed in range(64):
            for tid, rec in mc.draw_realized_demands(self.graph, rollout_seed=seed).items():
                if self.graph.task_by_id[tid].criticality != "HIGH":
                    self.assertNotEqual(rec["regime"], "overrun_above_hi")
                    self.assertIsNone(rec["demand_hi_equiv"])

    def test_regime_frequencies_roughly_match_declared_probabilities(self):
        doc = mc.load_uncertainty()
        declared = {r["regime_id"]: float(r["probability"]) for r in doc["regimes"]}
        counts = {k: 0 for k in declared}
        total = 0
        for seed in range(300):
            for rec in mc.draw_realized_demands(self.graph, rollout_seed=seed).values():
                counts[rec["regime"]] += 1
                total += 1
        # LOW/MEDIUM renormalisation moves mass out of the overrun regime, so only a
        # loose sanity band is asserted here (the exact draw is checked elsewhere).
        self.assertGreater(counts["nominal_below_lo"] / total, 0.45)
        self.assertLess(counts["nominal_below_lo"] / total, 0.95)
        self.assertGreater(counts["overrun_to_hi"] / total, 0.02)

    def test_no_nan_or_inf_anywhere(self):
        _assert_finite(self, mc.draw_realized_demands(self.graph, rollout_seed=99))
        _assert_finite(self, self.result)

    def test_realized_seconds_ref_is_consistent_with_equivalent_work(self):
        f_ref, xi = mc.reference_compute_model()
        for rec in self.result["execution"].values():
            self.assertAlmostEqual(
                mc.tier_duration_seconds(rec["realized_equiv"], f_ref, xi),
                rec["realized_equiv"] * 8.0 * xi / f_ref, places=12)


class ModeMachineTest(unittest.TestCase):
    def test_initial_mode_is_lo_and_low_demand_stays_lo(self):
        result = mc.mode_switch_fixtures()["below_clo"]["resolved_result"]
        self.assertEqual(result["initial_mode"], "LO")
        self.assertEqual(result["final_mode"], "LO")
        self.assertEqual(result["switches"], [])
        mc.assert_high_preserved(result, mc.build_fixture_graph())

    def test_lo_to_hi_on_high_demand_above_clo(self):
        fixture = mc.mode_switch_fixtures()["at_or_below_chi"]
        result = fixture["resolved_result"]
        self.assertEqual(result["initial_mode"], "LO")
        self.assertEqual(result["final_mode"], "HI")
        self.assertEqual(len(result["switches"]), 1)
        self.assertEqual(result["switches"][0]["triggering_task_id"], fixture["first_high_task_id"])
        self.assertEqual(result["switches"][0]["previous_mode"], "LO")
        self.assertEqual(result["switches"][0]["new_mode"], "HI")

    def test_above_chi_also_switches_and_is_capped_in_hi(self):
        fixture = mc.mode_switch_fixtures()["above_chi"]
        result = fixture["resolved_result"]
        self.assertEqual(result["final_mode"], "HI")
        first = fixture["first_high_task_id"]
        self.assertEqual(result["switches"][0]["triggering_task_id"], first)
        self.assertIn(first, result["capped_to_hi"])
        self.assertTrue(result["execution"][first]["capped_to_hi"])
        self.assertAlmostEqual(
            result["execution"][first]["effective_equiv"],
            result["execution"][first]["demand_hi_equiv"], places=9)

    def test_fixtures_force_the_intended_regime_for_the_first_high_task(self):
        fixtures = mc.mode_switch_fixtures()
        below = fixtures["below_clo"]
        to_hi = fixtures["at_or_below_chi"]
        above = fixtures["above_chi"]
        for fixture in (below, to_hi, above):
            self.assertEqual(fixture["expected_initial_mode"], "LO")
            self.assertEqual(fixture["first_high_task_id"], 2)
        for fixture, expected_regime, should_switch in (
            (below, "nominal_below_lo", False),
            (to_hi, "overrun_to_hi", True),
            (above, "overrun_above_hi", True),
        ):
            tid = fixture["first_high_task_id"]
            rec = fixture["realized"][tid]
            self.assertEqual(rec["regime"], expected_regime)
            if should_switch:
                self.assertGreater(rec["realized_equiv"], rec["demand_lo_equiv"])
                self.assertEqual(fixture["expected_final_mode"], "HI")
                self.assertEqual(fixture["triggering_task_id"], tid)
                self.assertEqual(fixture["resolved_result"]["switches"][0]["triggering_task_id"], tid)
            else:
                self.assertLessEqual(rec["realized_equiv"], rec["demand_lo_equiv"])
                self.assertEqual(fixture["expected_final_mode"], "LO")
                self.assertIsNone(fixture["triggering_task_id"])
                self.assertEqual(fixture["switch_count"], 0)
        # above C_HI really is above the HI budget; at-or-below C_HI really is inside it
        above_hi_tid = above["first_high_task_id"]
        self.assertGreater(above["realized"][above_hi_tid]["realized_equiv"],
                           above["realized"][above_hi_tid]["demand_hi_equiv"])
        to_hi_tid = to_hi["first_high_task_id"]
        self.assertLessEqual(to_hi["realized"][to_hi_tid]["realized_equiv"],
                             to_hi["realized"][to_hi_tid]["demand_hi_equiv"])

    def test_fixture_dropped_and_executed_sets_are_independently_specified(self):
        """Independent expectations, not the engine's own closure computation."""
        fixtures = mc.mode_switch_fixtures()
        self.assertEqual(fixtures["below_clo"]["resolved_result"]["executed_task_ids"],
                         [0, 1, 2, 3, 4, 5])
        self.assertEqual(fixtures["below_clo"]["resolved_result"]["dropped_task_ids"], [])
        for key in ("at_or_below_chi", "above_chi", "required_low_ancestor"):
            # LOW #0 is a droppable task by policy, but it is the required producer of
            # HIGH #2, so dependency-safety keeps it; only the unneeded LOW #3 is dropped.
            self.assertEqual(fixtures[key]["resolved_result"]["executed_task_ids"], [0, 1, 2, 4, 5])
            self.assertEqual(fixtures[key]["resolved_result"]["dropped_task_ids"], [3])
        self.assertEqual(fixtures["required_low_ancestor"]["expected_executed_task_ids"],
                         [0, 1, 2, 4, 5])
        self.assertEqual(fixtures["required_low_ancestor"]["expected_dropped_task_ids"], [3])

    def test_non_high_overrun_does_not_switch(self):
        graph = mc.build_fixture_graph()
        realized = mc.realized_fixture(graph, below=[t.task_id for t in graph.tasks if t.criticality == "HIGH"])
        # push a MEDIUM task into the to-C_HI band by hand (no C_HI required for MEDIUM)
        medium = [t.task_id for t in graph.tasks if t.criticality == "MEDIUM"][0]
        rec = realized[medium]
        rec["realized_equiv"] = 3.0 * rec["demand_lo_equiv"]
        rec["regime"] = "overrun_to_hi"
        result = mc.resolve_mode_and_execution(graph, realized)
        self.assertEqual(result["final_mode"], "LO")
        self.assertEqual(result["switches"], [])

    def test_switch_log_fields_are_exactly_the_nine_mandated(self):
        result = mc.mode_switch_fixtures()["at_or_below_chi"]["resolved_result"]
        for switch in result["switches"]:
            self.assertEqual(set(switch), set(mc.SWITCH_LOG_FIELDS))
            self.assertEqual(switch["criticality"], "HIGH")
            self.assertIn(switch["reason"], ("high_task_observed_execution_exceeds_empirical_execution_budget_lo",))
            self.assertIsInstance(switch["logical_schedule_point"], int)
            self.assertGreaterEqual(switch["logical_schedule_point"], 0)
            self.assertGreater(switch["observed_execution"], switch["C_LO"])
        positions = sorted(s["logical_schedule_point"] for s in result["switches"])
        self.assertEqual(positions, sorted(set(positions)))

    def test_switch_records_the_task_position_in_the_plan_order(self):
        graph = mc.build_fixture_graph()
        order = mc.plan_order_ids(graph)
        realized = mc.realized_fixture(graph, to_hi=[2], below=[4])
        for plan_order in (order, tuple(reversed(order))):
            if plan_order is not order:
                continue  # only topological orders are legal
            result = mc.resolve_mode_and_execution(graph, realized, plan_order)
            switch = result["switches"][0]
            self.assertEqual(switch["logical_schedule_point"], plan_order.index(switch["triggering_task_id"]))

    def test_hi_is_sticky_a_later_nominal_task_cannot_leave_hi(self):
        graph = mc.build_fixture_graph()
        # HIGH #2 overruns; HIGH #4 then draws nominal demand, strictly below its C_LO
        realized = mc.realized_fixture(graph, above_hi=[2], below=[0, 1, 3, 4, 5])
        result = mc.resolve_mode_and_execution(graph, realized)
        self.assertEqual(result["final_mode"], "HI")
        self.assertGreater(result["execution"][4]["realized_equiv"], 0.0)
        self.assertLess(result["execution"][4]["realized_equiv"], result["execution"][4]["demand_lo_equiv"])
        self.assertEqual(result["execution"][4]["mode_at_task"], "HI")
        self.assertEqual(result["initial_mode"], "LO")
        self.assertTrue(result["no_action_replanning_after_switch"])
        self.assertTrue(result["open_loop_fixed_placement_plan"])
        self.assertNotIn("LO", [s["new_mode"] for s in result["switches"]])

    def test_plan_order_must_be_topological(self):
        graph = mc.build_fixture_graph()
        realized = mc.realized_fixture(graph, to_hi=[2], below=[4])
        with self.assertRaises(mc.MCRuntimeError):
            mc.resolve_mode_and_execution(graph, realized, plan_order=[2, 1, 0, 3, 4, 5])
        with self.assertRaises(mc.MCRuntimeError):
            mc.resolve_mode_and_execution(graph, realized, plan_order=[0, 1, 2])


class DependencySafeDroppingTest(unittest.TestCase):
    def test_required_ancestor_low_survives_while_unneeded_low_is_dropped(self):
        fixture = mc.mode_switch_fixtures()["required_low_ancestor"]
        result = fixture["resolved_result"]
        self.assertEqual(result["final_mode"], "HI")
        ancestor = fixture["required_ancestor_task_id"]
        unneeded = fixture["unneeded_droppable_task_id"]
        graph = fixture["graph"]
        self.assertEqual(graph.task_by_id[ancestor].criticality, "LOW")
        self.assertTrue(graph.task_by_id[ancestor].drop_allowed_hi_mode)
        self.assertIn(ancestor, result["required_closure_task_ids"])
        self.assertIn(ancestor, result["executed_task_ids"])
        self.assertNotIn(ancestor, result["dropped_task_ids"])
        self.assertNotIn(unneeded, result["required_closure_task_ids"])
        self.assertIn(unneeded, result["dropped_task_ids"])
        self.assertEqual(fixture["expected_dropped_task_ids"], [unneeded])
        self.assertEqual(result["dropped_task_ids"], fixture["expected_dropped_task_ids"])

    def test_medium_is_never_dropped_even_when_the_policy_allows_degrade(self):
        fixture = mc.mode_switch_fixtures()["above_chi"]
        graph = fixture["graph"]
        result = fixture["resolved_result"]
        for task in graph.tasks:
            if task.criticality == "MEDIUM":
                self.assertIn(task.task_id, result["executed_task_ids"])
                self.assertNotIn(task.task_id, result["dropped_task_ids"])
                self.assertTrue(task.drop_allowed_hi_mode is False)

    def test_surviving_sub_dag_is_acyclic_with_all_predecessors_present(self):
        for fixture in mc.mode_switch_fixtures().values():
            if not isinstance(fixture, dict) or "resolved_result" not in fixture:
                continue
            graph = fixture["graph"]
            result = fixture["resolved_result"]
            check = mc.check_sub_dag(graph, result["executed_task_ids"])
            self.assertTrue(check["acyclic"])
            self.assertTrue(check["closure_complete"])
            for tid in result["executed_task_ids"]:
                for pred in graph.task_by_id[tid].predecessors:
                    self.assertIn(pred, result["executed_task_ids"])
            self.assertEqual(
                sorted(result["executed_task_ids"] + result["dropped_task_ids"]),
                sorted(t.task_id for t in graph.tasks))

    def test_dropped_set_is_sorted_and_disjoint_from_executed(self):
        for fixture in mc.mode_switch_fixtures().values():
            if not isinstance(fixture, dict) or "resolved_result" not in fixture:
                continue
            result = fixture["resolved_result"]
            self.assertEqual(result["executed_task_ids"], sorted(result["executed_task_ids"]))
            self.assertEqual(result["dropped_task_ids"], sorted(result["dropped_task_ids"]))
            self.assertFalse(set(result["executed_task_ids"]) & set(result["dropped_task_ids"]))

    def test_low_droppable_only_in_hi_mode(self):
        graph = mc.build_fixture_graph()
        realized = mc.realized_fixture(graph, below=[t.task_id for t in graph.tasks])
        result = mc.resolve_mode_and_execution(graph, realized)
        self.assertEqual(result["final_mode"], "LO")
        self.assertEqual(result["dropped_task_ids"], [])


class HighPreservationTest(unittest.TestCase):
    def setUp(self):
        self.fixtures = mc.mode_switch_fixtures()
        self.result = self.fixtures["above_chi"]["resolved_result"]
        self.graph = self.fixtures["above_chi"]["graph"]

    def test_high_tasks_are_executed_and_never_degraded(self):
        for task in self.graph.tasks:
            if task.criticality != "HIGH":
                continue
            self.assertIn(task.task_id, self.result["executed_task_ids"])
            self.assertNotIn(task.task_id, self.result["dropped_task_ids"])
            self.assertIn(self.result["degradation_status"][str(task.task_id)],
                          (mc.DEGRADATION_INTACT, mc.DEGRADATION_UNSUPPORTED))
            self.assertFalse(self.result["degrade_applied"][str(task.task_id)])
        self.assertTrue(self.result["high_preserved"])
        mc.assert_high_preserved(self.result, self.graph)

    def test_no_degradation_is_invented(self):
        for tid, status in self.result["degradation_status"].items():
            self.assertIn(status, (mc.DEGRADATION_INTACT, mc.DEGRADATION_UNSUPPORTED))
            if status == mc.DEGRADATION_UNSUPPORTED:
                self.assertFalse(self.result["degrade_applied"][tid])
                self.assertTrue(self.graph.task_by_id[int(tid)].degrade_allowed("HI"))

    def test_assert_high_preserved_raises_on_corrupted_result(self):
        dropped = copy.deepcopy(self.result)
        victim = dropped["high_task_ids"][0]
        dropped["executed_task_ids"] = [t for t in dropped["executed_task_ids"] if t != victim]
        dropped["dropped_task_ids"] = sorted(dropped["dropped_task_ids"] + [victim])
        with self.assertRaises(mc.MCRuntimeError):
            mc.assert_high_preserved(dropped, self.graph)

        degraded = copy.deepcopy(self.result)
        degraded["degradation_status"][str(victim)] = "degraded"
        degraded["degrade_applied"][str(victim)] = True
        with self.assertRaises(mc.MCRuntimeError):
            mc.assert_high_preserved(degraded, self.graph)

        flag = copy.deepcopy(self.result)
        flag["high_preserved"] = False
        with self.assertRaises(mc.MCRuntimeError):
            mc.assert_high_preserved(flag, self.graph)

    def test_high_capping_only_happens_in_hi_mode_and_respects_chi(self):
        graph = self.graph
        for tid, rec in self.result["execution"].items():
            task = graph.task_by_id[int(tid)]
            if task.criticality == "HIGH":
                self.assertIsNotNone(task.empirical_execution_budget_hi_s)
                if rec["capped_to_hi"]:
                    self.assertEqual(rec["mode_at_task"], "HI")
                    self.assertAlmostEqual(rec["effective_equiv"], rec["demand_hi_equiv"], places=9)
                self.assertLessEqual(rec["effective_equiv"], rec["demand_hi_equiv"] + 1e-9)
            else:
                self.assertIsNone(task.empirical_execution_budget_hi_s)
                self.assertFalse(rec["capped_to_hi"])
                self.assertEqual(rec["effective_equiv"], rec["realized_equiv"])

    def test_capped_list_matches_execution_flags(self):
        flagged = sorted(int(tid) for tid, rec in self.result["execution"].items() if rec["capped_to_hi"])
        self.assertEqual(self.result["capped_to_hi"], flagged)


class RealDatasetIntegrationTest(unittest.TestCase):
    """Integration over 3 real frozen graphs, read directly with json.loads."""

    def _load_graphs(self, limit: int = 3):
        from spec.automotive_training.automotive_loader import AutomotiveGraph, AutomotiveEdge, AutomotiveTask

        records = []
        with GRAPHS_PATH.open() as handle:
            for i, line in enumerate(handle):
                if i >= limit:
                    break
                records.append(json.loads(line))
        graphs = []
        for rec in records:
            tasks = tuple(AutomotiveTask(
                task_id=int(t["task_id"]), semantic_role=str(t["semantic_role"]),
                motif_id=str(t["motif_id"]), lineage_id=str(t["lineage_id"]),
                criticality=str(t["criticality"]), safety_scope=str(t.get("safety_scope", "")),
                compute_workload_bytes=int(t["compute_workload_bytes"]),
                t_ref_s=float(t["t_ref_s"]), task_output_bytes=int(t["task_output_bytes"]),
                external_input_bytes=int(t.get("external_input_bytes", 0)),
                output_payload_class=t.get("output_payload_class"),
                empirical_execution_budget_lo_s=float(t["empirical_execution_budget_lo_s"]),
                empirical_execution_budget_hi_s=(None if t.get("empirical_execution_budget_hi_s") is None
                                                 else float(t["empirical_execution_budget_hi_s"])),
                budget_hi_applicable=bool(t["budget_hi_applicable"]),
                budget_hi_status=str(t["budget_hi_status"]),
                drop_allowed_lo_mode=bool(t["drop_allowed_lo_mode"]),
                drop_allowed_hi_mode=bool(t["drop_allowed_hi_mode"]),
                degrade_allowed_lo_mode=bool(t["degrade_allowed_lo_mode"]),
                degrade_allowed_hi_mode=bool(t["degrade_allowed_hi_mode"]),
                E_s=float(t["E_s"]), L_s=float(t["L_s"]), deadline_s=float(t["deadline_s"]),
                slack_s=float(t["slack_s"]), deadline_type=str(t.get("deadline_type", "firm")),
                tardiness_weight=float(t.get("tardiness_weight", 1.0)),
                is_root=bool(t["is_root"]), is_sink=bool(t["is_sink"]),
                predecessors=tuple(int(x) for x in (t.get("predecessors") or [])),
                successors=tuple(int(x) for x in (t.get("successors") or [])),
            ) for t in rec["tasks"])
            graphs.append(AutomotiveGraph(
                graph_id=str(rec["graph_id"]), split="meta_train", role_in_split="train",
                application_family=str(rec["application_family"]),
                safety_scope=str(rec.get("safety_scope", "")), template_id=str(rec["template_id"]),
                template_lineage=str(rec["template_lineage"]),
                semantic_signature=str(rec["semantic_signature"]),
                topology_regime=str(rec["topology_regime"]),
                topology_signature=str(rec["topology_signature"]),
                workload_regime=str(rec["workload_regime"]),
                resource_profile=str(rec["resource_profile"]),
                resource_level=str(rec["resource_level"]),
                resource={k: float(v) for k, v in rec["resource"].items()},
                sla_id=str(rec["sla_id"]), sla_regime=str(rec["sla_regime"]),
                sla_class=rec.get("sla_class"), P_f_s=float(rec["P_f_s"]), D_G_s=float(rec["D_G_s"]),
                criticality_policy_id=str(rec["criticality_policy_id"]),
                criticality_counts={k: int(v) for k, v in rec["criticality_counts"].items()},
                criticality_mixture=str(rec["criticality_mixture"]),
                parent_seed=int(rec["parent_seed"]), graph_seed=int(rec["graph_seed"]),
                cell_id=str(rec["cell_id"]), canonical_sha256=str(rec["canonical_sha256"]),
                raw_sha256=str(rec["raw_sha256"]), tasks=tasks,
                edges=tuple(AutomotiveEdge(src=int(e["src"]), dst=int(e["dst"]),
                                           payload_class=str(e["payload_class"]),
                                           payload_model_ref=str(e["payload_model_ref"]),
                                           payload_bytes=int(e["payload_bytes"]),
                                           payload_rule_id=str(e["payload_rule_id"]),
                                           payload_evidence_class=str(e["payload_evidence_class"]))
                            for e in rec["edges"]),
                mode_semantics=dict(rec["mode_semantics"]), provenance_refs=dict(rec.get("provenance") or {}),
                certification={}, dataset_manifest={},
            ))
        return graphs

    @unittest.skipUnless(GRAPHS_PATH.exists(), "frozen dataset not materialized")
    def test_three_real_graphs_end_to_end(self):
        graphs = self._load_graphs(3)
        self.assertEqual(len(graphs), 3)
        # the overrun regime is ~5% of HIGH draws, so coverage is asserted over a
        # handful of rollout seeds per graph (still one draw per task per rollout).
        coverage_seeds = (101, 202, 303, 404, 505, 606, 707, 808)
        for graph in graphs:
            self.assertTrue(graph.high_task_ids())
            regimes_seen = set()
            for seed in coverage_seeds:
                regimes_seen.update(r["regime"] for r in mc.draw_realized_demands(graph, seed).values())
            self.assertEqual(regimes_seen,
                             {"nominal_below_lo", "overrun_to_hi", "overrun_above_hi"},
                             "%s does not reach every regime" % graph.graph_id)
            demands_a = mc.draw_realized_demands(graph, rollout_seed=101)
            demands_b = mc.draw_realized_demands(graph, rollout_seed=101)
            self.assertEqual(demands_a, demands_b)
            self.assertNotEqual(demands_a, mc.draw_realized_demands(graph, rollout_seed=202))
            for tid, rec in demands_a.items():
                task = graph.task_by_id[tid]
                expected_lo = mc.equivalent_work(task.empirical_execution_budget_lo_s, *mc.reference_compute_model())
                self.assertAlmostEqual(rec["demand_lo_equiv"], expected_lo, places=6)
                if task.criticality == "HIGH":
                    self.assertIsNotNone(rec["demand_hi_equiv"])
                    self.assertGreaterEqual(rec["demand_hi_equiv"], rec["demand_lo_equiv"])
                else:
                    self.assertIsNone(rec["demand_hi_equiv"])
            for tier in TIERS:
                result = mc.resolve_mode_and_execution(graph, _tag_placement(demands_a, tier))
                mc.assert_high_preserved(result, graph)
                self.assertTrue(result["high_preserved"])
                self.assertTrue(result["open_loop_fixed_placement_plan"])
                self.assertTrue(result["no_action_replanning_after_switch"])
                self.assertEqual(result["initial_mode"], "LO")
                self.assertIn(result["final_mode"], ("LO", "HI"))
                for high in graph.high_task_ids():
                    self.assertIn(high, result["executed_task_ids"])
                    self.assertNotIn(high, result["dropped_task_ids"])
                for tid, rec in result["execution"].items():
                    if rec["capped_to_hi"]:
                        self.assertEqual(graph.task_by_id[int(tid)].criticality, "HIGH")
                        self.assertEqual(rec["mode_at_task"], "HI")
                check = mc.check_sub_dag(graph, result["executed_task_ids"])
                self.assertTrue(check["acyclic"])
                _assert_finite(self, mc.draw_realized_demands(graph, rollout_seed=303))

    @unittest.skipUnless(GRAPHS_PATH.exists(), "frozen dataset not materialized")
    def test_real_graph_m7_policy_is_read_not_hard_coded(self):
        graph = self._load_graphs(1)[0]
        family = graph.mode_semantics["drop_degrade_policy"]
        for task in graph.tasks:
            policy = mc.task_policy(graph, task, "HI")
            self.assertEqual(policy["drop_allowed"], bool(task.drop_allowed_hi_mode))
            self.assertEqual(policy["degrade_allowed"], bool(task.degrade_allowed_hi_mode))
            self.assertEqual(policy["drop_allowed"], bool(family[task.criticality]["drop_allowed_hi_mode"]))


if __name__ == "__main__":
    unittest.main()
