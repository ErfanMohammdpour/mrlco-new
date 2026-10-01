#!/usr/bin/env python3
"""REV2 adversarial tests: lower bound, motif granularity, axis separation."""

from __future__ import annotations

import copy
import importlib.util
import sys
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "spec" / "automotive_mc_v1"
sys.path.insert(0, str(ROOT))


def _mod(name):
    spec = importlib.util.spec_from_file_location(name, BASE / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def _docs():
    return (yaml.safe_load((BASE / "workload_model_v2.yaml").read_text()),
            yaml.safe_load((BASE / "sla_registry_v2.yaml").read_text()),
            yaml.safe_load((BASE / "application_templates.yaml").read_text())["templates"],
            yaml.safe_load((BASE / "task_semantics.yaml").read_text()))


class TestREV2Invariants(unittest.TestCase):
    def setUp(self):
        self.v2 = _mod("deadline_model_v2")
        self.w2, self.s2, self.templates, self.semantics = _docs()
        self.tpl = self.templates[0]
        self.fam = self.tpl["family_id"]

    def test_co_locatable_edges_have_zero_lower_bound(self):
        lb, ref = self.v2.edge_lower_bound_costs(self.tpl, self.w2)
        self.assertTrue(all(q == 0.0 for q in lb.values()))
        self.assertTrue(all(q > 0.0 for q in ref.values()))   # reference scenario is not zero

    def test_non_co_locatable_edge_uses_minimum_legal_cost(self):
        w = copy.deepcopy(self.w2)
        w["payload_models"]["object_list"]["allowed_co_location"] = False
        w["payload_models"]["object_list"]["r_max_allowed_bps"] = 4.0e6
        lb, _r = self.v2.edge_lower_bound_costs(self.tpl, w)
        expected = 8.0 * 2000 / 4.0e6
        self.assertTrue(any(abs(q - expected) < 1e-15 for q in lb.values()))

    def test_reference_scenario_cannot_change_feasibility(self):
        base = self.v2.generate_deadlines_v2(self.tpl, self.w2, self.s2, self.fam)
        w = copy.deepcopy(self.w2)
        w["reference_scenario"]["rate_bps"] = 1.0e6
        probe = self.v2.generate_deadlines_v2(self.tpl, w, self.s2, self.fam)
        self.assertEqual(base.CP_ref, probe.CP_ref)
        self.assertNotEqual(base.CP_reference_scenario, probe.CP_reference_scenario)

    def test_motif_partition_preserves_the_aggregate(self):
        agg = self.w2["motif_aggregates"]["planning_motif"]
        split = self.w2["task_class_rules"][0]["t_ref_s"]
        self.assertAlmostEqual(sum(split.values()), agg["aggregate_s"], places=15)
        self.assertEqual(agg["partition_evidence_class"], "source_calibrated_synthetic")
        self.assertIn("ref_time_trajectory_single_thread", agg["subsumed_anchors"])

    def test_obi_anchors_are_not_double_counted(self):
        charged = []
        for rule in self.w2["task_class_rules"]:
            charged.extend(rule.get("applies_to_roles") or [])
        self.assertEqual(sorted(set(charged)), sorted(charged), "a role is charged twice")

    def test_P_f_and_D_f_are_independent_axes(self):
        base = self.v2.generate_deadlines_v2(self.tpl, self.w2, self.s2, self.fam)
        probe_sla = copy.deepcopy(self.s2)
        probe_sla["period_registry"]["families"][self.fam]["P_f_s"] *= 0.25
        probe = self.v2.generate_deadlines_v2(self.tpl, self.w2, probe_sla, self.fam)
        self.assertNotEqual(base.P_f, probe.P_f)
        self.assertEqual(base.D_G, probe.D_G)

    def test_D_G_cannot_depend_on_the_critical_path(self):
        base = self.v2.generate_deadlines_v2(self.tpl, self.w2, self.s2, self.fam)
        w = copy.deepcopy(self.w2)
        for rule in w["task_class_rules"]:
            # scale only range-based classes; the motif partition must stay invariant
            if rule.get("t_ref_s_range"):
                rule["t_ref_s_range"] = [5.0 * v for v in rule["t_ref_s_range"]]
        probe = self.v2.generate_deadlines_v2(self.tpl, w, self.s2, self.fam)
        self.assertEqual(base.D_G, probe.D_G)

    def test_post_freeze_mutation_changes_the_config_hash(self):
        probe = copy.deepcopy(self.s2)
        probe["deadline_registry"]["families"][self.fam]["D_f_s_range"] = [0.9, 1.2]
        self.assertNotEqual(self.v2.sla_config_sha256(self.s2), self.v2.sla_config_sha256(probe))

    def test_witness_cannot_be_passed_in(self):
        with self.assertRaises(TypeError):
            self.v2.generate_deadlines_v2(self.tpl, self.w2, self.s2, self.fam, witness=object())

    def test_determinism_and_decomposition(self):
        a = self.v2.generate_deadlines_v2(self.tpl, self.w2, self.s2, self.fam)
        b = self.v2.generate_deadlines_v2(self.tpl, self.w2, self.s2, self.fam)
        self.assertEqual(a.output_sha256, b.output_sha256)
        self.assertAlmostEqual(a.CP_ref, a.CP_compute + a.CP_communication_lb, places=15)

    def test_all_families_satisfy_the_bound_invariants(self):
        for tpl in self.templates:
            r = self.v2.generate_deadlines_v2(tpl, self.w2, self.s2, tpl["family_id"])
            self.assertEqual(r.status, "ok", f"{tpl['family_id']}: {r.status}")
            for t in r.tasks.values():
                self.assertLessEqual(t.E, t.d + 1e-12)
                self.assertLessEqual(t.d, t.L + 1e-12)
                self.assertGreaterEqual(t.slack, -1e-12)

    def test_sink_is_anchored_exactly_at_D_G(self):
        for tpl in self.templates:
            r = self.v2.generate_deadlines_v2(tpl, self.w2, self.s2, tpl["family_id"])
            self.assertTrue(any(abs(t.L - r.D_G) < 1e-12 for t in r.tasks.values()))

    def test_v1_references_are_rejected(self):
        self.assertEqual(self.semantics["defaults"]["workload_model_ref"], "workload_model_v2")

    def test_v2_references_resolve(self):
        for tpl in self.templates:
            for edge in tpl["edges"]:
                self.assertIn(edge["payload_class"], self.w2["payload_models"])

    def test_unknown_family_and_cycle_are_rejected(self):
        with self.assertRaises(self.v2.DeadlineError):
            self.v2.generate_deadlines_v2(self.tpl, self.w2, self.s2, "unknown_family")
        cyclic = copy.deepcopy(self.tpl)
        cyclic["edges"].append({"src": 16, "dst": 0, "payload_class": "control",
                                "payload_model_ref": "payload_control_v1",
                                "provenance_or_rule_ref": "x"})
        with self.assertRaises(self.v2.DeadlineError):
            self.v2.generate_deadlines_v2(cyclic, self.w2, self.s2, self.fam)

    def test_nonpositive_duration_is_rejected(self):
        w = copy.deepcopy(self.w2)
        w["task_class_rules"][1]["t_ref_s_range"] = [0.0, 1e-3]
        with self.assertRaises(self.v2.DeadlineError):
            self.v2.generate_deadlines_v2(self.tpl, w, self.s2, self.fam)


if __name__ == "__main__":
    unittest.main(verbosity=2)
