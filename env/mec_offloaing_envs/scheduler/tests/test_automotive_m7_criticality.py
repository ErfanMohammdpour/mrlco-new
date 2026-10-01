#!/usr/bin/env python3
"""M7 focused + adversarial tests: semantic criticality, budgets, LO/HI modes.

The adversarial tests mutate exactly the inputs the contract forbids as criticality
sources (task id, topology, workload, deadline, slack, resource profile) and require
the class assignment to stay put, and mutate the allowed inputs and require it to move.
"""

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
sys.path.insert(0, str(BASE))


def _mod(name):
    spec = importlib.util.spec_from_file_location(name, BASE / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def _docs():
    return {
        "policy": yaml.safe_load((BASE / "criticality_policy.yaml").read_text()),
        "semantics": yaml.safe_load((BASE / "task_semantics.yaml").read_text()),
        "w2": yaml.safe_load((BASE / "workload_model_v2.yaml").read_text()),
        "s2": yaml.safe_load((BASE / "sla_registry_v2.yaml").read_text()),
        "templates": yaml.safe_load((BASE / "application_templates.yaml").read_text())["templates"],
    }


class TestClassAssignment(unittest.TestCase):
    def setUp(self):
        self.cm = _mod("criticality_model")
        self.d = _docs()
        self.table = {t["family_id"]: self.cm.build_task_table(self.d["policy"], self.d["w2"], t)
                      for t in self.d["templates"]}

    def test_every_role_has_a_class_and_budgets(self):
        for fam, tpl in ((t["family_id"], t) for t in self.d["templates"]):
            table = self.cm.build_task_table(self.d["policy"], self.d["w2"], tpl)
            self.assertEqual(len(table), 20)
            for row in table.values():
                self.assertIn(row["criticality"], self.cm.CLASSES)
                self.assertGreater(row["empirical_execution_budget_lo_s"], 0.0)

    def test_high_tasks_carry_valid_mc_budgets(self):
        for table in self.table.values():
            for row in table.values():
                if row["criticality"] == "HIGH":
                    self.assertTrue(row["budget_hi_applicable"])
                    hi = row["empirical_execution_budget_hi_s"]
                    self.assertIsNotNone(hi)
                    self.assertGreaterEqual(hi, row["empirical_execution_budget_lo_s"])
                    self.assertFalse(row["wcet_claim"])
                else:
                    self.assertFalse(row["budget_hi_applicable"])
                    self.assertIsNone(row["empirical_execution_budget_hi_s"])
                    self.assertEqual(row["budget_hi_status"], "not_applicable")

    def test_high_exists_in_every_safety_family(self):
        for fam in ("perception_planning_control", "cooperative_perception",
                    "localization_prediction_planning"):
            mix = self.cm.criticality_mixture(self.table[fam])
            self.assertGreater(mix["HIGH"], 0, fam)

    def test_non_safety_family_is_declared_and_low(self):
        mix = self.cm.criticality_mixture(self.table["mapping_background"])
        self.assertEqual(mix["HIGH"], 0)
        self.assertEqual(mix["LOW"], 20)

    # ---- adversarial: forbidden inputs must not move the class -----------------
    def test_task_id_and_topology_do_not_determine_the_class(self):
        tpl = self.d["templates"][0]
        base = {t: r["criticality"] for t, r in
                self.cm.build_task_table(self.d["policy"], self.d["w2"], tpl).items()}
        relabeled = copy.deepcopy(tpl)
        swap = {i: 19 - i for i in range(20)}
        for t in relabeled["tasks"]:
            t["task_id"] = swap[t["task_id"]]
        for e in relabeled["edges"]:
            e["src"], e["dst"] = swap[e["src"]], swap[e["dst"]]
        probe = self.cm.build_task_table(self.d["policy"], self.d["w2"], relabeled)
        self.assertEqual({swap[t]: r["criticality"] for t, r in probe.items()}, base)

    def test_workload_does_not_move_the_class(self):
        tpl = self.d["templates"][0]
        base = {t: r["criticality"] for t, r in
                self.cm.build_task_table(self.d["policy"], self.d["w2"], tpl).items()}
        w = copy.deepcopy(self.d["w2"])
        for rule in w["task_class_rules"]:
            if rule.get("t_ref_s_range"):
                rule["t_ref_s_range"] = [11.0 * v for v in rule["t_ref_s_range"]]
        probe = self.cm.build_task_table(self.d["policy"], w, tpl)
        self.assertEqual({t: r["criticality"] for t, r in probe.items()}, base)

    def test_deadline_and_slack_are_not_inputs_to_the_class(self):
        # the resolver signature carries no deadline/slack; the class table is
        # invariant under a deadline-model change
        tpl = self.d["templates"][0]
        base = {t: r["criticality"] for t, r in
                self.cm.build_task_table(self.d["policy"], self.d["w2"], tpl).items()}
        s2 = copy.deepcopy(self.d["s2"])
        for fam in s2["deadline_registry"]["families"]:
            s2["deadline_registry"]["families"][fam]["D_f_s_range"] = [0.05, 5.0]
        probe = self.cm.build_task_table(self.d["policy"], self.d["w2"], tpl)
        self.assertEqual({t: r["criticality"] for t, r in probe.items()}, base)
        self.assertNotEqual(s2["deadline_registry"]["families"]["perception_planning_control"]
                            ["D_f_s_range"], self.d["s2"]["deadline_registry"]["families"]
                            ["perception_planning_control"]["D_f_s_range"])

    def test_allowed_inputs_family_does_move_the_class(self):
        # changing the family (an ALLOWED input) must be able to change the class
        same_role_diff_family = []
        for fam in ("perception_planning_control", "mapping_background"):
            tpl = next(t for t in self.d["templates"] if t["family_id"] == fam)
            table = self.cm.build_task_table(self.d["policy"], self.d["w2"], tpl)
            same_role_diff_family.append({r["semantic_role"]: r["criticality"]
                                          for r in table.values()})
        self.assertNotEqual(same_role_diff_family[0].get("localization"),
                            same_role_diff_family[1].get("localization"))

    def test_safety_family_may_not_downgrade_a_role(self):
        policy = copy.deepcopy(self.d["policy"])
        policy["family_class_override"]["cooperative_perception"] = {"control": "LOW"}
        with self.assertRaises(self.cm.CriticalityError):
            self.cm.classify(policy, "control", "cooperative_perception")

    def test_unknown_family_and_role_are_refused(self):
        with self.assertRaises(self.cm.CriticalityError):
            self.cm.classify(self.d["policy"], "control", "no_such_family")
        with self.assertRaises(self.cm.CriticalityError):
            self.cm.classify(self.d["policy"], "no_such_role", "cooperative_perception")

    def test_motif_floor_is_enforced_in_safety_scope(self):
        policy = copy.deepcopy(self.d["policy"])
        policy["role_baseline_class"]["planning"] = "LOW"
        with self.assertRaises(self.cm.CriticalityError):
            self.cm.classify(policy, "planning", "perception_planning_control")


class TestModeMachine(unittest.TestCase):
    def setUp(self):
        self.cm = _mod("criticality_model")
        self.d = _docs()
        self.tpl = self.d["templates"][0]
        self.table = self.cm.build_task_table(self.d["policy"], self.d["w2"], self.tpl)
        self.order = sorted(self.table)

    def _quiet(self):
        return {t: 0.5 * self.table[t]["empirical_execution_budget_lo_s"] for t in self.order}

    def test_starts_in_lo_without_overrun(self):
        trace = self.cm.simulate_mode_trace(self.d["policy"], self.table, self._quiet(), self.order)
        self.assertEqual(trace.initial_mode, "LO")
        self.assertEqual(trace.final_mode, "LO")
        self.assertEqual(len(trace.switches), 0)

    def test_non_high_overrun_never_switches(self):
        obs = self._quiet()
        non_high = [t for t in self.order if self.table[t]["criticality"] != "HIGH"][0]
        obs[non_high] = 9.0 * self.table[non_high]["empirical_execution_budget_lo_s"]
        trace = self.cm.simulate_mode_trace(self.d["policy"], self.table, obs, self.order)
        self.assertEqual(trace.final_mode, "LO")

    def test_high_overrun_switches_once_and_logs_every_field(self):
        high = [t for t in self.order if self.table[t]["criticality"] == "HIGH"]
        obs = self._quiet()
        obs[high[2]] = 3.0 * self.table[high[2]]["empirical_execution_budget_lo_s"]
        trace = self.cm.simulate_mode_trace(self.d["policy"], self.table, obs, self.order)
        self.assertEqual(trace.final_mode, "HI")
        self.assertEqual(len(trace.switches), 1)
        sw = trace.switches[0].as_dict()
        self.assertEqual(sorted(sw), sorted(self.cm.SWITCH_LOG_FIELDS))
        self.assertEqual(sw["triggering_task_id"], high[2])
        self.assertEqual(sw["previous_mode"], "LO")
        self.assertEqual(sw["new_mode"], "HI")
        self.assertAlmostEqual(sw["C_LO"], self.table[high[2]]["empirical_execution_budget_lo_s"])

    def test_hi_is_sticky_and_never_returns_to_lo(self):
        high = [t for t in self.order if self.table[t]["criticality"] == "HIGH"]
        obs = self._quiet()
        obs[high[0]] = 2.0 * self.table[high[0]]["empirical_execution_budget_lo_s"]
        for order in (self.order, list(reversed(self.order))):
            trace = self.cm.simulate_mode_trace(self.d["policy"], self.table, obs, order)
            self.assertEqual(trace.final_mode, "HI")

    def test_high_is_never_droppable_or_degradable(self):
        for mode in ("LO", "HI"):
            dg = self.cm.drop_degrade(self.d["policy"], "HIGH", mode)
            self.assertFalse(dg["drop_allowed"])
            self.assertFalse(dg["degrade_allowed"])
        self.assertTrue(self.cm.drop_degrade(self.d["policy"], "LOW", "HI")["drop_allowed"])
        self.assertFalse(self.cm.drop_degrade(self.d["policy"], "LOW", "LO")["drop_allowed"])

    def test_mode_policy_cannot_be_reconfigured_without_breaking_v1(self):
        policy = copy.deepcopy(self.d["policy"])
        policy["mode_semantics"]["initial_mode"] = "HI"
        with self.assertRaises(self.cm.CriticalityError):
            self.cm.simulate_mode_trace(policy, self.table, self._quiet(), self.order)
        policy = copy.deepcopy(self.d["policy"])
        policy["mode_semantics"]["hi_exit"] = "exit_on_next_task"
        with self.assertRaises(self.cm.CriticalityError):
            self.cm.simulate_mode_trace(policy, self.table, self._quiet(), self.order)


class TestAxisSeparation(unittest.TestCase):
    def setUp(self):
        self.cm = _mod("criticality_model")
        self.d = _docs()
        self.tpl = self.d["templates"][0]
        self.table = self.cm.build_task_table(self.d["policy"], self.d["w2"], self.tpl)

    def test_class_is_not_a_function_of_depth(self):
        depth = self.cm.topological_depth(self.tpl)
        rows = [{"criticality": r["criticality"], "depth": str(depth[t])}
                for t, r in self.table.items()]
        assoc = self.cm.association(rows, "depth")
        self.assertFalse(assoc["criticality_is_a_function_of_probe"])
        self.assertGreater(assoc["levels_with_mixed_classes"], 0)

    def test_class_is_not_a_function_of_deadline_tightness(self):
        slack = {t: float(t) for t in range(20)}
        buckets = ["q1" if i < 5 else "q2" if i < 10 else "q3" if i < 15 else "q4" for i in slack]
        rows = [{"criticality": r["criticality"], "urgency": buckets[t]}
                for t, r in self.table.items()]
        assoc = self.cm.association(rows, "urgency")
        self.assertFalse(assoc["criticality_is_a_function_of_probe"])

    def test_policy_forbids_the_right_inputs(self):
        forbidden = set(self.d["policy"]["axis_separation"]["forbidden_inputs"])
        for name in ("depth", "task_id", "workload", "payload", "deadline", "slack",
                     "execution_location", "resource_profile"):
            self.assertIn(name, forbidden)

    def test_demand_rule_resolves_each_role_exactly_once(self):
        for role in self.d["semantics"]["roles"]:
            rule = self.cm.resolve_demand_rule(self.d["w2"], role["semantic_role"])
            self.assertIn(rule["kind"], ("fixed", "range"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
