#!/usr/bin/env python3
"""M4: semantic DAG definitions and their validator."""

from __future__ import annotations

import copy
import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "spec" / "automotive_mc_v1"
sys.path.insert(0, str(ROOT))


def _m4():
    spec = importlib.util.spec_from_file_location(
        "validate_automotive_m4", BASE / "validate_automotive_m4.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _docs():
    import yaml

    return (yaml.safe_load((BASE / "task_semantics.yaml").read_text()),
            yaml.safe_load((BASE / "application_templates.yaml").read_text()))


class TestSemanticsFiles(unittest.TestCase):
    def setUp(self):
        self.m4 = _m4()
        self.semantics, self.templates = _docs()

    def test_semantics_pass_and_expose_the_frozen_vocabulary(self):
        violations, stats = self.m4.validate_semantics(self.semantics)
        self.assertEqual(violations, [])
        self.assertEqual(stats["families"], 4)
        self.assertEqual(stats["payload_classes"], 8)
        self.assertNotIn("raw_sensor", self.semantics["payload_classes"])

    def test_all_templates_are_20_task_dags(self):
        violations, stats = self.m4.validate_templates(self.templates, self.semantics)
        self.assertEqual(violations, [])
        self.assertEqual(stats["templates"], 4)
        self.assertEqual(stats["materializable_templates"], 4)

    def test_families_are_exactly_the_frozen_four(self):
        self.assertEqual(set(self.semantics["families"]),
                         {"perception_planning_control", "cooperative_perception",
                          "localization_prediction_planning", "mapping_background"})

    def test_cooperative_family_has_real_v2x_edges(self):
        tpl = next(t for t in self.templates["templates"]
                   if t["family_id"] == "cooperative_perception")
        classes = {e["payload_class"] for e in tpl["edges"]}
        self.assertIn("v2x_message", classes)
        roles = {t["semantic_role"] for t in tpl["tasks"]}
        self.assertIn("cooperative_perception", roles)

    def test_mapping_background_has_no_control_path(self):
        tpl = next(t for t in self.templates["templates"] if t["family_id"] == "mapping_background")
        roles = {t["semantic_role"] for t in tpl["tasks"]}
        self.assertNotIn("control", roles)
        self.assertIn("map_update", roles)

    def test_lineage_and_signature_present_for_splits(self):
        for tpl in self.templates["templates"]:
            self.assertTrue(tpl["template_lineage"])
            self.assertTrue(tpl["semantic_signature"])
            for task in tpl["tasks"]:
                self.assertTrue(task["lineage_id"])

    def test_no_compute_demand_leaks_into_edges(self):
        for tpl in self.templates["templates"]:
            for edge in tpl["edges"]:
                self.assertNotIn("compute_workload", edge)
                self.assertNotIn("W_i", edge)
                self.assertIn(edge["payload_class"], self.semantics["payload_classes"])


class TestValidatorNegatives(unittest.TestCase):
    def setUp(self):
        self.m4 = _m4()
        self.semantics, self.templates = _docs()

    def _tpl(self, family="perception_planning_control"):
        tpl = copy.deepcopy(next(t for t in self.templates["templates"] if t["family_id"] == family))
        doc = {"templates": [tpl]}
        return tpl, doc

    def test_cycle_is_detected(self):
        tpl, doc = self._tpl()
        tpl["edges"].append({"src": 16, "dst": 0, "payload_class": "control",
                             "payload_model_ref": "payload_control_v1",
                             "provenance_or_rule_ref": "x"})
        violations, _s = self.m4.validate_templates(doc, self.semantics)
        self.assertTrue(any("cycle" in v for v in violations), violations)

    def test_wrong_task_counts_fail(self):
        for drop, add in ((1, 0), (0, 1)):
            tpl, doc = self._tpl()
            if drop:
                tpl["tasks"].pop()
            if add:
                tpl["tasks"].append(dict(tpl["tasks"][-1], task_id=99, lineage_id="x#99"))
            violations, _s = self.m4.validate_templates(doc, self.semantics)
            self.assertTrue(any("task_count" in v for v in violations), violations)

    def test_duplicate_ids_fail(self):
        tpl, doc = self._tpl()
        tpl["tasks"][1]["task_id"] = tpl["tasks"][0]["task_id"]
        violations, _s = self.m4.validate_templates(doc, self.semantics)
        self.assertTrue(any("duplicate task ids" in v for v in violations), violations)

    def test_unknown_role_and_missing_semantics_fail(self):
        tpl, doc = self._tpl()
        tpl["tasks"][0]["semantic_role"] = "unknown_role"
        tpl["tasks"][1].pop("motif_id")
        violations, _s = self.m4.validate_templates(doc, self.semantics)
        self.assertTrue(any("unknown semantic_role" in v for v in violations), violations)
        self.assertTrue(any("missing motif_id" in v for v in violations), violations)

    def test_unresolved_payload_model_fails(self):
        tpl, doc = self._tpl()
        tpl["edges"][0]["payload_model_ref"] = "not_a_model"
        violations, _s = self.m4.validate_templates(doc, self.semantics)
        self.assertTrue(any("does not resolve" in v for v in violations), violations)

    def test_raw_sensor_class_fails(self):
        tpl, doc = self._tpl()
        sem = copy.deepcopy(self.semantics)
        sem["payload_classes"].append("raw_sensor")
        sem["payload_models"]["raw_sensor"] = "x"
        tpl["edges"][0]["payload_class"] = "raw_sensor"
        tpl["edges"][0]["payload_model_ref"] = "x"
        violations, _s = self.m4.validate_templates(doc, sem)
        self.assertTrue(any("forbidden payload class" in v for v in violations), violations)

    def test_unknown_family_fails(self):
        tpl, doc = self._tpl()
        tpl["family_id"] = "fifth_family"
        violations, _s = self.m4.validate_templates(doc, self.semantics)
        self.assertTrue(any("unknown family" in v for v in violations), violations)

    def test_invalid_edge_endpoint_fails(self):
        tpl, doc = self._tpl()
        tpl["edges"].append({"src": 0, "dst": 12345, "payload_class": "camera",
                             "payload_model_ref": "payload_raw_camera_v1",
                             "provenance_or_rule_ref": "x"})
        violations, _s = self.m4.validate_templates(doc, self.semantics)
        self.assertTrue(any("invalid edge endpoint" in v for v in violations), violations)

    def test_criticality_baked_into_a_template_fails(self):
        tpl, doc = self._tpl()
        tpl["tasks"][0]["criticality_class"] = "HIGH"
        violations, _s = self.m4.validate_templates(doc, self.semantics)
        self.assertTrue(any("criticality must not be baked" in v for v in violations), violations)

    def test_criticality_in_the_semantics_layer_fails(self):
        sem = copy.deepcopy(self.semantics)
        sem["roles"][0]["criticality_class"] = "HIGH"
        violations, _s = self.m4.validate_semantics(sem)
        self.assertTrue(any("must not be assigned" in v for v in violations), violations)

    def test_semantic_signatures_are_deterministic(self):
        import yaml

        again = yaml.safe_load((BASE / "application_templates.yaml").read_text())
        self.assertEqual([t["semantic_signature"] for t in self.templates["templates"]],
                         [t["semantic_signature"] for t in again["templates"]])


if __name__ == "__main__":
    unittest.main(verbosity=2)
