#!/usr/bin/env python3
"""M5: workload/payload/resource models, action invariance and numeric sanity."""

from __future__ import annotations

import copy
import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "spec" / "automotive_mc_v1"
sys.path.insert(0, str(ROOT))


def _m5():
    spec = importlib.util.spec_from_file_location(
        "validate_automotive_m5", BASE / "validate_automotive_m5.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


def _docs():
    import yaml

    return (yaml.safe_load((BASE / "workload_model.yaml").read_text()),
            yaml.safe_load((BASE / "resource_profiles.yaml").read_text()),
            yaml.safe_load((BASE / "task_semantics.yaml").read_text()),
            yaml.safe_load((BASE / "application_templates.yaml").read_text()))


class TestModelsPass(unittest.TestCase):
    def setUp(self):
        self.m5 = _m5()
        self.workload, self.resources, self.semantics, self.templates = _docs()

    def test_workload_and_payloads_valid(self):
        v, stats = self.m5.validate_workload(self.workload)
        self.assertEqual(v, [])
        self.assertEqual(stats["payload_models"], 8)
        self.assertEqual(stats["cpu_anchors"], 3)

    def test_resources_valid(self):
        v, stats = self.m5.validate_resources(self.resources)
        self.assertEqual(v, [])
        self.assertEqual(stats["profiles"], 3)

    def test_m4_cross_check_resolves(self):
        v, stats = self.m5.cross_check_m4(self.workload, self.semantics, self.templates)
        self.assertEqual(v, [])
        self.assertEqual(stats["roles_covered"], stats["roles_total"])

    def test_all_eight_payload_classes_have_a_distinct_rule(self):
        rules = {m["rule_id"] for m in self.workload["payload_models"].values()}
        self.assertEqual(len(rules), 8)
        self.assertNotIn("raw_sensor", self.workload["payload_models"])

    def test_scheduler_uses_achievable_capacity_only(self):
        self.assertEqual(self.resources["scheduler_uses"], "achievable_capacity")
        for prof in self.resources["profiles"].values():
            self.assertNotIn("service_required_rate", prof)
            self.assertNotIn("offered_traffic", prof)


class TestActionInvariance(unittest.TestCase):
    """W_i is a task property; only the execution time depends on the tier."""

    def setUp(self):
        self.workload, self.resources, _s, _t = _docs()
        ref = self.workload["reference_compute_model"]
        self.f_ref = float(ref["f_ref_hz"])
        self.xi = float(ref["xi_cycles_per_bit"])

    def _w(self, t_ref):
        return t_ref * self.f_ref / (8.0 * self.xi)

    def test_workload_identical_across_actions(self):
        rule = next(r for r in self.workload["task_class_rules"] if r.get("t_ref_s"))
        w = self._w(float(rule["t_ref_s"]))
        prof = self.resources["profiles"]["nominal"]
        times = {}
        for tier in ("f_ue_hz", "f_helper_hz", "f_mec_hz"):
            f = float(prof[tier]["range"][0])
            times[tier] = 8.0 * self.xi * w / f
            self.assertAlmostEqual(self._w(float(rule["t_ref_s"])), w, places=12)
        self.assertLess(times["f_mec_hz"], times["f_helper_hz"])
        self.assertLess(times["f_helper_hz"], times["f_ue_hz"])

    def test_transmission_depends_only_on_payload_and_rate(self):
        def t_tx(b, r):
            return 8.0 * b / r

        self.assertAlmostEqual(t_tx(2000, 10e6), 8.0 * 2000 / 10e6, places=15)
        self.assertLess(t_tx(1000, 10e6), t_tx(2000, 10e6))
        self.assertLess(t_tx(2000, 20e6), t_tx(2000, 10e6))


class TestNegatives(unittest.TestCase):
    def setUp(self):
        self.m5 = _m5()
        self.workload, self.resources, self.semantics, self.templates = _docs()

    def test_nonpositive_workload_inputs_fail(self):
        w = copy.deepcopy(self.workload)
        w["reference_compute_model"]["f_ref_hz"] = 0
        w["reference_compute_model"]["xi_cycles_per_bit"] = -1
        v, _s = self.m5.validate_workload(w)
        self.assertTrue(any("f_ref_hz must be positive" in x for x in v), v)
        self.assertTrue(any("xi must be positive" in x for x in v), v)

    def test_unknown_payload_model_fails(self):
        w = copy.deepcopy(self.workload)
        del w["payload_models"]["lidar"]
        v, _s = self.m5.validate_workload(w)
        self.assertTrue(any("payload model missing for class 'lidar'" in x for x in v), v)

    def test_mixed_statistic_fails(self):
        w = copy.deepcopy(self.workload)
        w["task_class_rules"][0]["t_ref_statistic"] = "max"
        v, _s = self.m5.validate_workload(w)
        self.assertTrue(any("mixes timing statistic" in x for x in v), v)

    def test_gpu_row_as_cpu_anchor_fails(self):
        w = copy.deepcopy(self.workload)
        rule = next(r for r in w["task_class_rules"] if r["evidence_class"] == "measured_cpu_anchor")
        rule["source_row"] = "ref_time_jetson_gpu_row"
        v, _s = self.m5.validate_workload(w)
        self.assertTrue(any("GPU/accelerator row used as a CPU anchor" in x for x in v), v)

    def test_payload_depending_on_w_i_fails(self):
        w = copy.deepcopy(self.workload)
        w["payload_models"]["control"]["formula"] = "B = W_i / 8"
        v, _s = self.m5.validate_workload(w)
        self.assertTrue(any("must not depend on compute demand" in x for x in v), v)

    def test_camera_channel_double_counting_fails(self):
        w = copy.deepcopy(self.workload)
        w["payload_models"]["camera"]["formula"] = "B = W * H * C * bits_per_pixel / 8"
        v, _s = self.m5.validate_workload(w)
        self.assertTrue(any("channels double-counted" in x for x in v), v)

    def test_raw_sensor_model_fails(self):
        w = copy.deepcopy(self.workload)
        w["payload_models"]["raw_sensor"] = {"formula": "x", "units": "bytes",
                                             "rule_id": "R", "evidence_class": "derived",
                                             "formula_kind": "generic"}
        v, _s = self.m5.validate_workload(w)
        self.assertTrue(any("raw_sensor" in x for x in v), v)

    def test_service_rate_as_capacity_fails(self):
        r = copy.deepcopy(self.resources)
        r["profiles"]["nominal"]["service_required_rate"] = 10
        v, _s = self.m5.validate_resources(r)
        self.assertTrue(any("service_required_rate must not be a scheduler input" in x for x in v), v)

    def test_nonpositive_capacity_range_fails(self):
        r = copy.deepcopy(self.resources)
        r["profiles"]["nominal"]["r_v2v_bps"]["range"] = [-1.0, 5.0]
        v, _s = self.m5.validate_resources(r)
        self.assertTrue(any("invalid range" in x for x in v), v)

    def test_missing_units_fails(self):
        r = copy.deepcopy(self.resources)
        del r["profiles"]["strong"]["f_mec_hz"]["units"]
        v, _s = self.m5.validate_resources(r)
        self.assertTrue(any("missing units" in x for x in v), v)

    def test_unresolved_workload_rule_fails(self):
        sem = copy.deepcopy(self.semantics)
        sem["roles"].append({"semantic_role": "unmapped_role", "motif_role": "sensing",
                             "families": ["mapping_background"], "payload_in": [],
                             "payload_out": []})
        v, _s = self.m5.cross_check_m4(self.workload, sem, self.templates)
        self.assertTrue(any("unmapped_role" in x for x in v), v)


class TestScaleAudit(unittest.TestCase):
    def test_representative_scale_is_sane(self):
        workload, resources, _s, _t = _docs()
        ref = workload["reference_compute_model"]
        f_ref, xi = float(ref["f_ref_hz"]), float(ref["xi_cycles_per_bit"])
        prof = resources["profiles"]["nominal"]
        durations = []
        for rule in workload["task_class_rules"]:
            if rule.get("t_ref_s"):
                w = float(rule["t_ref_s"]) * f_ref / (8.0 * xi)
            else:
                w = (rule["t_ref_s_range"][1]) * f_ref / (8.0 * xi)
            f = float(prof["f_ue_hz"]["range"][0])          # slowest tier
            durations.append(8.0 * xi * w / f)
        self.assertTrue(all(d > 0 for d in durations))
        self.assertLess(max(durations), 60.0, "a task must not take minutes on the slowest tier")
        t_tx = [8.0 * b / float(prof["r_mec_ul_bps"]["range"][0]) for b in (4000, 2000, 20000)]
        self.assertTrue(all(t > 0 for t in t_tx))


if __name__ == "__main__":
    unittest.main(verbosity=2)
