#!/usr/bin/env python3
"""The source registry's admission invariants must not regress."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
REGISTRY = ROOT / "spec" / "automotive_mc_v1" / "SOURCE_REGISTRY.yaml"
VALIDATOR = ROOT / "spec" / "automotive_mc_v1" / "validate_source_registry.py"


def _validator():
    spec = importlib.util.spec_from_file_location("validate_source_registry", VALIDATOR)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestSourceRegistryGate(unittest.TestCase):
    def setUp(self):
        import yaml

        if not REGISTRY.exists():
            self.skipTest("registry not present")
        self.module = _validator()
        self.doc = yaml.safe_load(REGISTRY.read_text())

    def test_no_admission_violations(self):
        violations, _stats = self.module.validate(self.doc)
        self.assertEqual(violations, [])

    def test_gate_is_a_conjunction(self):
        m = self.module
        self.assertTrue(m.expected_allowed(
            {"source_verification": "official_verified", "transcription_verified": True}))
        self.assertFalse(m.expected_allowed(
            {"source_verification": "official_verified", "transcription_verified": False}))
        self.assertFalse(m.expected_allowed(
            {"source_verification": "secondary_lead_unverified", "transcription_verified": True}))
        self.assertTrue(m.expected_allowed(
            {"source_verification": "synthetic_calibrated",
             "generation_rule_verified": True, "generation_rule_id": "E2E-SLA-V1"}))
        self.assertFalse(m.expected_allowed(
            {"source_verification": "synthetic_calibrated",
             "generation_rule_verified": True, "generation_rule_id": None}))

    def test_required_rows_carry_every_admission_field(self):
        _v, stats = self.module.validate(self.doc)
        self.assertIn("blocked_required_parameters", stats)
        self.assertGreaterEqual(stats["blocked_required_parameters"], 0)
        self.assertGreater(stats["required_by_generator"], 0)

    def _row(self, name, required, allowed, sv="official_verified", tv=False, **extra):
        row = {
            "parameter_name": name,
            "required_by_generator": required,
            "source_verification": sv,
            "transcription_verified": tv,
            "measurement_scope": "processing_stage",
            "semantic_role": "execution_time",
            "allowed_for_generation": allowed,
        }
        row.update(extra)
        return row

    def test_gate_counts_only_required_parameters(self):
        # a blocked REFERENCE row must not hold Step 2 back; only required ones count
        doc = {"sources": [{"source_id": "S1", "parameters": [
            self._row("req_blocked", True, False),
            self._row("ref_blocked", False, False, sv="secondary_lead_unverified"),
            self._row("req_allowed", True, True, tv=True),
        ]}]}
        violations, stats = self.module.validate(doc)
        self.assertEqual(violations, [])
        self.assertEqual(stats["required_by_generator"], 2)
        self.assertEqual(stats["blocked_required_parameters"], 1)

    def test_dg_guards(self):
        doc = {"sources": [{"source_id": "S1", "parameters": [
            self._row("comm_req", True, True, tv=True, measurement_scope="communication",
                      semantic_role="requirement", defines_D_G=True),
            self._row("measured", True, True, tv=True, semantic_role="measured_latency",
                      defines_D_G=True),
        ]}]}
        violations, _stats = self.module.validate(doc)
        self.assertEqual(len(violations), 2)

    def test_over_admission_is_caught(self):
        doc = {"sources": [{"source_id": "S1", "parameters": [
            self._row("over", True, True, tv=False),   # source ok but not transcribed
        ]}]}
        violations, _stats = self.module.validate(doc)
        self.assertEqual(len(violations), 1)
        self.assertIn("contradicts the gate", violations[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
