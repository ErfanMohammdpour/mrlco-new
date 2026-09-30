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
        if tv:
            # a transcribed EXECUTION row must also carry its measurement context
            if row["semantic_role"] in ("measured_latency", "execution_time"):
                row["execution_context"] = {
                    "platform": "unit-test fixture", "stage": "stage",
                    "reported_statistic": "mean", "unit": "ms",
                }
                row["conversion_to_cycles"] = "allowed"
            # a row claiming transcription must carry a complete, fresh ledger
            import hashlib

            evidence = _validator().EVIDENCE
            row["transcription_ledger"] = {
                "document_version": "TS 22.186 V16.2.0", "section": "5.3",
                "table": "Table 5.3-1", "requirement_id": "R.5.3-001",
                "column": "Max. end-to-end latency (ms)", "value": "10", "unit": "ms",
                "evidence_sha256": hashlib.sha256(evidence.read_bytes()).hexdigest(),
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


class TestRequiredManifest(unittest.TestCase):
    """Parameters that are not in the registry yet must not be invisible."""

    def setUp(self):
        import yaml

        self.module = _validator()
        self.rows = {"present_allowed": {"allowed_for_generation": True},
                     "present_blocked": {"allowed_for_generation": False}}
        self.base = {"application_families": ["fam_a"]}

    def _manifest(self, entries, families=("fam_a",)):
        return {"application_families": list(families), "required_parameters": entries}

    def test_absent_parameter_counts_as_missing(self):
        m = self.module
        man = self._manifest([
            {"id": "never_registered", "category": "payload", "requires_numeric": True, "registry_binding": []},
            {"id": "fam_a_e2e", "category": "application_e2e", "requires_numeric": True,
             "registry_binding": [], "required_for_families": ["fam_a"]},
        ])
        violations, stats = m.validate_manifest(man, self.rows)
        self.assertEqual(stats["missing_required_parameters"], 2)
        self.assertEqual(stats["blocked_required_parameters"], 0)

    def test_bound_but_not_allowed_counts_as_blocked(self):
        m = self.module
        man = self._manifest([
            {"id": "b", "category": "execution", "requires_numeric": True,
             "registry_binding": ["present_blocked"]},
            {"id": "fam_a_e2e", "category": "application_e2e", "requires_numeric": True,
             "registry_binding": [], "required_for_families": ["fam_a"]},
        ])
        _v, stats = m.validate_manifest(man, self.rows)
        self.assertEqual(stats["blocked_required_parameters"], 1)

    def test_rule_only_parameter_is_never_missing(self):
        m = self.module
        man = self._manifest([
            {"id": "r", "category": "rule", "requires_numeric": False, "registry_binding": []},
            {"id": "fam_a_e2e", "category": "application_e2e", "requires_numeric": True,
             "registry_binding": [], "required_for_families": ["fam_a"]},
        ])
        violations, stats = m.validate_manifest(man, self.rows)
        self.assertEqual(stats["rule_only_parameters"], 1)
        self.assertEqual(stats["missing_required_parameters"], 1)  # only the e2e anchor
        self.assertEqual(violations, [])

    def test_family_without_e2e_entry_is_a_violation(self):
        m = self.module
        man = self._manifest([], families=["fam_a", "fam_b"])
        violations, _s = m.validate_manifest(man, self.rows)
        self.assertEqual(len(violations), 2)

    def test_binding_an_unknown_row_is_a_violation(self):
        m = self.module
        man = self._manifest([
            {"id": "x", "category": "radio", "requires_numeric": True, "registry_binding": ["nope"]},
            {"id": "fam_a_e2e", "category": "application_e2e", "requires_numeric": True,
             "registry_binding": [], "required_for_families": ["fam_a"]},
        ])
        violations, _s = m.validate_manifest(man, self.rows)
        self.assertTrue(any("unknown registry rows" in v for v in violations))


class TestBindingPolicy(unittest.TestCase):
    """An aggregated value must not satisfy every family by accident."""

    def setUp(self):
        self.module = _validator()
        self.rows = {"row_a": {"allowed_for_generation": True},
                     "row_b": {"allowed_for_generation": False}}

    def _manifest(self, entries):
        return {"application_families": ["fam_a"], "required_parameters": list(entries) + [
            {"id": "fam_a_e2e", "category": "application_e2e", "requires_numeric": True,
             "registry_binding": [], "required_for_families": ["fam_a"]}]}

    def test_multi_row_binding_without_policy_is_a_violation(self):
        man = self._manifest([{"id": "agg", "category": "payload", "requires_numeric": True,
                               "registry_binding": ["row_a", "row_b"]}])
        violations, _s = self.module.validate_manifest(man, self.rows)
        self.assertTrue(any("binding_policy" in v for v in violations), violations)

    def test_use_case_map_must_cover_every_binding(self):
        man = self._manifest([{"id": "agg", "category": "payload", "requires_numeric": True,
                               "registry_binding": ["row_a", "row_b"],
                               "binding_policy": "use_case_conditioned",
                               "binding_map": {"uc": ["row_a"]}}])
        violations, _s = self.module.validate_manifest(man, self.rows)
        self.assertTrue(any("does not cover" in v for v in violations), violations)

    def test_complete_use_case_map_is_accepted_and_still_blocked(self):
        man = self._manifest([{"id": "agg", "category": "payload", "requires_numeric": True,
                               "registry_binding": ["row_a", "row_b"],
                               "binding_policy": "use_case_conditioned",
                               "binding_map": {"uc1": ["row_a"], "uc2": ["row_b"]}}])
        violations, stats = self.module.validate_manifest(man, self.rows)
        self.assertEqual(violations, [])
        self.assertEqual(stats["blocked_required_parameters"], 1)

    def test_binding_map_naming_an_unknown_row_is_a_violation(self):
        man = self._manifest([{"id": "agg", "category": "payload", "requires_numeric": True,
                               "registry_binding": ["row_a", "row_b"],
                               "binding_policy": "use_case_conditioned",
                               "binding_map": {"uc1": ["row_a"], "uc2": ["ghost"]}}])
        violations, _s = self.module.validate_manifest(man, self.rows)
        self.assertTrue(any("unknown row" in v for v in violations), violations)


class TestResourceOntology(unittest.TestCase):
    """Communication capacity and compute capacity are different resources."""

    def test_manifest_separates_capacity_kinds(self):
        import yaml

        man = yaml.safe_load(
            (Path(__file__).resolve().parents[4] / "spec" / "automotive_mc_v1"
             / "REQUIRED_PARAMETER_MANIFEST.yaml").read_text())
        by_cat = {}
        for entry in man["required_parameters"]:
            by_cat.setdefault(entry["category"], []).append(entry["id"])
        self.assertIn("communication_capacity", by_cat)
        self.assertIn("compute_capacity", by_cat)
        self.assertEqual(sorted(by_cat["communication_capacity"]),
                         ["mec_dl_achievable_capacity_profile",
                          "mec_ul_achievable_capacity_profile",
                          "v2v_achievable_capacity_profile"])
        self.assertEqual(sorted(by_cat["compute_capacity"]),
                         ["helper_compute_profile", "mec_compute_profile", "ue_compute_profile"])
        for entry in man["required_parameters"]:
            if len(entry.get("registry_binding") or []) > 1:
                self.assertIn(entry.get("binding_policy"),
                              ("single", "use_case_conditioned", "all_must_be_allowed"))


class TestConversionDeclaration(unittest.TestCase):
    """A GPU measurement must not silently become CPU cycles."""

    def setUp(self):
        self.module = _validator()

    def _row(self, platform, conv, **extra):
        row = {"parameter_name": "x", "required_by_generator": True,
               "source_verification": "official_verified", "transcription_verified": True,
               "measurement_scope": "processing_stage", "semantic_role": "measured_latency",
               "allowed_for_generation": True,
               "execution_context": {"platform": platform, "stage": "s", "reported_statistic": "mean", "unit": "ms"},
               "conversion_to_cycles": conv,
               "transcription_ledger": {
                   "document_version": "TS 22.186 V16.2.0", "section": "5.3",
                   "table": "Table 5.3-1", "requirement_id": "R.5.3-001",
                   "column": "x", "value": "10", "unit": "ms",
                   "evidence_sha256": __import__("hashlib").sha256(
                       _validator().EVIDENCE.read_bytes()).hexdigest()}}
        row.update(extra)
        return row

    def test_missing_declaration_is_a_violation(self):
        row = self._row("cpu-like embedded", "allowed")
        del row["conversion_to_cycles"]
        violations, _s = self.module.validate({"sources": [{"parameters": [row]}]})
        self.assertTrue(any("conversion_to_cycles" in v for v in violations), violations)

    def test_accelerator_without_rule_cannot_claim_conversion(self):
        violations, _s = self.module.validate(
            {"sources": [{"parameters": [self._row("Jetson Orin Nano GPU", "allowed")]}]})
        self.assertTrue(any("conversion_rule_id" in v for v in violations), violations)

    def test_accelerator_with_a_rule_is_accepted(self):
        violations, _s = self.module.validate(
            {"sources": [{"parameters": [self._row("Jetson Orin Nano GPU", "allowed",
                                                   conversion_rule_id="GPU-TO-CYCLES-V1")]}]})
        self.assertEqual(violations, [])

    def test_forbidden_conversion_is_accepted(self):
        violations, _s = self.module.validate(
            {"sources": [{"parameters": [self._row("Jetson Orin Nano GPU", "forbidden")]}]})
        self.assertEqual(violations, [])


class TestTranscriptionLedger(unittest.TestCase):
    """A verified row must carry a ledger, and the ledger must not go stale."""

    def setUp(self):
        self.module = _validator()

    def _row(self, **extra):
        row = {"parameter_name": "x", "required_by_generator": True,
               "source_verification": "official_verified", "transcription_verified": True,
               "measurement_scope": "communication", "semantic_role": "requirement",
               "allowed_for_generation": True}
        row.update(extra)
        return row

    def test_verified_row_without_ledger_is_a_violation(self):
        violations, _s = self.module.validate({"sources": [{"parameters": [self._row()]}]})
        self.assertTrue(any("without an explicit ledger" in v for v in violations), violations)

    def test_ledger_missing_a_field_is_a_violation(self):
        row = self._row(transcription_ledger={"document_version": "TS 22.186 V16.2.0",
                                              "section": "5.3", "table": "Table 5.3-1",
                                              "requirement_id": "R.5.3-001", "column": "x",
                                              "value": "10"})   # unit + evidence_sha256 absent
        violations, _s = self.module.validate({"sources": [{"parameters": [row]}]})
        self.assertTrue(any("ledger missing unit" in v for v in violations), violations)
        self.assertTrue(any("ledger missing evidence_sha256" in v for v in violations), violations)

    def test_stale_evidence_reference_is_a_violation(self):
        row = self._row(transcription_ledger={"document_version": "TS 22.186 V16.2.0",
                                              "section": "5.3", "table": "Table 5.3-1",
                                              "requirement_id": "R.5.3-001", "column": "x",
                                              "value": "10", "unit": "ms",
                                              "evidence_sha256": "0" * 64})
        violations, _s = self.module.validate({"sources": [{"parameters": [row]}]})
        self.assertTrue(any("stale evidence" in v for v in violations), violations)


class TestPin(unittest.TestCase):
    def test_real_pin_is_fresh(self):
        import hashlib

        from pathlib import Path as _P

        reg = _P(__file__).resolve().parents[4] / "spec" / "automotive_mc_v1" / "SOURCE_REGISTRY.yaml"
        if not reg.exists():
            self.skipTest("registry not present")
        violations, stats = _validator().check_pin(hashlib.sha256(reg.read_bytes()).hexdigest())
        self.assertEqual(violations, [])
        self.assertFalse(stats["registry_pin_stale"])

    def test_wrong_hash_is_stale(self):
        violations, stats = _validator().check_pin("0" * 64)
        self.assertTrue(stats["registry_pin_stale"])
        self.assertTrue(any("registry_pin_stale" in v for v in violations))


if __name__ == "__main__":
    unittest.main(verbosity=2)
