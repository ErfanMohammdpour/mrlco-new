#!/usr/bin/env python3
"""Step 3 lints: the dataset card/contract must not contradict machine-checked facts."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "spec" / "automotive_mc_v1"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _validator():
    spec = importlib.util.spec_from_file_location(
        "validate_source_registry", BASE / "validate_source_registry.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestStep2GateStillPasses(unittest.TestCase):
    def test_registry_and_manifest_consistent_and_pinned(self):
        import yaml

        module = _validator()
        doc = yaml.safe_load((BASE / "SOURCE_REGISTRY.yaml").read_text())
        rows = {q["parameter_name"]: q for s in doc["sources"] for q in (s.get("parameters") or [])}
        man = yaml.safe_load((BASE / "REQUIRED_PARAMETER_MANIFEST.yaml").read_text())
        row_v, _rs = module.validate(doc)
        man_v, stats = module.validate_manifest(man, rows)
        pin_v, pin = module.check_pin(
            hashlib.sha256((BASE / "SOURCE_REGISTRY.yaml").read_bytes()).hexdigest())
        self.assertEqual(row_v + man_v + pin_v, [])
        self.assertEqual(stats["missing_required_parameters"], 0)
        self.assertEqual(stats["blocked_required_parameters"], 0)
        self.assertFalse(pin["registry_pin_stale"])


class TestCardMatchesEvidence(unittest.TestCase):
    def setUp(self):
        import yaml

        self.registry = yaml.safe_load((BASE / "SOURCE_REGISTRY.yaml").read_text())
        self.rows = {q["parameter_name"]: q for s in self.registry["sources"]
                     for q in (s.get("parameters") or [])}
        self.manifest = yaml.safe_load((BASE / "REQUIRED_PARAMETER_MANIFEST.yaml").read_text())
        self.card = (BASE / "DATASET_CARD.md").read_text()
        self.contract = (BASE / "DATASET_CONTRACT.md").read_text()

    def test_card_declares_nature_and_non_claims(self):
        low = self.card.lower()
        self.assertIn("margo-automotive-mc-v1", low)
        self.assertIn("synthetic", low)
        self.assertIn("not", low)
        self.assertIn("measured production automotive", low)

    def test_reference_tier_matches_registry(self):
        row = self.rows["ref_tier_i9_12900hx"]
        self.assertEqual(str(row["value"]), "2.30e9")
        self.assertEqual(row["hardware_class"], "CPU")
        self.assertEqual(row["source_verification"], "peer_reviewed_verified")
        compact = self.card.replace(" ", "")
        self.assertIn("2.30e9", compact)
        self.assertIn("i9-12900HX", self.card)

    def test_xi_is_a_simulation_parameter(self):
        row = self.rows["cycles_per_bit_xu"]
        self.assertEqual(str(row["value"]), "300.0")
        self.assertEqual(row.get("parameter_class"), "simulation_parameter")
        self.assertIn("simulation parameter", self.card.lower())

    def test_families_and_rho_only_branch(self):
        families = self.manifest["application_families"]
        self.assertEqual(len(families), 4)
        ids = {e.get("id") for e in self.manifest["required_parameters"]}
        for family in families:
            self.assertIn(f"e2e_sla_{family}", ids)
        self.assertIn("D_G = rho_f * P_f", self.card)
        self.assertIn("is **not** active in v1", self.card)
        self.assertIn("rho_f * P_f", self.contract)

    def test_contract_locks_the_invariants(self):
        for heading in ("Compute demand is not communication payload",
                        "Reference tier coupling",
                        "What the reference execution tier is"):
            self.assertIn(heading, self.contract)

    def test_hashes_are_named_unambiguously(self):
        self.assertNotIn("manifest_sha:", self.card)
        self.assertIn("dataset_manifest_sha", self.card)
        self.assertIn("required_parameter_manifest_sha", self.card)


if __name__ == "__main__":
    unittest.main(verbosity=2)
