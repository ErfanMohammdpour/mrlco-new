#!/usr/bin/env python3
"""M10 focused + adversarial tests: certification statuses, bound validity, read-only."""

from __future__ import annotations

import copy
import importlib.util
import statistics
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "spec" / "automotive_mc_v1"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BASE))

ALLOWED = ("certified_feasible", "witness_not_found", "stress_or_infeasible")


def _mod(name):
    spec = importlib.util.spec_from_file_location(name, BASE / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


class TestCertification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cm = _mod("certify_automotive_m10")
        cls.ad = _mod("automotive_dataset")
        cls.asp = _mod("automotive_splits")
        cls.docs = cls.ad.load_inputs()
        cls.graphs, cls.dir = cls.asp.load_graphs()
        cls.ref_cp = {fam: cls.ad.reference_tier_critical_path(tpl, cls.docs["workload_model_v2.yaml"])
                      for fam, tpl in cls.ad.templates_by_family(cls.docs).items()}
        cls.sample = cls.graphs[::30][:8]

    def _cert(self, g):
        return self.cm.certify_graph(copy.deepcopy(g), self.ref_cp[g["application_family"]])

    def test_statuses_are_exactly_the_three_declared_ones(self):
        seen = set()
        for g in self.sample:
            r = self._cert(g)
            self.assertIn(r["status"], ALLOWED)
            seen.add(r["status"])
        self.assertTrue(seen <= set(ALLOWED))

    def test_lower_bound_never_exceeds_a_feasible_schedule(self):
        for g in self.sample:
            r = self._cert(g)
            for name, m in r["methods"].items():
                self.assertLessEqual(r["lower_bound_makespan_s"],
                                     m["makespan_s"] + 1e-12,
                                     f"{g['graph_id']}/{name}: schedule below the lower bound")

    def test_d_G_below_the_bound_is_stress_not_witness_not_found(self):
        g = self.sample[0]
        probe = copy.deepcopy(g)
        probe["D_G_s"] = self.cm.lower_bound_makespan(g) * 0.5
        r = self.cm.certify_graph(probe, self.ref_cp[g["application_family"]])
        self.assertEqual(r["status"], "stress_or_infeasible")
        self.assertNotEqual(r["status"], "witness_not_found")

    def test_no_witness_is_not_reported_as_infeasibility(self):
        g = next(x for x in self.graphs if x["application_family"] == "cooperative_perception")
        probe = copy.deepcopy(g)
        probe["D_G_s"] = self.cm.lower_bound_makespan(g) * 1.000001
        r = self.cm.certify_graph(probe, self.ref_cp[g["application_family"]])
        self.assertEqual(r["status"], "witness_not_found")
        self.assertNotEqual(r["status"], "stress_or_infeasible")
        self.assertFalse(r["status"] == "stress_or_infeasible")

    def test_rho_G_is_diagnostic_and_never_redefines_D_G(self):
        for g in self.sample:
            r = self._cert(g)
            self.assertAlmostEqual(r["rho_G"], r["best_makespan_s"] / g["D_G_s"], places=12)
            self.assertEqual(r["D_G_s"], float(g["D_G_s"]))

    def test_certification_is_read_only_in_memory(self):
        g = self.sample[1]
        snapshot = self.ad.sha256_json(g)
        self._cert(g)
        self.assertEqual(self.ad.sha256_json(g), snapshot)

    def test_certification_is_deterministic(self):
        g = self.sample[2]
        a = self._cert(g)
        b = self._cert(g)
        self.assertEqual(a["status"], b["status"])
        self.assertEqual(a["best_makespan_s"], b["best_makespan_s"])
        self.assertEqual(a["best_method"], b["best_method"])
        self.assertEqual(a["methods"]["mixed_search"]["plan"],
                         b["methods"]["mixed_search"]["plan"])

    def test_certified_feasible_implies_a_real_witness(self):
        for g in self.sample:
            r = self._cert(g)
            if r["status"] == "certified_feasible":
                self.assertLessEqual(r["best_makespan_s"], g["D_G_s"] + 1e-12)

    def test_mixed_placement_is_evaluated_and_compared_with_pure_plans(self):
        for g in self.sample:
            r = self._cert(g)
            self.assertIn("mixed_search", r["methods"])
            pure_best = r["pure_best_makespan_s"]
            self.assertLessEqual(r["best_makespan_s"], pure_best + 1e-12)
            self.assertEqual(r["pure_best_method"],
                             min((k for k in r["methods"] if r["methods"][k]["is_pure"]),
                                 key=lambda k: r["methods"][k]["makespan_s"]))

    def test_reference_tier_stress_flag_is_separate_from_status(self):
        g = self.sample[0]
        probe = copy.deepcopy(g)
        probe["D_G_s"] = self.ref_cp[g["application_family"]] * 0.5
        r = self.cm.certify_graph(probe, self.ref_cp[g["application_family"]])
        self.assertTrue(r["reference_tier_stress"])
        # the status must follow the TRUE bound, not the reference-tier critical path
        if r["D_G_s"] >= r["lower_bound_makespan_s"]:
            self.assertIn(r["status"], ("witness_not_found", "certified_feasible"))
        else:
            self.assertEqual(r["status"], "stress_or_infeasible")

    def test_search_budget_is_bounded(self):
        g = self.sample[3]
        r = self._cert(g)
        self.assertLessEqual(r["search_evaluations"], self.cm.SEARCH_BUDGET)

    def test_frozen_dataset_certification_rates_sum_to_one(self):
        rows = [self._cert(g) for g in self.sample]
        rates = {s: sum(1 for r in rows if r["status"] == s) / len(rows) for s in ALLOWED}
        self.assertAlmostEqual(sum(rates.values()), 1.0, places=12)
        self.assertIsInstance(statistics.median([r["rho_G"] for r in rows]), float)


if __name__ == "__main__":
    unittest.main(verbosity=2)
