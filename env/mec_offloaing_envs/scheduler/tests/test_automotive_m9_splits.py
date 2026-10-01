#!/usr/bin/env python3
"""M9 focused + adversarial tests: generating-factor splits and leakage detection."""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
import unittest
from pathlib import Path

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


class TestSplits(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.asp = _mod("automotive_splits")
        cls.ad = _mod("automotive_dataset")
        cls.graphs, cls.dir = cls.asp.load_graphs()
        cls.policy = json.loads((BASE / "split_policy.json").read_text())
        cls.res = cls.asp.run(cls.dir, check_only=True)
        cls.assignment = cls.res["assignment"]

    def test_every_graph_has_exactly_one_role(self):
        self.assertEqual(len(self.assignment), len(self.graphs))
        for g in self.graphs:
            a = self.assignment[g["graph_id"]]
            self.assertIn(a["split"], ("meta_train", "validation", "meta_test"))
            self.assertIn(a["role_in_split"], ("train", "support", "query"))

    def test_split_counts_and_support_query(self):
        summary = self.res["summary"]
        self.assertEqual(summary["counts"]["meta_train"], 120)
        self.assertEqual(summary["counts"]["validation"], 60)
        self.assertEqual(summary["counts"]["meta_test"], 60)
        self.assertEqual(summary["role_counts"]["validation/support"], 20)
        self.assertEqual(summary["role_counts"]["validation/query"], 40)
        self.assertEqual(summary["role_counts"]["meta_test/support"], 20)
        self.assertEqual(summary["role_counts"]["meta_test/query"], 40)

    def test_support_and_query_are_disjoint(self):
        for split in ("validation", "meta_test"):
            support = {gid for gid, a in self.assignment.items()
                       if a["split"] == split and a["role_in_split"] == "support"}
            query = {gid for gid, a in self.assignment.items()
                     if a["split"] == split and a["role_in_split"] == "query"}
            self.assertFalse(support & query)
            self.assertEqual(len(support | query), 60)

    def test_no_leakage_on_any_declared_key(self):
        self.assertEqual(self.asp.leakage_checks(self.graphs, self.assignment, self.policy), [])
        for key in self.policy["leakage_keys"]:
            buckets: dict[str, set] = {}
            for g in self.graphs:
                value = (self.asp.near_duplicate_signature(g)
                         if key == "near_duplicate_signature" else str(g[key]))
                buckets.setdefault(value, set()).add(self.assignment[g["graph_id"]]["split"])
            for value, splits in buckets.items():
                self.assertEqual(len(splits), 1, f"{key}={value} crosses {splits}")

    def test_sibling_groups_never_cross(self):
        groups: dict[int, set] = {}
        for g in self.graphs:
            groups.setdefault(g["parent_seed"], set()).add(self.assignment[g["graph_id"]]["split"])
        self.assertTrue(all(len(v) == 1 for v in groups.values()))

    def test_calibration_and_meta_test_isolation(self):
        summary = self.res["summary"]
        self.assertEqual(summary["calibration_source"], "meta_train")
        self.assertFalse(summary["meta_test_isolation"]["opened"])
        for gid, a in self.assignment.items():
            if a["split"] == "meta_train":
                self.assertEqual(a["role_in_split"], "train")

    def test_assignment_is_deterministic(self):
        again = self.asp.build_assignment(self.graphs, self.policy)
        self.assertEqual({k: (v["split"], v["role_in_split"]) for k, v in again.items()},
                         {k: (v["split"], v["role_in_split"]) for k, v in self.assignment.items()})

    # ---- adversarial: the checker must actually fire --------------------------
    def test_checker_detects_a_sibling_crossing(self):
        broken = copy.deepcopy(self.assignment)
        victim = next(g for g in self.graphs if g["application_family"] == "mapping_background")
        broken[victim["graph_id"]] = {"split": "meta_test", "role_in_split": "support",
                                      "stratum": None, "assignment_rank": None}
        violations = self.asp.leakage_checks(self.graphs, broken, self.policy)
        self.assertTrue(any("leakage" in v for v in violations), violations)
        self.assertTrue(any("parent_seed" in v for v in violations), violations)

    def test_checker_detects_support_query_overlap(self):
        broken = copy.deepcopy(self.assignment)
        target = next(gid for gid, a in broken.items()
                      if a["split"] == "validation" and a["role_in_split"] == "query")
        broken[target]["role_in_split"] = "support"
        violations = self.asp.leakage_checks(self.graphs, broken, self.policy)
        self.assertTrue(any("support count" in v or "overlap" in v for v in violations), violations)

    def test_checker_detects_a_missing_split_role(self):
        broken = copy.deepcopy(self.assignment)
        for gid, a in broken.items():
            if a["split"] == "meta_test":
                a["split"] = "meta_train"
                a["role_in_split"] = "train"
        violations = self.asp.leakage_checks(self.graphs, broken, self.policy)
        self.assertTrue(any("meta_test" in v for v in violations), violations)

    def test_checker_detects_opened_meta_test(self):
        policy = copy.deepcopy(self.policy)
        policy["meta_test_isolation"]["opened"] = True
        violations = self.asp.leakage_checks(self.graphs, self.assignment, policy)
        self.assertTrue(any("opened" in v for v in violations), violations)

    def test_clean_policy_passes(self):
        self.assertEqual(self.asp.leakage_checks(self.graphs, self.assignment, self.policy), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
