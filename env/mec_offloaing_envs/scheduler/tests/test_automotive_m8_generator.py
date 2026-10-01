#!/usr/bin/env python3
"""M8 focused + adversarial tests: deterministic generator and materialization."""

from __future__ import annotations

import copy
import importlib.util
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "spec" / "automotive_mc_v1"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BASE))

_DOCS = None
_GRAPHS = None


def _mod(name):
    spec = importlib.util.spec_from_file_location(name, BASE / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def docs():
    global _DOCS
    if _DOCS is None:
        _DOCS = _mod("automotive_dataset").load_inputs()
    return _DOCS


def graphs():
    global _GRAPHS
    if _GRAPHS is None:
        _GRAPHS = _mod("automotive_generator").generate_all(docs())
    return _GRAPHS


class TestGeneratorInvariants(unittest.TestCase):
    def setUp(self):
        self.ad = _mod("automotive_dataset")
        self.ag = _mod("automotive_generator")
        self.cm = _mod("criticality_model")
        self.d = docs()
        self.gs = graphs()

    def test_population_shape(self):
        self.assertEqual(len(self.gs), 240)
        self.assertEqual(len({g["application_family"] for g in self.gs}), 4)
        for g in self.gs:
            self.assertEqual(len(g["tasks"]), 20, g["graph_id"])
            self.assertEqual(len(g["edges"]), g["edge_count"])

    def test_dag_is_acyclic_and_topo_ordered(self):
        for g in self.gs[:40]:
            rank = {int(t["task_id"]): i for i, t in enumerate(g["tasks"])}
            for e in g["edges"]:
                self.assertLess(rank[e["src"]], rank[e["dst"]])

    def test_refs_resolve(self):
        sem = self.d["task_semantics.yaml"]
        classes = set(sem["payload_classes"])
        models = sem["payload_models"]
        roles = {r["semantic_role"]: r for r in sem["roles"]}
        for g in self.gs[:40]:
            self.assertIn(g["template_id"], {t["template_id"] for t in
                                             self.d["application_templates.yaml"]["templates"]})
            self.assertIn(g["resource_profile"], self.d["resource_profiles.yaml"]["profiles"])
            self.assertIn(g["sla_id"], {f["deadline_rule_id"] for f in
                                        self.d["sla_registry_v2.yaml"]["deadline_registry"]["families"].values()})
            for e in g["edges"]:
                self.assertIn(e["payload_class"], classes)
                self.assertEqual(e["payload_model_ref"], models[e["payload_class"]])
            for t in g["tasks"]:
                self.assertIn(t["semantic_role"], roles)
                self.assertEqual(t["motif_id"], roles[t["semantic_role"]]["motif_role"])
                self.assertEqual(t["lineage_id"], f"{g['template_id']}#{t['task_id']}")

    def test_criticality_comes_from_semantics_not_topology(self):
        for g in self.gs[::37]:
            family = g["application_family"]
            for t in g["tasks"]:
                expected = self.cm.classify(self.d["criticality_policy.yaml"],
                                            t["semantic_role"], family)["criticality"]
                self.assertEqual(t["criticality"], expected)

    def test_workload_is_action_and_profile_invariant(self):
        w2 = self.d["workload_model_v2.yaml"]
        f_ref = float(w2["reference_compute_model"]["f_ref_hz"])
        xi = float(w2["reference_compute_model"]["xi_cycles_per_bit"])
        for g in self.gs[::53]:
            probe = copy.deepcopy(g)
            probe["resource"] = {k: v * 3.0 for k, v in g["resource"].items()}
            for t in probe["tasks"]:
                self.assertEqual(t["compute_workload_bytes"],
                                 g["tasks"][t["task_id"]]["compute_workload_bytes"])
            for t in g["tasks"]:
                for forbidden in ("location", "action", "placement", "tier"):
                    self.assertNotIn(forbidden, t)
                self.assertEqual(int(t["compute_workload_bytes"]),
                                 int(round(float(t["t_ref_s"]) * f_ref / (8.0 * xi))))

    def test_compute_and_payload_do_not_mix(self):
        for g in self.gs[::31]:
            for e in g["edges"]:
                drawn = g["raw_draws"]["payload"][e["payload_class"]]["bytes"]
                self.assertEqual(int(e["payload_bytes"]), int(drawn))
            # changing the payload draw must not move any W_i, and vice versa
            probe = copy.deepcopy(g)
            for cls in probe["raw_draws"]["payload"]:
                probe["raw_draws"]["payload"][cls]["bytes"] = 1
            for t, t2 in zip(g["tasks"], probe["tasks"]):
                self.assertEqual(t["compute_workload_bytes"], t2["compute_workload_bytes"])

    def test_deadlines_are_the_frozen_construction_of_the_declared_D_G(self):
        for g in self.gs[::29]:
            sla = self.d["sla_registry_v2.yaml"]["deadline_registry"]["families"][g["application_family"]]
            lo, hi = (float(x) for x in sla["D_f_s_range"])
            self.assertAlmostEqual(float(g["D_G_s"]),
                                   lo + float(g["raw_draws"]["deadline_quantile"]) * (hi - lo),
                                   places=12)
            again = self.ad.deadlines_with_D_G(
                next(t for t in self.d["application_templates.yaml"]["templates"]
                     if t["family_id"] == g["application_family"]),
                self.d["workload_model_v2.yaml"],
                self.d["sla_registry_v2.yaml"], float(g["D_G_s"]))
            for t in g["tasks"]:
                r = again["tasks"][int(t["task_id"])]
                for field in ("E_s", "L_s", "deadline_s", "slack_s"):
                    self.assertAlmostEqual(float(t[field]), float(r[field]), places=15)

    def test_changing_D_G_moves_only_the_deadline_side(self):
        g = next(x for x in self.gs if x["application_family"] == "mapping_background"
                 and x["sla_regime"] == "loose")
        tpl = next(t for t in self.d["application_templates.yaml"]["templates"]
                   if t["family_id"] == g["application_family"])
        w2 = self.d["workload_model_v2.yaml"]
        f_ref = float(w2["reference_compute_model"]["f_ref_hz"])
        xi = float(w2["reference_compute_model"]["xi_cycles_per_bit"])
        probe = self.ad.deadlines_with_D_G(tpl, w2, self.d["sla_registry_v2.yaml"],
                                           float(g["D_G_s"]) * 0.5)
        self.assertEqual(probe["status"], "ok")
        for t in g["tasks"]:
            r = probe["tasks"][int(t["task_id"])]
            # E_i is a forward bound: it cannot depend on D_G at all
            self.assertAlmostEqual(r["E_s"], float(t["E_s"]), places=15)
            self.assertLessEqual(r["L_s"], float(t["L_s"]) + 1e-12)
            # W_i is a property of the task, never of the deadline
            self.assertEqual(int(t["compute_workload_bytes"]),
                             int(round(float(t["t_ref_s"]) * f_ref / (8.0 * xi))))
        for e in g["edges"]:
            drawn = g["raw_draws"]["payload"][e["payload_class"]]["bytes"]
            self.assertEqual(int(e["payload_bytes"]), int(drawn))

    def test_mc_budgets_are_valid_and_high_exists(self):
        high = 0
        for g in self.gs:
            high += int(g["criticality_counts"].get("HIGH", 0))
            for t in g["tasks"]:
                if t["criticality"] == "HIGH":
                    self.assertGreater(float(t["empirical_execution_budget_lo_s"]), 0.0)
                    self.assertGreaterEqual(float(t["empirical_execution_budget_hi_s"]),
                                            float(t["empirical_execution_budget_lo_s"]))
                    self.assertFalse(t["wcet_claim"])
                else:
                    self.assertIsNone(t["empirical_execution_budget_hi_s"])
        self.assertGreater(high, 0)

    def test_payload_classes_are_correct_and_disjoint(self):
        bands = self.ad.realised_payload_bands(self.gs)
        classes = set(self.d["task_semantics.yaml"]["payload_classes"])
        self.assertEqual(set(bands), classes)
        ordered = sorted(bands)
        for i, a in enumerate(ordered):
            for b in ordered[i + 1:]:
                self.assertTrue(bands[a][1] < bands[b][0] or bands[b][1] < bands[a][0],
                                f"{a}{bands[a]} overlaps {b}{bands[b]}")

    def test_canonical_hash_recomputes_and_is_stable(self):
        for g in self.gs[::17]:
            body = {k: v for k, v in g.items() if k != "canonical_sha256"}
            self.assertEqual(self.ad.sha256_json(body), g["canonical_sha256"])
        again = self.ag.generate_all(self.d)
        self.assertEqual([g["canonical_sha256"] for g in again],
                         [g["canonical_sha256"] for g in self.gs])
        self.assertEqual([g["raw_sha256"] for g in again], [g["raw_sha256"] for g in self.gs])

    def test_mode_semantics_annotation_is_complete(self):
        for g in self.gs[::13]:
            ms = g["mode_semantics"]
            self.assertEqual(ms["initial_mode"], "LO")
            self.assertEqual(ms["hi_exit"], "sticky_until_graph_completion")
            self.assertTrue(ms["high_never_silently_dropped"])
            for cls in ("LOW", "MEDIUM", "HIGH"):
                for mode in ("lo", "hi"):
                    self.assertIn(f"drop_allowed_{mode}_mode", ms["drop_degrade_policy"][cls])


class TestMaterializationDeterminism(unittest.TestCase):
    def setUp(self):
        self.ag = _mod("automotive_generator")
        self.ad = _mod("automotive_dataset")
        self.d = docs()

    def test_two_clean_directories_are_byte_identical(self):
        a = Path(tempfile.mkdtemp(prefix="m8_det_a_"))
        b = Path(tempfile.mkdtemp(prefix="m8_det_b_"))
        try:
            ra = self.ag.materialize(a, self.d)
            rb = self.ag.materialize(b, self.d)
            self.assertEqual(ra["graph_count"], rb["graph_count"])
            self.assertEqual(ra["graphs_sha256"], rb["graphs_sha256"])
            self.assertEqual(ra["dataset_manifest_sha256"], rb["dataset_manifest_sha256"])
            for name in self.ag.MATERIALIZED_FILES:
                self.assertEqual((a / name).read_bytes(), (b / name).read_bytes(), name)
        finally:
            shutil.rmtree(a, ignore_errors=True)
            shutil.rmtree(b, ignore_errors=True)

    def test_on_disk_dataset_is_not_stale(self):
        path = self.ad.DATASET_DIR / "graphs.jsonl"
        if not path.exists():
            self.skipTest("dataset not materialized in this checkout")
        fresh = "".join(self.ad.canonical_json(g) + "\n" for g in self.ag.generate_all(self.d))
        on_disk = path.read_text()
        self.assertEqual(self.ad.sha256_text(on_disk), self.ad.sha256_text(fresh),
                         "materialized dataset is stale; rerun automotive_generator.py")

    def test_generation_config_declares_no_timestamps(self):
        cfg = self.ad.load_inputs()["generation_config.yaml"]
        self.assertTrue(cfg["materialization"]["no_timestamps"])
        self.assertTrue(cfg["materialization"]["no_random_ordering"])
        self.assertEqual(cfg["topology_mutation"], "none")


if __name__ == "__main__":
    unittest.main(verbosity=2)
