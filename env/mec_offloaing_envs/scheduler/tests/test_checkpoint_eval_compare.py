#!/usr/bin/env python3
"""P1/P3: pure comparison + verdict builder for checkpoint evaluations."""

from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
for _name in ("gym", "gym.core"):
    sys.modules.setdefault(_name, types.ModuleType(_name))
sys.modules["gym.core"].Env = type("Env", (), {})


from spec.checkpoint_eval_compare import (  # noqa: E402
    LABELS,
    build,
    checks_and_verdict,
    itr0_row,
    label_row,
    main,
)


def _doc(label, k0, k3, *, all_mec=630.0, greedy=626.0, changed=True, det=True):
    return {
        "schema": "checkpoint_eval_v1",
        "deterministic_k0_fresh": det,
        "training_code_sha": "train-sha",
        "evaluation_code_sha": "eval-sha",
        "checkpoints": {
            label: {
                "k0": {"query_mean_latency": k0},
                "k3": {"query_mean_latency": k3,
                       "query_all_mec_latency": all_mec,
                       "query_greedy_latency": greedy},
                "weights_changed": changed,
                "checkpoint_sha256": "deadbeef",
            }
        },
    }


class TestRows(unittest.TestCase):
    def test_label_row_extracts_latency_and_gaps(self):
        row = label_row("final_itr24", _doc("final_itr24", 776.0, 795.0))
        self.assertAlmostEqual(row["k0"], 776.0)
        self.assertAlmostEqual(row["k3"], 795.0)
        self.assertAlmostEqual(row["gap_to_all_mec"], 165.0)
        self.assertAlmostEqual(row["gap_to_greedy"], 169.0)
        self.assertFalse(row["k3_better_than_k0"])
        self.assertTrue(row["weights_changed"])

    def test_missing_label_raises(self):
        with self.assertRaises(KeyError):
            label_row("absent", _doc("present", 1.0, 2.0))

    def test_itr0_row_reads_the_pilot_csv(self):
        header = ("Itr,validation_query_mean_latency_k0,validation_query_mean_latency_k3,"
                  "validation/validation_all_mec_latency,validation/validation_greedy_latency\n")
        body = "0,982.8035,893.3548,630.2798,626.6017\n50,970.0,880.0,630.2798,626.6017\n"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "progress.csv"
            path.write_text(header + body)
            row = itr0_row(path)
        self.assertEqual(row["label"], "itr0_from_pilot_a")
        self.assertAlmostEqual(row["k3"], 893.3548)
        self.assertAlmostEqual(row["gap_to_all_mec"], 893.3548 - 630.2798)

    def test_itr0_row_without_validation_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "progress.csv"
            path.write_text("Itr,validation_query_mean_latency_k3\n0,\n")
            with self.assertRaises(ValueError):
                itr0_row(path)


class TestVerdict(unittest.TestCase):
    def _rows(self, cand_k0, cand_k3, changeds=None, dets=None, baselines=None):
        docs = [("true_init", _doc("true_init", 1136.7784, 943.4189, changed=None))]
        docs.append(("final_itr24", _doc("final_itr24", cand_k0, cand_k3, changed=True)))
        if changeds is not None:
            for (label, doc), changed in zip(docs[1:], changeds):
                doc["checkpoints"][label]["weights_changed"] = changed
        if dets is not None:
            for (label, doc), det in zip(docs[1:], dets):
                doc["deterministic_k0_fresh"] = det
        if baselines is not None:
            for (label, doc), base in zip(docs[1:], baselines):
                ck = doc["checkpoints"][label]["k3"]
                ck["query_all_mec_latency"] = base
        return build(docs)

    def test_ready_requires_improvement_and_healthy_k3(self):
        out = self._rows(800.0, 700.0)
        self.assertTrue(out["improved_over_true_init"])
        self.assertTrue(out["k3_better_than_k0"])
        self.assertEqual(out["verdict"], LABELS[1])
        self.assertAlmostEqual(out["gain_vs_true_init"], 943.4189 - 700.0)

    def test_improved_but_unhealthy_k3_is_not_ready(self):
        out = self._rows(776.0014, 794.9655)
        self.assertTrue(out["improved_over_true_init"])
        self.assertFalse(out["k3_better_than_k0"])
        self.assertEqual(out["verdict"], LABELS[2])
        self.assertIn("k3 adaptation", out["verdict_detail"])

    def test_no_improvement_is_not_ready(self):
        out = self._rows(950.0, 960.0)
        self.assertFalse(out["improved_over_true_init"])
        self.assertEqual(out["verdict"], LABELS[2])

    def test_unchanged_weights_block(self):
        out = self._rows(800.0, 700.0, changeds=[False])
        self.assertFalse(out["checks"]["weights_changed"])
        self.assertEqual(out["verdict"], LABELS[0])

    def test_nondeterministic_eval_blocks(self):
        out = self._rows(800.0, 700.0, dets=[False])
        self.assertEqual(out["verdict"], LABELS[0])

    def test_inconsistent_baselines_block(self):
        out = self._rows(800.0, 700.0, baselines=[600.0])
        self.assertFalse(out["checks"]["baselines_consistent_across_labels"])
        self.assertEqual(out["verdict"], LABELS[0])

    def test_nonfinite_blocks(self):
        out = self._rows(800.0, float("inf"))
        self.assertEqual(out["verdict"], LABELS[0])

    def test_checks_and_verdict_matches_build(self):
        rows = [label_row("true_init", _doc("true_init", 1136.0, 943.0, changed=None)),
                label_row("final_itr24", _doc("final_itr24", 800.0, 700.0))]
        self.assertEqual(checks_and_verdict(rows)["verdict"], LABELS[1])


class TestRecoveredNumbers(unittest.TestCase):
    """The numbers recovered from the first attempt's stdout logs.

    Not evidence in themselves - the persisted JSON still has to be regenerated -
    but they pin the verdict the README documents for that evidence.
    """

    def _doc_for(self, label, k0, k3):
        return _doc(label, k0, k3, all_mec=630.2798, greedy=626.6017)

    def test_recovered_table_yields_long_run_allowed_but_not_ready(self):
        docs = [
            ("true_init", self._doc_for("true_init", 1136.7784, 943.4189)),
            ("final_itr24", self._doc_for("final_itr24", 776.0014, 794.9655)),
            ("final_itr39", self._doc_for("final_itr39", 834.1585, 842.8307)),
        ]
        out = build(docs)
        self.assertTrue(out["checks"]["all_finite"])
        self.assertTrue(out["checks"]["baselines_consistent_across_labels"])
        self.assertTrue(out["improved_over_true_init"])
        self.assertEqual(out["best_candidate"], "final_itr24")
        self.assertAlmostEqual(out["gain_vs_true_init"], 943.4189 - 794.9655)
        self.assertFalse(out["k3_better_than_k0"])
        self.assertEqual(out["verdict"], LABELS[2])

    def test_recovered_true_init_reruns_agree(self):
        first = label_row("true_init", self._doc_for("true_init", 1136.7784, 943.4189))
        repeat = label_row("true_init", self._doc_for("true_init", 1136.7784, 943.4189))
        self.assertEqual(first["k0"], repeat["k0"])
        self.assertEqual(first["k3"], repeat["k3"])


def _doc_with_objective(label, k0, k3, obj_k0, obj_k3, *, all_mec=630.2798, greedy=626.6017,
                        changed=True, det=True):
    """A post-contract document: carries query_discounted_return at k0 and k3."""
    doc = _doc(label, k0, k3, all_mec=all_mec, greedy=greedy, changed=changed, det=det)
    doc["checkpoints"][label]["k0"]["query_discounted_return"] = obj_k0
    doc["checkpoints"][label]["k3"]["query_discounted_return"] = obj_k3
    return doc


class TestObjectiveCriterion(unittest.TestCase):
    """objective_contract_v1: the discounted return selects, latency annotates."""

    def test_row_uses_the_objective_when_present(self):
        row = label_row("final_itr24", _doc_with_objective(
            "final_itr24", 776.0, 795.0, -0.776, -0.795))
        self.assertEqual(row["objective_source"], "query_discounted_return")
        self.assertAlmostEqual(row["objective_k3"], -0.795)
        self.assertTrue(row["k3_better_than_k0"] is False)
        # the seconds companion is still there
        self.assertAlmostEqual(row["k3"], 795.0)

    def test_legacy_documents_fall_back_with_a_label(self):
        row = label_row("final_itr24", _doc("final_itr24", 776.0, 795.0))
        self.assertEqual(row["objective_source"], "mean_latency_fallback(-seconds:k0)+"
                                                  "mean_latency_fallback(-seconds:k3)")
        self.assertAlmostEqual(row["objective_k3"], -795.0)

    def test_verdict_follows_the_objective_not_the_latency(self):
        # candidate is FASTER in seconds than true_init but WORSE on the objective:
        # the contract must not call this an improvement.
        docs = [
            ("true_init", _doc_with_objective("true_init", 1136.0, 943.0, -1.136, -0.943,
                                              changed=None)),
            ("final_itr24", _doc_with_objective("final_itr24", 700.0, 690.0, -0.700, -0.990)),
        ]
        out = build(docs)
        self.assertFalse(out["improved_over_true_init"])
        self.assertEqual(out["verdict"], LABELS[2])
        self.assertIn("objective", out["verdict_detail"])

    def test_verdict_ready_when_the_objective_and_gap_improve_and_k3_helps(self):
        docs = [
            ("true_init", _doc_with_objective("true_init", 1136.0, 943.0, -1.136, -0.943,
                                              changed=None)),
            ("final_itr24", _doc_with_objective("final_itr24", 800.0, 700.0, -0.800, -0.700)),
        ]
        out = build(docs)
        self.assertTrue(out["improved_over_true_init"])
        self.assertTrue(out["k3_better_than_k0"])
        self.assertEqual(out["verdict"], LABELS[1])
        self.assertAlmostEqual(out["gain_vs_true_init"], -0.700 - (-0.943))
        # latency gain is reported separately, never used for the verdict
        self.assertAlmostEqual(out["latency_gain_vs_true_init_seconds"], 943.0 - 700.0)

    def test_build_declares_the_criterion(self):
        out = build([("final_itr24", _doc("final_itr24", 800.0, 700.0))])
        self.assertIn("objective_contract_v1", out["criterion"])


class TestCli(unittest.TestCase):
    def test_cli_merges_documents_and_persists_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p_init = root / "true_init.json"
            p_ckpt = root / "final_itr24.json"
            p_init.write_text(json.dumps(_doc("true_init", 1136.7784, 943.4189, changed=None)))
            p_ckpt.write_text(json.dumps(_doc("final_itr24", 776.0014, 794.9655)))
            out = root / "comparison.json"
            code = main([
                "--eval", "true_init=%s" % p_init,
                "--eval", "final_itr24=%s" % p_ckpt,
                "--json", str(out),
            ])
            self.assertEqual(code, 0)
            payload = json.loads(out.read_text())
        self.assertEqual(payload["schema"], "checkpoint_eval_comparison_v1")
        self.assertEqual([r["label"] for r in payload["rows"]],
                         ["true_init", "final_itr24"])
        self.assertEqual(payload["verdict"], LABELS[2])
        self.assertEqual(payload["training_code_sha"], "train-sha")

    def test_cli_rejects_a_bare_path(self):
        with self.assertRaises(SystemExit):
            main(["--eval", "no_equals.json", "--json", "/tmp/x.json"])

    def test_cli_requires_at_least_one_eval(self):
        with self.assertRaises(SystemExit):
            main(["--json", "/tmp/x.json"])

    def test_cli_code_sha_flags_override_the_documents(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p_ckpt = root / "final_itr24.json"
            p_ckpt.write_text(json.dumps(_doc("final_itr24", 800.0, 700.0)))
            out = root / "comparison.json"
            main([
                "--eval", "final_itr24=%s" % p_ckpt,
                "--training-code-sha", "merge-train-sha",
                "--evaluation-code-sha", "merge-eval-sha",
                "--json", str(out),
            ])
            payload = json.loads(out.read_text())
        self.assertEqual(payload["training_code_sha"], "merge-train-sha")
        self.assertEqual(payload["evaluation_code_sha"], "merge-eval-sha")

    def test_build_falls_back_to_the_document_shas(self):
        out = build([("final_itr24", _doc("final_itr24", 800.0, 700.0))])
        self.assertEqual(out["training_code_sha"], "train-sha")
        self.assertEqual(out["evaluation_code_sha"], "eval-sha")


if __name__ == "__main__":
    unittest.main(verbosity=2)
