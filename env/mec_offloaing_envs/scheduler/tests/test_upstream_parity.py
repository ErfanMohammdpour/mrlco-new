#!/usr/bin/env python3
"""Part C: the upstream-parity diagnostic harness produces valid, gated evidence.

These tests are CPU-only and touch nothing in production.  They assert the
fair-comparison contract itself, not just that the harness exits zero:

  * the harness runs and its evidence JSON is serializable;
  * every makespan is finite and non-negative;
  * only binary Local(0)/MEC(1) actions appear, and no Helper/V2V result does;
  * the canonical row is exactly a direct ``schedule()`` call;
  * one graph / one decoder order / one plan / identical rates for all three
    schedulers, and latency-only semantics (energy accounting cannot move it);
  * when the upstream checkout is present, the upstream scheduler's own numbers
    are used and the harness's legacy control reproduces them;
  * a missing upstream scheduler is reported loudly and never fabricated.
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env.mec_offloaing_envs.scheduler import ResourceConfig, schedule  # noqa: E402
from spec.upstream_parity import (  # noqa: E402
    BINARY_ACTIONS,
    FACTOR_KEYS,
    SCHEMA,
    UpstreamUnavailableError,
    build_evidence,
    canonical_resources,
    load_upstream_scheduler,
    main,
    reference_contract,
)

REQUIRED_TASK_KEYS = {
    "task_id", "action", "location",
    "upload_bytes", "upload_s", "compute_s",
    "dependency_transfer_bytes", "dependency_transfer_s",
    "download_bytes", "download_s",
    "dependency_wait_s", "ready_s", "start_s", "finish_s", "return_s",
}


class TestUpstreamParity(unittest.TestCase):
    """One evidence build shared by the contract assertions."""

    @classmethod
    def setUpClass(cls):
        cls.evidence = build_evidence()

    # -- harness ------------------------------------------------------------
    def test_harness_runs_and_schema_is_pinned(self):
        self.assertEqual(self.evidence["schema"], SCHEMA)
        self.assertIn("all_checks_pass", self.evidence)
        self.assertIn("checks", self.evidence)

    def test_all_contract_checks_pass(self):
        for name, ok in self.evidence["checks"].items():
            self.assertTrue(ok, "check failed: %s" % name)
        self.assertTrue(self.evidence["all_checks_pass"])

    def test_evidence_is_json_serializable(self):
        blob = json.dumps(self.evidence, allow_nan=False)
        roundtrip = json.loads(blob)
        self.assertEqual(roundtrip["schema"], SCHEMA)
        self.assertTrue(roundtrip["checks"]["json_serializable"])

    def test_required_top_level_keys(self):
        required = {
            "schema", "upstream_available", "upstream_root", "contract",
            "schedulers", "differences", "summary", "checks", "all_checks_pass",
        }
        self.assertTrue(required <= set(self.evidence), required - set(self.evidence))

    # -- makespans ----------------------------------------------------------
    def test_makespans_finite_and_non_negative(self):
        makespans = {
            name: sched["makespan_seconds"]
            for name, sched in self.evidence["schedulers"].items()
        }
        self.assertTrue(makespans)
        for name, value in makespans.items():
            self.assertTrue(math.isfinite(value), name)
            self.assertGreaterEqual(value, 0.0, name)

    def test_three_schedulers_present(self):
        self.assertTrue(
            {"upstream_original", "legacy_control", "canonical"}
            <= set(self.evidence["schedulers"])
        )

    # -- contract: binary only ---------------------------------------------
    def test_binary_local_mec_only(self):
        plan = self.evidence["contract"]["plan"]
        self.assertTrue(plan)
        actions = [int(a) for _, a in plan]
        self.assertTrue(set(actions) <= set(BINARY_ACTIONS), actions)
        self.assertTrue(self.evidence["contract"]["binary_actions_only"])

    def test_helper_and_v2v_absent_everywhere(self):
        for name, sched in self.evidence["schedulers"].items():
            for row in sched["tasks"]:
                self.assertIn(row["action"], BINARY_ACTIONS, name)
                self.assertIn(row["location"], ("UE", "MEC"), name)
                self.assertNotEqual(row["location"], "HELPER", name)
        for transfer in self.evidence["schedulers"]["canonical"]["transfers"]:
            self.assertNotEqual(transfer["hop"], "V2V", transfer)
            self.assertNotIn("HELPER", (transfer["src_location"], transfer["dst_location"]))

    # -- contract: one graph / one decoder order / one plan -----------------
    def test_one_graph_one_decoder_order_one_plan(self):
        graph, decoder_order, actions, _ = reference_contract()
        contract = self.evidence["contract"]
        self.assertGreater(contract["graph"]["task_count"], 0)
        self.assertEqual(
            [int(x) for x in contract["decoder_order"]],
            [int(x) for x in decoder_order],
        )
        self.assertEqual(
            [[int(t), int(a)] for t, a in contract["plan"]],
            [[int(t), int(a)] for t, a in zip(decoder_order, actions)],
        )
        # every scheduler saw exactly the same task set
        for sched in self.evidence["schedulers"].values():
            self.assertEqual(
                sorted(row["task_id"] for row in sched["tasks"]),
                sorted(int(t) for t in graph.tasks),
            )
        self.assertTrue(contract["identical_rates"]["identical"])

    def test_per_task_rows_are_complete_and_finite(self):
        for name, sched in self.evidence["schedulers"].items():
            for row in sched["tasks"]:
                self.assertTrue(REQUIRED_TASK_KEYS <= set(row), (name, row))
                for key in REQUIRED_TASK_KEYS - {"task_id", "action", "location"}:
                    value = row[key]
                    self.assertTrue(math.isfinite(value), (name, row["task_id"], key))
                    self.assertGreaterEqual(value, 0.0, (name, row["task_id"], key))

    # -- contract: latency semantics only -----------------------------------
    def test_energy_accounting_does_not_influence_makespan(self):
        invariance = self.evidence["contract"]["energy_invariance"]
        self.assertTrue(invariance["equal"], invariance)
        self.assertEqual(
            invariance["legacy_energy_model_makespan_s"],
            invariance["physical_energy_model_makespan_s"],
        )
        self.assertTrue(self.evidence["contract"]["latency_semantics_only"])
        self.assertTrue(self.evidence["contract"]["energy_excluded_from_parity"])

    # -- canonical row == a direct schedule() call --------------------------
    def test_canonical_matches_direct_schedule_call(self):
        graph, decoder_order, actions, _ = reference_contract()
        direct = schedule(graph, decoder_order, actions, canonical_resources())
        row = self.evidence["schedulers"]["canonical"]
        self.assertAlmostEqual(row["makespan_seconds"], direct.makespan_seconds, places=12)
        self.assertEqual(len(row["tasks"]), len(direct.tasks))
        for task_row in row["tasks"]:
            rec = direct.tasks[int(task_row["task_id"])]
            self.assertEqual(task_row["location"], rec.location.value)
            self.assertAlmostEqual(task_row["start_s"], rec.start, places=12)
            self.assertAlmostEqual(task_row["finish_s"], rec.finish, places=12)
            self.assertAlmostEqual(task_row["return_s"], rec.all_consumers_ready, places=12)
        # the engine's topological order reproduced the handed-in decoder order
        self.assertEqual([int(t) for t in direct.topo_order], [int(t) for t in decoder_order])
        self.assertTrue(self.evidence["checks"]["canonical_matches_direct_schedule"])

    # -- factor decomposition ----------------------------------------------
    def test_all_seven_factors_are_stated(self):
        self.assertEqual(tuple(self.evidence["differences"]), FACTOR_KEYS)
        for key in FACTOR_KEYS:
            factor = self.evidence["differences"][key]
            self.assertIsInstance(factor["applies"], bool, key)
            self.assertTrue(factor["explanation"].strip(), key)
            self.assertIsInstance(factor["numbers"], dict, key)
            self.assertTrue(factor["numbers"], key)
        summary = self.evidence["summary"]
        self.assertEqual(
            sorted(summary["applying_factors"] + summary["non_applying_factors"]),
            sorted(FACTOR_KEYS),
        )

    def test_decoder_order_factor_is_stated_as_non_contributing(self):
        factor = self.evidence["differences"]["decoder_order"]
        self.assertFalse(factor["applies"])
        self.assertTrue(factor["numbers"]["orders_identical"])


class TestUpstreamAvailability(unittest.TestCase):
    """Upstream presence is reported honestly and never fabricated."""

    def test_missing_upstream_raises_loudly(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(UpstreamUnavailableError) as ctx:
                load_upstream_scheduler(Path(tmp) / "does-not-exist")
        message = str(ctx.exception)
        self.assertIn("upstream scheduler not found", message)
        self.assertIn("--upstream-root", message)

    def test_require_upstream_exits_nonzero_and_writes_no_lie(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "evidence.json"
            code = main(
                [
                    "--json",
                    str(out),
                    "--upstream-root",
                    str(Path(tmp) / "missing-upstream"),
                    "--require-upstream",
                    "--quiet",
                ]
            )
            self.assertNotEqual(code, 0)
            self.assertFalse(out.exists())

    def test_unavailable_upstream_is_recorded_not_faked(self):
        with tempfile.TemporaryDirectory() as tmp:
            evidence = build_evidence(Path(tmp) / "missing-upstream")
        self.assertFalse(evidence["upstream_available"])
        self.assertNotIn("upstream_original", evidence["schedulers"])
        self.assertTrue(evidence["upstream_error"])
        self.assertIn("upstream scheduler not found", evidence["upstream_error"])
        self.assertTrue(evidence["checks"]["upstream_status_recorded"])
        # only the upstream-dependent contract check may fail; nothing else
        failing = {name for name, ok in evidence["checks"].items() if not ok}
        self.assertEqual(failing, {"all_three_schedulers_present"}, evidence["checks"])
        self.assertFalse(evidence["all_checks_pass"])
        # no fabricated upstream numbers anywhere
        self.assertIsNone(evidence["summary"]["canonical_vs_upstream_delta_seconds"])
        self.assertIsNone(evidence["summary"]["legacy_control_equals_upstream"])

    def test_unavailable_upstream_exits_nonzero_but_still_writes_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "evidence.json"
            code = main(
                [
                    "--json",
                    str(out),
                    "--upstream-root",
                    str(Path(tmp) / "missing-upstream"),
                    "--quiet",
                ]
            )
            self.assertNotEqual(code, 0)
            self.assertTrue(out.is_file())
            payload = json.loads(out.read_text(encoding="utf-8"))
        self.assertFalse(payload["upstream_available"])
        self.assertTrue(payload["upstream_error"])

    def test_bogus_upstream_root_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            evidence = build_evidence(Path(tmp) / "nope")
        self.assertEqual(evidence["upstream_root"], str(Path(tmp) / "nope"))
        self.assertFalse(evidence["upstream_available"])

    def test_main_writes_evidence_and_exits_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "nested" / "evidence.json"
            code = main(["--json", str(out), "--quiet"])
            self.assertEqual(code, 0)
            self.assertTrue(out.is_file())
            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertTrue(payload["all_checks_pass"])


class TestUpstreamParityWithUpstream(unittest.TestCase):
    """When a checkout exists, upstream's own numbers drive the comparison."""

    @classmethod
    def setUpClass(cls):
        cls.evidence = build_evidence()
        cls.upstream_available = cls.evidence["upstream_available"]

    def setUp(self):
        if not self.upstream_available:
            self.skipTest("upstream metarl-offloading checkout not available")

    def test_upstream_row_uses_upstream_source(self):
        provenance = self.evidence["contract"]["scheduler_provenance"]["upstream_original"]
        self.assertTrue(provenance["available"])
        self.assertIn("get_scheduling_cost_step_by_step", provenance["extracted_symbols"])
        self.assertEqual(len(provenance["file_sha256"]), 64)
        self.assertEqual(len(provenance["scheduler_source_sha256"]), 64)

    def test_upstream_replay_reproduces_upstream_numbers(self):
        row = self.evidence["schedulers"]["upstream_original"]
        self.assertTrue(row["replay_matches_upstream"])
        self.assertLessEqual(row["max_abs_latency_delta_diff_s"], 1e-9)
        self.assertAlmostEqual(
            row["makespan_seconds"], row["upstream_makespan_seconds"], places=12
        )
        self.assertEqual(
            [round(x, 12) for x in row["latency_deltas"]],
            [round(x, 12) for x in row["upstream_latency_deltas"]],
        )
        self.assertAlmostEqual(
            sum(row["latency_deltas"]), row["makespan_seconds"], places=12
        )

    def test_legacy_control_reproduces_upstream_under_the_contract(self):
        """Binary + latency-only: the project's pre-canonical path was upstream."""
        makespans = self.evidence["summary"]["makespan_seconds"]
        self.assertAlmostEqual(
            makespans["legacy_control"], makespans["upstream_original"], places=12
        )
        self.assertTrue(self.evidence["summary"]["legacy_control_equals_upstream"])
        self.assertTrue(self.evidence["checks"]["legacy_control_equals_upstream_latency"])
        upstream = self.evidence["schedulers"]["upstream_original"]
        legacy = self.evidence["schedulers"]["legacy_control"]
        self.assertEqual(len(upstream["latency_deltas"]), len(legacy["latency_deltas"]))
        for a, b in zip(upstream["latency_deltas"], legacy["latency_deltas"]):
            self.assertAlmostEqual(a, b, places=9)

    def test_rate_identity_against_upstream_resources(self):
        rates = self.evidence["contract"]["identical_rates"]
        self.assertTrue(rates["identical"])
        self.assertAlmostEqual(
            rates["upstream_derived_ul_bytes_per_second"],
            rates["mec_uplink_bytes_per_second"],
            places=9,
        )
        self.assertAlmostEqual(
            rates["upstream_derived_dl_bytes_per_second"],
            rates["mec_downlink_bytes_per_second"],
            places=9,
        )


if __name__ == "__main__":
    unittest.main()
