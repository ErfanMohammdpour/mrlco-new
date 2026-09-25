#!/usr/bin/env python3
"""Part D tests: decoder-order / ranking audit (CPU only, no TensorFlow).

The audit is diagnostic: it must never change a production order. These tests
pin the methodology contract the user stated explicitly:

  * five orders are produced for every graph;
  * every order is a valid topological permutation of the task ids;
  * Kendall / Spearman vs the current legacy order are in range;
  * makespans are finite and non-negative;
  * the fixed-plan experiment is TRULY order-only (one frozen per-task action
    mapping for every order, reproduced exactly by an independent re-schedule);
  * the evidence JSON is serializable and `main()` exits non-zero on failure.
"""

from __future__ import annotations

import json
import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# `offloading_env` imports gym at module import time on this CPU host; the audit
# stubs it itself, but keep parity with the other scheduler tests.
for _name in ("gym", "gym.core"):
    if _name not in sys.modules:
        try:
            __import__(_name)
        except Exception:
            sys.modules[_name] = types.ModuleType(_name)
if not hasattr(sys.modules.get("gym.core", types.ModuleType("gym.core")), "Env"):
    sys.modules.setdefault("gym", types.ModuleType("gym"))
    sys.modules.setdefault("gym.core", types.ModuleType("gym.core"))
    sys.modules["gym.core"].Env = type("Env", (), {})

from env.mec_offloaing_envs.scheduler import schedule  # noqa: E402
from env.mec_offloaing_envs.scheduler.calendar import RESOURCE_NAMES  # noqa: E402
from env.mec_offloaing_envs.scheduler.primary_config import (  # noqa: E402
    resolved_primary_scheduler_config,
)
from spec import decoder_order_audit as audit  # noqa: E402

AUDIT_GRAPHS = ("syn_sparse", "syn_medium", "syn_dense", "syn_wide_shallow")


class DecoderOrderAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.resources = resolved_primary_scheduler_config()
        cls.evidence = audit.run(list(AUDIT_GRAPHS))
        cls.rows_by_graph: dict[str, list[dict]] = {}
        for row in cls.evidence["per_graph"]:
            cls.rows_by_graph.setdefault(row["graph"], []).append(row)

    # -- shape -------------------------------------------------------------
    def test_five_orders_produced_per_graph(self) -> None:
        self.assertEqual(set(self.rows_by_graph), set(AUDIT_GRAPHS))
        for graph, rows in self.rows_by_graph.items():
            self.assertEqual(sorted(r["order"] for r in rows), sorted(audit.ORDER_NAMES), graph)
            self.assertEqual(len(rows), len(audit.ORDER_NAMES), graph)

    def test_orders_list_documents_every_order(self) -> None:
        self.assertEqual(
            [entry["name"] for entry in self.evidence["orders"]], list(audit.ORDER_NAMES)
        )
        for entry in self.evidence["orders"]:
            self.assertTrue(entry["formula"])

    def test_required_top_level_keys(self) -> None:
        for key in (
            "schema",
            "orders",
            "graphs",
            "per_graph",
            "aggregate",
            "checks",
            "all_checks_pass",
        ):
            self.assertIn(key, self.evidence)
        self.assertEqual(self.evidence["schema"], "rank_audit_v1")

    def test_graph_set_covers_five_structural_kinds(self) -> None:
        graphs = audit.build_graphs(self.resources)
        for name in (
            "syn_sparse",
            "syn_medium",
            "syn_dense",
            "syn_narrow_deep",
            "syn_wide_shallow",
        ):
            self.assertIn(name, graphs)
        self.assertGreaterEqual(len(graphs), 5)

    # -- topological validity ---------------------------------------------
    def test_every_order_is_a_valid_topological_permutation(self) -> None:
        graphs = audit.build_graphs(self.resources)
        for graph, rows in self.rows_by_graph.items():
            dag = graphs[graph].dag
            expected = sorted(int(t) for t in dag.tasks)
            for row in rows:
                seq = row["order_sequence"]
                self.assertEqual(sorted(seq), expected, (graph, row["order"]))
                self.assertTrue(row["permutation_valid"], (graph, row["order"]))
                self.assertTrue(row["topological_valid"], (graph, row["order"]))
                self.assertTrue(audit.is_topological(dag, seq), (graph, row["order"]))
                ranks = {tid: i for i, tid in enumerate(seq)}
                for edge in dag.edges:
                    self.assertLess(
                        ranks[int(edge.src_task_id)], ranks[int(edge.dst_task_id)]
                    )

    # -- correlations ------------------------------------------------------
    def test_kendall_and_spearman_in_range(self) -> None:
        for row in self.evidence["per_graph"]:
            tau = float(row["kendall_tau"])
            self.assertGreaterEqual(tau, -1.0 - 1e-9, row)
            self.assertLessEqual(tau, 1.0 + 1e-9, row)
            foot = float(row["spearman_distance"])
            self.assertGreaterEqual(foot, 0.0, row)
            self.assertLessEqual(foot, 1.0, row)
            rho = float(row["spearman_rho"])
            self.assertGreaterEqual(rho, -1.0 - 1e-9, row)
            self.assertLessEqual(rho, 1.0 + 1e-9, row)

    def test_current_order_is_its_own_identity(self) -> None:
        for graph, rows in self.rows_by_graph.items():
            current = next(r for r in rows if r["order"] == "legacy_current")
            self.assertAlmostEqual(current["kendall_tau"], 1.0, places=12)
            self.assertAlmostEqual(current["spearman_distance"], 0.0, places=12)

    def test_kendall_and_spearman_helpers_agree_with_definitions(self) -> None:
        a = [0, 1, 2, 3]
        b = [3, 2, 1, 0]
        self.assertAlmostEqual(audit.kendall_tau(a, a), 1.0)
        self.assertAlmostEqual(audit.kendall_tau(a, b), -1.0)
        self.assertAlmostEqual(audit.spearman_rho(a, a), 1.0)
        self.assertAlmostEqual(audit.spearman_rho(a, b), -1.0)
        self.assertAlmostEqual(audit.spearman_footrule_distance(a, a), 0.0)
        self.assertAlmostEqual(audit.spearman_footrule_distance(a, b), 1.0)

    # -- latency -----------------------------------------------------------
    def test_makespans_finite_and_non_negative(self) -> None:
        for row in self.evidence["per_graph"]:
            for key in ("fixed_plan_latency_s", "greedy_latency_s", "static_bound_s"):
                value = row[key]
                self.assertIsNotNone(value, (row["graph"], row["order"], key))
                self.assertGreaterEqual(float(value), 0.0)
            self.assertGreater(float(row["fixed_plan_latency_s"]), 0.0)

    def test_static_bound_is_admissible_lower_bound(self) -> None:
        for row in self.evidence["per_graph"]:
            self.assertLessEqual(
                float(row["static_bound_s"]),
                float(row["fixed_plan_latency_s"]) + 1e-6,
                (row["graph"], row["order"]),
            )
            self.assertTrue(row["static_bound_valid"])
            self.assertGreaterEqual(float(row["static_bound_error_abs_s"]), -1e-9)
            self.assertIsNotNone(row["static_bound_error_rel"])
            self.assertGreaterEqual(float(row["static_bound_error_rel"]), -1e-9)

    # -- the methodology contract -----------------------------------------
    def test_fixed_plan_experiment_is_truly_order_only(self) -> None:
        graphs = audit.build_graphs(self.resources)
        for graph, rows in self.rows_by_graph.items():
            dag = graphs[graph].dag
            reference: dict[str, int] | None = None
            for row in rows:
                mapping = row["fixed_plan_actions_by_task"]
                # (a) the per-task assignment is exactly the frozen mapping
                for tid in dag.tasks:
                    self.assertEqual(
                        int(mapping[str(int(tid))]),
                        audit.fixed_action(int(tid)),
                        (graph, row["order"], tid),
                    )
                # (b) identical mapping for every order -> only order can differ
                if reference is None:
                    reference = mapping
                else:
                    self.assertEqual(mapping, reference, (graph, row["order"]))
                # (c) independent re-schedule reproduces the stored makespan
                order = row["order_sequence"]
                actions = [int(mapping[str(tid)]) for tid in order]
                recomputed = float(schedule(dag, order, actions, self.resources).makespan_seconds)
                self.assertAlmostEqual(
                    recomputed, float(row["fixed_plan_latency_s"]), places=12
                )
                # (d) re-running the same (order, actions) changes nothing
                self.assertLessEqual(float(row["fixed_plan_repeat_delta_s"]), 1e-9)

    def test_queue_utilization_covers_six_calendars_in_range(self) -> None:
        for row in self.evidence["per_graph"]:
            util = row["queue_utilization"]
            for name in RESOURCE_NAMES:
                self.assertIn(name, util)
                self.assertGreaterEqual(float(util[name]), 0.0)
                self.assertLessEqual(float(util[name]), 1.0)
            self.assertAlmostEqual(
                float(util["mean"]),
                sum(float(util[n]) for n in RESOURCE_NAMES) / len(RESOURCE_NAMES),
                places=12,
            )

    def test_orders_are_not_all_identical(self) -> None:
        sequences = {tuple(row["order_sequence"]) for row in self.evidence["per_graph"]}
        self.assertGreaterEqual(len(sequences), 2)

    def test_greedy_assignment_is_a_full_per_task_mapping(self) -> None:
        graphs = audit.build_graphs(self.resources)
        for row in self.evidence["per_graph"]:
            mapping = row["greedy_actions_by_task"]
            self.assertEqual(
                sorted(int(t) for t in mapping),
                sorted(int(t) for t in graphs[row["graph"]].dag.tasks),
            )
            self.assertEqual(sum(row["greedy_action_mix"].values()), len(mapping))

    def test_policy_action_mix_is_stated(self) -> None:
        for row in self.evidence["per_graph"]:
            self.assertIn(row["policy_action_mix"]["status"], ("available", "not available"))

    # -- production rank ---------------------------------------------------
    def test_legacy_order_matches_the_production_rank_values(self) -> None:
        """The production order must be a descending-rank topological order."""
        graphs = audit.build_graphs(self.resources)
        for graph, rows in self.rows_by_graph.items():
            dag = graphs[graph].dag
            current = next(r for r in rows if r["order"] == "legacy_current")
            seq = current["order_sequence"]
            ranks = current["legacy_rank_values"]
            values = [float(ranks[str(tid)]) for tid in seq]
            for i in range(len(values) - 1):
                self.assertGreaterEqual(
                    values[i] + 1e-9, values[i + 1], (graph, i)
                )
            # production rank is strictly greater than every successor's rank
            for edge in dag.edges:
                self.assertGreater(
                    float(ranks[str(int(edge.src_task_id))]),
                    float(ranks[str(int(edge.dst_task_id))]),
                    (graph, edge),
                )

    def test_legacy_order_is_reproducible_through_prioritize_tasks(self) -> None:
        from env.mec_offloaing_envs.offloading_task_graph import OffloadingTaskGraph

        graphs = audit.build_graphs(self.resources)
        for graph in AUDIT_GRAPHS:
            dag = graphs[graph].dag
            view = audit._LegacyGraphView(dag)
            cluster = audit._legacy_cluster(self.resources)
            produced = [int(t) for t in OffloadingTaskGraph.prioritize_tasks(view, cluster)]
            expected = [
                row["order_sequence"]
                for row in self.rows_by_graph[graph]
                if row["order"] == "legacy_current"
            ][0]
            self.assertEqual(produced, expected, graph)

    # -- serialization / entry point --------------------------------------
    def test_json_is_serializable(self) -> None:
        text = json.dumps(self.evidence, allow_nan=False, sort_keys=True)
        self.assertIn("rank_audit_v1", text)
        self.assertTrue(self.evidence["checks"]["json_serializable"])

    def test_all_checks_pass(self) -> None:
        failed = [k for k, v in self.evidence["checks"].items() if not v]
        self.assertEqual(failed, [], failed)
        self.assertTrue(self.evidence["all_checks_pass"])

    def test_main_writes_json_and_exits_zero(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rank_audit_evidence.json"
            code = audit.main(["--json", str(path), "--graphs", "syn_sparse,syn_dense"])
            self.assertEqual(code, 0)
            self.assertTrue(path.is_file())
            payload = json.loads(path.read_text())
            self.assertEqual(payload["schema"], "rank_audit_v1")
            self.assertTrue(payload["all_checks_pass"])
            self.assertEqual(len(payload["per_graph"]), 2 * len(audit.ORDER_NAMES))

    def test_main_exits_non_zero_on_failed_check(self) -> None:
        from unittest import mock

        import tempfile

        real = audit.run(["syn_dense"])
        real["checks"]["deliberately_false"] = False
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.json"
            with mock.patch.object(audit, "run", return_value=real):
                code = audit.main(["--json", str(path), "--graphs", "syn_dense"])
            self.assertEqual(code, 1)
            payload = json.loads(path.read_text())
            self.assertFalse(payload["all_checks_pass"])
            self.assertTrue(path.is_file())


if __name__ == "__main__":
    unittest.main()
