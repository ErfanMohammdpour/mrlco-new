#!/usr/bin/env python3
"""Regression tests for the corrected HEFT-v2 reference baseline.

Everything here reads the FROZEN dataset record-by-record with `json.loads`
(`env/mec_offloaing_envs/data/automotive_mc_v1/graphs.jsonl`).  The frozen M10
certifier is deliberately NOT imported: these tests pin the CORRECTED baseline and
independently reproduce the historical M10 defect through `historical_m10_alias`.

The six sampled graphs are the first record of six distinct frozen cells
(`lines = (0, 1, 6, 7, 8, 11)`): application family `cooperative_perception`, seed
`s0`, spanning whigh/wnominal x rlow/rhigh x dloose/dnominal/dtight.
"""

from __future__ import annotations

import json
import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spec.automotive_training.automotive_dag import (  # noqa: E402
    decoder_order,
    schedule_actions,
    tier_frequencies,
)
from spec.automotive_training.heft_reference_v2 import (  # noqa: E402
    historical_m10_alias,
    heft_reference_v2_plan,
    pure_plan_makespans,
    upward_ranks,
)

GRAPHS_JSONL = (ROOT / "env/mec_offloaing_envs/data/automotive_mc_v1/graphs.jsonl")
SAMPLE_LINES = (0, 1, 6, 7, 8, 11)
TOL = 1e-12


def load_frozen_graphs(lines):
    """Read the frozen JSONL directly; one `json.loads` per line, no M10 import."""
    wanted = set(lines)
    records = {}
    with GRAPHS_JSONL.open("r", encoding="utf-8") as handle:
        for index, raw in enumerate(handle):
            if index in wanted:
                records[index] = json.loads(raw)
            if len(records) == len(wanted):
                break
    missing = wanted - set(records)
    if missing:
        raise AssertionError("frozen graphs missing at lines %s" % sorted(missing))
    return [records[i] for i in lines]


def legacy_buggy_residual_plan(graph):
    """The pre-fix placement loop (unplaced tasks forced to UE), for the counter-example."""
    order = decoder_order(graph)
    rank = upward_ranks(graph)
    priority = sorted(order, key=lambda t: (-rank[t], t))
    chosen = {}
    for k, tid in enumerate(priority):
        best = None
        for action in (0, 1, 2):
            residual = [chosen.get(t, action if t == tid else 0) for t in order]
            metric = schedule_actions(graph, residual).makespan_seconds
            if best is None or metric < best[0] - TOL or (
                    abs(metric - best[0]) <= TOL and action < best[1]):
                best = (metric, action)
        chosen[tid] = int(best[1])
    actions = [chosen[t] for t in order]
    return actions, schedule_actions(graph, actions).makespan_seconds


class TestHeftReferenceV2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.graphs = load_frozen_graphs(SAMPLE_LINES)
        cls.evidence = {}
        for line, graph in zip(SAMPLE_LINES, cls.graphs):
            actions1, makespan1, _res1 = heft_reference_v2_plan(graph)
            actions2, makespan2, _res2 = heft_reference_v2_plan(graph)
            pure = pure_plan_makespans(graph)
            cls.evidence[line] = {
                "line": line,
                "graph_id": graph["graph_id"],
                "cell_id": graph["cell_id"],
                "n_tasks": len(decoder_order(graph)),
                "actions1": actions1,
                "actions2": actions2,
                "makespan1": makespan1,
                "makespan2": makespan2,
                "pure": pure,
                "best_pure": min(pure.values()),
                "counts": {a: actions1.count(a) for a in (0, 1, 2)},
                "alias": historical_m10_alias(graph),
                "ranks": upward_ranks(graph),
                "graph": graph,
            }

    # ---- frozen provenance -------------------------------------------------
    def test_six_real_frozen_graphs(self):
        self.assertEqual(len(self.graphs), 6)
        self.assertEqual(len(SAMPLE_LINES), 6)
        ids = set()
        for graph in self.graphs:
            self.assertEqual(graph["dataset_version"], "MARGO-AUTOMOTIVE-MC-v1")
            self.assertIn("resource", graph)
            self.assertTrue(graph["tasks"])
            self.assertTrue(graph["edges"])
            self.assertEqual(int(graph["task_count"]), len(graph["tasks"]))
            ids.add(graph["graph_id"])
        self.assertEqual(len(ids), 6)
        cells = {g["cell_id"] for g in self.graphs}
        self.assertEqual(len(cells), 6)

    # ---- corrected baseline is not a pure plan -----------------------------
    def test_v2_is_not_a_pure_plan_on_at_least_four_of_six(self):
        non_pure = [
            row for row in self.evidence.values()
            if not (row["counts"][0] == row["n_tasks"]
                    or row["counts"][1] == row["n_tasks"]
                    or row["counts"][2] == row["n_tasks"])
        ]
        self.assertGreaterEqual(
            len(non_pure), 4,
            "expected a genuinely mixed plan on >= 4/6 frozen graphs, got %d"
            % len(non_pure),
        )
        for row in self.evidence.values():
            self.assertFalse(row["counts"][2] == row["n_tasks"],
                             "v2 must not be the degenerate all_HELPER plan")

    # ---- dominance over the pure plans ------------------------------------
    def test_v2_makespan_le_all_helper_on_every_graph(self):
        for row in self.evidence.values():
            self.assertLessEqual(row["makespan1"], row["pure"]["all_HELPER"] + TOL,
                                 row["graph_id"])

    def test_v2_makespan_le_best_pure_on_at_least_four_of_six(self):
        within = [row for row in self.evidence.values()
                  if row["makespan1"] <= row["best_pure"] + TOL]
        self.assertGreaterEqual(len(within), 4,
                                [row["graph_id"] for row in self.evidence.values()])
        strict = [row for row in self.evidence.values()
                  if row["makespan1"] < row["best_pure"] - TOL]
        self.assertTrue(strict, "the corrected baseline must improve on the best pure plan")

    # ---- documented M10 regression ----------------------------------------
    def test_historical_alias_reproduces_degenerate_all_helper(self):
        for row in self.evidence.values():
            alias = row["alias"]
            self.assertEqual(alias["historical_tier_table"],
                             ["f_ue_hz", "f_helper_hz", "f_mec_hz"])
            self.assertEqual(alias["historical_fastest_action"], 2, row["graph_id"])
            self.assertTrue(alias["historical_plan_is_all_helper"], row["graph_id"])
            self.assertAlmostEqual(
                alias["historical_makespan_s"],
                row["pure"]["all_HELPER"],
                delta=1e-12,
                msg=row["graph_id"],
            )

    # ---- determinism -------------------------------------------------------
    def test_two_calls_are_identical(self):
        for row in self.evidence.values():
            self.assertEqual(row["actions1"], row["actions2"], row["graph_id"])
            self.assertAlmostEqual(row["makespan1"], row["makespan2"], delta=TOL,
                                   msg=row["graph_id"])

    # ---- ranks -------------------------------------------------------------
    def test_upward_ranks_finite_and_respect_the_dag(self):
        for row in self.evidence.values():
            graph, ranks = row["graph"], row["ranks"]
            self.assertEqual(set(ranks), set(decoder_order(graph)))
            for value in ranks.values():
                self.assertTrue(math.isfinite(value))
                self.assertGreater(value, 0.0)
            for edge in graph["edges"]:
                src, dst = int(edge["src"]), int(edge["dst"])
                self.assertGreater(ranks[src], ranks[dst],
                                   "%s: rank[%d] must exceed rank[%d]"
                                   % (row["graph_id"], src, dst))

    def test_tier_frequencies_uses_corrected_action_mapping(self):
        for row in self.evidence.values():
            r = row["graph"]["resource"]
            self.assertEqual(
                tier_frequencies(row["graph"]),
                {0: float(r["f_ue_hz"]), 1: float(r["f_mec_hz"]),
                 2: float(r["f_helper_hz"])},
            )
            # why the frozen swapped table degenerates: MEC is the fastest tier, so
            # the historical action 2 (costed with f_mec) always wins.
            self.assertGreater(float(r["f_mec_hz"]), float(r["f_helper_hz"]))
            self.assertGreater(float(r["f_mec_hz"]), float(r["f_ue_hz"]))

    # ---- plan plumbing -----------------------------------------------------
    def test_plan_length_and_makespan_are_consistent(self):
        for row in self.evidence.values():
            self.assertEqual(len(row["actions1"]), row["n_tasks"])
            self.assertTrue(all(a in (0, 1, 2) for a in row["actions1"]))
            recomputed = schedule_actions(row["graph"], row["actions1"]).makespan_seconds
            self.assertAlmostEqual(recomputed, row["makespan1"], delta=1e-12,
                                   msg=row["graph_id"])

    # ---- the fix is load-bearing ------------------------------------------
    def test_buggy_ue_padded_residual_is_worse_than_best_pure(self):
        """Counter-example: the pre-fix residual construction loses to a pure plan."""
        regressions = []
        for row in self.evidence.values():
            _actions, makespan = legacy_buggy_residual_plan(row["graph"])
            if makespan > row["best_pure"] + TOL:
                regressions.append((row["graph_id"], makespan, row["best_pure"]))
        self.assertTrue(
            regressions,
            "expected the UE-padded residual to lose to the best pure plan on >= 1 graph",
        )


if __name__ == "__main__":
    unittest.main()
