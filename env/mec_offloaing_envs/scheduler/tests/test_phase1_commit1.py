#!/usr/bin/env python3
"""Phase 1 Commit-1 tests: calendar, routes, canonical DAG, oracle smoke."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env.mec_offloaing_envs.scheduler import (  # noqa: E402
    CanonicalDAG,
    CanonicalTask,
    ConflictingDuplicateEdgeError,
    Location,
    ResourceCalendar,
    ResourceConfig,
    ROUTE_TABLE,
    route,
    schedule,
)


class TestCalendar(unittest.TestCase):
    def test_gap_reuse(self):
        cal = ResourceCalendar("V2V_CHANNEL")
        # Reserve late interval first
        cal.reserve(1.0, earliest=5.0)  # [5,6)
        # Early request should fit before, not after watermark
        s, e = cal.reserve(1.0, earliest=0.0)
        self.assertEqual((s, e), (0.0, 1.0))

    def test_no_overlap(self):
        cal = ResourceCalendar("MEC_DL")
        cal.reserve(1.0, earliest=0.0)
        s, e = cal.reserve(1.0, earliest=0.0)
        self.assertEqual((s, e), (1.0, 2.0))


class TestRoutes(unittest.TestCase):
    def test_all_nine(self):
        self.assertEqual(len(ROUTE_TABLE), 9)
        self.assertEqual(route(Location.UE, Location.MEC), ["MEC_UL"])
        self.assertEqual(route(Location.MEC, Location.HELPER), ["MEC_DL", "V2V"])
        self.assertEqual(route(Location.HELPER, Location.MEC), ["V2V", "MEC_UL"])
        self.assertEqual(route(Location.UE, Location.UE), [])


class TestCanonical(unittest.TestCase):
    def test_duplicate_collapse(self):
        tasks = [
            CanonicalTask(1, 10, 5, 10),
            CanonicalTask(2, 10, 5, 0),
        ]
        dag = CanonicalDAG.from_records(tasks, [(1, 2, 5), (1, 2, 5), (1, 2, 5)])
        self.assertEqual(dag.edge_record_count, 3)
        self.assertEqual(dag.unique_edge_count, 1)

    def test_conflict_rejected(self):
        tasks = [
            CanonicalTask(1, 10, 5, 10),
            CanonicalTask(2, 10, 5, 0),
        ]
        with self.assertRaises(ConflictingDuplicateEdgeError):
            CanonicalDAG.from_records(tasks, [(1, 2, 5), (1, 2, 7)])


class TestEngineOracleSmoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = ResourceConfig.from_frozen_yaml()

    def _schedule_oracle(self, name: str):
        import yaml

        path = ROOT / "spec" / "toy_oracles" / name
        doc = yaml.safe_load(path.read_text())
        tasks = [
            CanonicalTask(
                task_id=int(n["task_id"]),
                compute_workload_bytes=int(n["compute_workload_bytes"]),
                task_output_bytes=int(n["task_output_bytes"]),
                external_input_bytes=int(n.get("external_input_bytes", 0)),
            )
            for n in doc["nodes"]
        ]
        raw_edges = [
            (int(e["src_task_id"]), int(e["dst_task_id"]), int(e["edge_output_bytes"]))
            for e in doc.get("edges", [])
        ]
        dag = CanonicalDAG.from_records(tasks, raw_edges)
        order = sorted(dag.tasks)
        actions = doc["actions"]
        return schedule(dag, order, actions, self.cfg), doc["expected"]

    def test_01_all_local(self):
        got, exp = self._schedule_oracle("01_all-local-chain.yaml")
        self.assertAlmostEqual(got.makespan_seconds, exp["makespan_seconds"])
        self.assertAlmostEqual(got.total_mobile_joules, exp["total_mobile_joules"])

    def test_04_ue_to_mec(self):
        got, exp = self._schedule_oracle("04_ue-to-mec.yaml")
        self.assertAlmostEqual(got.makespan_seconds, exp["makespan_seconds"])
        self.assertAlmostEqual(got.total_mobile_joules, exp["total_mobile_joules"])
        # residency: task2 stays at MEC until sink return
        self.assertEqual(got.tasks[2].output_location, Location.MEC)

    def test_09_helper_to_mec(self):
        got, exp = self._schedule_oracle("09_helper-to-mec.yaml")
        self.assertAlmostEqual(got.makespan_seconds, exp["makespan_seconds"])
        self.assertAlmostEqual(got.total_mobile_joules, exp["total_mobile_joules"])

    def test_14_v2v_half_duplex(self):
        got, exp = self._schedule_oracle("14_v2v-half-duplex-contention.yaml")
        self.assertAlmostEqual(got.makespan_seconds, exp["makespan_seconds"])
        self.assertAlmostEqual(got.total_mobile_joules, exp["total_mobile_joules"])


if __name__ == "__main__":
    unittest.main()
