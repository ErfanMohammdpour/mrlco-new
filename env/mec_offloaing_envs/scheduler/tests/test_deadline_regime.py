#!/usr/bin/env python3
"""deadline_regime_v1: schema, generator (EFT/LFT) and stamper.

Hand-computed DAG tests first (chain / fork / join / multi-sink), then the
validation, idempotency and byte-exactness contracts. No TensorFlow needed.
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

from env.mec_offloaing_envs.scheduler.adapter import to_canonical_dag  # noqa: E402
from env.mec_offloaing_envs.scheduler.deadline_regime import (  # noqa: E402
    SCHEMA_VERSION,
    DeadlineRegime,
    DeadlineRegimeError,
    GraphDeadlines,
    TaskDeadline,
    build_regime,
    file_sha256,
    generate_graph_deadlines,
    graph_key,
    require_trainable,
    resources_sha256,
    stamp_task_graph,
    topological_order,
)
from env.mec_offloaing_envs.scheduler.model import CanonicalDAG, CanonicalTask  # noqa: E402
from env.mec_offloaing_envs.scheduler.resources import ResourceConfig  # noqa: E402

# 1 byte/s on every tier makes every compute time equal to the workload in bytes
UNIT = ResourceConfig(
    ue_cpu_bytes_per_second=1.0,
    mec_cpu_bytes_per_second=1.0,
    helper_cpu_bytes_per_second=1.0,
    mec_uplink_bytes_per_second=1.0,
    mec_downlink_bytes_per_second=1.0,
    v2v_bytes_per_second=1.0,
    rho_ue=1.0,
    f_l=1.0,
    zeta=2.0,
    ptx_mec_w=0.1,
    prx_mec_w=0.05,
    ptx_v2v_w=0.06,
    prx_v2v_w=0.03,
    rho_helper=0.7,
    f_v2v=1.0,
)


def dag_from(workloads, edges, out_bytes=None, external=None):
    """CanonicalDAG from {task_id: compute_bytes} and [(src, dst, bytes), ...]."""
    out_bytes = out_bytes or {}
    external = external or {}
    tasks = [
        CanonicalTask(
            task_id=tid,
            compute_workload_bytes=int(workloads[tid]),
            task_output_bytes=int(out_bytes.get(tid, 0)),
            external_input_bytes=int(external.get(tid, 0)),
        )
        for tid in sorted(workloads)
    ]
    return CanonicalDAG.from_records(tasks, list(edges))


def gen(dag, kappa=1.0, alpha=1.0, dtype="hard", order=None):
    return generate_graph_deadlines(
        dag,
        kappa=kappa,
        alpha=alpha,
        deadline_type=dtype,
        resources=UNIT,
        cycles_per_bit=1.0,
        order=order,
        content_sha256="0" * 64,
        graph_key_name="k",
    )


class TestChain(unittest.TestCase):
    """0 -> 1 -> 2, each task 10 bytes of compute: EFT = 10, 20, 30."""

    def setUp(self):
        self.dag = dag_from({0: 10, 1: 10, 2: 10}, [(0, 1, 0), (1, 2, 0)])

    def test_alpha_one_is_the_lft(self):
        # alpha=1 -> LFT: each task may finish as late as its successors allow,
        # only the sink reaches the graph budget
        entry = gen(self.dag, kappa=1.4, alpha=1.0)
        self.assertAlmostEqual(entry.bounds["graph_lb"], 30.0)
        self.assertAlmostEqual(entry.bounds["d_graph"], 42.0)
        self.assertAlmostEqual(entry.tasks[0].deadline_s, 22.0)
        self.assertAlmostEqual(entry.tasks[1].deadline_s, 32.0)
        self.assertAlmostEqual(entry.tasks[2].deadline_s, 42.0)

    def test_alpha_zero_is_the_tightest(self):
        entry = gen(self.dag, kappa=1.4, alpha=0.0)
        self.assertAlmostEqual(entry.tasks[0].deadline_s, 10.0)
        self.assertAlmostEqual(entry.tasks[1].deadline_s, 20.0)
        self.assertAlmostEqual(entry.tasks[2].deadline_s, 30.0)

    def test_intermediate_alpha_interpolates(self):
        entry = gen(self.dag, kappa=1.4, alpha=0.5)
        # d_i = EFT_i + 0.5 * (D_G - EFT_i); D_G - EFT_i = 12 for every task here
        self.assertAlmostEqual(entry.tasks[0].deadline_s, 16.0)
        self.assertAlmostEqual(entry.tasks[1].deadline_s, 26.0)
        self.assertAlmostEqual(entry.tasks[2].deadline_s, 36.0)

    def test_lft_not_below_eft_when_kappa_at_least_one(self):
        entry = gen(self.dag, kappa=1.0, alpha=1.0)
        self.assertEqual(entry.bounds["relaxation_infeasible_tasks"], 0.0)

    def test_infeasible_budget_is_counted_and_clamped(self):
        # kappa < 1 pushes LFT below EFT and the last deadlines below zero; they
        # are clamped so the schema stays valid, and the count is reported for the
        # witness step to reject the graph
        entry = gen(self.dag, kappa=0.5, alpha=1.0)
        self.assertGreater(entry.bounds["relaxation_infeasible_tasks"], 0.0)
        self.assertGreater(entry.bounds["clamped_tasks"], 0.0)
        for tid in (0, 1, 2):
            self.assertGreater(entry.tasks[tid].deadline_s, 0.0)


class TestForkJoin(unittest.TestCase):
    """0 -> {1,2} -> 3: parallel branch determines the lower bound."""

    def setUp(self):
        self.dag = dag_from(
            {0: 10, 1: 30, 2: 5, 3: 10},
            [(0, 1, 0), (0, 2, 0), (1, 3, 0), (2, 3, 0)],
        )

    def test_branch_bounds(self):
        entry = gen(self.dag, kappa=1.0, alpha=0.0)
        # EFT: t0=10, t1=40, t2=15, t3=max(40,15)+10=50
        self.assertAlmostEqual(entry.tasks[0].deadline_s, 10.0)
        self.assertAlmostEqual(entry.tasks[1].deadline_s, 40.0)
        self.assertAlmostEqual(entry.tasks[2].deadline_s, 15.0)
        self.assertAlmostEqual(entry.tasks[3].deadline_s, 50.0)
        self.assertAlmostEqual(entry.bounds["graph_lb"], 50.0)

    def test_join_lft_respects_the_longer_successor(self):
        entry = gen(self.dag, kappa=1.5, alpha=1.0)
        self.assertAlmostEqual(entry.bounds["d_graph"], 75.0)
        # root LFT is set by the SLOW branch (30 bytes) -> 75 - 10 - 30 = 35
        self.assertAlmostEqual(entry.tasks[0].deadline_s, 35.0)
        self.assertAlmostEqual(entry.tasks[1].deadline_s, 65.0)
        self.assertAlmostEqual(entry.tasks[2].deadline_s, 65.0)
        self.assertAlmostEqual(entry.tasks[3].deadline_s, 75.0)

    def test_deadlines_are_monotone_along_the_critical_path(self):
        entry = gen(self.dag, kappa=1.2, alpha=0.5)
        self.assertLessEqual(entry.tasks[0].deadline_s, entry.tasks[1].deadline_s)
        self.assertLessEqual(entry.tasks[1].deadline_s, entry.tasks[3].deadline_s)


class TestMultiSinkReturnHop(unittest.TestCase):
    """Two sinks; the sink deadline basis includes the UE return hop."""

    def setUp(self):
        self.dag = dag_from(
            {0: 10, 1: 10, 2: 10},
            [(0, 1, 0), (0, 2, 0)],
            out_bytes={1: 5000, 2: 100},
        )

    def test_sink_ready_bound_includes_return_hop_per_action(self):
        from env.mec_offloaing_envs.scheduler.static_bounds import static_action_bounds

        entry = gen(self.dag, kappa=1.0, alpha=0.0)
        bounds = static_action_bounds(self.dag, [0, 1, 2], UNIT, cycles_per_bit=1.0)
        # the MIN over actions is a co-located/UE option with no hop, which is why
        # it is the admissible deadline basis; the hop shows up per action
        for i, tid in enumerate((0, 1, 2)):
            self.assertAlmostEqual(entry.tasks[tid].deadline_s, min(bounds.ready_lb[i]))
        for tid in (1, 2):
            i = tid
            self.assertGreater(bounds.ready_lb[i][1], bounds.finish_lb[i][1])

    def test_bigger_payload_gets_the_larger_hop_bound(self):
        from env.mec_offloaing_envs.scheduler.static_bounds import static_action_bounds

        bounds = static_action_bounds(self.dag, [0, 1, 2], UNIT, cycles_per_bit=1.0)
        hop_big = bounds.ready_lb[1][1] - bounds.finish_lb[1][1]
        hop_small = bounds.ready_lb[2][1] - bounds.finish_lb[2][1]
        self.assertGreater(hop_big, hop_small)


class TestCriticalityAndWeights(unittest.TestCase):
    def test_class_and_weight_are_independent_fields(self):
        dag = dag_from({0: 10, 1: 10, 2: 10}, [(0, 1, 0), (1, 2, 0)])
        entry = gen(dag, kappa=1.2, alpha=0.5)
        tid = 2  # the sink
        self.assertEqual(entry.tasks[tid].criticality_class, "high")
        self.assertEqual(entry.tasks[tid].tardiness_weight, 2.0)
        # a custom weighting must not change the class
        custom = generate_graph_deadlines(
            dag,
            kappa=1.2,
            alpha=0.5,
            deadline_type="hard",
            resources=UNIT,
            cycles_per_bit=1.0,
            class_weights={"high": 7.0, "medium": 1.0, "low": 0.5},
        )
        self.assertEqual(custom.tasks[tid].criticality_class, "high")
        self.assertEqual(custom.tasks[tid].tardiness_weight, 7.0)

    def test_soft_and_firm_types_are_allowed(self):
        dag = dag_from({0: 10}, [])
        for dtype in ("soft", "firm", "hard"):
            self.assertEqual(gen(dag, dtype=dtype).tasks[0].deadline_type, dtype)


class TestSchemaValidation(unittest.TestCase):
    def _regime(self):
        dag = dag_from({0: 10, 1: 10}, [(0, 1, 0)])
        entry = gen(dag, kappa=1.0, alpha=1.0)
        entry = GraphDeadlines("k", entry.content_sha256, entry.tasks, entry.bounds)
        return DeadlineRegime(
            regime="t", kappa=1.0, alpha=1.0, deadline_type="hard",
            cycles_per_bit=1.0, graphs={"k": entry},
        )

    def test_round_trip(self):
        regime = self._regime()
        again = DeadlineRegime.from_json(regime.to_json())
        self.assertEqual(again.as_dict(), regime.as_dict())
        self.assertEqual(again.schema_version, SCHEMA_VERSION)

    def test_wrong_schema_version_is_rejected(self):
        blob = json.loads(self._regime().to_json())
        blob["schema_version"] = "deadline_regime_v0"
        with self.assertRaises(DeadlineRegimeError):
            DeadlineRegime.from_json(blob)

    def test_invalid_scalars_are_rejected(self):
        for kwargs in (
            {"kappa": 0.0}, {"kappa": -1.0}, {"kappa": float("nan")},
            {"alpha": -0.1}, {"alpha": 1.5}, {"alpha": float("inf")},
        ):
            base = dict(regime="t", kappa=1.0, alpha=1.0, deadline_type="hard",
                        cycles_per_bit=1.0, graphs=self._regime().graphs)
            base.update(kwargs)
            with self.assertRaises(DeadlineRegimeError, msg=str(kwargs)):
                DeadlineRegime(**base)

    def test_bad_deadline_type_is_rejected(self):
        with self.assertRaises(DeadlineRegimeError):
            TaskDeadline(1.0, "whenever", "high", 1.0)
        with self.assertRaises(DeadlineRegimeError):
            TaskDeadline(1.0, "hard", "urgent", 1.0)

    def test_bad_deadline_values_are_rejected(self):
        for value in (0.0, -1.0, float("nan"), float("inf")):
            with self.assertRaises(DeadlineRegimeError, msg=repr(value)):
                TaskDeadline(value, "hard", "high", 1.0)

    def test_bad_weight_is_rejected(self):
        for weight in (-0.5, float("nan"), float("inf")):
            with self.assertRaises(DeadlineRegimeError, msg=repr(weight)):
                TaskDeadline(1.0, "hard", "high", weight)

    def test_duplicate_task_id_in_json_is_rejected(self):
        blob = json.loads(self._regime().to_json())
        graph = blob["graphs"]["k"]
        graph["tasks"] = {"0": graph["tasks"]["0"], "1": graph["tasks"]["1"]}
        raw = json.dumps(blob)
        assert '"0"' in raw
        # duplicate keys are impossible in a JSON object, so simulate a bad
        # mapping instead: a task id that is not an integer
        graph["tasks"] = {"x": graph["tasks"]["0"]}
        with self.assertRaises(DeadlineRegimeError):
            DeadlineRegime.from_json(blob)

    def test_missing_tasks_are_rejected(self):
        blob = json.loads(self._regime().to_json())
        blob["graphs"]["k"]["tasks"] = {}
        with self.assertRaises(DeadlineRegimeError):
            DeadlineRegime.from_json(blob)

    def test_missing_content_hash_is_rejected(self):
        blob = json.loads(self._regime().to_json())
        del blob["graphs"]["k"]["content_sha256"]
        with self.assertRaises(DeadlineRegimeError):
            DeadlineRegime.from_json(blob)


class _FakeTask:
    def __init__(self, workload, out_bytes=0):
        self.processing_data_size = workload
        self.transmission_data_size = out_bytes


class _FakeGraph:
    """Minimal stand-in for OffloadingTaskGraph (what the stamper touches)."""

    def __init__(self, n, edges=()):
        self.task_number = n
        self.task_list = [_FakeTask(10) for _ in range(n)]
        # adapter reads edge[0]=src, edge[3]=bytes, edge[4]=dst
        self.edge_set = [
            [int(src), 0, 0, int(nbytes), int(dst), 0, 0] for src, dst, nbytes in edges
        ]
        self.prioritize_sequence = list(range(n))
        # to_canonical_dag reads pre_task_sets to decide which tasks are roots
        parents = {i: [] for i in range(n)}
        for src, dst, _nbytes in edges:
            parents[int(dst)].append(int(src))
        self.pre_task_sets = [set(parents[i]) for i in range(n)]
        self.succ_task_sets = [
            {int(dst) for src, dst, _b in edges if int(src) == i} for i in range(n)
        ]


class TestStamper(unittest.TestCase):
    def _entry(self, n=3):
        dag = dag_from({i: 10 for i in range(n)}, [(i, i + 1, 0) for i in range(n - 1)])
        return gen(dag, kappa=1.3, alpha=0.5)

    def test_stamp_sets_every_field(self):
        graph = _FakeGraph(3)
        entry = self._entry(3)
        written = stamp_task_graph(graph, entry, regime_name="r")
        self.assertEqual(written, 3)
        for i, task in enumerate(graph.task_list):
            self.assertAlmostEqual(task.deadline_s, entry.tasks[i].deadline_s)
            self.assertEqual(task.deadline_type, "hard")
            self.assertIn(task.criticality_class, ("low", "medium", "high"))
            self.assertGreater(task.tardiness_weight, 0.0)

    def test_task_number_mismatch_is_rejected(self):
        graph = _FakeGraph(2)
        with self.assertRaises(DeadlineRegimeError):
            stamp_task_graph(graph, self._entry(3), regime_name="r")

    def test_content_hash_mismatch_is_rejected(self):
        graph = _FakeGraph(3)
        with self.assertRaises(DeadlineRegimeError):
            stamp_task_graph(
                graph, self._entry(3), regime_name="r", content_sha256="f" * 64
            )

    def test_idempotent_and_overwrite_protected(self):
        graph = _FakeGraph(3)
        entry = self._entry(3)
        stamp_task_graph(graph, entry, regime_name="r")
        self.assertEqual(stamp_task_graph(graph, entry, regime_name="r"), 0)
        other = gen(
            dag_from({i: 10 for i in range(3)}, [(0, 1, 0), (1, 2, 0)]),
            kappa=2.0, alpha=0.5,
        )
        with self.assertRaises(DeadlineRegimeError):
            stamp_task_graph(graph, other, regime_name="other")
        # explicit override is allowed and reported
        self.assertEqual(
            stamp_task_graph(graph, other, regime_name="other", allow_overwrite=True), 3
        )

    def test_none_regime_writes_nothing(self):
        graph = _FakeGraph(3)
        empty = GraphDeadlines("k", "0" * 64, {}, {})
        self.assertEqual(stamp_task_graph(graph, empty, regime_name="none"), 0)
        for task in graph.task_list:
            self.assertFalse(hasattr(task, "deadline_s"))

    def test_none_regime_refuses_a_stamped_graph(self):
        graph = _FakeGraph(3)
        stamp_task_graph(graph, self._entry(3), regime_name="tight_hard")
        empty = GraphDeadlines("k", "0" * 64, {}, {})
        with self.assertRaises(DeadlineRegimeError):
            stamp_task_graph(graph, empty, regime_name="none")


class TestNoneRegimeIsByteExact(unittest.TestCase):
    def test_to_canonical_dag_is_unchanged(self):
        graph = _FakeGraph(3, edges=[(0, 1, 5), (1, 2, 7)])
        before = to_canonical_dag(graph)
        empty = GraphDeadlines("k", "0" * 64, {}, {})
        stamp_task_graph(graph, empty, regime_name="none")
        after = to_canonical_dag(graph)
        self.assertEqual(before.edges, after.edges)
        for tid, task in before.tasks.items():
            other = after.tasks[tid]
            self.assertIsNone(other.deadline_s)
            self.assertEqual(other.deadline_type, "none")
            self.assertEqual(task.compute_workload_bytes, other.compute_workload_bytes)
            self.assertEqual(task.deadline_s, other.deadline_s)


class TestDeterminismAndHashing(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.paths = []
        for name, workloads in (("a.gv", {0: 10, 1: 10}), ("b.gv", {0: 20, 1: 5})):
            path = Path(self.tmp.name) / name
            path.write_text("digraph { }" + chr(10))
            self.paths.append(path)
        self.graphs = [
            (1, self.paths[0], dag_from({0: 10, 1: 10}, [(0, 1, 0)])),
            (1, self.paths[1], dag_from({0: 20, 1: 5}, [(0, 1, 0)])),
        ]

    def _build(self, **overrides):
        kwargs = dict(regime="r", kappa=1.1, alpha=0.5, deadline_type="hard",
                      resources=UNIT, cycles_per_bit=1.0, seed=0)
        kwargs.update(overrides)
        return build_regime(self.graphs, **kwargs).to_json()

    def test_same_inputs_same_hash(self):
        self.assertEqual(self._build(), self._build())

    def test_different_kappa_changes_the_payload(self):
        self.assertNotEqual(self._build(kappa=1.1), self._build(kappa=1.2))

    def test_sidecar_records_the_content_hash_and_provenance(self):
        blob = json.loads(self._build())
        entry = blob["graphs"]["1/a.gv"]
        self.assertEqual(entry["content_sha256"], file_sha256(self.paths[0]))
        self.assertEqual(blob["resources_sha256"], resources_sha256(UNIT))
        self.assertIn("provenance", blob)

    def test_different_resources_change_the_hash(self):
        other = ResourceConfig(**{**UNIT.__dict__, "mec_cpu_bytes_per_second": 2.0})
        self.assertNotEqual(resources_sha256(UNIT), resources_sha256(other))

    def test_graph_key_and_file_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "random.20.7.gv"
            path.write_text("digraph {}\n")
            self.assertEqual(graph_key(3, path), "3/random.20.7.gv")
            self.assertEqual(file_sha256(path), file_sha256(path))
            path.write_text("digraph { a }\n")
            self.assertNotEqual(file_sha256(path), "0" * 64)

    def test_topological_order_is_deterministic(self):
        dag = dag_from({0: 1, 1: 1, 2: 1, 3: 1}, [(0, 2, 0), (1, 2, 0), (2, 3, 0)])
        self.assertEqual(topological_order(dag), [0, 1, 2, 3])
        self.assertEqual(topological_order(dag), topological_order(dag))


class TestTrainableGuard(unittest.TestCase):
    def test_infeasible_labelled_cannot_be_trained(self):
        entry = GraphDeadlines("k", "0" * 64, {0: TaskDeadline(1.0, "hard", "high", 1.0)}, {})
        labelled = DeadlineRegime(
            regime="infeasible_labelled", kappa=0.8, alpha=0.25, deadline_type="hard",
            cycles_per_bit=1.0, graphs={"k": entry},
        )
        self.assertFalse(labelled.is_trainable)
        with self.assertRaises(DeadlineRegimeError):
            require_trainable(labelled)

    def test_normal_regimes_are_trainable(self):
        entry = GraphDeadlines("k", "0" * 64, {0: TaskDeadline(1.0, "hard", "high", 1.0)}, {})
        regime = DeadlineRegime(
            regime="medium_hard", kappa=1.1, alpha=0.5, deadline_type="hard",
            cycles_per_bit=1.0, graphs={"k": entry},
        )
        self.assertTrue(regime.is_trainable)
        require_trainable(regime)


if __name__ == "__main__":
    unittest.main(verbosity=2)
