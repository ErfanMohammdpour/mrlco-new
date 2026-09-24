#!/usr/bin/env python3
"""Witness plans: hand-built DAGs, real scheduler replay, hard-deadline verdicts.

These tests use the ENGINE (through `schedule_via_adapter`), not the relaxation, so
they pin the semantics that matter: `all_consumers_ready` is the deadline basis,
the sink return hop is included, and an infeasible instance is rejected instead of
being certified by a bound.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env.mec_offloaing_envs.scheduler.deadline_regime import (  # noqa: E402
    generate_graph_deadlines,
    stamp_task_graph,
)
from env.mec_offloaing_envs.scheduler.model import CanonicalDAG, CanonicalTask  # noqa: E402
from env.mec_offloaing_envs.scheduler.resources import ResourceConfig  # noqa: E402
from env.mec_offloaing_envs.scheduler.witness import (  # noqa: E402
    find_fastest_plan,
    find_witness,
    mixed_action_witness,
    plan_for,
    replay_plan,
)

UNIT = ResourceConfig(
    ue_cpu_bytes_per_second=1.0,
    mec_cpu_bytes_per_second=1.0,
    helper_cpu_bytes_per_second=1.0,
    mec_uplink_bytes_per_second=1.0,
    mec_downlink_bytes_per_second=1.0,
    v2v_bytes_per_second=1.0,
    rho_ue=1.0, f_l=1.0, zeta=2.0,
    ptx_mec_w=0.1, prx_mec_w=0.05, ptx_v2v_w=0.06, prx_v2v_w=0.03,
    rho_helper=0.7, f_v2v=1.0,
)

# a SLOWER UE and HELPER than MEC, like the real resource profiles, so the agent
# has a reason to prefer MEC
TIERS = ResourceConfig(
    ue_cpu_bytes_per_second=1.0,
    mec_cpu_bytes_per_second=10.0,
    helper_cpu_bytes_per_second=5.0,
    mec_uplink_bytes_per_second=1.0,
    mec_downlink_bytes_per_second=1.0,
    v2v_bytes_per_second=1.0,
    rho_ue=1.0, f_l=1.0, zeta=2.0,
    ptx_mec_w=0.1, prx_mec_w=0.05, ptx_v2v_w=0.06, prx_v2v_w=0.03,
    rho_helper=0.7, f_v2v=1.0,
)


class FakeTask:
    def __init__(self, workload, out_bytes=0):
        self.processing_data_size = workload
        self.transmission_data_size = out_bytes


class FakeGraph:
    """The minimum `OffloadingTaskGraph` surface the adapter + stamper read."""

    def __init__(self, workloads, edges, out_bytes=None, order=None):
        out_bytes = out_bytes or {}
        n = len(workloads)
        self.task_number = n
        self.task_list = [FakeTask(workloads[i], out_bytes.get(i, 0)) for i in range(n)]
        self.edge_set = [
            [int(s), 0, 0, int(b), int(d), 0, 0] for s, d, b in edges
        ]
        parents = {i: set() for i in range(n)}
        for s, d, _b in edges:
            parents[int(d)].add(int(s))
        self.pre_task_sets = [parents[i] for i in range(n)]
        self.succ_task_sets = [
            {int(d) for s, d, _b in edges if int(s) == i} for i in range(n)
        ]
        self.prioritize_sequence = order or list(range(n))

    def canonical(self):
        tasks = [
            CanonicalTask(
                task_id=i,
                compute_workload_bytes=int(self.task_list[i].processing_data_size),
                task_output_bytes=int(self.task_list[i].transmission_data_size),
                external_input_bytes=0,
            )
            for i in range(self.task_number)
        ]
        return CanonicalDAG.from_records(tasks, [(s, d, b) for s, d, b in self.edge_set and []] or [])


def stamp(graph, *, kappa, alpha, dtype="hard", resources=UNIT, cycles_per_bit=1.0):
    """Generate deadlines from the graph's own records and stamp them on it."""
    tasks = [
        CanonicalTask(
            task_id=i,
            compute_workload_bytes=int(graph.task_list[i].processing_data_size),
            task_output_bytes=int(graph.task_list[i].transmission_data_size),
            external_input_bytes=0,
        )
        for i in range(graph.task_number)
    ]
    edges = [(int(e[0]), int(e[4]), int(e[3])) for e in graph.edge_set]
    dag = CanonicalDAG.from_records(tasks, edges)
    entry = generate_graph_deadlines(
        dag,
        kappa=kappa,
        alpha=alpha,
        deadline_type=dtype,
        resources=resources,
        cycles_per_bit=cycles_per_bit,
        content_sha256="0" * 64,
        graph_key_name="test",
    )
    stamp_task_graph(graph, entry, regime_name="test")
    return entry


def stamp_anchored(graph, *, kappa, alpha, dtype="hard", resources=UNIT, cycles_per_bit=1.0):
    """The production path: deadlines anchored to a REAL fastest plan."""
    tasks = [
        CanonicalTask(
            task_id=i,
            compute_workload_bytes=int(graph.task_list[i].processing_data_size),
            task_output_bytes=int(graph.task_list[i].transmission_data_size),
            external_input_bytes=0,
        )
        for i in range(graph.task_number)
    ]
    edges = [(int(e[0]), int(e[4]), int(e[3])) for e in graph.edge_set]
    dag = CanonicalDAG.from_records(tasks, edges)
    fastest = find_fastest_plan(graph, resources)
    entry = generate_graph_deadlines(
        dag,
        kappa=kappa,
        alpha=alpha,
        deadline_type=dtype,
        resources=resources,
        cycles_per_bit=cycles_per_bit,
        order=[int(t) for t in graph.prioritize_sequence],
        content_sha256="0" * 64,
        graph_key_name="test",
        anchor_ready_s=list(fastest.ready_s),
    )
    stamp_task_graph(graph, entry, regime_name="test")
    return entry, fastest


class TestChainWitness(unittest.TestCase):
    def setUp(self):
        self.graph = FakeGraph([10, 10, 10], [(0, 1, 0), (1, 2, 0)])

    def test_loose_deadlines_are_satisfiable(self):
        _entry, fastest = stamp_anchored(self.graph, kappa=1.0, alpha=1.0)
        witness = find_witness(self.graph, UNIT)
        # kappa = alpha = 1 puts every deadline exactly on a real plan's ready
        # times, so that plan IS the witness
        self.assertAlmostEqual(witness.makespan_s, fastest.makespan_s, places=6)
        self.assertEqual(witness.actions, fastest.actions)
        self.assertTrue(witness.found, witness.reason)
        self.assertEqual(witness.hard_miss_count, 0)
        # every task met its deadline in the real replay
        for row in (witness.per_task or {}).values():
            self.assertGreaterEqual(row["slack_s"], -1e-9)

    def test_relaxation_deadlines_can_be_exactly_achievable(self):
        # alpha=0 sets every deadline to its own lower bound; on a chain with no
        # transfer or contention the lower bound IS achievable, so this is a
        # legitimate witness -- the relaxation is not automatically infeasible
        stamp(self.graph, kappa=1.0, alpha=0.0)
        witness = find_witness(self.graph, UNIT)
        self.assertTrue(witness.found, witness.reason)
        for row in (witness.per_task or {}).values():
            self.assertAlmostEqual(row["slack_s"], 0.0, places=9)

    def test_impossible_deadlines_are_rejected_not_certified(self):
        # kappa < 1 with alpha=1 drives the LFT below the lower bound, so the
        # deadlines are clamped to the floor and no plan can meet them
        stamp(self.graph, kappa=0.5, alpha=1.0)
        witness = find_witness(self.graph, UNIT)
        self.assertFalse(witness.found)
        self.assertIn("no plan met every hard deadline", witness.reason)
        self.assertGreater(witness.hard_miss_count, 0)

    def test_deterministic(self):
        stamp_anchored(self.graph, kappa=1.5, alpha=0.5)
        first = find_witness(self.graph, UNIT)
        second = find_witness(self.graph, UNIT)
        self.assertEqual(first.actions, second.actions)
        self.assertAlmostEqual(first.makespan_s, second.makespan_s)
        self.assertEqual(first.method, second.method)

    def test_refuses_a_graph_without_hard_deadlines(self):
        with self.assertRaises(ValueError):
            find_witness(self.graph, UNIT)


class TestDeadlineBasisSemantics(unittest.TestCase):
    """A sink is judged on the return hop too, never on compute finish alone."""

    def setUp(self):
        # one task, one sink, 100 bytes of output
        self.graph = FakeGraph([10], [], out_bytes={0: 100})

    def test_replay_reports_the_hop(self):
        stamp(self.graph, kappa=100.0, alpha=1.0)
        result, table = replay_plan(self.graph, [1], UNIT)   # run on MEC
        row = table[0]
        self.assertGreater(row["all_consumers_ready_s"], row["finish_s"])
        self.assertAlmostEqual(
            row["all_consumers_ready_s"] - row["finish_s"], 100.0, places=6
        )
        self.assertAlmostEqual(row["slack_s"], row["deadline_s"] - row["all_consumers_ready_s"])

    def test_deadline_between_finish_and_ready_is_a_miss(self):
        # finish = 10, ready = 110 on MEC; set the deadline at 50
        self.graph.task_list[0].deadline_s = 50.0
        self.graph.task_list[0].deadline_type = "hard"
        result, table = replay_plan(self.graph, [1], UNIT)
        self.assertEqual(table[0]["missed"], 1.0)
        self.assertGreater(result.hard_miss_count, 0)
        # the same task on the UE meets it: no hop, ready == finish == 10
        result_ue, table_ue = replay_plan(self.graph, [0], UNIT)
        self.assertEqual(table_ue[0]["missed"], 0.0)
        self.assertEqual(result_ue.hard_miss_count, 0)

    def test_local_action_has_no_hop(self):
        stamp(self.graph, kappa=100.0, alpha=1.0)
        _r, table = replay_plan(self.graph, [0], UNIT)
        self.assertAlmostEqual(
            table[0]["all_consumers_ready_s"], table[0]["finish_s"], places=9
        )


class TestForkJoinWitness(unittest.TestCase):
    def test_fork_join_is_satisfiable_when_anchored(self):
        graph = FakeGraph([10, 20, 10, 10], [(0, 1, 0), (0, 2, 0), (1, 3, 0), (2, 3, 0)])
        stamp_anchored(graph, kappa=1.0, alpha=1.0, resources=TIERS)
        witness = find_witness(graph, TIERS)
        self.assertTrue(witness.found, witness.reason)
        self.assertEqual(len(witness.actions), 4)
        for row in (witness.per_task or {}).values():
            self.assertGreaterEqual(row["slack_s"], -1e-9)

    def test_multi_sink_is_satisfiable_when_anchored(self):
        graph = FakeGraph([10, 10, 10], [(0, 1, 0), (0, 2, 0)], out_bytes={1: 20, 2: 40})
        stamp_anchored(graph, kappa=1.2, alpha=1.0, resources=TIERS)
        witness = find_witness(graph, TIERS)
        self.assertTrue(witness.found, witness.reason)
        self.assertEqual(set(witness.per_task), {0, 1, 2})

    def test_witness_follows_the_physics_not_a_preference(self):
        # MEC computes 10x faster here, but the ROOT's 50-byte input has to be
        # uploaded at 1 byte/s while local execution needs no transfer, so the
        # fastest plan keeps the root local and sends the rest to MEC: the witness
        # is what the scheduler accepts, not what the action ordering suggests
        graph = FakeGraph([50, 50, 50], [(0, 1, 0), (1, 2, 0)])
        _entry, fastest = stamp_anchored(graph, kappa=1.5, alpha=1.0, resources=TIERS)
        witness = find_witness(graph, TIERS)
        self.assertTrue(witness.found, witness.reason)
        self.assertEqual(witness.actions, fastest.actions)
        self.assertEqual(witness.actions[0], 0)
        self.assertEqual(set(witness.actions[1:]), {1})
        self.assertTrue(mixed_action_witness(witness.actions))

    def test_local_only_plan_is_a_witness_for_the_root(self):
        graph = FakeGraph([50, 50, 50], [(0, 1, 0), (1, 2, 0)])
        stamp_anchored(graph, kappa=1.5, alpha=1.0, resources=TIERS)
        result, table = replay_plan(graph, [0, 1, 1], TIERS)
        self.assertEqual(result.hard_miss_count, 0)
        for row in table.values():
            self.assertGreaterEqual(row["slack_s"], -1e-9)


class TestRelaxationInvariant(unittest.TestCase):
    def test_witness_faster_than_the_bound_raises(self):
        graph = FakeGraph([10, 10], [(0, 1, 0)])
        stamp_anchored(graph, kappa=2.0, alpha=1.0)
        # claim an impossible lower bound: the witness is faster than it
        with self.assertRaises(ValueError):
            find_witness(graph, UNIT, relaxed_lb_s=1e6)


class TestPlanBinding(unittest.TestCase):
    def test_plan_uses_the_decoder_order(self):
        graph = FakeGraph([10, 10, 10], [(0, 1, 0), (1, 2, 0)], order=[2, 1, 0])
        plan = plan_for(graph, [2, 1, 0])
        self.assertEqual(plan, [(2, 2), (1, 1), (0, 0)])

    def test_length_mismatch_is_rejected(self):
        graph = FakeGraph([10, 10], [(0, 1, 0)])
        with self.assertRaises(ValueError):
            plan_for(graph, [0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
