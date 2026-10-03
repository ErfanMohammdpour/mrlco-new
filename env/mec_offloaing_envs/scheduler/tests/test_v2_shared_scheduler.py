#!/usr/bin/env python3
"""v2 shared multi-DAG scheduler tests: units, invariants, determinism, contention."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spec.automotive_training.automotive_env import AutomotiveEnv  # noqa: E402
from spec.automotive_training.automotive_loader import load_dataset  # noqa: E402
from spec.automotive_training.automotive_primary import (  # noqa: E402
    AutomotiveResourceCluster,
)
from spec.automotive_training.v2.adapters import (  # noqa: E402
    compute_spec,
    config_for,
    dag_spec_from_graph,
    link_spec,
    plan_map_from_actions,
    pure_plan,
)
from spec.automotive_training.v2.shared_scheduler import (  # noqa: E402
    HELPER,
    MEC,
    UE,
    V2ComputeSpec,
    V2DAGSpec,
    V2LinkSpec,
    V2ScheduleError,
    V2TaskSpec,
    schedule_shared,
)


def _env(graphs, seed=303):
    env = AutomotiveEnv(graphs, AutomotiveResourceCluster(), single_dist=True,
                        slots_per_task=len(graphs), base_seed=seed)
    env.set_task({"dist_index": 0, "graph_indices": np.arange(len(graphs), dtype=np.int32)})
    env.reset()
    return env


class TestSyntheticFixtures(unittest.TestCase):
    """Synthetic DAGs: the fixtures from the v2 adversarial list."""

    LINK = V2LinkSpec(mec_ul_bytes_per_s=20e6, mec_dl_bytes_per_s=20e6, v2v_bytes_per_s=10e6)
    COMPUTE = V2ComputeSpec(mec_cpu_bytes_per_s=10e6, mec_workers=1,
                            ue_cpu_bytes_per_s=(1e6,), helper_cpu_bytes_per_s=(2e6,))

    def _two_parallel(self):
        tasks = [V2TaskSpec(0, 10e6, 10e6, (), True, False, "HIGH", 0.5, 0),
                 V2TaskSpec(1, 10e6, 10e6, (), True, False, "HIGH", 0.5, 0),
                 V2TaskSpec(2, 1e6, 1e6, (0, 1), False, True, "HIGH", 0.5, 0)]
        return V2DAGSpec("parallel", 0, tasks, helper_id=0)

    def test_parallel_branches_use_two_workers(self):
        dag = self._two_parallel()
        plan = {"parallel": {0: 1, 1: 1, 2: 1}}
        one = schedule_shared([dag], plan, link=self.LINK, compute=self.COMPUTE)
        two = schedule_shared([dag], plan, link=self.LINK,
                              compute=V2ComputeSpec(10e6, 2, (1e6,), (2e6,)))
        self.assertLess(two.makespan_s, one.makespan_s,
                        "two independent MEC roots must overlap with 2 workers")
        self.assertLess(two.utilizations["MEC_CPU"], 1.0 + 1e-9)

    def test_serial_chain_all_mec_has_no_cut_transfer(self):
        tasks = [V2TaskSpec(i, 5e6, 5e6, ((i - 1,) if i else ()), i == 0, i == 4, "HIGH", 1.0, 0)
                 for i in range(5)]
        dag = V2DAGSpec("chain", 0, tasks)
        res = schedule_shared([dag], {"chain": pure_plan_stub(5, 1)},
                              link=self.LINK, compute=self.COMPUTE)
        # all-MEC: only the sink return (MEC->UE) moves over the radio
        self.assertEqual(res.mechanics["radio_events"], 1)
        mixed = pure_plan_stub(5, 1)
        mixed[2] = 2
        res2 = schedule_shared([dag], {"chain": mixed}, link=self.LINK, compute=self.COMPUTE)
        # a single HELPER token in the middle creates cut edges (in and out) + sink return
        self.assertGreaterEqual(res2.mechanics["radio_events"], 3)

    def test_overloaded_mec_makes_local_preferable(self):
        tasks = [V2TaskSpec(0, 30e6, 1e6, (), True, True, "LOW", 1.0, 0)]
        dag = V2DAGSpec("single", 0, tasks)
        # 8 concurrent single-task DAGs: the shared MEC queues, local UE does not
        # 16 concurrent DAGs vs a 10x faster single MEC server: 16*3 s of MEC service
        # beats neither, so local (1 s per DAG on its own vehicle) wins under overload
        dags = [V2DAGSpec("d%d" % i, i, [V2TaskSpec(0, 3e6, 1e6, (), True, True, "LOW", 1.0, i)])
                for i in range(16)]
        compute = V2ComputeSpec(10e6, 1, tuple([1e6] * 16), ())
        mec = schedule_shared(dags, {"d%d" % i: {0: 1} for i in range(16)},
                              link=self.LINK, compute=compute)
        ue = schedule_shared(dags, {"d%d" % i: {0: 0} for i in range(16)},
                             link=self.LINK, compute=compute)
        self.assertGreater(mec.queue_stats["cpu_wait_max_s"], 0.0)
        self.assertLess(ue.makespan_s, mec.makespan_s)

    def test_poor_v2v_makes_helper_unattractive(self):
        from spec.automotive_training.v2.helper_model import HelperState

        tasks = [V2TaskSpec(0, 10e6, 10e6, (), True, True, "MEDIUM", 1.0, 0)]
        dag = V2DAGSpec("h", 0, tasks, helper_id=0)
        state = HelperState(helper_id=0, cpu_bytes_per_s=2e6, contact_end_s=100.0,
                            predicted_contact_end_s=100.0)
        fast = schedule_shared([dag], {"h": {0: 2}}, link=V2LinkSpec(20e6, 20e6, 50e6),
                               compute=self.COMPUTE, helper_states={0: state})
        slow = schedule_shared([dag], {"h": {0: 2}}, link=V2LinkSpec(20e6, 20e6, 0.5e6),
                               compute=self.COMPUTE, helper_states={0: state})
        self.assertLess(fast.makespan_s, slow.makespan_s * 0.9)

    def test_later_long_reservation_does_not_block_earlier_short_transfer(self):
        """Regression: a shared channel booked for a later long transfer must not stall a
        short transfer that is ready earlier (FIFO-by-arrival did exactly that)."""
        from spec.automotive_training.v2.shared_scheduler import Calendar

        cal = Calendar("shared")
        late_start, late_end = cal.reserve(10.0, 5.0)      # long transfer booked first
        early_start, early_end = cal.reserve(0.0, 0.001)   # short one ready much earlier
        self.assertAlmostEqual(early_start, 0.0, places=9)
        self.assertLessEqual(early_end, 0.01)
        self.assertAlmostEqual(late_start, 10.0, places=9)
        self.assertAlmostEqual(late_end, 15.0, places=9)

    def test_helper_absent_degrades_to_local(self):
        tasks = [V2TaskSpec(0, 10e6, 1e6, (), True, True, "LOW", 1.0, 0)]
        dag = V2DAGSpec("n", 0, tasks, helper_id=None)
        res = schedule_shared([dag], {"n": {0: 2}}, link=self.LINK, compute=self.COMPUTE)
        self.assertEqual(res.timings[("n", 0)].location, UE)


def pure_plan_stub(n, action):
    return {i: action for i in range(n)}


class TestFrozenGraphIntegration(unittest.TestCase):
    def setUp(self):
        ds = load_dataset()
        self.graphs = ds.validation_query()[:3]
        self.env = _env(self.graphs)
        self.plans = [([1] * 20), ([1] * 20), ([1] * 20)]

    def test_single_dag_matches_v1_makespan_within_tolerance(self):
        graph = self.graphs[0]
        mc = self.env._slot_mc[0]
        v1, _e, _m = self.env._schedule(0, [1] * 20, mc)
        link = link_spec(graph)
        compute = compute_spec([graph])
        dag = dag_spec_from_graph(graph, dag_id="g0", owner=0, mc=mc)
        res = schedule_shared([dag], {"g0": plan_map_from_actions(graph, [1] * 20)},
                              link=link, compute=compute)
        v1_ms = float(v1.makespan_seconds)
        self.assertAlmostEqual(res.makespan_s, v1_ms, delta=0.10 * v1_ms,
                               msg="v2 single-DAG must be close to v1 (delta<=10%)")

    def test_two_concurrent_dags_queue_on_shared_mec(self):
        graph = self.graphs[0]
        mc = self.env._slot_mc[0]
        link = link_spec(graph)
        compute = compute_spec([graph, graph])
        dags = [dag_spec_from_graph(graph, dag_id="a", owner=0, mc=mc),
                dag_spec_from_graph(graph, dag_id="b", owner=1, mc=mc)]
        plans = {d.dag_id: plan_map_from_actions(graph, [1] * 20) for d in dags}
        serial = schedule_shared([dags[0]], {"a": plans["a"]}, link=link, compute=compute)
        both = schedule_shared(dags, plans, link=link, compute=compute)
        self.assertGreaterEqual(both.makespan_s, serial.makespan_s)
        self.assertGreater(both.utilizations["MEC_CPU"], 0.0)
        self.assertTrue(both.invariants["precedence_and_data_arrival"])
        self.assertTrue(both.invariants["utilization_bounds"])

    def test_determinism_same_inputs_bit_identical(self):
        graph = self.graphs[1]
        mc = self.env._slot_mc[1]
        link = link_spec(graph)
        compute = compute_spec([graph])
        dag = dag_spec_from_graph(graph, dag_id="g", owner=0, mc=mc)
        plan = plan_map_from_actions(graph, [1, 2] * 10)
        a = schedule_shared([dag], {"g": plan}, link=link, compute=compute)
        b = schedule_shared([dag], {"g": plan}, link=link, compute=compute)
        self.assertEqual(a.makespan_s, b.makespan_s)
        for key in a.timings:
            self.assertEqual(a.timings[key].finish_s, b.timings[key].finish_s)

    def test_n1_reduces_to_single_dag_semantics(self):
        graph = self.graphs[2]
        mc = self.env._slot_mc[2]
        link = link_spec(graph)
        compute = compute_spec([graph])
        dag = dag_spec_from_graph(graph, dag_id="solo", owner=0, mc=mc)
        res = schedule_shared([dag], {"solo": plan_map_from_actions(graph, [1] * 20)},
                              link=link, compute=compute)
        self.assertEqual(len(res.completion_by_dag), 1)
        self.assertLessEqual(res.utilizations["MEC_CPU"], 1.0 + 1e-9)
        self.assertEqual(res.invariants["completion_consistent"], True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
