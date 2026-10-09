#!/usr/bin/env python3
"""F04: the MC mode walk must use the SLOT'S OWN dataset graph order.

Audited at HEAD 9e80a16: `AutomotiveEnv._draw_slots` passed `self.orders[self.task_id]` to
`mc_runtime.resolve_mode_and_execution`. In the `single_dist` layout `task_id` is the
DISTRIBUTION index (always 0), so every slot's LO/HI mode walk used graph 0's decoder order.
The defect is inert only while every corpus graph shares one stored order; with a
topologically heterogeneous corpus `plan_order_ids` walks (or rejects) a foreign order.

The regression builds two graphs that share one DAG topology but carry DIFFERENT valid
decoder orders, and asserts that each slot's `logical_schedule_point` (its position in the
plan order the mode machine actually walked) is its OWN graph's position.
"""

from __future__ import annotations

import dataclasses
import heapq
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spec.automotive_training.automotive_dag import decoder_order  # noqa: E402
from spec.automotive_training.automotive_env import AutomotiveEnv  # noqa: E402
from spec.automotive_training.automotive_loader import load_dataset  # noqa: E402
from spec.automotive_training.automotive_primary import AutomotiveResourceCluster  # noqa: E402

SLOTS = 2


def _different_topological_order(graph) -> tuple:
    """A DIFFERENT valid topological order of the same DAG (task ids unchanged).

    Reverse-priority Kahn: ready tasks are consumed from the LARGEST id down, which is a
    valid topological order whenever the stored one is, and differs from it whenever the
    DAG has at least two independent ready tasks.
    """
    stored = [int(t) for t in decoder_order(graph.as_record())]
    indeg = {int(t.task_id): 0 for t in graph.tasks}
    succ = {int(t.task_id): [] for t in graph.tasks}
    for edge in graph.edges:
        src, dst = int(edge.src), int(edge.dst)
        succ[src].append(dst)
        indeg[dst] += 1
    ready = [-tid for tid, deg in indeg.items() if deg == 0]
    heapq.heapify(ready)
    out = []
    while ready:
        tid = -heapq.heappop(ready)
        out.append(tid)
        for child in succ[tid]:
            indeg[child] -= 1
            if indeg[child] == 0:
                heapq.heappush(ready, -child)
    if len(out) != len(stored) or sorted(out) != sorted(stored):
        raise AssertionError("alternative order is not a permutation of the task ids")
    # topological check against the frozen edges
    position = {tid: i for i, tid in enumerate(out)}
    for edge in graph.edges:
        assert position[int(edge.src)] < position[int(edge.dst)]
    return tuple(out)


class TestPerGraphMCOrder(unittest.TestCase):
    def _graphs(self):
        source = load_dataset().validation_query()[0]
        stored = tuple(decoder_order(source.as_record()))
        other = _different_topological_order(source)
        self.assertNotEqual(list(stored), list(other),
                            "the fixture DAG must admit a second topological order")
        by_id = {int(t.task_id): t for t in source.tasks}
        first = dataclasses.replace(source, graph_id="f04_order_a",
                                    tasks=tuple(by_id[tid] for tid in stored))
        second = dataclasses.replace(source, graph_id="f04_order_b",
                                     tasks=tuple(by_id[tid] for tid in other))
        self.assertEqual(decoder_order(first.as_record()), list(stored))
        self.assertEqual(decoder_order(second.as_record()), list(other))
        self.assertEqual([e for e in first.edges], [e for e in second.edges])
        return first, second, stored, other

    def test_each_slot_walks_its_own_graphs_plan_order(self):
        first, second, stored, other = self._graphs()
        env = AutomotiveEnv([first, second], AutomotiveResourceCluster(),
                            role="meta_train", slots_per_task=SLOTS, base_seed=0,
                            single_dist=True)
        env.set_task({"dist_index": 0, "graph_indices": np.arange(SLOTS, dtype=np.int32)})
        env.reset()
        expected_orders = {0: list(stored), 1: list(other)}
        for flat in range(SLOTS):
            graph_index = env._graph_index(flat)
            order = env.orders[graph_index]
            self.assertEqual(order, expected_orders[graph_index])
            mc = env._slot_mc[flat]
            executed = [int(t) for t in mc["executed_task_ids"]]
            self.assertTrue(executed, "slot %d executed no task" % flat)
            for tid in executed:
                self.assertEqual(
                    int(mc["execution"][tid]["logical_schedule_point"]), order.index(tid),
                    "slot %d (graph %d) walked a FOREIGN plan order: task %d has position "
                    "%d under the slot's own order and %d in the recorded MC"
                    % (flat, graph_index, tid, order.index(tid),
                       int(mc["execution"][tid]["logical_schedule_point"])))

    def test_replay_is_exact_and_slots_stay_independent_under_foreign_orders(self):
        """The F04 repair must not disturb per-slot seed independence or exact replay."""
        first, second, _stored, _other = self._graphs()

        def build():
            env = AutomotiveEnv([first, second], AutomotiveResourceCluster(),
                                role="meta_train", slots_per_task=SLOTS, base_seed=3,
                                single_dist=True)
            env.set_task({"dist_index": 0,
                          "graph_indices": np.arange(SLOTS, dtype=np.int32)})
            env.reset()
            return env

        def signature(env):
            return [
                (tuple(int(t) for t in mc["executed_task_ids"]),
                 tuple(int(mc["execution"][int(t)]["logical_schedule_point"])
                       for t in mc["executed_task_ids"]))
                for mc in env._slot_mc
            ]

        a, b = build(), build()
        self.assertEqual(signature(a), signature(b), "replay of the same seed is not exact")
        # per-slot independence: the seed is keyed by the FLAT slot position
        seeds = [a.rollout_seed(a.graph_objects[a._graph_index(flat)].graph_id, flat)
                 for flat in range(SLOTS)]
        self.assertEqual(len(set(seeds)), SLOTS)
        realized = [tuple(round(float(r["realized_equiv"]), 6)
                          for r in a._slot_realizations[flat].values())
                    for flat in range(SLOTS)]
        self.assertEqual(len(realized), SLOTS)
        self.assertGreater(len(set(realized)), 1,
                           "two slots reused one identical realized demand vector")


if __name__ == "__main__":
    unittest.main(verbosity=2)
