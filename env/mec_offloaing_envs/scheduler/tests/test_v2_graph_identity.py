#!/usr/bin/env python3
"""F01: the scheduled world, the MC source graph and the observation must be the SAME graph.

At the audited revision, ordinary training (single_dist=False) keeps `graph_indices` at all
zeros and resolves every slot of a task to `self.task_id`, but
`V2AutomotiveEnv._build_slot_world` read `graph_indices[flat]` directly and therefore built
graph0 for EVERY meta-task. The MC realization (drawn by the base env via `_graph_index`) and
the scheduled/observed world belonged to different graphs. This test pins the repair and the
acceptance criteria from F01.
"""
import sys
from pathlib import Path
import numpy as np
import unittest

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spec.automotive_training.automotive_loader import load_dataset  # noqa: E402
from spec.automotive_training.automotive_primary import AutomotiveResourceCluster  # noqa: E402
from spec.automotive_training.v2.env import V2AutomotiveEnv  # noqa: E402


class GraphIdentityTests(unittest.TestCase):
    def _env(self, *, single_dist, task_id=2, indices=None, slots=3):
        graphs = list(load_dataset().validation_query())[:6]
        # the corpus has DIFFERENT D_G_s per graph, so identity is observable
        assert len({round(g.D_G_s, 5) for g in graphs}) > 1
        env = V2AutomotiveEnv(graphs, AutomotiveResourceCluster(), role="meta_train",
                              slots_per_task=slots, base_seed=0, single_dist=single_dist)
        if single_dist:
            idx = np.arange(len(graphs), dtype=np.int32) if indices is None else indices
            env.set_task({"dist_index": 0, "graph_indices": idx})
        else:
            idx = np.zeros(slots, dtype=np.int32) if indices is None else indices
            env.set_task({"dist_index": task_id, "graph_indices": idx})
        env.reset()
        return env

    def test_non_single_dist_world_equals_task_graph(self):
        env = self._env(single_dist=False, task_id=2, slots=3)
        g2 = env.base.graph_objects[2]
        for flat in range(3):
            world = env._build_slot_world(flat)
            mc = env.base._slot_mc[flat]
            # MC source graph == task graph (base resolves _graph_index(slot) -> task_id)
            mc_graph = env.base.graph_objects[env.base._graph_index(flat)]
            self.assertEqual(world.dataset_graph_id, 2,
                             "world for flat %d must be the task graph" % flat)
            self.assertEqual(world.dataset_graph_id, env.base._graph_index(flat))
            self.assertEqual(mc_graph.graph_id, g2.graph_id)
            # observation graph id == world graph id
            ctx_graph = env.base.graph_objects[world.dataset_graph_id]
            self.assertEqual(ctx_graph.graph_id, mc_graph.graph_id)

    def test_single_dist_world_equals_slot_graph(self):
        env = self._env(single_dist=True, slots=6)
        for flat in range(6):
            world = env._build_slot_world(flat)
            self.assertEqual(world.dataset_graph_id, flat)

    def test_repeated_slots_get_same_graph_in_single_dist(self):
        # repeated graph IDs must map to the SAME dataset graph
        env = self._env(single_dist=True, indices=np.array([3, 3, 0, 0], dtype=np.int32),
                        slots=4)
        g = [env._build_slot_world(f).dataset_graph_id for f in range(4)]
        self.assertEqual(g[0], g[1])
        self.assertEqual(g[2], g[3])
        self.assertNotEqual(g[0], g[2])

    def test_nonzero_task_ids_across_the_corpus(self):
        # every nonzero task id must build ITS OWN graph, not graph0
        for tid in range(1, 6):
            env = self._env(single_dist=False, task_id=tid, slots=2)
            for flat in range(2):
                world = env._build_slot_world(flat)
                self.assertEqual(world.dataset_graph_id, tid,
                                 "task %d flat %d resolved to graph %d"
                                 % (tid, flat, world.dataset_graph_id))


if __name__ == "__main__":
    unittest.main(verbosity=2)


class MCIndependenceTests(unittest.TestCase):
    """F02: repeated support rows must NOT reuse the same MC realization."""

    def test_repeated_slots_draw_independent_realizations(self):
        graphs = list(load_dataset().validation_query())[:6]
        env = V2AutomotiveEnv(graphs, AutomotiveResourceCluster(), role="meta_train",
                              slots_per_task=20, base_seed=0, single_dist=False)
        env.set_task({"dist_index": 2, "graph_indices": np.zeros(20, dtype=np.int32)})
        env.reset()
        # the base _draw_slots seeds by FLAT position, so 20 trajectories get 20 draws
        seeds = [env.base.rollout_seed(
            env.base.graph_objects[env.base._graph_index(int(s))].graph_id, f)
            for f, s in enumerate(env.base.graph_indices)]
        self.assertEqual(len(set(seeds)), 20,
                         "repeated support rows reused one MC seed")
        # and the REALIZED DEMAND vectors actually stored are distinct (independent draws).
        # Survival sets MAY coincide if no seed crosses a LO/HI overrun threshold on this
        # particular graph; independence is about the draw, not the survivor count.
        realized = [tuple(round(float(r["realized_equiv"]), 6)
                          for r in env.base._slot_realizations[f].values())
                    for f in range(20)]
        self.assertGreater(len(set(realized)), 1,
                           "20 trajectories reused one identical realized demand vector")

    def test_replay_of_same_key_is_exact(self):
        graphs = list(load_dataset().validation_query())[:6]
        env = V2AutomotiveEnv(graphs, AutomotiveResourceCluster(), role="meta_train",
                              slots_per_task=4, base_seed=7, single_dist=False)
        env.set_task({"dist_index": 3, "graph_indices": np.zeros(4, dtype=np.int32)})
        env.reset()
        first = [tuple(env.base._slot_mc[f]['executed_task_ids']) for f in range(4)]
        env2 = V2AutomotiveEnv(graphs, AutomotiveResourceCluster(), role="meta_train",
                               slots_per_task=4, base_seed=7, single_dist=False)
        env2.set_task({"dist_index": 3, "graph_indices": np.zeros(4, dtype=np.int32)})
        env2.reset()
        second = [tuple(env2.base._slot_mc[f]['executed_task_ids']) for f in range(4)]
        self.assertEqual(first, second, "replay of the same key is not exact")
