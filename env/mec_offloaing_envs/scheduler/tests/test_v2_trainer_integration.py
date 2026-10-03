#!/usr/bin/env python3
"""Integration gaps found by the external review of HEAD ddfcfa7, now covered by tests.

1. `_packed_observation` mapped `graph_indices` (DATASET graph ids) to `_context_vector`
   (which expects FLAT SLOT POSITIONS). Two slots sharing a graph ([0, 0]) both received slot
   0's context, so contact/world/helper differences between trajectories were dropped from the
   observation.
2. The v2 telemetry carried only miss COUNTS, not the `violation/<NAME>` keys the frozen
   trainer observer reads, so `constraint_violations_from_telemetry` returned {} and a budget
   could never influence training.
3. `env.last_constraint_costs` was a plain list, but the frozen trainer reads
   `.active`/`.as_dict()` on it.
4. The v2 constraint manager lacked the members the frozen loop calls on a controller
   (`status`, `reset_batch`, `batch_size`) and had no way to accept the SIGNED costs the
   telemetry reports.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spec.automotive_training.automotive_loader import load_dataset  # noqa: E402
from spec.automotive_training.automotive_primary import AutomotiveResourceCluster  # noqa: E402
from spec.automotive_training.automotive_trainer import (  # noqa: E402
    constraint_violations_from_telemetry,
)
from spec.automotive_training.v2.constraint_channels import (  # noqa: E402
    channel_telemetry, v1_channels,
)
from spec.automotive_training.v2.constraints_v2 import (  # noqa: E402
    ConstraintCostBatch, V2ConstraintManager, calibrate_budgets,
)
from spec.automotive_training.v2.energy import (  # noqa: E402
    reference_ranges_from_plans, schedule_energy,
)
from spec.automotive_training.v2.env import V2AutomotiveEnv  # noqa: E402
from spec.automotive_training.v2.shared_scheduler import (  # noqa: E402
    V2ComputeSpec, V2DAGSpec, V2LinkSpec, V2TaskSpec, schedule_shared,
)

SHA = "d" * 64
REPEATED_INDICES = [0, 0]


def _env(indices=REPEATED_INDICES, constraints=True, seed=303, background=0):
    idx = np.asarray(indices, dtype=np.int32)
    graphs = load_dataset().validation_query()[:max(2, int(idx.max()) + 1, int(idx.size))]
    env = V2AutomotiveEnv(graphs, AutomotiveResourceCluster(), role="validation",
                          slots_per_task=int(idx.size), base_seed=seed, single_dist=True,
                          background_dags=int(background),
                          constraints_enabled=bool(constraints),
                          budget_fractions={"total_energy": 2.0},
                          scheduler_config_sha256=SHA)
    env.set_task({"dist_index": 0, "graph_indices": idx})
    return env


class TestSlotContextMapping(unittest.TestCase):
    def test_two_slots_sharing_a_graph_get_distinct_context_slots(self):
        env = _env(REPEATED_INDICES)
        env.reset()
        contexts = env._contexts()          # built BEFORE the spy so only the packing counts
        calls = []
        original = env._context_vector

        def spy(flat):
            calls.append(int(flat))
            return original(flat)

        env._context_vector = spy
        env._packed_observation(contexts)
        self.assertEqual(calls, list(range(len(REPEATED_INDICES))),
                         "the observation must resolve contexts by FLAT SLOT POSITION "
                         "([0, 1]), not by the dataset graph id ([0, 0])")

    def test_shared_graph_slots_still_differ_in_contact(self):
        import dataclasses

        env = _env(REPEATED_INDICES)
        env.reset()
        short = dataclasses.replace(env.helper_states[1][0], contact_end_s=0.4,
                                    predicted_contact_end_s=0.6)
        env._worlds[1].helpers = {0: short}
        env.helper_states[1] = env._worlds[1].helpers
        obs = env._packed_observation(env._contexts())
        self.assertEqual(np.asarray(obs).shape[0], 2)
        self.assertFalse(np.array_equal(obs[0], obs[1]),
                         "slots with the same graph but different contact must not share a "
                         "context row")

    def test_context_slot_and_dataset_graph_are_distinct_concepts(self):
        # `single_dist=True` requires one graph index per slot, and a nonzero DATASET index
        # must not be confused with the flat SLOT position
        env = _env([3, 0, 1, 2])
        env.reset()
        self.assertEqual(env.world_for_slot(0).slot_id, 0)
        self.assertEqual(env.world_for_slot(0).dataset_graph_id, 3)
        self.assertEqual(env.world_for_slot(1).dataset_graph_id, 0)


class TestTrainerConstraintChannels(unittest.TestCase):
    def test_the_frozen_observer_now_reads_non_empty_channels(self):
        env = _env()
        env.reset()
        _o, _r, _d, info = env.step(np.ones((2, 20), dtype=int))
        record = info[2][0]
        for name in ("C_GRAPH_HARD_DEADLINE", "C_HI_TASK_TARDINESS", "C_MED_TASK_TARDINESS"):
            self.assertIn("violation/%s" % name, record)
            self.assertIn("lambda/%s" % name, record)
            self.assertIn("n_violating/%s" % name, record)
        paths = {"t": [{"energy_telemetry": [record]}]}
        observed = constraint_violations_from_telemetry(paths)
        self.assertTrue(observed, "the frozen observer returned {} — a budget could not reach "
                                  "the controller")

    def test_channels_come_from_the_frozen_evaluator(self):
        env = _env(constraints=False)
        env.reset()
        result = env._schedule_slot(0, [1] * 20, validate=True)
        world = env.world_for_slot(0)
        graph = env.base.graph_objects[world.dataset_graph_id]
        channels = v1_channels(graph, result, foreground_dag=world.foreground,
                               d_g_s=float(graph.D_G_s))
        telemetry = channel_telemetry(graph, result, foreground_dag=world.foreground,
                                      d_g_s=float(graph.D_G_s), l_scale=float(graph.D_G_s))
        for name, channel in channels.items():
            if name.startswith("_"):
                continue
            self.assertEqual(telemetry["violation/%s" % name], float(channel["violation"]))
        # deadline channels are SECONDS of violation, not counts
        self.assertGreaterEqual(telemetry["violation/C_GRAPH_HARD_DEADLINE"], 0.0)
        self.assertEqual(telemetry["task_count"], len(result.timings))
        self.assertIn("unjudged_tasks_no_subdeadline", telemetry)

    def test_only_foreground_tasks_are_charged(self):
        env = _env(background=3)
        env.reset()
        result = env._schedule_slot(0, [1] * 20, validate=True)
        world = env.world_for_slot(0)
        graph = env.base.graph_objects[world.dataset_graph_id]
        channels = v1_channels(graph, result, foreground_dag=world.foreground,
                               d_g_s=float(graph.D_G_s))
        self.assertGreater(len(result.timings), len([t for t in world.foreground.tasks]))
        self.assertLessEqual(len([t for t in world.foreground.tasks]),
                             len([t for t in world.foreground.tasks]))


class TestConstraintBatchAndDuals(unittest.TestCase):
    def test_last_constraint_costs_behaves_like_the_frozen_single_cost(self):
        env = _env()
        env.reset()
        env.step(np.ones((2, 20), dtype=int))
        batch = env.last_constraint_batch
        self.assertIsInstance(batch, ConstraintCostBatch)
        self.assertTrue(batch.active)
        doc = batch.as_dict()
        self.assertIn("total_violation", doc)
        self.assertIn("episodes", doc)

    def test_signed_costs_move_lambda_up_and_down(self):
        env = _env()
        env.reset()
        _o, _r, _d, info = env.step(np.ones((2, 20), dtype=int))
        manager = env.constraint_manager
        rows = [{name: float(record["constraints"]["%s_signed" % name])
                 for name in record["constraints"]["names"]} for record in info[2]]
        self.assertEqual(manager.observe_signed(rows), 2)
        self.assertGreater(manager.batch_size(), 0)
        manager.dual_step()
        high = max(manager.lambdas)
        self.assertGreater(high, 0.0, "an over-budget batch must raise lambda")
        for _ in range(30):
            manager.observe_signed([{name: -1.0 for name in manager.names}])
        manager.dual_step()
        self.assertLess(max(manager.lambdas), high,
                        "signed ascent must be able to LOWER lambda below budget")
        self.assertGreaterEqual(max(manager.lambdas), 0.0)
        self.assertIsInstance(manager.status(), dict)
        manager.reset_batch()
        self.assertEqual(manager.batch_size(), 0)

    def test_the_manager_exposes_the_members_the_frozen_loop_calls(self):
        env = _env()
        env.reset()
        manager = env.constraint_manager
        for member in ("spec", "names", "lambdas", "dual_step", "status", "reset_batch",
                       "batch_size", "observe_signed", "lambdas_by_name"):
            self.assertTrue(hasattr(manager, member), member)
        self.assertTrue(manager.spec.enabled)
        self.assertEqual(manager.spec.mode, "lagrangian")

    def test_missing_signed_cost_is_an_error_not_a_zero(self):
        env = _env()
        env.reset()
        manager = env.constraint_manager
        with self.assertRaises(Exception):
            manager.observe_signed([{"total_energy": float("nan")}])
        self.assertEqual(manager.observe_signed([{"unknown_channel": 1.0}]), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
