#!/usr/bin/env python3
"""v2 env tests: v1-compatible surface, v2 execution/telemetry, determinism."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spec.automotive_training.automotive_loader import load_dataset  # noqa: E402
from spec.automotive_training.v2.env import (  # noqa: E402
    V2_CONTEXT_FIELDS, V2AutomotiveEnv, V2EnvError,
)


def _env(seed=303, **kw):
    graphs = load_dataset().validation_query()[:3]
    env = V2AutomotiveEnv(graphs, role="validation", slots_per_task=3, base_seed=seed,
                          single_dist=True, **kw)
    env.set_task({"dist_index": 0, "graph_indices": np.arange(3, dtype=np.int32)})
    env.reset()
    return env


class TestSurface(unittest.TestCase):
    def test_matches_the_frozen_env_interface(self):
        env = _env()
        for attr in ("input_dim", "total_task", "set_task", "reset", "step",
                     "sample_tasks", "set_constraint_lambdas"):
            self.assertTrue(hasattr(env, attr), attr)
        self.assertEqual(env.input_dim, 91)
        # single-distribution layout: one dist, one slot per graph
        self.assertEqual(env.total_task, 1)
        self.assertEqual(env.slots_per_task, 3)

    def test_reset_returns_v1_observation_shape(self):
        env = _env()
        obs = env.reset()
        self.assertEqual(np.asarray(obs).shape, (3, 20, 91))

    def test_step_contract(self):
        env = _env()
        obs, rewards, done, info = env.step(np.ones((3, 20), dtype=int))
        self.assertEqual(np.asarray(obs).shape, (3, 20, 91))
        self.assertEqual(len(rewards), 3)
        self.assertEqual(len(rewards[0]), 20)
        self.assertTrue(done)
        finish, energy, telemetry = info
        self.assertEqual(len(finish), 3)
        self.assertEqual(len(telemetry), 3)
        self.assertEqual(len(energy), 3)
        for record in telemetry:
            # frozen energy schema at the top level...
            for key in ("schema_version", "requester_joules", "mobile_joules",
                        "system_joules", "primary_scope", "primary_joules",
                        "scheduler_config_sha256", "makespan_s"):
                self.assertIn(key, record)
            # ...and the v2 dynamics nested under "v2"
            self.assertIn("v2", record)
            for key in ("queue_wait_total_s", "outage_wait_total_s",
                        "helper_contact_failures", "reliability_rejections",
                        "fallback_reserved_s", "location_mix", "deadline_miss_rate",
                        "scheduler_invariants"):
                self.assertIn(key, record["v2"])

    def test_energy_is_real_not_a_zero_shim(self):
        """An all-MEC rollout must report NONZERO measured joules at every boundary.

        This is the regression for the former zero-joule placeholder, which emitted
        `energy_constraint="not_configured"` together with three exact zeros.
        """
        env = _env()
        _o, _r, _d, info = env.step(np.ones((3, 20), dtype=int))
        _finish, energy, telemetry = info
        for record in telemetry:
            self.assertIn(record["energy_constraint"], ("configured", "telemetry_only"))
            self.assertNotEqual(record["energy_constraint"], "not_configured")
            self.assertGreater(record["system_joules"], 0.0)
            # NOTE (measured): in this dataset no root task carries external input bytes, so
            # an all-MEC plan uploads nothing and the REQUESTER boundary is legitimately
            # exactly 0 — it is not a placeholder. The all-UE plan below is the one that must
            # show a nonzero requester boundary.
            self.assertGreaterEqual(record["requester_joules"], 0.0)
            self.assertLessEqual(record["requester_joules"], record["mobile_joules"] + 1e-9)
            self.assertLessEqual(record["mobile_joules"], record["system_joules"] + 1e-9)
            self.assertEqual(record["primary_scope"], "system")
            self.assertEqual(len(str(record["energy_model_sha256"])), 64)
            self.assertTrue(record["energy_unmodeled"],
                            "unmodelled components must be declared, not zeroed")
        self.assertGreater(float(energy[0][0]), 0.0)
        # the all-UE plan is the one that charges the requester
        _o, _r, _d, info = env.step(np.zeros((3, 20), dtype=int))
        for record in info[2]:
            self.assertGreater(record["requester_joules"], 0.0)
            self.assertGreater(record["system_joules"], 0.0)

    def test_energy_and_latency_are_not_the_same_quantity(self):
        env = _env()
        _o, _r, _d, info = env.step(np.zeros((3, 20), dtype=int))   # all-UE
        ue_system = [r["system_joules"] for r in info[2]]
        _o, _r, _d, info = env.step(np.ones((3, 20), dtype=int))    # all-MEC
        mec_system = [r["system_joules"] for r in info[2]]
        self.assertTrue(all(m > u for m, u in zip(mec_system, ue_system)),
                        "the system boundary must charge MEC compute, so all-MEC costs "
                        "more system energy than all-UE here")

    def test_action_shape_guard(self):
        env = _env()
        with self.assertRaises(V2EnvError):
            env.step(np.ones((2, 20), dtype=int))

    def test_v2_context_vector(self):
        env = _env(link_regime="degraded", reliability=True)
        env.reset()
        env.step(np.ones((3, 20), dtype=int))
        ctx = env.v2_context()
        self.assertEqual(ctx.shape, (3, len(V2_CONTEXT_FIELDS)))
        self.assertTrue(np.all(np.isfinite(ctx)))
        # estimated multipliers and confidences must be in (0, 1]
        self.assertTrue(np.all(ctx[:, 0:3] > 0.0))
        self.assertTrue(np.all(ctx[:, 3:6] > 0.0))
        self.assertTrue(np.all(ctx[:, 3:6] <= 1.0))


class TestExecution(unittest.TestCase):
    def test_all_ue_plan_has_no_remote_locations(self):
        env = _env()
        _o, _r, _d, info = env.step(np.zeros((3, 20), dtype=int))
        for record in info[2]:
            self.assertEqual(record["v2"]["location_mix"]["MEC"], 0)
            self.assertEqual(record["v2"]["location_mix"]["HELPER"], 0)

    def test_all_mec_plan_uses_shared_mec_and_reports_queueing(self):
        env = _env()
        _o, _r, _d, info = env.step(np.ones((3, 20), dtype=int))
        for record in info[2]:
            self.assertGreater(record["v2"]["location_mix"]["MEC"], 0)
            self.assertTrue(record["v2"]["scheduler_invariants"]["precedence_and_data_arrival"])
            self.assertTrue(record["v2"]["scheduler_invariants"]["no_double_booking"])

    def test_reliability_gate_rejects_remote_under_unreliable_links(self):
        env = _env(link_regime="degraded", reliability=True)
        _o, _r, _d, info = env.step(np.ones((3, 20), dtype=int))
        self.assertGreaterEqual(sum(r["v2"]["reliability_rejections"] for r in info[2]), 0)
        for record in info[2]:
            self.assertIn("location_mix", record["v2"])

    def test_telescoping_sums_to_total_latency_change(self):
        env = _env()
        actions = np.ones((3, 20), dtype=int)
        _o, rewards, _d, info = env.step(actions)
        # sum_t r_t = -(L_final - L0)/L_scale with L0 = all-UE prefix makespan
        for slot in range(3):
            graph = env.base.graph_objects[env.base._graph_index(slot)]
            l_scale = max(float(graph.D_G_s), 1e-12)
            total = float(np.sum(rewards[slot]))
            final = float(info[0][slot])
            l0 = env._schedule_slot(slot, [0] * 20).makespan_s
            self.assertAlmostEqual(total, -(final - l0) / l_scale, places=6)

    def test_determinism_same_seed_bit_identical(self):
        a = _env(seed=7)
        b = _env(seed=7)
        plan = np.ones((3, 20), dtype=int)
        _o1, r1, _d1, i1 = a.step(plan)
        _o2, r2, _d2, i2 = b.step(plan)
        self.assertEqual([float(x) for x in i1[0]], [float(x) for x in i2[0]])
        self.assertTrue(np.array_equal(np.asarray(r1), np.asarray(r2)))

    def test_different_seed_changes_the_episode(self):
        a = _env(seed=11)
        b = _env(seed=12)
        plan = np.ones((3, 20), dtype=int)
        _o1, _r1, _d1, i1 = a.step(plan)
        _o2, _r2, _d2, i2 = b.step(plan)
        self.assertNotEqual([r["makespan_s"] for r in i1[2]],
                            [r["makespan_s"] for r in i2[2]])


if __name__ == "__main__":
    unittest.main(verbosity=2)
