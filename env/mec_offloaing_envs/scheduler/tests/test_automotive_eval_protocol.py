#!/usr/bin/env python3
"""Unit tests for the frozen paired multi-realization evaluation protocol.

The 5x500 pilot compared k0 and k3 across DIFFERENT mixed-criticality realizations
(k0 used base_seed 101, the k3 query used 303, and the k3 path resets the env one extra
time so even the same seed drew a different realization) and compared the model against
baselines computed on yet another seed. These tests pin the fix.
"""

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
    rollout_metric_block,
)
from spec.automotive_training.eval_protocol import (  # noqa: E402
    ADAPTATION_K_STEPS,
    BASELINE_CANDIDATES,
    EvalProtocolError,
    PRIMARY_DECODING,
    REALIZATION_SEEDS,
    REPORT_REPLICATES,
    SECONDARY_DECODING,
    SELECT_REPLICATES,
    aggregate,
    candidate_oracle,
    decoding_flags,
    frozen_protocol,
    protocol_sha,
    realization_seeds,
    regret,
)


class TestProtocolFreeze(unittest.TestCase):
    def test_seeds_are_committed_and_prefix_stable(self):
        self.assertEqual(len(REALIZATION_SEEDS), REPORT_REPLICATES)
        self.assertEqual(REALIZATION_SEEDS[0], 1000)
        self.assertEqual(REALIZATION_SEEDS[1] - REALIZATION_SEEDS[0], 17)
        self.assertEqual(realization_seeds(SELECT_REPLICATES), REALIZATION_SEEDS[:SELECT_REPLICATES])
        self.assertEqual(realization_seeds(REPORT_REPLICATES), REALIZATION_SEEDS)
        for bad in (0, -1, REPORT_REPLICATES + 1):
            with self.assertRaises(EvalProtocolError):
                realization_seeds(bad)

    def test_sha_is_stable_and_protocol_is_explicit(self):
        self.assertEqual(protocol_sha(), protocol_sha())
        doc = frozen_protocol()
        self.assertEqual(doc["select_replicates"], 5)
        self.assertEqual(doc["report_replicates"], 20)
        self.assertEqual(doc["primary_decoding"], "deterministic")
        self.assertEqual(list(ADAPTATION_K_STEPS), [0, 1, 2, 3, 5])
        self.assertIn("all_MEC", doc["baseline_candidates"])

    def test_decoding_flags(self):
        self.assertTrue(decoding_flags(PRIMARY_DECODING)["greedy"])
        self.assertFalse(decoding_flags(SECONDARY_DECODING)["greedy"])
        with self.assertRaises(EvalProtocolError):
            decoding_flags("whatever")

    def test_oracle_and_regret(self):
        panel = {"all_UE": 0.28, "all_MEC": 0.0262, "all_HELPER": 0.17,
                 "heft_reference_v2": 0.0273, "greedy_coordinate_descent_MC": 0.0348}
        name, best = candidate_oracle(panel)
        self.assertEqual(name, "all_MEC")
        self.assertAlmostEqual(best, 0.0262)
        self.assertAlmostEqual(regret(0.0300, panel), 0.0300 - 0.0262, places=12)
        self.assertAlmostEqual(regret(0.0200, panel), 0.0200 - 0.0262, places=12)  # negative = beats oracle
        with self.assertRaises(EvalProtocolError):
            candidate_oracle({"unknown": 1.0})
        self.assertEqual(list(BASELINE_CANDIDATES)[1], "all_MEC")

    def test_aggregate(self):
        rows = [{"x": 1.0}, {"x": 3.0}, {"x": 5.0}]
        out = aggregate(rows, ["x"])
        self.assertAlmostEqual(out["x"], 3.0)
        self.assertAlmostEqual(out["x_min"], 1.0)
        self.assertAlmostEqual(out["x_max"], 5.0)
        self.assertGreater(out["x_std"], 0.0)
        self.assertEqual(out["replicates"], 3)


class TestPairing(unittest.TestCase):
    """The pinned epoch must make the realization independent of reset_count."""

    def _env(self, base_seed, epoch, resets=0):
        graphs = load_dataset().validation_query()[:8]
        env = AutomotiveEnv(graphs, AutomotiveResourceCluster(), role="validation",
                            single_dist=True, slots_per_task=len(graphs),
                            base_seed=base_seed, mc_enabled=True)
        env.set_task({"dist_index": 0,
                      "graph_indices": np.arange(len(graphs), dtype=np.int32)})
        env.realization_epoch = epoch
        for _ in range(resets + 1):
            env.reset()
        return env

    def _signature(self, env):
        out = []
        for mc in env._slot_mc:
            out.append((
                str(mc.get("final_mode")),
                len(mc.get("switches", [])),
                len(mc.get("dropped_task_ids", [])),
                len(mc.get("capped_to_hi", [])),
                tuple(sorted((str(k), round(float(v), 9))
                             for k, v in (mc.get("realized_equiv_by_task") or {}).items()))[:4],
            ))
        return out

    def test_same_epoch_is_independent_of_reset_count(self):
        a = self._signature(self._env(1000, 0, resets=0))
        b = self._signature(self._env(1000, 0, resets=3))
        self.assertEqual(a, b, "the pinned epoch must survive extra resets (k3 path)")

    def test_different_epoch_or_seed_changes_the_realization(self):
        base = self._signature(self._env(1000, 0))
        self.assertNotEqual(base, self._signature(self._env(1000, 1)),
                            "a different epoch must draw a different realization")
        self.assertNotEqual(base, self._signature(self._env(1017, 0)),
                            "a different base seed must draw a different realization")

    def test_seed_reported_in_telemetry_matches_the_pinned_epoch(self):
        env = self._env(1000, 4)
        obs = env.reset()
        from spec.automotive_training.automotive_primary import per_graph_makespans
        _o, _r, _d, info = env.step(np.zeros((len(env.graph_indices), 20), dtype=int))
        records = [t for t in info[2]]
        expected = [env.rollout_seed(env.graph_objects[env._graph_index(slot)].graph_id, slot, 4)
                    for slot in range(len(env.graph_indices))]
        self.assertEqual([int(t["rollout_seed"]) for t in records], expected)


class TestMetricBlock(unittest.TestCase):
    def test_denominators_and_rates(self):
        def rec(slot, tardy_h, tardy_m, n_h=4, n_m=2, makespan=0.1):
            return {slot: {"graph_index": slot, "makespan_s": makespan, "hard_violation": 0.0,
                           "high_tardiness": 0.001 if tardy_h else 0.0,
                           "medium_tardiness": 0.002 if tardy_m else 0.0,
                           "firm_miss": tardy_h + tardy_m, "mode": "HI", "switches": 1.0,
                           "dropped": 2.0, "n_tasks_high": n_h, "n_tasks_medium": n_m,
                           "n_tardy_high": tardy_h, "n_tardy_medium": tardy_m,
                           "high_preserved": True}}
        block = rollout_metric_block({**rec(0, 1, 1), **rec(1, 0, 0)})
        self.assertAlmostEqual(block["high_task_tardiness_task_rate"], 1.0 / 8.0)
        self.assertAlmostEqual(block["medium_task_tardiness_task_rate"], 1.0 / 4.0)
        self.assertAlmostEqual(block["firm_task_miss_rate"], 2.0 / 12.0)
        self.assertAlmostEqual(block["graph_high_tardiness_incidence_rate"], 0.5)
        self.assertEqual(block["firm_task_miss_count"], 2.0)
        self.assertEqual(block["mc_policy_violation_count"], 0.0)
        self.assertAlmostEqual(block["hi_mode_rate"], 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
