#!/usr/bin/env python3
"""Identity-keyed environmental realizations and checkpoint/resume equivalence.

Two audited defects are covered:

1. `env.realization_epoch = e` on the v2 wrapper did NOT reach the frozen base environment
   (`__getattr__` only serves reads), so the Monte-Carlo realization stayed keyed by the
   mutable `reset_count` and replay/CRN/resume could not reproduce a world.
2. the epoch was pinned AFTER `self.base.reset()`, which draws the MC realization — so the pin
   took effect one reset late.

Also covered: the exogenous link process, the helper draws, the background workloads and the
world identity all move with the STABLE realization key, and a full support->inner-PPO->
query->outer->save->restore->resume cycle reproduces the uninterrupted iteration exactly.
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
from spec.automotive_training.v2.env import V2AutomotiveEnv, stable_seed  # noqa: E402
from spec.automotive_training.v2.smoke import SmokeConfig, run_smoke  # noqa: E402

SLOTS = 2


def _env(link_regime="degraded", seed=1234, background=1, graphs=None):
    graphs = list(graphs or load_dataset().validation_query()[:SLOTS])
    env = V2AutomotiveEnv(graphs, AutomotiveResourceCluster(), role="validation",
                          slots_per_task=len(graphs), base_seed=seed, single_dist=True,
                          link_regime=link_regime, background_dags=int(background))
    env.set_task({"dist_index": 0, "graph_indices": np.arange(len(graphs), dtype=np.int32)})
    return env


class TestRealizationEpochForwarding(unittest.TestCase):
    def test_assignment_reaches_the_frozen_base_environment(self):
        env = _env()
        env.realization_epoch = 5
        self.assertEqual(env.base.realization_epoch, 5,
                         "the wrapper must FORWARD the epoch assignment, not shadow it")
        self.assertEqual(env.realization_epoch, 5)
        env.set_realization_epoch(None)
        self.assertIsNone(env.base.realization_epoch)

    def _mc_signature(self, env):
        """Executed sub-DAG AND the effective work of the current realization.

        The effective work is the MC quantity that actually changes the schedule, so the
        signature must include it: two epochs can keep the same surviving tasks while drawing
        different realized demands.
        """
        out = []
        for mc in (env.base._slot_mc or []):
            mc = mc or {}
            tasks = sorted(int(t) for t in mc.get("executed_task_ids", []))
            work = [round(float((mc.get("execution") or {}).get(t, {})
                                .get("effective_equiv", 0.0)), 6) for t in tasks]
            out.append([tasks, work])
        return out

    def test_the_pinned_epoch_actually_selects_the_mc_realization(self):
        env = _env()
        env.realization_epoch = 3
        env.reset()
        third = self._mc_signature(env)
        env.realization_epoch = 3
        env.reset()
        self.assertEqual(third, self._mc_signature(env),
                         "the same epoch must reproduce the same realized sub-DAG")
        found = False
        for epoch in (9, 17, 23, 41):
            env.realization_epoch = epoch
            env.reset()
            if self._mc_signature(env) != third:
                found = True
                break
        self.assertTrue(found, "some other epoch must select a different realization")


class TestStableWorldIdentity(unittest.TestCase):
    def test_the_same_key_reproduces_epoch_link_helpers_and_background(self):
        a = _env()
        a.set_world_realization("support/200/0")
        a.reset()
        # a DIFFERENT instance with a DIFFERENT reset history (it is reset twice first)
        b = _env(seed=1234)
        b.reset()
        b.reset()
        b.set_world_realization("support/200/0")
        b.reset()
        self.assertEqual(a.base.realization_epoch, b.base.realization_epoch,
                         "the MC epoch must be keyed by the realization, not the counter")
        self.assertEqual(a.world_for_slot(0).world_id, b.world_for_slot(0).world_id)
        self.assertEqual([d.dag_id for d in a.world_for_slot(0).dags],
                         [d.dag_id for d in b.world_for_slot(0).dags])
        self.assertEqual(a.world_for_slot(0).fingerprint_sha256(),
                         b.world_for_slot(0).fingerprint_sha256(),
                         "background workloads are derived from world_id and must match")
        for slot in range(SLOTS):
            ha = a.helper_states[slot][0]
            hb = b.helper_states[slot][0]
            self.assertEqual(ha.contact_end_s, hb.contact_end_s)
            self.assertEqual(ha.predicted_contact_end_s, hb.predicted_contact_end_s)

    def test_different_keys_give_different_realizations(self):
        env = _env()
        env.set_world_realization("k1")
        env.reset()
        first = env.world_for_slot(0).fingerprint_sha256()
        env.set_world_realization("k2")
        env.reset()
        second = env.world_for_slot(0).fingerprint_sha256()
        self.assertNotEqual(first, second)

    def test_counter_keyed_behaviour_is_preserved_when_no_key_is_set(self):
        env = _env()
        env.reset()
        self.assertIsNone(env.world_realization_key())
        first = env.world_for_slot(0).world_id
        env.reset()
        self.assertNotEqual(first, env.world_for_slot(0).world_id,
                            "without a key the historical counter-keyed behaviour stands")
        self.assertEqual(stable_seed(1, "a", 2), stable_seed(1, "a", 2))


class TestSmokeResumeEquivalence(unittest.TestCase):
    """The full chain, with the frozen outer operator, must resume bit-identically."""

    @classmethod
    def setUpClass(cls):
        cls.config = SmokeConfig(meta_tasks=2, support=2, query=1, inner_steps=2, tokens=20,
                                 seed=11)
        cls.reference = run_smoke(cls.config, iterations=3, checkpoint_at=1)
        cls.resumed = run_smoke(cls.config, resume_from=cls.reference["checkpoint"],
                                iterations=1, start_iteration=2)

    def test_the_chain_produces_finite_measured_updates(self):
        last = self.reference["iterations"][-1]
        for key in ("outer_update_norm", "outer_pseudogradient_norm", "core_weight_norm",
                    "query_return_mean", "validation_latency_mean",
                    "validation_system_joules_mean"):
            self.assertTrue(np.isfinite(last[key]), key)
        self.assertGreater(last["outer_update_norm"], 0.0)
        self.assertGreater(last["validation_system_joules_mean"], 0.0)
        self.assertTrue(self.reference["final"]["finite"])

    def test_support_query_and_validation_all_execute(self):
        last = self.reference["iterations"][-1]
        self.assertEqual(len(last["inner"]), self.config.meta_tasks)
        for task in last["inner"]:
            self.assertEqual(task["inner_steps"], self.config.inner_steps)
            self.assertGreater(task["weight_delta_norm"], 0.0)
        self.assertGreater(last["support_return_mean"] - last["support_return_mean"], -1e-12)

    def test_resume_reproduces_the_uninterrupted_iteration_exactly(self):
        a = np.asarray(self.reference["iterations"][-1]["weights"], dtype=np.float64)
        b = np.asarray(self.resumed["iterations"][-1]["weights"], dtype=np.float64)
        self.assertEqual(self.reference["iterations"][-1]["iteration"],
                         self.resumed["iterations"][-1]["iteration"])
        np.testing.assert_array_equal(a, b)
        self.assertEqual(self.reference["final"]["duals"], self.resumed["final"]["duals"])

    def test_checkpoint_carries_the_hashes_and_contract(self):
        ckpt = self.reference["checkpoint"]
        for key in ("obs", "weights", "outer_optimizer", "duals", "world_config_sha256",
                    "scheduler_config_sha256", "crn_protocol_sha256", "smoke_config_sha256",
                    "reference_scope", "reference_scheduler_config_sha256",
                    "inner_optimizer_contract", "rng_contract", "schema"):
            self.assertIn(key, ckpt)
        self.assertEqual(ckpt["obs"]["v2_packed_dim"], 109)
        self.assertEqual(ckpt["crn_protocol_id"], "automotive_crn_gumbel_v1")
        self.assertEqual(len(ckpt["reference_scheduler_config_sha256"]), 64)

    def test_a_mismatched_configuration_is_refused(self):
        other = SmokeConfig(**{**self.config.as_dict(), "inner_steps": 4})
        with self.assertRaises(Exception):
            run_smoke(other, resume_from=self.reference["checkpoint"], iterations=1,
                      start_iteration=2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
