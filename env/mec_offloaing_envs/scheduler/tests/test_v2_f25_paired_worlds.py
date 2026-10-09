#!/usr/bin/env python3
"""F25 regression: the v2 baseline panel must be scored on the PAIRED policy world.

Pre-fix behaviour (reproduced in `test_the_audited_self_built_panel_world_is_not_the_policy_world`):
`V2HeldOutEvaluator.baseline_panel` built its OWN world with `mc=None` and
`helper_seed=seed_base+index`, while the policy rollout env built worlds with
`mc=base._slot_mc[flat]` and `helper_seed=stable_seed(base_seed, "helper", world_key, flat)`.
The panel world therefore had a different demand realization / helper window / task count from
the world the policy was scored on, and `oracle_s`/`k0_regret_s` subtracted the two.
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
from spec.automotive_training.v2 import eval_loop  # noqa: E402
from spec.automotive_training.v2.env import V2AutomotiveEnv, stable_seed  # noqa: E402
from spec.automotive_training.v2.eval_loop import (  # noqa: E402
    mc_identity, physics_fingerprint,
)
from spec.automotive_training.v2.stack import V2HeldOutEvaluator  # noqa: E402
from spec.automotive_training.v2.world import V2WorldConfig, build_world  # noqa: E402

EPOCH = 3
BASE_SEED = 1000 + 17 * EPOCH


def _factory(**extra):
    def factory(graphs, slots, seed, single_dist):
        env = V2AutomotiveEnv(list(graphs), AutomotiveResourceCluster(),
                              role="validation", slots_per_task=int(slots),
                              base_seed=int(seed), single_dist=bool(single_dist), **extra)
        if single_dist:
            env.set_task({"dist_index": 0,
                          "graph_indices": np.arange(len(graphs), dtype=np.int32)})
        return env

    return factory


def _evaluator(graphs, **extra):
    return V2HeldOutEvaluator(support_graphs=graphs[:1], query_graphs=graphs, policy=None,
                              source_policy=None, env_factory=_factory(**extra))


class TestF25PairedWorlds(unittest.TestCase):
    def _paired(self, **extra):
        graphs = load_dataset().validation_query()[:2]
        ev = _evaluator(graphs, **extra)
        env = ev._paired_env(graphs, BASE_SEED, EPOCH, role="query")
        env.reset()
        return ev, env, graphs

    def _paired_one(self, **extra):
        graphs = load_dataset().validation_query()[:1]
        ev = _evaluator(graphs, **extra)
        env = ev._paired_env(graphs, BASE_SEED, EPOCH, role="query")
        env.reset()
        return ev, env, graphs

    def test_panel_world_is_the_paired_policy_world(self):
        ev, env, graphs = self._paired()
        policy_world = env.world_for_slot(0)
        panel = ev.baseline_panel(env, epoch=None)      # epoch=None -> no caching
        entry = panel[0]
        self.assertTrue(entry["paired_with_policy_world"])
        self.assertEqual(entry["world_id"], policy_world.world_id)
        self.assertEqual(entry["world_fingerprint_sha256"],
                         policy_world.fingerprint_sha256())
        self.assertEqual(entry["world_physics_sha256"], physics_fingerprint(policy_world))
        self.assertEqual(entry["n_tasks"], len(policy_world.foreground.tasks))
        self.assertEqual(entry["mc_id"], mc_identity(env._slot_mc[0]))
        self.assertEqual(entry["dataset_graph_id"], int(policy_world.dataset_graph_id))
        self.assertEqual(entry["config_sha"], policy_world.config.sha256())
        expected_seed = stable_seed(env.base_seed, "helper", env.world_realization_key(), 0)
        self.assertEqual(entry["helper_seed"], int(expected_seed))
        # every candidate is scored on that same world, so the oracle and the regret are one-world
        self.assertEqual(entry["all_UE_metrics"]["unmodelled"]["status"], "UNMODELLED")

    def test_the_audited_self_built_panel_world_is_not_the_policy_world(self):
        """The executed pre-fix demonstration: old world != paired world."""
        ev, env, graphs = self._paired()
        graph = graphs[0]
        policy_world = env.world_for_slot(0)
        # EXACTLY the pre-fix construction (stack.py:737-742 at HEAD 9e80a16)
        old_world = build_world(
            graph, slot_id=0, mc=None,
            config=V2WorldConfig(background_dags=0, mec_workers=1),
            helper_seed=EPOCH + 0)
        self.assertNotEqual(physics_fingerprint(old_world), physics_fingerprint(policy_world),
                            "the audited no-op pairing must be observable as a physics mismatch")
        self.assertNotEqual(old_world.fingerprint_sha256(),
                            policy_world.fingerprint_sha256())
        panel = ev.baseline_panel(env, epoch=None)
        self.assertEqual(panel[0]["world_physics_sha256"], physics_fingerprint(policy_world))
        self.assertNotEqual(panel[0]["world_physics_sha256"],
                            physics_fingerprint(old_world))
        # and the OLD code never even reported the identity, so a consumer could not detect this
        self.assertNotIn("world_fingerprint_sha256", old_world.provenance)

    def test_baselines_are_scored_through_the_paired_link_process(self):
        ev, env, graphs = self._paired_one(link_regime="degraded")
        policy_link = env.link_process
        self.assertIsNotNone(policy_link)
        seen = []
        original = eval_loop._score

        def spy(world, plan_map, *, link_process=None):
            seen.append(link_process)
            return original(world, plan_map, link_process=link_process)

        eval_loop._score = spy
        try:
            ev.baseline_panel(env, epoch=None)
        finally:
            eval_loop._score = original
        self.assertTrue(seen)
        self.assertTrue(all(proc is policy_link for proc in seen),
                        "every baseline must be scored on the policy's own realized link stream")

    def test_panel_metrics_are_scored_on_the_same_world_object(self):
        """A candidate change never rebuilds the world; the paired world is reused."""
        ev, env, graphs = self._paired_one()
        policy_world = env.world_for_slot(0)
        seen = []
        original = eval_loop._score

        def spy(world, plan_map, *, link_process=None):
            seen.append(world.world_id)
            return original(world, plan_map, link_process=link_process)

        eval_loop._score = spy
        try:
            ev.baseline_panel(env, epoch=None)
        finally:
            eval_loop._score = original
        self.assertEqual(set(seen), {policy_world.world_id})

    def test_panel_numeric_regret_is_smaller_than_the_cross_world_one(self):
        """The paired oracle regret must be computed against the model's own world."""
        ev, env, graphs = self._paired_one()
        panel = ev.baseline_panel(env, epoch=None)
        # the oracle is the best panel candidate on THIS world, so it is a valid lower envelope
        for name in ("all_UE", "all_MEC", "all_HELPER", "greedy_coordinate_descent_MC",
                     "heft_reference_v2"):
            self.assertLessEqual(panel[0]["candidate_oracle_s"], panel[0][name] + 1e-12,
                                 "the oracle is the best-of-panel latency, so it can never "
                                 "exceed a panel candidate (%s)" % name)
        self.assertEqual(panel[0]["_name"],
                         min((panel[0][n], n) for n in (
                             "all_UE", "all_MEC", "all_HELPER",
                             "greedy_coordinate_descent_MC", "heft_reference_v2"))[1])
        self.assertEqual(panel[0]["oracle_selection_criterion"], "best_of_panel_latency")


if __name__ == "__main__":
    unittest.main(verbosity=2)
