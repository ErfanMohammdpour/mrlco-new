#!/usr/bin/env python3
"""v2 observation contract (audited items 6.1/6.2/6.3/6.4/6.6, CPU-accessible part).

Covered here:
* the width is DERIVED from one shared field list, used by the environment AND the frozen
  encoder (no duplicated names, no hardcoded 91/52);
* perturbing a world-level feature propagates into the policy input tensor at the declared
  column (the end-to-end perturbation test; the TF-encoder gradient check needs TensorFlow
  and is reported NOT RUN);
* two slots holding the SAME dataset graph under DIFFERENT contact conditions, and graphs at
  a NONZERO dataset index, each receive their own correct context;
* per-NODE epsilons, estimate age, decision-time waits/energies, the budget ratio and the
  broadcast duals are present and finite;
* the new channels are decision-time only: a different realized link process does not change
  them.
"""

from __future__ import annotations

import dataclasses
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spec.automotive_training.automotive_loader import load_dataset  # noqa: E402
from spec.automotive_training.automotive_primary import AutomotiveResourceCluster  # noqa: E402
from spec.automotive_training.v2.context_fields import (  # noqa: E402
    V2_CONTEXT_FIELDS, v2_context_feature_names, v2_context_dim,
)
from spec.automotive_training.v2.env import V2AutomotiveEnv  # noqa: E402
from spec.automotive_training.v2.observation import (  # noqa: E402
    V1_FEATURE_DIM, V1_PACKED_DIM, V2_CONTEXT_DIM, V2_FEATURE_DIM, V2_PACKED_DIM,
    pack_v2_row, split_v2_row,
)
from spec.automotive_training.v2.world import V2WorldConfig, build_world  # noqa: E402

SLOTS = 2
TOKENS = 20


def _env(indices=None, seed=909, constraints=False, background=0):
    """`single_dist=True` layout: `graph_indices[slot]` is the DATASET graph of that slot, so
    the array length is the number of slots and may repeat values or use nonzero indices."""
    idx = (np.arange(SLOTS, dtype=np.int32) if indices is None
           else np.asarray(indices, dtype=np.int32))
    need = max(int(idx.max()) + 1 if idx.size else 1, int(idx.size), 1)
    graphs = load_dataset().validation_query()[:need]
    env = V2AutomotiveEnv(graphs, AutomotiveResourceCluster(), role="validation",
                          slots_per_task=int(idx.size), base_seed=seed, single_dist=True,
                          background_dags=int(background), constraints_enabled=constraints,
                          budget_fractions={"total_energy": 2.0},
                          scheduler_config_sha256="c" * 64)
    env.set_task({"dist_index": 0, "graph_indices": idx})
    return env


#: 8 slots, dataset graph 5 in BOTH slot 0 and slot 1, then a permutation
REPEATED = [5, 5, 0, 1, 2, 3, 4, 6]
NONZERO = [5, 7, 0, 1, 2, 3, 4, 6]


class TestSingleContract(unittest.TestCase):
    def test_dimensions_are_derived_from_one_field_list(self):
        self.assertEqual(V2_CONTEXT_DIM, len(V2_CONTEXT_FIELDS))
        self.assertEqual(V2_FEATURE_DIM, V1_FEATURE_DIM + len(V2_CONTEXT_FIELDS))
        self.assertEqual(V2_PACKED_DIM, V1_PACKED_DIM + len(V2_CONTEXT_FIELDS))
        self.assertEqual(len(v2_context_feature_names()), v2_context_dim())
        self.assertEqual(len(set(V2_CONTEXT_FIELDS)), len(V2_CONTEXT_FIELDS))

    def test_the_encoder_reads_the_same_list(self):
        from env.mec_offloaing_envs.scheduler import encoder_obs

        self.assertTrue(encoder_obs.V2_CONTEXT_CONTRACT_AVAILABLE)
        self.assertEqual(len(encoder_obs.V2_CONTEXT_FEATURE_NAMES), len(V2_CONTEXT_FIELDS))
        self.assertEqual(
            tuple(encoder_obs.V2_CONTEXT_FEATURE_NAMES),
            tuple(v2_context_feature_names()))
        old = encoder_obs.OBS_VERSION
        try:
            encoder_obs.set_obs_version("automotive_v2_obs_v1")
            self.assertEqual(encoder_obs.FEATURE_DIM, V2_FEATURE_DIM)
            self.assertEqual(encoder_obs.PACKED_DIM, V2_PACKED_DIM)
        finally:
            encoder_obs.set_obs_version(old)

    def test_every_context_entry_is_observable_by_name(self):
        for field in V2_CONTEXT_FIELDS:
            self.assertTrue(field.startswith(("est_", "conf_", "helper_", "epsilon",
                                              "criticality", "mec_", "energy_", "lambda_",
                                              "contact_", "background_")),
                            "unexpected context field %r" % field)
        for forbidden in ("realized", "truth", "future", "oracle"):
            self.assertFalse(any(forbidden in f for f in V2_CONTEXT_FIELDS))


class TestPerturbationReachesTheTensor(unittest.TestCase):
    def _row_and_column(self, env, slot, column_name):
        ctx = env._context_vector(slot)
        col = list(V2_CONTEXT_FIELDS).index(column_name)
        rows = np.asarray(env._slice_current(env.base.encoder_batchs), dtype=np.float32)
        n_rows = int(rows.shape[0])
        v2_rows = np.stack([pack_v2_row(rows[r], ctx) for r in range(n_rows)])
        return v2_rows, V1_FEATURE_DIM + col

    def test_background_load_perturbation_appears_in_the_declared_column(self):
        quiet = _env(background=0)
        quiet.reset()
        crowd = _env(background=4)
        crowd.reset()
        a, col = self._row_and_column(quiet, 0, "background_dags")
        b, col2 = self._row_and_column(crowd, 0, "background_dags")
        self.assertEqual(col, col2)
        self.assertEqual(float(a[0, 0, col]), 0.0)
        self.assertEqual(float(b[0, 0, col]), 4.0)
        # ONLY that column changes: the perturbation is localised and traceable
        diff = np.flatnonzero(np.any(np.abs(a[0] - b[0]) > 1e-6, axis=0))
        self.assertEqual(set(diff.tolist()), {col})

    def test_mec_worker_perturbation_appears_in_the_declared_column(self):
        one = _env()
        one.reset()
        two = V2AutomotiveEnv(list(one.base.graph_objects),
                              AutomotiveResourceCluster(), role="validation",
                              slots_per_task=SLOTS, base_seed=909, single_dist=True,
                              world_config=dataclasses.replace(one.world_config,
                                                               mec_workers=3))
        two.set_task({"dist_index": 0, "graph_indices": np.arange(SLOTS, dtype=np.int32)})
        two.reset()
        a, col = self._row_and_column(one, 0, "mec_workers")
        b, _ = self._row_and_column(two, 0, "mec_workers")
        self.assertEqual(float(a[0, 0, col]), 1.0)
        self.assertEqual(float(b[0, 0, col]), 3.0)

    def test_the_tensor_round_trips_and_keeps_the_v1_prefix(self):
        env = _env()
        env.reset()
        rows = np.asarray(env._slice_current(env.base.encoder_batchs), dtype=np.float32)
        ctx = env._context_vector(0)
        packed = pack_v2_row(rows[0], ctx)
        self.assertEqual(packed.shape, (rows.shape[1], V2_PACKED_DIM))
        features, back, tail = split_v2_row(packed)
        np.testing.assert_array_equal(features, rows[0][:, :V1_FEATURE_DIM])
        np.testing.assert_array_equal(tail, rows[0][:, V1_FEATURE_DIM:])
        np.testing.assert_allclose(back, np.tile(ctx, (rows.shape[1], 1)), rtol=1e-6)


class TestSlotIdentity(unittest.TestCase):
    def test_same_graph_in_two_slots_with_different_contact_gets_its_own_context(self):
        env = _env(indices=REPEATED)              # SAME dataset graph in slots 0 and 1
        env.reset()
        self.assertEqual(env.world_for_slot(0).dataset_graph_id, 5)
        self.assertEqual(env.world_for_slot(1).dataset_graph_id, 5)
        # give slot 1 a much shorter contact than slot 0
        short = dataclasses.replace(env.helper_states[1][0], contact_end_s=0.5,
                                    predicted_contact_end_s=0.75)
        env._worlds[1].helpers = {0: short}
        env.helper_states[1] = env._worlds[1].helpers
        c0 = env._context_vector(0)
        c1 = env._context_vector(1)
        col = list(V2_CONTEXT_FIELDS).index("helper_contact_remaining_s")
        self.assertNotEqual(float(c0[col]), float(c1[col]),
                            "two slots with the same graph but different contact conditions "
                            "must not share one context vector")
        self.assertGreater(float(c0[col]), float(c1[col]))

    def test_nonzero_dataset_graph_index_maps_to_its_own_slot(self):
        env = _env(indices=NONZERO)
        env.reset()
        for slot in range(len(NONZERO)):
            world = env.world_for_slot(slot)
            self.assertEqual(world.slot_id, slot)
            self.assertEqual(world.dataset_graph_id, NONZERO[slot])
            self.assertEqual(env.base._graph_index(slot), NONZERO[slot])
            self.assertEqual(int(env.base.graph_indices[slot]), NONZERO[slot])
        self.assertEqual(len({env.world_for_slot(s).world_id
                              for s in range(len(NONZERO))}), len(NONZERO))

    def test_repeated_graph_slots_are_distinct_worlds(self):
        env = _env(indices=REPEATED)
        env.reset()
        a, b = env.world_for_slot(0), env.world_for_slot(1)
        self.assertEqual(a.dataset_graph_id, b.dataset_graph_id)
        self.assertNotEqual(a.world_id, b.world_id)
        self.assertNotEqual(a.fingerprint_sha256(), b.fingerprint_sha256())


class TestNewChannels(unittest.TestCase):
    def test_per_node_epsilon_channels_reflect_every_task(self):
        graphs = load_dataset().validation_query()[:6]
        mixed = None
        for g in graphs:
            classes = {str(t.criticality).upper() for t in g.tasks}
            if len(classes) > 1:
                mixed = g
                break
        self.assertIsNotNone(mixed, "no mixed-criticality graph in the validation query")
        env = _env(indices=[0, 0])
        env.reset()
        world = build_world(mixed, slot_id=0, config=V2WorldConfig(background_dags=0),
                            helper_seed=1)
        env._worlds[0] = world
        ctx = env._context_vector(0)
        lo = ctx[list(V2_CONTEXT_FIELDS).index("epsilon_node_min")]
        mid = ctx[list(V2_CONTEXT_FIELDS).index("epsilon_node_mean")]
        hi = ctx[list(V2_CONTEXT_FIELDS).index("epsilon_node_max")]
        self.assertLessEqual(lo, mid)
        self.assertLessEqual(mid, hi)
        self.assertLess(lo, hi, "a mixed-criticality graph must spread the node epsilons")

    def test_estimate_age_queue_energy_budget_and_lambda_channels_are_present(self):
        env = _env(constraints=True)
        env.reset()
        ctx = env._context_vector(0)
        self.assertEqual(ctx.shape[0], len(V2_CONTEXT_FIELDS))
        self.assertTrue(np.all(np.isfinite(ctx)))
        for name in ("est_age_ul_s", "est_wait_mec_s", "est_energy_mec_j",
                     "energy_budget_ratio", "lambda_total_energy", "contact_slack"):
            value = float(ctx[list(V2_CONTEXT_FIELDS).index(name)])
            self.assertGreaterEqual(value, 0.0, name)
        # the energy channels must be REAL plan energies, not zeros
        self.assertGreater(float(ctx[list(V2_CONTEXT_FIELDS).index("est_energy_mec_j")]), 0.0)
        self.assertGreater(float(ctx[list(V2_CONTEXT_FIELDS).index("est_energy_ue_j")]), 0.0)
        self.assertGreater(float(ctx[list(V2_CONTEXT_FIELDS).index("energy_budget_ratio")]), 0.0)

    def test_lambda_channel_follows_the_broadcast(self):
        env = _env(constraints=True)
        env.reset()
        col = list(V2_CONTEXT_FIELDS).index("lambda_total_energy")
        before = float(env._context_vector(0)[col])
        env.step(np.ones((SLOTS, TOKENS), dtype=int))
        env.constraint_manager.dual_step()
        env.last_v2_context = env._contexts()
        after = float(env._context_vector(0)[col])
        self.assertEqual(before, 0.0)
        self.assertGreater(after, 0.0, "the policy must be able to see the active lambda")

    def test_busy_seconds_and_busy_fraction_are_distinct_channels(self):
        env = _env()
        env.reset()
        env._worlds[0].helpers = {
            0: dataclasses.replace(env.helper_states[0][0], busy_until_s=1.0,
                                   contact_end_s=4.0, predicted_contact_end_s=4.0)}
        env.helper_states[0] = env._worlds[0].helpers
        ctx = env._context_vector(0)
        busy_s = float(ctx[list(V2_CONTEXT_FIELDS).index("helper_busy_s")])
        fraction = env.helper_busy_fraction(0)
        self.assertEqual(busy_s, 1.0)
        self.assertAlmostEqual(fraction, 0.25, places=6)

    def test_the_new_channels_do_not_depend_on_the_realized_link_process(self):
        env = _env()
        env.reset()
        base = env._context_vector(0)
        from spec.automotive_training.v2.link_model import make_process

        env.link_process = make_process("degraded", 12345)
        env.last_v2_context = env._contexts()
        other = env._context_vector(0)
        # the reference waits/energies are pure plan-time quantities and must be identical
        for name in ("est_wait_mec_s", "est_energy_ue_j", "est_energy_mec_j",
                     "est_energy_helper_j", "energy_budget_ratio"):
            i = list(V2_CONTEXT_FIELDS).index(name)
            self.assertAlmostEqual(float(base[i]), float(other[i]), places=6, msg=name)


if __name__ == "__main__":
    unittest.main(verbosity=2)
