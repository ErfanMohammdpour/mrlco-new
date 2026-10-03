#!/usr/bin/env python3
"""v2 observation extension: dims, prefix preservation, stats artifact."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spec.automotive_training.v2.env import V2_CONTEXT_FIELDS  # noqa: E402
from spec.automotive_training.v2.observation import (  # noqa: E402
    V1_FEATURE_DIM, V1_OBS_VERSION, V1_PACKED_DIM, V2_CONTEXT_DIM, V2_FEATURE_DIM,
    V2_OBS_VERSION, V2_PACKED_DIM, V2ObservationError, pack_v2_row, split_v2_row,
    v1_form_of_v2_row, validate_stats_file, v2_feature_names, write_v2_stats_file,
)


class TestSchema(unittest.TestCase):
    def test_dims_are_v1_plus_context(self):
        self.assertEqual(V2_CONTEXT_DIM, len(V2_CONTEXT_FIELDS))
        self.assertEqual(V2_FEATURE_DIM, V1_FEATURE_DIM + V2_CONTEXT_DIM)
        self.assertEqual(V2_PACKED_DIM, V1_PACKED_DIM + V2_CONTEXT_DIM)
        # dims are DERIVED from the declared field names, never hardcoded
        self.assertEqual(V2_FEATURE_DIM, V1_FEATURE_DIM + len(V2_CONTEXT_FIELDS))
        self.assertEqual(V2_PACKED_DIM, V1_PACKED_DIM + len(V2_CONTEXT_FIELDS))

    def test_feature_names_are_v1_prefix_then_context(self):
        names = v2_feature_names()
        self.assertEqual(len(names), V2_FEATURE_DIM)
        self.assertTrue(all(n.startswith(V2_OBS_VERSION + "_") for n in names[V1_FEATURE_DIM:]))
        self.assertEqual(names[V1_FEATURE_DIM:],
                         tuple("%s_%s" % (V2_OBS_VERSION, f) for f in V2_CONTEXT_FIELDS))

    def test_v1_schema_is_untouched(self):
        from env.mec_offloaing_envs.scheduler import encoder_obs

        previous = encoder_obs.OBS_VERSION
        try:
            encoder_obs.set_obs_version(V1_OBS_VERSION)
            self.assertEqual(encoder_obs.FEATURE_DIM, V1_FEATURE_DIM)
            self.assertEqual(encoder_obs.PACKED_DIM, V1_PACKED_DIM)
        finally:
            encoder_obs.set_obs_version(previous)

    def test_v2_version_is_wired_and_equals_v1_plus_context(self):
        import numpy as np
        from env.mec_offloaing_envs.scheduler import encoder_obs
        from spec.automotive_training.automotive_dag import canonical_dag, decoder_order
        from spec.automotive_training.automotive_env import AutomotiveEnv
        from spec.automotive_training.automotive_loader import load_dataset
        from spec.automotive_training.automotive_primary import AutomotiveResourceCluster

        graphs = load_dataset().validation_query()[:1]
        env = AutomotiveEnv(graphs, AutomotiveResourceCluster(), role="validation",
                            slots_per_task=1, base_seed=303, single_dist=True)
        rec = graphs[0].as_record()
        dag, order = canonical_dag(rec), decoder_order(rec)
        ctx = np.linspace(0.05, 0.95, V2_CONTEXT_DIM)
        previous = encoder_obs.OBS_VERSION
        try:
            encoder_obs.set_obs_version(V1_OBS_VERSION)
            self.assertEqual((encoder_obs.FEATURE_DIM, encoder_obs.PACKED_DIM),
                             (V1_FEATURE_DIM, V1_PACKED_DIM))
            v1 = np.asarray(encoder_obs.encode_canonical_dag(
                dag, order, stats=None, resources=env.configs[0],
                mc_context=env._mc_context(graphs[0])), dtype=np.float64)
            encoder_obs.set_obs_version(V2_OBS_VERSION)
            self.assertEqual((encoder_obs.FEATURE_DIM, encoder_obs.PACKED_DIM),
                             (V2_FEATURE_DIM, V2_PACKED_DIM))
            mc = dict(env._mc_context(graphs[0]) or {})
            mc["v2_context"] = ctx
            v2 = np.asarray(encoder_obs.encode_canonical_dag(
                dag, order, stats=None, resources=env.configs[0], mc_context=mc),
                dtype=np.float64)
        finally:
            encoder_obs.set_obs_version(previous)
        expected = pack_v2_row(v1.astype(np.float32), ctx.astype(np.float32)).astype(np.float64)
        self.assertTrue(np.allclose(v2, expected, atol=1e-6),
                        "v2 obs must equal v1 obs with the context block inserted")
        self.assertTrue(np.allclose(v1_form_of_v2_row(v2.astype(np.float32)),
                                    v1.astype(np.float32), atol=1e-6))


class TestPacking(unittest.TestCase):
    def test_pack_inserts_context_and_preserves_v1_columns(self):
        rng = np.random.RandomState(0)
        row = rng.rand(20, V1_PACKED_DIM).astype(np.float32)
        ctx = np.arange(V2_CONTEXT_DIM, dtype=np.float32)
        v2 = pack_v2_row(row, ctx)
        self.assertEqual(v2.shape, (20, V2_PACKED_DIM))
        features, extra, tail = split_v2_row(v2)
        self.assertTrue(np.array_equal(features, row[:, :V1_FEATURE_DIM]))
        self.assertTrue(np.array_equal(tail, row[:, V1_FEATURE_DIM:]))
        for col in range(V2_CONTEXT_DIM):
            self.assertTrue(np.allclose(extra[:, col], ctx[col]))
        self.assertTrue(np.array_equal(v1_form_of_v2_row(v2), row))

    def test_shape_and_finiteness_guards(self):
        with self.assertRaises(V2ObservationError):
            pack_v2_row(np.zeros((20, V1_PACKED_DIM - 1), dtype=np.float32),
                        np.zeros(V2_CONTEXT_DIM, dtype=np.float32))
        with self.assertRaises(V2ObservationError):
            pack_v2_row(np.zeros((20, V1_PACKED_DIM), dtype=np.float32),
                        np.zeros(V2_CONTEXT_DIM - 1, dtype=np.float32))
        bad = np.zeros(V2_CONTEXT_DIM, dtype=np.float32)
        bad[0] = np.nan
        with self.assertRaises(V2ObservationError):
            pack_v2_row(np.zeros((20, V1_PACKED_DIM), dtype=np.float32), bad)

    def test_stats_artifact_matches_the_schema(self):
        path = write_v2_stats_file()
        info = validate_stats_file(path)
        self.assertEqual(info["feature_dim"], V2_FEATURE_DIM)
        doc = json.loads(Path(path).read_text())
        self.assertEqual(doc["mean"][V1_FEATURE_DIM:], [0.0] * V2_CONTEXT_DIM)
        self.assertEqual(doc["std"][V1_FEATURE_DIM:], [1.0] * V2_CONTEXT_DIM)
        self.assertEqual(doc["derived_from"], "encoder_feature_stats_automotive_mc_v1.json")
        self.assertEqual(len(set(doc["feature_names"])), V2_FEATURE_DIM)


if __name__ == "__main__":
    unittest.main(verbosity=2)
