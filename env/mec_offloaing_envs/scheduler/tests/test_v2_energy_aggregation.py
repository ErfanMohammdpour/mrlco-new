#!/usr/bin/env python3
"""F12: one episode's joules must be aggregated ONCE.

The frozen trainer sums the `energy` channel over the token axis. The V2 env fills every token
with the SAME complete-episode value, so that sum multiplied one episode by the token count
(measured 20x) while still being labelled as a mobile average. The env now declares
`energy_channel_semantics = "episode_replicated"` and the aggregation takes the value once.
"""
import sys
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env.mec_offloaing_envs.scheduler.energy_telemetry import (  # noqa: E402
    aggregate_episode_energy,
)


class EpisodeEnergyAggregationTests(unittest.TestCase):
    def test_legacy_per_token_semantics_is_unchanged(self):
        ch = np.full((4, 20), 0.118, dtype=np.float32)
        out = aggregate_episode_energy(ch, semantics="per_token")
        # the historical 20x behaviour, preserved exactly (float32 accumulation included)
        np.testing.assert_allclose(out, ch.astype(np.float64).sum(axis=-1), rtol=1e-12)
        np.testing.assert_allclose(out, np.full(4, 0.118 * 20), rtol=1e-5)

    def test_episode_replicated_counts_once(self):
        ch = np.full((4, 20), 0.118, dtype=np.float32)
        out = aggregate_episode_energy(ch, semantics="episode_replicated")
        np.testing.assert_allclose(out, np.full(4, 0.118), rtol=1e-6)
        self.assertAlmostEqual(float(out[0]) / float(ch.sum(axis=-1)[0]), 1.0 / 20.0, places=6)

    def test_unequal_horizons_do_not_scale_the_value(self):
        a = aggregate_episode_energy(np.full((2, 20), 3.0), semantics="episode_replicated")
        b = aggregate_episode_energy(np.full((2, 7), 3.0), semantics="episode_replicated")
        np.testing.assert_allclose(a, b)

    def test_lying_marker_fails_loudly(self):
        with self.assertRaises(ValueError):
            aggregate_episode_energy(np.tile(np.arange(20.0), (3, 1)),
                                     semantics="episode_replicated")

    def test_empty_channel(self):
        self.assertEqual(aggregate_episode_energy(np.zeros((0, 20)),
                                                  semantics="episode_replicated").size, 0)

    def test_v2_env_declares_the_marker(self):
        from spec.automotive_training.v2.env import V2AutomotiveEnv
        self.assertEqual(getattr(V2AutomotiveEnv, "energy_channel_semantics", None),
                         "episode_replicated")


if __name__ == "__main__":
    unittest.main(verbosity=2)
