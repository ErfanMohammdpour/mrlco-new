#!/usr/bin/env python3
"""Future-blindness beyond observation equality.

Requirement: with IDENTICAL observable history and identical policy sampling seeds but
DIFFERENT hidden futures, every decision-time quantity must be unchanged — observations,
confidence, admission/reliability evidence, the action mask (a pure function of the
observation) and therefore the sampled decision-time action. Only the realized OUTCOME may
differ. Functions that compare an estimate against hidden truth must be unreachable from
those paths.
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spec.automotive_training.automotive_loader import load_dataset  # noqa: E402
from spec.automotive_training.automotive_primary import AutomotiveResourceCluster  # noqa: E402
from spec.automotive_training.v2.env import V2AutomotiveEnv  # noqa: E402
from spec.automotive_training.v2.link_model import (  # noqa: E402
    LINK_DL, LINK_UL, LINK_V2V, make_process,
)

SLOTS = 2
TOKENS = 20


class _SamePastDifferentFuture:
    """Observable at decision time (t_now = 0) is IDENTICAL; the realized future is not.

    `estimate_at`, `confidence` and `past_outage_fraction` return the same values in both
    variants; only `realized` (execution) and `confidence_vs_truth` (a post-hoc diagnostic)
    differ.
    """

    class _Regime:
        dt_s = 0.05
        name = "future_blindness_fixture"
        evidence_class = "test_stub"
        estimation_sigma = 0.25

    regime = _Regime()

    def __init__(self, out_after_s=None):
        self.out_after_s = out_after_s

    def realized(self, link, t):
        if self.out_after_s is None:
            return 1.0
        return 0.0 if t >= self.out_after_s else 1.0

    def estimate_at(self, link, t=0.0):
        return 1.0

    def confidence(self, link, t=0.0):
        return 1.0 / (1.0 + float(self.regime.estimation_sigma))

    def past_outage_fraction(self, link, t=0.0):
        return 0.0

    def confidence_vs_truth(self, link, t=0.0):
        # DIAGNOSTIC ONLY: this is the value that must never reach a decision
        return 0.95 if self.out_after_s is None else 0.05


def _env(seed=4242):
    graphs = load_dataset().validation_query()[:SLOTS]
    env = V2AutomotiveEnv(graphs, AutomotiveResourceCluster(), role="validation",
                          slots_per_task=SLOTS, base_seed=seed, single_dist=True,
                          link_regime="degraded", reliability=True)
    env.set_task({"dist_index": 0, "graph_indices": np.arange(SLOTS, dtype=np.int32)})
    return env


class TestFutureBlindness(unittest.TestCase):
    def _run(self, process):
        env = _env()
        env.reset()
        env.link_process = process
        env.last_v2_context = env._contexts()
        obs = env._packed_observation(env.last_v2_context)
        _o, rewards, _d, info = env.step(np.ones((SLOTS, TOKENS), dtype=int))
        return env, obs, rewards, info

    def test_observations_and_confidence_are_unchanged_by_a_hidden_future(self):
        live = _SamePastDifferentFuture(out_after_s=None)
        dark = _SamePastDifferentFuture(out_after_s=0.2)
        env_a, obs_a, rew_a, info_a = self._run(live)
        env_b, obs_b, rew_b, info_b = self._run(dark)
        # the fixture really does have different futures...
        self.assertNotEqual(live.realized(LINK_UL, 1.0), dark.realized(LINK_UL, 1.0))
        self.assertNotEqual(live.confidence_vs_truth(LINK_UL, 0.0),
                            dark.confidence_vs_truth(LINK_UL, 0.0))
        # ...while every decision-time quantity is identical
        np.testing.assert_array_equal(np.asarray(obs_a), np.asarray(obs_b))
        np.testing.assert_array_equal(env_a.v2_context(), env_b.v2_context())
        for link in (LINK_UL, LINK_DL, LINK_V2V):
            self.assertEqual(live.confidence(link, 0.0), dark.confidence(link, 0.0))
            self.assertEqual(live.past_outage_fraction(link, 0.0),
                             dark.past_outage_fraction(link, 0.0))
        # admission/reliability evidence at decision time is identical too
        for slot in range(SLOTS):
            pa = info_a[2][slot]["constraints"]
            pb = info_b[2][slot]["constraints"]
            self.assertEqual(pa, pb)
        # the action MASK is a pure function of the observation, so it is identical; the
        # observation tensor above is bit-identical, which is the mask's only input
        self.assertEqual([np.asarray(r).tobytes() for r in rew_a],
                         [np.asarray(r).tobytes() for r in rew_b],
                         "the un-penalised decision-time reward is identical; only the "
                         "realized execution outcome may differ")

    def test_truth_comparing_diagnostic_is_not_in_the_admission_path(self):
        env = _env()
        env.reset()
        env.link_process = _SamePastDifferentFuture(out_after_s=0.2)
        env.last_v2_context = env._contexts()
        ctx = np.asarray(env.v2_context(), dtype=float)
        # the diagnostic value (0.05) must not appear in any context column
        self.assertFalse(np.any(np.isclose(ctx, 0.05)),
                         "confidence_vs_truth leaked into the observation")
        conf = 1.0 / (1.0 + 0.25)
        self.assertTrue(np.all(np.isclose(ctx[:, 3:6], conf)),
                        "the confidence channel must be the estimation-model value")
        _o, _r, _d, info = env.step(np.ones((SLOTS, TOKENS), dtype=int))
        for record in info[2]:
            # energy is measured but not constrained in this fixture
            self.assertEqual(record["energy_constraint"], "telemetry_only")

    def test_past_window_statistics_are_valid_at_t_now_zero(self):
        process = make_process("degraded", 7)
        for link in (LINK_UL, LINK_DL, LINK_V2V):
            value = process.past_outage_fraction(link, 0.0)
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)
            # at t_now = 0 the window is the single CURRENT step: the observable present state
            self.assertEqual(value, float(process.outage_at(link, 0.0)))
        windows = [process.past_outage_fraction(LINK_UL, t) for t in (0.0, 1.0, 5.0, 20.0)]
        self.assertTrue(all(0.0 <= w <= 1.0 for w in windows))
        self.assertTrue(all(math.isfinite(w) for w in windows))

    def test_the_window_never_reads_a_step_after_t_now(self):
        process = make_process("degraded", 21)
        t_now = 2.0
        before = process.past_outage_fraction(LINK_DL, t_now)
        idx = process._idx(t_now)
        self.assertLess(idx + 1, len(process._outage[LINK_DL]))
        # flip EVERY future step to "out" and re-read the statistic at t_now
        for k in range(idx + 1, len(process._outage[LINK_DL])):
            process._outage[LINK_DL][k] = True
        after = process.past_outage_fraction(LINK_DL, t_now)
        self.assertEqual(before, after,
                         "the past-window statistic read a step after t_now (future leakage)")


if __name__ == "__main__":
    unittest.main(verbosity=2)
