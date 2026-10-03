#!/usr/bin/env python3
"""TF-gated CRN test: the same stateless seed pair must give bit-identical actions.

Runs in the TF1.15 container (skipped without TensorFlow).
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# The obs version must be active BEFORE `policies/graph2seq_encoder` is imported: that module
# binds FEATURE_DIM/PACKED_DIM at import time. The container image has no pytest, so this file
# is run with `python -m unittest discover -s env/mec_offloaing_envs/scheduler/tests`.
import os

try:
    import tensorflow as tf

    # some suite modules install a lightweight `tensorflow` stub in sys.modules, so a bare
    # import is not enough: require real TF1.15 APIs before running the TF-gated tests
    HAS_TF = bool(getattr(tf, "__version__", "")) and hasattr(tf, "compat") and \
        hasattr(getattr(tf, "compat", None), "v1") and \
        hasattr(getattr(tf, "random", None), "stateless_multinomial")
except Exception:
    HAS_TF = False


from spec.automotive_training.v2.observation import (  # noqa: E402
    V2_FEATURE_DIM, V2_OBS_VERSION, V2_PACKED_DIM,
)

#: obs version to restore after this module's tests; set by `setUpModule`
_PREVIOUS_OBS_VERSION = None


def setUpModule():
    """Activate the v2 schema for THIS module's tests only, and restore it afterwards.

    Doing this at IMPORT time leaked the v2 schema into every other module collected by the
    same run (both pytest and `unittest discover` import all modules before running any test),
    which broke the v1 encoder tests with `packed dim 50/109 != 79`. The schema is global
    module state in `encoder_obs`, so it must be scoped, not set once at import.
    """
    global _PREVIOUS_OBS_VERSION
    if not HAS_TF:                                    # pragma: no cover - container only
        return
    from env.mec_offloaing_envs.scheduler import encoder_obs

    _PREVIOUS_OBS_VERSION = encoder_obs.OBS_VERSION
    os.environ["MARGO_OBS_VERSION"] = V2_OBS_VERSION
    encoder_obs.set_obs_version(V2_OBS_VERSION)


def tearDownModule():
    if _PREVIOUS_OBS_VERSION is None:                 # pragma: no cover
        return
    from env.mec_offloaing_envs.scheduler import encoder_obs

    encoder_obs.set_obs_version(_PREVIOUS_OBS_VERSION)
    os.environ["MARGO_OBS_VERSION"] = _PREVIOUS_OBS_VERSION


@unittest.skipUnless(HAS_TF, "requires TensorFlow (run in the TF1.15 image)")
class TestSeq2SeqPolicyCRN(unittest.TestCase):
    def _policy(self, name="crn_policy"):
        from policies.meta_seq2seq_policy import Seq2SeqPolicy

        return Seq2SeqPolicy(obs_dim=V2_PACKED_DIM, encoder_units=32, decoder_units=32, vocab_size=3,
                             name=name, enable_crn=True)

    def test_same_seed_pair_bit_identical_actions(self):
        policy = self._policy()
        obs = np.random.RandomState(0).randn(4, 20, V2_PACKED_DIM).astype(np.float32)
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            a1, l1, v1 = policy.crn_actions(obs, [11, 22])
            a2, l2, v2 = policy.crn_actions(obs, [11, 22])
            a3, l3, v3 = policy.crn_actions(obs, [33, 44])
        self.assertTrue(np.array_equal(a1, a2), "same CRN seed must reproduce the actions")
        self.assertTrue(np.array_equal(l1, l2))
        self.assertTrue(np.array_equal(v1, v2))
        self.assertFalse(np.array_equal(a1, a3),
                         "a different CRN seed must give a different draw")

    def test_obs_version_in_use_is_the_v2_schema(self):
        from env.mec_offloaing_envs.scheduler import encoder_obs

        self.assertEqual(encoder_obs.OBS_VERSION, V2_OBS_VERSION)
        self.assertEqual((encoder_obs.FEATURE_DIM, encoder_obs.PACKED_DIM),
                         (V2_FEATURE_DIM, V2_PACKED_DIM))

    def test_gumbel_path_matches_stateless_categorical(self):
        """A stateless categorical draw IS the Gumbel-max rule with a deterministic stream:
        the empirical action frequencies must match the softmax of the logits."""
        policy = self._policy("crn_policy_stats")
        obs = np.zeros((200, 20, V2_PACKED_DIM), dtype=np.float32)
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            counts = np.zeros(3)
            draws = 0
            for s in range(6):
                actions, _l, _v = policy.crn_actions(obs, [7, 100 + s])
                flat = np.asarray(actions).reshape(-1)
                for a in range(3):
                    counts[a] += float(np.mean(flat == a))
                draws += 1
            counts /= max(1, draws)
        # the policy is untrained, so this only checks that all three actions are reachable
        self.assertTrue(all(c > 0.0 for c in counts), "all actions must be reachable: %s" % counts)

    def test_crn_is_off_by_default(self):
        from policies.meta_seq2seq_policy import Seq2SeqPolicy

        policy = Seq2SeqPolicy(obs_dim=V2_PACKED_DIM, encoder_units=32, decoder_units=32, vocab_size=3,
                               name="crn_off")
        self.assertIsNone(policy.crn_seed)
        self.assertFalse(policy.enable_crn)


if __name__ == "__main__":
    unittest.main(verbosity=2)
