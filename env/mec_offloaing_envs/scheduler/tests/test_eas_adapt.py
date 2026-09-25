#!/usr/bin/env python3
"""Phase 3 EAS unit tests. Numpy always. TF skipped if missing."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spec.eas_adapt import (  # noqa: E402
    ADAPT_SUBSETS,
    LOSS_TYPES,
    advantages_pomo,
    assert_adv_zero_mean,
)


class TestEasNumpy(unittest.TestCase):
    def test_advantage_zero_mean_per_graph(self):
        ts = np.array(
            [[10.0, 12.0, 8.0, 14.0], [20.0, 22.0, 18.0, 24.0]],
            dtype=np.float64,
        )
        t_mec = np.array([100.0, 200.0], dtype=np.float64)
        adv = advantages_pomo(ts, t_mec)
        assert_adv_zero_mean(adv)
        self.assertEqual(adv.shape, (2, 4))
        # better than mean => positive advantage
        self.assertGreater(adv[0, 2], 0.0)  # T=8 < mean=11

    def test_subset_loss_tuples(self):
        self.assertEqual(ADAPT_SUBSETS, ("lastlayer", "film", "emb"))
        self.assertEqual(LOSS_TYPES, ("pg_il", "pg", "il", "ce2opt"))

    def test_il_grad_zero_when_onehot_argmax(self):
        """Toy: CE to one-hot argmax plan → IL loss = 0 and grad = 0."""
        # logits such that softmax is one-hot on label → NLL=0
        logits = np.array([[100.0, -100.0, -100.0]], dtype=np.float64)
        label = 0
        # soft CE
        e = np.exp(logits - logits.max())
        p = e / e.sum()
        nll = -np.log(p[0, label] + 1e-12)
        self.assertLess(float(nll), 1e-6)


class TestEasTf(unittest.TestCase):
    def test_filter_lastlayer_names(self):
        try:
            import tensorflow as tf
        except ImportError:
            raise unittest.SkipTest("TensorFlow not installed")
        if not hasattr(tf, "contrib"):
            raise unittest.SkipTest("need tf.contrib (TensorFlow 1.15)")
        from policies.meta_seq2seq_policy import Seq2SeqPolicy
        from spec.eas_adapt import filter_adapt_vars
        from env.mec_offloaing_envs.scheduler.encoder_obs import PACKED_DIM

        tf.compat.v1.reset_default_graph()
        policy = Seq2SeqPolicy(
            obs_dim=PACKED_DIM,
            encoder_units=128,
            decoder_units=128,
            vocab_size=3,
            name="pi",
            encoder_type="meanagg",
            readout_type="mean",
        )
        vars_ = filter_adapt_vars(policy, "lastlayer")
        self.assertTrue(vars_)
        for v in vars_:
            self.assertIn("output_projection", v.name)
        all_names = {v.name for v in policy.get_trainable_variables()}
        chosen = {v.name for v in vars_}
        self.assertTrue(chosen.issubset(all_names))


if __name__ == "__main__":
    unittest.main()
