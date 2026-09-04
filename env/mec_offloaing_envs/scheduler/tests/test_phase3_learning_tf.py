#!/usr/bin/env python3
"""Phase 3 TensorFlow 1.15 learning-loop smoke.

Requires TensorFlow 1.x with tf.contrib. Missing TF is a failure, not a skip.
Does not train 3500 outer iterations. Does not use GPU.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ["CUDA_VISIBLE_DEVICES"] = ""


def _require_tf():
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError(
            "Phase 3 learning smoke requires TensorFlow 1.15; import failed"
        ) from exc
    if not hasattr(tf, "contrib"):
        raise RuntimeError("Phase 3 learning smoke requires tf.contrib (TensorFlow 1.15)")
    return tf


def _batch(rng, n=20, t=20, obs_dim=50, n_actions=3):
    observations = rng.randn(n, t, obs_dim).astype(np.float32)
    actions = rng.randint(0, n_actions, size=(n, t)).astype(np.int32)
    logits = rng.randn(n, t, n_actions).astype(np.float32)
    advantages = rng.randn(n, t).astype(np.float32)
    values = rng.randn(n, t).astype(np.float32)
    returns = rng.randn(n, t).astype(np.float32)
    return {
        "observations": observations,
        "actions": actions,
        "logits": logits,
        "advantages": advantages,
        "values": values,
        "returns": returns,
    }


class TestLearningTFSmoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tf = _require_tf()
        cls.tf = tf
        tfv1 = tf.compat.v1 if hasattr(tf, "compat") else tf
        cls.tfv1 = tfv1
        if hasattr(tfv1, "disable_eager_execution"):
            tfv1.disable_eager_execution()
        tfv1.reset_default_graph()
        if hasattr(tfv1, "set_random_seed"):
            tfv1.set_random_seed(0)
        np.random.seed(0)

        from env.mec_offloaing_envs.scheduler.encoder_obs import PACKED_DIM
        from meta_algos.MRLCO import MRLCO
        from meta_algos.ppo_offloading import PPO
        from policies.meta_seq2seq_policy import MetaSeq2SeqPolicy, Seq2SeqPolicy

        cls.obs_dim = int(PACKED_DIM)
        cls.meta_policy = MetaSeq2SeqPolicy(
            meta_batch_size=10,
            obs_dim=cls.obs_dim,
            encoder_units=32,
            decoder_units=32,
            vocab_size=3,
        )
        cls.mrlco = MRLCO(
            policy=cls.meta_policy,
            meta_batch_size=10,
            meta_sampler=None,
            meta_sampler_process=None,
            inner_lr=5e-4,
            outer_lr=5e-4,
            num_inner_grad_steps=3,
            clip_value=0.2,
            value_clip_epsilon=0.2,
            support_trajectories=20,
            ppo_batch_size_trajectories=20,
            rng=np.random.RandomState(0),
        )
        cls.ppo_policy = Seq2SeqPolicy(
            obs_dim=cls.obs_dim,
            encoder_units=32,
            decoder_units=32,
            vocab_size=3,
            name="phase3_ppo_smoke",
        )
        cls.ppo = PPO(
            policy=cls.ppo_policy,
            meta_sampler=None,
            meta_sampler_process=None,
            lr=5e-4,
            num_inner_grad_steps=3,
            clip_value=0.2,
            max_grad_norm=0.5,
            rng=np.random.RandomState(1),
        )
        cls.sess = tfv1.Session()
        cls._default = cls.sess.as_default()
        cls._default.__enter__()
        cls.sess.run(tfv1.global_variables_initializer())
        cls.mrlco.sync_task_policies_from_core()

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "_default", None) is not None:
            cls._default.__exit__(None, None, None)
        if getattr(cls, "sess", None) is not None:
            cls.sess.close()

    def test_tf_contrib_present(self):
        self.assertTrue(hasattr(self.tf, "contrib"))

    def test_sync_copies_core_to_task_slots(self):
        core = self.sess.run(self.meta_policy.core_policy.get_trainable_variables())
        task0 = self.sess.run(self.meta_policy.meta_policies[0].get_trainable_variables())
        for a, b in zip(core, task0):
            np.testing.assert_allclose(a, b)

    def test_inner_three_applies_then_one_outer(self):
        rng = np.random.RandomState(2)
        samples = [_batch(rng, obs_dim=self.obs_dim) for _ in range(10)]
        before_task = self.sess.run(self.meta_policy.meta_policies[0].get_trainable_variables())
        before_core = self.sess.run(self.meta_policy.core_policy.get_trainable_variables())
        policy_losses, value_losses = self.mrlco.UpdatePPOTarget(samples, batch_size=20)
        self.assertEqual(len(policy_losses), 10)
        self.assertEqual(len(policy_losses[0]), 3)
        self.assertEqual(len(value_losses[0]), 3)
        after_task = self.sess.run(self.meta_policy.meta_policies[0].get_trainable_variables())
        diverged = any(np.max(np.abs(a - b)) > 1e-8 for a, b in zip(before_task, after_task))
        self.assertTrue(diverged)
        self.mrlco.UpdateMetaPolicy()
        after_core = self.sess.run(self.meta_policy.core_policy.get_trainable_variables())
        core_moved = any(np.max(np.abs(a - b)) > 1e-12 for a, b in zip(before_core, after_core))
        self.assertTrue(core_moved)
        synced = self.sess.run(self.meta_policy.meta_policies[3].get_trainable_variables())
        for a, b in zip(after_core, synced):
            np.testing.assert_allclose(a, b, atol=1e-6)

    def test_wrong_inner_batch_is_rejected(self):
        rng = np.random.RandomState(3)
        samples = [_batch(rng, obs_dim=self.obs_dim) for _ in range(10)]
        with self.assertRaises(ValueError):
            self.mrlco.UpdatePPOTarget(samples, batch_size=10)

    def test_ppo_zero_shot_is_noop(self):
        rng = np.random.RandomState(4)
        data = _batch(rng, obs_dim=self.obs_dim)
        before = self.sess.run(self.ppo_policy.get_trainable_variables())
        policy_losses, value_losses = self.ppo.UpdatePPOTarget(data, batch_size=20, k_steps=0)
        self.assertEqual(policy_losses, [])
        self.assertEqual(value_losses, [])
        after = self.sess.run(self.ppo_policy.get_trainable_variables())
        for a, b in zip(before, after):
            np.testing.assert_allclose(a, b)

    def test_ppo_three_applies_and_fresh_adam(self):
        rng = np.random.RandomState(5)
        data = _batch(rng, obs_dim=self.obs_dim)
        policy_losses, value_losses = self.ppo.UpdatePPOTarget(data, batch_size=20, k_steps=3)
        self.assertEqual(len(policy_losses), 3)
        self.assertEqual(len(value_losses), 3)

    def test_value_clip_tf_matches_numpy(self):
        from spec.learning_ops import clipped_value_prediction

        v_old = np.array([[0.0, 1.0], [-1.0, 2.0]], dtype=np.float32)
        v_new = np.array([[5.0, 1.1], [-1.5, 1.9]], dtype=np.float32)
        old_ph = self.tfv1.placeholder(self.tf.float32, shape=[None, None])
        new_ph = self.tfv1.placeholder(self.tf.float32, shape=[None, None])
        clipped = old_ph + self.tf.clip_by_value(new_ph - old_ph, -0.2, 0.2)
        got = self.sess.run(clipped, feed_dict={old_ph: v_old, new_ph: v_new})
        np.testing.assert_allclose(got, clipped_value_prediction(v_old, v_new, 0.2), atol=1e-6)

    def test_assign_trainable_copies_core(self):
        from meta_algos.variable_io import assign_trainable

        assign_trainable(self.meta_policy.core_policy, self.ppo_policy, sess=self.sess)
        core = self.sess.run(self.meta_policy.core_policy.get_trainable_variables())
        dst = self.sess.run(self.ppo_policy.get_trainable_variables())
        self.assertEqual(len(core), len(dst))
        for a, b in zip(core, dst):
            np.testing.assert_allclose(a, b)


if __name__ == "__main__":
    unittest.main()
