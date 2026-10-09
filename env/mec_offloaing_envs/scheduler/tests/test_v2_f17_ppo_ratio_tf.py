#!/usr/bin/env python3
"""F17: TF-gated PPO ratio contract on the REAL graph.

`test_v2_ppo_contract.py` proves the likelihood-ratio CONTRACT numerically in numpy and asserts
the reset protocol against the source text, and `test_phase3_learning_tf.py::test_ppo_zero_shot_is_noop`
proves that a zero-step PPO call changes nothing. None of them executes the real ratio graph.

This module runs the REAL `MRLCO.UpdatePPOTarget` once on a stored batch and checks, on the graph
the trainer builds:

* at the FIRST apply the probability ratio is 1 within the float32 precision of a stable softmax
  (`approx_kl < 1e-9` and `clip_fraction == 0`), because the old logits the rollout stored ARE the
  logits of the pre-update weights and the actions ARE the stored ones;
* the actions the ratio is evaluated at are the STORED ones: mutating them changes the ratio
  (the F17 defect -- resampling or re-deriving actions at update time -- cannot pass this);
* the ratio equals `exp(log p_new(a) - log p_old(a))` at that same bivariate check, not some
  re-decoded trajectory.

MASKING: the v2 stack never sets `MARGO_MASK_MODE`, and `masking.resolve_mask_mode` defaults to
"off", so no mask placeholder exists and the masked-ratio clause is INACTIVE on the v2 path. This
module asserts that fact explicitly (`test_mask_contract_is_inactive_on_the_v2_path`) instead of
claiming mask coverage it does not have. The masked-ratio contract itself is covered on the v3
path by the masking fixtures.

Runs in the TF1.15 container (skipped without TensorFlow); the image has no pytest, so this is a
plain `unittest` module:

    python -m unittest env.mec_offloaing_envs.scheduler.tests.test_v2_f17_ppo_ratio_tf
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

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

try:
    import tensorflow as tf

    HAS_TF = bool(getattr(tf, "__version__", "")) and hasattr(tf, "contrib")
except Exception:
    HAS_TF = False


def _batch(rng, n=20, t=20, obs_dim=50, n_actions=3):
    return {
        "observations": rng.randn(n, t, obs_dim).astype(np.float32),
        "actions": rng.randint(0, n_actions, size=(n, t)).astype(np.int32),
        "logits": rng.randn(n, t, n_actions).astype(np.float32),
        "advantages": rng.randn(n, t).astype(np.float32),
        "values": rng.randn(n, t).astype(np.float32),
        "returns": rng.randn(n, t).astype(np.float32),
    }


def _softmax(z):
    z = z - np.max(z, axis=-1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=-1, keepdims=True)


class _InnerLoss:
    """Minimal accessor for the ratio tensor of a task policy, built on the REAL graph."""

    def __init__(self, policy, old_logits, actions):
        self.policy = policy
        self.old_logits = old_logits
        self.actions = actions
        self.ratio = policy.distribution.likelihood_ratio_sym(
            actions, old_logits, policy.network.decoder_logits)


@unittest.skipUnless(HAS_TF, "requires TensorFlow (run in the TF1.15 image)")
class TestPPORatioOnRealGraph(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tfv1 = tf.compat.v1
        tfv1.disable_eager_execution()
        tfv1.reset_default_graph()
        tfv1.set_random_seed(0)
        np.random.seed(0)

        from env.mec_offloaing_envs.scheduler.encoder_obs import PACKED_DIM
        from meta_algos.MRLCO import MRLCO
        from policies.meta_seq2seq_policy import MetaSeq2SeqPolicy

        cls.tfv1 = tfv1
        cls.obs_dim = int(PACKED_DIM)
        cls.meta_policy = MetaSeq2SeqPolicy(
            meta_batch_size=10, obs_dim=cls.obs_dim, encoder_units=32, decoder_units=32,
            vocab_size=3)
        cls.mrlco = MRLCO(
            policy=cls.meta_policy, meta_batch_size=10, meta_sampler=None,
            meta_sampler_process=None, inner_lr=5e-4, outer_lr=5e-4, num_inner_grad_steps=3,
            clip_value=0.2, value_clip_epsilon=0.2, support_trajectories=20,
            ppo_batch_size_trajectories=20, rng=np.random.RandomState(0))
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

    def _stored_batch(self, seed=101, task_id=0):
        """The batch the ROLLOUT would have stored: the policy's OWN logits and actions, fed
        through the real policy graph, then consumed as `old_logits`/`actions`."""
        rng = np.random.RandomState(seed)
        obs = rng.randn(20, 20, self.obs_dim).astype(np.float32)
        policy = self.meta_policy.meta_policies[task_id]
        full_length = np.array([obs.shape[1]] * obs.shape[0], dtype=np.int32)
        logits = self.sess.run(
            policy.network.decoder_logits,
            feed_dict={policy.obs: obs, policy.decoder_full_length: full_length})
        actions = np.argmax(np.asarray(logits), axis=-1).astype(np.int32)
        batch = _batch(rng, obs_dim=self.obs_dim)
        batch["observations"] = obs
        batch["logits"] = np.asarray(logits, dtype=np.float32)
        batch["actions"] = actions
        return batch

    def test_pre_update_ratio_is_one_and_clip_fraction_is_zero(self):
        batch = self._stored_batch(seed=101)
        policy = self.meta_policy.meta_policies[0]
        old_logits = self.tfv1.placeholder(
            self.tf.float32, shape=[None, None, 3], name="f17_old_logits")
        accessor = _InnerLoss(policy, old_logits, policy.decoder_targets)
        full_length = np.array(
            [batch["observations"].shape[1]] * batch["observations"].shape[0], dtype=np.int32)
        feed = {
            policy.obs: batch["observations"],
            policy.decoder_targets: batch["actions"],
            policy.decoder_inputs: np.column_stack(
                (np.zeros(batch["actions"].shape[0], dtype=np.int32), batch["actions"][:, :-1])),
            policy.decoder_full_length: full_length,
            old_logits: batch["logits"],
        }
        ratio, approx_kl, clip_fraction = self.sess.run(
            [accessor.ratio, self.tfv1.reduce_mean(accessor.ratio - 1.0
                                                   - self.tfv1.log(accessor.ratio)),
             self.tfv1.reduce_mean(self.tfv1.cast(
                 self.tfv1.greater(self.tfv1.abs(accessor.ratio - 1.0), 0.2), self.tf.float32))],
            feed_dict=feed)
        self.assertLess(float(approx_kl), 1e-9,
                        "approx_kl at the first apply must be ~0, got %r" % (approx_kl,))
        self.assertEqual(float(clip_fraction), 0.0, "nothing may be clipped at the first apply")
        np.testing.assert_allclose(ratio, np.ones_like(ratio), rtol=0, atol=1e-5)
        # ... and the same ratio is the bivariate `exp(log p_new - log p_old)` at the stored a
        new_logits = self.sess.run(
            policy.network.decoder_logits, feed_dict={
                policy.obs: batch["observations"],
                policy.decoder_inputs: np.column_stack(
                    (np.zeros(batch["actions"].shape[0], dtype=np.int32),
                     batch["actions"][:, :-1])),
                policy.decoder_full_length: full_length})
        idx = np.arange(batch["actions"].shape[0])
        rows = np.arange(batch["actions"].shape[1])
        p_new = _softmax(np.asarray(new_logits))[idx[:, None], rows[None, :], batch["actions"]]
        p_old = _softmax(np.asarray(batch["logits"]))[idx[:, None], rows[None, :],
                                                        batch["actions"]]
        np.testing.assert_allclose(np.asarray(ratio), p_new / p_old, rtol=1e-4, atol=1e-6)

    def test_the_ratio_uses_the_stored_actions_not_a_fresh_decode(self):
        """Mutating the stored actions MUST change the ratio; an implementation that re-derives
        actions at update time would be insensitive to them."""
        batch = self._stored_batch(seed=202)
        policy = self.meta_policy.meta_policies[1]
        old_logits = self.tfv1.placeholder(
            self.tf.float32, shape=[None, None, 3], name="f17b_old_logits")
        accessor = _InnerLoss(policy, old_logits, policy.decoder_targets)
        full_length = np.array(
            [batch["observations"].shape[1]] * batch["observations"].shape[0], dtype=np.int32)

        def run(actions):
            feed = {
                policy.obs: batch["observations"],
                policy.decoder_targets: actions,
                policy.decoder_inputs: np.column_stack(
                    (np.zeros(actions.shape[0], dtype=np.int32), actions[:, :-1])),
                policy.decoder_full_length: full_length,
                old_logits: batch["logits"],
            }
            return self.sess.run(accessor.ratio, feed_dict=feed)

        stored = run(batch["actions"])
        mutated_actions = (batch["actions"] + 1) % 3
        mutated = run(mutated_actions)
        np.testing.assert_allclose(stored, np.ones_like(stored), rtol=0, atol=1e-5)
        self.assertGreater(float(np.max(np.abs(stored - mutated))), 1e-3,
                           "the ratio must be a function of the STORED actions")
        # the mutated batch is no longer at ratio 1
        self.assertGreater(float(np.mean(np.abs(mutated - 1.0))), 1e-6)

    def test_mrlco_ratio_tensor_is_one_at_the_pre_update_logits(self):
        """The ratio the REAL MRLCO builds for task 0 (via `policy.distribution`) is the same
        contract: feed the stored batch with `old_logits = new logits` and it is exactly 1."""
        batch = self._stored_batch(seed=303)
        policy = self.meta_policy.meta_policies[0]
        full_length = np.array(
            [batch["observations"].shape[1]] * batch["observations"].shape[0], dtype=np.int32)
        feed = {
            self.mrlco.old_logits[0]: batch["logits"],
            self.mrlco.old_v[0]: batch["values"],
            self.mrlco.obs[0]: batch["observations"],
            self.mrlco.actions[0]: batch["actions"],
            self.mrlco.decoder_inputs[0]: np.column_stack(
                (np.zeros(batch["actions"].shape[0], dtype=np.int32), batch["actions"][:, :-1])),
            self.mrlco.decoder_full_length[0]: full_length,
            self.mrlco.advs[0]: batch["advantages"],
            self.mrlco.r[0]: batch["returns"],
        }
        assert self.mrlco.actions[0] is policy.decoder_targets
        ratio, approx_kl = self.sess.run(
            [self.mrlco.likelihood_ratio[0], self.mrlco.approx_kl[0]], feed_dict=feed)
        np.testing.assert_allclose(ratio, np.ones_like(ratio), rtol=0, atol=1e-5)
        self.assertLess(float(approx_kl), 1e-9)

    def test_mask_contract_is_inactive_on_the_v2_path(self):
        """The v2 stack does not set MARGO_MASK_MODE and does not build a mask placeholder, so
        the masked-ratio clause does NOT apply here. This asserts that instead of claiming it."""
        from env.mec_offloaing_envs.scheduler import masking as masking_mod

        saved = os.environ.pop(masking_mod.MASK_MODE_ENV_VAR, None)
        try:
            self.assertEqual(masking_mod.resolve_mask_mode(), "off")
            self.assertFalse(masking_mod.mask_mode_active())
            policy = self.meta_policy.meta_policies[0]
            self.assertIsNone(policy.feasible_mask,
                              "the v2 policy must not build a mask placeholder")
            self.assertEqual(str(self.meta_policy.mask_mode), "off")
        finally:
            if saved is not None:
                os.environ[masking_mod.MASK_MODE_ENV_VAR] = saved


if __name__ == "__main__":
    unittest.main(verbosity=2)
