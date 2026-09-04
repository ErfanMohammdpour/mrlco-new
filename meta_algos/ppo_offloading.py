# PPO inner adaptation for a single Seq2SeqPolicy (evaluator / held-out val).
# LEARNING_PROTOCOL.md v0.1: k_steps = Adam apply count, shuffle, fresh Adam, value clip.

import numpy as np
import tensorflow as tf

from spec.learning_ops import (
    expected_adam_apply_count,
    select_support_rows,
    shuffled_minibatch_slices,
)


class PPO:
    """Single-policy PPO matching the frozen inner contract."""

    def __init__(
        self,
        policy,
        meta_sampler,
        meta_sampler_process,
        lr=5e-4,
        num_inner_grad_steps=3,
        clip_value=0.2,
        value_clip_epsilon=0.2,
        vf_coef=0.5,
        max_grad_norm=0.5,
        support_trajectories=20,
        ppo_batch_size_trajectories=20,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        entropy_coefficient=0.0,
        rng=None,
    ):
        if int(num_inner_grad_steps) not in (0, 3):
            raise ValueError("v0.1 k_steps must be 0 (zero-shot) or 3 optimizer apply steps")
        if abs(float(lr) - 5e-4) > 1e-12:
            raise ValueError("v0.1 inner_learning_rate must be 5e-4")
        if abs(float(adam_epsilon) - 1e-8) > 1e-15:
            raise ValueError("v0.1 Adam epsilon must be 1e-8")
        if int(ppo_batch_size_trajectories) != 20:
            raise ValueError("v0.1 ppo_batch_size_trajectories must be 20")
        if int(support_trajectories) != 20:
            raise ValueError("v0.1 support trajectories per meta-task must be 20")
        if abs(float(entropy_coefficient)) > 1e-15:
            raise ValueError("v0.1 entropy_coefficient must be 0.0")
        if max_grad_norm is None or abs(float(max_grad_norm) - 0.5) > 1e-12:
            raise ValueError("v0.1 gradient_clip_norm must be 0.5")

        self.lr = float(lr)
        self.num_inner_grad_steps = int(num_inner_grad_steps)
        self.policy = policy
        self.meta_sampler = meta_sampler
        self.meta_sampler_process = meta_sampler_process
        self.clip_value = float(clip_value)
        self.value_clip_epsilon = float(value_clip_epsilon)
        self.vf_coef = float(vf_coef)
        self.max_grad_norm = float(max_grad_norm)
        self.support_trajectories = int(support_trajectories)
        self.ppo_batch_size_trajectories = int(ppo_batch_size_trajectories)
        self.entropy_coefficient = float(entropy_coefficient)
        self.rng = np.random.RandomState() if rng is None else rng

        self.optimizer = tf.compat.v1.train.AdamOptimizer(
            learning_rate=self.lr,
            beta1=adam_beta1,
            beta2=adam_beta2,
            epsilon=adam_epsilon,
            name="ppo_inner_adam",
        )
        self.build_graph()

    def build_graph(self):
        new_logits = self.policy.network.decoder_logits
        self.decoder_inputs = self.policy.decoder_inputs
        self.old_logits = tf.compat.v1.placeholder(
            dtype=tf.float32, shape=[None, None, self.policy.action_dim]
        )
        self.actions = self.policy.decoder_targets
        self.obs = self.policy.obs
        self.vpred = self.policy.vf
        self.decoder_full_length = self.policy.decoder_full_length

        self.old_v = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, None])
        self.advs = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, None])
        self.r = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, None])

        with tf.compat.v1.variable_scope("ppo_update"):
            likelihood_ratio = self.policy.distribution.likelihood_ratio_sym(
                self.actions, self.old_logits, new_logits
            )
            clipped_obj = tf.minimum(
                likelihood_ratio * self.advs,
                tf.clip_by_value(
                    likelihood_ratio, 1.0 - self.clip_value, 1.0 + self.clip_value
                )
                * self.advs,
            )
            self.surr_obj = -tf.reduce_mean(clipped_obj)

            vpredclipped = self.old_v + tf.clip_by_value(
                self.vpred - self.old_v, -self.value_clip_epsilon, self.value_clip_epsilon
            )
            vf_losses1 = tf.square(self.vpred - self.r)
            vf_losses2 = tf.square(vpredclipped - self.r)
            self.vf_loss = 0.5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))
            self.total_loss = self.surr_obj + self.vf_coef * self.vf_loss

            params = self.policy.network.get_trainable_variables()
            grads_and_var = self.optimizer.compute_gradients(self.total_loss, params)
            grads, var = zip(*grads_and_var)
            grads, _grad_norm = tf.clip_by_global_norm(grads, self.max_grad_norm)
            grads_and_var = list(zip(grads, var))
            self._train = self.optimizer.apply_gradients(grads_and_var)
            slot_vars = self.optimizer.variables()
            if slot_vars:
                self._inner_slot_init = tf.compat.v1.variables_initializer(slot_vars)
            else:
                self._inner_slot_init = tf.no_op()

    def reset_inner_optimizer(self):
        sess = tf.compat.v1.get_default_session()
        sess.run(self._inner_slot_init)

    def UpdatePPOTarget(self, task_samples, batch_size=20, k_steps=None):
        steps = self.num_inner_grad_steps if k_steps is None else int(k_steps)
        if steps < 0:
            raise ValueError("k_steps cannot be negative")
        if steps == 0:
            return [], []
        if int(batch_size) != self.ppo_batch_size_trajectories:
            raise ValueError(
                "batch_size=%s != frozen ppo_batch_size_trajectories=%s"
                % (batch_size, self.ppo_batch_size_trajectories)
            )

        self.reset_inner_optimizer()
        observations = np.asarray(task_samples["observations"])
        n = observations.shape[0]
        pick = select_support_rows(n, self.support_trajectories, self.rng)
        actions = np.asarray(task_samples["actions"])[pick]
        observations = observations[pick]
        logits = np.asarray(task_samples["logits"], dtype=np.float32)[pick]
        advantages = np.asarray(task_samples["advantages"], dtype=np.float32)[pick]
        values = np.asarray(task_samples["values"], dtype=np.float32)[pick]
        returns = np.asarray(task_samples["returns"], dtype=np.float32)[pick]
        shift_actions = np.column_stack(
            (np.zeros(actions.shape[0], dtype=np.int32), actions[:, 0:-1])
        )

        sess = tf.compat.v1.get_default_session()
        policy_losses = []
        value_losses = []
        apply_count = 0
        expect = expected_adam_apply_count(
            self.support_trajectories, batch_size, steps
        )
        for _epoch in range(steps):
            for idx in shuffled_minibatch_slices(
                self.support_trajectories, batch_size, self.rng
            ):
                obs_b = observations[idx]
                decoder_full_length = np.array(
                    [obs_b.shape[1]] * obs_b.shape[0], dtype=np.int32
                )
                feed_dict = {
                    self.old_logits: logits[idx],
                    self.old_v: values[idx],
                    self.obs: obs_b,
                    self.actions: actions[idx],
                    self.decoder_inputs: shift_actions[idx],
                    self.decoder_full_length: decoder_full_length,
                    self.advs: advantages[idx],
                    self.r: returns[idx],
                }
                _, value_loss, policy_loss = sess.run(
                    [self._train, self.vf_loss, self.surr_obj], feed_dict=feed_dict
                )
                apply_count += 1
                value_losses.append(value_loss)
                policy_losses.append(policy_loss)
        if apply_count != expect:
            raise RuntimeError(
                "k_steps=%d expected %d Adam apply calls, recorded %d"
                % (steps, expect, apply_count)
            )
        if apply_count != steps:
            raise RuntimeError(
                "k_steps=%d but recorded %d Adam apply calls" % (steps, apply_count)
            )
        return policy_losses, value_losses
