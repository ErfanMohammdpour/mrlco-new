"""MRLCO inner PPO + outer mean first-order pseudogradient (LEARNING_PROTOCOL.md)."""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from spec.learning_ops import (
    clipped_value_prediction,
    mean_pseudogradient,
    select_support_rows,
    shuffled_minibatch_slices,
)


class MRLCO:
    def __init__(
        self,
        policy,
        meta_batch_size,
        meta_sampler,
        meta_sampler_process,
        outer_lr=5e-4,
        inner_lr=5e-4,
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
        rng=None,
    ):
        if int(num_inner_grad_steps) != 3:
            raise ValueError("v0.1 k_steps must be 3 optimizer apply steps")
        if int(ppo_batch_size_trajectories) != 20:
            raise ValueError("v0.1 ppo_batch_size_trajectories must be 20")
        if int(support_trajectories) != 20:
            raise ValueError("v0.1 support trajectories per meta-task must be 20")
        self.outer_lr = float(outer_lr)
        self.inner_lr = float(inner_lr)
        self.num_inner_grad_steps = int(num_inner_grad_steps)
        self.policy = policy
        self.meta_sampler = meta_sampler
        self.meta_sampler_process = meta_sampler_process
        self.meta_batch_size = int(meta_batch_size)
        self.clip_value = float(clip_value)
        self.value_clip_epsilon = float(value_clip_epsilon)
        self.vf_coef = float(vf_coef)
        self.max_grad_norm = max_grad_norm
        self.support_trajectories = int(support_trajectories)
        self.ppo_batch_size_trajectories = int(ppo_batch_size_trajectories)
        self.rng = np.random.RandomState() if rng is None else rng

        self.inner_optimizers = []
        self.outer_optimizer = tf.compat.v1.train.AdamOptimizer(
            learning_rate=self.outer_lr,
            beta1=adam_beta1,
            beta2=adam_beta2,
            epsilon=adam_epsilon,
            name="outer_adam",
        )

        self.new_logits = []
        self.decoder_inputs = []
        self.old_logits = []
        self.actions = []
        self.obs = []
        self.vpred = []
        self.decoder_full_length = []
        self.old_v = []
        self.advs = []
        self.r = []
        self.surr_obj = []
        self.vf_loss = []
        self.likelihood_ratio = []
        self.clipped_obj = []
        self.total_loss = []
        self._train = []
        self._inner_slot_init = []

        self.build_graph(adam_beta1, adam_beta2, adam_epsilon)

    def build_graph(self, adam_beta1, adam_beta2, adam_epsilon):
        for i in range(self.meta_batch_size):
            self.new_logits.append(self.policy.meta_policies[i].network.decoder_logits)
            self.decoder_inputs.append(self.policy.meta_policies[i].decoder_inputs)
            self.old_logits.append(
                tf.compat.v1.placeholder(
                    dtype=tf.float32,
                    shape=[None, None, self.policy.action_dim],
                    name="old_logits_ph_task_" + str(i),
                )
            )
            self.actions.append(self.policy.meta_policies[i].decoder_targets)
            self.obs.append(self.policy.meta_policies[i].obs)
            self.vpred.append(self.policy.meta_policies[i].vf)
            self.decoder_full_length.append(self.policy.meta_policies[i].decoder_full_length)
            self.old_v.append(
                tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, None], name="old_v_ph_task_" + str(i))
            )
            self.advs.append(
                tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, None], name="advs_ph_task" + str(i))
            )
            self.r.append(
                tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, None], name="r_ph_task_" + str(i))
            )

            with tf.compat.v1.variable_scope("inner_update_parameters_task_" + str(i)):
                likelihood_ratio = self.policy.distribution.likelihood_ratio_sym(
                    self.actions[i], self.old_logits[i], self.new_logits[i]
                )
                self.likelihood_ratio.append(likelihood_ratio)
                clipped_obj = tf.minimum(
                    likelihood_ratio * self.advs[i],
                    tf.clip_by_value(likelihood_ratio, 1.0 - self.clip_value, 1.0 + self.clip_value)
                    * self.advs[i],
                )
                self.clipped_obj.append(clipped_obj)
                self.surr_obj.append(-tf.reduce_mean(clipped_obj))

                # LEARNING_PROTOCOL: v_old + clip(v_new - v_old, -eps, eps)
                vpredclipped = self.old_v[i] + tf.clip_by_value(
                    self.vpred[i] - self.old_v[i],
                    -self.value_clip_epsilon,
                    self.value_clip_epsilon,
                )
                vf_losses1 = tf.square(self.vpred[i] - self.r[i])
                vf_losses2 = tf.square(vpredclipped - self.r[i])
                self.vf_loss.append(0.5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2)))
                self.total_loss.append(self.surr_obj[i] + self.vf_coef * self.vf_loss[i])

                params = self.policy.meta_policies[i].network.get_trainable_variables()
                inner_opt = tf.compat.v1.train.AdamOptimizer(
                    learning_rate=self.inner_lr,
                    beta1=adam_beta1,
                    beta2=adam_beta2,
                    epsilon=adam_epsilon,
                    name="inner_adam_task_%d" % i,
                )
                self.inner_optimizers.append(inner_opt)
                grads_and_var = inner_opt.compute_gradients(self.total_loss[i], params)
                grads, var = zip(*grads_and_var)
                if self.max_grad_norm is not None:
                    grads, _grad_norm = tf.clip_by_global_norm(grads, self.max_grad_norm)
                grads_and_var = list(zip(grads, var))
                train_op = inner_opt.apply_gradients(grads_and_var)
                self._train.append(train_op)
                slot_vars = inner_opt.variables()
                if slot_vars:
                    self._inner_slot_init.append(tf.compat.v1.variables_initializer(slot_vars))
                else:
                    self._inner_slot_init.append(tf.no_op())

        with tf.compat.v1.variable_scope("outer_update_parameters"):
            core_network_parameters = self.policy.core_policy.get_trainable_variables()
            self.grads_placeholders = []
            for i, var in enumerate(core_network_parameters):
                self.grads_placeholders.append(
                    tf.compat.v1.placeholder(shape=var.shape, dtype=var.dtype, name="grads_" + str(i))
                )
            outer_grads_and_var = list(zip(self.grads_placeholders, core_network_parameters))
            self._outer_train = self.outer_optimizer.apply_gradients(outer_grads_and_var)

    def sync_task_policies_from_core(self):
        self.policy.async_parameters()

    def reset_inner_optimizer(self, task_id):
        sess = tf.compat.v1.get_default_session()
        sess.run(self._inner_slot_init[task_id])

    def UpdateMetaPolicy(self):
        """One order-invariant outer Adam step on mean first-order pseudogradient."""
        sess = tf.compat.v1.get_default_session()
        core_syms = self.policy.core_policy.get_trainable_variables()
        theta0 = sess.run(core_syms)
        adapted = []
        for task_id in range(self.meta_batch_size):
            adapted.append(sess.run(self.policy.meta_policies[task_id].get_trainable_variables()))
        grads = mean_pseudogradient(theta0, adapted, self.inner_lr, self.num_inner_grad_steps)
        feed = {ph: g.astype(ph.dtype.as_numpy_dtype) for ph, g in zip(self.grads_placeholders, grads)}
        sess.run(self._outer_train, feed_dict=feed)
        self.sync_task_policies_from_core()

    def UpdatePPOTarget(self, task_samples, batch_size=20):
        if int(batch_size) != self.ppo_batch_size_trajectories:
            raise ValueError(
                "batch_size=%s != frozen ppo_batch_size_trajectories=%s"
                % (batch_size, self.ppo_batch_size_trajectories)
            )
        total_policy_losses = []
        total_value_losses = []
        for task_id in range(self.meta_batch_size):
            policy_losses, value_losses = self.UpdatePPOTargetPerTask(task_samples[task_id], task_id, batch_size)
            total_policy_losses.append(policy_losses)
            total_value_losses.append(value_losses)
        return total_policy_losses, total_value_losses

    def UpdatePPOTargetPerTask(self, task_samples, task_id, batch_size=20):
        self.reset_inner_optimizer(task_id)
        observations = np.asarray(task_samples["observations"])
        pick = select_support_rows(observations.shape[0], self.support_trajectories, self.rng)
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
        for _epoch in range(self.num_inner_grad_steps):
            for idx in shuffled_minibatch_slices(
                self.support_trajectories, batch_size, self.rng
            ):
                obs_b = observations[idx]
                decoder_full_length = np.array([obs_b.shape[1]] * obs_b.shape[0], dtype=np.int32)
                feed_dict = {
                    self.old_logits[task_id]: logits[idx],
                    self.old_v[task_id]: values[idx],
                    self.obs[task_id]: obs_b,
                    self.actions[task_id]: actions[idx],
                    self.decoder_inputs[task_id]: shift_actions[idx],
                    self.decoder_full_length[task_id]: decoder_full_length,
                    self.advs[task_id]: advantages[idx],
                    self.r[task_id]: returns[idx],
                }
                _, value_loss, policy_loss = sess.run(
                    [self._train[task_id], self.vf_loss[task_id], self.surr_obj[task_id]],
                    feed_dict=feed_dict,
                )
                apply_count += 1
                value_losses.append(value_loss)
                policy_losses.append(policy_loss)
        if apply_count != self.num_inner_grad_steps:
            raise RuntimeError(
                "k_steps=%d but recorded %d Adam apply calls" % (self.num_inner_grad_steps, apply_count)
            )
        return policy_losses, value_losses


_ = clipped_value_prediction
