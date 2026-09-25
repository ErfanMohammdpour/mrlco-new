"""MRLCO inner PPO + outer mean first-order pseudogradient (LEARNING_PROTOCOL.md)."""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from spec.learning_ops import (
    clipped_value_prediction,
    instance_ids_from_order,
    mean_pseudogradient,
    select_elite_per_instance,
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
        support_select="random",
        bc_kl_coef=0.0,
    ):
        if int(num_inner_grad_steps) != 3:
            raise ValueError("v0.1 k_steps must be 3 optimizer apply steps")
        if int(ppo_batch_size_trajectories) != 20:
            raise ValueError("v0.1 ppo_batch_size_trajectories must be 20")
        if int(support_trajectories) != 20:
            raise ValueError("v0.1 support trajectories per meta-task must be 20")
        if int(meta_batch_size) != 10:
            raise ValueError("v0.1 meta_batch_size_distributions must be 10")
        if abs(float(inner_lr) - 5e-4) > 1e-12:
            raise ValueError("v0.1 inner_learning_rate must be 5e-4")
        if abs(float(outer_lr) - 5e-4) > 1e-12:
            raise ValueError("v0.1 outer Adam learning_rate must be 5e-4")
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
        if support_select not in ("random", "elite"):
            raise ValueError("support_select must be random or elite, got %r" % (support_select,))
        self.support_select = support_select
        self.bc_kl_coef = float(bc_kl_coef)
        if self.bc_kl_coef < 0:
            raise ValueError("bc_kl_coef must be >= 0")
        self.last_kl_bc = None
        self.last_update_mode = "publication"

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
        self.approx_kl = []
        self.clip_fraction = []
        self.grad_norm = []
        self.last_approx_kl: float | None = None
        self.last_clip_fraction: float | None = None
        self.last_grad_norm: float | None = None
        self._task_diagnostics: list[tuple[float, float, float]] = []
        self.clipped_obj = []
        self.total_loss = []
        self._train = []
        self._train_kl = []
        self._train_vf_only = []
        self.bc_logits = []
        self.kl_bc = []
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
                # P1 additive diagnostics from the same likelihood ratio
                self.approx_kl.append(
                    tf.reduce_mean(likelihood_ratio - 1.0 - tf.math.log(likelihood_ratio))
                )
                self.clip_fraction.append(
                    tf.reduce_mean(
                        tf.cast(
                            tf.greater(
                                tf.abs(likelihood_ratio - 1.0), self.clip_value
                            ),
                            tf.float32,
                        )
                    )
                )

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
                    grads, grad_norm = tf.clip_by_global_norm(grads, self.max_grad_norm)
                else:
                    grad_norm = tf.constant(0.0)
                self.grad_norm.append(grad_norm)
                grads_and_var = list(zip(grads, var))
                train_op = inner_opt.apply_gradients(grads_and_var)
                self._train.append(train_op)

                if self.bc_kl_coef > 0:
                    self.bc_logits.append(
                        tf.compat.v1.placeholder(
                            dtype=tf.float32,
                            shape=[None, None, self.policy.action_dim],
                            name="bc_logits_ph_task_" + str(i),
                        )
                    )
                    # KL(π || π_BC): current logits vs frozen BC logits. Not PPO π_old.
                    self.kl_bc.append(
                        tf.reduce_mean(
                            self.policy.distribution.kl_sym(self.new_logits[i], self.bc_logits[i])
                        )
                    )
                    kl_total = self.surr_obj[i] + self.vf_coef * self.vf_loss[i] + self.bc_kl_coef * self.kl_bc[i]
                    kl_gvs = inner_opt.compute_gradients(kl_total, params)
                    kl_grads, kl_vars = zip(*kl_gvs)
                    if self.max_grad_norm is not None:
                        kl_grads, _ = tf.clip_by_global_norm(kl_grads, self.max_grad_norm)
                    self._train_kl.append(inner_opt.apply_gradients(list(zip(kl_grads, kl_vars))))
                    vf_params = [v for v in params if "qvalue_layer" in v.name]
                    if not vf_params:
                        raise ValueError("qvalue_layer vars missing for critic warmup task %d" % i)
                    vf_gvs = inner_opt.compute_gradients(self.vf_loss[i], vf_params)
                    vf_grads, vf_vars = zip(*vf_gvs)
                    if self.max_grad_norm is not None:
                        vf_grads, _ = tf.clip_by_global_norm(vf_grads, self.max_grad_norm)
                    self._train_vf_only.append(inner_opt.apply_gradients(list(zip(vf_grads, vf_vars))))
                else:
                    self.bc_logits.append(None)
                    self.kl_bc.append(None)
                    self._train_kl.append(None)
                    self._train_vf_only.append(None)

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

    def _pick_support_rows(self, task_samples, n):
        k = self.support_trajectories
        if self.support_select != "elite":
            return select_support_rows(n, k, self.rng)
        ft = np.asarray(task_samples["finish_time"], dtype=np.float64).reshape(n, -1)[:, -1]
        ids = instance_ids_from_order(n, k)
        return select_elite_per_instance(ft, ids, k)

    def UpdatePPOTarget(self, task_samples, batch_size=20, update_mode="publication", bc_policy=None):
        if int(batch_size) != self.ppo_batch_size_trajectories:
            raise ValueError(
                "batch_size=%s != frozen ppo_batch_size_trajectories=%s"
                % (batch_size, self.ppo_batch_size_trajectories)
            )
        if update_mode not in ("publication", "kl_bc", "vf_only"):
            raise ValueError("update_mode must be publication, kl_bc, or vf_only, got %r" % (update_mode,))
        if update_mode != "publication" and self.bc_kl_coef <= 0:
            raise ValueError("update_mode=%s requires bc_kl_coef > 0" % update_mode)
        if update_mode == "kl_bc" and bc_policy is None:
            raise ValueError("kl_bc update needs frozen bc_policy")
        self.last_update_mode = update_mode
        self.last_kl_bc = []
        self._task_diagnostics = []
        total_policy_losses = []
        total_value_losses = []
        for task_id in range(self.meta_batch_size):
            policy_losses, value_losses = self.UpdatePPOTargetPerTask(
                task_samples[task_id],
                task_id,
                batch_size,
                update_mode=update_mode,
                bc_policy=bc_policy,
            )
            total_policy_losses.append(policy_losses)
            total_value_losses.append(value_losses)
        if self._task_diagnostics:
            n = float(len(self._task_diagnostics))
            self.last_approx_kl = sum(d[0] for d in self._task_diagnostics) / n
            self.last_clip_fraction = sum(d[1] for d in self._task_diagnostics) / n
            self.last_grad_norm = sum(d[2] for d in self._task_diagnostics) / n
        else:
            self.last_approx_kl = None
            self.last_clip_fraction = None
            self.last_grad_norm = None
        return total_policy_losses, total_value_losses

    def _frozen_bc_logits(self, bc_policy, obs_b, shift_b, actions_b):
        sess = tf.compat.v1.get_default_session()
        decoder_full_length = np.array([obs_b.shape[1]] * obs_b.shape[0], dtype=np.int32)
        feed_dict = {
            bc_policy.obs: obs_b,
            bc_policy.decoder_inputs: shift_b,
            bc_policy.decoder_targets: actions_b,
            bc_policy.decoder_full_length: decoder_full_length,
        }
        # The BC reference stays UNMASKED: all-feasible is the byte-exact legacy
        # distribution, and the KL is driven by the masked policy side (p=0 on
        # infeasible actions contributes nothing).
        mask_ph = getattr(bc_policy, "feasible_mask", None)
        if mask_ph is not None:
            feed_dict[mask_ph] = np.ones(
                (obs_b.shape[0], obs_b.shape[1], bc_policy.action_dim), dtype=np.float32
            )
        return sess.run(bc_policy.network.decoder_logits, feed_dict=feed_dict)

    def _mask_ph(self, task_id):
        return getattr(self.policy.meta_policies[task_id], "feasible_mask", None)

    def _feasible_batch(self, task_samples, pick, n_rows, n_slots, task_id):
        """Stored rollout mask for the selected rows (None when masking is off)."""
        if self._mask_ph(task_id) is None:
            return None
        if "feasible" not in task_samples or task_samples["feasible"] is None:
            raise ValueError(
                "masking is active but task %d batch has no 'feasible' entry; the "
                "rollout mask must be stored, never recomputed at update time" % task_id
            )
        feasible = np.asarray(task_samples["feasible"], dtype=np.float32)
        if feasible.ndim == 2:
            feasible = feasible[None, ...]
        if feasible.shape[0] != n_rows:
            raise ValueError(
                "feasible rows %d != trajectory rows %d" % (feasible.shape[0], n_rows)
            )
        feasible = feasible[pick]
        want = (feasible.shape[0], n_slots, self.policy.action_dim)
        if tuple(feasible.shape[1:]) != want[1:]:
            raise ValueError(
                "feasible mask shape %s != expected %s"
                % (tuple(feasible.shape), want)
            )
        return feasible

    def UpdatePPOTargetPerTask(self, task_samples, task_id, batch_size=20, update_mode="publication", bc_policy=None):
        self.reset_inner_optimizer(task_id)
        observations = np.asarray(task_samples["observations"])
        n_rows = observations.shape[0]
        pick = self._pick_support_rows(task_samples, n_rows)
        actions = np.asarray(task_samples["actions"])[pick]
        feasible = self._feasible_batch(task_samples, pick, n_rows, observations.shape[1], task_id)
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
        kl_values = []
        clip_values = []
        grad_values = []
        apply_count = 0
        if update_mode == "kl_bc":
            train_op = self._train_kl[task_id]
        elif update_mode == "vf_only":
            train_op = self._train_vf_only[task_id]
        else:
            train_op = self._train[task_id]
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
                if feasible is not None:
                    feed_dict[self._mask_ph(task_id)] = feasible[idx]
                fetches = [
                    train_op,
                    self.vf_loss[task_id],
                    self.surr_obj[task_id],
                    self.approx_kl[task_id],
                    self.clip_fraction[task_id],
                    self.grad_norm[task_id],
                ]
                if update_mode == "kl_bc":
                    feed_dict[self.bc_logits[task_id]] = self._frozen_bc_logits(
                        bc_policy, obs_b, shift_actions[idx], actions[idx]
                    )
                    fetches.append(self.kl_bc[task_id])
                out = sess.run(fetches, feed_dict=feed_dict)
                value_loss = out[1]
                policy_loss = out[2]
                kl_values.append(float(out[3]))
                clip_values.append(float(out[4]))
                grad_values.append(float(out[5]))
                if update_mode == "kl_bc":
                    self.last_kl_bc.append(float(out[6]))
                apply_count += 1
                value_losses.append(value_loss)
                policy_losses.append(policy_loss)
        if apply_count != self.num_inner_grad_steps:
            raise RuntimeError(
                "k_steps=%d but recorded %d Adam apply calls" % (self.num_inner_grad_steps, apply_count)
            )
        if kl_values:
            self._task_diagnostics.append(
                (
                    float(np.mean(kl_values)),
                    float(np.mean(clip_values)),
                    float(np.mean(grad_values)),
                )
            )
        return policy_losses, value_losses


_ = clipped_value_prediction
