import tensorflow as tf
import numpy as np
import itertools

# this is the tf graph version of reptile:
class MRLCO():
    def __init__(self,
                 policy,
                 meta_batch_size,
                 meta_sampler,
                 meta_sampler_process,
                 outer_lr=1e-4,
                 inner_lr=0.1,
                 num_inner_grad_steps=4,
                 clip_value = 0.2,
                 vf_coef=0.5,
                 max_grad_norm=0.5):
        self.outer_lr = outer_lr
        self.inner_lr = inner_lr
        self.num_inner_grad_steps=num_inner_grad_steps
        self.policy = policy
        self.meta_sampler = meta_sampler
        self.meta_sampler_process = meta_sampler_process
        self.meta_batch_size = meta_batch_size
        self.update_numbers = 1

        #self.optimizer = MpiAdamOptimizer(MPI.COMM_WORLD, learning_rate=self.lr, epsilon=1e-5)
        #self.inner_optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=self.inner_lr)
        self.inner_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.inner_lr)
        self.outer_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.outer_lr)
        self.clip_value = clip_value
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm

        # initialize the place hoder for each task place holder.
        self.new_logits = []
        self.decoder_inputs =[]
        self.old_logits = []
        self.actions = []
        self.obs = []
        self.vpred = []
        self.decoder_full_length = []
        self.adjacency_matrix = []  # Add adjacency matrix placeholders

        self.old_v =[]
        self.advs = []
        self.r = []

        self.surr_obj = []
        self.vf_loss = []
        self.likelihood_ratio = []
        self.clipped_obj = []
        self.total_loss = []
        self._train = []

        self.build_graph()

    def build_graph(self):
        # build inner update for each tasks
        for i in range(self.meta_batch_size):
            self.new_logits.append(self.policy.meta_policies[i].network.decoder_logits)
            self.decoder_inputs.append(self.policy.meta_policies[i].decoder_inputs)
            self.old_logits.append(tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, None, self.policy.action_dim], name='old_logits_ph_task_'+str(i)))
            self.actions.append(self.policy.meta_policies[i].decoder_targets)
            self.obs.append(self.policy.meta_policies[i].obs)
            self.vpred.append(self.policy.meta_policies[i].vf)
            self.decoder_full_length.append(self.policy.meta_policies[i].decoder_full_length)
            # Add adjacency matrix placeholder if it exists
            if hasattr(self.policy.meta_policies[i], 'adjacency_matrix') and self.policy.meta_policies[i].adjacency_matrix is not None:
                self.adjacency_matrix.append(self.policy.meta_policies[i].adjacency_matrix)
            else:
                self.adjacency_matrix.append(None)

            self.old_v.append(tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, None], name='old_v_ph_task_'+str(i)))
            self.advs.append(tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, None], name='advs_ph_task'+str(i)))
            self.r.append(tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, None], name='r_ph_task_'+str(i)))

            with tf.compat.v1.variable_scope("inner_update_parameters_task_"+str(i)) as scope:
                # likelihood_ratio = self.policy.distribution.likelihood_ratio_sym(self.actions[i], self.old_logits[i], self.new_logits[i])
                # self.likelihood_ratio.append(likelihood_ratio)
                #
                # clipped_obj = tf.minimum(likelihood_ratio * self.advs[i] ,
                #                          tf.clip_by_value(likelihood_ratio,
                #                                           1.0 - self.clip_value,
                #                                           1.0 + self.clip_value) * self.advs[i])
                # self.clipped_obj.append(clipped_obj)
                # self.surr_obj.append(-tf.reduce_mean(clipped_obj))
                #
                # vpredclipped = self.vpred[i] + tf.clip_by_value(self.vpred[i] - self.old_v[i], -self.clip_value, self.clip_value)
                # vf_losses1 = tf.square(self.vpred[i] - self.r[i])
                # vf_losses2 = tf.square(vpredclipped - self.r[i])
                #
                # self.vf_loss.append( .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2)) )
                #
                # self.total_loss.append( self.surr_obj[i] + self.vf_coef * self.vf_loss[i])

                likelihood_ratio = self.policy.distribution.likelihood_ratio_sym(
                    self.actions[i], self.old_logits[i], self.new_logits[i]
                )
                self.likelihood_ratio.append(likelihood_ratio)

                ratio = likelihood_ratio
                ratio_clipped = tf.clip_by_value(ratio, 1.0 - self.clip_value, 1.0 + self.clip_value)
                surr1 = ratio * self.advs[i]
                surr2 = ratio_clipped * self.advs[i]
                clipped_obj = tf.where(self.advs[i] >= 0.0, tf.minimum(surr1, surr2), tf.maximum(surr1, surr2))
                self.clipped_obj.append(clipped_obj)

                max_T = tf.shape(self.new_logits[i])[1]
                mask = tf.sequence_mask(self.decoder_full_length[i], maxlen=max_T, dtype=tf.float32)  # [B,T]

                self.surr_obj.append(- tf.reduce_sum(clipped_obj * mask) / (tf.reduce_sum(mask) + 1e-8))

                vpredclipped = self.vpred[i] + tf.clip_by_value(self.vpred[i] - self.old_v[i], -self.clip_value,
                                                                self.clip_value)
                vf_losses1 = tf.square(self.vpred[i] - self.r[i])
                vf_losses2 = tf.square(vpredclipped - self.r[i])
                vf_err = tf.maximum(vf_losses1, vf_losses2) * mask
                self.vf_loss.append(0.5 * tf.reduce_sum(vf_err) / (tf.reduce_sum(mask) + 1e-8))

                logits = self.new_logits[i]
                a0 = logits - tf.reduce_max(logits, axis=-1, keepdims=True)
                ea0 = tf.exp(a0);
                z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
                p0 = ea0 / z0
                ent_per_t = tf.reduce_sum(p0 * (tf.log(z0) - a0), axis=-1)  # [B,T]
                entropy_coef = 1e-2
                ent_mean = tf.reduce_sum(ent_per_t * mask) / (tf.reduce_sum(mask) + 1e-8)

                self.total_loss.append(self.surr_obj[i] + self.vf_coef * self.vf_loss[i] - entropy_coef * ent_mean)


                params = self.policy.meta_policies[i].network.get_trainable_variables()

                grads_and_var = self.inner_optimizer.compute_gradients(self.total_loss[i], params)
                grads, var = zip(*grads_and_var)

                if self.max_grad_norm is not None:
                    # Clip the gradients (normalize)
                    grads, _grad_norm = tf.clip_by_global_norm(grads, self.max_grad_norm)
                grads_and_var = list(zip(grads, var))

                self._train.append(self.inner_optimizer.apply_gradients(grads_and_var))

        # Outer update for the parameters
        # feed in the parameters of inner policy network and update outer parameters.
        with tf.compat.v1.variable_scope("outer_update_parameters") as scope:
            core_network_parameters = self.policy.core_policy.get_trainable_variables()

            # Placeholders to feed precomputed meta-gradients (same shape/dtype as params)
            self.grads_placeholders = []
            for i, var in enumerate(core_network_parameters):
                self.grads_placeholders.append(
                    tf.compat.v1.placeholder(shape=var.shape, dtype=var.dtype, name="grads_%d" % i)
                )

            # Apply the fed gradients to the core parameters
            outer_grads_and_var = list(zip(self.grads_placeholders, core_network_parameters))
            self._outer_train = self.outer_optimizer.apply_gradients(outer_grads_and_var)

    def UpdateMetaPolicy(self):
        """
        TF1-style outer update:
        - Pull numpy values for core & per-task params
        - First-order meta-grad approximation: (core - task)/(inner_lr*K*M*update_numbers)
        - Feed the grads via placeholders and run self._outer_train
        """
        sess = tf.compat.v1.get_default_session()

        core_params_sym = self.policy.core_policy.get_trainable_variables()
        grads_accum = None  # numpy accumulators

        for t in range(self.meta_batch_size):
            task_params_sym = self.policy.meta_policies[t].get_trainable_variables()

            # Fetch actual numpy values
            core_vals, task_vals = sess.run([core_params_sym, task_params_sym])

            # First-order meta-gradient (Reptile-style)
            scale = (self.inner_lr * self.num_inner_grad_steps *
                     self.meta_batch_size * self.update_numbers)
            if grads_accum is None:
                grads_accum = []
                for core_v, task_v in zip(core_vals, task_vals):
                    g = (core_v - task_v) / scale
                    grads_accum.append(g)
            else:
                for i, (core_v, task_v) in enumerate(zip(core_vals, task_vals)):
                    grads_accum[i] += (core_v - task_v) / scale

        # Actually apply meta gradients
        feed = {ph: g for ph, g in zip(self.grads_placeholders, grads_accum)}
        sess.run(self._outer_train, feed_dict=feed)

        print("async core policy to meta-policy")
        self.policy.async_parameters()

    def UpdatePPOTarget(self, task_samples, batch_size=50):
        total_policy_losses = []
        total_value_losses = []
        for i in range(self.meta_batch_size):
            policy_losses, value_losses = self.UpdatePPOTargetPerTask(task_samples[i], i, batch_size)
            total_policy_losses.append(policy_losses)
            total_value_losses.append(value_losses)

        return total_policy_losses, total_value_losses

    def UpdatePPOTargetPerTask(self, task_samples, task_id, batch_size=50):
        policy_losses = []
        value_losses = []

        batch_number = int(task_samples['observations'].shape[0] / batch_size)
        self.update_numbers = batch_number
        #:q!
        # print("update number is: ", self.update_numbers)
        #observations = task_samples['observations']

        shift_actions = np.column_stack(
                    (np.zeros(task_samples['actions'].shape[0], dtype=np.int32), task_samples['actions'][:, 0:-1]))

        observations_batchs = np.split(np.array(task_samples['observations']), batch_number)
        actions_batchs = np.split(np.array(task_samples['actions']), batch_number)
        shift_action_batchs = np.split(np.array(shift_actions), batch_number)

        old_logits_batchs = np.split(np.array(task_samples["logits"], dtype=np.float32 ), batch_number)
        advs_batchs = np.split(np.array(task_samples['advantages'], dtype=np.float32), batch_number)
        oldvpred = np.split(np.array(task_samples['values'], dtype=np.float32), batch_number)
        returns = np.split(np.array(task_samples['returns'], dtype=np.float32), batch_number)

        sess = tf.compat.v1.get_default_session()

        vf_loss = 0.0
        pg_loss = 0.0
        dec_full_lens_batchs = np.split(np.array(task_samples['decoder_full_lengths'], dtype=np.int32), batch_number)

        # copy_policy.set_weights(self.policy.get_weights())
        for i in range(self.num_inner_grad_steps):
            # action, old_logits, _ = copy_policy(observations)
            for old_logits, old_v, observations, actions, shift_actions, advs, r, dec_lens  in zip(old_logits_batchs, oldvpred, observations_batchs, actions_batchs,
                                                                                        shift_action_batchs, advs_batchs, returns, dec_full_lens_batchs):
                # decoder_full_length = np.array([observations.shape[1]] * observations.shape[0], dtype=np.int32)

                feed_dict = {self.old_logits[task_id]: old_logits, self.old_v[task_id]: old_v, self.obs[task_id]: observations, self.actions[task_id]: actions,
                            self.decoder_inputs[task_id]: shift_actions,
                             self.decoder_full_length[task_id]: dec_lens, self.advs[task_id]: advs, self.r[task_id]: r}
                
                # Add adjacency matrix to feed_dict if placeholder exists
                if self.adjacency_matrix[task_id] is not None:
                    # Try to use real adjacency from samples_data if available
                    if 'adjacency_matrices' in task_samples and task_samples['adjacency_matrices'] is not None:
                        # Use real DAG adjacency with edge weights
                        adjacency = task_samples['adjacency_matrices']
                        # Ensure correct shape
                        if len(adjacency.shape) == 2:
                            adjacency = adjacency[np.newaxis, :]
                        feed_dict[self.adjacency_matrix[task_id]] = adjacency
                        # Debug log for first iteration
                        if i == 0 and task_id == 0:
                            print(f"[DEBUG] Using REAL adjacency matrix from task graph")
                            print(f"        Shape: {adjacency.shape}, Sparsity: {np.mean(adjacency == 0):.2%}")
                            print(f"        Edge weight range: [{np.min(adjacency[adjacency > 0]):.3f}, {np.max(adjacency[adjacency > 0]):.3f}]")
                    else:
                        # Fallback to default fully connected adjacency matrix
                        batch_size_adj = observations.shape[0]
                        num_nodes = observations.shape[1]
                        default_adjacency = np.ones((batch_size_adj, num_nodes, num_nodes), dtype=np.float32)
                        feed_dict[self.adjacency_matrix[task_id]] = default_adjacency
                        # Debug log for first iteration
                        if i == 0 and task_id == 0:
                            print(f"[DEBUG] Using DEFAULT fully-connected adjacency (fallback)")
                            print(f"        Reason: {'adjacency_matrices' not in task_samples if 'adjacency_matrices' not in task_samples else 'adjacency_matrices is None'}")
                            print(f"        Shape: {default_adjacency.shape}")

                _, value_loss, policy_loss, likelihood_ratio_val, advs_val, clipped_obj_val = sess.run(
                    [self._train[task_id], self.vf_loss[task_id], self.surr_obj[task_id],
                     self.likelihood_ratio[task_id], self.advs[task_id], self.clipped_obj[task_id]],
                    feed_dict=feed_dict)

                # Debug logging
                if i == 0 and task_id == 0:  # Log only for first iteration and task
                    print(f"\n[DEBUG] Loss calculation details:")
                    print(f"  Policy loss (surr_obj): {policy_loss}")
                    print(f"  Value loss: {value_loss}")
                    print(f"  Likelihood ratio mean: {np.mean(likelihood_ratio_val)}")
                    print(f"  Likelihood ratio std: {np.std(likelihood_ratio_val)}")
                    print(f"  Advantages mean: {np.mean(advs_val)}")
                    print(f"  Advantages std: {np.std(advs_val)}")
                    print(f"  Clipped objective mean: {np.mean(clipped_obj_val)}")

                vf_loss += value_loss
                pg_loss += policy_loss

            vf_loss = vf_loss / float(self.num_inner_grad_steps)
            pg_loss = pg_loss / float(self.num_inner_grad_steps)

            value_losses.append(vf_loss)
            policy_losses.append(pg_loss)

        return policy_losses, value_losses
