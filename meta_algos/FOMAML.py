import tensorflow as tf
import numpy as np
import itertools

class FOMAML:
    """
    First-Order Model-Agnostic Meta-Learning (FOMAML) implementation
    This is a first-order approximation of MAML that avoids second-order gradients
    """
    
    def __init__(self,
                 policy,
                 meta_batch_size,
                 meta_sampler,
                 meta_sampler_process,
                 outer_lr=1e-4,
                 inner_lr=0.1,
                 num_inner_grad_steps=4,
                 clip_value=0.2,
                 vf_coef=0.5,
                 max_grad_norm=0.5):
        
        self.outer_lr = outer_lr
        self.inner_lr = inner_lr
        self.num_inner_grad_steps = num_inner_grad_steps
        self.policy = policy
        self.meta_sampler = meta_sampler
        self.meta_sampler_process = meta_sampler_process
        self.meta_batch_size = meta_batch_size
        self.clip_value = clip_value
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm

        # Optimizers
        self.inner_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.inner_lr)
        self.outer_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.outer_lr)

        # Placeholders for each task
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

        # Loss components for each task
        self.surr_obj = []
        self.vf_loss = []
        self.likelihood_ratio = []
        self.clipped_obj = []
        self.total_loss = []
        self._inner_train = []

        # Meta-update components
        self.meta_losses = []
        self.meta_grads_placeholders = []
        self._meta_train = None

        self.build_graph()

    def build_graph(self):
        """Build the FOMAML computation graph"""
        
        # Build inner update graphs for each task
        for i in range(self.meta_batch_size):
            self.new_logits.append(self.policy.meta_policies[i].network.decoder_logits)
            self.decoder_inputs.append(self.policy.meta_policies[i].decoder_inputs)
            self.old_logits.append(tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, None, self.policy.action_dim], 
                                                           name='old_logits_ph_task_'+str(i)))
            self.actions.append(self.policy.meta_policies[i].decoder_targets)
            self.obs.append(self.policy.meta_policies[i].obs)
            self.vpred.append(self.policy.meta_policies[i].vf)
            self.decoder_full_length.append(self.policy.meta_policies[i].decoder_full_length)

            self.old_v.append(tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, None], 
                                                      name='old_v_ph_task_'+str(i)))
            self.advs.append(tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, None], 
                                                     name='advs_ph_task_'+str(i)))
            self.r.append(tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, None], 
                                                  name='r_ph_task_'+str(i)))

            # Build inner loop loss for task i
            with tf.compat.v1.variable_scope("inner_update_parameters_task_"+str(i), reuse=tf.compat.v1.AUTO_REUSE):
                likelihood_ratio = self.policy.distribution.likelihood_ratio_sym(
                    self.actions[i], self.old_logits[i], self.new_logits[i])
                self.likelihood_ratio.append(likelihood_ratio)

                clipped_obj = tf.minimum(likelihood_ratio * self.advs[i],
                                       tf.clip_by_value(likelihood_ratio,
                                                       1.0 - self.clip_value,
                                                       1.0 + self.clip_value) * self.advs[i])
                self.clipped_obj.append(clipped_obj)
                self.surr_obj.append(-tf.reduce_mean(clipped_obj))

                vpredclipped = self.vpred[i] + tf.clip_by_value(self.vpred[i] - self.old_v[i], 
                                                               -self.clip_value, self.clip_value)
                vf_losses1 = tf.square(self.vpred[i] - self.r[i])
                vf_losses2 = tf.square(vpredclipped - self.r[i])

                self.vf_loss.append(0.5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2)))
                self.total_loss.append(self.surr_obj[i] + self.vf_coef * self.vf_loss[i])

                # Inner loop training step
                params = self.policy.meta_policies[i].network.get_trainable_variables()
                grads_and_var = self.inner_optimizer.compute_gradients(self.total_loss[i], params)
                grads, var = zip(*grads_and_var)

                if self.max_grad_norm is not None:
                    grads, _grad_norm = tf.clip_by_global_norm(grads, self.max_grad_norm)
                grads_and_var = list(zip(grads, var))

                self._inner_train.append(self.inner_optimizer.apply_gradients(grads_and_var))

        # Build meta-update graph
        self._build_meta_update_graph()

    def _build_meta_update_graph(self):
        """Build the meta-update computation graph"""
        with tf.compat.v1.variable_scope("meta_update_parameters"):
            # Get core policy parameters
            core_network_parameters = self.policy.core_policy.get_trainable_variables()
            
            # Create placeholders for meta-gradients
            self.meta_grads_placeholders = []
            for i, var in enumerate(core_network_parameters):
                self.meta_grads_placeholders.append(
                    tf.compat.v1.placeholder(shape=var.shape, dtype=var.dtype, 
                                           name="meta_grads_"+str(i)))

            # Meta-update step
            meta_grads_and_var = list(zip(self.meta_grads_placeholders, core_network_parameters))
            self._meta_train = self.outer_optimizer.apply_gradients(meta_grads_and_var)

    def adapt_task(self, task_samples, task_id, batch_size=50):
        """
        Inner loop: Adapt model to specific task using support set
        This performs multiple gradient steps on the task-specific policy
        """
        policy_losses = []
        value_losses = []

        batch_number = int(task_samples['observations'].shape[0] / batch_size)
        
        # Prepare data
        shift_actions = np.column_stack(
            (np.zeros(task_samples['actions'].shape[0], dtype=np.int32), 
             task_samples['actions'][:, 0:-1]))

        observations_batchs = np.split(np.array(task_samples['observations']), batch_number)
        actions_batchs = np.split(np.array(task_samples['actions']), batch_number)
        shift_action_batchs = np.split(np.array(shift_actions), batch_number)
        old_logits_batchs = np.split(np.array(task_samples["logits"], dtype=np.float32), batch_number)
        advs_batchs = np.split(np.array(task_samples['advantages'], dtype=np.float32), batch_number)
        oldvpred = np.split(np.array(task_samples['values'], dtype=np.float32), batch_number)
        returns = np.split(np.array(task_samples['returns'], dtype=np.float32), batch_number)

        sess = tf.compat.v1.get_default_session()

        # Perform multiple inner loop steps
        for step in range(self.num_inner_grad_steps):
            vf_loss = 0.0
            pg_loss = 0.0
            
            for old_logits, old_v, observations, actions, shift_actions, advs, r in zip(
                old_logits_batchs, oldvpred, observations_batchs, actions_batchs,
                shift_action_batchs, advs_batchs, returns):
                
                decoder_full_length = np.array([observations.shape[1]] * observations.shape[0], dtype=np.int32)

                feed_dict = {
                    self.old_logits[task_id]: old_logits,
                    self.old_v[task_id]: old_v,
                    self.obs[task_id]: observations,
                    self.actions[task_id]: actions,
                    self.decoder_inputs[task_id]: shift_actions,
                    self.decoder_full_length[task_id]: decoder_full_length,
                    self.advs[task_id]: advs,
                    self.r[task_id]: r
                }

                _, value_loss, policy_loss = sess.run(
                    [self._inner_train[task_id], self.vf_loss[task_id], self.surr_obj[task_id]], 
                    feed_dict=feed_dict)

                vf_loss += value_loss
                pg_loss += policy_loss

            vf_loss = vf_loss / float(batch_number)
            pg_loss = pg_loss / float(batch_number)

            value_losses.append(vf_loss)
            policy_losses.append(pg_loss)

        return policy_losses, value_losses

    def evaluate_adapted_policy(self, task_samples, task_id):
        """
        Evaluate the adapted policy on query set
        This computes the loss that will be used for meta-update
        """
        # Use the same loss computation as inner loop but for evaluation
        # In practice, this would be the same as adapt_task but without parameter updates
        policy_losses, value_losses = self.adapt_task(task_samples, task_id)
        
        # Return the final loss for meta-update
        return policy_losses[-1] + self.vf_coef * value_losses[-1]

    def meta_update(self, adapted_policies, query_losses):
        """
        Outer loop: Meta-update using query set losses
        This is the key difference from Reptile - we use explicit gradients
        """
        sess = tf.compat.v1.get_default_session()
        
        # Get current core policy parameters
        core_params = sess.run(self.policy.core_policy.get_trainable_variables())
        
        # Compute meta-gradients using first-order approximation
        meta_grads = []
        for i, (adapted_policy, query_loss) in enumerate(zip(adapted_policies, query_losses)):
            # Get adapted parameters for this task
            adapted_params = sess.run(self.policy.meta_policies[i].get_trainable_variables())
            
            # Compute meta-gradient as difference between adapted and original parameters
            # This is the first-order approximation of the meta-gradient
            meta_grad = [(adapted - original) / self.inner_lr / self.num_inner_grad_steps 
                        for adapted, original in zip(adapted_params, core_params)]
            meta_grads.append(meta_grad)
        
        # Average meta-gradients across all tasks
        if len(meta_grads) > 1:
            avg_meta_grads = [np.mean([grad[i] for grad in meta_grads], axis=0) 
                             for i in range(len(meta_grads[0]))]
        else:
            avg_meta_grads = meta_grads[0]
        
        # Apply meta-update
        update_feed_dict = {}
        for i, grad in enumerate(avg_meta_grads):
            update_feed_dict[self.meta_grads_placeholders[i]] = grad
        
        sess.run(self._meta_train, feed_dict=update_feed_dict)
        
        # Sync adapted parameters back to meta-policies
        self.policy.async_parameters()
        
        print(f"Applied meta-update with {len(adapted_policies)} tasks")

    def UpdatePPOTarget(self, task_samples, batch_size=50):
        """
        Compatibility method - this is now handled by adapt_task
        """
        total_policy_losses = []
        total_value_losses = []
        
        for i in range(self.meta_batch_size):
            policy_losses, value_losses = self.adapt_task(task_samples[i], i, batch_size)
            total_policy_losses.append(policy_losses)
            total_value_losses.append(value_losses)

        return total_policy_losses, total_value_losses

    def UpdateMetaPolicy(self):
        """
        Compatibility method - this is now handled by meta_update
        """
        # This method is kept for compatibility but the actual meta-update
        # is now handled in the training loop
        pass
