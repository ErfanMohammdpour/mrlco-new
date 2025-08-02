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
        # MIGRATION: Use Keras optimizers
        self.inner_optimizer = tf.keras.optimizers.Adam(learning_rate=self.inner_lr)
        self.outer_optimizer = tf.keras.optimizers.Adam(learning_rate=self.outer_lr)
        self.clip_value = clip_value
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm

        # MIGRATION: No placeholders or build_graph in TF2
        # Training will use @tf.function and GradientTape
        # TODO(runtime): Verify meta-learning parameter updates

    # MIGRATION: build_graph replaced with TF2 training functions
    @tf.function
    def train_step_per_task(self, task_idx, observations, actions, decoder_inputs, 
                           decoder_full_length, old_logits, old_v, advs, r):
        """Training step for a single task"""
        # Get the specific task policy
        task_policy = self.policy.meta_policies[task_idx]
        
        with tf.GradientTape() as tape:
            # Forward pass
            # TODO(runtime): Verify policy forward pass interface
            new_logits, vpred = task_policy.call_with_inputs(
                observations, decoder_inputs, decoder_full_length
            )
            
            # PPO loss computation
            likelihood_ratio = self.policy.distribution.likelihood_ratio_sym(
                actions, old_logits, new_logits
            )
            
            clipped_obj = tf.minimum(
                likelihood_ratio * advs,
                tf.clip_by_value(likelihood_ratio, 1.0 - self.clip_value, 1.0 + self.clip_value) * advs
            )
            surr_obj = -tf.reduce_mean(clipped_obj)
            
            # Value loss with clipping
            vpredclipped = old_v + tf.clip_by_value(vpred - old_v, -self.clip_value, self.clip_value)
            vf_losses1 = tf.square(vpred - r)
            vf_losses2 = tf.square(vpredclipped - r)
            vf_loss = 0.5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))
            
            total_loss = surr_obj + self.vf_coef * vf_loss
        
        # Compute and apply gradients
        params = task_policy.network.get_trainable_variables()
        grads = tape.gradient(total_loss, params)
        
        if self.max_grad_norm is not None:
            grads, _grad_norm = tf.clip_by_global_norm(grads, self.max_grad_norm)
        
        self.inner_optimizer.apply_gradients(zip(grads, params))
        
        return vf_loss, surr_obj
    
    # MIGRATION: Removed build_graph_legacy method - now using eager execution
    # All placeholder-based graph building replaced with tensor-based method calls in UpdatePPOTargetPerTask

    def UpdateMetaPolicy(self):
        # EAGER: Meta-policy update using first-order approximation
        # TODO(runtime): Verify meta-learning gradient computation
        
        # Collect gradients from all tasks
        accumulated_grads = []
        core_params = self.policy.core_policy.get_trainable_variables()
        
        # Initialize gradient accumulators
        for param in core_params:
            accumulated_grads.append(tf.zeros_like(param))
        
        # Compute meta-gradients from each task
        for task_idx in range(self.meta_batch_size):
            task_params = self.policy.meta_policies[task_idx].get_trainable_variables()
            
            # First-order approximation: (theta_core - theta_task) / (alpha * K * M)
            for i, (core_var, task_var) in enumerate(zip(core_params, task_params)):
                grad = (core_var - task_var) / self.inner_lr / self.num_inner_grad_steps / self.meta_batch_size / self.update_numbers
                accumulated_grads[i] = accumulated_grads[i] + grad
        
        # Apply accumulated gradients to core policy
        self.outer_optimizer.apply_gradients(zip(accumulated_grads, core_params))
        
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
        # EAGER: Refactored to use TF2 training step
        policy_losses = []
        value_losses = []

        batch_number = int(task_samples['observations'].shape[0] / batch_size)
        self.update_numbers = batch_number

        shift_actions = np.column_stack(
                    (np.zeros(task_samples['actions'].shape[0], dtype=np.int32), task_samples['actions'][:, 0:-1]))

        observations_batchs = np.split(np.array(task_samples['observations']), batch_number)
        actions_batchs = np.split(np.array(task_samples['actions']), batch_number)
        shift_action_batchs = np.split(np.array(shift_actions), batch_number)

        old_logits_batchs = np.split(np.array(task_samples["logits"], dtype=np.float32 ), batch_number)
        advs_batchs = np.split(np.array(task_samples['advantages'], dtype=np.float32), batch_number)
        oldvpred = np.split(np.array(task_samples['values'], dtype=np.float32), batch_number)
        returns = np.split(np.array(task_samples['returns'], dtype=np.float32), batch_number)

        vf_loss = 0.0
        pg_loss = 0.0
        
        for i in range(self.num_inner_grad_steps):
            for old_logits, old_v, observations, actions, shift_actions, advs, r in zip(
                    old_logits_batchs, oldvpred, observations_batchs, actions_batchs,
                    shift_action_batchs, advs_batchs, returns):
                decoder_full_length = np.array([observations.shape[1]] * observations.shape[0], dtype=np.int32)

                # EAGER: Use TF2 training step
                value_loss, policy_loss = self.train_step_per_task(
                    task_id, observations, actions, shift_actions,
                    decoder_full_length, old_logits, old_v, advs, r
                )
                
                # TODO(runtime): Add debug logging if needed
                
                vf_loss += value_loss.numpy()
                pg_loss += policy_loss.numpy()

            vf_loss = vf_loss / float(self.num_inner_grad_steps)
            pg_loss = pg_loss / float(self.num_inner_grad_steps)

            value_losses.append(vf_loss)
            policy_losses.append(pg_loss)

        return policy_losses, value_losses
