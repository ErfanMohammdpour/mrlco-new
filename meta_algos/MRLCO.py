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

        # TF2: Create task-specific optimizers
        self.task_optimizers = []
        for i in range(meta_batch_size):
            opt = tf.keras.optimizers.Adam(learning_rate=self.inner_lr)
            self.task_optimizers.append(opt)
        
        self.outer_optimizer = tf.keras.optimizers.Adam(learning_rate=self.outer_lr)
        self.clip_value = clip_value
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm

    def UpdateMetaPolicy(self):
        """Update meta policy using first-order MAML approximation"""
        # Get parameters of core policy
        core_params = self.policy.core_policy.get_trainable_variables()
        
        # Calculate meta gradients
        meta_grads = []
        for i, core_var in enumerate(core_params):
            grad_sum = None
            
            for task_id in range(self.meta_batch_size):
                task_params = self.policy.meta_policies[task_id].get_trainable_variables()
                if i < len(task_params):
                    # Calculate gradient as difference between core and task parameters
                    grad = (core_var.numpy() - task_params[i].numpy()) / (
                        self.inner_lr * self.num_inner_grad_steps * self.meta_batch_size * self.update_numbers
                    )
                    
                    if grad_sum is None:
                        grad_sum = grad
                    else:
                        grad_sum += grad
            
            if grad_sum is not None:
                meta_grads.append(grad_sum)
            else:
                meta_grads.append(np.zeros_like(core_var.numpy()))
        
        # Apply gradients to core policy
        self.outer_optimizer.apply_gradients(zip(meta_grads, core_params))
        
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

    @tf.function
    def _train_step(self, task_id, old_logits, old_v, observations, actions, decoder_inputs, decoder_full_length, advs, r):
        """Single training step for a task using TF2 GradientTape
        
        Expected shapes:
        - old_logits: [batch, sequence, vocab_size]
        - old_v: [batch, sequence]
        - observations: [batch, sequence, obs_dim]
        - actions: [batch, sequence]
        - advs: [batch, sequence]
        - r: [batch, sequence]
        """
        with tf.GradientTape() as tape:
            # Get logits and value predictions from policy
            new_logits, vpred = self.policy.meta_policies[task_id].forward_train(
                observations, decoder_inputs, actions, decoder_full_length
            )
            
            # Calculate likelihood ratio for PPO
            # Both old_logits and new_logits have shape [batch, sequence, vocab_size]
            # actions has shape [batch, sequence]
            # likelihood_ratio will have shape [batch, sequence]
            likelihood_ratio = self.policy.distribution.likelihood_ratio_sym(
                actions, old_logits, new_logits
            )
            
            # PPO clipped objective with per-timestep advantages
            # likelihood_ratio shape: [batch, sequence]
            # advs shape: [batch, sequence]
            clipped_obj = tf.minimum(
                likelihood_ratio * advs,
                tf.clip_by_value(likelihood_ratio, 1.0 - self.clip_value, 1.0 + self.clip_value) * advs
            )
            surr_obj = -tf.reduce_mean(clipped_obj)
            
            # Value function loss with clipping
            # vpred shape: [batch, sequence]
            # old_v shape: [batch, sequence]
            # r shape: [batch, sequence]
            vpredclipped = old_v + tf.clip_by_value(vpred - old_v, -self.clip_value, self.clip_value)
            vf_losses1 = tf.square(vpred - r)
            vf_losses2 = tf.square(vpredclipped - r)
            vf_loss = 0.5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))
            
            # Total loss
            total_loss = surr_obj + self.vf_coef * vf_loss
        
        # Get trainable variables
        params = self.policy.meta_policies[task_id].get_trainable_variables()
        
        # Compute gradients
        grads = tape.gradient(total_loss, params)
        
        # Clip gradients if specified
        if self.max_grad_norm is not None:
            grads, _grad_norm = tf.clip_by_global_norm(grads, self.max_grad_norm)
        
        # Apply gradients using task-specific optimizer
        self.task_optimizers[task_id].apply_gradients(zip(grads, params))
        
        # Return scalar losses and mean of likelihood ratio for logging
        return vf_loss, surr_obj, tf.reduce_mean(likelihood_ratio), clipped_obj

    def UpdatePPOTargetPerTask(self, task_samples, task_id, batch_size=50):
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
        
        # Build task optimizer with policy variables if not already done
        policy_vars = self.policy.meta_policies[task_id].get_trainable_variables()
        if not hasattr(self.task_optimizers[task_id], '_built'):
            self.task_optimizers[task_id].build(policy_vars)
            self.task_optimizers[task_id]._built = True

        vf_loss = 0.0
        pg_loss = 0.0
        
        for i in range(self.num_inner_grad_steps):
            for old_logits, old_v, observations, actions, shift_actions, advs, r in zip(
                    old_logits_batchs, oldvpred, observations_batchs, actions_batchs,
                    shift_action_batchs, advs_batchs, returns):
                
                decoder_full_length = np.array([observations.shape[1]] * observations.shape[0], dtype=np.int32)
                
                # Convert to tensors
                old_logits_tf = tf.convert_to_tensor(old_logits, dtype=tf.float32)
                old_v_tf = tf.convert_to_tensor(old_v, dtype=tf.float32)
                observations_tf = tf.convert_to_tensor(observations, dtype=tf.float32)
                actions_tf = tf.convert_to_tensor(actions, dtype=tf.int32)
                decoder_inputs_tf = tf.convert_to_tensor(shift_actions, dtype=tf.int32)
                decoder_full_length_tf = tf.convert_to_tensor(decoder_full_length, dtype=tf.int32)
                advs_tf = tf.convert_to_tensor(advs, dtype=tf.float32)
                r_tf = tf.convert_to_tensor(r, dtype=tf.float32)
                
                # Run training step
                value_loss, policy_loss, likelihood_ratio_val, clipped_obj_val = self._train_step(
                    task_id, old_logits_tf, old_v_tf, observations_tf, actions_tf,
                    decoder_inputs_tf, decoder_full_length_tf, advs_tf, r_tf
                )
                
                # Debug logging for first iteration
                if i == 0 and task_id == 0:
                    print(f"\n[DEBUG] Loss calculation details:")
                    print(f"  Policy loss (surr_obj): {policy_loss.numpy()}")
                    print(f"  Value loss: {value_loss.numpy()}")
                    print(f"  Likelihood ratio mean: {np.mean(likelihood_ratio_val.numpy())}")
                    print(f"  Likelihood ratio std: {np.std(likelihood_ratio_val.numpy())}")
                    print(f"  Advantages mean: {np.mean(advs)}")
                    print(f"  Advantages std: {np.std(advs)}")
                    print(f"  Clipped objective mean: {np.mean(clipped_obj_val.numpy())}")
                
                vf_loss += value_loss.numpy()
                pg_loss += policy_loss.numpy()

            vf_loss = vf_loss / float(self.num_inner_grad_steps)
            pg_loss = pg_loss / float(self.num_inner_grad_steps)

            value_losses.append(vf_loss)
            policy_losses.append(pg_loss)

        return policy_losses, value_losses