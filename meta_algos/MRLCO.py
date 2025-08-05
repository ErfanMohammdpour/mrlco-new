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
        self.inner_optimizer = tf.keras.optimizers.Adam(learning_rate=self.inner_lr)
        self.outer_optimizer = tf.keras.optimizers.Adam(learning_rate=self.outer_lr)
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

        self.old_v =[]
        self.advs = []
        self.r = []

        self.surr_obj = []
        self.vf_loss = []
        self.likelihood_ratio = []
        self.clipped_obj = []
        self.total_loss = []

        self.build_graph()

    def build_graph(self):
        # TF2: No need to build graph - we'll use eager execution
        pass

    @tf.function
    def update_task_gradients(self, task_id, old_logits, old_v, observations, actions, decoder_inputs, decoder_full_length, advs, r):
        """TF2 training step for a single task using GradientTape"""
        with tf.GradientTape() as tape:
            # Forward pass through the policy network
            new_logits, vpred = self.policy.meta_policies[task_id].forward_train(
                observations, decoder_inputs, actions, decoder_full_length
            )
            
            # Calculate PPO loss components
            likelihood_ratio = self.policy.distribution.likelihood_ratio_sym(actions, old_logits, new_logits)
            
            clipped_obj = tf.minimum(
                likelihood_ratio * advs,
                tf.clip_by_value(likelihood_ratio, 1.0 - self.clip_value, 1.0 + self.clip_value) * advs
            )
            surr_obj = -tf.reduce_mean(clipped_obj)
            
            # Value function loss with clipping
            vpredclipped = old_v + tf.clip_by_value(vpred - old_v, -self.clip_value, self.clip_value)
            vf_losses1 = tf.square(vpred - r)
            vf_losses2 = tf.square(vpredclipped - r)
            vf_loss = 0.5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))
            
            total_loss = surr_obj + self.vf_coef * vf_loss
        
        # Compute gradients
        params = self.policy.meta_policies[task_id].network.get_trainable_variables()
        grads = tape.gradient(total_loss, params)
        
        if self.max_grad_norm is not None:
            grads, _grad_norm = tf.clip_by_global_norm(grads, self.max_grad_norm)
        
        # Apply gradients
        self.inner_optimizer.apply_gradients(zip(grads, params))
        
        return vf_loss, surr_obj, likelihood_ratio, advs, clipped_obj

    def UpdateMetaPolicy(self):
        # get the parameters value of the policy network
        core_params = [v.numpy() for v in self.policy.core_policy.get_trainable_variables()]
        
        # Calculate meta gradients by averaging differences across tasks
        meta_grads = []
        for i, core_param in enumerate(core_params):
            task_grads = []
            
            for task_id in range(self.meta_batch_size):
                task_params = [v.numpy() for v in self.policy.meta_policies[task_id].get_trainable_variables()]
                if i < len(task_params) and task_params[i].shape == core_param.shape:
                    # Calculate gradient as difference between core and task parameters
                    grad = (core_param - task_params[i]) / (self.inner_lr * self.num_inner_grad_steps * self.meta_batch_size * self.update_numbers)
                    task_grads.append(grad)
            
            if task_grads:
                # Average gradients across tasks
                avg_grad = np.mean(task_grads, axis=0)
                meta_grads.append(avg_grad)
            else:
                meta_grads.append(np.zeros_like(core_param))
        
        # Apply meta gradients
        core_vars = self.policy.core_policy.get_trainable_variables()
        grads_and_vars = list(zip(meta_grads, core_vars))
        self.outer_optimizer.apply_gradients(grads_and_vars)

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
            for old_logits, old_v, observations, actions, shift_actions, advs, r in zip(old_logits_batchs, oldvpred, observations_batchs, actions_batchs,
                                                                                        shift_action_batchs, advs_batchs, returns):
                decoder_full_length = np.array([observations.shape[1]] * observations.shape[0], dtype=np.int32)

                # Convert numpy arrays to tensors
                old_logits_tensor = tf.convert_to_tensor(old_logits, dtype=tf.float32)
                old_v_tensor = tf.convert_to_tensor(old_v, dtype=tf.float32)  
                observations_tensor = tf.convert_to_tensor(observations, dtype=tf.float32)
                actions_tensor = tf.convert_to_tensor(actions, dtype=tf.int32)
                shift_actions_tensor = tf.convert_to_tensor(shift_actions, dtype=tf.int32)
                decoder_full_length_tensor = tf.convert_to_tensor(decoder_full_length, dtype=tf.int32)
                advs_tensor = tf.convert_to_tensor(advs, dtype=tf.float32)
                r_tensor = tf.convert_to_tensor(r, dtype=tf.float32)

                # Call TF2 training function
                value_loss, policy_loss, likelihood_ratio_val, advs_val, clipped_obj_val = self.update_task_gradients(
                    task_id, old_logits_tensor, old_v_tensor, observations_tensor, actions_tensor, 
                    shift_actions_tensor, decoder_full_length_tensor, advs_tensor, r_tensor
                )
                
                # Debug logging
                if i == 0 and task_id == 0:  # Log only for first iteration and task
                    print(f"\n[DEBUG] Loss calculation details:")
                    print(f"  Policy loss (surr_obj): {policy_loss.numpy()}")
                    print(f"  Value loss: {value_loss.numpy()}")
                    print(f"  Likelihood ratio mean: {np.mean(likelihood_ratio_val.numpy())}")
                    print(f"  Likelihood ratio std: {np.std(likelihood_ratio_val.numpy())}")
                    print(f"  Advantages mean: {np.mean(advs_val.numpy())}")
                    print(f"  Advantages std: {np.std(advs_val.numpy())}")
                    print(f"  Clipped objective mean: {np.mean(clipped_obj_val.numpy())}")

                vf_loss += value_loss.numpy()
                pg_loss += policy_loss.numpy()

            vf_loss = vf_loss / float(self.num_inner_grad_steps)
            pg_loss = pg_loss / float(self.num_inner_grad_steps)

            value_losses.append(vf_loss)
            policy_losses.append(pg_loss)

        return policy_losses, value_losses