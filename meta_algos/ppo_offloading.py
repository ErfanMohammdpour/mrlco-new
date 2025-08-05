# TF2.19 version of PPO offloading

import tensorflow as tf
import numpy as np
from utils.mpi_adam_optimizer import MpiAdamOptimizer
from policies.meta_seq2seq_policy import Seq2SeqPolicy
import itertools

class PPO():
    """
    Provides ppo based offloading training - TF2.19 compatible version
    """
    def __init__(self,
                 policy,
                 meta_sampler,
                 meta_sampler_process,
                 lr=1e-4,
                 num_inner_grad_steps=4,
                 clip_value = 0.2,
                 vf_coef=0.5,
                 max_grad_norm=0.5):
        self.lr = lr
        self.num_inner_grad_steps=num_inner_grad_steps
        self.policy = policy
        self.meta_sampler = meta_sampler
        self.meta_sampler_process = meta_sampler_process

        #self.optimizer = MpiAdamOptimizer(MPI.COMM_WORLD, learning_rate=self.lr, epsilon=1e-5)
        # TF2.19: Use Keras optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr, epsilon=1e-5)
        self.clip_value = clip_value
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm

    @tf.function
    def train_step(self, observations, actions, decoder_inputs, decoder_full_length, 
                    old_logits, old_v, advs, r):
        """TF2.19 training step using GradientTape
        
        Args:
            observations: [batch, time, obs_dim]
            actions: [batch, time] decoder targets 
            decoder_inputs: [batch, time] shifted actions
            decoder_full_length: [batch] sequence lengths
            old_logits: [batch, time, action_dim] from previous policy
            old_v: [batch, time] old value predictions
            advs: [batch, time] advantages
            r: [batch, time] returns
        """
        with tf.GradientTape() as tape:
            # Update policy inputs
            self.policy.obs.assign(observations)
            self.policy.decoder_inputs.assign(decoder_inputs)
            self.policy.decoder_targets.assign(actions)
            self.policy.decoder_full_length.assign(decoder_full_length)
            
            # Forward pass through policy
            new_logits = self.policy.network.decoder_logits
            vpred = self.policy.vf
            
            # Compute PPO loss
            likelihood_ratio = self.policy.distribution.likelihood_ratio_sym(
                actions, old_logits, new_logits
            )
            
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
        
        # Compute and apply gradients
        params = self.policy.network.get_trainable_variables()
        grads = tape.gradient(total_loss, params)
        
        if self.max_grad_norm is not None:
            grads, _grad_norm = tf.clip_by_global_norm(grads, self.max_grad_norm)
        
        self.optimizer.apply_gradients(zip(grads, params))
        
        return vf_loss, surr_obj

    def UpdatePPOTarget(self, task_samples, batch_size=50):
        # TF2.19: Refactored to use TF2 training step
        policy_losses = []
        value_losses = []

        batch_number = int(task_samples['observations'].shape[0] / batch_size)

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
            for batch_idx, (old_logits, old_v, observations, actions, shift_actions, advs, r) in enumerate(zip(
                    old_logits_batchs, oldvpred, observations_batchs, actions_batchs,
                    shift_action_batchs, advs_batchs, returns)):
                decoder_full_length = np.array([observations.shape[1]] * observations.shape[0], dtype=np.int32)

                # Convert to tensors
                observations_tensor = tf.convert_to_tensor(observations, dtype=tf.float32)
                actions_tensor = tf.convert_to_tensor(actions, dtype=tf.int32)
                shift_actions_tensor = tf.convert_to_tensor(shift_actions, dtype=tf.int32)
                decoder_full_length_tensor = tf.convert_to_tensor(decoder_full_length, dtype=tf.int32)
                old_logits_tensor = tf.convert_to_tensor(old_logits, dtype=tf.float32)
                old_v_tensor = tf.convert_to_tensor(old_v, dtype=tf.float32)
                advs_tensor = tf.convert_to_tensor(advs, dtype=tf.float32)
                r_tensor = tf.convert_to_tensor(r, dtype=tf.float32)

                # TF2.19: Call TF2 training step
                value_loss, policy_loss = self.train_step(
                    observations_tensor, actions_tensor, shift_actions_tensor, decoder_full_length_tensor,
                    old_logits_tensor, old_v_tensor, advs_tensor, r_tensor
                )

                vf_loss += value_loss.numpy()
                pg_loss += policy_loss.numpy()
                
                # Log device placement on first iteration
                if i == 0 and batch_idx == 0:
                    print(f"\n[DEBUG] PPO Loss calculation details:")
                    print(f"  Policy loss: {policy_loss.numpy()}")
                    print(f"  Value loss: {value_loss.numpy()}")

            vf_loss = vf_loss / float(self.num_inner_grad_steps)
            pg_loss = pg_loss / float(self.num_inner_grad_steps)

            value_losses.append(vf_loss)
            policy_losses.append(pg_loss)

        return policy_losses, value_losses