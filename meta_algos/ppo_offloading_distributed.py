# Distributed PPO for TF2/eager execution with multi-GPU support

import tensorflow as tf
import numpy as np
from utils.mpi_adam_optimizer import MpiAdamOptimizer
from mpi4py import MPI
from policies.meta_seq2seq_policy import Seq2SeqPolicy
import itertools
from utils.gpu import log_tensor_device


class PPODistributed():
    """
    Provides PPO based offloading training with distributed multi-GPU support
    """
    def __init__(self,
                 policy,
                 meta_sampler,
                 meta_sampler_process,
                 lr=1e-4,
                 num_inner_grad_steps=4,
                 clip_value=0.2,
                 vf_coef=0.5,
                 max_grad_norm=0.5,
                 strategy=None):
        self.lr = lr
        self.num_inner_grad_steps = num_inner_grad_steps
        self.policy = policy
        self.meta_sampler = meta_sampler
        self.meta_sampler_process = meta_sampler_process
        self.strategy = strategy
        
        # Create optimizer within strategy scope
        if self.strategy:
            with self.strategy.scope():
                self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr, epsilon=1e-5)
        else:
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr, epsilon=1e-5)
            
        self.clip_value = clip_value
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm

    @tf.function
    def train_step(self, observations, actions, decoder_inputs, decoder_full_length, 
                   old_logits, old_v, advs, r):
        """Single PPO training step using GradientTape
        
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
            # Forward pass through policy
            new_logits, vpred = self.policy.call_with_inputs(
                observations, decoder_inputs, decoder_full_length
            )
            
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
    
    def distributed_train_step(self, dataset_inputs):
        """Distributed training step that runs across multiple devices"""
        per_replica_losses = self.strategy.run(self.train_step, args=dataset_inputs)
        
        # Reduce losses across replicas
        vf_loss = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses[0], axis=None)
        surr_obj = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses[1], axis=None)
        
        return vf_loss, surr_obj

    def UpdatePPOTarget(self, task_samples, batch_size=50):
        """Update PPO target with distributed training support"""
        policy_losses = []
        value_losses = []

        batch_number = int(task_samples['observations'].shape[0] / batch_size)
        
        # Log device info on first batch
        if hasattr(self, '_first_batch'):
            self._first_batch = False
            if self.strategy and self.strategy.num_replicas_in_sync > 1:
                print(f"[Step 0] PPO training using {self.strategy.num_replicas_in_sync} devices")
        
        for batch_id in range(batch_number):
            batch_indexs = np.arange(batch_id * batch_size, (batch_id + 1) * batch_size)
            
            # Prepare batch data
            batch_obs = task_samples['observations'][batch_indexs]
            batch_actions = task_samples['actions'][batch_indexs]
            batch_decoder_inputs = task_samples['decoder_inputs'][batch_indexs]
            batch_decoder_full_length = task_samples['decoder_full_length'][batch_indexs]
            batch_old_logits = task_samples['logits'][batch_indexs]
            batch_old_v = task_samples["values"][batch_indexs]
            batch_advs = task_samples["advantages"][batch_indexs]
            batch_returns = task_samples["returns"][batch_indexs]
            
            if self.strategy and self.strategy.num_replicas_in_sync > 1:
                # Create dataset for distributed training
                dataset = tf.data.Dataset.from_tensor_slices((
                    batch_obs, batch_actions, batch_decoder_inputs, batch_decoder_full_length,
                    batch_old_logits, batch_old_v, batch_advs, batch_returns
                ))
                
                # Batch for each replica
                per_replica_batch_size = batch_size // self.strategy.num_replicas_in_sync
                dataset = dataset.batch(per_replica_batch_size)
                
                # Distribute dataset
                dist_dataset = self.strategy.experimental_distribute_dataset(dataset)
                
                # Run distributed training
                for step, inputs in enumerate(dist_dataset):
                    vf_loss, surr_loss = self.distributed_train_step(inputs)
                    policy_losses.append(surr_loss.numpy())
                    value_losses.append(vf_loss.numpy())
            else:
                # Single device training
                vf_loss, surr_loss = self.train_step(
                    batch_obs, batch_actions, batch_decoder_inputs, batch_decoder_full_length,
                    batch_old_logits, batch_old_v, batch_advs, batch_returns
                )
                policy_losses.append(surr_loss.numpy())
                value_losses.append(vf_loss.numpy())

        return policy_losses, value_losses