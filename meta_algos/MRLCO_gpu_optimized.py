import tensorflow as tf
import numpy as np
from tensorflow.keras import mixed_precision

class MRLCOGPUOptimized:
    """GPU-optimized MRLCO with proper tf.function and strategy support"""
    
    def __init__(self, policy, meta_batch_size, meta_sampler, meta_sampler_process,
                 outer_lr=1e-4, inner_lr=0.1, num_inner_grad_steps=4,
                 clip_value=0.2, vf_coef=0.5, max_grad_norm=0.5, strategy=None):
        self.outer_lr = outer_lr
        self.inner_lr = inner_lr
        self.num_inner_grad_steps = num_inner_grad_steps
        self.policy = policy
        self.meta_sampler = meta_sampler
        self.meta_sampler_process = meta_sampler_process
        self.meta_batch_size = meta_batch_size
        self.update_numbers = 1
        self.clip_value = clip_value
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        
        # Use provided strategy or create one
        if strategy is None:
            gpus = tf.config.list_physical_devices('GPU')
            if len(gpus) >= 2:
                self.strategy = tf.distribute.MirroredStrategy()
            elif len(gpus) == 1:
                self.strategy = tf.distribute.OneDeviceStrategy("/GPU:0")
            else:
                self.strategy = tf.distribute.OneDeviceStrategy("/CPU:0")
        else:
            self.strategy = strategy
        
        print(f"MRLCOGPUOptimized using strategy: {self.strategy}")
        
        # Create optimizers within strategy scope
        with self.strategy.scope():
            # Inner optimizers for each task
            self.inner_optimizers = []
            for i in range(meta_batch_size):
                optimizer = tf.keras.optimizers.Adam(learning_rate=self.inner_lr, epsilon=1e-5)
                # Wrap with mixed precision loss scale optimizer
                optimizer = mixed_precision.LossScaleOptimizer(optimizer)
                self.inner_optimizers.append(optimizer)
            
            # Outer optimizer for meta-policy
            self.outer_optimizer = tf.keras.optimizers.Adam(learning_rate=self.outer_lr, epsilon=1e-5)
            self.outer_optimizer = mixed_precision.LossScaleOptimizer(self.outer_optimizer)
    
    @tf.function
    def compute_ppo_loss(self, new_logits, vpred, actions, old_logits, old_v, advs, r):
        """Compute PPO loss components"""
        # Cast to float32 for stability
        new_logits = tf.cast(new_logits, tf.float32)
        vpred = tf.cast(vpred, tf.float32)
        old_logits = tf.cast(old_logits, tf.float32)
        old_v = tf.cast(old_v, tf.float32)
        advs = tf.cast(advs, tf.float32)
        r = tf.cast(r, tf.float32)
        
        # Compute likelihood ratio
        old_neg_log_p = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=actions, logits=old_logits
        )
        new_neg_log_p = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=actions, logits=new_logits
        )
        likelihood_ratio = tf.exp(old_neg_log_p - new_neg_log_p)
        
        # Clipped surrogate objective
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
        
        return total_loss, surr_obj, vf_loss
    
    @tf.function
    def distributed_train_step(self, batch_data, task_id):
        """Single training step executed via strategy.run"""
        def step_fn(inputs):
            observations = inputs['observations']
            actions = inputs['actions']
            decoder_inputs = inputs['decoder_inputs']
            decoder_full_length = inputs['decoder_full_length']
            old_logits = inputs['logits']
            old_v = inputs['values']
            advs = inputs['advantages']
            r = inputs['returns']
            
            with tf.GradientTape() as tape:
                # Forward pass through policy
                # Get current policy for this task
                task_policy = self.policy.meta_policies[task_id]
                
                # Run model forward pass
                # Note: This assumes the policy has been properly set up
                new_logits = task_policy.network.decoder_logits
                vpred = task_policy.vf
                
                # Compute loss
                total_loss, surr_obj, vf_loss = self.compute_ppo_loss(
                    new_logits, vpred, actions, old_logits, old_v, advs, r
                )
                
                # Scale loss for mixed precision
                scaled_loss = self.inner_optimizers[task_id].get_scaled_loss(total_loss)
            
            # Compute gradients
            trainable_vars = task_policy.network.get_trainable_variables()
            scaled_gradients = tape.gradient(scaled_loss, trainable_vars)
            gradients = self.inner_optimizers[task_id].get_unscaled_gradients(scaled_gradients)
            
            # Clip gradients
            if self.max_grad_norm is not None:
                gradients, _ = tf.clip_by_global_norm(gradients, self.max_grad_norm)
            
            # Apply gradients
            self.inner_optimizers[task_id].apply_gradients(zip(gradients, trainable_vars))
            
            return surr_obj, vf_loss
        
        # Execute step function on strategy
        per_replica_losses = self.strategy.run(step_fn, args=(batch_data,))
        
        # Reduce losses across replicas
        surr_loss = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses[0], axis=None)
        vf_loss = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses[1], axis=None)
        
        return surr_loss, vf_loss
    
    def create_dataset(self, samples_data, batch_size):
        """Create optimized tf.data.Dataset from samples"""
        # Convert numpy arrays to tensors
        dataset = tf.data.Dataset.from_tensor_slices({
            'observations': tf.constant(samples_data['observations'], dtype=tf.float32),
            'actions': tf.constant(samples_data['actions'], dtype=tf.int32),
            'decoder_inputs': tf.constant(samples_data['decoder_inputs'], dtype=tf.int32),
            'decoder_full_length': tf.constant(samples_data['decoder_full_length'], dtype=tf.int32),
            'logits': tf.constant(samples_data['logits'], dtype=tf.float32),
            'values': tf.constant(samples_data['values'], dtype=tf.float32),
            'advantages': tf.constant(samples_data['advantages'], dtype=tf.float32),
            'returns': tf.constant(samples_data['returns'], dtype=tf.float32)
        })
        
        # Apply optimizations
        dataset = dataset.cache()
        dataset = dataset.shuffle(buffer_size=min(1000, len(samples_data['observations'])))
        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        # Distribute dataset across replicas
        dataset = self.strategy.experimental_distribute_dataset(dataset)
        
        return dataset
    
    def UpdatePPOTarget(self, task_samples, batch_size=50, sess=None):
        """Update PPO targets for all tasks with GPU optimization"""
        total_policy_losses = []
        total_value_losses = []
        
        with self.strategy.scope():
            for task_id in range(self.meta_batch_size):
                # Create dataset for this task
                dataset = self.create_dataset(task_samples[task_id], batch_size)
                
                task_policy_losses = []
                task_value_losses = []
                
                # Train for multiple epochs
                for epoch in range(self.num_inner_grad_steps):
                    epoch_policy_losses = []
                    epoch_value_losses = []
                    
                    # Iterate through batches
                    for batch in dataset:
                        surr_loss, vf_loss = self.distributed_train_step(batch, task_id)
                        epoch_policy_losses.append(surr_loss.numpy())
                        epoch_value_losses.append(vf_loss.numpy())
                    
                    task_policy_losses.extend(epoch_policy_losses)
                    task_value_losses.extend(epoch_value_losses)
                
                total_policy_losses.append(task_policy_losses)
                total_value_losses.append(task_value_losses)
        
        return total_policy_losses, total_value_losses
    
    @tf.function
    def compute_meta_gradients(self):
        """Compute meta-gradients using first-order approximation"""
        # Get core policy parameters
        core_params = self.policy.core_policy.network.get_trainable_variables()
        
        # Initialize gradient accumulators
        accumulated_grads = [tf.zeros_like(param) for param in core_params]
        
        # Compute meta-gradients from each task
        for task_idx in range(self.meta_batch_size):
            task_params = self.policy.meta_policies[task_idx].network.get_trainable_variables()
            
            # First-order approximation: (theta_core - theta_task) / (alpha * K * M)
            for i, (core_var, task_var) in enumerate(zip(core_params, task_params)):
                grad = (core_var - task_var) / (self.inner_lr * self.num_inner_grad_steps * 
                                               self.meta_batch_size * self.update_numbers)
                accumulated_grads[i] = accumulated_grads[i] + grad
        
        return accumulated_grads, core_params
    
    def UpdateMetaPolicy(self, sess=None):
        """Update meta-policy with GPU optimization"""
        with self.strategy.scope():
            # Compute meta-gradients
            grads, params = self.compute_meta_gradients()
            
            # Apply gradients
            self.outer_optimizer.apply_gradients(zip(grads, params))
        
        # Sync parameters
        print("Syncing core policy to meta-policies...")
        self.policy.async_parameters()
    
    def log_gpu_memory(self):
        """Log current GPU memory usage"""
        try:
            gpus = tf.config.list_physical_devices('GPU')
            for gpu in gpus:
                # This requires TF 2.3+
                memory_info = tf.config.experimental.get_memory_info(gpu.name)
                current_mb = memory_info['current'] / 1024 / 1024
                peak_mb = memory_info['peak'] / 1024 / 1024
                print(f"{gpu.name} - Current: {current_mb:.1f}MB, Peak: {peak_mb:.1f}MB")
        except:
            # Fallback for older TF versions
            pass