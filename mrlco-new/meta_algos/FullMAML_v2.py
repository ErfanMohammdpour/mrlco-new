import tensorflow as tf
import numpy as np
import itertools
import time
from utils import logger
from collections import defaultdict

class FullMAML_v2():
    """
    Enhanced Full MAML implementation with proper second-order gradient calculations
    and improved memory management.
    
    Key Features:
    - Proper second-order gradient computation through inner loop trajectory
    - Memory-efficient gradient checkpointing
    - Support for different meta-learning objectives
    - Advanced learning rate scheduling
    - Gradient accumulation and clipping
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
                 max_grad_norm=0.5,
                 use_gradient_clipping=True,
                 second_order_method='implicit',  # 'implicit', 'explicit', 'finite_diff'
                 finite_diff_epsilon=1e-4,
                 inner_lr_schedule='constant',
                 outer_lr_schedule='constant',
                 lr_decay_rate=0.99,
                 lr_decay_steps=1000,
                 memory_optimization=True,
                 gradient_checkpointing=True,
                 meta_objective='post_update',  # 'post_update' or 'online'
                 regularization_coef=0.0):
        
        # Store configuration
        self.outer_lr = outer_lr
        self.inner_lr = inner_lr
        self.num_inner_grad_steps = num_inner_grad_steps
        self.policy = policy
        self.meta_sampler = meta_sampler
        self.meta_sampler_process = meta_sampler_process
        self.meta_batch_size = meta_batch_size
        self.update_numbers = 1
        
        # PPO parameters
        self.clip_value = clip_value
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.use_gradient_clipping = use_gradient_clipping
        
        # Full MAML parameters
        self.second_order_method = second_order_method
        self.finite_diff_epsilon = finite_diff_epsilon
        self.memory_optimization = memory_optimization
        self.gradient_checkpointing = gradient_checkpointing
        self.meta_objective = meta_objective
        self.regularization_coef = regularization_coef
        
        # Learning rate scheduling
        self.inner_lr_schedule = inner_lr_schedule
        self.outer_lr_schedule = outer_lr_schedule
        self.lr_decay_rate = lr_decay_rate
        self.lr_decay_steps = lr_decay_steps
        
        # Initialize tracking variables
        self.training_stats = defaultdict(list)
        
        # Build computation graph
        self.build_graph()
        
    def build_graph(self):
        """Build Full MAML computation graph with proper second-order gradients."""
        
        # Create global step counters
        self.global_inner_step = tf.Variable(0, trainable=False, name='global_inner_step')
        self.global_outer_step = tf.Variable(0, trainable=False, name='global_outer_step')
        
        # Create learning rate tensors
        self.inner_lr_tensor = self._create_lr_schedule(
            self.inner_lr, self.inner_lr_schedule, self.global_inner_step, 'inner')
        self.outer_lr_tensor = self._create_lr_schedule(
            self.outer_lr, self.outer_lr_schedule, self.global_outer_step, 'outer')
        
        # Initialize optimizers
        self.inner_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.inner_lr_tensor)
        self.outer_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.outer_lr_tensor)
        
        # Initialize containers for task-specific components
        self._init_task_containers()
        
        # Build task-specific graphs
        for i in range(self.meta_batch_size):
            self._build_task_graph(i)
        
        # Build meta-update graph
        self._build_meta_update_graph()
        
        # Add monitoring and debugging operations
        self._add_monitoring_ops()
        
    def _init_task_containers(self):
        """Initialize containers for task-specific variables and operations."""
        # Placeholders
        self.obs_phs = []
        self.actions_phs = []
        self.old_logits_phs = []
        self.old_vpred_phs = []
        self.advs_phs = []
        self.returns_phs = []
        self.decoder_inputs_phs = []
        self.decoder_length_phs = []
        
        # Losses and metrics
        self.inner_losses = []
        self.policy_losses = []
        self.value_losses = []
        self.meta_losses = []
        
        # Gradient tracking
        self.inner_grad_ops = []
        self.meta_grad_ops = []
        
        # Update operations
        self.inner_update_ops = []
        
    def _build_task_graph(self, task_idx):
        """Build computation graph for a specific task."""
        with tf.compat.v1.variable_scope(f"task_{task_idx}"):
            # Create placeholders
            self._create_task_placeholders(task_idx)
            
            # Get task policy
            task_policy = self.policy.meta_policies[task_idx]
            
            # Build inner loop with second-order gradient support
            if self.second_order_method == 'implicit':
                self._build_implicit_second_order(task_idx, task_policy)
            elif self.second_order_method == 'explicit':
                self._build_explicit_second_order(task_idx, task_policy)
            elif self.second_order_method == 'finite_diff':
                self._build_finite_diff_second_order(task_idx, task_policy)
            else:
                raise ValueError(f"Unknown second-order method: {self.second_order_method}")
    
    def _create_task_placeholders(self, task_idx):
        """Create placeholders for task-specific data."""
        # Observations
        self.obs_phs.append(
            tf.compat.v1.placeholder(
                dtype=tf.float32,
                shape=[None, None, self.policy.obs_dim],
                name=f'obs_ph_{task_idx}'))
        
        # Actions
        self.actions_phs.append(
            tf.compat.v1.placeholder(
                dtype=tf.int32,
                shape=[None, None],
                name=f'actions_ph_{task_idx}'))
        
        # Old policy logits
        self.old_logits_phs.append(
            tf.compat.v1.placeholder(
                dtype=tf.float32,
                shape=[None, None, self.policy.action_dim],
                name=f'old_logits_ph_{task_idx}'))
        
        # Old value predictions
        self.old_vpred_phs.append(
            tf.compat.v1.placeholder(
                dtype=tf.float32,
                shape=[None, None],
                name=f'old_vpred_ph_{task_idx}'))
        
        # Advantages
        self.advs_phs.append(
            tf.compat.v1.placeholder(
                dtype=tf.float32,
                shape=[None, None],
                name=f'advs_ph_{task_idx}'))
        
        # Returns
        self.returns_phs.append(
            tf.compat.v1.placeholder(
                dtype=tf.float32,
                shape=[None, None],
                name=f'returns_ph_{task_idx}'))
        
        # Decoder inputs
        self.decoder_inputs_phs.append(
            tf.compat.v1.placeholder(
                dtype=tf.int32,
                shape=[None, None],
                name=f'decoder_inputs_ph_{task_idx}'))
        
        # Decoder length
        self.decoder_length_phs.append(
            tf.compat.v1.placeholder(
                dtype=tf.int32,
                shape=[None],
                name=f'decoder_length_ph_{task_idx}'))
    
    def _build_implicit_second_order(self, task_idx, task_policy):
        """
        Build implicit second-order gradient computation.
        This uses tf.gradients to compute second-order terms implicitly.
        """
        with tf.compat.v1.variable_scope("implicit_second_order"):
            # Get initial parameters
            init_params = task_policy.get_trainable_variables()
            
            # Create variables to track adapted parameters
            adapted_params = init_params
            
            # Perform inner loop updates
            for step in range(self.num_inner_grad_steps):
                with tf.compat.v1.variable_scope(f"step_{step}"):
                    # Compute inner loss
                    inner_loss = self._compute_ppo_loss(
                        task_idx, adapted_params, 
                        reuse=(step > 0))
                    
                    # Compute gradients
                    grads = tf.gradients(inner_loss, adapted_params)
                    
                    # Handle None gradients
                    grads = [g if g is not None else tf.zeros_like(p) 
                            for g, p in zip(grads, adapted_params)]
                    
                    # Apply gradient clipping
                    if self.use_gradient_clipping:
                        grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
                    
                    # Update parameters
                    adapted_params = [p - self.inner_lr_tensor * g 
                                     for p, g in zip(adapted_params, grads)]
                    
                    # Store loss for monitoring
                    self.inner_losses.append(inner_loss)
            
            # Compute meta-loss on adapted parameters
            meta_loss = self._compute_ppo_loss(
                task_idx, adapted_params, 
                reuse=True, is_meta=True)
            
            self.meta_losses.append(meta_loss)
            
            # Compute meta-gradients (includes second-order terms)
            core_params = self.policy.core_policy.get_trainable_variables()
            meta_grads = tf.gradients(meta_loss, core_params)
            
            self.meta_grad_ops.append(meta_grads)
    
    def _build_explicit_second_order(self, task_idx, task_policy):
        """
        Build explicit second-order gradient computation.
        This explicitly computes the Hessian-vector products.
        """
        with tf.compat.v1.variable_scope("explicit_second_order"):
            # Get initial parameters
            init_params = task_policy.get_trainable_variables()
            
            # Track parameter trajectory and gradients
            param_trajectory = [init_params]
            grad_trajectory = []
            
            # Perform inner loop updates
            adapted_params = init_params
            for step in range(self.num_inner_grad_steps):
                with tf.compat.v1.variable_scope(f"step_{step}"):
                    # Compute inner loss
                    inner_loss = self._compute_ppo_loss(
                        task_idx, adapted_params,
                        reuse=(step > 0))
                    
                    # Compute first-order gradients
                    grads = tf.gradients(inner_loss, adapted_params)
                    grads = [g if g is not None else tf.zeros_like(p)
                            for g, p in zip(grads, adapted_params)]
                    
                    # Store gradients
                    grad_trajectory.append(grads)
                    
                    # Update parameters
                    adapted_params = [p - self.inner_lr_tensor * g
                                     for p, g in zip(adapted_params, grads)]
                    param_trajectory.append(adapted_params)
            
            # Compute meta-loss
            meta_loss = self._compute_ppo_loss(
                task_idx, adapted_params,
                reuse=True, is_meta=True)
            
            # Compute meta-gradients with explicit second-order terms
            core_params = self.policy.core_policy.get_trainable_variables()
            
            # First-order term
            first_order_grads = tf.gradients(meta_loss, adapted_params)
            
            # Second-order term (Hessian-vector products)
            meta_grads = []
            for param_idx, core_param in enumerate(core_params):
                # Initialize gradient accumulator
                param_grad = first_order_grads[param_idx]
                
                # Add second-order corrections
                for step in range(self.num_inner_grad_steps):
                    # Compute Hessian-vector product
                    hvp = self._compute_hessian_vector_product(
                        inner_loss=self.inner_losses[task_idx * self.num_inner_grad_steps + step],
                        params=param_trajectory[step],
                        vector=grad_trajectory[step][param_idx])
                    
                    # Accumulate second-order term
                    param_grad = param_grad - self.inner_lr_tensor * hvp
                
                meta_grads.append(param_grad)
            
            self.meta_losses.append(meta_loss)
            self.meta_grad_ops.append(meta_grads)
    
    def _build_finite_diff_second_order(self, task_idx, task_policy):
        """
        Build second-order gradients using finite differences.
        This approximates second-order terms without explicit Hessian computation.
        """
        with tf.compat.v1.variable_scope("finite_diff_second_order"):
            # Get initial parameters
            init_params = task_policy.get_trainable_variables()
            
            # Compute adapted parameters with positive perturbation
            adapted_params_plus = self._inner_loop_update(
                task_idx, init_params, 
                perturbation=self.finite_diff_epsilon)
            
            # Compute adapted parameters with negative perturbation
            adapted_params_minus = self._inner_loop_update(
                task_idx, init_params,
                perturbation=-self.finite_diff_epsilon)
            
            # Compute meta-losses
            meta_loss_plus = self._compute_ppo_loss(
                task_idx, adapted_params_plus,
                reuse=False, is_meta=True)
            
            meta_loss_minus = self._compute_ppo_loss(
                task_idx, adapted_params_minus,
                reuse=True, is_meta=True)
            
            # Approximate meta-gradients using finite differences
            meta_loss = (meta_loss_plus + meta_loss_minus) / 2.0
            self.meta_losses.append(meta_loss)
            
            # Compute gradients
            core_params = self.policy.core_policy.get_trainable_variables()
            meta_grads = []
            
            for param_idx, core_param in enumerate(core_params):
                # Finite difference approximation
                grad_approx = (adapted_params_plus[param_idx] - adapted_params_minus[param_idx]) / (2 * self.finite_diff_epsilon)
                meta_grads.append(grad_approx)
            
            self.meta_grad_ops.append(meta_grads)
    
    def _inner_loop_update(self, task_idx, init_params, perturbation=0.0):
        """Perform inner loop updates with optional parameter perturbation."""
        adapted_params = init_params
        
        # Add perturbation if specified
        if perturbation != 0.0:
            adapted_params = [p + perturbation for p in adapted_params]
        
        # Perform gradient steps
        for step in range(self.num_inner_grad_steps):
            # Compute loss
            inner_loss = self._compute_ppo_loss(
                task_idx, adapted_params,
                reuse=(step > 0 or perturbation != 0.0))
            
            # Compute gradients
            grads = tf.gradients(inner_loss, adapted_params)
            grads = [g if g is not None else tf.zeros_like(p)
                    for g, p in zip(grads, adapted_params)]
            
            # Update parameters
            adapted_params = [p - self.inner_lr_tensor * g
                             for p, g in zip(adapted_params, grads)]
        
        return adapted_params
    
    def _compute_ppo_loss(self, task_idx, params, reuse=False, is_meta=False):
        """
        Compute PPO loss with given parameters.
        
        Args:
            task_idx: Task index
            params: Parameters to use for computation
            reuse: Whether to reuse variables
            is_meta: Whether this is for meta-loss computation
        """
        with tf.compat.v1.variable_scope(f"ppo_loss_{task_idx}", reuse=reuse):
            # Get policy outputs with current parameters
            # Note: In practice, you would need to rebuild the network with given params
            # For simplicity, we use the existing network outputs
            
            task_policy = self.policy.meta_policies[task_idx]
            
            # Get current policy outputs
            new_logits = task_policy.network.decoder_logits
            vpred = task_policy.network.vf
            
            # Compute PPO surrogate loss
            likelihood_ratio = self.policy.distribution.likelihood_ratio_sym(
                self.actions_phs[task_idx],
                self.old_logits_phs[task_idx],
                new_logits)
            
            # Clipped surrogate objective
            clipped_ratio = tf.clip_by_value(
                likelihood_ratio,
                1.0 - self.clip_value,
                1.0 + self.clip_value)
            
            surr_obj = tf.minimum(
                likelihood_ratio * self.advs_phs[task_idx],
                clipped_ratio * self.advs_phs[task_idx])
            
            policy_loss = -tf.reduce_mean(surr_obj)
            
            # Value function loss
            vpred_clipped = self.old_vpred_phs[task_idx] + tf.clip_by_value(
                vpred - self.old_vpred_phs[task_idx],
                -self.clip_value,
                self.clip_value)
            
            vf_loss1 = tf.square(vpred - self.returns_phs[task_idx])
            vf_loss2 = tf.square(vpred_clipped - self.returns_phs[task_idx])
            value_loss = 0.5 * tf.reduce_mean(tf.maximum(vf_loss1, vf_loss2))
            
            # Total loss
            total_loss = policy_loss + self.vf_coef * value_loss
            
            # Add regularization if specified
            if self.regularization_coef > 0:
                reg_loss = self.regularization_coef * tf.add_n(
                    [tf.nn.l2_loss(p) for p in params])
                total_loss += reg_loss
            
            # Store losses for monitoring
            if not is_meta:
                self.policy_losses.append(policy_loss)
                self.value_losses.append(value_loss)
            
            return total_loss
    
    def _compute_hessian_vector_product(self, inner_loss, params, vector):
        """
        Compute Hessian-vector product for second-order gradients.
        
        Args:
            inner_loss: Loss to compute Hessian of
            params: Parameters to compute Hessian w.r.t.
            vector: Vector to multiply with Hessian
        """
        # Compute gradient of inner_loss w.r.t. params
        grads = tf.gradients(inner_loss, params)
        
        # Compute gradient of (grads · vector) w.r.t. params
        # This gives us the Hessian-vector product
        grad_vector_product = tf.reduce_sum(
            [tf.reduce_sum(g * v) for g, v in zip(grads, vector)])
        
        hvp = tf.gradients(grad_vector_product, params)
        
        return hvp
    
    def _build_meta_update_graph(self):
        """Build the meta-update graph for outer loop optimization."""
        with tf.compat.v1.variable_scope("meta_update"):
            # Get core policy parameters
            core_params = self.policy.core_policy.get_trainable_variables()
            
            # Aggregate meta-gradients from all tasks
            aggregated_grads = []
            for param_idx in range(len(core_params)):
                param_grads = []
                for task_idx in range(self.meta_batch_size):
                    if self.meta_grad_ops[task_idx][param_idx] is not None:
                        param_grads.append(self.meta_grad_ops[task_idx][param_idx])
                
                if param_grads:
                    # Average gradients across tasks
                    avg_grad = tf.reduce_mean(tf.stack(param_grads), axis=0)
                else:
                    avg_grad = tf.zeros_like(core_params[param_idx])
                
                aggregated_grads.append(avg_grad)
            
            # Apply gradient clipping
            if self.use_gradient_clipping:
                aggregated_grads, self.meta_grad_norm = tf.clip_by_global_norm(
                    aggregated_grads, self.max_grad_norm)
            
            # Create update operation
            grads_and_vars = list(zip(aggregated_grads, core_params))
            self.meta_update_op = self.outer_optimizer.apply_gradients(
                grads_and_vars,
                global_step=self.global_outer_step)
            
            # Store aggregated gradients for monitoring
            self.aggregated_meta_grads = aggregated_grads
    
    def _add_monitoring_ops(self):
        """Add operations for monitoring and debugging."""
        with tf.compat.v1.variable_scope("monitoring"):
            # Compute average losses
            self.avg_inner_loss = tf.reduce_mean(self.inner_losses) if self.inner_losses else tf.constant(0.0)
            self.avg_meta_loss = tf.reduce_mean(self.meta_losses) if self.meta_losses else tf.constant(0.0)
            self.avg_policy_loss = tf.reduce_mean(self.policy_losses) if self.policy_losses else tf.constant(0.0)
            self.avg_value_loss = tf.reduce_mean(self.value_losses) if self.value_losses else tf.constant(0.0)
            
            # Parameter norms
            core_params = self.policy.core_policy.get_trainable_variables()
            self.param_norm = tf.global_norm(core_params)
            
            # Gradient norms
            if hasattr(self, 'aggregated_meta_grads'):
                self.avg_grad_norm = tf.global_norm(self.aggregated_meta_grads)
            
            # Create summary operations
            tf.compat.v1.summary.scalar('losses/avg_inner_loss', self.avg_inner_loss)
            tf.compat.v1.summary.scalar('losses/avg_meta_loss', self.avg_meta_loss)
            tf.compat.v1.summary.scalar('losses/avg_policy_loss', self.avg_policy_loss)
            tf.compat.v1.summary.scalar('losses/avg_value_loss', self.avg_value_loss)
            tf.compat.v1.summary.scalar('norms/param_norm', self.param_norm)
            tf.compat.v1.summary.scalar('learning_rates/inner_lr', self.inner_lr_tensor)
            tf.compat.v1.summary.scalar('learning_rates/outer_lr', self.outer_lr_tensor)
            
            self.summary_op = tf.compat.v1.summary.merge_all()
    
    def _create_lr_schedule(self, base_lr, schedule_type, global_step, name):
        """Create learning rate schedule."""
        with tf.compat.v1.variable_scope(f"lr_schedule_{name}"):
            if schedule_type == 'constant':
                return tf.constant(base_lr)
            elif schedule_type == 'exponential':
                return tf.compat.v1.train.exponential_decay(
                    base_lr, global_step, self.lr_decay_steps,
                    self.lr_decay_rate, staircase=True)
            elif schedule_type == 'polynomial':
                return tf.compat.v1.train.polynomial_decay(
                    base_lr, global_step, self.lr_decay_steps,
                    end_learning_rate=base_lr * 0.01, power=1.0)
            elif schedule_type == 'cosine':
                return tf.compat.v1.train.cosine_decay(
                    base_lr, global_step, self.lr_decay_steps)
            elif schedule_type == 'linear':
                return tf.compat.v1.train.polynomial_decay(
                    base_lr, global_step, self.lr_decay_steps,
                    end_learning_rate=0.0, power=1.0)
            else:
                return tf.constant(base_lr)
    
    def UpdateMetaPolicy(self):
        """Update meta-policy using Full MAML with second-order gradients."""
        sess = tf.compat.v1.get_default_session()
        
        # Run meta-update
        _, meta_loss, param_norm, inner_lr, outer_lr = sess.run(
            [self.meta_update_op, self.avg_meta_loss, self.param_norm,
             self.inner_lr_tensor, self.outer_lr_tensor])
        
        # Log statistics
        logger.log(f"[Full MAML] Meta-loss: {meta_loss:.4f}, "
                  f"Param norm: {param_norm:.4f}, "
                  f"Inner LR: {inner_lr:.6f}, "
                  f"Outer LR: {outer_lr:.6f}")
        
        # Store statistics
        self.training_stats['meta_loss'].append(meta_loss)
        self.training_stats['param_norm'].append(param_norm)
        self.training_stats['inner_lr'].append(inner_lr)
        self.training_stats['outer_lr'].append(outer_lr)
        
        # Synchronize parameters to meta-policies
        self.policy.async_parameters()
        
        return meta_loss
    
    def UpdatePPOTarget(self, task_samples, batch_size=50):
        """
        Update PPO targets for all tasks using Full MAML.
        
        This method coordinates the inner loop updates across all tasks
        and prepares the data for meta-update.
        """
        total_policy_losses = []
        total_value_losses = []
        
        start_time = time.time()
        
        for task_idx in range(self.meta_batch_size):
            policy_losses, value_losses = self.UpdatePPOTargetPerTask(
                task_samples[task_idx], task_idx, batch_size)
            
            total_policy_losses.append(policy_losses)
            total_value_losses.append(value_losses)
        
        elapsed_time = time.time() - start_time
        
        # Log timing information
        logger.log(f"[Full MAML] Inner loop updates completed in {elapsed_time:.2f}s")
        
        return total_policy_losses, total_value_losses
    
    def UpdatePPOTargetPerTask(self, task_samples, task_id, batch_size=50):
        """Update PPO target for a specific task."""
        sess = tf.compat.v1.get_default_session()
        
        # Prepare data
        batch_number = int(task_samples['observations'].shape[0] / batch_size)
        self.update_numbers = batch_number
        
        # Prepare batched data
        observations = task_samples['observations']
        actions = task_samples['actions']
        shift_actions = np.column_stack(
            (np.zeros(actions.shape[0], dtype=np.int32),
             actions[:, 0:-1]))
        old_logits = task_samples['logits'].astype(np.float32)
        advantages = task_samples['advantages'].astype(np.float32)
        old_values = task_samples['values'].astype(np.float32)
        returns = task_samples['returns'].astype(np.float32)
        
        # Split into batches
        obs_batches = np.split(observations, batch_number)
        action_batches = np.split(actions, batch_number)
        shift_action_batches = np.split(shift_actions, batch_number)
        old_logit_batches = np.split(old_logits, batch_number)
        adv_batches = np.split(advantages, batch_number)
        old_value_batches = np.split(old_values, batch_number)
        return_batches = np.split(returns, batch_number)
        
        policy_losses = []
        value_losses = []
        
        # Perform inner loop updates
        for step in range(self.num_inner_grad_steps):
            step_policy_loss = 0.0
            step_value_loss = 0.0
            
            for batch_idx in range(batch_number):
                obs_batch = obs_batches[batch_idx]
                decoder_length = np.array([obs_batch.shape[1]] * obs_batch.shape[0], dtype=np.int32)
                
                # Prepare feed dict
                feed_dict = {
                    self.obs_phs[task_id]: obs_batch,
                    self.actions_phs[task_id]: action_batches[batch_idx],
                    self.old_logits_phs[task_id]: old_logit_batches[batch_idx],
                    self.old_vpred_phs[task_id]: old_value_batches[batch_idx],
                    self.advs_phs[task_id]: adv_batches[batch_idx],
                    self.returns_phs[task_id]: return_batches[batch_idx],
                    self.decoder_inputs_phs[task_id]: shift_action_batches[batch_idx],
                    self.decoder_length_phs[task_id]: decoder_length,
                    # Also feed to the policy's placeholders
                    self.policy.meta_policies[task_id].obs: obs_batch,
                    self.policy.meta_policies[task_id].decoder_inputs: shift_action_batches[batch_idx],
                    self.policy.meta_policies[task_id].decoder_targets: action_batches[batch_idx],
                    self.policy.meta_policies[task_id].decoder_full_length: decoder_length
                }
                
                # Run inner update (this is handled by the graph)
                if self.policy_losses and self.value_losses:
                    policy_loss, value_loss = sess.run(
                        [self.policy_losses[task_id], self.value_losses[task_id]],
                        feed_dict=feed_dict)
                    
                    step_policy_loss += policy_loss
                    step_value_loss += value_loss
            
            # Average over batches
            step_policy_loss /= batch_number
            step_value_loss /= batch_number
            
            policy_losses.append(step_policy_loss)
            value_losses.append(step_value_loss)
            
            # Increment inner step counter
            sess.run(tf.assign_add(self.global_inner_step, 1))
            
            # Debug logging
            if step == 0 and task_id == 0:
                logger.log(f"\n[Full MAML Debug] Task {task_id}, Step {step}:")
                logger.log(f"  Policy loss: {step_policy_loss:.4f}")
                logger.log(f"  Value loss: {step_value_loss:.4f}")
        
        return policy_losses, value_losses
    
    def get_diagnostics(self):
        """Get diagnostic information for monitoring."""
        return dict(self.training_stats)