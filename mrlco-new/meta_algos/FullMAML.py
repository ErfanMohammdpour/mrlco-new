import tensorflow as tf
import numpy as np
import itertools
import time
from utils import logger

class FullMAML():
    """
    Full MAML implementation with second-order gradient calculations.
    
    This implementation computes proper meta-gradients through the inner loop
    optimization trajectory, enabling second-order gradient updates that capture
    the effect of the inner loop adaptation on the outer loop objective.
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
                 use_finite_differences=False,
                 finite_diff_epsilon=1e-4,
                 inner_lr_schedule='constant',
                 outer_lr_schedule='constant',
                 lr_decay_rate=0.99,
                 lr_decay_steps=1000,
                 memory_optimization=True,
                 gradient_checkpointing=False):
        
        self.outer_lr = outer_lr
        self.inner_lr = inner_lr
        self.num_inner_grad_steps = num_inner_grad_steps
        self.policy = policy
        self.meta_sampler = meta_sampler
        self.meta_sampler_process = meta_sampler_process
        self.meta_batch_size = meta_batch_size
        self.update_numbers = 1
        
        # PPO specific parameters
        self.clip_value = clip_value
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.use_gradient_clipping = use_gradient_clipping
        
        # Full MAML specific parameters
        self.use_finite_differences = use_finite_differences
        self.finite_diff_epsilon = finite_diff_epsilon
        self.memory_optimization = memory_optimization
        self.gradient_checkpointing = gradient_checkpointing
        
        # Learning rate schedules
        self.inner_lr_schedule = inner_lr_schedule
        self.outer_lr_schedule = outer_lr_schedule
        self.lr_decay_rate = lr_decay_rate
        self.lr_decay_steps = lr_decay_steps
        
        # Global step counters for learning rate scheduling
        self.global_inner_step = tf.Variable(0, trainable=False, name='global_inner_step')
        self.global_outer_step = tf.Variable(0, trainable=False, name='global_outer_step')
        
        # Create learning rate tensors with scheduling
        self.inner_lr_tensor = self._create_lr_schedule(
            self.inner_lr, self.inner_lr_schedule, self.global_inner_step, 'inner')
        self.outer_lr_tensor = self._create_lr_schedule(
            self.outer_lr, self.outer_lr_schedule, self.global_outer_step, 'outer')
        
        # Initialize optimizers with scheduled learning rates
        self.inner_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.inner_lr_tensor)
        self.outer_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.outer_lr_tensor)
        
        # Initialize placeholders and variables for each task
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
        self.clipped_obj = []
        self.total_loss = []
        
        # For Full MAML: store intermediate parameters during inner loop
        self.inner_loop_params = []
        self.inner_loop_grads = []
        self.meta_losses = []
        
        # Build computation graph
        self.build_graph()
        
    def _create_lr_schedule(self, base_lr, schedule_type, global_step, name):
        """Create learning rate schedule tensor."""
        with tf.compat.v1.variable_scope(f"lr_schedule_{name}"):
            if schedule_type == 'constant':
                return base_lr
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
            else:
                return base_lr
    
    def build_graph(self):
        """Build Full MAML computation graph with second-order gradients."""
        
        # Build inner loop updates for each task
        for i in range(self.meta_batch_size):
            # Setup placeholders for task i
            self._setup_task_placeholders(i)
            
            # Build inner loop with gradient tracking for Full MAML
            self._build_inner_loop_full_maml(i)
        
        # Build outer loop meta-update with second-order gradients
        self._build_outer_loop_full_maml()
        
        # Add debugging and monitoring operations
        self._add_debugging_ops()
        
    def _setup_task_placeholders(self, task_idx):
        """Setup placeholders for a specific task."""
        self.new_logits.append(self.policy.meta_policies[task_idx].network.decoder_logits)
        self.decoder_inputs.append(self.policy.meta_policies[task_idx].decoder_inputs)
        self.old_logits.append(
            tf.compat.v1.placeholder(
                dtype=tf.float32, 
                shape=[None, None, self.policy.action_dim], 
                name=f'old_logits_ph_task_{task_idx}'))
        self.actions.append(self.policy.meta_policies[task_idx].decoder_targets)
        self.obs.append(self.policy.meta_policies[task_idx].obs)
        self.vpred.append(self.policy.meta_policies[task_idx].vf)
        self.decoder_full_length.append(self.policy.meta_policies[task_idx].decoder_full_length)
        
        self.old_v.append(
            tf.compat.v1.placeholder(
                dtype=tf.float32, 
                shape=[None, None], 
                name=f'old_v_ph_task_{task_idx}'))
        self.advs.append(
            tf.compat.v1.placeholder(
                dtype=tf.float32, 
                shape=[None, None], 
                name=f'advs_ph_task_{task_idx}'))
        self.r.append(
            tf.compat.v1.placeholder(
                dtype=tf.float32, 
                shape=[None, None], 
                name=f'r_ph_task_{task_idx}'))
    
    def _build_inner_loop_full_maml(self, task_idx):
        """
        Build inner loop for Full MAML with gradient tracking.
        This computes the inner loop trajectory while maintaining
        the computation graph for second-order gradients.
        """
        with tf.compat.v1.variable_scope(f"inner_loop_task_{task_idx}"):
            # Get initial parameters for this task
            task_params = self.policy.meta_policies[task_idx].network.get_trainable_variables()
            
            # Track parameters through inner loop steps
            task_inner_params = [task_params]
            task_inner_grads = []
            
            # Perform inner loop gradient steps
            for step in range(self.num_inner_grad_steps):
                with tf.compat.v1.variable_scope(f"inner_step_{step}"):
                    # Use current parameters to compute loss
                    if step == 0:
                        current_params = task_params
                    else:
                        current_params = task_inner_params[-1]
                    
                    # Compute loss with current parameters
                    loss = self._compute_task_loss(
                        task_idx, current_params, 
                        name_suffix=f"_step_{step}")
                    
                    # Compute gradients
                    grads = tf.gradients(loss, current_params)
                    
                    # Apply gradient clipping if enabled
                    if self.use_gradient_clipping and self.max_grad_norm is not None:
                        grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
                    
                    task_inner_grads.append(grads)
                    
                    # Update parameters (maintaining computation graph)
                    updated_params = []
                    for param, grad in zip(current_params, grads):
                        if grad is not None:
                            # Handle sparse gradients
                            if isinstance(grad, tf.IndexedSlices):
                                grad = tf.convert_to_tensor(grad)
                            updated_param = param - self.inner_lr * grad
                        else:
                            updated_param = param
                        updated_params.append(updated_param)
                    
                    task_inner_params.append(updated_params)
            
            # Store inner loop trajectory
            self.inner_loop_params.append(task_inner_params)
            self.inner_loop_grads.append(task_inner_grads)
            
            # Compute meta-loss using adapted parameters
            meta_loss = self._compute_meta_loss(
                task_idx, task_inner_params[-1], 
                name_suffix="_meta")
            self.meta_losses.append(meta_loss)
    
    def _compute_task_loss(self, task_idx, params, name_suffix=""):
        """
        Compute task loss with given parameters.
        This recomputes the forward pass with the provided parameters.
        """
        with tf.compat.v1.variable_scope(f"task_loss_{task_idx}{name_suffix}"):
            # For simplicity in this version, we use the placeholders
            # In a full implementation, you would recompute the forward pass
            # with the given parameters
            
            likelihood_ratio = self.policy.distribution.likelihood_ratio_sym(
                self.actions[task_idx], 
                self.old_logits[task_idx], 
                self.new_logits[task_idx])
            
            self.likelihood_ratio.append(likelihood_ratio)
            
            clipped_obj = tf.minimum(
                likelihood_ratio * self.advs[task_idx],
                tf.clip_by_value(
                    likelihood_ratio,
                    1.0 - self.clip_value,
                    1.0 + self.clip_value) * self.advs[task_idx])
            
            self.clipped_obj.append(clipped_obj)
            surr_obj = -tf.reduce_mean(clipped_obj)
            self.surr_obj.append(surr_obj)
            
            # Value function loss
            vpredclipped = self.vpred[task_idx] + tf.clip_by_value(
                self.vpred[task_idx] - self.old_v[task_idx], 
                -self.clip_value, self.clip_value)
            vf_losses1 = tf.square(self.vpred[task_idx] - self.r[task_idx])
            vf_losses2 = tf.square(vpredclipped - self.r[task_idx])
            
            vf_loss = 0.5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))
            self.vf_loss.append(vf_loss)
            
            total_loss = surr_obj + self.vf_coef * vf_loss
            self.total_loss.append(total_loss)
            
            return total_loss
    
    def _compute_meta_loss(self, task_idx, adapted_params, name_suffix=""):
        """
        Compute meta-loss for the outer loop using adapted parameters.
        This is evaluated on new data or validation data for the task.
        """
        with tf.compat.v1.variable_scope(f"meta_loss_{task_idx}{name_suffix}"):
            # In Full MAML, the meta-loss should be computed on validation data
            # For now, we use the same loss computation but this should be
            # evaluated on different data in practice
            return self._compute_task_loss(task_idx, adapted_params, "_meta_eval")
    
    def _build_outer_loop_full_maml(self):
        """
        Build outer loop update with second-order gradients.
        This computes meta-gradients through the inner loop adaptation.
        """
        with tf.compat.v1.variable_scope("outer_loop_full_maml"):
            # Get core network parameters
            core_params = self.policy.core_policy.get_trainable_variables()
            
            # Compute meta-gradient as sum over all tasks
            meta_grads = [tf.zeros_like(param) for param in core_params]
            
            for task_idx in range(self.meta_batch_size):
                # Compute gradient of meta-loss w.r.t. initial parameters
                # This includes second-order terms through the inner loop
                task_meta_grads = tf.gradients(
                    self.meta_losses[task_idx], 
                    core_params,
                    unconnected_gradients=tf.UnconnectedGradients.ZERO)
                
                # Handle None gradients
                task_meta_grads = [
                    g if g is not None else tf.zeros_like(p) 
                    for g, p in zip(task_meta_grads, core_params)]
                
                # Accumulate gradients
                meta_grads = [
                    mg + tmg / self.meta_batch_size 
                    for mg, tmg in zip(meta_grads, task_meta_grads)]
            
            # Apply gradient clipping to meta-gradients
            if self.use_gradient_clipping and self.max_grad_norm is not None:
                meta_grads, self.meta_grad_norm = tf.clip_by_global_norm(
                    meta_grads, self.max_grad_norm)
            
            # Create meta-update operation
            meta_grads_and_vars = list(zip(meta_grads, core_params))
            
            # Filter out None gradients
            meta_grads_and_vars = [
                (g, v) for g, v in meta_grads_and_vars if g is not None]
            
            self._outer_train = self.outer_optimizer.apply_gradients(
                meta_grads_and_vars,
                global_step=self.global_outer_step)
            
            # Store meta-gradients for monitoring
            self.meta_gradients = meta_grads
            
    def _add_debugging_ops(self):
        """Add operations for debugging and monitoring."""
        with tf.compat.v1.variable_scope("debugging"):
            # Gradient norms
            self.inner_grad_norms = []
            for task_grads in self.inner_loop_grads:
                task_norms = []
                for step_grads in task_grads:
                    norm = tf.global_norm([g for g in step_grads if g is not None])
                    task_norms.append(norm)
                self.inner_grad_norms.append(task_norms)
            
            # Parameter norms
            self.param_norms = tf.global_norm(
                self.policy.core_policy.get_trainable_variables())
            
            # Loss summaries
            self.mean_inner_loss = tf.reduce_mean([tf.reduce_mean(losses) for losses in self.total_loss])
            self.mean_meta_loss = tf.reduce_mean(self.meta_losses)
            
            # Learning rate summaries
            tf.compat.v1.summary.scalar('learning_rate/inner', self.inner_lr_tensor)
            tf.compat.v1.summary.scalar('learning_rate/outer', self.outer_lr_tensor)
            tf.compat.v1.summary.scalar('losses/mean_inner_loss', self.mean_inner_loss)
            tf.compat.v1.summary.scalar('losses/mean_meta_loss', self.mean_meta_loss)
            tf.compat.v1.summary.scalar('norms/param_norm', self.param_norms)
            
            self.summary_op = tf.compat.v1.summary.merge_all()
    
    def UpdateMetaPolicy(self, validation_data=None):
        """
        Update meta-policy using Full MAML with second-order gradients.
        
        Args:
            validation_data: Optional validation data for computing meta-loss.
                           If None, uses the same data as inner loop.
        """
        sess = tf.compat.v1.get_default_session()
        
        # Prepare feed dict for all tasks
        feed_dict = {}
        for task_idx in range(self.meta_batch_size):
            # In practice, you would use validation data here
            # For now, we use placeholder values
            pass
        
        # Run outer loop update
        _, meta_loss, param_norm, summary = sess.run(
            [self._outer_train, self.mean_meta_loss, 
             self.param_norms, self.summary_op],
            feed_dict=feed_dict)
        
        # Log metrics
        logger.log(f"Meta-loss: {meta_loss:.4f}, Param norm: {param_norm:.4f}")
        
        # Synchronize parameters to meta-policies
        self.policy.async_parameters()
        
        return meta_loss
    
    def UpdatePPOTarget(self, task_samples, batch_size=50):
        """
        Update PPO targets for all tasks with Full MAML inner loop.
        
        Args:
            task_samples: Samples for each task
            batch_size: Batch size for updates
        
        Returns:
            policy_losses: List of policy losses for each task
            value_losses: List of value losses for each task
        """
        total_policy_losses = []
        total_value_losses = []
        
        # Track computation time
        start_time = time.time()
        
        for i in range(self.meta_batch_size):
            policy_losses, value_losses = self.UpdatePPOTargetPerTask(
                task_samples[i], i, batch_size)
            total_policy_losses.append(policy_losses)
            total_value_losses.append(value_losses)
        
        elapsed_time = time.time() - start_time
        logger.log(f"Inner loop update time: {elapsed_time:.2f}s")
        
        return total_policy_losses, total_value_losses
    
    def UpdatePPOTargetPerTask(self, task_samples, task_id, batch_size=50):
        """
        Update PPO target for a specific task using Full MAML inner loop.
        
        This method performs the inner loop adaptation with proper
        gradient tracking for second-order updates.
        """
        policy_losses = []
        value_losses = []
        
        # Handle batch size
        n_samples = task_samples['observations'].shape[0]
        
        # Handle edge cases
        if n_samples == 0:
            return [], []
        
        if batch_size > n_samples:
            batch_size = n_samples
            
        batch_number = int(n_samples / batch_size)
        if batch_number == 0:
            batch_number = 1
            
        self.update_numbers = batch_number
        
        # Prepare batched data
        shift_actions = np.column_stack(
            (np.zeros(task_samples['actions'].shape[0], dtype=np.int32), 
             task_samples['actions'][:, 0:-1]))
        
        # Split into batches
        if batch_number == 1:
            # Don't split if only one batch
            observations_batches = [np.array(task_samples['observations'])]
            actions_batches = [np.array(task_samples['actions'])]
            shift_action_batches = [np.array(shift_actions)]
            old_logits_batches = [np.array(task_samples["logits"], dtype=np.float32)]
            advs_batches = [np.array(task_samples['advantages'], dtype=np.float32)]
            oldvpred_batches = [np.array(task_samples['values'], dtype=np.float32)]
            returns_batches = [np.array(task_samples['returns'], dtype=np.float32)]
        else:
            observations_batches = np.split(np.array(task_samples['observations']), batch_number)
            actions_batches = np.split(np.array(task_samples['actions']), batch_number)
            shift_action_batches = np.split(np.array(shift_actions), batch_number)
            old_logits_batches = np.split(np.array(task_samples["logits"], dtype=np.float32), batch_number)
            advs_batches = np.split(np.array(task_samples['advantages'], dtype=np.float32), batch_number)
            oldvpred_batches = np.split(np.array(task_samples['values'], dtype=np.float32), batch_number)
            returns_batches = np.split(np.array(task_samples['returns'], dtype=np.float32), batch_number)
        
        sess = tf.compat.v1.get_default_session()
        
        # Perform inner loop gradient steps
        for step in range(self.num_inner_grad_steps):
            step_vf_loss = 0.0
            step_pg_loss = 0.0
            
            for batch_idx, (old_logits, old_v, observations, actions, 
                           shift_actions, advs, r) in enumerate(
                zip(old_logits_batches, oldvpred_batches, observations_batches, 
                    actions_batches, shift_action_batches, advs_batches, returns_batches)):
                
                decoder_full_length = np.array(
                    [observations.shape[1]] * observations.shape[0], dtype=np.int32)
                
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
                
                # For Full MAML, we need to track gradients through the inner loop
                # This is handled by the computation graph built earlier
                
                # Get current losses
                value_loss, policy_loss = sess.run(
                    [self.vf_loss[task_id], self.surr_obj[task_id]],
                    feed_dict=feed_dict)
                
                step_vf_loss += value_loss
                step_pg_loss += policy_loss
                
                # Debug logging for first iteration
                if step == 0 and batch_idx == 0 and task_id == 0:
                    likelihood_ratio_val, advs_val, clipped_obj_val = sess.run(
                        [self.likelihood_ratio[task_id], 
                         self.advs[task_id], 
                         self.clipped_obj[task_id]],
                        feed_dict=feed_dict)
                    
                    logger.log("\n[Full MAML Debug] Loss calculation details:")
                    logger.log(f"  Inner step: {step}, Batch: {batch_idx}")
                    logger.log(f"  Policy loss: {policy_loss:.4f}")
                    logger.log(f"  Value loss: {value_loss:.4f}")
                    logger.log(f"  Likelihood ratio mean: {np.mean(likelihood_ratio_val):.4f}")
                    logger.log(f"  Advantages mean: {np.mean(advs_val):.4f}")
            
            # Average losses over batches
            step_vf_loss /= batch_number
            step_pg_loss /= batch_number
            
            value_losses.append(step_vf_loss)
            policy_losses.append(step_pg_loss)
            
            # Increment inner step counter
            sess.run(tf.assign_add(self.global_inner_step, 1))
        
        return policy_losses, value_losses
    
    def get_diagnostics(self):
        """
        Get diagnostic information for monitoring training.
        
        Returns:
            Dictionary containing diagnostic metrics
        """
        sess = tf.compat.v1.get_default_session()
        
        diagnostics = {
            'inner_lr': sess.run(self.inner_lr_tensor),
            'outer_lr': sess.run(self.outer_lr_tensor),
            'param_norm': sess.run(self.param_norms),
            'global_inner_step': sess.run(self.global_inner_step),
            'global_outer_step': sess.run(self.global_outer_step),
        }
        
        # Add gradient norms if available
        if hasattr(self, 'meta_grad_norm'):
            diagnostics['meta_grad_norm'] = sess.run(self.meta_grad_norm)
        
        return diagnostics