import tensorflow as tf
import numpy as np
import itertools
from utils.gpu import log_tensor_device
from utils.distributed_tf1 import aggregate_gradients


class MRLCODistributed():
    """Distributed version of MRLCO that supports multi-GPU training with TF1 compatibility"""
    
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
                 device_list=None):
        self.outer_lr = outer_lr
        self.inner_lr = inner_lr
        self.num_inner_grad_steps = num_inner_grad_steps
        self.policy = policy
        self.meta_sampler = meta_sampler
        self.meta_sampler_process = meta_sampler_process
        self.meta_batch_size = meta_batch_size
        self.update_numbers = 1
        self.device_list = device_list or ['/GPU:0']
        self.num_devices = len(self.device_list)
        
        # TF1-style optimizers
        self.inner_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.inner_lr)
        self.outer_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.outer_lr)
        self.clip_value = clip_value
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        
        # Build distributed graph
        self.build_distributed_graph()
    
    def build_distributed_graph(self):
        """Build training graph with multi-GPU support"""
        # Storage for per-device operations
        self.device_losses = []
        self.device_gradients = []
        self.device_train_ops = []
        
        # Build operations for each device
        tasks_per_device = self.meta_batch_size // self.num_devices
        remaining_tasks = self.meta_batch_size % self.num_devices
        
        # Placeholders and ops for each meta-task
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
        self._train = []
        
        task_idx = 0
        for device_idx, device in enumerate(self.device_list):
            device_losses = []
            device_grads = []
            
            # Calculate number of tasks for this device
            num_tasks = tasks_per_device + (1 if device_idx < remaining_tasks else 0)
            
            with tf.device(device):
                for local_task_idx in range(num_tasks):
                    i = task_idx
                    
                    # Create placeholders and get model outputs
                    self.new_logits.append(self.policy.meta_policies[i].network.decoder_logits)
                    self.decoder_inputs.append(self.policy.meta_policies[i].decoder_inputs)
                    self.old_logits.append(tf.compat.v1.placeholder(
                        dtype=tf.float32, 
                        shape=[None, None, self.policy.action_dim], 
                        name=f'old_logits_ph_task_{i}'
                    ))
                    self.actions.append(self.policy.meta_policies[i].decoder_targets)
                    self.obs.append(self.policy.meta_policies[i].obs)
                    self.vpred.append(self.policy.meta_policies[i].vf)
                    self.decoder_full_length.append(self.policy.meta_policies[i].decoder_full_length)
                    
                    self.old_v.append(tf.compat.v1.placeholder(
                        dtype=tf.float32, 
                        shape=[None, None], 
                        name=f'old_v_ph_task_{i}'
                    ))
                    self.advs.append(tf.compat.v1.placeholder(
                        dtype=tf.float32, 
                        shape=[None, None], 
                        name=f'advs_ph_task_{i}'
                    ))
                    self.r.append(tf.compat.v1.placeholder(
                        dtype=tf.float32, 
                        shape=[None, None], 
                        name=f'r_ph_task_{i}'
                    ))
                    
                    with tf.compat.v1.variable_scope(f"inner_update_parameters_task_{i}") as scope:
                        # Compute PPO loss
                        likelihood_ratio = self.policy.distribution.likelihood_ratio_sym(
                            self.actions[i], self.old_logits[i], self.new_logits[i]
                        )
                        self.likelihood_ratio.append(likelihood_ratio)
                        
                        clipped_obj = tf.minimum(
                            likelihood_ratio * self.advs[i],
                            tf.clip_by_value(likelihood_ratio, 1.0 - self.clip_value, 1.0 + self.clip_value) * self.advs[i]
                        )
                        self.clipped_obj.append(clipped_obj)
                        self.surr_obj.append(-tf.reduce_mean(clipped_obj))
                        
                        # Value function loss
                        vpredclipped = self.vpred[i] + tf.clip_by_value(
                            self.vpred[i] - self.old_v[i], -self.clip_value, self.clip_value
                        )
                        vf_losses1 = tf.square(self.vpred[i] - self.r[i])
                        vf_losses2 = tf.square(vpredclipped - self.r[i])
                        
                        self.vf_loss.append(0.5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2)))
                        
                        # Total loss
                        self.total_loss.append(self.surr_obj[i] + self.vf_coef * self.vf_loss[i])
                        device_losses.append(self.total_loss[i])
                        
                        # Compute gradients
                        params = self.policy.meta_policies[i].network.get_trainable_variables()
                        grads_and_var = self.inner_optimizer.compute_gradients(self.total_loss[i], params)
                        grads, var = zip(*grads_and_var)
                        
                        if self.max_grad_norm is not None:
                            grads, _grad_norm = tf.clip_by_global_norm(grads, self.max_grad_norm)
                        
                        device_grads.append(list(zip(grads, var)))
                        
                        # Create train op
                        self._train.append(self.inner_optimizer.apply_gradients(
                            zip(grads, var), 
                            global_step=tf.compat.v1.train.get_or_create_global_step()
                        ))
                    
                    task_idx += 1
            
            self.device_losses.append(device_losses)
            self.device_gradients.append(device_grads)
        
        # Create outer update operations
        # Get the trainable variables from core policy to know shapes
        core_trainable_vars = self.policy.core_policy.network.get_trainable_variables()
        
        self.theta_vars = []
        with tf.compat.v1.variable_scope("meta_policy_variables") as scope:
            for i, var in enumerate(core_trainable_vars):
                self.theta_vars.append(tf.compat.v1.placeholder(
                    dtype=tf.float32, 
                    shape=var.get_shape().as_list(), 
                    name=f'theta_placeholder_{i}'
                ))
        
        # Meta update operation
        self.meta_params = self.policy.core_policy.network.get_trainable_variables()
        self.meta_param_values = self.policy.core_policy.network.get_variable_values()
        
        print(f"[DEBUG] Number of theta_vars: {len(self.theta_vars)}")
        print(f"[DEBUG] Number of meta_params: {len(self.meta_params)}")
        print(f"[DEBUG] Number of meta_param_values: {len(self.meta_param_values)}")
        
        # Compute meta gradients across all devices
        all_meta_grads = []
        for device_idx, device in enumerate(self.device_list):
            with tf.device(device):
                device_meta_grads = []
                for i, param in enumerate(self.meta_params):
                    adapted_values = []
                    # Collect adapted values from tasks on this device
                    start_idx = sum(tasks_per_device + (1 if d < remaining_tasks else 0) 
                                  for d in range(device_idx))
                    end_idx = start_idx + tasks_per_device + (1 if device_idx < remaining_tasks else 0)
                    
                    for task_idx in range(start_idx, end_idx):
                        if task_idx < self.meta_batch_size:
                            # Use task_idx to get the corresponding adapted parameter value
                            if i < len(self.theta_vars):
                                adapted_values.append(self.theta_vars[i])
                            else:
                                # Fallback to the original parameter if theta_vars doesn't have enough elements
                                adapted_values.append(param)
                    
                    if adapted_values:
                        # Average adapted parameters from tasks on this device
                        avg_adapted = tf.reduce_mean(tf.stack(adapted_values), axis=0)
                        grad = avg_adapted - param
                        device_meta_grads.append((grad, param))
                
                all_meta_grads.append(device_meta_grads)
        
        # Aggregate gradients across devices
        if len(all_meta_grads) > 1:
            self.aggregated_meta_grads = aggregate_gradients(all_meta_grads)
        else:
            self.aggregated_meta_grads = all_meta_grads[0]
        
        # Apply meta update
        self._meta_train = self.outer_optimizer.apply_gradients(
            self.aggregated_meta_grads,
            global_step=tf.compat.v1.train.get_or_create_global_step()
        )
    
    def UpdatePPOTarget(self, samples_data, batch_size=50, sess=None):
        """Update target policies using distributed training"""
        policy_losses = []
        value_losses = []
        
        for i in range(self.meta_batch_size):
            batch_number = int(samples_data[i]['observations'].shape[0] / batch_size)
            policy_losses_i = []
            value_losses_i = []
            
            for batch_id in range(batch_number):
                batch_indexs = np.arange(batch_id * batch_size, (batch_id + 1) * batch_size)
                
                # Prepare feed dict
                feed_dict = {
                    self.obs[i]: samples_data[i]['observations'][batch_indexs],
                    self.actions[i]: samples_data[i]['actions'][batch_indexs],
                    self.decoder_inputs[i]: samples_data[i]['decoder_inputs'][batch_indexs],
                    self.decoder_full_length[i]: samples_data[i]['decoder_full_length'][batch_indexs],
                    self.old_logits[i]: samples_data[i]['logits'][batch_indexs],
                    self.old_v[i]: samples_data[i]["values"][batch_indexs],
                    self.advs[i]: samples_data[i]["advantages"][batch_indexs],
                    self.r[i]: samples_data[i]["returns"][batch_indexs]
                }
                
                # Run training operation
                if sess is not None:
                    # Log device info on first batch
                    if i == 0 and batch_id == 0:
                        device_idx = i % self.num_devices
                        print(f"[Step 0] Training task {i} on device: {self.device_list[device_idx]}")
                    
                    _, surr_loss_i, vf_loss_i = sess.run(
                        [self._train[i], self.surr_obj[i], self.vf_loss[i]], 
                        feed_dict=feed_dict
                    )
                else:
                    # For TF2 eager mode (used in meta_evaluator)
                    _, surr_loss_i, vf_loss_i = tf.compat.v1.get_default_session().run(
                        [self._train[i], self.surr_obj[i], self.vf_loss[i]], 
                        feed_dict=feed_dict
                    )
                
                policy_losses_i.append(surr_loss_i)
                value_losses_i.append(vf_loss_i)
            
            policy_losses.append(np.mean(policy_losses_i))
            value_losses.append(np.mean(value_losses_i))
        
        return policy_losses, value_losses
    
    def UpdateMetaPolicy(self, sess=None):
        """Update meta policy using distributed aggregation"""
        feed_dict = {}
        
        # Get the actual variable values from the session for each adapted policy
        if sess is None:
            sess = tf.compat.v1.get_default_session()
        
        # Get adapted parameter values after inner updates
        for j, var in enumerate(self.meta_params):
            # Collect adapted values from all tasks for this parameter
            adapted_values = []
            for i in range(self.meta_batch_size):
                # Make sure the task policy has a network
                if self.policy.meta_policies[i].network is not None:
                    task_vars = self.policy.meta_policies[i].network.get_trainable_variables()
                    if j < len(task_vars):
                        task_var = task_vars[j]
                        adapted_value = sess.run(task_var)
                        adapted_values.append(adapted_value)
                    else:
                        # If task doesn't have this variable, use the meta param value
                        adapted_values.append(sess.run(var))
                else:
                    # If network not initialized, use the meta param value
                    adapted_values.append(sess.run(var))
            
            # Average the adapted values across tasks
            avg_adapted_value = np.mean(adapted_values, axis=0)
            feed_dict[self.theta_vars[j]] = avg_adapted_value
        
        # Run meta update
        sess.run(self._meta_train, feed_dict=feed_dict)
        
        # Sync all task policies with the updated meta policy
        for i in range(self.meta_batch_size):
            self.policy.assign_old_eq_new_tasks[i]()