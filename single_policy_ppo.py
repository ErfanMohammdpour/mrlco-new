import tensorflow as tf
import numpy as np
from utils.mpi_adam_optimizer import MpiAdamOptimizer
from mpi4py import MPI
from policies.meta_seq2seq_policy import Seq2SeqPolicy
import itertools

class SinglePolicyPPO():
    """
    Provides PPO based offloading training for single-policy RL
    """
    def __init__(self,
                 policy,
                 sampler,
                 sampler_process,
                 lr=1e-4,
                 num_grad_steps=4,
                 clip_value=0.2,
                 vf_coef=0.5,
                 max_grad_norm=0.5):
        self.lr = lr
        self.num_grad_steps = num_grad_steps
        self.policy = policy
        self.sampler = sampler
        self.sampler_process = sampler_process

        # Use Adam optimizer for single-policy training
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.lr, epsilon=1e-5)
        self.clip_value = clip_value
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm

        self.build_graph()

    def build_graph(self):
        # Build the computational graph for PPO updates
        new_logits = self.policy.network.decoder_logits
        self.decoder_inputs = self.policy.decoder_inputs
        self.old_logits = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, None, self.policy.action_dim], name='old_logits_ph')
        self.actions = self.policy.decoder_targets
        self.obs = self.policy.obs
        self.vpred = self.policy.vf
        self.decoder_full_length = self.policy.decoder_full_length

        self.old_v = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, None], name='old_v_ph')
        self.advs = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, None], name='advs_ph')
        self.r = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, None], name='r_ph')

        with tf.compat.v1.variable_scope("ppo_update") as scope:
            # Compute likelihood ratio for PPO
            likelihood_ratio = self.policy.distribution.likelihood_ratio_sym(self.actions, self.old_logits, new_logits)

            # Compute clipped objective
            clipped_obj = tf.minimum(likelihood_ratio * self.advs,
                                   tf.clip_by_value(likelihood_ratio,
                                                    1.0 - self.clip_value,
                                                    1.0 + self.clip_value) * self.advs)
            self.surr_obj = -tf.reduce_mean(clipped_obj)

            # Value function loss with clipping
            vpredclipped = self.vpred + tf.clip_by_value(self.vpred - self.old_v, -self.clip_value, self.clip_value)
            vf_losses1 = tf.square(self.vpred - self.r)
            vf_losses2 = tf.square(vpredclipped - self.r)

            self.vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))

            # Total loss combining policy and value function losses
            self.total_loss = self.surr_obj + self.vf_coef * self.vf_loss

            # Get trainable parameters
            params = self.policy.network.get_trainable_variables()

            # Compute gradients
            grads_and_var = self.optimizer.compute_gradients(self.total_loss, params)

            grads, var = zip(*grads_and_var)

            # Gradient clipping
            if self.max_grad_norm is not None:
                grads, _grad_norm = tf.clip_by_global_norm(grads, self.max_grad_norm)
            grads_and_var = list(zip(grads, var))

            # Apply gradients
            self._train = self.optimizer.apply_gradients(grads_and_var)

    def UpdatePPOTarget(self, samples_data, batch_size=50):
        """
        Update the policy using PPO algorithm
        
        Args:
            samples_data (dict): Dictionary containing training data
            batch_size (int): Batch size for training
            
        Returns:
            tuple: (policy_losses, value_losses)
        """
        policy_losses = []
        value_losses = []

        batch_number = int(samples_data['observations'].shape[0] / batch_size)

        # Prepare shifted actions for decoder input
        shift_actions = np.column_stack(
            (np.zeros(samples_data['actions'].shape[0], dtype=np.int32), 
             samples_data['actions'][:, 0:-1]))

        # Split data into batches
        observations_batches = np.split(np.array(samples_data['observations']), batch_number)
        actions_batches = np.split(np.array(samples_data['actions']), batch_number)
        shift_action_batches = np.split(np.array(shift_actions), batch_number)

        old_logits_batches = np.split(np.array(samples_data["logits"], dtype=np.float32), batch_number)
        advs_batches = np.split(np.array(samples_data['advantages'], dtype=np.float32), batch_number)
        oldvpred = np.split(np.array(samples_data['values'], dtype=np.float32), batch_number)
        returns = np.split(np.array(samples_data['returns'], dtype=np.float32), batch_number)

        sess = tf.compat.v1.get_default_session()

        vf_loss = 0.0
        pg_loss = 0.0

        # Perform multiple gradient steps
        for i in range(self.num_grad_steps):
            for old_logits, old_v, observations, actions, shift_actions, advs, r in zip(
                old_logits_batches, oldvpred, observations_batches, actions_batches,
                shift_action_batches, advs_batches, returns):
                
                decoder_full_length = np.array([observations.shape[1]] * observations.shape[0], dtype=np.int32)

                feed_dict = {
                    self.old_logits: old_logits, 
                    self.old_v: old_v, 
                    self.obs: observations, 
                    self.actions: actions,
                    self.decoder_inputs: shift_actions,
                    self.decoder_full_length: decoder_full_length, 
                    self.advs: advs, 
                    self.r: r
                }

                _, value_loss, policy_loss = sess.run([self._train, self.vf_loss, self.surr_obj], feed_dict=feed_dict)

                vf_loss += value_loss
                pg_loss += policy_loss

            # Average losses over gradient steps
            vf_loss = vf_loss / float(self.num_grad_steps)
            pg_loss = pg_loss / float(self.num_grad_steps)

            value_losses.append(vf_loss)
            policy_losses.append(pg_loss)

        return policy_losses, value_losses
