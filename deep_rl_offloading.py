"""
Deep RL Implementation for Task Offloading
Replaces Meta-RL with single-policy Deep RL approach
"""

import tensorflow as tf
import numpy as np
import os
import joblib
from collections import deque
import random

class DeepRLOffloadingAgent:
    """
    Deep RL Agent for task offloading using Actor-Critic with Graph2Seq encoder
    """
    
    def __init__(self, 
                 obs_dim=17,
                 action_dim=2,
                 encoder_units=128,
                 decoder_units=128,
                 learning_rate=3e-4,
                 gamma=0.99,
                 tau=0.005,
                 epsilon_start=1.0,
                 epsilon_end=0.01,
                 epsilon_decay=0.995,
                 buffer_size=100000,
                 batch_size=64,
                 update_frequency=4,
                 target_update_frequency=100,
                 num_layers=2):
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.encoder_units = encoder_units
        self.decoder_units = decoder_units
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.tau = tau
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.update_frequency = update_frequency
        self.target_update_frequency = target_update_frequency
        
        # Experience replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size, batch_size)
        
        # Build networks
        self._build_networks()
        self._build_training_ops()
        
        # Initialize target networks
        self.update_target_networks(tau=1.0)
        
        # Training counters
        self.update_count = 0
        self.episode_count = 0
        
    def _build_networks(self):
        """Build Actor and Critic networks with Graph2Seq encoder"""
        
        # Placeholders
        self.obs_ph = tf.placeholder(tf.float32, [None, None, self.obs_dim], name='observations')
        self.actions_ph = tf.placeholder(tf.int32, [None, None], name='actions')
        self.rewards_ph = tf.placeholder(tf.float32, [None, None], name='rewards')
        self.next_obs_ph = tf.placeholder(tf.float32, [None, None, self.obs_dim], name='next_observations')
        self.dones_ph = tf.placeholder(tf.bool, [None, None], name='dones')
        self.sequence_length_ph = tf.placeholder(tf.int32, [None], name='sequence_length')
        
        # Actor Network (Policy)
        with tf.variable_scope('actor'):
            self.actor_policy = self._build_actor_network(self.obs_ph, self.sequence_length_ph)
            
        # Critic Network (Value Function)
        with tf.variable_scope('critic'):
            self.critic_value = self._build_critic_network(self.obs_ph, self.sequence_length_ph)
            
        # Target Networks
        with tf.variable_scope('target_actor'):
            self.target_actor_policy = self._build_actor_network(self.obs_ph, self.sequence_length_ph)
            
        with tf.variable_scope('target_critic'):
            self.target_critic_value = self._build_critic_network(self.obs_ph, self.sequence_length_ph)
    
    def _build_actor_network(self, obs, sequence_length):
        """Build Actor network using Graph2Seq encoder + LSTM decoder"""
        from policies.graph2seq_encoder import create_graph2seq_encoder
        
        # Graph2Seq Encoder
        encoder_outputs, encoder_state = create_graph2seq_encoder(
            encoder_inputs=obs,
            encoder_units=self.encoder_units,
            num_layers=self.num_layers,
            is_bidirectional=True,
            mode='train',
            scope_name="actor_encoder"
        )
        
        # LSTM Decoder for action sequence
        with tf.variable_scope("actor_decoder"):
            # Create LSTM cell
            lstm_cell = tf.nn.rnn_cell.LSTMCell(self.decoder_units)
            
            # Attention mechanism
            attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                self.decoder_units, encoder_outputs
            )
            
            decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
                lstm_cell, attention_mechanism,
                attention_layer_size=self.decoder_units
            )
            
            # Initial state
            decoder_initial_state = decoder_cell.zero_state(
                tf.shape(obs)[0], dtype=tf.float32
            ).clone(cell_state=encoder_state)
            
            # Helper for training
            helper = tf.contrib.seq2seq.TrainingHelper(
                tf.zeros([tf.shape(obs)[0], sequence_length[0], self.action_dim]),
                sequence_length
            )
            
            # Decoder
            decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=decoder_cell,
                helper=helper,
                initial_state=decoder_initial_state,
                output_layer=tf.layers.Dense(self.action_dim)
            )
            
            outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder)
            
            # Get logits and probabilities
            logits = outputs.rnn_output
            action_probs = tf.nn.softmax(logits)
            
        # Reshape logits for multinomial sampling
        batch_size = tf.shape(logits)[0]
        seq_len = tf.shape(logits)[1]
        action_dim = tf.shape(logits)[2]
        
        # Reshape to [batch_size * seq_len, action_dim] for multinomial
        logits_reshaped = tf.reshape(logits, [-1, action_dim])
        
        # Sample actions - multinomial returns [batch_size * seq_len, 1]
        sampled_actions = tf.multinomial(logits_reshaped, 1)
        
        # Reshape back to [batch_size, seq_len] and squeeze the last dimension
        sampled_actions = tf.reshape(sampled_actions, [batch_size, seq_len, 1])
        sampled_actions = tf.squeeze(sampled_actions, axis=2)  # Remove the last dimension
        
        return {
            'logits': logits,
            'probs': action_probs,
            'sample': sampled_actions
        }
    
    def _build_critic_network(self, obs, sequence_length):
        """Build Critic network using Graph2Seq encoder + value head"""
        from policies.graph2seq_encoder import create_graph2seq_encoder
        
        # Graph2Seq Encoder
        encoder_outputs, encoder_state = create_graph2seq_encoder(
            encoder_inputs=obs,
            encoder_units=self.encoder_units,
            num_layers=self.num_layers,
            is_bidirectional=True,
            mode='train',
            scope_name="critic_encoder"
        )
        
        # Value head
        with tf.variable_scope("critic_value"):
            # Global average pooling over sequence
            pooled_output = tf.reduce_mean(encoder_outputs, axis=1)
            
            # Dense layers for value estimation
            hidden = tf.layers.dense(pooled_output, 256, activation=tf.nn.relu)
            hidden = tf.layers.dense(hidden, 128, activation=tf.nn.relu)
            values = tf.layers.dense(hidden, 1, activation=None)
            
        return values
    
    def _build_training_ops(self):
        """Build training operations for Actor-Critic"""
        
        # Actor loss (Policy Gradient)
        with tf.variable_scope('actor_loss'):
            # Get action probabilities
            action_probs = self.actor_policy['probs']
            
            # One-hot encode actions
            actions_one_hot = tf.one_hot(self.actions_ph, self.action_dim)
            
            # Compute log probabilities
            log_probs = tf.log(action_probs + 1e-8)
            selected_log_probs = tf.reduce_sum(actions_one_hot * log_probs, axis=-1)
            
            # Compute advantages (TD error)
            # Expand critic_value to match rewards_ph shape [batch_size, seq_len]
            critic_values_expanded = tf.tile(self.critic_value, [1, tf.shape(self.rewards_ph)[1]])
            advantages = self.rewards_ph - tf.stop_gradient(critic_values_expanded)
            
            # Actor loss (negative log probability weighted by advantages)
            self.actor_loss = -tf.reduce_mean(selected_log_probs * advantages)
        
        # Critic loss (Mean Squared Error)
        with tf.variable_scope('critic_loss'):
            # Expand target_critic_value to match rewards_ph shape [batch_size, seq_len]
            target_values_expanded = tf.tile(self.target_critic_value, [1, tf.shape(self.rewards_ph)[1]])
            target_values = self.rewards_ph + self.gamma * tf.stop_gradient(
                target_values_expanded * (1.0 - tf.cast(self.dones_ph, tf.float32))
            )
            # Expand critic_value to match target_values shape
            critic_values_expanded = tf.tile(self.critic_value, [1, tf.shape(self.rewards_ph)[1]])
            self.critic_loss = tf.reduce_mean(tf.square(critic_values_expanded - target_values))
        
        # Combined loss
        self.total_loss = self.actor_loss + self.critic_loss
        
        # Optimizers
        self.actor_optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.critic_optimizer = tf.train.AdamOptimizer(self.learning_rate)
        
        # Training operations
        self.actor_train_op = self.actor_optimizer.minimize(self.actor_loss)
        self.critic_train_op = self.critic_optimizer.minimize(self.critic_loss)
        
        # Target network update operations
        self.update_target_ops = self._build_target_update_ops()
    
    def _build_target_update_ops(self):
        """Build operations for updating target networks"""
        update_ops = []
        
        # Get all trainable variables
        actor_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor')
        critic_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic')
        target_actor_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target_actor')
        target_critic_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target_critic')
        
        # Create update operations
        for var, target_var in zip(actor_vars, target_actor_vars):
            update_ops.append(target_var.assign(self.tau * var + (1 - self.tau) * target_var))
            
        for var, target_var in zip(critic_vars, target_critic_vars):
            update_ops.append(target_var.assign(self.tau * var + (1 - self.tau) * target_var))
        
        return update_ops
    
    def get_action(self, obs, sequence_length, training=True):
        """Get action from policy"""
        sess = tf.get_default_session()
        
        if training and random.random() < self.epsilon:
            # Random action for exploration
            batch_size = obs.shape[0]
            seq_len = sequence_length[0]
            return np.random.randint(0, self.action_dim, size=(batch_size, seq_len))
        
        # Use policy network
        feed_dict = {
            self.obs_ph: obs,
            self.sequence_length_ph: sequence_length
        }
        
        if training:
            actions = sess.run(self.actor_policy['sample'], feed_dict=feed_dict)
        else:
            # Greedy action selection
            probs = sess.run(self.actor_policy['probs'], feed_dict=feed_dict)
            actions = np.argmax(probs, axis=-1)
        
        return actions
    
    def store_experience(self, obs, actions, rewards, next_obs, dones, sequence_length):
        """Store experience in replay buffer"""
        self.replay_buffer.add(obs, actions, rewards, next_obs, dones, sequence_length)
    
    def update(self):
        """Update networks using experience replay"""
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Sample batch from replay buffer
        batch = self.replay_buffer.sample()
        
        # Update networks
        sess = tf.get_default_session()
        
        # Update actor and critic
        feed_dict = {
            self.obs_ph: batch['obs'],
            self.actions_ph: batch['actions'],
            self.rewards_ph: batch['rewards'],
            self.next_obs_ph: batch['next_obs'],
            self.dones_ph: batch['dones'],
            self.sequence_length_ph: batch['sequence_length']
        }
        
        _, actor_loss, critic_loss = sess.run([
            [self.actor_train_op, self.critic_train_op],
            self.actor_loss,
            self.critic_loss
        ], feed_dict=feed_dict)
        
        # Update target networks
        self.update_count += 1
        if self.update_count % self.target_update_frequency == 0:
            sess.run(self.update_target_ops)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay
        
        return actor_loss, critic_loss
    
    def update_target_networks(self, tau=1.0):
        """Update target networks"""
        sess = tf.get_default_session()
        old_tau = self.tau
        self.tau = tau
        sess.run(self.update_target_ops)
        self.tau = old_tau
    
    def save_model(self, save_path):
        """Save model parameters"""
        sess = tf.get_default_session()
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        ps = sess.run(variables)
        save_dict = {v.name: value for v, value in zip(variables, ps)}
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(save_dict, save_path)
    
    def load_model(self, load_path):
        """Load model parameters"""
        sess = tf.get_default_session()
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        loaded_params = joblib.load(load_path)
        
        restores = []
        for v in variables:
            if v.name in loaded_params:
                restores.append(v.assign(loaded_params[v.name]))
        
        sess.run(restores)


class ReplayBuffer:
    """Experience Replay Buffer for Deep RL"""
    
    def __init__(self, capacity, batch_size):
        self.capacity = capacity
        self.batch_size = batch_size
        self.buffer = deque(maxlen=capacity)
    
    def add(self, obs, actions, rewards, next_obs, dones, sequence_length):
        """Add experience to buffer"""
        experience = {
            'obs': obs,
            'actions': actions,
            'rewards': rewards,
            'next_obs': next_obs,
            'dones': dones,
            'sequence_length': sequence_length
        }
        self.buffer.append(experience)
    
    def sample(self):
        """Sample batch from buffer"""
        batch = random.sample(self.buffer, self.batch_size)
        
        # Stack experiences
        return {
            'obs': np.stack([e['obs'] for e in batch]),
            'actions': np.stack([e['actions'] for e in batch]),
            'rewards': np.stack([e['rewards'] for e in batch]),
            'next_obs': np.stack([e['next_obs'] for e in batch]),
            'dones': np.stack([e['dones'] for e in batch]),
            'sequence_length': np.stack([e['sequence_length'] for e in batch])
        }
    
    def __len__(self):
        return len(self.buffer)
