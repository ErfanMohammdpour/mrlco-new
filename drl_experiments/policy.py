"""
DRL Policy implementation with autoregressive decoder for task offloading.
"""

import numpy as np
import tensorflow as tf
from policies.graph2seq_encoder import create_graph2seq_encoder


class DRLPolicy:
    """
    Deep RL Policy for task offloading with autoregressive action generation.
    """
    
    def __init__(self, obs_dim=17, action_dim=2, encoder_units=128, decoder_units=128, 
                 num_layers=1, scope_name="drl_policy"):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.encoder_units = encoder_units
        self.decoder_units = decoder_units
        self.num_layers = num_layers
        self.scope_name = scope_name
        
        # Build policy components
        self._build_encoder()
        self._build_decoder()
        self._build_heads()
        
        # Collect trainable variables
        self.trainable_variables = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope_name
        )
    
    def _build_encoder(self):
        """Build encoder for processing observations."""
        with tf.variable_scope(self.scope_name + "/encoder"):
            # Input projection layer
            self.input_projection = tf.layers.Dense(
                self.encoder_units, 
                activation=tf.nn.relu,
                name="input_projection"
            )
            
            # Initialize with dummy input to create variables
            dummy_input = tf.placeholder(tf.float32, [None, None, self.obs_dim], name="dummy_obs")
            _ = self.input_projection(dummy_input)
    
    def _build_decoder(self):
        """Build autoregressive decoder for action generation."""
        with tf.variable_scope(self.scope_name + "/decoder"):
            # LSTM cell for autoregressive generation
            self.lstm_cell = tf.nn.rnn_cell.LSTMCell(self.decoder_units)
    
    def _build_heads(self):
        """Build policy and value heads."""
        with tf.variable_scope(self.scope_name + "/heads"):
            # Policy head (binary classification)
            self.policy_head = tf.layers.Dense(self.action_dim, name="policy_head")
            
            # Value head (scalar)
            self.value_head = tf.layers.Dense(1, name="value_head")
            
            # Initialize the heads with dummy input to create variables
            dummy_input = tf.placeholder(tf.float32, [None, self.decoder_units], name="dummy_input")
            _ = self.policy_head(dummy_input)
            _ = self.value_head(dummy_input)
    
    def _apply_ready_mask(self, logits, timestep):
        """
        Apply ready mask to logits based on timestep.
        In our DAG, prioritize_sequence ensures topological order,
        so node at timestep t should always be ready.
        """
        # For now, we assume all nodes are ready (topological order)
        # This can be extended to implement actual ready masking
        return logits
    
    def sample_action(self, obs, timestep):
        """
        Sample action for given timestep autoregressively.
        
        Args:
            obs: observations [B, T, F] where T is current timestep + 1
            timestep: current timestep (0-indexed)
            
        Returns:
            action: sampled action [B]
            log_prob: log probability of action [B]
            value: value estimate [B]
        """
        with tf.variable_scope(self.scope_name, reuse=tf.AUTO_REUSE):
            # Convert numpy array to tensor if needed
            if isinstance(obs, np.ndarray):
                obs = tf.constant(obs, dtype=tf.float32)
            
            # Input projection to match encoder units
            projected_obs = self.input_projection(obs)
            
            # Get encoder outputs
            encoder_outputs, encoder_state = create_graph2seq_encoder(
                encoder_inputs=projected_obs,
                encoder_units=self.encoder_units,
                num_layers=self.num_layers,
                is_bidirectional=True,
                mode='train',
                scope_name="drl_encoder"
            )
            
            # Simple approach: use encoder output at the desired timestep
            seq_len = obs.shape[1] if hasattr(obs, 'shape') else tf.shape(obs)[1]
            
            # Ensure timestep is within bounds
            if isinstance(timestep, int):
                if isinstance(seq_len, tf.Tensor):
                    seq_len_val = tf.cast(seq_len, tf.int32)
                    actual_timestep = tf.minimum(timestep, seq_len_val - 1)
                else:
                    actual_timestep = min(timestep, seq_len - 1)
            else:
                actual_timestep = tf.minimum(timestep, seq_len - 1)
            
            # Use encoder output directly (no decoder for simplicity)
            final_output = encoder_outputs[:, actual_timestep, :]
            
            # Policy head
            logits = self.policy_head(final_output)  # [B, action_dim]
            
            # Apply ready mask
            logits = self._apply_ready_mask(logits, timestep)
            
            # Sample action
            action_probs = tf.nn.softmax(logits)
            action = tf.multinomial(logits, 1)[:, 0]  # [B]
            
            # Compute log probability
            log_prob = tf.log(action_probs + 1e-8)
            action_one_hot = tf.one_hot(action, self.action_dim)
            log_prob = tf.reduce_sum(action_one_hot * log_prob, axis=1)  # [B]
            
            # Value head
            value = self.value_head(final_output)[:, 0]  # [B]
            
            return action, log_prob, value
    
    def evaluate_actions(self, obs, actions):
        """
        Evaluate actions for given observations.
        
        Args:
            obs: observations [B, T, F]
            actions: actions [B, T]
            
        Returns:
            log_probs: log probabilities [B, T]
            entropy: entropy [B, T]
            values: value estimates [B, T]
        """
        with tf.variable_scope(self.scope_name, reuse=tf.AUTO_REUSE):
            # Convert numpy arrays to tensors if needed
            if isinstance(obs, np.ndarray):
                obs = tf.constant(obs, dtype=tf.float32)
            if isinstance(actions, np.ndarray):
                actions = tf.constant(actions, dtype=tf.int32)
            
            # Input projection to match encoder units
            projected_obs = self.input_projection(obs)
            
            # Get encoder outputs
            encoder_outputs, encoder_state = create_graph2seq_encoder(
                encoder_inputs=projected_obs,
                encoder_units=self.encoder_units,
                num_layers=self.num_layers,
                is_bidirectional=True,
                mode='train',
                scope_name="drl_encoder"
            )
            
            # Simple approach: use encoder outputs directly
            decoder_outputs = encoder_outputs  # [B, T, F]
            
            # Process all timesteps
            # Reshape decoder_outputs for processing
            feat_dim = tf.shape(decoder_outputs)[2]
            
            # Reshape to [B*T, F] for processing
            decoder_outputs_flat = tf.reshape(decoder_outputs, [-1, feat_dim])
            
            # Apply policy head
            logits_flat = self.policy_head(decoder_outputs_flat)  # [B*T, action_dim]
            
            # Reshape back to [B, T, action_dim]
            logits = tf.reshape(logits_flat, [tf.shape(decoder_outputs)[0], tf.shape(decoder_outputs)[1], self.action_dim])
            
            # Compute probabilities
            action_probs = tf.nn.softmax(logits)
            
            # Compute log probabilities
            log_prob = tf.log(action_probs + 1e-8)
            action_one_hot = tf.one_hot(actions, self.action_dim)
            log_probs = tf.reduce_sum(action_one_hot * log_prob, axis=2)  # [B, T]
            
            # Compute entropy
            entropy = -tf.reduce_sum(action_probs * tf.log(action_probs + 1e-8), axis=2)  # [B, T]
            
            # Value head
            values_flat = self.value_head(decoder_outputs_flat)[:, 0]  # [B*T]
            values = tf.reshape(values_flat, [tf.shape(decoder_outputs)[0], tf.shape(decoder_outputs)[1]])  # [B, T]
            
            return log_probs, entropy, values
    
    def get_value(self, obs):
        """
        Get value estimates for observations.
        
        Args:
            obs: observations [B, T, F]
            
        Returns:
            values: value estimates [B, T]
        """
        # Create dummy actions for evaluation
        batch_size = tf.shape(obs)[0]
        seq_len = tf.shape(obs)[1]
        dummy_actions = tf.zeros([batch_size, seq_len], dtype=tf.int32)
        
        _, _, values = self.evaluate_actions(obs, dummy_actions)
        return values
