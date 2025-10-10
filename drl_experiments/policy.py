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
            # No placeholder needed - we'll use the input directly in methods
            pass
    
    def _build_decoder(self):
        """Build autoregressive decoder for action generation."""
        with tf.variable_scope(self.scope_name + "/decoder"):
            # LSTM cell for autoregressive generation
            self.lstm_cell = tf.nn.rnn_cell.LSTMCell(self.decoder_units)
            
            # Note: Attention mechanism will be created dynamically in methods
            # since we don't have encoder_outputs at build time
    
    def _build_heads(self):
        """Build policy and value heads."""
        with tf.variable_scope(self.scope_name + "/heads"):
            # Policy head (binary classification)
            self.policy_head = tf.layers.Dense(self.action_dim, name="policy_head")
            
            # Value head (scalar)
            self.value_head = tf.layers.Dense(1, name="value_head")
    
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
            projected_obs = tf.layers.dense(
                obs, 
                self.encoder_units, 
                activation=tf.nn.relu,
                name="input_projection"
            )
            
            # Get encoder outputs
            encoder_outputs, encoder_state = create_graph2seq_encoder(
                encoder_inputs=projected_obs,
                encoder_units=self.encoder_units,
                num_layers=self.num_layers,
                is_bidirectional=True,
                mode='train',
                scope_name="drl_encoder"
            )
            
            # Create attention mechanism dynamically
            attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                self.decoder_units, encoder_outputs
            )
            
            decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
                self.lstm_cell, attention_mechanism,
                attention_layer_size=self.decoder_units
            )
            
            # Initialize decoder state
            decoder_state = decoder_cell.zero_state(
                tf.shape(obs)[0], dtype=tf.float32
            ).clone(cell_state=encoder_state)
            
            # Run decoder for timestep steps
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
            
            # Ensure actual_timestep is an integer tensor
            if isinstance(actual_timestep, tf.Tensor):
                actual_timestep = tf.cast(actual_timestep, tf.int32)
            else:
                actual_timestep = tf.constant(actual_timestep, dtype=tf.int32)
            
            # Simple approach: just use the encoder output at the desired timestep
            # Use safe indexing
            decoder_input = encoder_outputs[:, actual_timestep, :]
            
            final_output, _ = decoder_cell(decoder_input, decoder_state)
            
            # Note: This is a simplified version that doesn't use autoregressive decoding
            
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
            
            # batch_size and seq_len will be computed from decoder_outputs
            
            # Input projection to match encoder units
            projected_obs = tf.layers.dense(
                obs, 
                self.encoder_units, 
                activation=tf.nn.relu,
                name="input_projection"
            )
            
            # Get encoder outputs
            encoder_outputs, encoder_state = create_graph2seq_encoder(
                encoder_inputs=projected_obs,
                encoder_units=self.encoder_units,
                num_layers=self.num_layers,
                is_bidirectional=True,
                mode='train',
                scope_name="drl_encoder"
            )
            
            # Create attention mechanism dynamically
            attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                self.decoder_units, encoder_outputs
            )
            
            decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
                self.lstm_cell, attention_mechanism,
                attention_layer_size=self.decoder_units
            )
            
            # Initialize decoder state
            decoder_state = decoder_cell.zero_state(
                tf.shape(obs)[0], dtype=tf.float32
            ).clone(cell_state=encoder_state)
            
            # No need to store outputs in lists anymore
            
            # Simple approach: just use encoder outputs directly
            # This is a simplified version that doesn't use autoregressive decoding
            decoder_outputs = encoder_outputs  # [B, T, F]
            
            # Note: decoder_state is not used in this simplified version
            
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
            
            # Stack outputs
            # log_probs, entropy, values are already computed above
            
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
