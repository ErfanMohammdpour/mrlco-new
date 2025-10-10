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
            # Placeholder for observations
            self.obs_ph = tf.placeholder(tf.float32, [None, None, self.obs_dim], name='observations')
            
            # Use existing Graph2Seq encoder
            self.encoder_outputs, self.encoder_state = create_graph2seq_encoder(
                encoder_inputs=self.obs_ph,
                encoder_units=self.encoder_units,
                num_layers=self.num_layers,
                is_bidirectional=True,
                mode='train',
                scope_name="drl_encoder"
            )
    
    def _build_decoder(self):
        """Build autoregressive decoder for action generation."""
        with tf.variable_scope(self.scope_name + "/decoder"):
            # LSTM cell for autoregressive generation
            self.lstm_cell = tf.nn.rnn_cell.LSTMCell(self.decoder_units)
            
            # Attention mechanism
            self.attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                self.decoder_units, self.encoder_outputs
            )
            
            self.decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
                self.lstm_cell, self.attention_mechanism,
                attention_layer_size=self.decoder_units
            )
            
            # Initial state
            self.decoder_initial_state = self.decoder_cell.zero_state(
                tf.shape(self.obs_ph)[0], dtype=tf.float32
            ).clone(cell_state=self.encoder_state)
    
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
            
            # Get encoder outputs
            encoder_outputs, encoder_state = create_graph2seq_encoder(
                encoder_inputs=obs,
                encoder_units=self.encoder_units,
                num_layers=self.num_layers,
                is_bidirectional=True,
                mode='train',
                scope_name="drl_encoder"
            )
            
            # Initialize decoder state
            decoder_state = self.decoder_cell.zero_state(
                tf.shape(obs)[0], dtype=tf.float32
            ).clone(cell_state=encoder_state)
            
            # Run decoder for timestep steps
            for t in range(timestep + 1):
                if t == 0:
                    # First timestep: use encoder output
                    decoder_input = encoder_outputs[:, t, :]
                else:
                    # Subsequent timesteps: use previous action embedding
                    # For simplicity, we'll use encoder output at current timestep
                    decoder_input = encoder_outputs[:, t, :]
                
                # Run decoder step
                decoder_output, decoder_state = self.decoder_cell(
                    decoder_input, decoder_state
                )
            
            # Get final decoder output
            final_output = decoder_output
            
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
            
            batch_size = tf.shape(obs)[0]
            seq_len = tf.shape(obs)[1]
            
            # Get encoder outputs
            encoder_outputs, encoder_state = create_graph2seq_encoder(
                encoder_inputs=obs,
                encoder_units=self.encoder_units,
                num_layers=self.num_layers,
                is_bidirectional=True,
                mode='train',
                scope_name="drl_encoder"
            )
            
            # Initialize decoder state
            decoder_state = self.decoder_cell.zero_state(
                batch_size, dtype=tf.float32
            ).clone(cell_state=encoder_state)
            
            # Store outputs
            log_probs_list = []
            entropy_list = []
            values_list = []
            
            # Run decoder autoregressively
            for t in range(seq_len):
                if t == 0:
                    decoder_input = encoder_outputs[:, t, :]
                else:
                    decoder_input = encoder_outputs[:, t, :]
                
                # Run decoder step
                decoder_output, decoder_state = self.decoder_cell(
                    decoder_input, decoder_state
                )
                
                # Policy head
                logits = self.policy_head(decoder_output)  # [B, action_dim]
                logits = self._apply_ready_mask(logits, t)
                
                # Compute probabilities
                action_probs = tf.nn.softmax(logits)
                
                # Get action for this timestep
                action = actions[:, t]  # [B]
                
                # Compute log probability
                log_prob = tf.log(action_probs + 1e-8)
                action_one_hot = tf.one_hot(action, self.action_dim)
                log_prob = tf.reduce_sum(action_one_hot * log_prob, axis=1)  # [B]
                
                # Compute entropy
                entropy = -tf.reduce_sum(action_probs * tf.log(action_probs + 1e-8), axis=1)  # [B]
                
                # Value head
                value = self.value_head(decoder_output)[:, 0]  # [B]
                
                log_probs_list.append(log_prob)
                entropy_list.append(entropy)
                values_list.append(value)
            
            # Stack outputs
            log_probs = tf.stack(log_probs_list, axis=1)  # [B, T]
            entropy = tf.stack(entropy_list, axis=1)  # [B, T]
            values = tf.stack(values_list, axis=1)  # [B, T]
            
            return log_probs, entropy, values
    
    def get_value(self, obs):
        """
        Get value estimates for observations.
        
        Args:
            obs: observations [B, T, F]
            
        Returns:
            values: value estimates [B, T]
        """
        _, _, values = self.evaluate_actions(obs, tf.zeros_like(obs[:, :, 0]))
        return values
