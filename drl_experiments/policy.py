"""
DRL Policy implementation with advanced graph-aware encoder for task offloading.
"""

import numpy as np
import tensorflow as tf


class GraphConvolutionLayer:
    """Graph Convolution Layer similar to Graph2Seq aggregators."""
    
    def __init__(self, input_dim, output_dim, activation=tf.nn.relu, name="gcn"):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation
        self.name = name
        
        with tf.variable_scope(name):
            # Self weights
            self.self_weights = tf.get_variable(
                "self_weights", 
                [input_dim, output_dim],
                initializer=tf.glorot_uniform_initializer()
            )
            
            # Neighbor weights  
            self.neigh_weights = tf.get_variable(
                "neigh_weights",
                [input_dim, output_dim], 
                initializer=tf.glorot_uniform_initializer()
            )
            
            # Bias
            self.bias = tf.get_variable(
                "bias",
                [output_dim],
                initializer=tf.zeros_initializer()
            )
    
    def __call__(self, node_features, adjacency_matrix):
        """
        Apply graph convolution.
        
        Args:
            node_features: [num_nodes, input_dim]
            adjacency_matrix: [num_nodes, num_nodes]
            
        Returns:
            output: [num_nodes, output_dim]
        """
        # Self connection
        self_output = tf.matmul(node_features, self.self_weights)
        
        # Neighbor aggregation
        neigh_features = tf.matmul(adjacency_matrix, node_features)  # [num_nodes, input_dim]
        neigh_output = tf.matmul(neigh_features, self.neigh_weights)
        
        # Combine and add bias
        output = self_output + neigh_output + self.bias
        
        if self.activation:
            output = self.activation(output)
            
        return output


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
        """Build advanced graph-aware encoder."""
        with tf.variable_scope(self.scope_name + "/encoder"):
            # Input projection layer
            self.input_projection = tf.layers.Dense(
                self.encoder_units, 
                activation=tf.nn.relu,
                name="input_projection"
            )
            
            # Graph Convolution layers (like Graph2Seq)
            self.gcn_layers = []
            for i in range(2):  # 2 GCN layers
                layer = GraphConvolutionLayer(
                    input_dim=self.encoder_units,
                    output_dim=self.encoder_units,
                    activation=tf.nn.relu,
                    name=f"gcn_layer_{i}"
                )
                self.gcn_layers.append(layer)
            
            # Attention mechanism for graph nodes
            self.node_attention = tf.layers.Dense(1, name="node_attention")
            
            # LSTM encoder for temporal modeling
            self.lstm_encoder = tf.nn.rnn_cell.LSTMCell(self.encoder_units)
            
            # Initialize with dummy input to create variables
            dummy_input = tf.placeholder(tf.float32, [None, None, self.obs_dim], name="dummy_obs")
            projected = self.input_projection(dummy_input)
            
            # Initialize GCN layers
            batch_size = tf.shape(projected)[0]
            seq_len = tf.shape(projected)[1]
            dummy_graph_features = tf.reshape(projected, [-1, self.encoder_units])
            dummy_adj = self._create_dummy_adjacency(seq_len)
            
            for layer in self.gcn_layers:
                dummy_graph_features = layer(dummy_graph_features, dummy_adj)
            
            # Initialize attention
            _ = self.node_attention(dummy_graph_features)
            
            # Initialize LSTM
            _, _ = tf.nn.dynamic_rnn(self.lstm_encoder, projected, dtype=tf.float32)
    
    def _create_dummy_adjacency(self, seq_len):
        """Create dummy adjacency matrix for initialization."""
        # Create a simple adjacency matrix (fully connected)
        adj = tf.ones([seq_len, seq_len], dtype=tf.float32)
        # Normalize by degree
        degree = tf.reduce_sum(adj, axis=1, keepdims=True)
        adj_normalized = adj / (degree + 1e-8)
        return adj_normalized
    
    def _create_task_adjacency(self, seq_len):
        """Create task-specific adjacency matrix based on DAG structure."""
        # For task offloading, create adjacency based on task dependencies
        # This simulates the DAG structure where tasks have dependencies
        
        # Create lower triangular matrix (tasks depend on previous tasks)
        indices = tf.range(seq_len)
        i_indices, j_indices = tf.meshgrid(indices, indices, indexing='ij')
        
        # Task i can depend on task j if j < i (topological order)
        mask = tf.cast(j_indices <= i_indices, tf.float32)
        
        # Add some randomness to make it more realistic
        random_mask = tf.random.uniform([seq_len, seq_len]) > 0.3
        mask = mask * tf.cast(random_mask, tf.float32)
        
        # Normalize by degree
        degree = tf.reduce_sum(mask, axis=1, keepdims=True)
        adj_normalized = mask / (degree + 1e-8)
        
        return adj_normalized
    
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
            # encoder_outputs has dimension encoder_units (LSTM)
            dummy_input = tf.placeholder(tf.float32, [None, self.encoder_units], name="dummy_input")
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
            
            # Get sequence length
            seq_len = obs.shape[1] if hasattr(obs, 'shape') else tf.shape(obs)[1]
            batch_size = tf.shape(obs)[0]
            
            # Create task-specific adjacency matrix
            adj_matrix = self._create_task_adjacency(seq_len)
            
            # Apply Graph Convolution layers
            # Reshape to [batch_size * seq_len, encoder_units] for GCN
            graph_features = tf.reshape(projected_obs, [-1, self.encoder_units])
            
            # Apply GCN layers
            for layer in self.gcn_layers:
                graph_features = layer(graph_features, adj_matrix)
            
            # Apply node attention
            attention_scores = self.node_attention(graph_features)  # [batch_size * seq_len, 1]
            attention_weights = tf.nn.softmax(tf.reshape(attention_scores, [batch_size, seq_len]))
            
            # Reshape back to [batch_size, seq_len, encoder_units]
            graph_outputs = tf.reshape(graph_features, [batch_size, seq_len, self.encoder_units])
            
            # Apply attention
            attended_outputs = graph_outputs * tf.expand_dims(attention_weights, -1)
            
            # Apply LSTM for temporal modeling
            encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
                self.lstm_encoder, 
                attended_outputs, 
                dtype=tf.float32
            )
            
            # Ensure timestep is within bounds
            if isinstance(timestep, int):
                if isinstance(seq_len, tf.Tensor):
                    seq_len_val = tf.cast(seq_len, tf.int32)
                    actual_timestep = tf.minimum(timestep, seq_len_val - 1)
                else:
                    actual_timestep = min(timestep, seq_len - 1)
            else:
                actual_timestep = tf.minimum(timestep, seq_len - 1)
            
            # Use encoder output directly
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
            
            # Get sequence length
            seq_len = obs.shape[1] if hasattr(obs, 'shape') else tf.shape(obs)[1]
            batch_size = tf.shape(obs)[0]
            
            # Create task-specific adjacency matrix
            adj_matrix = self._create_task_adjacency(seq_len)
            
            # Apply Graph Convolution layers
            # Reshape to [batch_size * seq_len, encoder_units] for GCN
            graph_features = tf.reshape(projected_obs, [-1, self.encoder_units])
            
            # Apply GCN layers
            for layer in self.gcn_layers:
                graph_features = layer(graph_features, adj_matrix)
            
            # Apply node attention
            attention_scores = self.node_attention(graph_features)  # [batch_size * seq_len, 1]
            attention_weights = tf.nn.softmax(tf.reshape(attention_scores, [batch_size, seq_len]))
            
            # Reshape back to [batch_size, seq_len, encoder_units]
            graph_outputs = tf.reshape(graph_features, [batch_size, seq_len, self.encoder_units])
            
            # Apply attention
            attended_outputs = graph_outputs * tf.expand_dims(attention_weights, -1)
            
            # Apply LSTM for temporal modeling
            encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
                self.lstm_encoder, 
                attended_outputs, 
                dtype=tf.float32
            )
            
            # Use encoder outputs directly
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
