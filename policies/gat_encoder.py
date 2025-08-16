"""
Graph Attention Network (GAT) Encoder implementation for MRLCO.
This implementation follows the GAT architecture from "Graph Attention Networks" (Velickovic et al., 2018).
"""

import tensorflow as tf
import numpy as np
from .base_encoder import BaseEncoder


class GATEncoder(BaseEncoder):
    """
    Graph Attention Network (GAT) encoder that implements BaseEncoder interface.
    
    The GAT encoder applies multiple attention layers to learn representations
    from graph-structured input data. It supports multiple attention heads
    and can be configured for different aggregation strategies.
    
    Args:
        input_dim (int): Input feature dimension per node
        hidden_dim (int): Hidden dimension for each attention head
        num_heads (int): Number of attention heads per layer
        num_layers (int): Number of GAT layers
        concat (bool): Whether to concatenate or average multiple heads
        dropout (float): Dropout rate for training
        mode (str): Training mode ('train', 'eval', 'infer')
        
    Input Shape:
        encoder_inputs: [batch_size, num_nodes, input_dim]
        
    Output Shape:
        encoder_outputs: [batch_size, num_nodes, output_dim]
        encoder_state: Compatible state for decoder initialization
        
    Note:
        For sequence-to-sequence compatibility, this encoder treats
        sequences as fully connected graphs where each position
        connects to all other positions.
    """
    
    def __init__(self, input_dim, hidden_dim, num_heads=8, num_layers=2, 
                 concat=True, dropout=0.1, mode='train', decoder_num_layers=None):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.concat = concat
        self.dropout = dropout if mode == 'train' else 0.0
        self.mode = mode
        # Use decoder_num_layers if provided, otherwise use encoder num_layers
        self.decoder_num_layers = decoder_num_layers if decoder_num_layers is not None else num_layers
        
        # Derived parameters
        if self.concat:
            self.output_dim_per_layer = self.hidden_dim * self.num_heads
        else:
            self.output_dim_per_layer = self.hidden_dim
            
        self.final_output_dim = self.output_dim_per_layer
        
    def encode(self, encoder_inputs):
        """
        Build GAT encoder graph and return (outputs, state).
        
        Args:
            encoder_inputs: Tensor of shape [batch_size, num_nodes, input_dim]
            
        Returns:
            tuple: (encoder_outputs, encoder_state)
                encoder_outputs: [batch_size, num_nodes, output_dim] 
                encoder_state: Pooled state for decoder initialization
        """
        batch_size = tf.shape(encoder_inputs)[0]
        num_nodes = tf.shape(encoder_inputs)[1]
        
        # Create adjacency matrix for fully connected graph
        # Each node connects to all nodes (including itself)
        adjacency_matrix = tf.ones([batch_size, num_nodes, num_nodes], dtype=tf.float32)
        
        # Initialize node features
        node_features = encoder_inputs
        
        # Apply GAT layers
        for layer_idx in range(self.num_layers):
            with tf.variable_scope(f"gat_layer_{layer_idx}"):
                node_features = self._gat_layer(
                    node_features, 
                    adjacency_matrix,
                    input_dim=node_features.get_shape()[-1].value or self.input_dim,
                    output_dim=self.hidden_dim,
                    num_heads=self.num_heads,
                    concat=self.concat and (layer_idx < self.num_layers - 1),  # Don't concat last layer if specified
                    dropout=self.dropout,
                    activation=tf.nn.elu if layer_idx < self.num_layers - 1 else None
                )
        
        encoder_outputs = node_features
        
        # Create encoder state for decoder compatibility
        # Use mean pooling over nodes to create graph-level representation
        graph_embedding = tf.reduce_mean(encoder_outputs, axis=1)  # [batch_size, output_dim]
        
        # Project to appropriate state size if needed
        state_dim = self.hidden_dim  # Assuming decoder expects this dimension
        if graph_embedding.get_shape()[-1].value != state_dim:
            with tf.variable_scope("state_projection"):
                graph_embedding = tf.layers.dense(
                    graph_embedding, 
                    state_dim, 
                    activation=None,
                    name="state_dense"
                )
        
        # Create LSTM-compatible state tuple for decoder
        # The decoder expects multi-layer state if decoder_num_layers > 1
        if self.decoder_num_layers == 1:
            encoder_state = tf.nn.rnn_cell.LSTMStateTuple(
                c=graph_embedding, 
                h=graph_embedding
            )
        else:
            # Create tuple of identical states for each layer
            encoder_state = tuple([
                tf.nn.rnn_cell.LSTMStateTuple(c=graph_embedding, h=graph_embedding)
                for _ in range(self.decoder_num_layers)
            ])
        
        return encoder_outputs, encoder_state
        
    def _gat_layer(self, node_features, adjacency_matrix, input_dim, output_dim, 
                   num_heads, concat=True, dropout=0.0, activation=None):
        """
        Apply single GAT layer with multiple attention heads.
        
        Args:
            node_features: [batch_size, num_nodes, input_dim]
            adjacency_matrix: [batch_size, num_nodes, num_nodes]
            input_dim: Input feature dimension
            output_dim: Output feature dimension per head
            num_heads: Number of attention heads
            concat: Whether to concatenate or average heads
            dropout: Dropout rate
            activation: Activation function
            
        Returns:
            Tensor: [batch_size, num_nodes, output_dim * num_heads] if concat
                   else [batch_size, num_nodes, output_dim]
        """
        batch_size = tf.shape(node_features)[0]
        num_nodes = tf.shape(node_features)[1]
        
        # Collect outputs from all attention heads
        head_outputs = []
        
        for head_idx in range(num_heads):
            with tf.variable_scope(f"attention_head_{head_idx}"):
                head_output = self._attention_head(
                    node_features, 
                    adjacency_matrix,
                    input_dim, 
                    output_dim,
                    dropout
                )
                head_outputs.append(head_output)
        
        # Combine heads
        if concat:
            # Concatenate all heads: [batch_size, num_nodes, output_dim * num_heads]
            combined_output = tf.concat(head_outputs, axis=-1)
        else:
            # Average all heads: [batch_size, num_nodes, output_dim]
            combined_output = tf.reduce_mean(tf.stack(head_outputs, axis=-1), axis=-1)
        
        # Apply activation
        if activation is not None:
            combined_output = activation(combined_output)
            
        return combined_output
    
    def _attention_head(self, node_features, adjacency_matrix, input_dim, output_dim, dropout=0.0):
        """
        Apply single attention head computation.
        
        Implements the GAT attention mechanism:
        1. Linear transformation: W * h_i
        2. Attention coefficients: LeakyReLU(a^T [W*h_i || W*h_j])
        3. Softmax normalization: alpha_ij = softmax_j(e_ij)
        4. Weighted aggregation: h'_i = sum_j(alpha_ij * W*h_j)
        
        Args:
            node_features: [batch_size, num_nodes, input_dim]
            adjacency_matrix: [batch_size, num_nodes, num_nodes]
            input_dim: Input feature dimension
            output_dim: Output feature dimension
            dropout: Dropout rate
            
        Returns:
            Tensor: [batch_size, num_nodes, output_dim]
        """
        batch_size = tf.shape(node_features)[0]
        num_nodes = tf.shape(node_features)[1]
        
        # Linear transformation: W * h
        # Weight matrix: [input_dim, output_dim]
        W = tf.get_variable(
            "weight_matrix",
            shape=[input_dim, output_dim],
            initializer=tf.glorot_uniform_initializer()
        )
        
        # Apply transformation: [batch_size, num_nodes, output_dim]
        transformed_features = tf.tensordot(node_features, W, axes=[[2], [0]])
        
        # Attention mechanism
        # Attention vector: [2 * output_dim, 1]
        attention_vector = tf.get_variable(
            "attention_vector",
            shape=[2 * output_dim, 1],
            initializer=tf.glorot_uniform_initializer()
        )
        
        # Compute attention coefficients
        # For each pair (i,j), compute e_ij = LeakyReLU(a^T [W*h_i || W*h_j])
        
        # Prepare features for attention computation
        # [batch_size, num_nodes, 1, output_dim] - for broadcasting
        features_i = tf.expand_dims(transformed_features, axis=2)
        # [batch_size, 1, num_nodes, output_dim] - for broadcasting  
        features_j = tf.expand_dims(transformed_features, axis=1)
        
        # Concatenate features: [batch_size, num_nodes, num_nodes, 2 * output_dim]
        concatenated_features = tf.concat([
            tf.tile(features_i, [1, 1, num_nodes, 1]),  # Repeat i for all j
            tf.tile(features_j, [1, num_nodes, 1, 1])   # Repeat j for all i
        ], axis=-1)
        
        # Compute attention coefficients: [batch_size, num_nodes, num_nodes, 1]
        attention_logits = tf.tensordot(
            concatenated_features, 
            attention_vector, 
            axes=[[3], [0]]
        )
        
        # Apply LeakyReLU
        attention_logits = tf.nn.leaky_relu(attention_logits, alpha=0.2)
        
        # Remove last dimension: [batch_size, num_nodes, num_nodes]
        attention_logits = tf.squeeze(attention_logits, axis=-1)
        
        # Mask attention logits using adjacency matrix
        # Set attention to very negative value for non-connected nodes
        mask = tf.equal(adjacency_matrix, 0.0)
        masked_attention_logits = tf.where(
            mask,
            tf.fill(tf.shape(attention_logits), -1e9),
            attention_logits
        )
        
        # Apply softmax to get attention weights: [batch_size, num_nodes, num_nodes]
        attention_weights = tf.nn.softmax(masked_attention_logits, axis=-1)
        
        # Apply dropout to attention weights during training
        if dropout > 0.0:
            attention_weights = tf.layers.dropout(
                attention_weights, 
                rate=dropout, 
                training=(self.mode == 'train')
            )
        
        # Weighted aggregation: [batch_size, num_nodes, output_dim]
        # attention_weights: [batch_size, num_nodes, num_nodes]
        # transformed_features: [batch_size, num_nodes, output_dim]
        output_features = tf.matmul(attention_weights, transformed_features)
        
        # Apply dropout to output features during training
        if dropout > 0.0:
            output_features = tf.layers.dropout(
                output_features,
                rate=dropout,
                training=(self.mode == 'train')
            )
        
        return output_features
    
    def get_output_dim(self):
        """
        Return the final output feature dimension of the encoder.
        
        Returns:
            int: Output dimension size
        """
        return self.final_output_dim