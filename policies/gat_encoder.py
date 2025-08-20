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
                 concat=True, dropout=0.1, mode='train', decoder_num_layers=None,
                 pooling_method='mean_max', state_projection_dim=None):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.concat = concat
        self.dropout = dropout if mode == 'train' else 0.0
        self.mode = mode
        # Use decoder_num_layers if provided, otherwise use encoder num_layers
        self.decoder_num_layers = decoder_num_layers if decoder_num_layers is not None else num_layers
        self.pooling_method = pooling_method  # 'mean', 'max', or 'mean_max'
        self.state_projection_dim = state_projection_dim  # Optional custom state dimension
        
        # Derived parameters
        # Note: final_output_dim will be computed dynamically in get_output_dim()
        # based on whether the final layer concatenates or averages
        
    def encode(self, encoder_inputs, adjacency_matrix=None, edge_list=None, edge_weights=None):
        """
        Build GAT encoder graph and return (outputs, state).
        
        Args:
            encoder_inputs: Tensor of shape [batch_size, num_nodes, input_dim]
            adjacency_matrix: Optional tensor of shape [batch_size, num_nodes, num_nodes]
                             Values can be edge weights (e.g., latency, cost) or binary.
                             If None, will use fully connected graph.
            edge_list: Optional alternative to adjacency_matrix. List of edges as
                      [batch_size, num_edges, 2] where last dim is [source, target].
                      Will be converted to adjacency matrix internally.
            edge_weights: Optional tensor of shape [batch_size, num_edges] containing
                         the weight for each edge in edge_list. If None, uses 1.0.
            
        Returns:
            tuple: (encoder_outputs, encoder_state)
                encoder_outputs: [batch_size, num_nodes, output_dim] 
                encoder_state: Pooled state for decoder initialization
        """
        batch_size = tf.shape(encoder_inputs)[0]
        num_nodes = tf.shape(encoder_inputs)[1]
        
        # Handle adjacency input
        if adjacency_matrix is None and edge_list is None:
            # Default to fully connected graph if no adjacency provided
            adjacency_matrix = tf.ones([batch_size, num_nodes, num_nodes], dtype=tf.float32)
        elif edge_list is not None:
            # Convert edge list to adjacency matrix with optional weights
            adjacency_matrix = self._edge_list_to_adjacency(edge_list, batch_size, num_nodes, edge_weights)
        # else: use provided adjacency_matrix directly (can contain weights)
        
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
        # Implement more expressive pooling: mean + max concatenation
        if self.pooling_method == 'mean':
            graph_embedding = tf.reduce_mean(encoder_outputs, axis=1)  # [batch_size, output_dim]
        elif self.pooling_method == 'max':
            graph_embedding = tf.reduce_max(encoder_outputs, axis=1)  # [batch_size, output_dim]
        elif self.pooling_method == 'mean_max':
            # Concatenate mean and max pooling
            mean_pool = tf.reduce_mean(encoder_outputs, axis=1)  # [batch_size, output_dim]
            max_pool = tf.reduce_max(encoder_outputs, axis=1)    # [batch_size, output_dim]
            graph_embedding = tf.concat([mean_pool, max_pool], axis=-1)  # [batch_size, 2*output_dim]
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling_method}")
        
        # Project to appropriate state size if needed
        state_dim = self.state_projection_dim if self.state_projection_dim else self.hidden_dim
        current_dim = graph_embedding.get_shape()[-1].value
        if current_dim != state_dim:
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
        Apply single attention head computation with improved efficiency.
        
        Implements the GAT attention mechanism efficiently:
        1. Linear transformation: W * h_i
        2. Attention coefficients: LeakyReLU(a^T [W*h_i || W*h_j]) for neighbors only
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
        # Get dynamic shapes
        batch_size = tf.shape(node_features)[0]
        num_nodes = tf.shape(node_features)[1]
        
        # Ensure adjacency matrix has the correct batch size
        adj_batch_size = tf.shape(adjacency_matrix)[0]
        
        # Handle batch size mismatch by taking the minimum
        # This can happen when different batches have different sizes
        actual_batch_size = tf.minimum(batch_size, adj_batch_size)
        
        # Slice to ensure consistent batch sizes
        node_features = node_features[:actual_batch_size]
        adjacency_matrix = adjacency_matrix[:actual_batch_size]
        
        # Update batch size to the actual consistent size
        batch_size = actual_batch_size
        
        # Linear transformation: W * h
        W = tf.get_variable(
            "weight_matrix",
            shape=[input_dim, output_dim],
            initializer=tf.glorot_uniform_initializer()
        )
        
        # Apply transformation: [batch_size, num_nodes, output_dim]
        transformed_features = tf.tensordot(node_features, W, axes=[[2], [0]])
        
        # Efficient attention mechanism using masked computation
        # Use separate attention parameters for source and target
        attention_source = tf.get_variable(
            "attention_source",
            shape=[output_dim, 1],
            initializer=tf.glorot_uniform_initializer()
        )
        attention_target = tf.get_variable(
            "attention_target",
            shape=[output_dim, 1],
            initializer=tf.glorot_uniform_initializer()
        )
        
        # Compute attention components efficiently
        # Source component: [batch_size, num_nodes, 1]
        source_scores = tf.matmul(transformed_features, attention_source)
        # Target component: [batch_size, num_nodes, 1]
        target_scores = tf.matmul(transformed_features, attention_target)
        
        # For sparse graphs, use adjacency to avoid computing unnecessary attention scores
        # Get sparsity ratio to decide computation strategy
        sparsity = tf.reduce_mean(adjacency_matrix)
        
        # Use conditional computation based on sparsity
        def sparse_attention():
            """Compute attention only for existing edges."""
            # Get edge indices where adjacency > 0 (including weighted edges)
            edge_indices = tf.where(tf.not_equal(adjacency_matrix, 0.0))
            
            # Extract source and target features for edges
            batch_idx = edge_indices[:, 0]
            src_idx = edge_indices[:, 1]
            tgt_idx = edge_indices[:, 2]
            
            # Gather scores for edges (ensure indices are valid)
            valid_batch_mask = tf.less(batch_idx, batch_size)
            valid_indices = tf.boolean_mask(edge_indices, valid_batch_mask)
            valid_batch_idx = valid_indices[:, 0]
            valid_src_idx = valid_indices[:, 1]
            valid_tgt_idx = valid_indices[:, 2]
            
            src_scores = tf.gather_nd(source_scores, tf.stack([valid_batch_idx, valid_src_idx], axis=1))
            tgt_scores = tf.gather_nd(target_scores, tf.stack([valid_batch_idx, valid_tgt_idx], axis=1))
            
            # Compute attention for edges only
            edge_attention = tf.nn.leaky_relu(src_scores + tgt_scores, alpha=0.2)
            
            # Scatter back to full attention matrix with consistent batch size
            attention_shape = tf.stack([batch_size, num_nodes, num_nodes])
            edge_attention_flat = tf.reshape(edge_attention, [-1])
            sparse_logits = tf.scatter_nd(valid_indices, edge_attention_flat, attention_shape)
            
            # Apply mask for non-edges with consistent shapes
            mask = tf.equal(adjacency_matrix, 0.0)
            masked_logits = tf.where(
                mask, 
                tf.ones_like(sparse_logits) * (-1e9),
                sparse_logits
            )
            return masked_logits
        
        def dense_attention():
            """Standard dense attention computation."""
            # Use broadcasting to compute all pairwise attention scores efficiently
            # source_scores: [batch_size, num_nodes, 1]
            # target_scores: [batch_size, num_nodes, 1]
            
            # Expand for broadcasting: [batch_size, num_nodes, 1, 1]
            source_expanded = tf.expand_dims(source_scores, axis=2)  
            # Expand for broadcasting: [batch_size, 1, num_nodes, 1]
            target_expanded = tf.expand_dims(target_scores, axis=1)
            
            # Broadcast to [batch_size, num_nodes, num_nodes, 1]
            attention_logits = source_expanded + target_expanded
            # Remove last dimension: [batch_size, num_nodes, num_nodes]
            attention_logits = tf.squeeze(attention_logits, axis=-1)
            attention_logits = tf.nn.leaky_relu(attention_logits, alpha=0.2)
            
            # Apply adjacency mask - ensure consistent shapes
            # adjacency_matrix: [batch_size, num_nodes, num_nodes]
            mask = tf.equal(adjacency_matrix, 0.0)
            # Create masked attention with same shape as attention_logits
            masked_logits = tf.where(
                mask, 
                tf.ones_like(attention_logits) * (-1e9),
                attention_logits
            )
            return masked_logits
        
        # Choose computation path based on sparsity (if < 30% edges, use sparse)
        # Both functions already apply adjacency masking internally
        masked_attention_logits = tf.cond(
            sparsity < 0.3,
            sparse_attention,
            dense_attention
        )
        
        # Apply softmax to get attention weights: [batch_size, num_nodes, num_nodes]
        attention_weights = tf.nn.softmax(masked_attention_logits, axis=-1)
        
        # Incorporate edge weights from adjacency matrix after softmax
        # This scales attention by the edge weight (e.g., latency, cost)
        # Only apply weights where edges exist (non-zero adjacency)
        edge_mask = tf.equal(adjacency_matrix, 0.0)
        edge_weight_mask = tf.where(
            edge_mask,
            tf.zeros_like(adjacency_matrix),
            adjacency_matrix
        )
        attention_weights = attention_weights * edge_weight_mask
        
        # Renormalize attention weights after applying edge weights
        # This ensures attention weights sum to 1 for each node
        attention_sum = tf.reduce_sum(attention_weights, axis=-1, keepdims=True)
        # Broadcast attention_sum to match attention_weights shape
        # attention_sum: [batch_size, num_nodes, 1]
        # attention_weights: [batch_size, num_nodes, num_nodes]
        attention_sum_broadcast = tf.tile(attention_sum, [1, 1, num_nodes])
        # Create condition with correct shape
        condition = tf.greater(attention_sum_broadcast, 0)
        attention_weights = tf.where(
            condition,
            attention_weights / (attention_sum + 1e-10),  # attention_sum will broadcast correctly here
            attention_weights
        )
        
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
    
    def _edge_list_to_adjacency(self, edge_list, batch_size, num_nodes, edge_weights=None):
        """
        Convert edge list to weighted adjacency matrix.
        
        Args:
            edge_list: [batch_size, num_edges, 2] tensor where last dim is [source, target]
            batch_size: Batch size
            num_nodes: Number of nodes
            edge_weights: Optional [batch_size, num_edges] tensor of edge weights.
                         If None, uses 1.0 for all edges.
            
        Returns:
            adjacency_matrix: [batch_size, num_nodes, num_nodes] weighted tensor
        """
        # Get batch indices
        num_edges = tf.shape(edge_list)[1]
        batch_indices = tf.reshape(
            tf.tile(tf.expand_dims(tf.range(batch_size), 1), [1, num_edges]),
            [-1]
        )
        
        # Flatten edge list
        edges_flat = tf.reshape(edge_list, [-1, 2])
        
        # Create indices for scatter update
        indices = tf.stack([
            batch_indices,
            edges_flat[:, 0],  # source nodes
            edges_flat[:, 1]   # target nodes
        ], axis=1)
        
        # Handle edge weights
        if edge_weights is not None:
            # Flatten edge weights to match indices
            updates = tf.reshape(edge_weights, [-1])
        else:
            # Default to weight of 1.0 for all edges
            updates = tf.ones(tf.shape(indices)[0], dtype=tf.float32)
        
        # Create weighted adjacency matrix using scatter
        adjacency_matrix = tf.scatter_nd(indices, updates, 
                                        [batch_size, num_nodes, num_nodes])
        
        # Add self-loops with weight 1.0 (each node connects to itself)
        eye = tf.eye(num_nodes, batch_shape=[batch_size], dtype=tf.float32)
        # Use maximum to preserve existing edge weights while adding self-loops
        # This ensures self-loops have weight 1.0 and doesn't overwrite existing edges
        adjacency_matrix = tf.where(
            tf.equal(adjacency_matrix, 0.0),
            eye,
            adjacency_matrix
        )
        
        return adjacency_matrix
    
    def get_output_dim(self):
        """
        Return the final output feature dimension of the encoder.
        
        Computes the correct dimension based on the final layer behavior:
        - If num_layers == 1: uses self.concat directly
        - If num_layers > 1 and concat=True: last layer averages (hidden_dim)
        - If concat=False: always averages (hidden_dim)
        
        Returns:
            int: Output dimension size
        """
        # Check final layer behavior based on line 111 logic:
        # concat=self.concat and (layer_idx < self.num_layers - 1)
        if self.num_layers == 1:
            # Single layer: uses self.concat directly
            if self.concat:
                return self.hidden_dim * self.num_heads
            else:
                return self.hidden_dim
        else:
            # Multiple layers: last layer doesn't concat when concat=True
            # because (layer_idx < self.num_layers - 1) is False for last layer
            return self.hidden_dim