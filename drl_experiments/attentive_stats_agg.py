"""
Optional AttentiveStatisticsAggregator for DRL experiments.
Provides an alternative to MeanAggregator with attention-based pooling.
"""

import tensorflow as tf
import numpy as np


class AttentiveStatisticsAggregator:
    """
    Attention-based statistics aggregator for graph neural networks.
    Computes mean and standard deviation with attention weights.
    """
    
    def __init__(self, input_dim, output_dim, concat=True, mode='train'):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.concat = concat
        self.mode = mode
        
        # Attention parameters
        self.attention_dim = min(input_dim, 64)  # Small attention dimension for stability
        
        # Initialize attention weights
        self.attention_W = tf.get_variable(
            'attention_W', 
            shape=[input_dim, self.attention_dim],
            initializer=tf.contrib.layers.xavier_initializer()
        )
        self.attention_b = tf.get_variable(
            'attention_b',
            shape=[self.attention_dim],
            initializer=tf.zeros_initializer()
        )
        self.attention_v = tf.get_variable(
            'attention_v',
            shape=[self.attention_dim, 1],
            initializer=tf.contrib.layers.xavier_initializer()
        )
        
        # Output projection
        if concat:
            self.output_W = tf.get_variable(
                'output_W',
                shape=[input_dim * 2, output_dim],  # Mean + Std concatenated
                initializer=tf.contrib.layers.xavier_initializer()
            )
        else:
            self.output_W = tf.get_variable(
                'output_W',
                shape=[input_dim, output_dim],
                initializer=tf.contrib.layers.xavier_initializer()
            )
        
        self.output_b = tf.get_variable(
            'output_b',
            shape=[output_dim],
            initializer=tf.zeros_initializer()
        )
        
        # Layer normalization
        self.layer_norm = tf.contrib.layers.layer_norm
    
    def __call__(self, inputs):
        """
        Apply attentive statistics aggregation.
        
        Args:
            inputs: tuple of (node_features, neighbor_features, neighbor_lengths)
                - node_features: [batch_size, input_dim]
                - neighbor_features: [batch_size, max_neighbors, input_dim]
                - neighbor_lengths: [batch_size] - actual number of neighbors per node
        
        Returns:
            aggregated_features: [batch_size, output_dim]
        """
        node_features, neighbor_features, neighbor_lengths = inputs
        
        batch_size = tf.shape(node_features)[0]
        max_neighbors = tf.shape(neighbor_features)[1]
        
        # Compute attention scores
        # Reshape neighbor features for attention computation
        neighbor_flat = tf.reshape(neighbor_features, [-1, self.input_dim])  # [batch*max_neighbors, input_dim]
        
        # Compute attention scores
        attention_scores = tf.matmul(neighbor_flat, self.attention_W) + self.attention_b  # [batch*max_neighbors, attention_dim]
        attention_scores = tf.tanh(attention_scores)
        attention_scores = tf.matmul(attention_scores, self.attention_v)  # [batch*max_neighbors, 1]
        attention_scores = tf.reshape(attention_scores, [batch_size, max_neighbors])  # [batch_size, max_neighbors]
        
        # Apply mask for actual neighbors
        neighbor_mask = tf.sequence_mask(neighbor_lengths, max_neighbors, dtype=tf.float32)  # [batch_size, max_neighbors]
        attention_scores = attention_scores * neighbor_mask
        
        # Normalize attention scores (softmax)
        attention_scores = attention_scores - tf.reduce_max(attention_scores, axis=1, keepdims=True)  # Numerical stability
        attention_weights = tf.nn.softmax(attention_scores, axis=1)  # [batch_size, max_neighbors]
        
        # Apply attention weights
        attention_weights_expanded = tf.expand_dims(attention_weights, axis=2)  # [batch_size, max_neighbors, 1]
        weighted_features = neighbor_features * attention_weights_expanded  # [batch_size, max_neighbors, input_dim]
        
        # Compute statistics
        # Mean
        attended_mean = tf.reduce_sum(weighted_features, axis=1)  # [batch_size, input_dim]
        
        # Standard deviation
        neighbor_mean_expanded = tf.expand_dims(attended_mean, axis=1)  # [batch_size, 1, input_dim]
        diff = neighbor_features - neighbor_mean_expanded  # [batch_size, max_neighbors, input_dim]
        diff_squared = tf.square(diff)  # [batch_size, max_neighbors, input_dim]
        
        # Weighted variance
        weighted_var = tf.reduce_sum(diff_squared * attention_weights_expanded, axis=1)  # [batch_size, input_dim]
        attended_std = tf.sqrt(weighted_var + 1e-8)  # Add small epsilon for numerical stability
        
        # Combine mean and std
        if self.concat:
            combined_features = tf.concat([attended_mean, attended_std], axis=1)  # [batch_size, input_dim*2]
        else:
            # Residual connection
            combined_features = attended_mean + attended_std
        
        # Apply output projection
        output = tf.matmul(combined_features, self.output_W) + self.output_b  # [batch_size, output_dim]
        
        # Apply layer normalization
        output = self.layer_norm(output)
        
        # Apply activation
        output = tf.nn.relu(output)
        
        return output


def create_attentive_aggregator(input_dim, output_dim, concat=True, mode='train'):
    """
    Factory function to create AttentiveStatisticsAggregator.
    
    Args:
        input_dim: input feature dimension
        output_dim: output feature dimension
        concat: whether to concatenate mean and std
        mode: 'train' or 'eval'
    
    Returns:
        AttentiveStatisticsAggregator instance
    """
    return AttentiveStatisticsAggregator(input_dim, output_dim, concat, mode)
