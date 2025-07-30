"""
Feature transformation module for converting 17-dim features to 72-dim features.
This module handles the transformation from old node features to new enhanced features.
"""
import tensorflow as tf
import numpy as np

# Global constant for new node feature dimensions
IN_NODE_DIM = 72

class FeatureTransformer:
    """
    Transforms node features from the original format to the new 72-dimensional format.
    
    Original format: [task_index, local_process_cost, up_link_cost, mec_process_cost, down_link_cost, 
                     pre_task_indices..., succ_task_indices...]
    New format: cost_embed (64) + id_embed (8) = 72 dimensions
    """
    
    def __init__(self, max_task_id, training=True, name="feature_transformer"):
        self.max_task_id = max_task_id
        self.training = training
        self.name = name
        
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            # Shared MLP for cost embedding: Dense(4->32) -> ReLU -> Dense(32->64)
            self.cost_dense1 = tf.layers.Dense(32, activation=None, 
                                             kernel_initializer=tf.variance_scaling_initializer(mode='fan_in'),
                                             name="cost_dense1")
            self.cost_dense2 = tf.layers.Dense(64, activation=None,
                                             kernel_initializer=tf.variance_scaling_initializer(mode='fan_in'), 
                                             name="cost_dense2")
            self.cost_layer_norm = tf.layers.BatchNormalization(name="cost_layer_norm")
            
            # Task index embedding: Embedding(num_ids=max_id+1, dim=8)
            self.task_embedding = tf.Variable(
                tf.random_uniform([self.max_task_id + 1, 8], -0.1, 0.1),
                name="task_embedding_matrix",
                dtype=tf.float32
            )
            
            # Dropout for training
            self.dropout_rate = 0.1
    
    def transform(self, old_features):
        """
        Transform old 17-dim features to new 72-dim features.
        
        Args:
            old_features: Tensor of shape [batch_size, seq_len, 5]
                         Format: [task_index, local_process_cost, up_link_cost, 
                                 mec_process_cost, down_link_cost]
        
        Returns:
            new_features: Tensor of shape [batch_size, seq_len, 72]
        """
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            batch_size = tf.shape(old_features)[0]
            seq_len = tf.shape(old_features)[1]
            
            # Extract components from old features
            task_indices = tf.cast(old_features[:, :, 0], tf.int32)  # [batch_size, seq_len]
            cost_vector = old_features[:, :, 1:5]  # [batch_size, seq_len, 4]
            
            # Handle invalid task indices (-1 used for padding)
            # Clip task indices to valid range [0, max_task_id]
            task_indices = tf.maximum(task_indices, 0)
            task_indices = tf.minimum(task_indices, self.max_task_id)
            
            # 1. Cost vector (4 scalars) -> shared MLP -> cost_embed (64 dims)
            # Dense(4->32) -> ReLU -> LayerNorm -> Dense(32->64)
            cost_hidden = self.cost_dense1(cost_vector)  # [batch_size, seq_len, 32]
            cost_hidden = tf.nn.relu(cost_hidden)
            cost_hidden = self.cost_layer_norm(cost_hidden, training=self.training)
            cost_embed = self.cost_dense2(cost_hidden)  # [batch_size, seq_len, 64]
            
            # 2. Task index -> embedding -> id_embed (8 dims)
            id_embed = tf.nn.embedding_lookup(self.task_embedding, task_indices)  # [batch_size, seq_len, 8]
            
            # 3. Concatenate: cost_embed + id_embed = 72 dims
            node_feat_new = tf.concat([cost_embed, id_embed], axis=-1)  # [batch_size, seq_len, 72]
            
            # 4. Apply Dropout during training
            if self.training:
                node_feat_new = tf.layers.dropout(node_feat_new, rate=self.dropout_rate, training=self.training)
            
            return node_feat_new
    
    def add_shape_check(self, tensor, expected_shape_suffix, name):
        """Add shape consistency check"""
        with tf.name_scope("shape_check_" + name):
            actual_shape = tf.shape(tensor)
            expected_last_dim = expected_shape_suffix[-1]
            
            check_op = tf.assert_equal(
                actual_shape[-1], expected_last_dim,
                message="Shape mismatch in " + name
            )
            
            with tf.control_dependencies([check_op]):
                return tf.identity(tensor, name="checked_" + name)


def create_feature_transformer(max_task_id, training=True):
    """Factory function to create feature transformer"""
    return FeatureTransformer(max_task_id=max_task_id, training=training)


def add_shape_consistency_check(tensor, expected_shape, checkpoint_name):
    """
    Add shape consistency check at various points in the pipeline.
    
    Args:
        tensor: The tensor to check
        expected_shape: Expected shape (can use None for dynamic dimensions)
        checkpoint_name: Name for this checkpoint
    """
    with tf.name_scope("shape_check_" + checkpoint_name):
        actual_shape = tf.shape(tensor)
        
        # Check the last dimension (feature dimension)
        if expected_shape[-1] is not None:
            check_op = tf.assert_equal(
                actual_shape[-1], expected_shape[-1],
                message="Shape check failed at " + checkpoint_name
            )
            with tf.control_dependencies([check_op]):
                return tf.identity(tensor, name="shape_checked_" + checkpoint_name)
        
        return tensor