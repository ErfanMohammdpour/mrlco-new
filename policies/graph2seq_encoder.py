"""
This implementation is imported from IBM/Graph2Seq. Interfaces and tensor shapes are preserved to match the original Metarl-Offloading encoder.
"""
import tensorflow as tf
import numpy as np

# Import Graph2Seq modules from local copies
from .graph2seq_modules.neigh_samplers import UniformNeighborSampler
from .graph2seq_modules.aggregators import MeanAggregator, MaxPoolingAggregator, GatedMeanAggregator
from .graph2seq_modules.inits import glorot, zeros
from .graph2seq_modules.layers import Layer


class Graph2SeqEncoderAdapter:
    """
    Adapter class that wraps Graph2Seq encoder to be compatible with metarl-offloading.
    Converts sequence inputs to graph representation and maintains interface compatibility.
    """
    
    def __init__(self, input_dim, hidden_dim, num_layers=2, bidirectional=False, mode='train', decoder_hidden_unit=None):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.mode = mode
        self.decoder_hidden_unit = decoder_hidden_unit if decoder_hidden_unit is not None else hidden_dim
        
        # Graph2Seq parameters
        self.sample_layer_size = 2  # Number of GCN layers
        self.concat = True
        self.dropout = 0.0 if mode != 'train' else 0.1
        
        # Pre-create aggregators for all layers to avoid creating variables in tf.function
        self.fw_aggregators = []
        self.bw_aggregators = []
        
        for layer in range(self.num_layers):
            if layer == 0:
                # First layer takes input_dim as input
                input_size = self.input_dim
            else:
                # Subsequent layers take concatenated output from previous layer
                input_size = 2 * self.hidden_dim if self.concat else self.hidden_dim
            
            # Create forward aggregator
            fw_aggregator = MeanAggregator(
                input_size,
                self.hidden_dim,
                concat=self.concat,
                mode=self.mode
            )
            self.fw_aggregators.append(fw_aggregator)
            
            # Create backward aggregator if bidirectional
            if self.bidirectional:
                bw_aggregator = MeanAggregator(
                    input_size,
                    self.hidden_dim,
                    concat=self.concat,
                    mode=self.mode
                )
                self.bw_aggregators.append(bw_aggregator)
        
        # Pre-create state projection layer to avoid creating variables in tf.function
        self.state_projection_layer = tf.keras.layers.Dense(
            self.decoder_hidden_unit, 
            activation=None, 
            name="state_dense"
        )
        
    def sequence_to_graph(self, sequence_inputs):
        """
        Convert sequence inputs to graph representation.
        Input: [batch_size, seq_len, feature_dim]
        Output: graph adjacency and feature tensors
        """
        batch_size = tf.shape(sequence_inputs)[0]
        seq_len = tf.shape(sequence_inputs)[1]
        feature_dim = tf.shape(sequence_inputs)[2]
        
        # Total number of nodes across all batches
        total_nodes = batch_size * seq_len
        
        # Create adjacency info for fully connected graph within each sequence
        # Each node connects to all nodes in its sequence
        
        # Method: Create an adjacency matrix where each node points to all nodes in its sequence
        # For node i in batch b, it connects to nodes [b*seq_len, (b+1)*seq_len)
        
        # Create base indices for one sequence
        seq_indices = tf.range(seq_len)
        
        # Tile to create adjacency for all nodes
        # Shape: [seq_len, seq_len] - each row is the adjacency list for that position
        single_seq_adj = tf.tile(tf.expand_dims(seq_indices, 0), [seq_len, 1])
        
        # Create batch offsets
        # Shape: [batch_size, 1, 1]
        batch_offsets = tf.reshape(tf.range(batch_size) * seq_len, [batch_size, 1, 1])
        
        # Expand single sequence adjacency to all batches
        # Shape: [batch_size, seq_len, seq_len]
        batch_adj = tf.expand_dims(single_seq_adj, 0) + batch_offsets
        
        # Reshape to [total_nodes, seq_len]
        fw_adj_info = tf.reshape(batch_adj, [total_nodes, seq_len])
        bw_adj_info = fw_adj_info  # Same for backward
        
        # Flatten sequence features for graph processing
        # Shape: [total_nodes, feature_dim]
        feature_info = tf.reshape(sequence_inputs, [total_nodes, feature_dim])
        
        # Create batch nodes tensor - identifies which nodes belong to which batch
        # Shape: [batch_size, seq_len]
        batch_nodes = tf.reshape(tf.range(total_nodes), [batch_size, seq_len])
        
        return fw_adj_info, bw_adj_info, feature_info, batch_nodes
        
    def encode(self, encoder_inputs):
        """
        Main encoding function that maintains compatibility with metarl-offloading.
        Input: encoder_inputs [batch_size, seq_len, input_dim]
        Output: (encoder_outputs, encoder_state) matching original interface
        """
        # Convert sequence to graph representation
        fw_adj_info, bw_adj_info, feature_info, batch_nodes = self.sequence_to_graph(encoder_inputs)
        
        batch_size = tf.shape(encoder_inputs)[0]
        seq_len = tf.shape(encoder_inputs)[1]
        single_graph_nodes_size = seq_len
        sample_size_per_layer = seq_len  # Sample all nodes
        
        # Initialize node embeddings
        embedded_node_rep = feature_info
        
        # Create samplers
        fw_sampler = UniformNeighborSampler(fw_adj_info)
        if self.bidirectional:
            bw_sampler = UniformNeighborSampler(bw_adj_info)
        
        nodes = tf.reshape(batch_nodes, [-1])
        
        # Initial hidden states
        fw_hidden = tf.nn.embedding_lookup(embedded_node_rep, nodes)
        if self.bidirectional:
            bw_hidden = tf.nn.embedding_lookup(embedded_node_rep, nodes)
        
        # Sample neighbors
        fw_sampled_neighbors = fw_sampler((nodes, sample_size_per_layer))
        if self.bidirectional:
            bw_sampled_neighbors = bw_sampler((nodes, sample_size_per_layer))
        
        # Create neighbor length tensors - all nodes have access to full sequence
        fw_sampled_neighbors_len = tf.fill([batch_size * seq_len], seq_len)
        if self.bidirectional:
            bw_sampled_neighbors_len = tf.fill([batch_size * seq_len], seq_len)
        
        # Graph convolution layers
        for layer in range(self.sample_layer_size):
            # Use pre-created aggregator
            fw_aggregator = self.fw_aggregators[layer]
            
            # Get neighbor embeddings
            if layer == 0:
                neigh_vec_hidden = tf.nn.embedding_lookup(embedded_node_rep, fw_sampled_neighbors)
            else:
                # For subsequent layers, use the concatenated output from previous layer
                # The output dimension is 2*hidden_dim due to concat=True
                padded_hidden = tf.concat([fw_hidden, tf.zeros([1, 2 * self.hidden_dim])], 0)
                neigh_vec_hidden = tf.nn.embedding_lookup(padded_hidden, fw_sampled_neighbors)
            
            # Aggregate
            fw_hidden = fw_aggregator((fw_hidden, neigh_vec_hidden, fw_sampled_neighbors_len))
            
            if self.bidirectional:
                bw_aggregator = self.bw_aggregators[layer]
                
                if layer == 0:
                    neigh_vec_hidden = tf.nn.embedding_lookup(embedded_node_rep, bw_sampled_neighbors)
                else:
                    padded_hidden = tf.concat([bw_hidden, tf.zeros([1, 2 * self.hidden_dim])], 0)
                    neigh_vec_hidden = tf.nn.embedding_lookup(padded_hidden, bw_sampled_neighbors)
                
                bw_hidden = bw_aggregator((bw_hidden, neigh_vec_hidden, bw_sampled_neighbors_len))
        
        # Reshape hidden states
        fw_hidden = tf.reshape(fw_hidden, [batch_size, seq_len, 2 * self.hidden_dim])
        
        if self.bidirectional:
            bw_hidden = tf.reshape(bw_hidden, [batch_size, seq_len, 2 * self.hidden_dim])
            encoder_outputs = tf.concat([fw_hidden, bw_hidden], axis=2)
        else:
            encoder_outputs = fw_hidden
        
        encoder_outputs = tf.nn.relu(encoder_outputs)
        
        # Create encoder state compatible with LSTM decoder
        # Use max pooling over sequence to get final state
        final_state = tf.reduce_max(encoder_outputs, axis=1)
        
        # Create LSTM-compatible state tuple
        if self.bidirectional:
            state_size = 4 * self.hidden_dim
        else:
            state_size = 2 * self.hidden_dim
            
        # For multi-layer compatibility, create tuple of states
        # The decoder expects states with dimension matching decoder_units
        # Project the state to match decoder expectations using pre-created layer
        final_state_proj = self.state_projection_layer(final_state)
            
        if self.num_layers == 1:
            # TF2: Create LSTM state tuple manually
            encoder_state = (final_state_proj, final_state_proj)  # (c, h) for LSTM
        else:
            encoder_state = tuple([
                (final_state_proj, final_state_proj)  # (c, h) for each layer
                for _ in range(self.num_layers)
            ])
        
        return encoder_outputs, encoder_state


def create_graph2seq_encoder(encoder_inputs, encoder_units, num_layers, is_bidirectional, mode, scope_name="encoder"):
    """
    Factory function to create Graph2Seq encoder matching the original interface.
    """
    # TF2: Use name_scope instead of variable_scope
    with tf.name_scope(scope_name):
        # TF2: Use shape property instead of get_shape().value
        input_dim = encoder_inputs.shape[-1]
        encoder_adapter = Graph2SeqEncoderAdapter(
            input_dim=input_dim,
            hidden_dim=encoder_units,
            num_layers=num_layers,
            bidirectional=is_bidirectional,
            mode=mode
        )
        
        encoder_outputs, encoder_state = encoder_adapter.encode(encoder_inputs)
        
    return encoder_outputs, encoder_state