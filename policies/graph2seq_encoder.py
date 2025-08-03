"""
Graph2Seq Encoder refactored as Keras Layer for TF2
Maintains exact interface and tensor shapes from original implementation
"""
import tensorflow as tf
import numpy as np

# Import Graph2Seq modules from local copies
from .graph2seq_modules.neigh_samplers import UniformNeighborSampler
from .graph2seq_modules.aggregators import MeanAggregator, MaxPoolingAggregator, GatedMeanAggregator
from .graph2seq_modules.inits import glorot, zeros
from .graph2seq_modules.layers import Layer


class Graph2SeqEncoder(tf.keras.layers.Layer):
    """
    Graph2Seq Encoder as a Keras Layer
    Converts sequence inputs to graph representation and encodes using GCN layers
    """
    
    def __init__(self, hidden_dim, num_layers=2, bidirectional=False, 
                 concat=True, dropout=0.1, name="graph2seq_encoder", **kwargs):
        super(Graph2SeqEncoder, self).__init__(name=name, **kwargs)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.concat = concat
        self.dropout = dropout
        
        # Graph2Seq parameters
        self.sample_layer_size = 2  # Number of GCN layers
        
        # Aggregators will be created in build()
        self.fw_aggregators = []
        self.bw_aggregators = []
        
        # State projection layer
        self.state_projection = None
        self._built_aggregators = False
        
    def build(self, input_shape):
        """Build the layer - create aggregators and projection layer"""
        batch_size, seq_len, input_dim = input_shape
        self.input_dim = input_dim
        
        # Create aggregators for each GCN layer
        for layer in range(self.sample_layer_size):
            if layer == 0:
                dim_mul = 1
                input_hidden_dim = self.input_dim
            else:
                dim_mul = 2
                input_hidden_dim = dim_mul * self.hidden_dim
                
            # Forward aggregator
            fw_aggregator = MeanAggregator(
                input_hidden_dim,  # neigh_input_dim
                self.hidden_dim,   # output_dim
                concat=self.concat,
                mode='train'  # Will be controlled by training parameter in call()
            )
            self.fw_aggregators.append(fw_aggregator)
            
            # Backward aggregator if bidirectional
            if self.bidirectional:
                bw_aggregator = MeanAggregator(
                    input_hidden_dim,
                    self.hidden_dim,
                    concat=self.concat,
                    mode='train'
                )
                self.bw_aggregators.append(bw_aggregator)
        
        # State projection layer
        if self.bidirectional:
            state_size = 4 * self.hidden_dim  # concat fw and bw, each is 2*hidden_dim
        else:
            state_size = 2 * self.hidden_dim  # concat mode doubles dim
            
        if state_size > self.hidden_dim:
            self.state_projection = tf.keras.layers.Dense(
                self.hidden_dim,
                activation=None,
                name="state_projection"
            )
            # Build the projection layer immediately with known input shape
            self.state_projection.build((None, state_size))
        
        super(Graph2SeqEncoder, self).build(input_shape)
    
    def sequence_to_graph(self, sequence_inputs):
        """
        Convert sequence inputs to graph representation.
        Input: [batch_size, seq_len, feature_dim]
        Output: graph adjacency and feature tensors
        """
        batch_size = tf.shape(sequence_inputs)[0]
        seq_len = tf.shape(sequence_inputs)[1]
        
        # Total number of nodes across all batches
        total_nodes = batch_size * seq_len
        
        # Create adjacency info for fully connected graph within each sequence
        # Create base indices for one sequence
        seq_indices = tf.range(seq_len)
        
        # Tile to create adjacency for all nodes
        single_seq_adj = tf.tile(tf.expand_dims(seq_indices, 0), [seq_len, 1])
        
        # Create batch offsets
        batch_offsets = tf.reshape(tf.range(batch_size) * seq_len, [batch_size, 1, 1])
        
        # Expand single sequence adjacency to all batches
        batch_adj = tf.expand_dims(single_seq_adj, 0) + batch_offsets
        
        # Reshape to [total_nodes, seq_len]
        fw_adj_info = tf.reshape(batch_adj, [total_nodes, seq_len])
        bw_adj_info = fw_adj_info  # Same for backward
        
        # Flatten sequence features for graph processing
        feature_info = tf.reshape(sequence_inputs, [total_nodes, -1])
        
        # Create batch nodes tensor
        batch_nodes = tf.reshape(tf.range(total_nodes), [batch_size, seq_len])
        
        return fw_adj_info, bw_adj_info, feature_info, batch_nodes
        
    def call(self, inputs, mask=None, training=None, adj=None, **kwargs):
        """
        Main encoding function
        Args:
            inputs: encoder_inputs [batch_size, seq_len, input_dim]
            mask: optional mask (not used currently)
            training: boolean for training mode
            adj: optional adjacency (not used - we create fully connected)
        Returns:
            encoder_outputs: [batch_size, seq_len, 2*hidden_dim or 4*hidden_dim]
            encoder_state: [batch_size, hidden_dim] projected final state
        """
        # Handle dropout based on training mode
        if training is None:
            training = tf.keras.backend.learning_phase()
            
        # Convert sequence to graph representation
        fw_adj_info, bw_adj_info, feature_info, batch_nodes = self.sequence_to_graph(inputs)
        
        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]
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
        
        # Create neighbor length tensors
        fw_sampled_neighbors_len = tf.fill([batch_size * seq_len], seq_len)
        if self.bidirectional:
            bw_sampled_neighbors_len = tf.fill([batch_size * seq_len], seq_len)
        
        # Graph convolution layers
        for layer in range(self.sample_layer_size):
            if layer == 0:
                dim_mul = 1
                input_hidden_dim = self.input_dim
            else:
                dim_mul = 2
                input_hidden_dim = dim_mul * self.hidden_dim
                
            # Get aggregator for this layer
            fw_aggregator = self.fw_aggregators[layer]
            
            # Set training mode for aggregator
            # TODO(runtime): Verify aggregator training mode handling
            
            # Get neighbor embeddings
            if layer == 0:
                neigh_vec_hidden = tf.nn.embedding_lookup(embedded_node_rep, fw_sampled_neighbors)
            else:
                # Pad hidden states for lookup
                padded_hidden = tf.concat([fw_hidden, tf.zeros([1, dim_mul * self.hidden_dim])], 0)
                neigh_vec_hidden = tf.nn.embedding_lookup(padded_hidden, fw_sampled_neighbors)
            
            # Apply dropout if training
            if training and self.dropout > 0:
                fw_hidden = tf.nn.dropout(fw_hidden, rate=self.dropout)
                neigh_vec_hidden = tf.nn.dropout(neigh_vec_hidden, rate=self.dropout)
            
            # Aggregate - let Keras handle automatic building
            fw_hidden = fw_aggregator((fw_hidden, neigh_vec_hidden, fw_sampled_neighbors_len), training=training)
            
            if self.bidirectional:
                bw_aggregator = self.bw_aggregators[layer]
                
                if layer == 0:
                    neigh_vec_hidden = tf.nn.embedding_lookup(embedded_node_rep, bw_sampled_neighbors)
                else:
                    padded_hidden = tf.concat([bw_hidden, tf.zeros([1, dim_mul * self.hidden_dim])], 0)
                    neigh_vec_hidden = tf.nn.embedding_lookup(padded_hidden, bw_sampled_neighbors)
                
                if training and self.dropout > 0:
                    bw_hidden = tf.nn.dropout(bw_hidden, rate=self.dropout)
                    neigh_vec_hidden = tf.nn.dropout(neigh_vec_hidden, rate=self.dropout)
                
                # Aggregate - let Keras handle automatic building
                bw_hidden = bw_aggregator((bw_hidden, neigh_vec_hidden, bw_sampled_neighbors_len), training=training)
        
        # Reshape hidden states back to sequence format
        fw_hidden = tf.reshape(fw_hidden, [batch_size, seq_len, 2 * self.hidden_dim])
        
        if self.bidirectional:
            bw_hidden = tf.reshape(bw_hidden, [batch_size, seq_len, 2 * self.hidden_dim])
            encoder_outputs = tf.concat([fw_hidden, bw_hidden], axis=2)
        else:
            encoder_outputs = fw_hidden
        
        encoder_outputs = tf.nn.relu(encoder_outputs)
        
        # Create encoder state - use max pooling over sequence
        final_state = tf.reduce_max(encoder_outputs, axis=1)
        
        # Project state if needed
        if self.state_projection is not None:
            final_state = self.state_projection(final_state)
        else:
            # If no projection needed, extract the first hidden_dim dimensions
            final_state = final_state[..., :self.hidden_dim]
        
        return encoder_outputs, final_state


# Global cache for encoder instances to avoid creating duplicates
_encoder_cache = {}

def create_graph2seq_encoder(encoder_inputs, encoder_units, num_layers, is_bidirectional, mode, scope_name="encoder"):
    """
    Factory function to create Graph2Seq encoder matching the original interface.
    MIGRATION: Now creates a Keras layer instead of using variable_scope with caching
    """
    # Get input dimensions for cache key
    input_shape = encoder_inputs.shape
    input_dim = input_shape[-1] if len(input_shape) >= 2 else None
    
    # Create a cache key based on encoder configuration and input dimensions
    cache_key = (encoder_units, num_layers, is_bidirectional, scope_name, input_dim)
    
    # Check if encoder already exists in cache
    if cache_key not in _encoder_cache:
        # Create the encoder layer
        encoder = Graph2SeqEncoder(
            hidden_dim=encoder_units,
            num_layers=num_layers,
            bidirectional=is_bidirectional,
            dropout=0.1 if mode == 'train' else 0.0,
            name=scope_name
        )
        _encoder_cache[cache_key] = encoder
    else:
        encoder = _encoder_cache[cache_key]
    
    # Call the encoder
    training = (mode == 'train')
    encoder_outputs, encoder_state = encoder(encoder_inputs, training=training)
    
    # Convert final state to LSTM-compatible format
    # Create LSTM state tuple for decoder compatibility
    if num_layers == 1:
        # Single layer - create LSTMStateTuple
        # Using compat layer for compatibility
        from compat import rnn as compat_rnn
        encoder_state_tuple = (encoder_state, encoder_state)  # (c, h)
    else:
        # Multi-layer - create tuple of states
        encoder_state_tuple = tuple([
            (encoder_state, encoder_state)  # (c, h) for each layer
            for _ in range(num_layers)
        ])
    
    return encoder_outputs, encoder_state_tuple