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
    
    def __init__(self, input_dim, hidden_dim, num_layers=2, bidirectional=False, mode='train'):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.mode = mode
        
        # Graph2Seq parameters
        self.sample_layer_size = 2  # Number of GCN layers
        self.concat = True
        self.dropout = 0.0 if mode != 'train' else 0.1
        
        # Initialize aggregators
        self.fw_aggregators = []
        self.bw_aggregators = []
        
    def sequence_to_graph(self, sequence_inputs):
        """
        Convert sequence inputs to graph representation.
        Input: [batch_size, seq_len, feature_dim]
        Output: graph adjacency and feature tensors
        """
        batch_size = tf.shape(sequence_inputs)[0]
        seq_len = tf.shape(sequence_inputs)[1]
        feature_dim = tf.shape(sequence_inputs)[2]
        
        # Create fully connected graph adjacency (each node connected to all others)
        # For sequences, we can use temporal adjacency (each node connected to previous/next)
        # Forward adjacency: each node connected to next nodes
        fw_adj = []
        bw_adj = []
        
        # Create adjacency lists for sequential connections
        # Each position connects to its neighbors within a window
        window_size = min(5, seq_len)  # Connect to up to 5 neighbors
        
        # Placeholder for actual graph construction - for now using fully connected
        # In practice, you might want to use attention scores or learned adjacency
        all_indices = tf.range(seq_len)
        fw_adj_info = tf.tile(tf.expand_dims(all_indices, 0), [batch_size * seq_len, 1])
        bw_adj_info = tf.tile(tf.expand_dims(all_indices, 0), [batch_size * seq_len, 1])
        
        # Flatten sequence for graph processing
        feature_info = tf.reshape(sequence_inputs, [batch_size * seq_len, feature_dim])
        
        # Create batch nodes tensor
        batch_nodes = tf.reshape(tf.range(batch_size * seq_len), [batch_size, seq_len])
        
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
        
        fw_sampled_neighbors_len = tf.constant(seq_len, shape=[batch_size * seq_len])
        if self.bidirectional:
            bw_sampled_neighbors_len = tf.constant(seq_len, shape=[batch_size * seq_len])
        
        # Graph convolution layers
        for layer in range(self.sample_layer_size):
            if layer == 0:
                dim_mul = 1
            else:
                dim_mul = 2
                
            # Create aggregator
            fw_aggregator = MeanAggregator(
                dim_mul * self.hidden_dim, 
                self.hidden_dim, 
                concat=self.concat, 
                mode=self.mode
            )
            self.fw_aggregators.append(fw_aggregator)
            
            # Get neighbor embeddings
            if layer == 0:
                neigh_vec_hidden = tf.nn.embedding_lookup(embedded_node_rep, fw_sampled_neighbors)
            else:
                # Pad hidden states for lookup
                padded_hidden = tf.concat([fw_hidden, tf.zeros([1, dim_mul * self.hidden_dim])], 0)
                neigh_vec_hidden = tf.nn.embedding_lookup(padded_hidden, fw_sampled_neighbors)
            
            # Aggregate
            fw_hidden = fw_aggregator((fw_hidden, neigh_vec_hidden, fw_sampled_neighbors_len))
            
            if self.bidirectional:
                bw_aggregator = MeanAggregator(
                    dim_mul * self.hidden_dim, 
                    self.hidden_dim, 
                    concat=self.concat, 
                    mode=self.mode
                )
                self.bw_aggregators.append(bw_aggregator)
                
                if layer == 0:
                    neigh_vec_hidden = tf.nn.embedding_lookup(embedded_node_rep, bw_sampled_neighbors)
                else:
                    padded_hidden = tf.concat([bw_hidden, tf.zeros([1, dim_mul * self.hidden_dim])], 0)
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
        if self.num_layers == 1:
            encoder_state = tf.nn.rnn_cell.LSTMStateTuple(c=final_state, h=final_state)
        else:
            encoder_state = tuple([
                tf.nn.rnn_cell.LSTMStateTuple(c=final_state, h=final_state)
                for _ in range(self.num_layers)
            ])
        
        return encoder_outputs, encoder_state


def create_graph2seq_encoder(encoder_inputs, encoder_units, num_layers, is_bidirectional, mode, scope_name="encoder"):
    """
    Factory function to create Graph2Seq encoder matching the original interface.
    """
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
        input_dim = encoder_inputs.get_shape()[-1].value
        encoder_adapter = Graph2SeqEncoderAdapter(
            input_dim=input_dim,
            hidden_dim=encoder_units,
            num_layers=num_layers,
            bidirectional=is_bidirectional,
            mode=mode
        )
        
        encoder_outputs, encoder_state = encoder_adapter.encode(encoder_inputs)
        
    return encoder_outputs, encoder_state