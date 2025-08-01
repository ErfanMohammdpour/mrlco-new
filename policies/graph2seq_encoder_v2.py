"""
Improved Graph2Seq encoder with:
- Edge-Gated Convolution
- Residual connections
- LayerNorm
- Virtual node support
"""
import tensorflow as tf
import numpy as np

from .graph2seq_modules.neigh_samplers import UniformNeighborSampler
from .graph2seq_modules.edge_gated_aggregator import EdgeGatedAggregator
from .graph2seq_modules.inits import glorot, zeros
from .graph2seq_modules.layers import Layer


class ImprovedGraph2SeqEncoder:
    """
    Enhanced Graph2Seq encoder with gated edge convolution, residual connections,
    and layer normalization.
    """
    
    def __init__(self, input_dim, hidden_dim, num_layers=2, bidirectional=False, 
                 mode='train', feature_mode='full17', use_virtual_node=True):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.mode = mode
        self.feature_mode = feature_mode
        self.use_virtual_node = use_virtual_node
        
        # Set number of GNN blocks based on feature mode
        self.num_blocks = 3 if feature_mode == 'core5' else 2
        
        self.concat = True
        self.dropout = 0.0 if mode != 'train' else 0.1
        
        # Initialize aggregators list
        self.fw_aggregators = []
        self.bw_aggregators = []
        
    def add_virtual_node(self, feature_info, batch_nodes, fw_adj_info, bw_adj_info):
        """
        Add a virtual node to the graph that connects to all other nodes.
        """
        batch_size = tf.shape(batch_nodes)[0]
        seq_len = tf.shape(batch_nodes)[1]
        total_nodes = batch_size * seq_len
        
        # Create virtual node features (learnable or zeros)
        with tf.variable_scope("virtual_node"):
            virtual_features = tf.get_variable(
                name="virtual_features",
                shape=[1, self.input_dim],
                initializer=tf.zeros_initializer(),
                dtype=tf.float32
            )
            # Tile for each batch
            virtual_features_batch = tf.tile(virtual_features, [batch_size, 1])
            
        # Append virtual node features
        # Original: [total_nodes, input_dim], Virtual: [batch_size, input_dim]
        # Combined: [total_nodes + batch_size, input_dim]
        feature_info_with_virtual = tf.concat([feature_info, virtual_features_batch], axis=0)
        
        # Update adjacency for Layer 1: real nodes -> virtual node
        # Each real node in batch b connects to virtual node at index (total_nodes + b)
        virtual_node_indices = tf.range(batch_size) + total_nodes
        
        # For each node in each batch, add connection to its batch's virtual node
        batch_ids = tf.reshape(tf.tile(tf.expand_dims(tf.range(batch_size), 1), [1, seq_len]), [-1])
        virtual_connections = tf.gather(virtual_node_indices, batch_ids)
        virtual_connections = tf.expand_dims(virtual_connections, 1)
        
        # Update forward adjacency for Layer 1
        fw_adj_layer1 = tf.concat([fw_adj_info, virtual_connections], axis=1)
        
        # For Layer 2: virtual node -> real nodes
        # Virtual nodes connect back to all nodes in their batch
        virtual_to_real = tf.reshape(batch_nodes, [batch_size, seq_len])
        # Pad with -1 for other batches
        max_neighbors = seq_len + 1  # +1 for self-connection
        
        # Create adjacency for virtual nodes
        virtual_adj = tf.pad(virtual_to_real, [[0, 0], [0, 1]], constant_values=-1)
        
        # Combine adjacencies
        # Real nodes keep their original adjacency for layer 2
        # Virtual nodes use virtual_adj
        fw_adj_layer2 = tf.concat([fw_adj_info, virtual_adj], axis=0)
        
        # For backward direction (if bidirectional)
        if self.bidirectional:
            bw_adj_layer1 = fw_adj_layer1
            bw_adj_layer2 = fw_adj_layer2
        else:
            bw_adj_layer1 = None
            bw_adj_layer2 = None
            
        # Update batch_nodes to include virtual nodes
        virtual_node_ids = tf.reshape(virtual_node_indices, [batch_size, 1])
        batch_nodes_with_virtual = tf.concat([batch_nodes, virtual_node_ids], axis=1)
        
        return (feature_info_with_virtual, batch_nodes_with_virtual, 
                fw_adj_layer1, fw_adj_layer2, bw_adj_layer1, bw_adj_layer2)
        
    def gnn_block(self, hidden, fw_sampled_neighbors, fw_sampled_neighbors_len,
                  bw_sampled_neighbors=None, bw_sampled_neighbors_len=None,
                  block_idx=0, layer_idx=0, input_dim=None):
        """
        A single GNN block with edge-gated convolution, residual connection, and layer norm.
        """
        if input_dim is None:
            input_dim = 2 * self.hidden_dim
            
        # Save input for residual connection
        residual = hidden
        
        # Forward aggregator
        fw_aggregator = EdgeGatedAggregator(
            input_dim=input_dim,
            output_dim=self.hidden_dim,
            concat=self.concat,
            dropout=self.dropout,
            name=f"fw_block{block_idx}_layer{layer_idx}"
        )
        self.fw_aggregators.append(fw_aggregator)
        
        # Get neighbor embeddings
        if layer_idx == 0 and block_idx == 0:
            # First layer uses original features
            neigh_vec_hidden = tf.nn.embedding_lookup(hidden, fw_sampled_neighbors)
        else:
            # Subsequent layers use computed hidden states
            # Pad with zeros for invalid indices
            padded_hidden = tf.concat([hidden, tf.zeros([1, input_dim])], 0)
            neigh_vec_hidden = tf.nn.embedding_lookup(padded_hidden, fw_sampled_neighbors)
            
        # Apply aggregator
        fw_hidden = fw_aggregator((hidden, neigh_vec_hidden))
        
        # Handle bidirectional
        if self.bidirectional and bw_sampled_neighbors is not None:
            bw_aggregator = EdgeGatedAggregator(
                input_dim=input_dim,
                output_dim=self.hidden_dim,
                concat=self.concat,
                dropout=self.dropout,
                name=f"bw_block{block_idx}_layer{layer_idx}"
            )
            self.bw_aggregators.append(bw_aggregator)
            
            if layer_idx == 0 and block_idx == 0:
                neigh_vec_hidden = tf.nn.embedding_lookup(hidden, bw_sampled_neighbors)
            else:
                padded_hidden = tf.concat([hidden, tf.zeros([1, input_dim])], 0)
                neigh_vec_hidden = tf.nn.embedding_lookup(padded_hidden, bw_sampled_neighbors)
                
            bw_hidden = bw_aggregator((hidden, neigh_vec_hidden))
            
            # Concatenate forward and backward
            hidden = tf.concat([fw_hidden, bw_hidden], axis=-1)
        else:
            hidden = fw_hidden
            
        # Residual connection + LayerNorm
        if hidden.get_shape()[-1] == residual.get_shape()[-1]:
            # Direct residual connection
            hidden = hidden + residual
        else:
            # Project residual to match dimensions
            with tf.variable_scope(f"residual_proj_block{block_idx}_layer{layer_idx}"):
                residual_proj = tf.layers.dense(residual, hidden.get_shape()[-1], 
                                               activation=None, use_bias=False)
                hidden = hidden + residual_proj
                
        # Layer normalization
        hidden = tf.contrib.layers.layer_norm(hidden, 
                                             scope=f"layer_norm_block{block_idx}_layer{layer_idx}")
        
        return hidden
        
    def sequence_to_graph(self, sequence_inputs):
        """
        Convert sequence inputs to graph representation.
        """
        batch_size = tf.shape(sequence_inputs)[0]
        seq_len = tf.shape(sequence_inputs)[1]
        feature_dim = tf.shape(sequence_inputs)[2]
        
        # Total number of nodes across all batches
        total_nodes = batch_size * seq_len
        
        # Create base indices for one sequence
        seq_indices = tf.range(seq_len)
        
        # Create fully connected adjacency within each sequence
        single_seq_adj = tf.tile(tf.expand_dims(seq_indices, 0), [seq_len, 1])
        
        # Create batch offsets
        batch_offsets = tf.reshape(tf.range(batch_size) * seq_len, [batch_size, 1, 1])
        
        # Expand to all batches
        batch_adj = tf.expand_dims(single_seq_adj, 0) + batch_offsets
        
        # Reshape to [total_nodes, seq_len]
        fw_adj_info = tf.reshape(batch_adj, [total_nodes, seq_len])
        bw_adj_info = fw_adj_info  # Same for backward
        
        # Flatten sequence features
        feature_info = tf.reshape(sequence_inputs, [total_nodes, feature_dim])
        
        # Create batch nodes tensor
        batch_nodes = tf.reshape(tf.range(total_nodes), [batch_size, seq_len])
        
        return fw_adj_info, bw_adj_info, feature_info, batch_nodes
        
    def encode(self, encoder_inputs):
        """
        Main encoding function with improved architecture.
        """
        # Convert sequence to graph
        fw_adj_info, bw_adj_info, feature_info, batch_nodes = self.sequence_to_graph(encoder_inputs)
        
        batch_size = tf.shape(encoder_inputs)[0]
        seq_len = tf.shape(encoder_inputs)[1]
        
        # Add virtual node if enabled
        if self.use_virtual_node:
            (feature_info, batch_nodes_with_virtual, 
             fw_adj_layer1, fw_adj_layer2, 
             bw_adj_layer1, bw_adj_layer2) = self.add_virtual_node(
                feature_info, batch_nodes, fw_adj_info, bw_adj_info)
            
            # Update seq_len to include virtual node
            seq_len_with_virtual = seq_len + 1
            total_nodes_with_virtual = batch_size * seq_len + batch_size
        else:
            fw_adj_layer1 = fw_adj_layer2 = fw_adj_info
            bw_adj_layer1 = bw_adj_layer2 = bw_adj_info
            batch_nodes_with_virtual = batch_nodes
            seq_len_with_virtual = seq_len
            total_nodes_with_virtual = batch_size * seq_len
            
        # Initialize node embeddings
        embedded_node_rep = feature_info
        
        # Create samplers for different layers
        fw_sampler_layer1 = UniformNeighborSampler(fw_adj_layer1)
        fw_sampler_layer2 = UniformNeighborSampler(fw_adj_layer2)
        if self.bidirectional:
            bw_sampler_layer1 = UniformNeighborSampler(bw_adj_layer1)
            bw_sampler_layer2 = UniformNeighborSampler(bw_adj_layer2)
            
        # Flatten nodes for processing
        nodes = tf.reshape(batch_nodes_with_virtual, [-1])
        
        # Initial hidden states
        hidden = tf.nn.embedding_lookup(embedded_node_rep, nodes)
        
        # Apply GNN blocks
        for block_idx in range(self.num_blocks):
            # Layer 1: real -> virtual (if virtual node enabled)
            if self.use_virtual_node and block_idx == 0:
                sampler = fw_sampler_layer1
                sample_size = seq_len + 1  # Include virtual node
            else:
                sampler = fw_sampler_layer2
                sample_size = seq_len_with_virtual
                
            fw_sampled_neighbors = sampler((nodes, sample_size))
            fw_sampled_neighbors_len = tf.fill([total_nodes_with_virtual], sample_size)
            
            if self.bidirectional:
                if self.use_virtual_node and block_idx == 0:
                    bw_sampler = bw_sampler_layer1
                else:
                    bw_sampler = bw_sampler_layer2
                bw_sampled_neighbors = bw_sampler((nodes, sample_size))
                bw_sampled_neighbors_len = fw_sampled_neighbors_len
            else:
                bw_sampled_neighbors = None
                bw_sampled_neighbors_len = None
                
            # Determine input dimension
            if block_idx == 0:
                input_dim = self.input_dim
            else:
                input_dim = 2 * self.hidden_dim if not self.bidirectional else 4 * self.hidden_dim
                
            # Apply GNN block
            hidden = self.gnn_block(
                hidden, fw_sampled_neighbors, fw_sampled_neighbors_len,
                bw_sampled_neighbors, bw_sampled_neighbors_len,
                block_idx=block_idx, layer_idx=0, input_dim=input_dim
            )
            
            # For layer 2: virtual -> real (if virtual node enabled)
            if self.use_virtual_node and block_idx < self.num_blocks - 1:
                # Use layer 2 adjacency
                fw_sampled_neighbors = fw_sampler_layer2((nodes, sample_size))
                if self.bidirectional:
                    bw_sampled_neighbors = bw_sampler_layer2((nodes, sample_size))
                    
                hidden = self.gnn_block(
                    hidden, fw_sampled_neighbors, fw_sampled_neighbors_len,
                    bw_sampled_neighbors, bw_sampled_neighbors_len,
                    block_idx=block_idx, layer_idx=1,
                    input_dim=2 * self.hidden_dim if not self.bidirectional else 4 * self.hidden_dim
                )
                
        # Reshape hidden states (excluding virtual nodes if present)
        if self.use_virtual_node:
            # Extract only real node hidden states
            real_node_indices = tf.reshape(batch_nodes, [-1])
            hidden_real = tf.gather(hidden, real_node_indices)
            final_hidden_dim = hidden.get_shape()[-1]
            hidden_real = tf.reshape(hidden_real, [batch_size, seq_len, final_hidden_dim])
        else:
            final_hidden_dim = hidden.get_shape()[-1]
            hidden_real = tf.reshape(hidden, [batch_size, seq_len, final_hidden_dim])
            
        encoder_outputs = hidden_real
        
        # Create encoder state compatible with LSTM decoder
        final_state = tf.reduce_max(encoder_outputs, axis=1)
        
        # Project state if needed
        if final_hidden_dim > self.hidden_dim:
            with tf.variable_scope("state_projection"):
                final_state_proj = tf.layers.dense(final_state, self.hidden_dim, 
                                                  activation=None, name="state_dense")
        else:
            final_state_proj = final_state
            
        # Create LSTM-compatible state
        if self.num_layers == 1:
            encoder_state = tf.nn.rnn_cell.LSTMStateTuple(c=final_state_proj, h=final_state_proj)
        else:
            encoder_state = tuple([
                tf.nn.rnn_cell.LSTMStateTuple(c=final_state_proj, h=final_state_proj)
                for _ in range(self.num_layers)
            ])
            
        return encoder_outputs, encoder_state


def create_improved_graph2seq_encoder(encoder_inputs, encoder_units, num_layers, 
                                     is_bidirectional, mode, feature_mode='full17',
                                     use_virtual_node=True, scope_name="encoder"):
    """
    Factory function to create improved Graph2Seq encoder.
    """
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
        input_dim = encoder_inputs.get_shape()[-1].value
        encoder = ImprovedGraph2SeqEncoder(
            input_dim=input_dim,
            hidden_dim=encoder_units,
            num_layers=num_layers,
            bidirectional=is_bidirectional,
            mode=mode,
            feature_mode=feature_mode,
            use_virtual_node=use_virtual_node
        )
        
        encoder_outputs, encoder_state = encoder.encode(encoder_inputs)
        
    return encoder_outputs, encoder_state