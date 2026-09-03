"""
Graph2Seq encoder over canonical DAG adjacency packed into observations.

Neighbor indices live in the observation tail so the sampler stays a single
ndarray. Node-feature channels are embedded; adjacency indices are not.
"""
import tensorflow as tf

from env.mec_offloaing_envs.scheduler.encoder_obs import FEATURE_DIM, MAX_NEIGH, PACKED_DIM
from .graph2seq_modules.neigh_samplers import UniformNeighborSampler
from .graph2seq_modules.aggregators import MeanAggregator


class Graph2SeqEncoderAdapter:
    """
    Adapter class that wraps Graph2Seq encoder to be compatible with metarl-offloading.
    Converts packed observations to DAG adjacency + node embeddings.
    """

    def __init__(self, input_dim, hidden_dim, num_layers=2, bidirectional=False, mode='train'):
        if input_dim is not None and int(input_dim) != PACKED_DIM:
            raise ValueError(f"encoder packed dim {input_dim} != {PACKED_DIM}")
        self.input_dim = PACKED_DIM
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.mode = mode

        self.sample_layer_size = 2
        self.concat = True
        self.dropout = 0.0 if mode != 'train' else 0.1

        self.fw_aggregators = []
        self.bw_aggregators = []

    def packed_to_graph(self, packed_inputs):
        """Unpack packed obs [B, N, PACKED_DIM] into DAG adj + raw node features."""
        feature_slice = packed_inputs[:, :, :FEATURE_DIM]
        fw_local = tf.cast(packed_inputs[:, :, FEATURE_DIM:FEATURE_DIM + MAX_NEIGH], tf.int32)
        bw_local = tf.cast(
            packed_inputs[:, :, FEATURE_DIM + MAX_NEIGH:FEATURE_DIM + 2 * MAX_NEIGH],
            tf.int32,
        )
        node_mask = packed_inputs[:, :, -1]

        batch_size = tf.shape(packed_inputs)[0]
        seq_len = tf.shape(packed_inputs)[1]
        total_nodes = batch_size * seq_len
        dummy = total_nodes

        def _globalize(local_adj):
            valid = tf.greater_equal(local_adj, 0)
            batch_offsets = tf.reshape(tf.range(batch_size) * seq_len, [batch_size, 1, 1])
            shifted = local_adj + batch_offsets
            dummy_fill = tf.fill(tf.shape(local_adj), dummy)
            global_adj = tf.where(valid, shifted, dummy_fill)
            lengths = tf.reshape(tf.reduce_sum(tf.cast(valid, tf.int32), axis=2), [total_nodes])
            return tf.reshape(global_adj, [total_nodes, MAX_NEIGH]), lengths

        fw_adj_info, fw_len = _globalize(fw_local)
        bw_adj_info, bw_len = _globalize(bw_local)
        batch_nodes = tf.reshape(tf.range(total_nodes), [batch_size, seq_len])
        return fw_adj_info, bw_adj_info, feature_slice, batch_nodes, fw_len, bw_len, node_mask

    def encode(self, encoder_inputs):
        """
        Main encoding function that maintains compatibility with metarl-offloading.
        Input: packed encoder_inputs [batch_size, seq_len, PACKED_DIM]
        Output: (encoder_outputs, encoder_state) matching original interface
        """
        fw_adj_info, bw_adj_info, feature_slice, batch_nodes, fw_len, bw_len, node_mask = (
            self.packed_to_graph(encoder_inputs)
        )

        batch_size = tf.shape(encoder_inputs)[0]
        seq_len = tf.shape(encoder_inputs)[1]
        sample_size_per_layer = MAX_NEIGH

        embedded_nodes = tf.layers.dense(
            feature_slice,
            self.hidden_dim,
            activation=None,
            name="node_feature_embed",
        )
        feature_info = tf.reshape(embedded_nodes, [batch_size * seq_len, self.hidden_dim])
        embedded_node_rep = tf.concat([feature_info, tf.zeros([1, self.hidden_dim])], 0)

        fw_sampler = UniformNeighborSampler(fw_adj_info)
        if self.bidirectional:
            bw_sampler = UniformNeighborSampler(bw_adj_info)

        nodes = tf.reshape(batch_nodes, [-1])
        fw_hidden = tf.nn.embedding_lookup(embedded_node_rep, nodes)
        if self.bidirectional:
            bw_hidden = tf.nn.embedding_lookup(embedded_node_rep, nodes)

        fw_sampled_neighbors = fw_sampler((nodes, sample_size_per_layer))
        if self.bidirectional:
            bw_sampled_neighbors = bw_sampler((nodes, sample_size_per_layer))

        fw_sampled_neighbors_len = fw_len
        if self.bidirectional:
            bw_sampled_neighbors_len = bw_len

        for layer in range(self.sample_layer_size):
            if layer == 0:
                dim_mul = 1
            else:
                dim_mul = 2

            fw_aggregator = MeanAggregator(
                dim_mul * self.hidden_dim,
                self.hidden_dim,
                concat=self.concat,
                mode=self.mode,
                dropout=self.dropout,
            )
            self.fw_aggregators.append(fw_aggregator)

            if layer == 0:
                neigh_vec_hidden = tf.nn.embedding_lookup(embedded_node_rep, fw_sampled_neighbors)
            else:
                padded_hidden = tf.concat([fw_hidden, tf.zeros([1, dim_mul * self.hidden_dim])], 0)
                neigh_vec_hidden = tf.nn.embedding_lookup(padded_hidden, fw_sampled_neighbors)

            fw_hidden = fw_aggregator((fw_hidden, neigh_vec_hidden, fw_sampled_neighbors_len))

            if self.bidirectional:
                bw_aggregator = MeanAggregator(
                    dim_mul * self.hidden_dim,
                    self.hidden_dim,
                    concat=self.concat,
                    mode=self.mode,
                    dropout=self.dropout,
                )
                self.bw_aggregators.append(bw_aggregator)

                if layer == 0:
                    neigh_vec_hidden = tf.nn.embedding_lookup(embedded_node_rep, bw_sampled_neighbors)
                else:
                    padded_hidden = tf.concat([bw_hidden, tf.zeros([1, dim_mul * self.hidden_dim])], 0)
                    neigh_vec_hidden = tf.nn.embedding_lookup(padded_hidden, bw_sampled_neighbors)

                bw_hidden = bw_aggregator((bw_hidden, neigh_vec_hidden, bw_sampled_neighbors_len))

        fw_hidden = tf.reshape(fw_hidden, [batch_size, seq_len, 2 * self.hidden_dim])

        if self.bidirectional:
            bw_hidden = tf.reshape(bw_hidden, [batch_size, seq_len, 2 * self.hidden_dim])
            encoder_outputs = tf.concat([fw_hidden, bw_hidden], axis=2)
        else:
            encoder_outputs = fw_hidden

        encoder_outputs = tf.nn.relu(encoder_outputs)

        mask = tf.expand_dims(node_mask, axis=-1)
        neg_inf = (1.0 - mask) * (-1e9)
        attn_logits = tf.layers.dense(encoder_outputs, 1, activation=None, name="readout_attn_logits")
        attn_weights = tf.nn.softmax(attn_logits + neg_inf, axis=1)
        attn_pool = tf.reduce_sum(encoder_outputs * attn_weights, axis=1)
        mask_sum = tf.maximum(tf.reduce_sum(mask, axis=1), 1e-8)
        mean_pool = tf.reduce_sum(encoder_outputs * mask, axis=1) / mask_sum
        max_pool = tf.reduce_max(encoder_outputs + neg_inf, axis=1)
        final_state = tf.layers.dense(
            tf.concat([mean_pool, max_pool, attn_pool], axis=-1),
            units=(4 * self.hidden_dim) if self.bidirectional else (2 * self.hidden_dim),
            activation=tf.tanh,
            name="readout_proj"
        )
        if self.bidirectional:
            state_size = 4 * self.hidden_dim
        else:
            state_size = 2 * self.hidden_dim

        if state_size > self.hidden_dim:
            with tf.variable_scope("state_projection"):
                final_state_proj = tf.layers.dense(final_state, self.hidden_dim,
                                                  activation=None,
                                                  name="state_dense")
        else:
            final_state_proj = final_state

        if self.num_layers == 1:
            encoder_state = tf.nn.rnn_cell.LSTMStateTuple(c=final_state_proj, h=final_state_proj)
        else:
            encoder_state = tuple([
                tf.nn.rnn_cell.LSTMStateTuple(c=final_state_proj, h=final_state_proj)
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
