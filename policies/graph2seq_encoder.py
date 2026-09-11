"""
Graph2Seq encoder over canonical DAG adjacency packed into observations.

Neighbor indices live in the observation tail so the sampler stays a single
ndarray. Node-feature channels are embedded; adjacency indices are not.

Frozen v0.1: predecessor AND successor aggregators, summed; dropout 0.
v0.3: encoder_type in {meanagg, gatv2, dagformer}; default meanagg is the
v0.1 path. readout_type default triple.
"""
import tensorflow as tf

from env.mec_offloaing_envs.scheduler.encoder_obs import (
    DIRECTION_COMBINE,
    ENCODER_DROPOUT,
    FEATURE_DIM,
    FEATURE_NAMES,
    GNN_LAYERS,
    MAX_NEIGH,
    MAX_TASKS,
    PACKED_DIM,
    default_feature_stats,
)
from .graph2seq_modules.neigh_samplers import UniformNeighborSampler
from .graph2seq_modules.aggregators import MeanAggregator

VALID_ENCODER_TYPES = ("meanagg", "gatv2", "dagformer")
VALID_READOUT_TYPES = ("triple", "mean", "max", "attn", "zero")
GAT_HEADS = 4
GAT_HEAD_DIM = 32
# Width cut so n_params ≤ 2× meanagg (smoke: meanagg 429185, dagformer d=256 was 1290369).
DAGFORMER_D = 128
DAGFORMER_OUT = 256
DAGFORMER_HEADS = 4
DAGFORMER_FF = 256
DAGFORMER_LAYERS = 2
DEPTH_INDEX = FEATURE_NAMES.index("depth")


class Graph2SeqEncoderAdapter:
    """
    Adapter class that wraps Graph2Seq encoder to be compatible with metarl-offloading.
    Converts packed observations to DAG adjacency + node embeddings.
    Always consumes both successor (fw) and predecessor (bw) tables.
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_layers=2,
        bidirectional=True,
        mode="eval",
        encoder_type="meanagg",
        readout_type="triple",
        reachability_mask=None,
    ):
        if input_dim is not None and int(input_dim) != PACKED_DIM:
            raise ValueError(f"encoder packed dim {input_dim} != {PACKED_DIM}")
        if DIRECTION_COMBINE != "sum":
            raise ValueError(f"unsupported direction_combine: {DIRECTION_COMBINE}")
        encoder_type = str(encoder_type)
        readout_type = str(readout_type)
        if encoder_type not in VALID_ENCODER_TYPES:
            raise ValueError("encoder_type must be one of %s, got %r" % (VALID_ENCODER_TYPES, encoder_type))
        if readout_type not in VALID_READOUT_TYPES:
            raise ValueError("readout_type must be one of %s, got %r" % (VALID_READOUT_TYPES, readout_type))
        self.input_dim = PACKED_DIM
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = True
        self.mode = mode
        self.encoder_type = encoder_type
        self.readout_type = readout_type
        self.reachability_mask = reachability_mask
        self.sample_layer_size = GNN_LAYERS
        self.concat = True
        self.dropout = ENCODER_DROPOUT
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

    def _run_direction(self, hidden, sampled_neighbors, sampled_len, embedded_node_rep, aggregators):
        for layer in range(self.sample_layer_size):
            dim_mul = 1 if layer == 0 else 2
            aggregator = MeanAggregator(
                dim_mul * self.hidden_dim,
                self.hidden_dim,
                concat=self.concat,
                mode=self.mode,
                dropout=self.dropout,
            )
            aggregators.append(aggregator)
            if layer == 0:
                neigh_vec_hidden = tf.nn.embedding_lookup(embedded_node_rep, sampled_neighbors)
            else:
                padded_hidden = tf.concat([hidden, tf.zeros([1, dim_mul * self.hidden_dim])], 0)
                neigh_vec_hidden = tf.nn.embedding_lookup(padded_hidden, sampled_neighbors)
            hidden = aggregator((hidden, neigh_vec_hidden, sampled_len))
        return hidden

    def _encode_meanagg(self, encoder_inputs, unpacked):
        fw_adj_info, bw_adj_info, feature_slice, batch_nodes, fw_len, bw_len, _node_mask = unpacked
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
        bw_sampler = UniformNeighborSampler(bw_adj_info)
        nodes = tf.reshape(batch_nodes, [-1])
        fw_hidden = tf.nn.embedding_lookup(embedded_node_rep, nodes)
        bw_hidden = tf.nn.embedding_lookup(embedded_node_rep, nodes)
        fw_sampled_neighbors = fw_sampler((nodes, sample_size_per_layer))
        bw_sampled_neighbors = bw_sampler((nodes, sample_size_per_layer))

        fw_hidden = self._run_direction(
            fw_hidden, fw_sampled_neighbors, fw_len, embedded_node_rep, self.fw_aggregators
        )
        bw_hidden = self._run_direction(
            bw_hidden, bw_sampled_neighbors, bw_len, embedded_node_rep, self.bw_aggregators
        )

        fw_hidden = tf.reshape(fw_hidden, [batch_size, seq_len, 2 * self.hidden_dim])
        bw_hidden = tf.reshape(bw_hidden, [batch_size, seq_len, 2 * self.hidden_dim])
        encoder_outputs = fw_hidden + bw_hidden
        encoder_outputs = tf.nn.relu(encoder_outputs)
        return encoder_outputs

    def _gatv2_layer(self, hidden, neigh_idx, neigh_len, in_dim, name):
        """One GATv2 layer: 4 heads x 32, concat self||neigh → 256, ELU scores, no self-loop."""
        n_nodes = tf.shape(hidden)[0]
        padded = tf.concat([hidden, tf.zeros([1, in_dim])], 0)
        neigh_h = tf.nn.embedding_lookup(padded, neigh_idx)
        valid = tf.sequence_mask(neigh_len, maxlen=MAX_NEIGH)
        valid_f = tf.expand_dims(tf.cast(valid, tf.float32), -1)
        self_exp = tf.expand_dims(hidden, 1)
        self_tile = tf.tile(self_exp, [1, MAX_NEIGH, 1])
        pair = tf.concat([self_tile, neigh_h], axis=-1)
        heads = []
        with tf.variable_scope(name):
            for h in range(GAT_HEADS):
                u = tf.layers.dense(pair, GAT_HEAD_DIM, activation=tf.nn.elu, name="attn_%d" % h)
                e = tf.squeeze(tf.layers.dense(u, 1, activation=None, use_bias=False, name="score_%d" % h), -1)
                e = e + (1.0 - tf.squeeze(valid_f, -1)) * (-1e9)
                alpha = tf.nn.softmax(e, axis=1)
                val = tf.layers.dense(neigh_h, GAT_HEAD_DIM, activation=None, name="val_%d" % h)
                heads.append(tf.reduce_sum(tf.expand_dims(alpha, -1) * val * valid_f, axis=1))
            neigh_cat = tf.concat(heads, axis=-1)
            self_proj = tf.layers.dense(hidden, self.hidden_dim, activation=None, name="self_proj")
            out = tf.concat([self_proj, neigh_cat], axis=-1)
            out = tf.nn.relu(out)
        return out

    def _encode_gatv2(self, encoder_inputs, unpacked):
        fw_adj_info, bw_adj_info, feature_slice, batch_nodes, fw_len, bw_len, _node_mask = unpacked
        batch_size = tf.shape(encoder_inputs)[0]
        seq_len = tf.shape(encoder_inputs)[1]
        embedded_nodes = tf.layers.dense(
            feature_slice,
            self.hidden_dim,
            activation=None,
            name="node_feature_embed",
        )
        hidden0 = tf.reshape(embedded_nodes, [batch_size * seq_len, self.hidden_dim])
        nodes = tf.reshape(batch_nodes, [-1])
        fw_neigh = tf.nn.embedding_lookup(fw_adj_info, nodes)
        bw_neigh = tf.nn.embedding_lookup(bw_adj_info, nodes)

        with tf.variable_scope("gatv2_fw"):
            fw = self._gatv2_layer(hidden0, fw_neigh, fw_len, self.hidden_dim, "layer0")
            fw = self._gatv2_layer(fw, fw_neigh, fw_len, 2 * self.hidden_dim, "layer1")
        with tf.variable_scope("gatv2_bw"):
            bw = self._gatv2_layer(hidden0, bw_neigh, bw_len, self.hidden_dim, "layer0")
            bw = self._gatv2_layer(bw, bw_neigh, bw_len, 2 * self.hidden_dim, "layer1")
        fw = tf.reshape(fw, [batch_size, seq_len, 2 * self.hidden_dim])
        bw = tf.reshape(bw, [batch_size, seq_len, 2 * self.hidden_dim])
        return tf.nn.relu(fw + bw)

    def _ln(self, x, name):
        dim = x.get_shape().as_list()[-1]
        with tf.variable_scope(name):
            gain = tf.get_variable("gain", [dim], initializer=tf.ones_initializer())
            bias = tf.get_variable("bias", [dim], initializer=tf.zeros_initializer())
            mean, var = tf.nn.moments(x, [-1], keep_dims=True)
            return gain * (x - mean) * tf.rsqrt(var + 1e-5) + bias

    def _depth_ids(self, feature_slice):
        stats = default_feature_stats()
        mean = tf.constant(float(stats.mean[DEPTH_INDEX]), dtype=tf.float32)
        std = tf.constant(float(stats.std[DEPTH_INDEX]), dtype=tf.float32)
        raw = feature_slice[:, :, DEPTH_INDEX] * std + mean
        return tf.clip_by_value(tf.cast(tf.round(raw), tf.int32), 0, MAX_TASKS - 1)

    def _mha(self, x, mask, name):
        head_dim = DAGFORMER_D // DAGFORMER_HEADS
        with tf.variable_scope(name):
            q = tf.layers.dense(x, DAGFORMER_D, use_bias=False, name="q")
            k = tf.layers.dense(x, DAGFORMER_D, use_bias=False, name="k")
            v = tf.layers.dense(x, DAGFORMER_D, use_bias=False, name="v")
            batch = tf.shape(x)[0]
            slen = tf.shape(x)[1]
            def _split(t):
                t = tf.reshape(t, [batch, slen, DAGFORMER_HEADS, head_dim])
                return tf.transpose(t, [0, 2, 1, 3])
            q, k, v = _split(q), _split(k), _split(v)
            scale = tf.rsqrt(tf.cast(head_dim, tf.float32))
            scores = tf.matmul(q, k, transpose_b=True) * scale
            mask_b = tf.expand_dims(mask, 1)
            scores = scores + (1.0 - mask_b) * (-1e9)
            attn = tf.nn.softmax(scores, axis=-1)
            ctx = tf.matmul(attn, v)
            ctx = tf.transpose(ctx, [0, 2, 1, 3])
            ctx = tf.reshape(ctx, [batch, slen, DAGFORMER_D])
            return tf.layers.dense(ctx, DAGFORMER_D, use_bias=False, name="o")

    def _encode_dagformer(self, encoder_inputs, unpacked):
        _fw, _bw, feature_slice, _nodes, _fw_len, _bw_len, node_mask = unpacked
        batch_size = tf.shape(encoder_inputs)[0]
        seq_len = tf.shape(encoder_inputs)[1]
        x = tf.layers.dense(feature_slice, DAGFORMER_D, activation=None, name="node_feature_embed")
        depth_ids = self._depth_ids(feature_slice)
        depth_table = tf.get_variable(
            "depth_embedding",
            [MAX_TASKS, DAGFORMER_D],
            initializer=tf.random_uniform_initializer(-0.05, 0.05),
        )
        x = x + tf.nn.embedding_lookup(depth_table, depth_ids)
        if self.reachability_mask is None:
            raise ValueError("dagformer requires reachability_mask placeholder [B,N,N]")
        mask = tf.cast(self.reachability_mask, tf.float32)
        node_m = tf.expand_dims(node_mask, 1) * tf.expand_dims(node_mask, 2)
        mask = mask * node_m
        for layer in range(DAGFORMER_LAYERS):
            with tf.variable_scope("dagformer_layer_%d" % layer):
                y = self._ln(x, "ln1")
                x = x + self._mha(y, mask, "mha")
                y = self._ln(x, "ln2")
                h = tf.layers.dense(y, DAGFORMER_FF, activation=tf.nn.relu, name="ff1")
                h = tf.layers.dense(h, DAGFORMER_D, activation=None, name="ff2")
                x = x + h
        return tf.layers.dense(x, DAGFORMER_OUT, activation=None, name="node_out_proj")

    def _readout(self, encoder_outputs, node_mask):
        mask = tf.expand_dims(node_mask, axis=-1)
        neg_inf = (1.0 - mask) * (-1e9)
        mask_sum = tf.maximum(tf.reduce_sum(mask, axis=1), 1e-8)
        mean_pool = tf.reduce_sum(encoder_outputs * mask, axis=1) / mask_sum
        max_pool = tf.reduce_max(encoder_outputs + neg_inf, axis=1)
        if self.readout_type == "zero":
            batch = tf.shape(encoder_outputs)[0]
            final_state_proj = tf.zeros([batch, self.hidden_dim], dtype=encoder_outputs.dtype)
            return final_state_proj
        if self.readout_type == "triple" or self.readout_type == "attn":
            attn_logits = tf.layers.dense(encoder_outputs, 1, activation=None, name="readout_attn_logits")
            attn_weights = tf.nn.softmax(attn_logits + neg_inf, axis=1)
            attn_pool = tf.reduce_sum(encoder_outputs * attn_weights, axis=1)
        else:
            attn_pool = mean_pool
        if self.readout_type == "triple":
            pooled = tf.concat([mean_pool, max_pool, attn_pool], axis=-1)
        elif self.readout_type == "mean":
            pooled = mean_pool
        elif self.readout_type == "max":
            pooled = max_pool
        elif self.readout_type == "attn":
            pooled = attn_pool
        else:
            raise ValueError("readout_type %r" % self.readout_type)
        if self.readout_type == "triple":
            final_state = tf.layers.dense(
                pooled,
                units=2 * self.hidden_dim,
                activation=tf.tanh,
                name="readout_proj",
            )
            state_size = 2 * self.hidden_dim
            if state_size > self.hidden_dim:
                with tf.variable_scope("state_projection"):
                    final_state_proj = tf.layers.dense(
                        final_state, self.hidden_dim, activation=None, name="state_dense"
                    )
            else:
                final_state_proj = final_state
            return final_state_proj
        final_state_proj = tf.layers.dense(
            pooled, self.hidden_dim, activation=tf.tanh, name="readout_proj"
        )
        return final_state_proj

    def encode(self, encoder_inputs):
        """
        Main encoding function that maintains compatibility with metarl-offloading.
        Input: packed encoder_inputs [batch_size, seq_len, PACKED_DIM]
        Output: (encoder_outputs, encoder_state) matching original interface
        """
        unpacked = self.packed_to_graph(encoder_inputs)
        node_mask = unpacked[-1]
        if self.encoder_type == "meanagg":
            encoder_outputs = self._encode_meanagg(encoder_inputs, unpacked)
        elif self.encoder_type == "gatv2":
            encoder_outputs = self._encode_gatv2(encoder_inputs, unpacked)
        else:
            encoder_outputs = self._encode_dagformer(encoder_inputs, unpacked)

        final_state_proj = self._readout(encoder_outputs, node_mask)

        if self.num_layers == 1:
            encoder_state = tf.nn.rnn_cell.LSTMStateTuple(c=final_state_proj, h=final_state_proj)
        else:
            encoder_state = tuple([
                tf.nn.rnn_cell.LSTMStateTuple(c=final_state_proj, h=final_state_proj)
                for _ in range(self.num_layers)
            ])

        return encoder_outputs, encoder_state


def create_graph2seq_encoder(
    encoder_inputs,
    encoder_units,
    num_layers,
    is_bidirectional,
    mode,
    scope_name="encoder",
    encoder_type="meanagg",
    readout_type="triple",
    reachability_mask=None,
):
    """
    Factory function to create Graph2Seq encoder matching the original interface.

    `is_bidirectional` and `mode` are ignored for neighborhood/dropout: v0.1 always
    aggregates predecessors and successors and freezes encoder dropout to 0.
    """
    del is_bidirectional, mode
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
        input_dim = encoder_inputs.get_shape()[-1].value
        encoder_adapter = Graph2SeqEncoderAdapter(
            input_dim=input_dim,
            hidden_dim=encoder_units,
            num_layers=num_layers,
            bidirectional=True,
            mode="eval",
            encoder_type=encoder_type,
            readout_type=readout_type,
            reachability_mask=reachability_mask,
        )
        encoder_outputs, encoder_state = encoder_adapter.encode(encoder_inputs)
    return encoder_outputs, encoder_state


def count_encoder_params(scope_name="encoder"):
    total = 0
    prefix = scope_name + "/"
    for var in tf.compat.v1.trainable_variables():
        if var.name.startswith(prefix) or ("/" + prefix) in ("/" + var.name):
            n = 1
            for d in var.get_shape().as_list():
                n *= int(d or 1)
            total += n
    return int(total)
