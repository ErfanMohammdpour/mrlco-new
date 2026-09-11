import os
import joblib

import numpy as np
import tensorflow as tf
import policies.model_helper as model_helper
from policies.graph2seq_encoder import create_graph2seq_encoder

from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops.distributions import categorical
from policies.distributions.categorical_pd import CategoricalPd
import utils as U
from utils.utils import zipsame

try:
    tf.get_logger().setLevel('WARNING')
except AttributeError:
    pass

def _nucleus_logits(logits, top_p):
    """Keep smallest prefix whose preceding mass < top_p. Always keep top-1."""
    vocab = tf.shape(logits)[-1]
    batch = tf.shape(logits)[0]
    probs = tf.nn.softmax(logits)
    sorted_probs, sorted_idx = tf.nn.top_k(probs, k=vocab, sorted=True)
    keep_sorted = tf.less(tf.cumsum(sorted_probs, axis=-1) - sorted_probs, top_p)
    keep_sorted = tf.logical_or(
        keep_sorted, tf.equal(tf.range(vocab)[None, :], 0)
    )
    b_idx = tf.tile(tf.expand_dims(tf.range(batch), 1), [1, vocab])
    scatter_idx = tf.stack([b_idx, sorted_idx], axis=-1)
    keep = tf.scatter_nd(
        scatter_idx, tf.cast(keep_sorted, tf.float32), tf.shape(logits)
    )
    neg = tf.fill(tf.shape(logits), tf.cast(-1e9, logits.dtype))
    return tf.where(tf.greater(keep, 0.5), logits, neg)


class FixedSequenceLearningSampleEmbedingHelper(tf.contrib.seq2seq.SampleEmbeddingHelper):
    def __init__(
        self,
        sequence_length,
        embedding,
        start_tokens,
        end_token,
        softmax_temperature=None,
        seed=None,
        top_p=None,
    ):
        super(FixedSequenceLearningSampleEmbedingHelper, self).__init__(
            embedding, start_tokens, end_token, softmax_temperature, seed
        )
        self._top_p = top_p
        self._sequence_length = ops.convert_to_tensor(
            sequence_length, name="sequence_length")
        if self._sequence_length.get_shape().ndims != 1:
            raise ValueError(
                "Expected sequence_length to be a vector, but received shape: %s" %
                self._sequence_length.get_shape())

    def sample(self, time, outputs, state, name=None):
        """sample for SampleEmbeddingHelper."""
        del time, state  # unused by sample_fn
        # Outputs are logits, we sample instead of argmax (greedy).
        if not isinstance(outputs, ops.Tensor):
            raise TypeError("Expected outputs to be a single Tensor, got: %s" %
                            type(outputs))
        if self._softmax_temperature is None:
            logits = outputs
        else:
            logits = outputs / self._softmax_temperature

        def _plain():
            return categorical.Categorical(logits=logits).sample(seed=self._seed)

        if self._top_p is None:
            return _plain()

        def _nucleus():
            masked = _nucleus_logits(logits, self._top_p)
            return categorical.Categorical(logits=masked).sample(seed=self._seed)

        return control_flow_ops.cond(
            math_ops.greater(self._top_p, 0.0), _nucleus, _plain
        )

    def next_inputs(self, time, outputs, state, sample_ids, name=None):
        """next_inputs_fn for GreedyEmbeddingHelper."""
        del outputs  # unused by next_inputs_fn

        next_time = time + 1
        finished = (next_time >= self._sequence_length)
        all_finished = math_ops.reduce_all(finished)

        next_inputs = control_flow_ops.cond(
            all_finished,
            # If we're finished, the next_inputs value doesn't matter
            lambda: self._start_inputs,
            lambda: self._embedding_fn(sample_ids))
        return (finished, next_inputs, state)


class Seq2SeqNetwork():
    def __init__(self, name,
                 hparams, reuse,
                 encoder_inputs,
                 decoder_inputs,
                 decoder_full_length,
                 decoder_targets,
                 dist_ids=None,
                 reachability_mask=None,
                 ctx_obs=None,
                 ctx_acts=None,
                 ctx_t=None,
                 ctx_zero=None):
        self.encoder_hidden_unit = hparams.encoder_units
        self.decoder_hidden_unit = hparams.decoder_units
        self.is_bidencoder = hparams.is_bidencoder
        self.reuse = reuse

        self.n_features = hparams.n_features
        self.time_major = hparams.time_major
        self.is_attention = hparams.is_attention

        self.unit_type = hparams.unit_type

        # default setting
        self.mode = tf.contrib.learn.ModeKeys.TRAIN

        self.num_layers = hparams.num_layers
        self.num_residual_layers = hparams.num_residual_layers

        self.single_cell_fn = None
        self.start_token = hparams.start_token
        self.end_token = hparams.end_token

        self.encoder_inputs = encoder_inputs
        self.decoder_inputs = decoder_inputs
        self.decoder_targets = decoder_targets

        self.decoder_full_length = decoder_full_length
        self.enable_cavia = bool(getattr(hparams, "enable_cavia", False))
        self.cavia_z_dim = int(getattr(hparams, "cavia_z_dim", 32))
        self.cavia_z = None
        self.cavia_film_trainable = bool(getattr(hparams, "cavia_film_trainable", False))
        self.enable_eas_emb = bool(getattr(hparams, "enable_eas_emb", False))
        self.eas_emb_delta = None
        self.enable_context_encoder = bool(getattr(hparams, "enable_context_encoder", False))
        self.ctx_obs = ctx_obs
        self.ctx_acts = ctx_acts
        self.ctx_t = ctx_t
        self.ctx_zero = ctx_zero
        self.context_z = None

        with tf.compat.v1.variable_scope(name, reuse=self.reuse, initializer=tf.glorot_normal_initializer()):
            self.scope = tf.compat.v1.get_variable_scope().name
            self.embeddings = tf.Variable(tf.random.uniform(
                [self.n_features,
                 self.encoder_hidden_unit],
                -1.0, 1.0), dtype=tf.float32)

            self.decoder_embeddings = tf.nn.embedding_lookup(self.embeddings,
                                                             self.decoder_inputs)

            self.enable_oracle_dist = bool(getattr(hparams, "enable_oracle_dist", False))
            self.oracle_n_dist = int(getattr(hparams, "oracle_n_dist", 26))
            self.oracle_z_dim = int(getattr(hparams, "oracle_z_dim", 32))
            self.dist_ids = dist_ids
            self.oracle_dist_table = None
            self._dist_delta = None
            if self.enable_oracle_dist:
                if dist_ids is None:
                    raise ValueError("enable_oracle_dist requires dist_ids")
                self._dist_delta = self._build_oracle_dist_delta()
                self.decoder_embeddings = self.decoder_embeddings + self._dist_delta[:, None, :]

            self.decoder_targets_embeddings = tf.one_hot(self.decoder_targets,
                                                         self.n_features,
                                                         dtype=tf.float32)

            self.output_layer = tf.compat.v1.layers.Dense(self.n_features, use_bias=False, name="output_projection")

            if self.enable_cavia and not self.enable_context_encoder:
                self.cavia_z = tf.compat.v1.get_variable(
                    "cavia_z",
                    shape=[self.cavia_z_dim],
                    dtype=tf.float32,
                    initializer=tf.zeros_initializer(),
                    trainable=True,
                )
            if self.enable_context_encoder:
                if ctx_obs is None or ctx_acts is None or ctx_t is None:
                    raise ValueError("enable_context_encoder needs ctx placeholders")
                self.context_z = self._build_context_z()
                self.enable_cavia = True
                self.cavia_film_trainable = True

            # Packed obs already carries DAG adj; embed node features inside Graph2Seq.
            self.encoder_type = str(getattr(hparams, "encoder_type", "meanagg"))
            self.readout_type = str(getattr(hparams, "readout_type", "triple"))
            self.encoder_outputs, self.encoder_state = create_graph2seq_encoder(
                encoder_inputs=self.encoder_inputs,
                encoder_units=self.encoder_hidden_unit,
                num_layers=self.num_layers,
                is_bidirectional=self.is_bidencoder,
                mode=self.mode,
                scope_name="encoder",
                encoder_type=str(getattr(hparams, "encoder_type", "meanagg")),
                readout_type=str(getattr(hparams, "readout_type", "triple")),
                reachability_mask=reachability_mask,
            )
            # EAS-Emb: residual [20, 256] on node embeddings; zero-init => identity.
            if self.enable_eas_emb:
                out_dim = int(self.encoder_outputs.get_shape().as_list()[-1] or 256)
                self.eas_emb_delta = tf.compat.v1.get_variable(
                    "eas_emb_delta",
                    shape=[20, out_dim],
                    dtype=tf.float32,
                    initializer=tf.zeros_initializer(),
                    trainable=True,
                )
                self.encoder_outputs = self.encoder_outputs + self.eas_emb_delta[None, :, :]

            # training decoder
            self.decoder_outputs, self.decoder_state = self.create_decoder(hparams, self.encoder_outputs,
                                                                           self.encoder_state, model="train")
            self.decoder_logits = self.decoder_outputs.rnn_output
            self.pi = tf.nn.softmax(self.decoder_logits)
            self.q = tf.compat.v1.layers.dense(self.decoder_logits, self.n_features, activation=None,
                                     reuse=tf.compat.v1.AUTO_REUSE, name="qvalue_layer")
            self.vf = tf.reduce_sum(self.pi * self.q, axis=-1)

            self.decoder_prediction = self.decoder_outputs.sample_id

            self.sample_softmax_temperature = tf.compat.v1.placeholder_with_default(
                tf.constant(1.0, dtype=tf.float32),
                shape=(),
                name="sample_softmax_temperature",
            )
            self.sample_top_p = tf.compat.v1.placeholder_with_default(
                tf.constant(-1.0, dtype=tf.float32),
                shape=(),
                name="sample_top_p",
            )

            # sample decoder
            self.sample_decoder_outputs, self.sample_decoder_state = self.create_decoder(hparams, self.encoder_outputs,
                                                                           self.encoder_state, model="sample")
            self.sample_decoder_logits = self.sample_decoder_outputs.rnn_output
            self.sample_pi = tf.nn.softmax(self.sample_decoder_logits)
            self.sample_q = tf.compat.v1.layers.dense(self.sample_decoder_logits, self.n_features,
                                            activation=None, reuse=tf.compat.v1.AUTO_REUSE, name="qvalue_layer")

            self.sample_vf = tf.reduce_sum(self.sample_pi*self.sample_q, axis=-1)

            self.sample_decoder_prediction = self.sample_decoder_outputs.sample_id

            # Note: we can't use sparse_softmax_cross_entropy_with_logits
            self.sample_decoder_embeddings = tf.one_hot(self.sample_decoder_prediction,
                                                        self.n_features,
                                                        dtype=tf.float32)

            self.sample_neglogp = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.sample_decoder_embeddings,
                                                                             logits=self.sample_decoder_logits)

            # greedy decoder
            self.greedy_decoder_outputs, self.greedy_decoder_state = self.create_decoder(hparams, self.encoder_outputs,
                                                                           self.encoder_state, model="greedy")
            self.greedy_decoder_logits = self.greedy_decoder_outputs.rnn_output
            self.greedy_pi = tf.nn.softmax(self.greedy_decoder_logits)
            self.greedy_q = tf.compat.v1.layers.dense(self.greedy_decoder_logits, self.n_features, activation=None, reuse=tf.compat.v1.AUTO_REUSE,
                                     name="qvalue_layer")
            self.greedy_vf = tf.reduce_sum(self.greedy_pi * self.greedy_q, axis=-1)

            self.greedy_decoder_prediction = self.greedy_decoder_outputs.sample_id

    def predict_training(self, sess, encoder_input_batch, decoder_input, decoder_full_length):
        return sess.run([self.decoder_prediction, self.pi],
                        feed_dict={
                            self.encoder_inputs: encoder_input_batch,
                            self.decoder_inputs: decoder_input,
                            self.decoder_full_length: decoder_full_length
                        })

    def kl(self, other):
        a0 = self.decoder_logits - tf.reduce_max(self.decoder_logits, axis=-1, keepdims=True)
        a1 = other.decoder_logits - tf.reduce_max(other.decoder_logits, axis=-1, keepdims=True)
        ea0 = tf.exp(a0)
        ea1 = tf.exp(a1)
        z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
        z1 = tf.reduce_sum(ea1, axis=-1, keepdims=True)
        p0 = ea0 / z0
        return tf.reduce_sum(p0 * (a0 - tf.log(z0) - a1 + tf.log(z1)), axis=-1)

    def entropy(self):
        a0 = self.decoder_logits - tf.reduce_max(self.decoder_logits, axis=-1, keepdims=True)
        ea0 = tf.exp(a0)
        z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
        p0 = ea0 / z0
        return tf.reduce_sum(p0 * (tf.log(z0) - a0), axis=-1)

    def neglogp(self):
        # return tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=x)
        # Note: we can't use sparse_softmax_cross_entropy_with_logits because
        #       the implementation does not allow second-order derivatives...
        return tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=self.decoder_logits,
            labels=self.decoder_targets_embeddings)

    def logp(self):
        return -self.neglogp()

    # DEPRECATED: Original encoder cell builder - replaced by Graph2Seq encoder
    # def _build_encoder_cell(self, hparams, num_layers, num_residual_layers, base_gpu=0):
    #     """Build a multi-layer RNN cell that can be used by encoder."""
    #     return model_helper.create_rnn_cell(
    #         unit_type=hparams.unit_type,
    #         num_units=hparams.encoder_units,
    #         num_layers=num_layers,
    #         num_residual_layers=num_residual_layers,
    #         forget_bias=hparams.forget_bias,
    #         dropout=hparams.dropout,
    #         num_gpus=hparams.num_gpus,
    #         mode=self.mode,
    #         base_gpu=base_gpu,
    #         single_cell_fn=self.single_cell_fn)

    def _build_oracle_dist_delta(self):
        """True dist_id embed added at every decoder step. Zero-init proj => identity."""
        with tf.compat.v1.variable_scope("oracle_dist"):
            self.oracle_dist_table = tf.compat.v1.get_variable(
                "table",
                shape=[self.oracle_n_dist, self.oracle_z_dim],
                dtype=tf.float32,
                initializer=tf.glorot_normal_initializer(),
                trainable=True,
            )
            ids = tf.clip_by_value(self.dist_ids, 0, self.oracle_n_dist - 1)
            z = tf.nn.embedding_lookup(self.oracle_dist_table, ids)
            delta = tf.compat.v1.layers.dense(
                z,
                self.encoder_hidden_unit,
                activation=None,
                use_bias=False,
                kernel_initializer=tf.zeros_initializer(),
                name="delta",
            )
        return delta

    def _build_context_z(self):
        """Permutation-invariant PEARL-style z from support (obs, plan, T). No VAE."""
        obs_m = tf.reduce_mean(self.ctx_obs, axis=1)
        act_oh = tf.one_hot(self.ctx_acts, depth=self.n_features, dtype=tf.float32)
        act_m = tf.reduce_mean(act_oh, axis=1)
        t = tf.reshape(tf.math.log1p(tf.maximum(self.ctx_t, 0.0)) / 6.5, [-1, 1])
        x = tf.concat([obs_m, act_m, t], axis=-1)
        h = tf.compat.v1.layers.dense(x, 64, activation=tf.nn.relu, name="context_fc1")
        h = tf.compat.v1.layers.dense(h, 64, activation=tf.nn.relu, name="context_fc2")
        pooled = tf.reduce_mean(h, axis=0, keepdims=True)
        z = tf.compat.v1.layers.dense(
            pooled,
            self.cavia_z_dim,
            activation=None,
            use_bias=False,
            name="context_z",
        )
        z = tf.reshape(z, [self.cavia_z_dim])
        if self.ctx_zero is not None:
            z = tf.cond(
                tf.convert_to_tensor(self.ctx_zero),
                lambda: tf.zeros_like(z),
                lambda: z,
            )
        return z

    def _decoder_token_embed(self, ids):
        tok = tf.nn.embedding_lookup(self.embeddings, ids)
        if self._dist_delta is None:
            return tok
        return tok + self._dist_delta

    def _film_encoder_state(self, encoder_state):
        """Additive residual from z. z=0 => identity (bias-free dense)."""
        z_src = self.context_z if self.context_z is not None else self.cavia_z
        if (not self.enable_cavia) or z_src is None:
            return encoder_state
        batch = tf.size(self.decoder_full_length)
        z_b = tf.tile(tf.reshape(z_src, [1, self.cavia_z_dim]), [batch, 1])
        with tf.compat.v1.variable_scope("cavia_film", reuse=tf.compat.v1.AUTO_REUSE):
            delta = tf.compat.v1.layers.dense(
                z_b,
                self.decoder_hidden_unit,
                activation=None,
                use_bias=False,
                trainable=bool(self.cavia_film_trainable),
                name="delta",
                kernel_initializer=tf.zeros_initializer()
                if self.cavia_film_trainable
                else tf.glorot_normal_initializer(),
            )

        def _one(st):
            return tf.nn.rnn_cell.LSTMStateTuple(c=st.c + delta, h=st.h + delta)

        if isinstance(encoder_state, (tuple, list)):
            return tuple(_one(st) for st in encoder_state)
        return _one(encoder_state)

    def _build_decoder_cell(self, hparams, num_layers, num_residual_layers, base_gpu=0):
        """Build a multi-layer RNN cell that can be used by decoder"""
        return model_helper.create_rnn_cell(
            unit_type=hparams.unit_type,
            num_units=hparams.decoder_units,
            num_layers=num_layers,
            num_residual_layers=num_residual_layers,
            forget_bias=hparams.forget_bias,
            dropout=hparams.dropout,
            num_gpus=hparams.num_gpus,
            mode=self.mode,
            base_gpu=base_gpu,
            single_cell_fn=self.single_cell_fn)

    # DEPRECATED: Original RNN encoder - replaced by Graph2Seq encoder
    # def create_encoder(self, hparams):
    #     # Build RNN cell
    #     with tf.compat.v1.variable_scope("encoder", reuse=tf.compat.v1.AUTO_REUSE) as scope:
    #         encoder_cell = self._build_encoder_cell(hparams=hparams,
    #                                                 num_layers=self.num_layers,
    #                                                 num_residual_layers=self.num_residual_layers)
    #
    #         # encoder_cell = tf.contrib.rnn.GRUCell(self.encoder_hidden_unit)
    #         # currently only consider the normal dynamic rnn
    #         encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
    #             cell=encoder_cell,
    #             sequence_length = None,
    #             inputs=self.encoder_embeddings,
    #             dtype=tf.float32,
    #             time_major=self.time_major,
    #             swap_memory=True,
    #             scope=scope
    #         )
    #
    #     return encoder_outputs, encoder_state

    # DEPRECATED: Original bidirectional RNN encoder - replaced by Graph2Seq encoder
    # def create_bidrect_encoder(self, hparams):
    #     with tf.compat.v1.variable_scope("encoder", reuse=tf.compat.v1.AUTO_REUSE) as scope:
    #         num_bi_layers = int(self.num_layers / 2)
    #         num_bi_residual_layers = int(self.num_residual_layers / 2)
    #         forward_cell = self._build_encoder_cell(hparams=hparams,
    #                                                 num_layers=num_bi_layers,
    #                                                 num_residual_layers=num_bi_residual_layers)
    #         backward_cell = self._build_encoder_cell(hparams=hparams,
    #                                                  num_layers=num_bi_layers,
    #                                                  num_residual_layers=num_bi_residual_layers)
    #
    #         bi_outputs, bi_state = tf.nn.bidirectional_dynamic_rnn(
    #             forward_cell,
    #             backward_cell,
    #             inputs=self.encoder_embeddings,
    #             time_major=self.time_major,
    #             swap_memory=True,
    #             dtype=tf.float32)
    #
    #         encoder_outputs = tf.concat(bi_outputs, -1)
    #
    #         if num_bi_layers == 1:
    #             encoder_state = bi_state
    #         else:
    #             encoder_state = []
    #             for layer_id in range(num_bi_layers):
    #                 encoder_state.append(bi_state[0][layer_id])  # forward
    #                 encoder_state.append(bi_state[1][layer_id])  # backward
    #
    #             encoder_state = tuple(encoder_state)
    #
    #         return encoder_outputs, encoder_state

    def create_decoder(self, hparams, encoder_outputs, encoder_state, model):
        with tf.compat.v1.variable_scope("decoder", reuse=tf.compat.v1.AUTO_REUSE) as decoder_scope:
            embed = self._decoder_token_embed if self.enable_oracle_dist else self.embeddings
            if model == "greedy":
                helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                    embed,
                    # Batchsize * Start_token
                    start_tokens=tf.fill([tf.size(self.decoder_full_length)], self.start_token),
                    end_token=self.end_token
                )

            elif model == "sample":
                helper = FixedSequenceLearningSampleEmbedingHelper(
                    sequence_length=self.decoder_full_length,
                    embedding=embed,
                    start_tokens=tf.fill([tf.size(self.decoder_full_length)], self.start_token),
                    end_token=self.end_token,
                    softmax_temperature=self.sample_softmax_temperature,
                    top_p=self.sample_top_p,
                )

            elif model == "train":
                helper = tf.contrib.seq2seq.TrainingHelper(
                    self.decoder_embeddings,
                    self.decoder_full_length,
                    time_major=self.time_major)
            else:
                helper = tf.contrib.seq2seq.TrainingHelper(
                    self.decoder_embeddings,
                    self.decoder_full_length,
                    time_major=self.time_major)

            if self.is_attention:
                decoder_cell = self._build_decoder_cell(hparams=hparams,
                                                        num_layers=self.num_layers,
                                                        num_residual_layers=self.num_residual_layers)
                # decoder_cell = tf.contrib.rnn.GRUCell(self.decoder_hidden_unit)
                if self.time_major:
                    # [batch_size, max_time, num_nunits]
                    attention_states = tf.transpose(encoder_outputs, [1, 0, 2])
                else:
                    attention_states = encoder_outputs

                attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                    self.decoder_hidden_unit, attention_states)

                decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
                    decoder_cell, attention_mechanism,
                    attention_layer_size=self.decoder_hidden_unit)

                film_state = self._film_encoder_state(encoder_state)
                decoder_initial_state = (
                    decoder_cell.zero_state(tf.size(self.decoder_full_length),
                                            dtype=tf.float32).clone(
                        cell_state=film_state))
            else:
                decoder_cell = self._build_decoder_cell(hparams=hparams,
                                                        num_layers=self.num_layers,
                                                        num_residual_layers=self.num_residual_layers)

                decoder_initial_state = self._film_encoder_state(encoder_state)

            decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=decoder_cell,
                helper=helper,
                initial_state=decoder_initial_state,
                output_layer=self.output_layer)

            outputs, last_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder,
                                                                       output_time_major=self.time_major,
                                                                       maximum_iterations=self.decoder_full_length[0])
        return outputs, last_state

    def get_variables(self):
        return tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, self.scope)


class Seq2SeqPolicy():
    def __init__(self, obs_dim, encoder_units,
                 decoder_units, vocab_size, name="pi", enable_cavia=False,
                 cavia_z_dim=32, enable_oracle_dist=False, oracle_n_dist=26,
                 oracle_z_dim=32, encoder_type="meanagg", readout_type="triple",
                 enable_eas_emb=False, cavia_film_trainable=False,
                 enable_context_encoder=False):
        self.decoder_targets = tf.compat.v1.placeholder(shape=[None, None], dtype=tf.int32, name="decoder_targets_ph_"+name)
        self.decoder_inputs = tf.compat.v1.placeholder(shape=[None, None], dtype=tf.int32, name="decoder_inputs_ph"+name)
        self.obs = tf.compat.v1.placeholder(shape=[None, None, obs_dim], dtype=tf.float32, name="obs_ph"+name)
        self.decoder_full_length = tf.compat.v1.placeholder(shape=[None], dtype=tf.int32, name="decoder_full_length"+name)
        self.encoder_type = str(encoder_type)
        self.readout_type = str(readout_type)
        self.reachability_mask = None
        if self.encoder_type == "dagformer":
            self.reachability_mask = tf.compat.v1.placeholder(
                shape=[None, None, None], dtype=tf.float32, name="reachability_mask_ph_" + name
            )
        self.enable_oracle_dist = bool(enable_oracle_dist)
        self.dist_ids = None
        if self.enable_oracle_dist:
            self.dist_ids = tf.compat.v1.placeholder(
                shape=[None], dtype=tf.int32, name="dist_ids_ph_" + name
            )
        self.enable_context_encoder = bool(enable_context_encoder)
        self.ctx_obs = None
        self.ctx_acts = None
        self.ctx_t = None
        self.ctx_zero = None
        if self.enable_context_encoder:
            self.ctx_obs = tf.compat.v1.placeholder(
                shape=[None, None, obs_dim], dtype=tf.float32, name="ctx_obs_ph_" + name
            )
            self.ctx_acts = tf.compat.v1.placeholder(
                shape=[None, None], dtype=tf.int32, name="ctx_acts_ph_" + name
            )
            self.ctx_t = tf.compat.v1.placeholder(
                shape=[None], dtype=tf.float32, name="ctx_t_ph_" + name
            )
            self.ctx_zero = tf.compat.v1.placeholder_with_default(
                False, shape=(), name="ctx_zero_ph_" + name
            )

        self.action_dim = vocab_size
        self.name = name

        hparams = tf.contrib.training.HParams(
            unit_type="lstm",
            encoder_units=encoder_units,
            decoder_units=decoder_units,

            n_features=vocab_size,
            time_major=False,
            is_attention=True,
            forget_bias=1.0,
            dropout=0,
            num_gpus=1,
            num_layers=2,
            num_residual_layers=0,
            start_token=0,
            # Sentinel outside the action set {0,...,vocab_size-1}.
            # Ternary: vocab=3, end=3. Binary: vocab=2, end=2.
            # GreedyEmbeddingHelper stops at end_token, so it must not be an action.
            end_token=int(vocab_size),
            is_bidencoder=False,
            enable_cavia=bool(enable_cavia) or bool(enable_context_encoder),
            cavia_z_dim=int(cavia_z_dim),
            cavia_film_trainable=bool(cavia_film_trainable) or bool(enable_context_encoder),
            enable_eas_emb=bool(enable_eas_emb),
            enable_oracle_dist=bool(enable_oracle_dist),
            oracle_n_dist=int(oracle_n_dist),
            oracle_z_dim=int(oracle_z_dim),
            encoder_type=str(encoder_type),
            readout_type=str(readout_type),
            enable_context_encoder=bool(enable_context_encoder),
        )

        self.network = Seq2SeqNetwork( hparams = hparams, reuse=tf.compat.v1.AUTO_REUSE,
                 encoder_inputs=self.obs,
                 decoder_inputs=self.decoder_inputs,
                 decoder_full_length=self.decoder_full_length,
                 decoder_targets=self.decoder_targets,name = name,
                 dist_ids=self.dist_ids,
                 reachability_mask=self.reachability_mask,
                 ctx_obs=self.ctx_obs,
                 ctx_acts=self.ctx_acts,
                 ctx_t=self.ctx_t,
                 ctx_zero=self.ctx_zero)

        self.vf = self.network.vf

        self._dist = CategoricalPd(vocab_size)

    def get_actions(self, observations):
        sess = tf.compat.v1.get_default_session()

        decoder_full_length = np.array( [observations.shape[1]] * observations.shape[0] , dtype=np.int32)

        actions, logits, v_value = sess.run([self.network.sample_decoder_prediction,
                                             self.network.sample_decoder_logits,
                                             self.network.sample_vf],
                                            feed_dict={self.obs: observations, self.decoder_full_length: decoder_full_length})

        return actions, logits, v_value

    @property
    def distribution(self):
        return self._dist

    def get_variables(self):
        return self.network.get_variables()

    def get_trainable_variables(self):
        return self.network.get_trainable_variables()

    def cavia_trainable_variables(self):
        if self.network.cavia_z is None:
            return []
        return [self.network.cavia_z]

    def reset_cavia_z(self, sess=None):
        if self.network.cavia_z is None:
            raise ValueError("reset_cavia_z requires enable_cavia=True")
        sess = sess or tf.compat.v1.get_default_session()
        sess.run(self.network.cavia_z.initializer)

    def save_variables(self, save_path, sess=None):
        sess = sess or tf.compat.v1.get_default_session()
        variables = self.get_variables()

        ps = sess.run(variables)
        save_dict = {v.name: value for v, value in zip(variables, ps)}

        dirname = os.path.dirname(save_path)
        if any(dirname):
            os.makedirs(dirname, exist_ok=True)

        joblib.dump(save_dict, save_path)

    def load_variables(self, load_path, sess=None):
        sess = sess or tf.compat.v1.get_default_session()
        variables = self.get_variables()

        loaded_params = joblib.load(os.path.expanduser(load_path))
        restores = []

        if isinstance(loaded_params, list):
            assert len(loaded_params) == len(variables), 'number of variables loaded mismatches len(variables)'
            for d, v in zip(loaded_params, variables):
                restores.append(v.assign(d))
        else:
            for v in variables:
                restores.append(v.assign(loaded_params[v.name]))

        sess.run(restores)


class MetaSeq2SeqPolicy():
    def __init__(self, meta_batch_size, obs_dim, encoder_units, decoder_units,
                 vocab_size, encoder_type="meanagg", readout_type="triple"):

        self.meta_batch_size = meta_batch_size
        self.obs_dim = obs_dim
        self.action_dim = vocab_size
        self.encoder_type = str(encoder_type)
        self.readout_type = str(readout_type)

        self.core_policy = Seq2SeqPolicy(
            obs_dim, encoder_units, decoder_units, vocab_size, name='core_policy',
            encoder_type=self.encoder_type, readout_type=self.readout_type,
        )


        self.meta_policies = []

        self.assign_old_eq_new_tasks = []

        for i in range(meta_batch_size):
            self.meta_policies.append(Seq2SeqPolicy(
                obs_dim, encoder_units, decoder_units,
                vocab_size, name="task_"+str(i)+"_policy",
                encoder_type=self.encoder_type, readout_type=self.readout_type,
            ))

            self.assign_old_eq_new_tasks.append(
                U.function([], [], updates=[tf.compat.v1.assign(oldv, newv)
                                            for (oldv, newv) in
                                            zipsame(self.meta_policies[i].get_variables(), self.core_policy.get_variables())])
                )

        self._dist = CategoricalPd(vocab_size)


    def get_actions(self, observations):
        assert len(observations) == self.meta_batch_size

        meta_actions = []
        meta_logits = []
        meta_v_values = []
        for i, obser_per_task in enumerate(observations):
            action, logits, v_value = self.meta_policies[i].get_actions(obser_per_task)

            meta_actions.append(np.array(action))
            meta_logits.append(np.array(logits))
            meta_v_values.append(np.array(v_value))

        return meta_actions, meta_logits, meta_v_values

    def async_parameters(self):
        # async_parameters.
        for i in range(self.meta_batch_size):
            self.assign_old_eq_new_tasks[i]()

    @property
    def distribution(self):
        return self._dist

