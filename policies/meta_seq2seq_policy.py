import os
import joblib

import numpy as np
import tensorflow as tf
import policies.model_helper as model_helper
from policies.graph2seq_encoder import create_graph2seq_encoder
from policies.graph2seq_encoder_v2 import create_improved_graph2seq_encoder

from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops.distributions import categorical
from policies.distributions.categorical_pd import CategoricalPd
import utils as U
from utils.utils import zipsame

tf.get_logger().setLevel('WARNING')

class FixedSequenceLearningSampleEmbedingHelper(tf.contrib.seq2seq.SampleEmbeddingHelper):
    def __init__(self, sequence_length, embedding, start_tokens, end_token, softmax_temperature=None, seed=None):
        super(FixedSequenceLearningSampleEmbedingHelper, self).__init__(
            embedding, start_tokens, end_token, softmax_temperature, seed
        )
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

        sample_id_sampler = categorical.Categorical(logits=logits)
        sample_ids = sample_id_sampler.sample(seed=self._seed)

        return sample_ids

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
                 feature_mode='full17'):
        self.encoder_hidden_unit = hparams.encoder_units
        self.decoder_hidden_unit = hparams.decoder_units
        self.is_bidencoder = hparams.is_bidencoder
        self.reuse = reuse

        self.n_features = hparams.n_features
        self.time_major = hparams.time_major
        self.is_attention = hparams.is_attention
        self.feature_mode = feature_mode

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

        with tf.compat.v1.variable_scope(name, reuse=self.reuse, initializer=tf.glorot_normal_initializer()):
            self.scope = tf.compat.v1.get_variable_scope().name
            self.embeddings = tf.Variable(tf.random.uniform(
                [self.n_features,
                 self.encoder_hidden_unit],
                -1.0, 1.0), dtype=tf.float32)

            # Assert correct input dimensions based on feature mode
            expected_input_dim = 13 if self.feature_mode == 'core5' else 17
            input_shape = tf.shape(self.encoder_inputs)
            actual_input_dim = self.encoder_inputs.get_shape()[-1]
            
            # Add assertion for input dimension
            dimension_assertion = tf.debugging.assert_equal(
                actual_input_dim, expected_input_dim,
                message=f"Expected input dimension {expected_input_dim} for {self.feature_mode} mode, but got {actual_input_dim}"
            )
            with tf.control_dependencies([dimension_assertion]):
                self.encoder_embeddings = tf.contrib.layers.fully_connected(self.encoder_inputs,
                                                                            self.encoder_hidden_unit,
                                                                            activation_fn = None,
                                                                            scope="encoder_embeddings",
                                                                            reuse=tf.compat.v1.AUTO_REUSE)

            self.decoder_embeddings = tf.nn.embedding_lookup(self.embeddings,
                                                             self.decoder_inputs)

            self.decoder_targets_embeddings = tf.one_hot(self.decoder_targets,
                                                         self.n_features,
                                                         dtype=tf.float32)

            self.output_layer = tf.compat.v1.layers.Dense(self.n_features, use_bias=False, name="output_projection")

            # Use improved Graph2Seq encoder with gated convolution
            self.encoder_outputs, self.encoder_state = create_improved_graph2seq_encoder(
                encoder_inputs=self.encoder_embeddings,
                encoder_units=self.encoder_hidden_unit,
                num_layers=self.num_layers,
                is_bidirectional=self.is_bidencoder,
                mode=self.mode,
                feature_mode=feature_mode,
                use_virtual_node=True,
                scope_name="encoder"
            )

            # training decoder
            self.decoder_outputs, self.decoder_state = self.create_decoder(hparams, self.encoder_outputs,
                                                                           self.encoder_state, model="train")
            self.decoder_logits = self.decoder_outputs.rnn_output
            self.pi = tf.nn.softmax(self.decoder_logits)
            self.q = tf.compat.v1.layers.dense(self.decoder_logits, self.n_features, activation=None,
                                     reuse=tf.compat.v1.AUTO_REUSE, name="qvalue_layer")
            self.vf = tf.reduce_sum(self.pi * self.q, axis=-1)

            self.decoder_prediction = self.decoder_outputs.sample_id

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
        # x_shape = x.get_shape()
        # logits_shape = self.logits.get_shape()
        # x = tf.one_hot(x, logits_shape[2])
        # Sparse to dense
        return tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=self.decoder_logits,
            labels=self.decoder_targets_embeddings)

    def _build_decoder_cell(self, hparams, num_layers, num_residual_layers, base_gpu=0):

        """Build an RNN cell that can be used by decoder."""
        # We only make use of encoder_unit_type in the decoder
        # unit_type = hparams.encoder_unit_type
        num_units = self.decoder_hidden_unit
        dropout = hparams.dropout

        # Cell Type
        # if unit_type == "lstm":
        single_cell = tf.contrib.rnn.BasicLSTMCell(num_units, forget_bias=hparams.forget_bias)
        # elif unit_type == "gru":
        #    single_cell = tf.contrib.rnn.GRUCell(num_units)
        # elif unit_type == "layer_norm_lstm":
        #    single_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(num_units, forget_bias=hparams.forget_bias,
        # 							    layer_norm=True)
        # elif unit_type == "nas":
        # 	single_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(num_units)
        # else:
        # 	raise ValueError("Unknown unit type %s!" % unit_type)

        cell_list = model_helper._cell_list(unit_type=hparams.unit_type,
                                            num_units=num_units,
                                            num_layers=num_layers,
                                            num_residual_layers=num_residual_layers,
                                            forget_bias=hparams.forget_bias,
                                            dropout=dropout,
                                            num_gpus=hparams.num_gpus,
                                            mode=self.mode,
                                            base_gpu=base_gpu,
                                            single_cell_fn=None
                                            )

        if len(cell_list) == 1:
            return cell_list[0]
        else:
            return tf.contrib.rnn.MultiRNNCell(cell_list)

    def create_decoder(self, hparams, encoder_outputs, encoder_state, model):
        with tf.compat.v1.variable_scope("decoder", reuse=tf.compat.v1.AUTO_REUSE) as decoder_scope:
            if model == "greedy":
                helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                    self.embeddings,
                    # Batchsize * Start_token
                    start_tokens=tf.fill([tf.size(self.decoder_full_length)], self.start_token),
                    end_token=self.end_token
                )

            elif model == "sample":
                helper = FixedSequenceLearningSampleEmbedingHelper(
                    sequence_length=self.decoder_full_length,
                    embedding=self.embeddings,
                    start_tokens=tf.fill([tf.size(self.decoder_full_length)], self.start_token),
                    end_token=self.end_token
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

                decoder_initial_state = (
                    decoder_cell.zero_state(tf.size(self.decoder_full_length),
                                            dtype=tf.float32).clone(
                        cell_state=encoder_state))
            else:
                decoder_cell = self._build_decoder_cell(hparams=hparams,
                                                        num_layers=self.num_layers,
                                                        num_residual_layers=self.num_residual_layers)

                decoder_initial_state = encoder_state

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
                 decoder_units, vocab_size, feature_mode='full17', name="pi"):
        self.decoder_targets = tf.compat.v1.placeholder(shape=[None, None], dtype=tf.int32, name="decoder_targets_ph_"+name)
        self.decoder_inputs = tf.compat.v1.placeholder(shape=[None, None], dtype=tf.int32, name="decoder_inputs_ph"+name)
        
        # Store feature_mode before using it
        self.feature_mode = feature_mode
        self.obs_dim = obs_dim
        self.action_dim = vocab_size
        self.name = name
        
        # Determine the input dimension based on feature mode
        # For the new 72dim system, we receive 5-dim raw features that get transformed
        if feature_mode == 'core5':
            input_raw_dim = 5  # New format: [task_index, local_cost, up_cost, mec_cost, down_cost]
        else:
            input_raw_dim = 17  # Legacy format: full 17-dimensional features
            
        self.obs_raw = tf.compat.v1.placeholder(shape=[None, None, input_raw_dim], dtype=tf.float32, name="obs_raw_ph"+name)
        
        # Feature transformation based on mode
        if feature_mode == 'core5':
            # Extract core features and add embeddings
            self.obs = self._build_core5_features(self.obs_raw, name)
        else:
            # Use full 17-dimensional features
            self.obs = self.obs_raw
            
        self.decoder_full_length = tf.compat.v1.placeholder(shape=[None], dtype=tf.int32, name="decoder_full_length"+name)

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
            end_token=2,
            is_bidencoder=False
        )

        self.network = Seq2SeqNetwork( hparams = hparams, reuse=tf.compat.v1.AUTO_REUSE,
                 encoder_inputs=self.obs,
                 decoder_inputs=self.decoder_inputs,
                 decoder_full_length=self.decoder_full_length,
                 decoder_targets=self.decoder_targets,
                 feature_mode=self.feature_mode,
                 name = name)

        self.vf = self.network.vf
        self._dist = CategoricalPd(vocab_size)
                 
    def _build_core5_features(self, obs_raw, name):
        """
        Transform 5-dim raw features to 13-dim features for core5 mode:
        - Input: [task_index, local_cost, up_cost, mec_cost, down_cost] (5 dims)
        - Add depth normalization (1 scalar) - computed from task_index
        - Add 8-dim task embedding
        - Output: 13 dimensions total
        """
        with tf.compat.v1.variable_scope(name + "_core5_transform", reuse=tf.compat.v1.AUTO_REUSE):
            # Input is already the 5 core scalars: [task_index, local_cost, up_cost, mec_cost, down_cost]
            task_index = obs_raw[:, :, 0:1]  # Keep dims
            core_scalars = obs_raw[:, :, 1:5]  # local, up, mec, down costs
            
            # For depth calculation in the new system, we'll use the task_index position
            # as a proxy for depth (tasks earlier in sequence tend to have fewer dependencies)
            # This is a simplification since we don't have the predecessor info in the 5-dim format
            batch_size = tf.shape(obs_raw)[0]
            seq_len = tf.shape(obs_raw)[1]
            
            # For TensorFlow 1.x compatibility, use a simpler approach for depth proxy
            # Since we know the max sequence length is typically 20 for this problem,
            # we'll create a fixed-size range and slice it dynamically
            max_seq_len = 50  # Conservative upper bound
            fixed_range = tf.cast(tf.range(max_seq_len), tf.float32)
            
            # Create position indices for each batch
            position_indices = tf.tile(tf.expand_dims(fixed_range, 0), [batch_size, 1])
            # Slice to actual sequence length
            position_indices = position_indices[:, :seq_len]
            position_indices = tf.expand_dims(position_indices, -1)
            
            # Normalize position to [0, 1] as depth proxy
            seq_len_float = tf.cast(seq_len, tf.float32)
            depth_norm = position_indices / tf.maximum(seq_len_float - 1.0, 1.0)
            
            # Create task embedding
            # Use task_index as integer for embedding lookup
            task_idx_int = tf.cast(task_index, tf.int32)
            task_idx_int = tf.squeeze(task_idx_int, axis=-1)  # Remove last dim for embedding lookup
            
            # Create embedding layer (20 tasks, 8-dim embedding)
            embedding_table = tf.compat.v1.get_variable(
                name="task_embedding",
                shape=[20, 8],
                initializer=tf.glorot_uniform_initializer(),
                dtype=tf.float32
            )
            
            # Lookup embeddings
            task_embeddings = tf.nn.embedding_lookup(embedding_table, task_idx_int)
            
            # Concatenate all features: [core_scalars(4), depth_norm(1), task_embedding(8)] = 13 dims
            core5_features = tf.concat([core_scalars, depth_norm, task_embeddings], axis=-1)
            
            # Assert shape is correct
            shape_assertion = tf.debugging.assert_equal(
                tf.shape(core5_features)[-1], 13,
                message="Core5 features should have 13 dimensions"
            )
            # Mark assertion as used to avoid TensorFlow warnings
            with tf.control_dependencies([shape_assertion]):
                core5_features = tf.identity(core5_features)
            
            return core5_features

    def get_actions(self, observations):
        sess = tf.compat.v1.get_default_session()

        decoder_full_length = np.array( [observations.shape[1]] * observations.shape[0] , dtype=np.int32)

        actions, logits, v_value = sess.run([self.network.sample_decoder_prediction,
                                             self.network.sample_decoder_logits,
                                             self.network.sample_vf],
                                            feed_dict={self.obs_raw: observations, self.decoder_full_length: decoder_full_length})

        return actions, logits, v_value

    @property
    def distribution(self):
        return self._dist

    def get_variables(self):
        return self.network.get_variables()

    def get_trainable_variables(self):
        return self.network.get_trainable_variables()

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
                 vocab_size, feature_mode='full17'):

        self.meta_batch_size = meta_batch_size
        self.obs_dim = obs_dim
        self.action_dim = vocab_size
        self.feature_mode = feature_mode

        self.core_policy = Seq2SeqPolicy(obs_dim, encoder_units, decoder_units, vocab_size, 
                                        feature_mode=feature_mode, name='core_policy')


        self.meta_policies = []

        self.assign_old_eq_new_tasks = []

        for i in range(meta_batch_size):
            self.meta_policies.append(Seq2SeqPolicy(obs_dim, encoder_units, decoder_units,
                                                    vocab_size, feature_mode=feature_mode, 
                                                    name="task_"+str(i)+"_policy"))

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
        for i in range(self.meta_batch_size):
            self.assign_old_eq_new_tasks[i]()

    @property
    def distribution(self):
        return self._dist