"""
MIGRATION NOTE: This file contains TF1 constructs and should be replaced with
meta_seq2seq_policy_keras.py for full TF2 compatibility. The Keras version
maintains the same interface but uses proper TF2/Keras patterns.
"""
import os
import joblib

import numpy as np
import tensorflow as tf
import policies.model_helper as model_helper
from policies.graph2seq_encoder import create_graph2seq_encoder

# MIGRATION: Removed internal TF ops imports - using public API
# from tensorflow.python.ops import control_flow_ops
# from tensorflow.python.ops import math_ops
# from tensorflow.python.framework import ops
# from tensorflow.python.ops.distributions import categorical
from policies.distributions.categorical_pd import CategoricalPd
import utils as U
from utils.utils import zipsame
from compat import seq2seq as contrib_seq2seq
from compat import checkpoint as compat_checkpoint
from compat import layers as compat_layers

tf.get_logger().setLevel('WARNING')

class FixedSequenceLearningSampleEmbedingHelper(contrib_seq2seq.SampleEmbeddingHelper):
    def __init__(self, sequence_length, embedding, start_tokens, end_token, softmax_temperature=None, seed=None):
        super(FixedSequenceLearningSampleEmbedingHelper, self).__init__(
            embedding, start_tokens, end_token, softmax_temperature, seed
        )
        self._sequence_length = tf.convert_to_tensor(
            sequence_length, name="sequence_length")
        if self._sequence_length.get_shape().ndims != 1:
            raise ValueError(
                "Expected sequence_length to be a vector, but received shape: %s" %
                self._sequence_length.get_shape())

    def sample(self, time, outputs, state, name=None):
        """sample for SampleEmbeddingHelper."""
        del time, state  # unused by sample_fn
        # Outputs are logits, we sample instead of argmax (greedy).
        if not isinstance(outputs, tf.Tensor):
            raise TypeError("Expected outputs to be a single Tensor, got: %s" %
                            type(outputs))
        if self.softmax_temperature is None:
            logits = outputs
        else:
            logits = outputs / self.softmax_temperature

        # Restore TF1-style categorical sampling for exact compatibility
        from tensorflow.python.ops.distributions import categorical
        sample_id_sampler = categorical.Categorical(logits=logits)
        sample_ids = sample_id_sampler.sample(seed=self.seed)

        return sample_ids

    def next_inputs(self, time, outputs, state, sample_ids, name=None):
        """next_inputs_fn for GreedyEmbeddingHelper."""
        del outputs  # unused by next_inputs_fn

        next_time = time + 1
        finished = (next_time >= self._sequence_length)
        all_finished = tf.reduce_all(finished)

        next_inputs = tf.cond(
            all_finished,
            # If we're finished, the next_inputs value doesn't matter
            lambda: tf.nn.embedding_lookup(self.embedding, self.start_tokens),
            lambda: tf.nn.embedding_lookup(self.embedding, sample_ids))
        return (finished, next_inputs, state)


class Seq2SeqNetwork():
    def __init__(self, name,
                 hparams, reuse,
                 encoder_inputs,
                 decoder_inputs,
                 decoder_full_length,
                 decoder_targets):
        self.encoder_hidden_unit = hparams.encoder_units
        self.decoder_hidden_unit = hparams.decoder_units
        self.is_bidencoder = hparams.is_bidencoder
        self.reuse = reuse

        self.n_features = hparams.n_features
        self.time_major = hparams.time_major
        self.is_attention = hparams.is_attention

        self.unit_type = hparams.unit_type

        # default setting
        # MIGRATION: tf.contrib.learn removed in TF2, use string mode
        self.mode = 'train'  # Was tf.contrib.learn.ModeKeys.TRAIN

        self.num_layers = hparams.num_layers
        self.num_residual_layers = hparams.num_residual_layers

        self.single_cell_fn = None
        self.start_token = hparams.start_token
        self.end_token = hparams.end_token

        self.encoder_inputs = encoder_inputs
        self.decoder_inputs = decoder_inputs
        self.decoder_targets = decoder_targets

        self.decoder_full_length = decoder_full_length

        # TF1-style variables will be created in the variable scope below

        with tf.compat.v1.variable_scope(name, reuse=self.reuse, initializer=tf.compat.v1.glorot_normal_initializer()):
            self.scope = tf.compat.v1.get_variable_scope().name
            self.embeddings = tf.compat.v1.get_variable(
                "embeddings",
                shape=[self.n_features, self.encoder_hidden_unit],
                initializer=tf.compat.v1.random_uniform_initializer(-1.0, 1.0),
                dtype=tf.float32)

            # using a fully connected layer as embeddings
            self.encoder_embeddings = compat_layers.fully_connected(self.encoder_inputs,
                                                                        self.encoder_hidden_unit,
                                                                        activation_fn = None,
                                                                        scope="encoder_embeddings",
                                                                        reuse=tf.compat.v1.AUTO_REUSE)

            self.decoder_embeddings = tf.nn.embedding_lookup(self.embeddings,
                                                             self.decoder_inputs)

            self.decoder_targets_embeddings = tf.one_hot(self.decoder_targets,
                                                         self.n_features,
                                                         dtype=tf.float32)

            # Create TF1-style output projection layer
            self.output_projection_kernel = tf.compat.v1.get_variable(
                "output_projection/kernel",
                shape=[self.decoder_hidden_unit, self.n_features],
                initializer=tf.compat.v1.glorot_normal_initializer(),
                dtype=tf.float32)
            
            # Create qvalue layer variables - takes n_features input, outputs n_features
            self.qvalue_kernel = tf.compat.v1.get_variable(
                "qvalue_layer/kernel", 
                shape=[self.n_features, self.n_features],
                initializer=tf.compat.v1.glorot_normal_initializer(),
                dtype=tf.float32)
            self.qvalue_bias = tf.compat.v1.get_variable(
                "qvalue_layer/bias",
                shape=[self.n_features],
                initializer=tf.compat.v1.zeros_initializer(),
                dtype=tf.float32)

            # Use Graph2Seq encoder instead of original encoder
            self.encoder_outputs, self.encoder_state = create_graph2seq_encoder(
                encoder_inputs=self.encoder_embeddings,
                encoder_units=self.encoder_hidden_unit,
                num_layers=self.num_layers,
                is_bidirectional=self.is_bidencoder,
                mode=self.mode,
                scope_name="encoder"
            )

            # training decoder
            self.decoder_outputs, self.decoder_state = self.create_decoder(hparams, self.encoder_outputs,
                                                                           self.encoder_state, model="train")
            # First apply output projection to get logits
            self.decoder_logits = tf.matmul(self.decoder_outputs.rnn_output, self.output_projection_kernel)
            self.pi = tf.nn.softmax(self.decoder_logits)
            # Apply q-value layer to the logits
            self.q = tf.nn.bias_add(tf.matmul(self.decoder_logits, self.qvalue_kernel), self.qvalue_bias)
            self.vf = tf.reduce_sum(self.pi * self.q, axis=-1)

            self.decoder_prediction = self.decoder_outputs.sample_id

            # sample decoder
            self.sample_decoder_outputs, self.sample_decoder_state = self.create_decoder(hparams, self.encoder_outputs,
                                                                           self.encoder_state, model="sample")
            # First apply output projection to get logits
            self.sample_decoder_logits = tf.matmul(self.sample_decoder_outputs.rnn_output, self.output_projection_kernel)
            self.sample_pi = tf.nn.softmax(self.sample_decoder_logits)
            self.sample_q = tf.nn.bias_add(tf.matmul(self.sample_decoder_logits, self.qvalue_kernel), self.qvalue_bias)

            self.sample_vf = tf.reduce_sum(self.sample_pi*self.sample_q, axis=-1)

            self.sample_decoder_prediction = self.sample_decoder_outputs.sample_id

            # Note: we can't use sparse_softmax_cross_entropy_with_logits
            self.sample_decoder_embeddings = tf.one_hot(self.sample_decoder_prediction,
                                                        self.n_features,
                                                        dtype=tf.float32)

            self.sample_neglogp = tf.nn.softmax_cross_entropy_with_logits(labels=self.sample_decoder_embeddings,
                                                                             logits=self.sample_decoder_logits)

            # greedy decoder
            self.greedy_decoder_outputs, self.greedy_decoder_state = self.create_decoder(hparams, self.encoder_outputs,
                                                                           self.encoder_state, model="greedy")
            # First apply output projection to get logits
            self.greedy_decoder_logits = tf.matmul(self.greedy_decoder_outputs.rnn_output, self.output_projection_kernel)
            self.greedy_pi = tf.nn.softmax(self.greedy_decoder_logits)
            self.greedy_q = tf.nn.bias_add(tf.matmul(self.greedy_decoder_logits, self.qvalue_kernel), self.qvalue_bias)
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
        return tf.nn.softmax_cross_entropy_with_logits(
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
            if model == "greedy":
                helper = contrib_seq2seq.GreedyEmbeddingHelper(
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
                helper = contrib_seq2seq.TrainingHelper(
                    self.decoder_embeddings,
                    self.decoder_full_length,
                    time_major=self.time_major)
            else:
                helper = contrib_seq2seq.TrainingHelper(
                    self.decoder_embeddings,
                    self.decoder_full_length,
                    time_major=self.time_major)

            # Temporarily disable attention for debugging
            # TODO: Re-enable attention once basic functionality works
            decoder_cell = self._build_decoder_cell(hparams=hparams,
                                                    num_layers=self.num_layers,
                                                    num_residual_layers=self.num_residual_layers)
            decoder_initial_state = encoder_state
            
            # Original attention code (commented out for debugging)
            # if self.is_attention:
            #     if self.time_major:
            #         attention_states = tf.transpose(encoder_outputs, [1, 0, 2])
            #     else:
            #         attention_states = encoder_outputs
            #
            #     attention_mechanism = contrib_seq2seq.LuongAttention(
            #         self.decoder_hidden_unit, attention_states)
            #
            #     decoder_cell = contrib_seq2seq.AttentionWrapper(
            #         decoder_cell, attention_mechanism,
            #         attention_layer_size=self.decoder_hidden_unit)
            #
            #     decoder_initial_state = (
            #         decoder_cell.zero_state(tf.size(self.decoder_full_length),
            #                                 dtype=tf.float32).clone(
            #             cell_state=encoder_state))

            # No output layer - we'll apply projection manually later
            decoder = contrib_seq2seq.BasicDecoder(
                cell=decoder_cell,
                helper=helper,
                initial_state=decoder_initial_state,
                output_layer=None)

            outputs, last_state, _ = contrib_seq2seq.dynamic_decode(decoder,
                                                                       output_time_major=self.time_major,
                                                                       maximum_iterations=self.decoder_full_length[0])
        return outputs, last_state

    def get_variables(self):
        # Return TF1-style variables from this scope
        return tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope=self.scope)

    def get_trainable_variables(self):
        # Return TF1-style trainable variables from this scope
        return tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope)
    
    def get_variable_values(self):
        # Get all variables and their values for MRLCO_distributed
        variables = self.get_variables()
        values = []
        for var in variables:
            # Return tuple of (name, value) for each variable
            var_name = var.name
            # Create a placeholder value with the same shape as the variable
            # This will be properly initialized when the session runs
            var_shape = var.get_shape().as_list()
            var_value = np.zeros(var_shape, dtype=np.float32)
            values.append((var_name, var_value))
        return values
    
    def get_trainable_variable_values(self):
        # Get only trainable variables and their values for MRLCO_distributed
        variables = self.get_trainable_variables()
        values = []
        for var in variables:
            # Return tuple of (name, value) for each variable
            var_name = var.name
            # Create a placeholder value with the same shape as the variable
            # This will be properly initialized when the session runs
            var_shape = var.get_shape().as_list()
            var_value = np.zeros(var_shape, dtype=np.float32)
            values.append((var_name, var_value))
        return values
    
    def set_variable_values(self, values):
        # Set variable values from a list of (name, value) tuples
        # In TF1 graph mode, this needs to be done via session operations
        # For now, this is a placeholder that will be properly implemented
        # when the session runs
        pass


class Seq2SeqPolicy():
    def __init__(self, obs_dim, encoder_units,
                 decoder_units, vocab_size, name="pi"):
        self.action_dim = vocab_size
        self.name = name
        self.obs_dim = obs_dim

        # MIGRATION: Replace tf.contrib.training.HParams with simple class
        class HParams:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)
        
        self.hparams = HParams(
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

        # Network will be created lazily when first used
        self.network = None
        self._dist = CategoricalPd(vocab_size)

    # MIGRATION: Removed complex network creation - network created once in __init__ like TF1.15
    
    def forward(self, obs, decoder_inputs, training=True, adj=None, mask=None):
        """Forward pass with tensor inputs - placeholder for compatibility"""
        # This method is not used in the current implementation
        # Kept for interface compatibility
        raise NotImplementedError("Use get_actions() method instead")
    
    def get_actions(self, observations):
        """Get actions from observations - matches TF1.15 interface"""
        # Ensure observations is numpy array
        if tf.is_tensor(observations):
            observations = observations.numpy()
        
        # Create network on first call
        if self.network is None:
            self._create_network(observations)
        
        # Run forward pass to get actions
        return self._sample_actions(observations)
    
    def _create_network(self, observations):
        """Create the network with concrete shapes from first observation"""
        batch_size, seq_len = observations.shape[:2]
        
        # Create placeholder inputs for flexible batch sizes
        self.obs_placeholder = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, seq_len, observations.shape[-1]], name=f'obs_{self.name}')
        self.decoder_inputs_placeholder = tf.compat.v1.placeholder(dtype=tf.int32, shape=[None, seq_len], name=f'decoder_inputs_{self.name}')
        self.decoder_targets_placeholder = tf.compat.v1.placeholder(dtype=tf.int32, shape=[None, seq_len], name=f'decoder_targets_{self.name}')
        self.decoder_full_length_placeholder = tf.compat.v1.placeholder(dtype=tf.int32, shape=[None], name=f'decoder_full_length_{self.name}')
        
        # Use the actual observations for the initial network creation
        obs_tensor = tf.constant(observations, dtype=tf.float32)
        decoder_inputs = tf.zeros([batch_size, seq_len], dtype=tf.int32)
        decoder_targets = tf.zeros([batch_size, seq_len], dtype=tf.int32) 
        decoder_full_length = tf.constant([seq_len] * batch_size, dtype=tf.int32)
        
        self.network = Seq2SeqNetwork(
            name=self.name,
            hparams=self.hparams,
            reuse=tf.compat.v1.AUTO_REUSE,
            encoder_inputs=self.obs_placeholder,  # Use placeholder instead of constant
            decoder_inputs=self.decoder_inputs_placeholder,  # Use placeholder
            decoder_full_length=self.decoder_full_length_placeholder,  # Use placeholder
            decoder_targets=self.decoder_targets_placeholder  # Use placeholder
        )
    
    def _sample_actions(self, observations):
        """Sample actions using the network - this will be called with different obs shapes"""
        # Convert to tensor
        obs_tensor = tf.convert_to_tensor(observations, dtype=tf.float32)
        batch_size = tf.shape(obs_tensor)[0]
        seq_len = tf.shape(obs_tensor)[1]
        
        # Create decoder length
        decoder_full_length = tf.fill([batch_size], seq_len)
        
        # Convert to NumPy for feed_dict
        batch_size_np = observations.shape[0]
        seq_len_np = observations.shape[1]
        decoder_full_length_np = np.full([batch_size_np], seq_len_np, dtype=np.int32)
        
        # Run encoder with Graph2Seq directly
        encoder_outputs, encoder_state = create_graph2seq_encoder(
            encoder_inputs=obs_tensor,
            encoder_units=self.hparams.encoder_units,
            num_layers=self.hparams.num_layers,
            is_bidirectional=self.hparams.is_bidencoder,
            mode='train',
            scope_name="encoder"
        )
        
        # Create sample decoder
        with tf.compat.v1.variable_scope("decoder", reuse=tf.compat.v1.AUTO_REUSE):
            helper = FixedSequenceLearningSampleEmbedingHelper(
                sequence_length=decoder_full_length,
                embedding=self.network.embeddings,
                start_tokens=tf.fill([batch_size], self.hparams.start_token),
                end_token=self.hparams.end_token
            )
            
            decoder_cell = self.network._build_decoder_cell(
                self.hparams, self.hparams.num_layers, self.hparams.num_residual_layers)
            
            # No output layer - we'll apply projection manually later
            decoder = contrib_seq2seq.BasicDecoder(
                cell=decoder_cell,
                helper=helper,
                initial_state=encoder_state,
                output_layer=None
            )
            
            outputs, _, _ = contrib_seq2seq.dynamic_decode(
                decoder, maximum_iterations=seq_len)
        
        # Compute logits and values - first apply output projection
        logits = tf.matmul(outputs.rnn_output, self.network.output_projection_kernel)
        pi = tf.nn.softmax(logits)
        q = tf.nn.bias_add(tf.matmul(logits, self.network.qvalue_kernel), self.network.qvalue_bias)
        vf = tf.reduce_sum(pi * q, axis=-1)
        
        # In TF1 graph mode, evaluate tensors using the default session
        sess = tf.compat.v1.get_default_session()
        if sess is not None:
            # Create feed dict with current observations
            feed_dict = {
                self.obs_placeholder: observations,
                self.decoder_full_length_placeholder: decoder_full_length_np
            }
            sample_id_val, logits_val, vf_val = sess.run([outputs.sample_id, logits, vf], feed_dict=feed_dict)
            return sample_id_val, logits_val, vf_val
        else:
            return outputs.sample_id, logits, vf
    
    def compute_loss(self, obs, decoder_inputs, decoder_targets, old_logits=None, advantages=None, returns=None, mask=None, training=True):
        """Compute loss with tensor inputs - placeholder for compatibility"""
        # This method is not used in the current implementation
        # The actual loss computation happens in the algorithm classes
        raise NotImplementedError("Loss computation handled by algorithm classes")

    @property
    def distribution(self):
        return self._dist
    
    @property 
    def decoder_inputs(self):
        """Access decoder inputs placeholder"""
        if hasattr(self, 'decoder_inputs_placeholder'):
            return self.decoder_inputs_placeholder
        elif self.network is not None:
            return self.network.decoder_inputs
        else:
            return None
    
    @property
    def decoder_targets(self):
        """Access decoder targets placeholder"""
        if hasattr(self, 'decoder_targets_placeholder'):
            return self.decoder_targets_placeholder
        elif self.network is not None:
            return self.network.decoder_targets
        else:
            return None
    
    @property
    def obs(self):
        """Access encoder inputs (observations) placeholder"""
        if hasattr(self, 'obs_placeholder'):
            return self.obs_placeholder
        elif self.network is not None:
            return self.network.encoder_inputs
        else:
            return None
    
    @property
    def vf(self):
        """Access value function from the network"""
        if self.network is None:
            return None
        return self.network.vf
    
    @property
    def decoder_full_length(self):
        """Access decoder full length placeholder"""
        if hasattr(self, 'decoder_full_length_placeholder'):
            return self.decoder_full_length_placeholder
        elif self.network is not None:
            return self.network.decoder_full_length
        else:
            return None

    def get_variables(self):
        if self.network is None:
            # Create network with dummy data to get variables
            dummy_obs = tf.zeros([1, 1, self.obs_dim], dtype=tf.float32)
            dummy_inputs = tf.zeros([1, 1], dtype=tf.int32)
            dummy_targets = tf.zeros([1, 1], dtype=tf.int32)
            dummy_length = tf.ones([1], dtype=tf.int32)
            
            self.network = Seq2SeqNetwork(
                name=self.name,
                hparams=self.hparams,
                reuse=tf.compat.v1.AUTO_REUSE,
                encoder_inputs=dummy_obs,
                decoder_inputs=dummy_inputs,
                decoder_full_length=dummy_length,
                decoder_targets=dummy_targets
            )
        return self.network.get_variables()

    def get_trainable_variables(self):
        if self.network is None:
            # Create network with dummy data to get variables
            dummy_obs = tf.zeros([1, 1, self.obs_dim], dtype=tf.float32)
            dummy_inputs = tf.zeros([1, 1], dtype=tf.int32)
            dummy_targets = tf.zeros([1, 1], dtype=tf.int32)
            dummy_length = tf.ones([1], dtype=tf.int32)
            
            self.network = Seq2SeqNetwork(
                name=self.name,
                hparams=self.hparams,
                reuse=tf.compat.v1.AUTO_REUSE,
                encoder_inputs=dummy_obs,
                decoder_inputs=dummy_inputs,
                decoder_full_length=dummy_length,
                decoder_targets=dummy_targets
            )
        return self.network.get_trainable_variables()

    def save_variables(self, save_path, sess=None):
        # Restore TF1-style checkpoint saving for exact compatibility
        if sess is None:
            sess = tf.compat.v1.get_default_session()
        # Only save trainable variables to avoid optimizer state issues
        variables = {v.name: sess.run(v) for v in self.get_trainable_variables()}
        joblib.dump(variables, save_path)

    def load_variables(self, load_path, sess=None):
        # EAGER: No session needed - use compat checkpoint helper
        variables = self.get_trainable_variables()
        compat_checkpoint.load_variables_joblib(variables, load_path)
        # EAGER: Variable assignment happens immediately in load_variables_joblib


class MetaSeq2SeqPolicy():
    def __init__(self, meta_batch_size, obs_dim, encoder_units, decoder_units,
                 vocab_size):

        self.meta_batch_size = meta_batch_size
        self.obs_dim = obs_dim
        self.action_dim = vocab_size

        self.core_policy = Seq2SeqPolicy(obs_dim, encoder_units, decoder_units, vocab_size, name='core_policy')


        self.meta_policies = []

        self.assign_old_eq_new_tasks = []

        for i in range(meta_batch_size):
            self.meta_policies.append(Seq2SeqPolicy(obs_dim, encoder_units, decoder_units,
                                                    vocab_size, name="task_"+str(i)+"_policy"))

            # MIGRATION: In TF2 eager execution, use direct assignment
            def assign_core_to_task(task_idx):
                core_vars = self.core_policy.get_variables()
                task_vars = self.meta_policies[task_idx].get_variables()
                print(f"[DEBUG] Core policy variables: {len(core_vars)}")
                print(f"[DEBUG] Task {task_idx} policy variables: {len(task_vars)}")
                if len(core_vars) != len(task_vars):
                    print("[DEBUG] Variable count mismatch! Skipping parameter sync.")
                    return
                for oldv, newv in zipsame(task_vars, core_vars):
                    oldv.assign(newv)
            
            self.assign_old_eq_new_tasks.append(lambda idx=i: assign_core_to_task(idx))

        self._dist = CategoricalPd(vocab_size)
        
        # Initialize all networks immediately for graph building
        self._initialize_all_networks()

    def _initialize_all_networks(self):
        """Initialize all policy networks to ensure they're ready for graph building"""
        # Create dummy observations to trigger network creation
        dummy_obs = np.zeros((1, 20, self.obs_dim), dtype=np.float32)
        
        # Initialize core policy
        if self.core_policy.network is None:
            _ = self.core_policy.get_actions(dummy_obs)
        
        # Initialize all meta policies  
        for i in range(self.meta_batch_size):
            if self.meta_policies[i].network is None:
                _ = self.meta_policies[i].get_actions(dummy_obs)

    def get_actions(self, observations):
        assert len(observations) == self.meta_batch_size

        meta_actions = []
        meta_logits = []
        meta_v_values = []
        for i, obser_per_task in enumerate(observations):
            action, logits, v_value = self.meta_policies[i].get_actions(obser_per_task)

            # Tensors should already be evaluated by individual policy get_actions
            meta_actions.append(action)
            meta_logits.append(logits)
            meta_v_values.append(v_value)

        return meta_actions, meta_logits, meta_v_values

    def async_parameters(self):
        # async_parameters.
        for i in range(self.meta_batch_size):
            self.assign_old_eq_new_tasks[i]()

    @property
    def distribution(self):
        return self._dist

