import os
import joblib

import numpy as np
import tensorflow as tf
import policies.model_helper as model_helper
from policies.graph2seq_encoder import create_graph2seq_encoder

from policies.distributions.categorical_pd import CategoricalPd
import utils as U
from utils.utils import zipsame

tf.get_logger().setLevel('WARNING')


class TF2BasicDecoder(tf.keras.layers.Layer):
    """TF2 implementation of BasicDecoder for seq2seq"""
    def __init__(self, cell, output_layer=None, **kwargs):
        super(TF2BasicDecoder, self).__init__(**kwargs)
        self.cell = cell
        self.output_layer = output_layer
        
    def call(self, inputs, initial_state, sequence_length, training=True):
        """Run the decoder
        Args:
            inputs: decoder inputs [batch_size, seq_len, embedding_dim]
            initial_state: initial state for decoder
            sequence_length: length of sequences [batch_size]
            training: whether in training mode
        Returns:
            outputs: decoder outputs
            final_state: final decoder state
        """
        batch_size = tf.shape(inputs)[0]
        max_time = tf.shape(inputs)[1]
        
        # Initialize output arrays
        outputs_ta = tf.TensorArray(dtype=tf.float32, size=max_time, element_shape=None)
        
        def loop_fn(time, cell_state, outputs_ta):
            # Get input for this timestep
            cell_input = inputs[:, time, :]
            
            # Run RNN cell
            if hasattr(self.cell, '__call__'):
                cell_output, next_state = self.cell(cell_input, cell_state, training=training)
            else:
                cell_output, next_state = self.cell(cell_input, cell_state)
            
            # Apply output layer if provided
            if self.output_layer is not None:
                output = self.output_layer(cell_output)
            else:
                output = cell_output
            
            # Write to output array
            outputs_ta = outputs_ta.write(time, output)
            
            return time + 1, next_state, outputs_ta
        
        # Run the loop
        _, final_state, outputs_ta = tf.while_loop(
            lambda time, _, __: time < max_time,
            loop_fn,
            loop_vars=(tf.constant(0), initial_state, outputs_ta)
        )
        
        # Stack outputs
        outputs = tf.transpose(outputs_ta.stack(), [1, 0, 2])
        
        return outputs, final_state


class TF2GreedyEmbeddingHelper:
    """TF2 compatible GreedyEmbeddingHelper for decoding"""
    def __init__(self, embedding, start_tokens, end_token):
        self.embedding = embedding
        self.start_tokens = start_tokens
        self.end_token = end_token
        self._batch_size = tf.size(start_tokens)
        
    def initialize(self):
        finished = tf.zeros([self._batch_size], dtype=tf.bool)
        next_inputs = tf.nn.embedding_lookup(self.embedding, self.start_tokens)
        return finished, next_inputs
    
    def sample(self, outputs):
        """Greedy sampling - take argmax"""
        sample_ids = tf.argmax(outputs, axis=-1, output_type=tf.int32)
        return sample_ids
    
    def next_inputs(self, sample_ids):
        finished = tf.equal(sample_ids, self.end_token)
        next_inputs = tf.nn.embedding_lookup(self.embedding, sample_ids)
        return finished, next_inputs


class TF2SampleEmbeddingHelper:
    """TF2 compatible SampleEmbeddingHelper for sampling during training"""
    def __init__(self, embedding, sequence_length, start_tokens, end_token, seed=None):
        self.embedding = embedding
        self.sequence_length = sequence_length
        self.start_tokens = start_tokens
        self.end_token = end_token
        self.seed = seed
        self._batch_size = tf.size(start_tokens)
        
    def initialize(self):
        finished = tf.zeros([self._batch_size], dtype=tf.bool)
        next_inputs = tf.nn.embedding_lookup(self.embedding, self.start_tokens)
        return finished, next_inputs
    
    def sample(self, outputs):
        """Sample from categorical distribution"""
        sample_ids = tf.random.categorical(outputs, 1, seed=self.seed)
        sample_ids = tf.squeeze(sample_ids, axis=-1)
        sample_ids = tf.cast(sample_ids, tf.int32)
        return sample_ids
    
    def next_inputs(self, sample_ids, time):
        finished = time >= self.sequence_length
        all_finished = tf.reduce_all(finished)
        next_inputs = tf.cond(
            all_finished,
            lambda: tf.nn.embedding_lookup(self.embedding, self.start_tokens),
            lambda: tf.nn.embedding_lookup(self.embedding, sample_ids)
        )
        return finished, next_inputs


class TF2TrainingHelper:
    """TF2 compatible TrainingHelper for teacher forcing"""
    def __init__(self, inputs, sequence_length):
        self.inputs = inputs
        self.sequence_length = sequence_length
        self._batch_size = tf.shape(inputs)[0]
        
    def initialize(self):
        finished = tf.zeros([self._batch_size], dtype=tf.bool)
        next_inputs = self.inputs[:, 0, :]
        return finished, next_inputs
    
    def next_inputs(self, time):
        next_time = time + 1
        finished = next_time >= self.sequence_length
        next_inputs = self.inputs[:, next_time, :]
        return finished, next_inputs


class Seq2SeqNetwork(tf.keras.Model):
    def __init__(self, hparams, name="seq2seq"):
        super(Seq2SeqNetwork, self).__init__(name=name)
        
        self.encoder_hidden_unit = hparams.encoder_units
        self.decoder_hidden_unit = hparams.decoder_units
        self.is_bidencoder = hparams.is_bidencoder
        
        self.n_features = hparams.n_features
        self.time_major = hparams.time_major
        self.is_attention = hparams.is_attention
        
        self.unit_type = hparams.unit_type
        self.mode = 'train'  # Default mode, can be changed
        
        self.num_layers = hparams.num_layers
        self.num_residual_layers = hparams.num_residual_layers
        
        self.start_token = hparams.start_token
        self.end_token = hparams.end_token
        
        # Create embeddings
        self.embeddings = self.add_weight(
            name="embeddings",
            shape=[self.n_features, self.encoder_hidden_unit],
            initializer=tf.keras.initializers.RandomUniform(-1.0, 1.0),
            dtype=tf.float32,
            trainable=True
        )
        
        # Encoder embedding layer
        self.encoder_embedding_layer = tf.keras.layers.Dense(
            self.encoder_hidden_unit,
            activation=None,
            name="encoder_embeddings"
        )
        
        # Output projection layer
        self.output_layer = tf.keras.layers.Dense(
            self.n_features,
            use_bias=False,
            name="output_projection"
        )
        
        # Value function layers
        self.q_layer = tf.keras.layers.Dense(
            self.n_features,
            activation=None,
            name="qvalue_layer"
        )
        
        # Pre-create Graph2Seq encoder to avoid creating variables in tf.function
        from policies.graph2seq_encoder import Graph2SeqEncoderAdapter
        # Graph2Seq expects hidden_dim to be the actual output dimension divided by 2
        # since it concatenates forward and backward for bidirectional
        graph2seq_hidden = 32 if self.is_bidencoder else 64
        self.graph2seq_encoder = Graph2SeqEncoderAdapter(
            input_dim=self.encoder_hidden_unit,  # Input dimension after embedding
            hidden_dim=graph2seq_hidden,  # Will output 2*hidden_dim=64 for concat
            num_layers=self.num_layers,
            bidirectional=self.is_bidencoder,
            mode='train',  # Use train mode to include dropout
            decoder_hidden_unit=self.decoder_hidden_unit  # Pass decoder size for proper state projection
        )
        
        # Build decoder cell
        self.decoder_cell = self._build_decoder_cell(hparams)
        
        # Attention mechanism (if enabled)
        if self.is_attention:
            from tensorflow.keras.layers import Attention
            self.attention_layer = tf.keras.layers.Dense(
                self.decoder_hidden_unit,
                name="attention_layer"
            )
            # Pre-create query projection layer for attention mechanism
            # Graph2Seq outputs 64 dims (2*32), decoder uses 128 dims
            # So we need to project query from decoder_hidden_unit to encoder output dimension
            expected_encoder_dim = 64  # Graph2Seq output: 2*hidden_dim where hidden_dim=32
            self.query_projection_layer = tf.keras.layers.Dense(
                expected_encoder_dim,
                use_bias=False,
                name="query_projection"
            )
        
    def _build_decoder_cell(self, hparams):
        """Build decoder RNN cell"""
        cells = []
        for i in range(self.num_layers):
            if self.unit_type == "lstm":
                cell = tf.keras.layers.LSTMCell(self.decoder_hidden_unit)
            elif self.unit_type == "gru":
                cell = tf.keras.layers.GRUCell(self.decoder_hidden_unit)
            else:
                cell = tf.keras.layers.SimpleRNNCell(self.decoder_hidden_unit)
            
            # Add dropout wrapper if in training mode
            if hparams.dropout > 0:
                cell = tf.keras.layers.Dropout(hparams.dropout)(cell)
            
            cells.append(cell)
        
        if len(cells) == 1:
            return cells[0]
        else:
            return tf.keras.layers.StackedRNNCells(cells)
    
    def encode(self, encoder_inputs):
        """Run encoder"""
        # Apply encoder embeddings
        encoder_embeddings = self.encoder_embedding_layer(encoder_inputs)
        
        # Use pre-created Graph2Seq encoder
        encoder_outputs, encoder_state = self.graph2seq_encoder.encode(encoder_embeddings)
        
        return encoder_outputs, encoder_state
    
    def decode(self, decoder_inputs, encoder_outputs, encoder_state, sequence_length, mode="train"):
        """Run decoder
        Args:
            decoder_inputs: decoder input ids or embeddings
            encoder_outputs: encoder outputs for attention
            encoder_state: encoder final state
            sequence_length: decoder sequence lengths
            mode: "train", "sample", or "greedy"
        """
        batch_size = tf.shape(decoder_inputs)[0] if len(decoder_inputs.shape) > 1 else tf.shape(sequence_length)[0]
        
        # Get decoder embeddings
        if mode == "train":
            # decoder_inputs are already IDs, need embedding lookup
            decoder_embeddings = tf.nn.embedding_lookup(self.embeddings, decoder_inputs)
        else:
            # For sample/greedy mode, we'll generate embeddings step by step
            decoder_embeddings = None
        
        # Setup decoder based on mode
        if self.is_attention:
            # Simple attention mechanism without contrib
            # We'll use the encoder outputs to compute attention
            
            # Initialize decoder state
            if isinstance(encoder_state, tuple):
                # For LSTM state
                decoder_initial_state = encoder_state
            else:
                decoder_initial_state = encoder_state
            
            # Run decoder with attention
            if mode == "train":
                # Teacher forcing mode
                outputs = []
                state = decoder_initial_state
                
                # Manually unroll the loop for fixed sequence length
                # This avoids issues with symbolic tensors in graph mode
                seq_len = tf.shape(decoder_embeddings)[1]
                
                # Process all timesteps - fixed unrolling for graph mode
                for t in range(20):  # Assuming max sequence length of 20
                    input_t = decoder_embeddings[:, t, :]
                    
                    # Compute attention (simplified)
                    if self.is_attention and encoder_outputs is not None:
                        # Extract query from state
                        if isinstance(state, tuple):
                            # Check if this is a multi-layer state (tuple of tuples)
                            if len(state) > 0 and isinstance(state[0], tuple):
                                # Multi-layer: state is tuple of (c, h) tuples
                                # Use the hidden state from the last layer
                                query = state[-1][1] if len(state[-1]) == 2 else state[-1][0]
                            elif len(state) == 2 and not isinstance(state[0], tuple):
                                # Single layer: state is (c, h) tuple
                                query = state[1]
                            else:
                                # Fallback
                                query = state[0] if len(state) > 0 else state
                        elif isinstance(state, list):
                            # If state is a list (shouldn't happen but handle it)
                            if len(state) > 0 and isinstance(state[0], tuple):
                                query = state[-1][1] if len(state[-1]) == 2 else state[-1][0]
                            else:
                                query = state[-1] if len(state) > 0 else tf.zeros((batch_size, self.decoder_units))
                        else:
                            query = state
                        
                        # Make absolutely sure query is a tensor
                        if isinstance(query, (list, tuple)):
                            # This shouldn't happen after our fixes above, but be defensive
                            query = query[0] if len(query) > 0 else tf.zeros((batch_size, self.decoder_units))
                        
                        # Project query to match encoder output dimension if needed
                        encoder_dim = encoder_outputs.shape[-1]
                        query_dim = query.shape[-1]
                        
                        if encoder_dim != query_dim:
                            # Project query to encoder dimension using pre-created layer
                            query_proj = self.query_projection_layer(query)
                        else:
                            query_proj = query
                        
                        # Compute attention scores
                        scores = tf.matmul(encoder_outputs, tf.expand_dims(query_proj, 2))
                        scores = tf.squeeze(scores, axis=2)
                        attention_weights = tf.nn.softmax(scores, axis=1)
                        
                        # Compute context vector
                        context = tf.reduce_sum(
                            encoder_outputs * tf.expand_dims(attention_weights, 2),
                            axis=1
                        )
                        
                        # Combine context with input
                        input_t = tf.concat([input_t, context], axis=1)
                        input_t = self.attention_layer(input_t)
                    
                    # Run decoder cell
                    output, state = self.decoder_cell(input_t, state, training=True)
                    outputs.append(output)
                
                outputs = tf.stack(outputs, axis=1)
                final_state = state
                
            else:
                # Sampling or greedy mode - generate step by step
                outputs = []
                state = decoder_initial_state
                
                if mode == "greedy":
                    next_input = tf.nn.embedding_lookup(
                        self.embeddings,
                        tf.fill([batch_size], self.start_token)
                    )
                else:  # sample mode
                    next_input = tf.nn.embedding_lookup(
                        self.embeddings,
                        tf.fill([batch_size], self.start_token)
                    )
                
                # Fixed unrolling for graph mode - assume max length of 20
                for t in range(20):  # Fixed max length
                    # Compute attention if needed
                    if self.is_attention and encoder_outputs is not None:
                        # Extract query from state
                        if isinstance(state, tuple):
                            # Check if this is a multi-layer state (tuple of tuples)
                            if len(state) > 0 and isinstance(state[0], tuple):
                                # Multi-layer: state is tuple of (c, h) tuples
                                # Use the hidden state from the last layer
                                query = state[-1][1] if len(state[-1]) == 2 else state[-1][0]
                            elif len(state) == 2 and not isinstance(state[0], tuple):
                                # Single layer: state is (c, h) tuple
                                query = state[1]
                            else:
                                # Fallback
                                query = state[0] if len(state) > 0 else state
                        elif isinstance(state, list):
                            # If state is a list (shouldn't happen but handle it)
                            if len(state) > 0 and isinstance(state[0], tuple):
                                query = state[-1][1] if len(state[-1]) == 2 else state[-1][0]
                            else:
                                query = state[-1] if len(state) > 0 else tf.zeros((batch_size, self.decoder_units))
                        else:
                            query = state
                        
                        # Make absolutely sure query is a tensor
                        if isinstance(query, (list, tuple)):
                            # This shouldn't happen after our fixes above, but be defensive
                            query = query[0] if len(query) > 0 else tf.zeros((batch_size, self.decoder_units))
                        
                        # Project query to match encoder output dimension if needed
                        encoder_dim = encoder_outputs.shape[-1]
                        query_dim = query.shape[-1]
                        
                        if encoder_dim != query_dim:
                            # Project query to encoder dimension using pre-created layer
                            query_proj = self.query_projection_layer(query)
                        else:
                            query_proj = query
                        
                        scores = tf.matmul(encoder_outputs, tf.expand_dims(query_proj, 2))
                        scores = tf.squeeze(scores, axis=2)
                        attention_weights = tf.nn.softmax(scores, axis=1)
                        context = tf.reduce_sum(
                            encoder_outputs * tf.expand_dims(attention_weights, 2),
                            axis=1
                        )
                        
                        cell_input = tf.concat([next_input, context], axis=1)
                        cell_input = self.attention_layer(cell_input)
                    else:
                        cell_input = next_input
                    
                    # Run decoder cell
                    output, state = self.decoder_cell(cell_input, state, training=False)
                    outputs.append(output)
                    
                    # Get next input
                    logits = self.output_layer(output)
                    
                    if mode == "greedy":
                        next_token = tf.argmax(logits, axis=-1, output_type=tf.int32)
                    else:  # sample
                        next_token = tf.random.categorical(logits, 1)
                        next_token = tf.squeeze(next_token, axis=-1)
                        next_token = tf.cast(next_token, tf.int32)
                    
                    next_input = tf.nn.embedding_lookup(self.embeddings, next_token)
                
                outputs = tf.stack(outputs, axis=1)
                final_state = state
                
        else:
            # No attention - simple decoder
            if mode == "train":
                # Use teacher forcing
                outputs = []
                state = encoder_state
                
                seq_len = tf.shape(decoder_embeddings)[1]
                for t in tf.range(seq_len):
                    input_t = decoder_embeddings[:, t, :]
                    output, state = self.decoder_cell(input_t, state, training=True)
                    outputs.append(output)
                
                outputs = tf.stack(outputs, axis=1)
                final_state = state
            else:
                # Generate step by step
                outputs = []
                state = encoder_state
                next_input = tf.nn.embedding_lookup(
                    self.embeddings,
                    tf.fill([batch_size], self.start_token)
                )
                
                for t in range(sequence_length[0]):
                    output, state = self.decoder_cell(next_input, state, training=False)
                    outputs.append(output)
                    
                    logits = self.output_layer(output)
                    if mode == "greedy":
                        next_token = tf.argmax(logits, axis=-1, output_type=tf.int32)
                    else:
                        next_token = tf.random.categorical(logits, 1)
                        next_token = tf.squeeze(next_token, axis=-1)
                        next_token = tf.cast(next_token, tf.int32)
                    
                    next_input = tf.nn.embedding_lookup(self.embeddings, next_token)
                
                outputs = tf.stack(outputs, axis=1)
                final_state = state
        
        # Apply output projection
        logits = self.output_layer(outputs)
        
        return logits, outputs, final_state
    
    def call(self, encoder_inputs, decoder_inputs, decoder_targets, decoder_full_length, training=True):
        """Forward pass through the network"""
        # Encode
        encoder_outputs, encoder_state = self.encode(encoder_inputs)
        
        # Decode
        if training:
            decoder_logits, decoder_outputs, decoder_state = self.decode(
                decoder_inputs, encoder_outputs, encoder_state, 
                decoder_full_length, mode="train"
            )
        else:
            decoder_logits, decoder_outputs, decoder_state = self.decode(
                decoder_inputs, encoder_outputs, encoder_state,
                decoder_full_length, mode="sample"
            )
        
        # Compute value function
        pi = tf.nn.softmax(decoder_logits)
        q = self.q_layer(decoder_logits)
        vf = tf.reduce_sum(pi * q, axis=-1)
        
        # Get predictions
        decoder_prediction = tf.argmax(decoder_logits, axis=-1, output_type=tf.int32)
        
        return decoder_logits, vf, decoder_prediction, decoder_outputs
    
    def get_trainable_variables(self):
        return self.trainable_variables


class Seq2SeqPolicy:
    def __init__(self, obs_dim, encoder_units,
                 decoder_units, vocab_size, name="pi"):
        self.obs_dim = obs_dim
        self.action_dim = vocab_size
        self.name = name
        
        # Create hyperparameters using a simple class
        class HParams:
            pass
        
        self.hparams = HParams()
        self.hparams.__dict__.update(dict(
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
        ))
        
        # Create network
        self.network = Seq2SeqNetwork(self.hparams, name=name)
        
        # Build the network by calling it once with dummy inputs
        dummy_obs = tf.zeros((1, 1, obs_dim), dtype=tf.float32)
        dummy_decoder_inputs = tf.zeros((1, 1), dtype=tf.int32)
        dummy_decoder_targets = tf.zeros((1, 1), dtype=tf.int32)
        dummy_length = tf.constant([1], dtype=tf.int32)
        
        _ = self.network(dummy_obs, dummy_decoder_inputs, dummy_decoder_targets, dummy_length, training=False)
        
        self._dist = CategoricalPd(vocab_size)
    
    def forward_train(self, observations, decoder_inputs, decoder_targets, decoder_full_length):
        """Forward pass for training - returns logits and value function"""
        logits, vf, _, _ = self.network(
            observations, decoder_inputs, decoder_targets, decoder_full_length, training=True
        )
        return logits, vf
    
    def get_actions(self, observations):
        """Get actions for sampling"""
        batch_size = observations.shape[0]
        seq_len = observations.shape[1]
        
        decoder_full_length = np.array([seq_len] * batch_size, dtype=np.int32)
        decoder_full_length_tf = tf.convert_to_tensor(decoder_full_length, dtype=tf.int32)
        
        # Dummy decoder inputs for sampling mode
        dummy_decoder_inputs = tf.zeros((batch_size, seq_len), dtype=tf.int32)
        dummy_decoder_targets = tf.zeros((batch_size, seq_len), dtype=tf.int32)
        
        # Run network in sampling mode
        logits, vf, predictions, _ = self.network(
            observations, dummy_decoder_inputs, dummy_decoder_targets,
            decoder_full_length_tf, training=False
        )
        
        return predictions.numpy(), logits.numpy(), vf.numpy()
    
    @property
    def distribution(self):
        return self._dist
    
    def get_variables(self):
        return self.network.variables
    
    def get_trainable_variables(self):
        return self.network.trainable_variables
    
    def save_variables(self, save_path):
        """Save variables to file"""
        variables = self.get_variables()
        save_dict = {v.name: v.numpy() for v in variables}
        
        dirname = os.path.dirname(save_path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        
        joblib.dump(save_dict, save_path)
    
    def load_variables(self, load_path):
        """Load variables from file"""
        variables = self.get_variables()
        loaded_params = joblib.load(os.path.expanduser(load_path))
        
        for v in variables:
            if v.name in loaded_params:
                v.assign(loaded_params[v.name])


class MetaSeq2SeqPolicy:
    def __init__(self, meta_batch_size, obs_dim, encoder_units, decoder_units,
                 vocab_size):
        
        self.meta_batch_size = meta_batch_size
        self.obs_dim = obs_dim
        self.action_dim = vocab_size
        
        # Create core policy
        self.core_policy = Seq2SeqPolicy(obs_dim, encoder_units, decoder_units, vocab_size, name='core_policy')
        
        # Create meta policies for each task
        self.meta_policies = []
        for i in range(meta_batch_size):
            self.meta_policies.append(
                Seq2SeqPolicy(obs_dim, encoder_units, decoder_units, vocab_size, name=f"task_{i}_policy")
            )
        
        self._dist = CategoricalPd(vocab_size)
        
        # Initialize meta policies with core policy weights
        self.async_parameters()
    
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
        """Copy parameters from core policy to meta policies"""
        core_vars = self.core_policy.get_variables()
        
        for i in range(self.meta_batch_size):
            meta_vars = self.meta_policies[i].get_variables()
            
            for core_var, meta_var in zip(core_vars, meta_vars):
                if core_var.shape == meta_var.shape:
                    meta_var.assign(core_var)
    
    @property
    def distribution(self):
        return self._dist