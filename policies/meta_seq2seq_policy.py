import os
import joblib

import numpy as np
import tensorflow as tf
import policies.model_helper as model_helper
from policies.graph2seq_encoder import create_graph2seq_encoder

from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops
from policies.distributions.categorical_pd import CategoricalPd
import utils as U
from utils.utils import zipsame

tf.get_logger().setLevel('WARNING')


class FixedSequenceLearningSampleEmbedingHelper:
    """TF2.19 compatible SampleEmbeddingHelper"""
    def __init__(self, sequence_length, embedding, start_tokens, end_token, softmax_temperature=None, seed=None):
        self.embedding = embedding
        self.start_tokens = start_tokens
        self.end_token = end_token
        self.softmax_temperature = softmax_temperature
        self.seed = seed
        self._sequence_length = tf.convert_to_tensor(
            sequence_length, name="sequence_length")
        if len(self._sequence_length.shape) != 1:
            raise ValueError(
                "Expected sequence_length to be a vector, but received shape: %s" %
                self._sequence_length.shape)

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

        sample_ids = tf.random.categorical(logits, 1, seed=self.seed)
        sample_ids = tf.squeeze(sample_ids, axis=-1)
        sample_ids = tf.cast(sample_ids, tf.int32)  # Ensure int32 for TensorArray

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


class TF2LuongAttention(tf.keras.layers.Layer):
    """TF2.19 replacement for tf.contrib.seq2seq.LuongAttention"""
    def __init__(self, num_units, memory, memory_sequence_length=None, scale=False, **kwargs):
        super(TF2LuongAttention, self).__init__(**kwargs)
        self.num_units = num_units
        self.memory = memory
        self.memory_sequence_length = memory_sequence_length
        self.scale = scale
        
        # Don't create layers in __init__ - let Keras handle it automatically
        self.query_layer = None
        self.memory_layer = None
        self._processed_memory = None
            
    def build(self, input_shape):
        """Build method for Keras compatibility"""
        if self.query_layer is None:
            # Create sub-layers in build method
            self.query_layer = tf.keras.layers.Dense(self.num_units, use_bias=False, name="attention_query_layer")
            self.memory_layer = tf.keras.layers.Dense(self.num_units, use_bias=False, name="attention_memory_layer")
            
            # Build the memory layer immediately with known memory shape
            self.memory_layer.build(self.memory.shape)
            
            # Pre-process memory once during build
            self._processed_memory = self.memory_layer(self.memory)
        
        super(TF2LuongAttention, self).build(input_shape)
        
    def call(self, query):
        # query: [batch_size, num_units]
        # memory: [batch_size, max_time, memory_depth]
        
        # Process memory dynamically to match current batch size
        current_batch_size = tf.shape(query)[0]
        memory_batch_size = tf.shape(self.memory)[0]
        
        # Handle batch size mismatch - always use minimum of the two batch sizes
        actual_batch_size = tf.minimum(current_batch_size, memory_batch_size)
        
        # Slice both query and memory to matching batch size
        query_sliced = query[:actual_batch_size]
        processed_query = self.query_layer(query_sliced)
        processed_query = tf.expand_dims(processed_query, 1)  # [batch_size, 1, num_units]
        
        memory_for_context = self.memory[:actual_batch_size]
        processed_memory = self.memory_layer(memory_for_context)
        
        # Calculate alignment scores
        score = tf.reduce_sum(processed_query * processed_memory, axis=2)  # [batch_size, max_time]
        
        if self.scale:
            score = score / tf.sqrt(tf.cast(self.num_units, dtype=score.dtype))
        
        # Mask if sequence lengths provided
        if self.memory_sequence_length is not None:
            max_time = tf.shape(memory_for_context)[1]
            # Handle sequence length batch size mismatch
            seq_length_to_use = self.memory_sequence_length[:actual_batch_size]
            mask = tf.sequence_mask(seq_length_to_use, max_time, dtype=score.dtype)
            score = score * mask + (1.0 - mask) * tf.float32.min
        
        # Apply softmax
        alignments = tf.nn.softmax(score, axis=1)
        
        # Calculate context vector using the same memory slice used for scores
        expanded_alignments = tf.expand_dims(alignments, 2)  # [batch_size, max_time, 1]
        context = tf.reduce_sum(expanded_alignments * memory_for_context, axis=1)  # [batch_size, memory_depth]
        
        # If we had to slice due to batch size mismatch, pad the results back to original query batch size
        if current_batch_size > actual_batch_size:
            # Pad context and alignments to match original query batch size
            memory_depth = tf.shape(context)[1]
            max_time = tf.shape(alignments)[1]
            
            context_padding = tf.zeros([current_batch_size - actual_batch_size, memory_depth], dtype=context.dtype)
            context = tf.concat([context, context_padding], axis=0)
            
            alignments_padding = tf.zeros([current_batch_size - actual_batch_size, max_time], dtype=alignments.dtype)
            alignments = tf.concat([alignments, alignments_padding], axis=0)
        
        return context, alignments
    
    def __call__(self, query):
        """Compatibility wrapper for call method"""
        return self.call(query)


class TF2AttentionWrapper(tf.keras.layers.Layer):
    """TF2.19 replacement for tf.contrib.seq2seq.AttentionWrapper"""
    def __init__(self, cell, attention_mechanism, attention_layer_size=None, **kwargs):
        super(TF2AttentionWrapper, self).__init__(**kwargs)
        self.cell = cell
        self.attention_mechanism = attention_mechanism
        self.attention_layer_size = attention_layer_size or attention_mechanism.num_units
        
        # Don't create layers in __init__ - let Keras handle it automatically
        self.attention_layer = None
        self.output_layer = None
        
    def build(self, input_shape):
        """Build method for Keras compatibility"""
        if self.attention_layer is None:
            # Create sub-layers in build method
            self.attention_layer = tf.keras.layers.Dense(
                self.attention_layer_size, 
                use_bias=False, 
                name="attention_layer"
            )
            
            # Combined output layer
            memory_depth = self.attention_mechanism.memory.shape[-1]
            cell_output_size = getattr(self.cell, 'output_size', self.cell.units) if hasattr(self.cell, 'units') else self.cell.output_size
            combined_size = cell_output_size + memory_depth
            self.output_layer = tf.keras.layers.Dense(
                self.attention_layer_size,
                use_bias=False,
                name="attention_output_layer"
            )
        
        super(TF2AttentionWrapper, self).build(input_shape)
        
    def zero_state(self, batch_size, dtype):
        """Create zero state for attention wrapper"""
        # TF2.19: get_initial_state for Keras RNN cells - signature is different
        if hasattr(self.cell, 'get_initial_state') and hasattr(self.cell, 'cells'):
            # For StackedRNNCells in TF2, only takes batch_size parameter
            cell_state = self.cell.get_initial_state(batch_size=batch_size)
        elif hasattr(self.cell, 'zero_state'):
            # Fallback for TF1-style cells
            cell_state = self.cell.zero_state(batch_size, dtype)
        else:
            # For individual Keras cells
            cell_state = self.cell.get_initial_state(batch_size=batch_size)
        
        # Attention context starts as zeros
        # Handle case where memory shape might be dynamic
        memory_depth = self.attention_mechanism.memory.shape[-1]
        if memory_depth is None:
            memory_depth = tf.shape(self.attention_mechanism.memory)[-1]
        attention_context = tf.zeros([batch_size, memory_depth], dtype=dtype)
        return (cell_state, attention_context)
    
    def call(self, inputs, state=None):
        """Run one step of attention wrapper"""
        cell_state, prev_attention_context = state
        
        # Concatenate input with previous attention context
        cell_input = tf.concat([inputs, prev_attention_context], axis=-1)
        
        # Run the wrapped cell
        cell_output, new_cell_state = self.cell(cell_input, cell_state)
        
        # Apply attention mechanism
        attention_context, alignments = self.attention_mechanism(cell_output)
        
        # Combine cell output with attention context
        combined = tf.concat([cell_output, attention_context], axis=-1)
        attention_output = self.output_layer(combined)
        
        new_state = (new_cell_state, attention_context)
        return attention_output, new_state
    
    def __call__(self, inputs, state):
        """Compatibility wrapper for call method"""
        return self.call(inputs, state)
    
    @property
    def state_size(self):
        """State size for Keras RNN compatibility"""
        # Return tuple of (cell_state_size, attention_context_size)
        cell_state_size = self.cell.state_size if hasattr(self.cell, 'state_size') else self.cell.units
        attention_size = self.attention_mechanism.memory.shape[-1]
        return (cell_state_size, attention_size)
    
    @property
    def output_size(self):
        """Output size for Keras RNN compatibility"""
        return self.attention_layer_size
    
    @property
    def compute_dtype(self):
        """Compute dtype for Keras RNN compatibility"""
        return tf.float32


class TF2DecoderOutput:
    """Container for decoder outputs"""
    def __init__(self, rnn_output, sample_id):
        self.rnn_output = rnn_output
        self.sample_id = sample_id


class TF2BasicDecoder:
    """TF2.19 replacement for tf.contrib.seq2seq.BasicDecoder"""
    def __init__(self, cell, helper, initial_state, output_layer=None):
        self.cell = cell
        self.helper = helper
        self.initial_state = initial_state
        self.output_layer = output_layer


def tf2_dynamic_decode(decoder, output_time_major=False, maximum_iterations=None):
    """TF2.19 replacement for tf.contrib.seq2seq.dynamic_decode"""
    
    # Get initial values
    batch_size = tf.shape(decoder.initial_state[0])[0] if isinstance(decoder.initial_state, tuple) else tf.shape(decoder.initial_state)[0]
    
    # For training helper (teacher forcing)
    if hasattr(decoder.helper, 'inputs'):
        # Training mode with teacher forcing
        inputs = decoder.helper.inputs
        sequence_length = decoder.helper.sequence_length
        max_time = tf.shape(inputs)[1]
        
        # Initialize output arrays
        output_ta = tf.TensorArray(dtype=tf.float32, size=max_time)
        
        # Manually unroll the RNN
        state = decoder.initial_state
        
        # Convert tf.range loop to use explicit shape handling
        time_steps = tf.unstack(tf.range(max_time))
        
        for time_step in time_steps:
            # Get input for this timestep
            input_t = inputs[:, time_step, :]
            
            # Call cell
            output_t, new_state = decoder.cell(input_t, state)
            
            # Ensure state shape consistency by setting explicit shape
            if isinstance(state, tuple):
                # For structured states (like attention wrapper)
                new_state = tf.nest.map_structure(
                    lambda old_s, new_s: tf.ensure_shape(new_s, old_s.shape) if hasattr(old_s, 'shape') else new_s,
                    state, new_state
                )
            else:
                # For simple tensor states
                new_state = tf.ensure_shape(new_state, state.shape)
            
            state = new_state
            
            # Store output
            output_ta = output_ta.write(time_step, output_t)
        
        # Stack outputs
        outputs = output_ta.stack()  # [time, batch, features]
        outputs = tf.transpose(outputs, [1, 0, 2])  # [batch, time, features]
        
        # Handle time_major output transpose
        if output_time_major:
            outputs = tf.transpose(outputs, [1, 0, 2])
        
        # Apply output projection
        if decoder.output_layer is not None:
            logits = decoder.output_layer(outputs)
        else:
            logits = outputs
            
        # Sample IDs (for training this is usually the target sequence)
        sample_ids = tf.argmax(logits, axis=-1, output_type=tf.int32)
        
        return TF2DecoderOutput(logits, sample_ids), state, tf.reduce_max(sequence_length)
    
    else:
        # Inference mode - step by step decoding
        max_iterations = maximum_iterations or 50
        
        # Get output shape for TensorArray initialization
        # Run one dummy step to get output shape
        state = decoder.initial_state
        if hasattr(decoder.helper, 'start_tokens'):
            dummy_input = tf.nn.embedding_lookup(decoder.helper.embedding, decoder.helper.start_tokens)
        else:
            dummy_input = tf.zeros([batch_size, decoder.helper.embedding.shape[-1]], dtype=tf.float32)
        
        dummy_output, _ = decoder.cell(dummy_input, state)
        if decoder.output_layer is not None:
            dummy_logits = decoder.output_layer(dummy_output)
        else:
            dummy_logits = dummy_output
        
        output_size = tf.shape(dummy_logits)[-1]
        
        # Initialize TensorArrays
        outputs_ta = tf.TensorArray(dtype=tf.float32, size=max_iterations, dynamic_size=False)
        sample_ids_ta = tf.TensorArray(dtype=tf.int32, size=max_iterations, dynamic_size=False)
        
        # Reset state for actual computation
        state = decoder.initial_state
        
        # Get first input
        if hasattr(decoder.helper, 'start_tokens'):
            current_input = tf.nn.embedding_lookup(decoder.helper.embedding, decoder.helper.start_tokens)
        else:
            current_input = tf.zeros([batch_size, decoder.helper.embedding.shape[-1]], dtype=tf.float32)
        
        # Unroll the loop using tf.range to ensure proper tensor graph construction
        for time in tf.range(max_iterations):
            # Run one step
            cell_output, new_state = decoder.cell(current_input, state)
            
            # Ensure state maintains its original structure (convert lists to tuples)
            # Use a simpler approach that works with TF autograph
            if isinstance(state, tuple) and len(state) == 2:
                # Handle the specific case of ((state1, state2), attention_context)
                rnn_states, attention_context = new_state
                
                # Convert RNN states from list to tuple if needed
                if isinstance(rnn_states, list):
                    # Convert each layer state from list to tuple
                    fixed_rnn_states = tuple(
                        tuple(s) if isinstance(s, list) else s 
                        for s in rnn_states
                    )
                    state = (fixed_rnn_states, attention_context)
                else:
                    state = new_state
            else:
                state = new_state
            
            # Apply output layer
            if decoder.output_layer is not None:
                logits = decoder.output_layer(cell_output)
            else:
                logits = cell_output
            
            # Sample next token
            sample_id = decoder.helper.sample(time, logits, state)
            
            # Store outputs in TensorArrays
            outputs_ta = outputs_ta.write(time, logits)
            sample_ids_ta = sample_ids_ta.write(time, sample_id)
            
            # Get next input
            current_input = tf.nn.embedding_lookup(decoder.helper.embedding, sample_id)
        
        # Stack the results from TensorArrays
        outputs = outputs_ta.stack()  # [time, batch, features]
        sample_ids = sample_ids_ta.stack()  # [time, batch]
        
        # Transpose if needed
        if not output_time_major:
            outputs = tf.transpose(outputs, [1, 0, 2])  # [time, batch, features] -> [batch, time, features]
            sample_ids = tf.transpose(sample_ids, [1, 0])  # [time, batch] -> [batch, time]
        
        return TF2DecoderOutput(outputs, sample_ids), state, max_iterations


class TrainingHelper:
    """TF2.19 replacement for tf.contrib.seq2seq.TrainingHelper"""
    def __init__(self, inputs, sequence_length, time_major=False):
        self.inputs = inputs
        self.sequence_length = sequence_length
        self.time_major = time_major


class GreedyEmbeddingHelper:
    """TF2.19 replacement for tf.contrib.seq2seq.GreedyEmbeddingHelper"""
    def __init__(self, embedding, start_tokens, end_token):
        self.embedding = embedding
        self.start_tokens = start_tokens
        self.end_token = end_token
    
    def sample(self, time, outputs, state, name=None):
        """Greedy sampling (argmax)"""
        del time, state  # unused
        return tf.argmax(outputs, axis=-1, output_type=tf.int32)


class Seq2SeqNetwork():
    def __init__(self, name,
                 hparams, reuse,
                 encoder_inputs=None,
                 decoder_inputs=None,
                 decoder_full_length=None,
                 decoder_targets=None):
        self.encoder_hidden_unit = hparams.encoder_units
        self.decoder_hidden_unit = hparams.decoder_units
        self.is_bidencoder = hparams.is_bidencoder
        self.reuse = reuse

        self.n_features = hparams.n_features
        self.time_major = hparams.time_major
        self.is_attention = hparams.is_attention

        self.unit_type = hparams.unit_type

        # default setting
        self.mode = 'train'

        self.num_layers = hparams.num_layers
        self.num_residual_layers = hparams.num_residual_layers

        self.single_cell_fn = None
        self.start_token = hparams.start_token
        self.end_token = hparams.end_token

        # Initialize without placeholders - tensors will be passed to methods
        self.encoder_inputs = encoder_inputs
        self.decoder_inputs = decoder_inputs
        self.decoder_targets = decoder_targets
        self.decoder_full_length = decoder_full_length
        
        self.scope = name
        self.hparams = hparams
        # Create embeddings as a persistent variable
        self.embeddings = tf.Variable(tf.random.uniform(
            [self.n_features,
             self.encoder_hidden_unit],
            -1.0, 1.0), dtype=tf.float32, name=f"{name}_embeddings")

        # Create layers that will be used during forward pass
        self.encoder_embedding_layer = tf.keras.layers.Dense(
            self.encoder_hidden_unit,
            activation=None,
            name=f"{name}_encoder_embeddings"
        )
        
        self.output_layer = tf.keras.layers.Dense(
            self.n_features, 
            use_bias=False, 
            name=f"{name}_output_projection"
        )
        
        # Pre-create attention mechanism and cell to avoid creating them inside @tf.function
        self._attention_mechanism = None
        self._attention_decoder_cell = None
        
        # Build the network structure if inputs are provided
        if encoder_inputs is not None:
            self._build_network()

        # Create Q-value layer
        self.q_layer = tf.keras.layers.Dense(
            self.n_features, 
            activation=None, 
            name=f"{name}_qvalue_layer"
        )
    
    def _build_network(self):
        """Build network components when inputs are available"""
        # Process encoder inputs
        self.encoder_embeddings = self.encoder_embedding_layer(self.encoder_inputs)
        
        # Look up decoder embeddings
        if self.decoder_inputs is not None:
            self.decoder_embeddings = tf.nn.embedding_lookup(self.embeddings, self.decoder_inputs)
        
        # Create target embeddings
        if self.decoder_targets is not None:
            self.decoder_targets_embeddings = tf.one_hot(
                self.decoder_targets,
                self.n_features,
                dtype=tf.float32
            )
        
        # Use Graph2Seq encoder
        self.encoder_outputs, self.encoder_state = create_graph2seq_encoder(
            encoder_inputs=self.encoder_embeddings,
            encoder_units=self.encoder_hidden_unit,
            num_layers=self.num_layers,
            is_bidirectional=self.is_bidencoder,
            mode=self.mode,
            scope_name=f"{self.scope}_encoder"
        )
        
        # Pre-create attention mechanism if needed (outside @tf.function)
        if self.is_attention and self._attention_mechanism is None:
            if self.time_major:
                attention_states = tf.transpose(self.encoder_outputs, [1, 0, 2])
            else:
                attention_states = self.encoder_outputs
                
            self._attention_mechanism = TF2LuongAttention(
                self.decoder_hidden_unit, attention_states)
            
            # Build attention mechanism immediately 
            dummy_query_shape = [tf.shape(attention_states)[0], self.decoder_hidden_unit]
            self._attention_mechanism.build(dummy_query_shape)
            
            # Also pre-create the attention wrapper cell
            decoder_cell = self._build_decoder_cell(hparams=self.hparams,
                                                  num_layers=self.num_layers,
                                                  num_residual_layers=self.num_residual_layers)
            self._attention_decoder_cell = TF2AttentionWrapper(
                    decoder_cell, self._attention_mechanism,
                    attention_layer_size=self.decoder_hidden_unit)
            # Build the wrapper with dummy input to ensure variables are created
            dummy_input_shape = [None, self.decoder_hidden_unit + self._attention_mechanism.memory.shape[-1]]
            self._attention_decoder_cell.build(dummy_input_shape)
        
        # Build decoders for different modes
        self._build_training_decoder()
        self._build_sample_decoder()
        self._build_greedy_decoder()
    
    def _build_training_decoder(self):
        """Build training decoder"""
        if self.decoder_inputs is not None and self.decoder_full_length is not None:
            self.decoder_outputs, self.decoder_state = self.create_decoder(
                self.hparams, self.encoder_outputs, self.encoder_state, model="train"
            )
            self.decoder_logits = self.decoder_outputs.rnn_output
            self.pi = tf.nn.softmax(self.decoder_logits)
            self.q = self.q_layer(self.decoder_logits)
            self.vf = tf.reduce_sum(self.pi * self.q, axis=-1)
            self.decoder_prediction = self.decoder_outputs.sample_id
    
    def _build_sample_decoder(self):
        """Build sample decoder"""
        if self.decoder_full_length is not None:
            self.sample_decoder_outputs, self.sample_decoder_state = self.create_decoder(
                self.hparams, self.encoder_outputs, self.encoder_state, model="sample"
            )
            self.sample_decoder_logits = self.sample_decoder_outputs.rnn_output
            self.sample_pi = tf.nn.softmax(self.sample_decoder_logits)
            self.sample_q = self.q_layer(self.sample_decoder_logits)
            self.sample_vf = tf.reduce_sum(self.sample_pi * self.sample_q, axis=-1)
            self.sample_decoder_prediction = self.sample_decoder_outputs.sample_id
            
            # Compute neglogp
            self.sample_decoder_embeddings = tf.one_hot(
                self.sample_decoder_prediction,
                self.n_features,
                dtype=tf.float32
            )
            self.sample_neglogp = tf.nn.softmax_cross_entropy_with_logits(
                labels=self.sample_decoder_embeddings,
                logits=self.sample_decoder_logits
            )
    
    def _build_greedy_decoder(self):
        """Build greedy decoder"""
        if self.decoder_full_length is not None:
            self.greedy_decoder_outputs, self.greedy_decoder_state = self.create_decoder(
                self.hparams, self.encoder_outputs, self.encoder_state, model="greedy"
            )
            self.greedy_decoder_logits = self.greedy_decoder_outputs.rnn_output
            self.greedy_pi = tf.nn.softmax(self.greedy_decoder_logits)
            self.greedy_q = self.q_layer(self.greedy_decoder_logits)
            self.greedy_vf = tf.reduce_sum(self.greedy_pi * self.greedy_q, axis=-1)
            self.greedy_decoder_prediction = self.greedy_decoder_outputs.sample_id

    @tf.function
    def predict_training(self, encoder_input_batch, decoder_input, decoder_full_length):
        """TF2 eager execution - pass tensors directly"""
        # Update network inputs
        self.encoder_inputs = encoder_input_batch
        self.decoder_inputs = decoder_input
        self.decoder_full_length = decoder_full_length
        
        # Recompute outputs
        self.encoder_embeddings = tf.keras.layers.Dense(
            self.encoder_hidden_unit,
            activation=None,
            name="encoder_embeddings"
        )(self.encoder_inputs)
        
        self.decoder_embeddings = tf.nn.embedding_lookup(self.embeddings, self.decoder_inputs)
        
        # Re-run encoder and decoder
        self.encoder_outputs, self.encoder_state = create_graph2seq_encoder(
            encoder_inputs=self.encoder_embeddings,
            encoder_units=self.encoder_hidden_unit,
            num_layers=self.num_layers,
            is_bidirectional=self.is_bidencoder,
            mode=self.mode,
            scope_name=f"{self.scope}_encoder"
        )
        
        self.decoder_outputs, self.decoder_state = self.create_decoder(
            self.hparams, self.encoder_outputs, self.encoder_state, model="train"
        )
        
        return self.decoder_prediction, self.pi

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

    def create_decoder(self, hparams, encoder_outputs, encoder_state, model):
        # TF2: No need for variable_scope
        if model == "greedy":
            helper = GreedyEmbeddingHelper(
                    self.embeddings,
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
            helper = TrainingHelper(
                    self.decoder_embeddings,
                    self.decoder_full_length,
                    time_major=self.time_major)
        else:
            helper = TrainingHelper(
                    self.decoder_embeddings,
                    self.decoder_full_length,
                    time_major=self.time_major)

        if self.is_attention:
            # Use pre-created attention decoder cell
            decoder_cell = self._attention_decoder_cell

            batch_size = tf.size(self.decoder_full_length)
            decoder_initial_state = decoder_cell.zero_state(batch_size, dtype=tf.float32)
            # Update the cell state from encoder
            if isinstance(encoder_state, tuple) and len(encoder_state) > 0:
                # Multi-layer case or tuple state
                if isinstance(decoder_initial_state[0], tuple):
                    # Multi-layer: encoder_state should match structure
                    decoder_initial_state = (encoder_state, decoder_initial_state[1])
                else:
                    # Single layer
                    decoder_initial_state = (encoder_state, decoder_initial_state[1])
            else:
                # Single state
                decoder_initial_state = (encoder_state, decoder_initial_state[1])
        else:
            decoder_cell = self._build_decoder_cell(hparams=hparams,
                                                    num_layers=self.num_layers,
                                                    num_residual_layers=self.num_residual_layers)

            decoder_initial_state = encoder_state

        decoder = TF2BasicDecoder(
            cell=decoder_cell,
            helper=helper,
            initial_state=decoder_initial_state,
            output_layer=self.output_layer)

        outputs, last_state, _ = tf2_dynamic_decode(decoder,
                                                   output_time_major=self.time_major,
                                                   maximum_iterations=self.decoder_full_length[0])
        return outputs, last_state

    def get_variables(self):
        """Get all variables in this network"""
        variables = [self.embeddings]
        variables.extend(self.encoder_embedding_layer.variables)
        variables.extend(self.output_layer.variables)
        variables.extend(self.q_layer.variables)
        # Add any other layer variables
        return variables

    def get_trainable_variables(self):
        """Get trainable variables in this network"""
        return [v for v in self.get_variables() if v.trainable]


class Seq2SeqPolicy():
    def __init__(self, obs_dim, encoder_units,
                 decoder_units, vocab_size, name="pi"):
        self.obs_dim = obs_dim
        self.action_dim = vocab_size
        self.name = name
        self.encoder_units = encoder_units
        self.decoder_units = decoder_units
        self.vocab_size = vocab_size

        # Create HParams manually since tf.contrib.training is removed
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

        # Initialize network eagerly
        self.network = Seq2SeqNetwork(
            name=name,
            hparams=self.hparams,
            reuse=False
        )

        self._dist = CategoricalPd(vocab_size)

    def get_actions(self, observations):
        """TF2 eager execution - pass tensors directly"""
        batch_size = tf.shape(observations)[0]
        seq_length = tf.shape(observations)[1]
        decoder_full_length = tf.fill([batch_size], seq_length)
        
        # Create decoder inputs (shifted actions starting with start token)
        decoder_inputs = tf.zeros([batch_size, seq_length], dtype=tf.int32)
        
        # Forward pass through network
        return self.forward_sample(observations, decoder_inputs, decoder_full_length)

    @property
    def distribution(self):
        return self._dist

    def forward_sample(self, observations, decoder_inputs, decoder_full_length):
        """Forward pass for sampling"""
        # Update network inputs
        self.network.encoder_inputs = observations
        self.network.decoder_inputs = decoder_inputs
        self.network.decoder_full_length = decoder_full_length
        
        # Build network components
        self.network._build_network()
        
        # Get predictions
        actions = self.network.sample_decoder_prediction
        logits = self.network.sample_decoder_logits
        v_value = self.network.sample_vf
        
        return actions, logits, v_value
    
    def forward_train(self, observations, decoder_inputs, decoder_targets, decoder_full_length):
        """Forward pass for training"""
        # Update network inputs
        self.network.encoder_inputs = observations
        self.network.decoder_inputs = decoder_inputs
        self.network.decoder_targets = decoder_targets
        self.network.decoder_full_length = decoder_full_length
        
        # Build network components
        self.network._build_network()
        
        return self.network.decoder_logits, self.network.vf

    def get_variables(self):
        return self.network.get_variables()

    def get_trainable_variables(self):
        return self.network.get_trainable_variables()

    def save_variables(self, save_path):
        """TF2 eager save variables"""
        variables = self.get_variables()

        ps = [v.numpy() for v in variables]
        save_dict = {v.name: value for v, value in zip(variables, ps)}

        dirname = os.path.dirname(save_path)
        if any(dirname):
            os.makedirs(dirname, exist_ok=True)

        joblib.dump(save_dict, save_path)

    def load_variables(self, load_path):
        """TF2 eager load variables"""
        variables = self.get_variables()

        loaded_params = joblib.load(os.path.expanduser(load_path))

        if isinstance(loaded_params, list):
            assert len(loaded_params) == len(variables), 'number of variables loaded mismatches len(variables)'
            for d, v in zip(loaded_params, variables):
                v.assign(d)
        else:
            for v in variables:
                v.assign(loaded_params[v.name])


class MetaSeq2SeqPolicy():
    def __init__(self, meta_batch_size, obs_dim, encoder_units, decoder_units,
                 vocab_size):

        self.meta_batch_size = meta_batch_size
        self.obs_dim = obs_dim
        self.action_dim = vocab_size

        self.core_policy = Seq2SeqPolicy(obs_dim, encoder_units, decoder_units, vocab_size, name='core_policy')

        self.meta_policies = []

        for i in range(meta_batch_size):
            self.meta_policies.append(Seq2SeqPolicy(obs_dim, encoder_units, decoder_units,
                                                    vocab_size, name="task_"+str(i)+"_policy"))

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
        """TF2 eager parameter synchronization"""
        # Copy core policy parameters to all meta policies
        core_vars = self.core_policy.get_variables() if self.core_policy.network else []
        
        for i in range(self.meta_batch_size):
            if self.meta_policies[i].network and core_vars:
                meta_vars = self.meta_policies[i].get_variables()
                for meta_v, core_v in zip(meta_vars, core_vars):
                    if meta_v.shape == core_v.shape:
                        meta_v.assign(core_v)

    @property
    def distribution(self):
        return self._dist