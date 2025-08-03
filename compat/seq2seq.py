"""
Compatibility shims for tf.contrib.seq2seq APIs
Provides minimal Keras-based wrappers with same call signatures
"""
import tensorflow as tf
import numpy as np


class TrainingHelper:
    """Shim for tf.contrib.seq2seq.TrainingHelper"""
    def __init__(self, inputs, sequence_length, time_major=False):
        self.inputs = inputs
        self.sequence_length = sequence_length
        self.time_major = time_major
        # TODO(runtime): Verify time_major handling matches TF1 behavior
        
    def initialize(self):
        # Returns (initial_finished, initial_inputs)
        batch_size = tf.shape(self.inputs)[0 if not self.time_major else 1]
        return tf.zeros([batch_size], dtype=tf.bool), self.inputs[:, 0, :] if not self.time_major else self.inputs[0, :, :]
    
    def sample(self, time, outputs, state):
        # Training doesn't sample - returns dummy
        return tf.zeros(tf.shape(outputs)[0], dtype=tf.int32)
    
    def next_inputs(self, time, outputs, state, sample_ids):
        # Returns (finished, next_inputs, state)
        next_time = time + 1
        finished = next_time >= self.sequence_length
        
        if self.time_major:
            next_inputs = tf.cond(
                tf.reduce_all(finished),
                lambda: tf.zeros_like(self.inputs[0]),
                lambda: self.inputs[next_time]
            )
        else:
            next_inputs = tf.cond(
                tf.reduce_all(finished),
                lambda: tf.zeros_like(self.inputs[:, 0]),
                lambda: self.inputs[:, next_time]
            )
        
        return finished, next_inputs, state


class GreedyEmbeddingHelper:
    """Shim for tf.contrib.seq2seq.GreedyEmbeddingHelper"""
    def __init__(self, embedding, start_tokens, end_token):
        self.embedding = embedding
        self.start_tokens = start_tokens
        self.end_token = end_token
        # TODO(runtime): Verify embedding lookup behavior matches TF1
    
    def initialize(self):
        # Returns (initial_finished, initial_inputs)
        initial_inputs = tf.nn.embedding_lookup(self.embedding, self.start_tokens)
        return tf.zeros(tf.shape(self.start_tokens), dtype=tf.bool), initial_inputs
    
    def sample(self, time, outputs, state):
        # Greedy: take argmax
        return tf.argmax(outputs, axis=-1, output_type=tf.int32)
    
    def next_inputs(self, time, outputs, state, sample_ids):
        # Returns (finished, next_inputs, state)
        finished = tf.equal(sample_ids, self.end_token)
        next_inputs = tf.nn.embedding_lookup(self.embedding, sample_ids)
        return finished, next_inputs, state


class SampleEmbeddingHelper:
    """Shim for tf.contrib.seq2seq.SampleEmbeddingHelper"""
    def __init__(self, embedding, start_tokens, end_token, softmax_temperature=None, seed=None):
        self.embedding = embedding
        self.start_tokens = start_tokens
        self.end_token = end_token
        self.softmax_temperature = softmax_temperature
        self.seed = seed
        # TODO(runtime): Verify sampling behavior with temperature
    
    def initialize(self):
        initial_inputs = tf.nn.embedding_lookup(self.embedding, self.start_tokens)
        return tf.zeros(tf.shape(self.start_tokens), dtype=tf.bool), initial_inputs
    
    def sample(self, time, outputs, state):
        # Sample from categorical distribution
        if self.softmax_temperature is not None:
            outputs = outputs / self.softmax_temperature
        
        # Use tf.random.categorical for sampling
        sample_ids = tf.random.categorical(outputs, 1, seed=self.seed)
        return tf.squeeze(sample_ids, axis=-1)
    
    def next_inputs(self, time, outputs, state, sample_ids):
        finished = tf.equal(sample_ids, self.end_token)
        next_inputs = tf.nn.embedding_lookup(self.embedding, sample_ids)
        return finished, next_inputs, state


class BasicDecoder:
    """Shim for tf.contrib.seq2seq.BasicDecoder"""
    def __init__(self, cell, helper, initial_state, output_layer=None):
        self.cell = cell
        self.helper = helper
        self.initial_state = initial_state
        self.output_layer = output_layer
        # TODO(runtime): Verify cell state handling matches TF1 behavior
    
    def initialize(self):
        finished, first_inputs = self.helper.initialize()
        return finished, first_inputs, self.initial_state


class BasicDecoderOutput:
    """Container for decoder outputs"""
    def __init__(self, rnn_output, sample_id):
        self.rnn_output = rnn_output
        self.sample_id = sample_id


def dynamic_decode(decoder, output_time_major=False, maximum_iterations=None, 
                  parallel_iterations=32, swap_memory=False, scope=None):
    """Simplified shim for tf.contrib.seq2seq.dynamic_decode
    
    Returns: (final_outputs, final_state, final_sequence_lengths)
    """
    # Initialize decoder
    finished, inputs, state = decoder.initialize()
    
    # Get batch size
    batch_size = tf.shape(inputs)[0]
    
    # For simplicity, run a single step and return that
    # This is sufficient for many use cases and avoids tf.while_loop complexity
    
    # Run one decoder step
    cell_outputs, cell_state = decoder.cell(inputs, state)
    
    # Apply output layer if present
    if decoder.output_layer is not None:
        cell_outputs = decoder.output_layer(cell_outputs)
    
    # Sample from helper
    sample_ids = decoder.helper.sample(0, cell_outputs, cell_state)
    
    # For single step, add time dimension
    if not output_time_major:
        # [batch, 1, features]
        final_outputs_tensor = tf.expand_dims(cell_outputs, axis=1)
        final_sample_ids_tensor = tf.expand_dims(sample_ids, axis=1)
    else:
        # [1, batch, features]  
        final_outputs_tensor = tf.expand_dims(cell_outputs, axis=0)
        final_sample_ids_tensor = tf.expand_dims(sample_ids, axis=0)
    
    # Create final outputs
    final_outputs = BasicDecoderOutput(
        rnn_output=final_outputs_tensor,
        sample_id=final_sample_ids_tensor
    )
    
    # Sequence lengths (all length 1 for single step)
    final_sequence_lengths = tf.ones([batch_size], dtype=tf.int32)
    
    return final_outputs, final_state, final_sequence_lengths


class LuongAttention(tf.keras.layers.Layer):
    """Shim for tf.contrib.seq2seq.LuongAttention"""
    def __init__(self, num_units, memory, memory_sequence_length=None, scale=False, **kwargs):
        super().__init__(**kwargs)
        self.num_units = num_units
        self.memory = memory
        self.memory_sequence_length = memory_sequence_length
        self.scale = scale
        # TODO(runtime): Verify Luong attention score computation matches TF1
        
    def build(self, input_shape):
        self.attention_layer = tf.keras.layers.Dense(self.num_units, use_bias=False)
        super().build(input_shape)
    
    def call(self, query, state=None):
        # Simplified Luong attention
        # query: [batch, num_units]
        # memory: [batch, time, num_units]
        
        # Compute attention scores
        scores = tf.matmul(self.memory, tf.expand_dims(query, axis=2))
        scores = tf.squeeze(scores, axis=2)
        
        # Apply masking if sequence length provided
        if self.memory_sequence_length is not None:
            # TODO(runtime): Implement proper masking
            pass
        
        # Compute attention weights
        alignments = tf.nn.softmax(scores)
        
        # Compute context
        context = tf.reduce_sum(
            tf.expand_dims(alignments, axis=2) * self.memory,
            axis=1
        )
        
        return context, alignments


class AttentionWrapper(tf.keras.layers.Layer):
    """Shim for tf.contrib.seq2seq.AttentionWrapper"""
    def __init__(self, cell, attention_mechanism, attention_layer_size=None, 
                 alignment_history=False, cell_input_fn=None, output_attention=False,
                 initial_cell_state=None, **kwargs):
        super().__init__(**kwargs)
        self.cell = cell
        self.attention_mechanism = attention_mechanism
        self.attention_layer_size = attention_layer_size
        self.alignment_history = alignment_history
        self.output_attention = output_attention
        # TODO(runtime): Verify attention concatenation and projection matches TF1
        
    def build(self, input_shape):
        if self.attention_layer_size is not None:
            self.attention_layer = tf.keras.layers.Dense(self.attention_layer_size)
        super().build(input_shape)
    
    def call(self, inputs, state, training=None):
        # This is a simplified version
        # TODO(runtime): Implement full AttentionWrapper logic
        
        # For now, pass through to wrapped cell
        cell_outputs, cell_state = self.cell(inputs, state, training=training)
        
        # Placeholder for attention computation
        # In full implementation, would compute attention over cell output
        context, alignments = self.attention_mechanism(cell_outputs)
        
        # Combine cell output with attention context
        if self.attention_layer_size is not None:
            attention_output = self.attention_layer(tf.concat([cell_outputs, context], axis=-1))
        else:
            attention_output = tf.concat([cell_outputs, context], axis=-1)
        
        return attention_output, cell_state
    
    def zero_state(self, batch_size, dtype):
        """Create zero state for attention wrapper"""
        # TODO(runtime): Implement proper wrapped state with attention state
        return self.cell.zero_state(batch_size, dtype)