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
    """Full implementation of tf.contrib.seq2seq.dynamic_decode for TF2
    
    Returns: (final_outputs, final_state, final_sequence_lengths)
    """
    # Initialize decoder
    finished, inputs, state = decoder.initialize()
    
    # Get batch size and dtype
    batch_size = tf.shape(inputs)[0]
    
    # Use maximum_iterations if provided, otherwise use a default
    if maximum_iterations is None:
        maximum_iterations = tf.constant(100)  # Default max length
    else:
        # Handle case where maximum_iterations is a tensor
        if hasattr(maximum_iterations, 'shape') and len(maximum_iterations.shape) > 0:
            # If it's a vector, take the first element
            maximum_iterations = maximum_iterations[0]
    
    # Initialize loop variables
    time = tf.constant(0)
    outputs_ta = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    sample_ids_ta = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
    
    def condition(time, finished, inputs, state, outputs_ta, sample_ids_ta):
        return tf.logical_and(
            tf.logical_not(tf.reduce_all(finished)),
            tf.less(time, maximum_iterations)
        )
    
    def body(time, finished, inputs, state, outputs_ta, sample_ids_ta):
        # Run decoder cell
        if hasattr(decoder.cell, '__call__'):
            cell_outputs, cell_state = decoder.cell(inputs, state)
        else:
            # Handle wrapped cells
            cell_outputs, cell_state = decoder.cell(inputs, state)
        
        # Apply output layer if present
        if decoder.output_layer is not None:
            cell_outputs = decoder.output_layer(cell_outputs)
        
        # Sample from helper
        sample_ids = decoder.helper.sample(time, cell_outputs, cell_state)
        
        # Get next inputs
        finished, next_inputs, next_state = decoder.helper.next_inputs(
            time, cell_outputs, cell_state, sample_ids
        )
        
        # Write outputs
        outputs_ta = outputs_ta.write(time, cell_outputs)
        sample_ids_ta = sample_ids_ta.write(time, sample_ids)
        
        return time + 1, finished, next_inputs, next_state, outputs_ta, sample_ids_ta
    
    # Run the decoding loop
    final_time, final_finished, _, final_state, final_outputs_ta, final_sample_ids_ta = tf.while_loop(
        condition,
        body,
        loop_vars=[time, finished, inputs, state, outputs_ta, sample_ids_ta],
        parallel_iterations=parallel_iterations,
        swap_memory=swap_memory
    )
    
    # Stack outputs
    final_outputs_tensor = final_outputs_ta.stack()
    final_sample_ids_tensor = final_sample_ids_ta.stack()
    
    # Transpose if needed
    if not output_time_major:
        # Convert from [time, batch, ...] to [batch, time, ...]
        final_outputs_tensor = tf.transpose(final_outputs_tensor, [1, 0, 2])
        final_sample_ids_tensor = tf.transpose(final_sample_ids_tensor, [1, 0])
    
    # Create final outputs
    final_outputs = BasicDecoderOutput(
        rnn_output=final_outputs_tensor,
        sample_id=final_sample_ids_tensor
    )
    
    # Compute sequence lengths
    # Count the number of time steps before the first finished flag
    if output_time_major:
        # final_sample_ids_tensor is [time, batch]
        # Create mask for end tokens
        not_finished = tf.not_equal(final_sample_ids_tensor, decoder.helper.end_token)
        # Sum along time axis
        final_sequence_lengths = tf.reduce_sum(tf.cast(not_finished, tf.int32), axis=0)
    else:
        # final_sample_ids_tensor is [batch, time]
        # Handle different helper types - some don't have end_token
        if hasattr(decoder.helper, 'end_token'):
            not_finished = tf.not_equal(final_sample_ids_tensor, decoder.helper.end_token)
        else:
            # For TrainingHelper and others without end_token, assume no early termination
            not_finished = tf.ones_like(final_sample_ids_tensor, dtype=tf.bool)
        final_sequence_lengths = tf.reduce_sum(tf.cast(not_finished, tf.int32), axis=1)
    
    # Ensure minimum length of 1
    final_sequence_lengths = tf.maximum(final_sequence_lengths, 1)
    
    return final_outputs, final_state, final_sequence_lengths


class LuongAttention:
    """Simple shim for tf.contrib.seq2seq.LuongAttention"""
    def __init__(self, num_units, memory, memory_sequence_length=None, scale=False):
        self.num_units = num_units
        self.memory = memory
        self.memory_sequence_length = memory_sequence_length
        self.scale = scale
    
    def __call__(self, query, state=None):
        # Simplified Luong attention
        # query: [batch, num_units]
        # memory: [batch, time, num_units]
        
        # Compute attention scores
        scores = tf.matmul(self.memory, tf.expand_dims(query, axis=2))
        scores = tf.squeeze(scores, axis=2)
        
        # Apply masking if sequence length provided
        if self.memory_sequence_length is not None:
            # Create mask for sequence lengths
            mask = tf.sequence_mask(self.memory_sequence_length, tf.shape(self.memory)[1])
            scores = tf.where(mask, scores, tf.fill(tf.shape(scores), -float('inf')))
        
        # Compute attention weights
        alignments = tf.nn.softmax(scores)
        
        # Compute context
        context = tf.reduce_sum(
            tf.expand_dims(alignments, axis=2) * self.memory,
            axis=1
        )
        
        return context, alignments


class AttentionWrapper:
    """Simple shim for tf.contrib.seq2seq.AttentionWrapper"""
    def __init__(self, cell, attention_mechanism, attention_layer_size=None, 
                 alignment_history=False, cell_input_fn=None, output_attention=False,
                 initial_cell_state=None):
        self.cell = cell
        self.attention_mechanism = attention_mechanism
        self.attention_layer_size = attention_layer_size
        self.alignment_history = alignment_history
        self.output_attention = output_attention
        
        # Create attention projection layer if needed
        if attention_layer_size is not None:
            self.attention_layer = tf.keras.layers.Dense(attention_layer_size)
        else:
            self.attention_layer = None
    
    def __call__(self, inputs, state, training=None):
        # Run cell forward pass
        cell_outputs, cell_state = self.cell(inputs, state)
        
        # Compute attention
        context, alignments = self.attention_mechanism(cell_outputs)
        
        # Combine cell output with attention context
        if self.attention_layer is not None:
            attention_output = self.attention_layer(tf.concat([cell_outputs, context], axis=-1))
        else:
            attention_output = tf.concat([cell_outputs, context], axis=-1)
        
        return attention_output, cell_state
    
    def zero_state(self, batch_size, dtype):
        """Create zero state for attention wrapper"""
        # Return a simple wrapper around the cell's zero state
        class AttentionWrapperState:
            def __init__(self, cell_state):
                self.cell_state = cell_state
            
            def clone(self, cell_state=None):
                """Clone method expected by tf.contrib patterns"""
                if cell_state is not None:
                    return AttentionWrapperState(cell_state)
                return AttentionWrapperState(self.cell_state)
        
        # Get cell's zero state
        if hasattr(self.cell, 'zero_state'):
            cell_state = self.cell.zero_state(batch_size, dtype)
        else:
            # For Keras cells, create zero state manually
            state_size = getattr(self.cell, 'state_size', self.cell.units)
            if isinstance(state_size, (list, tuple)):
                # LSTM has tuple state (c, h)
                cell_state = tuple(tf.zeros([batch_size, size], dtype=dtype) for size in state_size)
            else:
                cell_state = tf.zeros([batch_size, state_size], dtype=dtype)
        
        return AttentionWrapperState(cell_state)