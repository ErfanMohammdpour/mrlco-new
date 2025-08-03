"""
Compatibility shims for tf.contrib.rnn APIs
Provides Keras-based replacements for RNN cells
"""
import tensorflow as tf


class BasicLSTMCell(tf.keras.layers.Layer):
    """Shim for tf.contrib.rnn.BasicLSTMCell"""
    
    def __init__(self, num_units, forget_bias=1.0, state_is_tuple=True, 
                 activation=None, reuse=None, name=None, dtype=None, **kwargs):
        super().__init__(name=name, dtype=dtype)
        self.num_units = num_units
        self._forget_bias = forget_bias
        self._state_is_tuple = state_is_tuple
        self._activation = activation or 'tanh'
        
        # Create the underlying Keras LSTMCell
        self._keras_cell = tf.keras.layers.LSTMCell(
            units=num_units,
            activation=self._activation,
            recurrent_activation='sigmoid',
            use_bias=True,
            kernel_initializer='glorot_uniform',
            recurrent_initializer='orthogonal',
            bias_initializer='zeros',
            unit_forget_bias=True,
            dropout=0.0,
            recurrent_dropout=0.0,
            **kwargs
        )
        
    @property
    def state_size(self):
        if self._state_is_tuple:
            return (self.num_units, self.num_units)  # (c, h)
        else:
            return self.num_units * 2
            
    @property 
    def output_size(self):
        return self.num_units
        
    def call(self, inputs, states, training=None):
        """Call the underlying Keras cell"""
        if self._state_is_tuple:
            # states is (c, h) tuple for LSTM
            output, new_state = self._keras_cell(inputs, states, training=training)
        else:
            # Convert concatenated state to tuple
            c, h = tf.split(states, 2, axis=-1)
            output, (new_c, new_h) = self._keras_cell(inputs, (c, h), training=training)
            new_state = tf.concat([new_c, new_h], axis=-1)
        return output, new_state
    
    def __call__(self, inputs, state, scope=None, training=None):
        """Backward compatibility for function-style calls"""
        return self.call(inputs, state, training=training)
    
    def zero_state(self, batch_size, dtype):
        """Create zero state for LSTM"""
        if self._state_is_tuple:
            return (
                tf.zeros([batch_size, self.num_units], dtype=dtype),  # c
                tf.zeros([batch_size, self.num_units], dtype=dtype)   # h
            )
        else:
            return tf.zeros([batch_size, self.num_units * 2], dtype=dtype)
        
    @property
    def trainable_variables(self):
        return self._keras_cell.trainable_variables
        
    @property
    def variables(self):
        return self._keras_cell.variables


class GRUCell(tf.keras.layers.Layer):
    """Shim for tf.contrib.rnn.GRUCell"""
    
    def __init__(self, num_units, activation=None, reuse=None, 
                 kernel_initializer=None, bias_initializer=None, 
                 name=None, dtype=None, **kwargs):
        super().__init__(name=name, dtype=dtype)
        self.num_units = num_units
        self._activation = activation or 'tanh'
        
        # Create the underlying Keras GRUCell
        self._keras_cell = tf.keras.layers.GRUCell(
            units=num_units,
            activation=self._activation,
            recurrent_activation='sigmoid',
            use_bias=True,
            kernel_initializer=kernel_initializer or 'glorot_uniform',
            recurrent_initializer='orthogonal',
            bias_initializer=bias_initializer or 'zeros',
            dropout=0.0,
            recurrent_dropout=0.0,
            reset_after=True,  # TF1 GRU behavior
            **kwargs
        )
        
    @property
    def state_size(self):
        return self.num_units
    
    @property
    def output_size(self):
        return self.num_units
        
    def call(self, inputs, states, training=None):
        """Call the underlying Keras cell"""
        return self._keras_cell(inputs, states, training=training)
    
    def __call__(self, inputs, state, scope=None, training=None):
        """Backward compatibility for function-style calls"""
        return self.call(inputs, state, training=training)
    
    def zero_state(self, batch_size, dtype):
        """Create zero state for GRU"""
        return tf.zeros([batch_size, self.num_units], dtype=dtype)
        
    @property
    def trainable_variables(self):
        return self._keras_cell.trainable_variables
        
    @property
    def variables(self):
        return self._keras_cell.variables


class LayerNormBasicLSTMCell(tf.keras.layers.LSTMCell):
    """Shim for tf.contrib.rnn.LayerNormBasicLSTMCell"""
    def __init__(self, num_units, forget_bias=1.0, activation=None,
                 layer_norm=True, norm_gain=1.0, norm_shift=0.0,
                 dropout_keep_prob=1.0, dropout_prob_seed=None,
                 reuse=None, name=None, **kwargs):
        super().__init__(
            units=num_units,
            activation=activation or 'tanh',
            name=name,
            **kwargs
        )
        self.num_units = num_units
        self._layer_norm = layer_norm
        self._norm_gain = norm_gain
        self._norm_shift = norm_shift
        # TODO(runtime): Implement layer normalization within LSTM cell
        
        if layer_norm:
            self.layer_norm_layers = {
                'input': tf.keras.layers.LayerNormalization(
                    gamma_initializer=tf.constant_initializer(norm_gain),
                    beta_initializer=tf.constant_initializer(norm_shift)
                ),
                'hidden': tf.keras.layers.LayerNormalization(
                    gamma_initializer=tf.constant_initializer(norm_gain),
                    beta_initializer=tf.constant_initializer(norm_shift)
                ),
                'cell': tf.keras.layers.LayerNormalization(
                    gamma_initializer=tf.constant_initializer(norm_gain),
                    beta_initializer=tf.constant_initializer(norm_shift)
                )
            }


class NASCell(tf.keras.layers.Layer):
    """Shim for tf.contrib.rnn.NASCell - Neural Architecture Search Cell"""
    def __init__(self, num_units, num_proj=None, use_bias=True,
                 reuse=None, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_units = num_units
        self.num_proj = num_proj
        self.use_bias = use_bias
        # TODO(runtime): Implement full NAS cell architecture
        # For now, using LSTM as placeholder
        self._cell = tf.keras.layers.LSTMCell(num_units)
    
    def call(self, inputs, states, training=None):
        return self._cell(inputs, states, training=training)
    
    @property
    def state_size(self):
        return self._cell.state_size
    
    @property
    def output_size(self):
        return self.num_proj if self.num_proj else self.num_units


class DropoutWrapper(tf.keras.layers.Layer):
    """Shim for tf.contrib.rnn.DropoutWrapper"""
    def __init__(self, cell, input_keep_prob=1.0, output_keep_prob=1.0,
                 state_keep_prob=1.0, variational_recurrent=False,
                 input_size=None, dtype=None, seed=None, **kwargs):
        super().__init__(**kwargs)
        self.cell = cell
        self.input_keep_prob = input_keep_prob
        self.output_keep_prob = output_keep_prob
        self.state_keep_prob = state_keep_prob
        # TODO(runtime): Implement variational recurrent dropout
        
    def call(self, inputs, states, training=None):
        # Apply dropout to inputs
        if training and self.input_keep_prob < 1.0:
            inputs = tf.nn.dropout(inputs, rate=1.0 - self.input_keep_prob)
        
        # Call wrapped cell
        output, new_states = self.cell(inputs, states, training=training)
        
        # Apply dropout to outputs
        if training and self.output_keep_prob < 1.0:
            output = tf.nn.dropout(output, rate=1.0 - self.output_keep_prob)
        
        # TODO(runtime): Apply state dropout if needed
        
        return output, new_states
    
    @property
    def state_size(self):
        return self.cell.state_size
    
    @property
    def output_size(self):
        return self.cell.output_size
    
    def zero_state(self, batch_size, dtype):
        return self.cell.zero_state(batch_size, dtype)


class ResidualWrapper(tf.keras.layers.Layer):
    """Shim for tf.contrib.rnn.ResidualWrapper"""
    def __init__(self, cell, residual_fn=None, **kwargs):
        super().__init__(**kwargs)
        self.cell = cell
        self.residual_fn = residual_fn
        
    def call(self, inputs, states, training=None):
        # Call wrapped cell
        output, new_states = self.cell(inputs, states, training=training)
        
        # Apply residual connection
        if self.residual_fn is not None:
            output = self.residual_fn(inputs, output)
        else:
            # Default residual connection: input + output
            output = inputs + output
        
        return output, new_states
    
    @property
    def state_size(self):
        return self.cell.state_size
    
    @property
    def output_size(self):
        return self.cell.output_size
    
    def zero_state(self, batch_size, dtype):
        return self.cell.zero_state(batch_size, dtype)


class MultiRNNCell(tf.keras.layers.Layer):
    """Shim for tf.contrib.rnn.MultiRNNCell"""
    def __init__(self, cells, state_is_tuple=True, **kwargs):
        super().__init__(**kwargs)
        self.cells = cells
        self._state_is_tuple = state_is_tuple
        # TODO(runtime): Verify state tuple handling matches TF1
        
    def call(self, inputs, states, training=None):
        new_states = []
        
        for i, cell in enumerate(self.cells):
            if self._state_is_tuple:
                state = states[i]
            else:
                # Handle non-tuple state
                state = states
            
            inputs, state = cell(inputs, state, training=training)
            new_states.append(state)
        
        if self._state_is_tuple:
            new_states = tuple(new_states)
        
        return inputs, new_states
    
    @property
    def state_size(self):
        if self._state_is_tuple:
            return tuple(cell.state_size for cell in self.cells)
        else:
            # Concatenated state size
            return sum(cell.state_size for cell in self.cells)
    
    @property
    def output_size(self):
        return self.cells[-1].output_size
    
    def zero_state(self, batch_size, dtype):
        if self._state_is_tuple:
            return tuple(cell.zero_state(batch_size, dtype) for cell in self.cells)
        else:
            # TODO(runtime): Handle concatenated state
            return tf.concat([cell.zero_state(batch_size, dtype) for cell in self.cells], axis=-1)