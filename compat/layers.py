"""
Compatibility layer for tf.contrib.layers functions in TensorFlow 2.x
"""
import tensorflow as tf


def fully_connected(inputs, num_outputs, activation_fn=None, scope=None, reuse=None):
    """
    Replacement for tf.contrib.layers.fully_connected
    Creates a fully connected layer with proper variable scoping
    """
    with tf.compat.v1.variable_scope(scope, default_name='fully_connected', reuse=reuse):
        # Get input shape
        input_shape = inputs.get_shape()
        if input_shape.ndims is None:
            raise ValueError("Input shape must be known")
        
        # Create weight variable
        input_size = input_shape[-1].value if hasattr(input_shape[-1], 'value') else input_shape[-1]
        w = tf.compat.v1.get_variable(
            "weights",
            shape=[input_size, num_outputs],
            initializer=tf.compat.v1.keras.initializers.GlorotUniform()
        )
        
        # Create bias variable
        b = tf.compat.v1.get_variable(
            "biases",
            shape=[num_outputs],
            initializer=tf.compat.v1.zeros_initializer()
        )
        
        # Compute output
        output = tf.matmul(inputs, w) + b
        
        # Apply activation if specified
        if activation_fn is not None:
            output = activation_fn(output)
            
        return output