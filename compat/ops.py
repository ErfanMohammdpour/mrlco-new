"""
Compatibility shims for TensorFlow operations
Handles control dependencies and other graph-based operations
"""
import tensorflow as tf
from contextlib import contextmanager


@contextmanager
def control_dependencies(control_inputs):
    """Shim for tf.control_dependencies
    
    In TF2 eager mode, operations execute in order automatically.
    This is a no-op context manager for compatibility.
    
    Args:
        control_inputs: List of operations that must complete before
    
    Note: In TF2 with @tf.function, use tf.control_dependencies directly
    """
    # EAGER: In eager mode, ops execute in program order
    yield
    

def get_variable(name, shape=None, dtype=None, initializer=None,
                regularizer=None, trainable=True, collections=None,
                caching_device=None, partitioner=None, validate_shape=True,
                use_resource=None, custom_getter=None, constraint=None,
                synchronization=None, aggregation=None):
    """Shim for tf.get_variable
    
    In TF2, use tf.Variable directly or create variables in keras layers.
    This function provides a compatibility layer.
    """
    
    if initializer is None:
        initializer = tf.keras.initializers.GlorotUniform()
    elif callable(initializer):
        # It's already an initializer function
        pass
    else:
        # Assume it's a value
        initializer = tf.constant_initializer(initializer)
    
    if shape is not None:
        initial_value = initializer(shape, dtype=dtype)
    else:
        initial_value = initializer
    
    return tf.Variable(
        initial_value=initial_value,
        trainable=trainable,
        name=name,
        dtype=dtype,
        constraint=constraint
    )


@contextmanager
def variable_scope(name_or_scope, default_name=None, values=None,
                  initializer=None, regularizer=None, caching_device=None,
                  partitioner=None, custom_getter=None, reuse=None,
                  dtype=None, use_resource=None, constraint=None,
                  auxiliary_name_scope=True):
    """Shim for tf.variable_scope
    
    In TF2, variable scoping is handled by keras layers and name scopes.
    This provides a minimal compatibility layer.
    """
    # EAGER: Variable scoping is handled differently in TF2
    # Variables should be created in layer __init__ or build() methods
    
    # For now, just use name scope for compatibility
    with tf.name_scope(name_or_scope or default_name):
        yield


class VariableScope:
    """Minimal VariableScope object for compatibility"""
    def __init__(self, name):
        self.name = name
    
    def get_variable(self, name, *args, **kwargs):
        return get_variable(f"{self.name}/{name}", *args, **kwargs)


def get_collection(key, scope=None):
    """Shim for tf.get_collection
    
    In TF2, collections are not used. Models track their own variables.
    """
    # In TF2, use model.trainable_variables, model.variables, etc.
    return []


def add_to_collection(key, value):
    """Shim for tf.add_to_collection
    
    In TF2, collections are not used. This is a no-op.
    """
    # EAGER: Collections not used in TF2
    pass


def global_variables():
    """Shim for tf.global_variables
    
    In TF2, there's no global collection of variables.
    """
    return []


def trainable_variables():
    """Shim for tf.trainable_variables
    
    In TF2, use model.trainable_variables instead.
    """
    return []


def moving_average_variables():
    """Shim for tf.moving_average_variables
    
    In TF2, track moving averages explicitly in your model.
    """
    return []


def model_variables():
    """Shim for tf.model_variables
    
    In TF2, use model.variables instead.
    """
    return []


def local_variables():
    """Shim for tf.local_variables
    
    In TF2, there's no distinction of local variables.
    """
    return []


def get_variable_scope():
    """Shim for tf.get_variable_scope
    
    Returns a minimal VariableScope object for compatibility.
    """
    return VariableScope("") 


def convert_to_tensor(value, dtype=None, name=None, preferred_dtype=None):
    """Shim for tf.convert_to_tensor with TF1 signature"""
    return tf.convert_to_tensor(value, dtype=dtype or preferred_dtype, name=name)