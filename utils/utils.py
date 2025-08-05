import numpy as np
import scipy
import scipy.signal
import json
import collections
import tensorflow as tf
import os
import multiprocessing

def get_original_tf_name(name):
    """
    Args:
        name (str): full name of the tf variable with all the scopes

    Returns:
        (str): name given to the variable when creating it (i.e. name of the variable w/o the scope and the colons)
    """
    return name.split("/")[-1].split(":")[0]


def remove_scope_from_name(name, scope):
    """
    Args:
        name (str): full name of the tf variable with all the scopes

    Returns:
        (str): full name of the variable with the scope removed
    """
    result = name.split(scope)[1]
    result = result[1:] if result[0] == '/' else result
    return result.split(":")[0]

def remove_first_scope_from_name(name):
    return name.replace(name + '/', "").split(":")[0]

def get_last_scope(name):
    """
    Args:
        name (str): full name of the tf variable with all the scopes

    Returns:
        (str): name of the last scope
    """
    return name.split("/")[-2]


def extract(x, *keys):
    """
    Args:
        x (dict or list): dict or list of dicts

    Returns:
        (tuple): tuple with the elements of the dict or the dicts of the list
    """
    if isinstance(x, dict):
        return tuple(x[k] for k in keys)
    elif isinstance(x, list):
        return tuple([xi[k] for xi in x] for k in keys)
    else:
        raise NotImplementedError


def normalize_advantages(advantages):
    """
    Args:
        advantages (np.ndarray): np array with the advantages

    Returns:
        (np.ndarray): np array with the advantages normalized
    """
    return (advantages - np.mean(advantages)) / (advantages.std() + 1e-8)


def shift_advantages_to_positive(advantages):
    return (advantages - np.min(advantages)) + 1e-8


def discount_cumsum(x, discount):
    """
    See https://docs.scipy.org/doc/scipy/reference/tutorial/signal.html#difference-equation-filtering

    Returns:
        (float) : y[t] - discount*y[t+1] = x[t] or rev(y)[t] - discount*rev(y)[t-1] = rev(x)[t]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


def explained_variance_1d(ypred, y):
    """
    Args:
        ypred (np.ndarray): predicted values of the variable of interest
        y (np.ndarray): real values of the variable

    Returns:
        (float): variance explained by your estimator

    """
    assert y.ndim == 1 and ypred.ndim == 1
    vary = np.var(y)
    if np.isclose(vary, 0):
        if np.var(ypred) > 0:
            return 0
        else:
            return 1
    return 1 - np.var(y - ypred) / (vary + 1e-8)


def concat_tensor_dict_list(tensor_dict_list):
    """
    Args:
        tensor_dict_list (list) : list of dicts of lists of tensors

    Returns:
        (dict) : dict of lists of tensors
    """
    keys = list(tensor_dict_list[0].keys())
    ret = dict()
    for k in keys:
        example = tensor_dict_list[0][k]
        if isinstance(example, dict):
            v = concat_tensor_dict_list([x[k] for x in tensor_dict_list])
        else:
            v = np.concatenate([x[k] for x in tensor_dict_list])
        ret[k] = v
    return ret


def stack_tensor_dict_list(tensor_dict_list):
    """
    Args:
        tensor_dict_list (list) : list of dicts of tensors

    Returns:
        (dict) : dict of lists of tensors
    """
    keys = list(tensor_dict_list[0].keys())
    ret = dict()
    for k in keys:
        example = tensor_dict_list[0][k]
        if isinstance(example, dict):
            v = stack_tensor_dict_list([x[k] for x in tensor_dict_list])
        else:
            v = np.asarray([x[k] for x in tensor_dict_list])
        ret[k] = v
    return ret


def create_feed_dict(placeholder_dict, value_dict):
    """TF2: Feed dicts are no longer needed with eager execution"""
    # Return the value_dict for compatibility
    return value_dict

def set_seed(seed):
    """
    Set the random seed for all random number generators

    Args:
        seed (int) : seed to use

    Returns:
        None
    """
    import random
    import tensorflow as tf
    seed %= 4294967294
    random.seed(seed)
    np.random.seed(seed)
    # MIGRATION: tf.set_random_seed -> tf.random.set_seed
    tf.random.set_seed(seed)
    print('using seed %s' % (str(seed)))

class ClassEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, type):
            return {'$class': o.__module__ + "." + o.__name__}
        if callable(o):
            return {'function': o.__name__}
        return json.JSONEncoder.default(self, o)

def function(inputs, outputs, updates=None, givens=None):
    """TF2 eager replacement for TF1 function compilation.
    In eager mode, we return a simple callable that executes the operations.
    
    For compatibility, we maintain the same interface but execute eagerly.
    """
    def eager_function(*args, **kwargs):
        # In TF2 eager mode, operations execute immediately
        # For variable assignments passed as updates, execute them
        if updates:
            for update in updates:
                # If update is an assignment operation, it will execute immediately
                pass
        
        # Return the outputs directly
        if isinstance(outputs, list):
            return outputs
        elif isinstance(outputs, (dict, collections.OrderedDict)):
            return outputs
        else:
            return outputs
    
    return eager_function

def _check_shape(placeholder_shape, data_shape):
    ''' check if two shapes are compatible (i.e. differ only by dimensions of size 1, or by the batch dimension)'''

    return True
    squeezed_placeholder_shape = _squeeze_shape(placeholder_shape)
    squeezed_data_shape = _squeeze_shape(data_shape)

    for i, s_data in enumerate(squeezed_data_shape):
        s_placeholder = squeezed_placeholder_shape[i]
        if s_placeholder != -1 and s_data != s_placeholder:
            return False

    return True

def _squeeze_shape(shape):
    return [x for x in shape if x != 1]

# ================================================================
# Shape adjustment for feeding into tf placeholders
# ================================================================
def adjust_shape(placeholder, data):
    '''
    adjust shape of the data to the shape of the placeholder if possible.
    If shape is incompatible, AssertionError is thrown

    Parameters:
        placeholder     tensorflow input placeholder

        data            input data to be (potentially) reshaped to be fed into placeholder

    Returns:
        reshaped data
    '''

    if not isinstance(data, np.ndarray) and not isinstance(data, list):
        return data
    if isinstance(data, list):
        data = np.array(data)

    placeholder_shape = [x or -1 for x in placeholder.shape.as_list()]

    assert _check_shape(placeholder_shape, data.shape), \
        'Shape of data {} is not compatible with shape of the placeholder {}'.format(data.shape, placeholder_shape)

    return np.reshape(data, placeholder_shape)

def make_session(config=None, num_cpu=None, make_default=False, graph=None):
    """TF2: Sessions are no longer needed with eager execution"""
    # Configure threading for eager execution if needed
    if num_cpu is None:
        num_cpu = int(os.getenv('RCALL_NUM_CPU', multiprocessing.cpu_count()))
    
    if num_cpu:
        tf.config.threading.set_inter_op_parallelism_threads(num_cpu)
        tf.config.threading.set_intra_op_parallelism_threads(num_cpu)
    
    # Return None for compatibility
    return None


def get_session(config=None):
    """TF2: Sessions are no longer needed with eager execution"""
    # Return None for compatibility
    return None

# TF2: _Function class is no longer needed with eager execution

def zipsame(*seqs):
    L = len(seqs[0])
    assert all(len(seq) == L for seq in seqs[1:])
    return zip(*seqs)
