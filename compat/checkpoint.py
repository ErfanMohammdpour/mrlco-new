"""
Compatibility layer for checkpoint handling
Provides functions to bridge between joblib and TF2 checkpoints
"""
import os
import joblib
import tensorflow as tf


def save_variables_joblib(variables, save_path, sess=None):
    """Save variables in joblib format for backward compatibility
    
    Args:
        variables: List of tf.Variable objects
        save_path: Path to save the checkpoint
        sess: Not used in TF2 (kept for API compatibility)
    """
    # TODO(runtime): Verify variable values are correctly extracted in eager mode
    save_dict = {}
    
    for var in variables:
        # In TF2 eager mode, we can directly access variable values
        save_dict[var.name] = var.numpy()
    
    # Create directory if needed
    dirname = os.path.dirname(save_path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    
    joblib.dump(save_dict, save_path)


def load_variables_joblib(variables, load_path, sess=None):
    """Load variables from joblib format
    
    Args:
        variables: List of tf.Variable objects to load into
        load_path: Path to load the checkpoint from
        sess: Not used in TF2 (kept for API compatibility)
    """
    # TODO(runtime): Verify variable assignment works correctly in eager mode
    loaded_params = joblib.load(os.path.expanduser(load_path))
    
    if isinstance(loaded_params, list):
        # Legacy format: list of values in same order as variables
        assert len(loaded_params) == len(variables), \
            'number of variables loaded mismatches len(variables)'
        for var, value in zip(variables, loaded_params):
            var.assign(value)
    else:
        # Dict format: map by variable name
        for var in variables:
            if var.name in loaded_params:
                var.assign(loaded_params[var.name])
            else:
                # Try without the :0 suffix that TF sometimes adds
                name_without_suffix = var.name.rstrip(':0')
                if name_without_suffix in loaded_params:
                    var.assign(loaded_params[name_without_suffix])
                else:
                    print(f"Warning: Variable {var.name} not found in checkpoint")


def create_tf2_checkpoint(model, optimizer=None):
    """Create a TF2 checkpoint object
    
    Args:
        model: Keras model or object with variables
        optimizer: Optional optimizer to include in checkpoint
    
    Returns:
        tf.train.Checkpoint object
    """
    if optimizer is not None:
        checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
    else:
        checkpoint = tf.train.Checkpoint(model=model)
    
    return checkpoint


def save_tf2_checkpoint(checkpoint, save_path):
    """Save TF2 checkpoint
    
    Args:
        checkpoint: tf.train.Checkpoint object
        save_path: Path prefix for checkpoint files
    """
    # TODO(runtime): Verify checkpoint saving in production environment
    checkpoint.save(save_path)


def load_tf2_checkpoint(checkpoint, load_path):
    """Load TF2 checkpoint
    
    Args:
        checkpoint: tf.train.Checkpoint object
        load_path: Path prefix for checkpoint files
    
    Returns:
        Status object from checkpoint restoration
    """
    # TODO(runtime): Verify checkpoint restoration and handle missing variables
    status = checkpoint.restore(load_path)
    return status


def convert_joblib_to_tf2(joblib_path, model, save_path):
    """Convert joblib checkpoint to TF2 format
    
    Args:
        joblib_path: Path to joblib checkpoint
        model: Keras model to load weights into
        save_path: Path to save TF2 checkpoint
    """
    # Load joblib checkpoint
    variables = model.variables
    load_variables_joblib(variables, joblib_path)
    
    # Create and save TF2 checkpoint
    checkpoint = create_tf2_checkpoint(model)
    save_tf2_checkpoint(checkpoint, save_path)


def map_variable_names(old_names, new_variables):
    """Map old variable names to new tf.Variable objects
    
    Args:
        old_names: List of old variable names from TF1
        new_variables: List of new tf.Variable objects from TF2 model
    
    Returns:
        Dict mapping old names to new variables
    """
    # TODO(runtime): Implement smart name mapping heuristics
    name_map = {}
    
    # Simple mapping by stripping prefixes and matching substrings
    for old_name in old_names:
        # Remove common TF1 prefixes
        clean_name = old_name
        for prefix in ['core_policy/', 'task_0_policy/', 'encoder/', 'decoder/']:
            if clean_name.startswith(prefix):
                clean_name = clean_name[len(prefix):]
        
        # Find best match in new variables
        for new_var in new_variables:
            new_name = new_var.name
            # Strip TF2 suffixes
            new_name_clean = new_name.rstrip(':0')
            
            if clean_name in new_name_clean or new_name_clean.endswith(clean_name):
                name_map[old_name] = new_var
                break
    
    return name_map