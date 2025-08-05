"""Simple GPU detection and memory growth setup for TensorFlow 2.x"""
import os
import tensorflow as tf


def setup_gpu_memory_growth():
    """Enable memory growth for all visible GPUs"""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
                print(f"Memory growth enabled for: {gpu.name}")
            except RuntimeError as e:
                print(f"Could not set memory growth for {gpu.name}: {e}")
    return gpus


def print_device_info():
    """Print current device configuration"""
    print(f"\n=== Device Configuration ===")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"Found {len(gpus)} GPU(s):")
        for i, gpu in enumerate(gpus):
            print(f"  [{i}] {gpu.name}")
    else:
        print("No GPUs found. Using CPU.")
    
    print("===========================\n")
    return len(gpus)


# Stub functions for backward compatibility
def setup_gpu_and_strategy(tf1_compatibility_mode=False):
    """Deprecated - kept for compatibility"""
    print("WARNING: setup_gpu_and_strategy is deprecated. Device selection is automatic.")
    setup_gpu_memory_growth()
    return None, {'num_gpus': len(tf.config.list_physical_devices('GPU'))}


def log_tensor_device(tensor, tensor_name="tensor", step=None):
    """Log tensor device placement"""
    if hasattr(tensor, 'device'):
        device = tensor.device
        if step is not None:
            print(f"[Step {step}] {tensor_name} device: {device}")
        else:
            print(f"{tensor_name} device: {device}")
    return device


def ensure_tensor_conversion(data):
    """Convert data to TensorFlow tensor"""
    if not isinstance(data, tf.Tensor):
        return tf.convert_to_tensor(data)
    return data