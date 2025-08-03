"""GPU detection and automatic strategy selection for TensorFlow 2.x"""
import os
import tensorflow as tf


def setup_gpu_and_strategy(tf1_compatibility_mode=False):
    """
    Automatically detect and configure GPU/CPU strategy for TensorFlow.
    
    Args:
        tf1_compatibility_mode: If True, only use single GPU strategies (no MirroredStrategy)
                               for compatibility with TF1-style code
    
    Returns:
        tuple: (strategy, device_info_dict) where:
            - strategy: tf.distribute.Strategy instance to use
            - device_info_dict: Dictionary with device configuration details
    """
    # Get environment info
    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')
    tf_version = tf.__version__
    
    # Get visible GPUs
    physical_gpus = tf.config.list_physical_devices('GPU')
    num_gpus = len(physical_gpus)
    
    # Log initial device information
    print(f"\n========== TensorFlow Device Configuration ==========")
    print(f"TensorFlow version: {tf_version}")
    print(f"CUDA_VISIBLE_DEVICES: {cuda_visible_devices}")
    print(f"Visible GPUs: {num_gpus}")
    
    if num_gpus > 0:
        print(f"GPU devices found:")
        for i, gpu in enumerate(physical_gpus):
            print(f"  [{i}] {gpu.name}")
    
    # Enable memory growth for all GPUs to avoid pre-allocating all GPU memory
    if num_gpus > 0:
        for gpu in physical_gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
                print(f"Memory growth enabled for: {gpu.name}")
            except RuntimeError as e:
                print(f"Could not set memory growth for {gpu.name}: {e}")
    
    # Select strategy based on available devices
    if num_gpus >= 2 and not tf1_compatibility_mode:
        strategy = tf.distribute.MirroredStrategy()
        strategy_name = "MirroredStrategy"
        selected_devices = [f"/GPU:{i}" for i in range(num_gpus)]
        print(f"\nUsing MirroredStrategy with {num_gpus} GPUs")
    elif num_gpus >= 2 and tf1_compatibility_mode:
        # For TF1 compatibility, use only the first GPU
        strategy = tf.distribute.OneDeviceStrategy("/GPU:0")
        strategy_name = "OneDeviceStrategy"
        selected_devices = ["/GPU:0"]
        print(f"\nTF1 compatibility mode: Using single GPU (GPU:0) out of {num_gpus} available GPUs")
    elif num_gpus == 1:
        strategy = tf.distribute.OneDeviceStrategy("/GPU:0")
        strategy_name = "OneDeviceStrategy"
        selected_devices = ["/GPU:0"]
        print(f"\nUsing OneDeviceStrategy with single GPU")
    else:
        # CPU fallback
        strategy = tf.distribute.OneDeviceStrategy("/CPU:0")
        strategy_name = "OneDeviceStrategy"
        selected_devices = ["/CPU:0"]
        print(f"\nNo GPUs found. Using CPU fallback")
    
    print(f"Selected strategy: {strategy_name}")
    print(f"Devices in use: {selected_devices}")
    
    # Perform warmup operation to verify device placement
    print(f"\nPerforming warmup operation...")
    with strategy.scope():
        # Small matmul operation for warmup
        warmup_a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        warmup_b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
        warmup_result = tf.matmul(warmup_a, warmup_b)
        
        # Get device placement
        if hasattr(warmup_result, 'device'):
            device_placement = warmup_result.device
        else:
            # For distributed strategies, check the actual device
            with tf.device(selected_devices[0]):
                test_tensor = tf.constant(1.0)
                device_placement = test_tensor.device
        
        print(f"Warmup operation completed on device: {device_placement}")
    
    print(f"====================================================\n")
    
    # Return strategy and device info
    device_info = {
        'tf_version': tf_version,
        'cuda_visible_devices': cuda_visible_devices,
        'num_gpus': num_gpus,
        'physical_gpus': physical_gpus,
        'strategy_name': strategy_name,
        'selected_devices': selected_devices,
        'device_placement': device_placement
    }
    
    return strategy, device_info


def log_tensor_device(tensor, tensor_name="tensor", step=None):
    """
    Log the device placement of a tensor.
    
    Args:
        tensor: TensorFlow tensor to check
        tensor_name: Name of the tensor for logging
        step: Optional step number for logging
    """
    if hasattr(tensor, 'device'):
        device = tensor.device
    else:
        # Handle cases where device attribute might not be directly accessible
        device = "Unknown (distributed tensor)"
    
    if step is not None:
        print(f"[Step {step}] {tensor_name} device: {device}")
    else:
        print(f"{tensor_name} device: {device}")
    
    return device


def ensure_tensor_conversion(data):
    """
    Ensure data is converted to TensorFlow tensor for GPU execution.
    
    Args:
        data: Input data (numpy array, list, or already a tensor)
    
    Returns:
        tf.Tensor: Converted tensor
    """
    if not isinstance(data, tf.Tensor):
        return tf.convert_to_tensor(data)
    return data


def setup_data_pipeline_optimization(dataset):
    """
    Apply GPU-friendly optimizations to a tf.data pipeline.
    
    Args:
        dataset: tf.data.Dataset instance
    
    Returns:
        tf.data.Dataset: Optimized dataset
    """
    # Apply prefetching for better GPU utilization
    return dataset.prefetch(tf.data.AUTOTUNE)


def run_device_diagnostics(strategy, device_info, detailed=True):
    """
    Run comprehensive device diagnostics and warmup
    
    Args:
        strategy: tf.distribute.Strategy instance
        device_info: Device information dictionary
        detailed: Whether to run detailed diagnostics
        
    Returns:
        dict: Diagnostic results
    """
    print("\n========== Running Device Diagnostics ==========")
    results = {}
    
    # Test 1: Basic computation on each device
    print("\n1. Testing basic computation on each device...")
    for device in device_info['selected_devices']:
        with tf.device(device):
            # Matrix multiplication test
            size = 1000
            a = tf.random.normal([size, size])
            b = tf.random.normal([size, size])
            
            # Warmup
            _ = tf.matmul(a, b)
            
            # Timed run
            import time
            start_time = time.time()
            c = tf.matmul(a, b)
            # Force execution
            _ = c.numpy() if tf.executing_eagerly() else tf.compat.v1.get_default_session().run(c)
            end_time = time.time()
            
            elapsed = end_time - start_time
            results[f'{device}_matmul_time'] = elapsed
            print(f"  {device}: {elapsed:.4f} seconds for {size}x{size} matmul")
    
    # Test 2: Memory bandwidth test
    if detailed:
        print("\n2. Testing memory bandwidth...")
        for device in device_info['selected_devices']:
            with tf.device(device):
                # Large tensor copy
                size = 100 * 1024 * 1024  # 100MB
                data = tf.random.normal([size])
                
                start_time = time.time()
                data_copy = tf.identity(data)
                # Force execution
                if tf.executing_eagerly():
                    _ = data_copy.numpy()
                else:
                    sess = tf.compat.v1.get_default_session()
                    if sess:
                        _ = sess.run(data_copy)
                end_time = time.time()
                
                elapsed = end_time - start_time
                bandwidth_gb_s = (size * 4 * 2) / (elapsed * 1e9)  # 4 bytes per float32, 2x for read+write
                results[f'{device}_bandwidth_gb_s'] = bandwidth_gb_s
                print(f"  {device}: {bandwidth_gb_s:.2f} GB/s memory bandwidth")
    
    # Test 3: Multi-device synchronization (if multiple devices)
    if len(device_info['selected_devices']) > 1:
        print("\n3. Testing multi-device synchronization...")
        
        @tf.function
        def distributed_computation():
            # Each device computes a sum
            local_sum = tf.reduce_sum(tf.random.normal([1000, 1000]))
            return local_sum
        
        start_time = time.time()
        if hasattr(strategy, 'run'):
            results_per_replica = strategy.run(distributed_computation)
            total = strategy.reduce(tf.distribute.ReduceOp.SUM, results_per_replica, axis=None)
        else:
            total = distributed_computation()
        
        # Force execution
        if tf.executing_eagerly():
            _ = total.numpy()
        else:
            sess = tf.compat.v1.get_default_session()
            if sess:
                _ = sess.run(total)
        
        end_time = time.time()
        
        sync_time = end_time - start_time
        results['multi_device_sync_time'] = sync_time
        print(f"  Multi-device synchronization: {sync_time:.4f} seconds")
    
    print("\n================================================\n")
    return results


def create_profiling_callback(log_dir="./profiling_logs", profile_batch='10,20'):
    """
    Create a TensorBoard profiling callback for performance analysis
    
    Args:
        log_dir: Directory to save profiling logs
        profile_batch: Batch range to profile (e.g., '10,20' profiles batches 10-20)
        
    Returns:
        tf.keras.callbacks.TensorBoard callback with profiling enabled
    """
    import os
    os.makedirs(log_dir, exist_ok=True)
    
    # Parse profile batch range
    if isinstance(profile_batch, str) and ',' in profile_batch:
        start, end = map(int, profile_batch.split(','))
        profile_batch = (start, end)
    
    callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        profile_batch=profile_batch,
        write_graph=True,
        write_images=False,
        update_freq='batch'
    )
    
    print(f"Profiling callback created. Logs will be saved to: {log_dir}")
    print(f"To view profiling results, run: tensorboard --logdir={log_dir}")
    
    return callback


def get_gpu_memory_info():
    """
    Get current GPU memory usage information
    
    Returns:
        dict: Memory information for each GPU
    """
    gpus = tf.config.list_physical_devices('GPU')
    memory_info = {}
    
    for i, gpu in enumerate(gpus):
        try:
            # This requires nvidia-ml-py
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            
            memory_info[f'GPU:{i}'] = {
                'total_mb': info.total / 1024 / 1024,
                'used_mb': info.used / 1024 / 1024,
                'free_mb': info.free / 1024 / 1024,
                'utilization_percent': (info.used / info.total) * 100
            }
        except ImportError:
            # Fallback to TensorFlow's limited memory info
            memory_info[f'GPU:{i}'] = {
                'status': 'nvidia-ml-py not installed for detailed memory info'
            }
        except Exception as e:
            memory_info[f'GPU:{i}'] = {
                'error': str(e)
            }
    
    return memory_info