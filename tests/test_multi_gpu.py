#!/usr/bin/env python3
"""
Test script to verify multi-GPU execution for the MRLCO project
"""

import os
import tensorflow as tf
import numpy as np
import subprocess
import time


def check_gpu_availability():
    """Check and report GPU availability"""
    print("\n" + "="*60)
    print("GPU AVAILABILITY CHECK")
    print("="*60)
    
    # Check CUDA_VISIBLE_DEVICES
    cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')
    print(f"CUDA_VISIBLE_DEVICES: {cuda_devices}")
    
    # Check TensorFlow GPU support
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")
    
    # List physical devices
    gpus = tf.config.list_physical_devices('GPU')
    print(f"Number of GPUs available: {len(gpus)}")
    
    if gpus:
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu}")
            
        # Check GPU details using nvidia-smi
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.free', 
                                   '--format=csv,noheader'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print("\nGPU Details (from nvidia-smi):")
                for i, line in enumerate(result.stdout.strip().split('\n')):
                    print(f"  GPU {i}: {line}")
        except Exception as e:
            print(f"Could not run nvidia-smi: {e}")
    else:
        print("No GPUs found!")
    
    return len(gpus)


def test_single_gpu_computation():
    """Test computation on a single GPU"""
    print("\n" + "="*60)
    print("SINGLE GPU COMPUTATION TEST")
    print("="*60)
    
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        print("No GPUs available for testing")
        return
    
    # Enable memory growth
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    
    # Test on first GPU
    with tf.device('/GPU:0'):
        print("\nTesting on GPU:0...")
        
        # Matrix multiplication test
        size = 4096
        a = tf.random.normal([size, size])
        b = tf.random.normal([size, size])
        
        # Warmup
        _ = tf.matmul(a, b)
        
        # Timed run
        start = time.time()
        c = tf.matmul(a, b)
        c_numpy = c.numpy()  # Force execution
        end = time.time()
        
        print(f"Matrix multiplication ({size}x{size}) completed in {end-start:.3f} seconds")
        print(f"Result shape: {c_numpy.shape}")
        print(f"Device placement: {c.device}")


def test_multi_gpu_strategy():
    """Test multi-GPU execution with MirroredStrategy"""
    print("\n" + "="*60)
    print("MULTI-GPU STRATEGY TEST")
    print("="*60)
    
    gpus = tf.config.list_physical_devices('GPU')
    if len(gpus) < 2:
        print(f"Need at least 2 GPUs for multi-GPU test, found {len(gpus)}")
        if len(gpus) == 1:
            print("Testing with OneDeviceStrategy instead...")
            strategy = tf.distribute.OneDeviceStrategy("/GPU:0")
        else:
            print("Testing with CPU fallback...")
            strategy = tf.distribute.OneDeviceStrategy("/CPU:0")
    else:
        print(f"Creating MirroredStrategy with {len(gpus)} GPUs...")
        strategy = tf.distribute.MirroredStrategy()
    
    print(f"Number of replicas: {strategy.num_replicas_in_sync}")
    
    # Test distributed computation
    @tf.function
    def distributed_computation():
        # Each replica computes its own random matrix multiplication
        size = 2048
        a = tf.random.normal([size, size])
        b = tf.random.normal([size, size])
        local_result = tf.matmul(a, b)
        return tf.reduce_sum(local_result)
    
    print("\nRunning distributed computation...")
    start = time.time()
    
    # Run on all devices
    per_replica_result = strategy.run(distributed_computation)
    
    # Reduce across replicas
    global_sum = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_result, axis=None)
    
    end = time.time()
    
    print(f"Distributed computation completed in {end-start:.3f} seconds")
    print(f"Global sum: {global_sum.numpy()}")
    
    # Test data distribution
    print("\nTesting data distribution across devices...")
    
    # Create a simple dataset
    dataset = tf.data.Dataset.from_tensor_slices(tf.range(100))
    dataset = dataset.batch(10)
    
    # Distribute the dataset
    dist_dataset = strategy.experimental_distribute_dataset(dataset)
    
    @tf.function
    def process_batch(x):
        # Simple processing - square the values
        return tf.square(x)
    
    print("Processing distributed batches...")
    results = []
    batch_count = 0
    for batch in dist_dataset:
        if batch_count >= 3:
            break
        result = strategy.run(process_batch, args=(batch,))
        # Gather results from all replicas
        gathered = strategy.gather(result, axis=0)
        results.append(gathered.numpy())
        print(f"  Processed batch shape: {gathered.shape}")
        batch_count += 1
    
    print("Data distribution test completed")


def test_memory_usage():
    """Test and report GPU memory usage"""
    print("\n" + "="*60)
    print("GPU MEMORY USAGE TEST")
    print("="*60)
    
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        print("No GPUs available for memory test")
        return
    
    # Get initial memory state
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.free', 
                               '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("Initial GPU memory state:")
            for i, line in enumerate(result.stdout.strip().split('\n')):
                used, free = map(int, line.split(', '))
                total = used + free
                print(f"  GPU {i}: {used}MB used, {free}MB free (total: {total}MB)")
                
            # Allocate some tensors
            print("\nAllocating large tensors...")
            tensors = []
            for i in range(len(gpus)):
                with tf.device(f'/GPU:{i}'):
                    # Allocate ~1GB per GPU
                    tensor = tf.random.normal([8192, 8192])  # ~256MB float32
                    tensors.append(tensor)
            
            # Check memory after allocation
            time.sleep(1)  # Give time for allocation
            result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.free', 
                                   '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print("\nGPU memory state after allocation:")
                for i, line in enumerate(result.stdout.strip().split('\n')):
                    used, free = map(int, line.split(', '))
                    total = used + free
                    print(f"  GPU {i}: {used}MB used, {free}MB free (total: {total}MB)")
                    
    except Exception as e:
        print(f"Could not check memory usage: {e}")


def run_mini_training_test():
    """Run a mini training loop to test multi-GPU training"""
    print("\n" + "="*60)
    print("MINI TRAINING LOOP TEST")
    print("="*60)
    
    # Setup strategy
    gpus = tf.config.list_physical_devices('GPU')
    if len(gpus) >= 2:
        strategy = tf.distribute.MirroredStrategy()
    elif len(gpus) == 1:
        strategy = tf.distribute.OneDeviceStrategy("/GPU:0")
    else:
        strategy = tf.distribute.OneDeviceStrategy("/CPU:0")
    
    print(f"Using strategy: {type(strategy).__name__} with {strategy.num_replicas_in_sync} replicas")
    
    # Create a simple model
    with strategy.scope():
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(10,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        
        optimizer = tf.keras.optimizers.Adam(0.001)
    
    # Create dummy data
    num_samples = 1000
    x_train = np.random.randn(num_samples, 10).astype(np.float32)
    y_train = np.random.randn(num_samples, 1).astype(np.float32)
    
    # Create distributed dataset
    batch_size = 32 * strategy.num_replicas_in_sync  # Scale batch size
    dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    dist_dataset = strategy.experimental_distribute_dataset(dataset)
    
    # Define training step
    @tf.function
    def train_step(inputs, labels):
        with tf.GradientTape() as tape:
            predictions = model(inputs, training=True)
            loss = tf.reduce_mean(tf.square(predictions - labels))
        
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss
    
    @tf.function
    def distributed_train_step(inputs, labels):
        per_replica_losses = strategy.run(train_step, args=(inputs, labels))
        return strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses, axis=None)
    
    # Run training
    print("\nRunning mini training loop...")
    start = time.time()
    
    for epoch in range(3):
        total_loss = 0.0
        num_batches = 0
        
        for x_batch, y_batch in dist_dataset:
            loss = distributed_train_step(x_batch, y_batch)
            total_loss += loss
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch + 1}: avg_loss = {avg_loss:.4f}")
    
    end = time.time()
    print(f"\nTraining completed in {end-start:.3f} seconds")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("MRLCO MULTI-GPU EXECUTION TEST")
    print("="*80)
    
    # Run all tests
    num_gpus = check_gpu_availability()
    
    if num_gpus > 0:
        test_single_gpu_computation()
        test_multi_gpu_strategy()
        test_memory_usage()
        run_mini_training_test()
    else:
        print("\nNo GPUs found. Skipping GPU-specific tests.")
        print("The project will fall back to CPU execution.")
    
    print("\n" + "="*80)
    print("TEST COMPLETED")
    print("="*80)