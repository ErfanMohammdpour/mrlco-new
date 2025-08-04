#!/usr/bin/env python3
"""Test script to verify GPU setup and logging"""

import os
import tensorflow as tf
from utils.gpu import setup_gpu_and_strategy, log_tensor_device

print("=" * 70)
print("GPU SETUP TEST SCRIPT")
print("=" * 70)

# Test with different CUDA_VISIBLE_DEVICES settings
test_scenarios = [
    ("2 GPUs", "0,1"),
    ("1 GPU", "0"),
    ("No GPU", "")
]

for scenario_name, cuda_devices in test_scenarios:
    print(f"\n{'=' * 70}")
    print(f"Testing scenario: {scenario_name}")
    print(f"Setting CUDA_VISIBLE_DEVICES={cuda_devices}")
    print(f"{'=' * 70}")
    
    # Set environment variable
    os.environ['CUDA_VISIBLE_DEVICES'] = cuda_devices
    
    # Clear any existing TF devices
    tf.config.set_visible_devices([], 'GPU')
    if cuda_devices:
        # Re-detect GPUs
        physical_devices = tf.config.list_physical_devices('GPU')
        if len(physical_devices) > 0:
            tf.config.set_visible_devices(physical_devices[:len(cuda_devices.split(','))], 'GPU')
    
    # Set up strategy
    strategy, device_info = setup_gpu_and_strategy()
    
    # Test computation
    print("\nRunning test computation...")
    with strategy.scope():
        # Create test tensors
        a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        b = tf.constant([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]])
        
        # Perform computation
        result = tf.matmul(a, b)
        
        # Log device placement
        log_tensor_device(result, "Matrix multiplication result")
        
        print(f"Computation result shape: {result.shape}")
        print(f"Computation completed successfully!")
    
    print("\n" + "-" * 70 + "\n")

print("\nAll GPU setup tests completed!")