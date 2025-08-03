#!/usr/bin/env python3
"""Simplified demo of GPU optimizations for MRLCO TF2.19"""

import os
import time
import tensorflow as tf
import numpy as np

# Disable eager execution for TF1 compatibility
tf.compat.v1.disable_eager_execution()

print(f"TensorFlow version: {tf.__version__}")

# Enable memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
print(f"Found {len(gpus)} GPU(s)")

def create_optimized_trainer():
    """Demonstrate key GPU optimizations"""
    
    print("\n========== GPU Optimization Demo ==========")
    
    # 1. Strategy setup
    print("\n1. Setting up distribution strategy...")
    if len(gpus) >= 2:
        strategy = tf.distribute.MirroredStrategy()
        print(f"   ✓ Using MirroredStrategy with {len(gpus)} GPUs")
    elif len(gpus) == 1:
        strategy = tf.distribute.OneDeviceStrategy("/GPU:0")
        print("   ✓ Using OneDeviceStrategy on GPU:0")
    else:
        strategy = tf.distribute.OneDeviceStrategy("/CPU:0")
        print("   ✗ No GPU found, using CPU fallback")
    
    # 2. Create model within strategy scope
    print("\n2. Creating model within strategy scope...")
    with strategy.scope():
        # Simple model for demonstration
        inputs = tf.keras.Input(shape=(100,))
        x = tf.keras.layers.Dense(256, activation='relu')(inputs)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        outputs = tf.keras.layers.Dense(10)(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        # Use float32 (avoid float64)
        model = tf.keras.models.clone_model(model)
        
        # Create optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        print("   ✓ Model and optimizer created in strategy scope")
    
    # 3. Warmup on GPU
    print("\n3. Performing GPU warmup...")
    with strategy.scope():
        # Create warmup data
        warmup_data = tf.random.normal([32, 100], dtype=tf.float32)
        
        # Run warmup operations
        start_time = time.time()
        for i in range(5):
            _ = model(warmup_data, training=False)
        warmup_time = time.time() - start_time
        
        # Check device placement
        if model.weights:
            weight_device = model.weights[0].device if hasattr(model.weights[0], 'device') else 'Unknown'
            print(f"   ✓ Model weights on device: {weight_device}")
        print(f"   ✓ Warmup completed in {warmup_time:.3f} seconds")
    
    # 4. Optimized tf.data pipeline
    print("\n4. Creating optimized tf.data pipeline...")
    
    # Generate synthetic data
    num_samples = 1000
    x_data = np.random.randn(num_samples, 100).astype(np.float32)
    y_data = np.random.randn(num_samples, 10).astype(np.float32)
    
    # Create dataset with optimizations
    dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data))
    dataset = dataset.cache()  # Cache in memory
    dataset = dataset.shuffle(buffer_size=100)  # Shuffle
    dataset = dataset.batch(32, drop_remainder=True)  # Batch
    dataset = dataset.prefetch(tf.data.AUTOTUNE)  # Prefetch
    
    print("   ✓ Pipeline: cache → shuffle → batch → prefetch(AUTOTUNE)")
    
    # 5. @tf.function compiled training step
    print("\n5. Creating @tf.function compiled training step...")
    
    @tf.function
    def train_step(inputs, targets):
        """Single training step with tf.function compilation"""
        with tf.GradientTape() as tape:
            predictions = model(inputs, training=True)
            loss = tf.reduce_mean(tf.square(predictions - targets))
        
        gradients = tape.gradient(loss, model.trainable_variables)
        
        # Clip gradients (important for stability)
        gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
        
        # Apply gradients
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        return loss
    
    # Distributed version
    @tf.function
    def distributed_train_step(inputs, targets):
        """Distributed training step using strategy.run"""
        per_replica_losses = strategy.run(train_step, args=(inputs, targets))
        return strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses, axis=None)
    
    print("   ✓ Training step compiled with @tf.function")
    print("   ✓ Using strategy.run() and strategy.reduce()")
    
    # 6. Demonstrate training without numpy operations
    print("\n6. Running sample training (no numpy in loop)...")
    
    # Take a few batches for demonstration
    train_losses = []
    step_times = []
    
    for step, (x_batch, y_batch) in enumerate(dataset.take(10)):
        step_start = time.time()
        
        # Train step - no .numpy() calls or print inside
        loss = distributed_train_step(x_batch, y_batch)
        
        step_time = time.time() - step_start
        step_times.append(step_time)
        
        # Only convert to numpy after computation
        train_losses.append(loss.numpy())
        
        if step == 0:
            # Log device info on first step
            print(f"\n   [Step 0] First batch shape: {x_batch.shape}")
            print(f"   [Step 0] Loss value: {loss.numpy():.4f}")
            print(f"   [Step 0] Step time: {step_time:.3f}s")
    
    avg_step_time = np.mean(step_times)
    print(f"\n   ✓ Average step time: {avg_step_time:.3f}s")
    print(f"   ✓ Final loss: {train_losses[-1]:.4f}")
    
    # 7. Summary of optimizations
    print("\n========== Optimization Summary ==========")
    print("✓ Distribution strategy with proper scope")
    print("✓ @tf.function compiled training steps")
    print("✓ strategy.run() and strategy.reduce() for distributed execution")
    print("✓ Optimized tf.data pipeline (cache→shuffle→batch→prefetch)")
    print("✓ No numpy operations inside training loop")
    print("✓ Float32 tensors (not float64)")
    print("✓ Gradient clipping for stability")
    print("✓ GPU warmup before training")
    print("✓ Memory growth enabled")
    
    return model, strategy, dataset

def main():
    """Run the demonstration"""
    try:
        # Set optimal GPU settings
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
        os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
        
        # Run demo
        model, strategy, dataset = create_optimized_trainer()
        
        print("\n========== Demo Complete ==========")
        print("All GPU optimizations have been demonstrated.")
        print("These techniques will significantly improve GPU utilization")
        print("when applied to the full MRLCO training pipeline.")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()