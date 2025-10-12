#!/usr/bin/env python3
"""
Quick test to verify if the policy is learning
"""

import numpy as np
import tensorflow as tf
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pretrain_on_maps import pretrain_policy

def test_learning():
    """Test if the policy learns with the corrected reward function"""
    print("=" * 60)
    print("TESTING POLICY LEARNING")
    print("=" * 60)
    
    # Initialize TensorFlow session
    sess = tf.compat.v1.Session()
    with sess.as_default():
        sess.run(tf.compat.v1.global_variables_initializer())
        
        try:
            # Run pre-training for a few iterations
            print("Running pre-training for 5 iterations...")
            avg_ret, avg_loss, avg_latencies = pretrain_policy(
                num_maps=2,  # Use only 2 maps for quick test
                iterations=5,  # Only 5 iterations
                batch_size=50,  # Smaller batch size
                learning_rate=5e-4,
                save_interval=5
            )
            
            print("\n" + "=" * 60)
            print("LEARNING TEST RESULTS")
            print("=" * 60)
            
            if len(avg_ret) > 1:
                reward_improvement = avg_ret[-1] - avg_ret[0]
                print(f"Reward improvement: {reward_improvement:.4f}")
                if reward_improvement > 0:
                    print("✅ LEARNING DETECTED: Reward improved over time!")
                else:
                    print("❌ NO LEARNING: Reward did not improve")
            
            if len(avg_latencies) > 1:
                latency_improvement = avg_latencies[0] - avg_latencies[-1]
                print(f"Latency improvement: {latency_improvement:.4f}")
                if latency_improvement > 0:
                    print("✅ LEARNING DETECTED: Latency decreased over time!")
                else:
                    print("❌ NO LEARNING: Latency did not decrease")
            
            print(f"\nFinal average reward: {avg_ret[-1] if avg_ret else 'N/A'}")
            print(f"Final average latency: {avg_latencies[-1] if avg_latencies else 'N/A'}")
            
            return True
            
        except Exception as e:
            print(f"❌ Learning test failed: {e}")
            import traceback
            traceback.print_exc()
            return False

if __name__ == "__main__":
    success = test_learning()
    if success:
        print("\n🎉 Learning test completed successfully!")
    else:
        print("\n💥 Learning test failed!")
