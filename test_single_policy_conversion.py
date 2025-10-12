#!/usr/bin/env python3
"""
Test script for single-policy RL conversion

This script tests the basic functionality of the converted single-policy RL system
without running the full training loop.
"""

import tensorflow as tf
import numpy as np
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all required modules can be imported"""
    print("Testing imports...")
    
    try:
        from single_policy_trainer import SinglePolicyTrainer
        print("✓ SinglePolicyTrainer imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import SinglePolicyTrainer: {e}")
        return False
    
    try:
        from env.single_policy_offloading_env import SinglePolicyOffloadingEnvironment, Resources
        print("✓ SinglePolicyOffloadingEnvironment imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import SinglePolicyOffloadingEnvironment: {e}")
        return False
    
    try:
        from policies.meta_seq2seq_policy import Seq2SeqPolicy
        print("✓ Seq2SeqPolicy imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import Seq2SeqPolicy: {e}")
        return False
    
    try:
        from samplers.seq2seq_sampler import Seq2SeqSampler
        print("✓ Seq2SeqSampler imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import Seq2SeqSampler: {e}")
        return False
    
    try:
        from samplers.seq2seq_sampler_process import Seq2SeqSamplerProcessor
        print("✓ Seq2SeqSamplerProcessor imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import Seq2SeqSamplerProcessor: {e}")
        return False
    
    try:
        from baselines.vf_baseline import ValueFunctionBaseline
        print("✓ ValueFunctionBaseline imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import ValueFunctionBaseline: {e}")
        return False
    
    try:
        from single_policy_ppo import SinglePolicyPPO
        print("✓ SinglePolicyPPO imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import SinglePolicyPPO: {e}")
        return False
    
    return True

def test_environment_creation():
    """Test that the environment can be created"""
    print("\nTesting environment creation...")
    
    try:
        from env.single_policy_offloading_env import SinglePolicyOffloadingEnvironment, Resources
        
        # Create resource cluster
        resource_cluster = Resources(
            mec_process_capable=(10.0 * 1024 * 1024),
            mobile_process_capable=(1.0 * 1024 * 1024),
            bandwidth_up=7.0,
            bandwidth_dl=7.0
        )
        print("✓ Resource cluster created successfully")
        
        # Create environment with minimal configuration
        env = SinglePolicyOffloadingEnvironment(
            resource_cluster=resource_cluster,
            batch_size=10,  # Small batch for testing
            graph_number=10,  # Small number for testing
            graph_file_paths=["./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_1/random.20."],
            time_major=False
        )
        print("✓ SinglePolicyOffloadingEnvironment created successfully")
        
        # Test basic environment methods
        obs = env.reset()
        print(f"✓ Environment reset successful, observation shape: {obs.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Environment creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_policy_creation():
    """Test that the policy can be created"""
    print("\nTesting policy creation...")
    
    try:
        from policies.meta_seq2seq_policy import Seq2SeqPolicy
        
        # Create single policy
        policy = Seq2SeqPolicy(
            obs_dim=17,
            encoder_units=64,  # Smaller for testing
            decoder_units=64,
            vocab_size=2
        )
        print("✓ Seq2SeqPolicy created successfully")
        
        # Test policy methods
        print(f"✓ Policy action dimension: {policy.action_dim}")
        print(f"✓ Policy name: {policy.name}")
        
        return True
        
    except Exception as e:
        print(f"✗ Policy creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_algorithm_creation():
    """Test that the PPO algorithm can be created"""
    print("\nTesting algorithm creation...")
    
    try:
        from single_policy_ppo import SinglePolicyPPO
        from policies.meta_seq2seq_policy import Seq2SeqPolicy
        from samplers.seq2seq_sampler import Seq2SeqSampler
        from samplers.seq2seq_sampler_process import Seq2SeqSamplerProcessor
        from baselines.vf_baseline import ValueFunctionBaseline
        from env.single_policy_offloading_env import SinglePolicyOffloadingEnvironment, Resources
        
        # Create minimal components
        resource_cluster = Resources(
            mec_process_capable=(10.0 * 1024 * 1024),
            mobile_process_capable=(1.0 * 1024 * 1024),
            bandwidth_up=7.0,
            bandwidth_dl=7.0
        )
        
        env = SinglePolicyOffloadingEnvironment(
            resource_cluster=resource_cluster,
            batch_size=5,
            graph_number=5,
            graph_file_paths=["./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_1/random.20."],
            time_major=False
        )
        
        policy = Seq2SeqPolicy(obs_dim=17, encoder_units=32, decoder_units=32, vocab_size=2)
        sampler = Seq2SeqSampler(env=env, policy=policy, rollouts_per_task=1, max_path_length=100, parallel=False)
        baseline = ValueFunctionBaseline()
        sample_processor = Seq2SeqSamplerProcessor(baseline=baseline, discount=0.99, gae_lambda=0.95, normalize_adv=True, positive_adv=False)
        
        # Create PPO algorithm
        algo = SinglePolicyPPO(
            policy=policy,
            sampler=sampler,
            sampler_process=sample_processor,
            lr=1e-4,
            num_grad_steps=1,
            clip_value=0.2
        )
        print("✓ SinglePolicyPPO created successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Algorithm creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("=" * 60)
    print("SINGLE-POLICY RL CONVERSION TEST")
    print("=" * 60)
    
    # Suppress TensorFlow warnings
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    
    tests = [
        ("Import Test", test_imports),
        ("Environment Creation Test", test_environment_creation),
        ("Policy Creation Test", test_policy_creation),
        ("Algorithm Creation Test", test_algorithm_creation),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 40)
        if test_func():
            passed += 1
            print(f"✓ {test_name} PASSED")
        else:
            print(f"✗ {test_name} FAILED")
    
    print("\n" + "=" * 60)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("🎉 All tests passed! The conversion appears to be successful.")
        print("\nYou can now run the full training with:")
        print("python train_single_policy.py")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
