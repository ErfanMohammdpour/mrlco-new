#!/usr/bin/env python3
"""
Test Script for DRL Fine-tuning System

This script tests the basic functionality of the fine-tuning system
without running the full training loops.
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
        from weight_manager import WeightManager
        print("✓ WeightManager imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import WeightManager: {e}")
        return False
    
    try:
        from pretrain_on_maps import pretrain_policy
        print("✓ pretrain_policy imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import pretrain_policy: {e}")
        return False
    
    try:
        from finetune_on_map import finetune_policy
        print("✓ finetune_policy imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import finetune_policy: {e}")
        return False
    
    try:
        from evaluate_policy import evaluate_policy
        print("✓ evaluate_policy imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import evaluate_policy: {e}")
        return False
    
    try:
        from finetuning_workflow import main as workflow_main
        print("✓ finetuning_workflow imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import finetuning_workflow: {e}")
        return False
    
    return True

def test_weight_manager():
    """Test weight manager functionality"""
    print("\nTesting weight manager...")
    
    try:
        from weight_manager import WeightManager
        
        # Create weight manager
        wm = WeightManager()
        print("✓ WeightManager created successfully")
        
        # Test directory creation
        assert os.path.exists(wm.pretrained_dir), "Pre-trained directory not created"
        assert os.path.exists(wm.finetuned_dir), "Fine-tuned directory not created"
        print("✓ Directories created successfully")
        
        # Test weight listing
        weights = wm.list_available_weights()
        assert isinstance(weights, dict), "Weight listing should return dict"
        assert 'pretrained' in weights, "Should have pretrained weights list"
        assert 'finetuned' in weights, "Should have finetuned weights list"
        print("✓ Weight listing works")
        
        return True
        
    except Exception as e:
        print(f"✗ Weight manager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_environment_creation():
    """Test that environments can be created"""
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
            batch_size=5,  # Small batch for testing
            graph_number=5,  # Small number for testing
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
    """Test that policies can be created"""
    print("\nTesting policy creation...")
    
    try:
        # Clear TensorFlow graph to avoid variable sharing conflicts
        tf.compat.v1.reset_default_graph()
        
        from policies.meta_seq2seq_policy import Seq2SeqPolicy
        
        # Create single policy
        policy = Seq2SeqPolicy(
            obs_dim=17,
            encoder_units=32,  # Smaller for testing
            decoder_units=32,
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
    """Test that algorithms can be created"""
    print("\nTesting algorithm creation...")
    
    try:
        # Clear TensorFlow graph to avoid variable sharing conflicts
        tf.compat.v1.reset_default_graph()
        
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
            batch_size=2,
            graph_number=2,
            graph_file_paths=["./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_1/random.20."],
            time_major=False
        )
        
        policy = Seq2SeqPolicy(obs_dim=17, encoder_units=16, decoder_units=16, vocab_size=2)
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

def test_map_creation():
    """Test that map environments can be created"""
    print("\nTesting map environment creation...")
    
    try:
        # Clear TensorFlow graph to avoid variable sharing conflicts
        tf.compat.v1.reset_default_graph()
        
        from finetune_on_map import create_single_map_environment
        from env.single_policy_offloading_env import Resources
        
        # Create resource cluster
        resource_cluster = Resources(
            mec_process_capable=(10.0 * 1024 * 1024),
            mobile_process_capable=(1.0 * 1024 * 1024),
            bandwidth_up=7.0,
            bandwidth_dl=7.0
        )
        
        # Test creating environment for map 1
        env, map_file_path = create_single_map_environment(1, resource_cluster)
        print(f"✓ Map 1 environment created successfully")
        print(f"✓ Map file path: {map_file_path}")
        
        # Test environment reset
        obs = env.reset()
        print(f"✓ Map environment reset successful, observation shape: {obs.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Map environment creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("=" * 60)
    print("DRL FINE-TUNING SYSTEM TEST")
    print("=" * 60)
    
    # Suppress TensorFlow warnings
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    
    tests = [
        ("Import Test", test_imports),
        ("Weight Manager Test", test_weight_manager),
        ("Environment Creation Test", test_environment_creation),
        ("Policy Creation Test", test_policy_creation),
        ("Algorithm Creation Test", test_algorithm_creation),
        ("Map Environment Test", test_map_creation),
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
        print("🎉 All tests passed! The fine-tuning system is ready to use.")
        print("\nYou can now run the fine-tuning workflow:")
        print("1. Pre-train: python finetuning_workflow.py --mode pretrain --maps 5 --iterations 100")
        print("2. Fine-tune: python finetuning_workflow.py --mode finetune --map_id 1 --steps 20")
        print("3. Evaluate: python finetuning_workflow.py --mode evaluate --map_id 1 --episodes 5")
        print("4. Complete workflow: python finetuning_workflow.py --mode workflow --maps 5 --workflow_maps 1 2 3")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
