#!/usr/bin/env python3
"""
Test script for FOMAML implementation
This script tests the FOMAML implementation to ensure it works correctly
"""

import tensorflow as tf
import numpy as np
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from meta_algos.FOMAML import FOMAML
from policies.meta_seq2seq_policy import MetaSeq2SeqPolicy
from samplers.seq2seq_meta_sampler import Seq2SeqMetaSampler
from samplers.seq2seq_meta_sampler_process import Seq2SeqMetaSamplerProcessor
from baselines.vf_baseline import ValueFunctionBaseline
from env.mec_offloaing_envs.offloading_env import Resources, OffloadingEnvironment

def test_fomaml_initialization():
    """Test FOMAML initialization"""
    print("Testing FOMAML initialization...")
    
    # Create a simple environment for testing
    resource_cluster = Resources(
        mec_process_capable=(10.0 * 1024 * 1024),
        mobile_process_capable=(1.0 * 1024 * 1024),
        bandwidth_up=7.0, 
        bandwidth_dl=7.0
    )
    
    env = OffloadingEnvironment(
        resource_cluster=resource_cluster,
        batch_size=10,  # Small batch for testing
        graph_number=10,
        graph_file_paths=[
            "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_1/random.20."
        ],
        time_major=False
    )
    
    # Create policy
    meta_policy = MetaSeq2SeqPolicy(
        meta_batch_size=2,  # Small batch for testing
        obs_dim=17, 
        encoder_units=64,  # Smaller for testing
        decoder_units=64, 
        vocab_size=2
    )
    
    # Create sampler
    sampler = Seq2SeqMetaSampler(
        env=env,
        policy=meta_policy,
        rollouts_per_meta_task=1,
        meta_batch_size=2,
        max_path_length=100,  # Smaller for testing
        parallel=False,
    )
    
    # Create sample processor
    baseline = ValueFunctionBaseline()
    sample_processor = Seq2SeqMetaSamplerProcessor(
        baseline=baseline,
        discount=0.99,
        gae_lambda=0.95,
        normalize_adv=True,
        positive_adv=False
    )
    
    # Create FOMAML algorithm
    try:
        algo = FOMAML(
            policy=meta_policy,
            meta_sampler=sampler,
            meta_sampler_process=sample_processor,
            inner_lr=1e-3,
            outer_lr=1e-4,
            meta_batch_size=2,
            num_inner_grad_steps=1,
            clip_value=0.2
        )
        print("✅ FOMAML initialization successful!")
        return algo, env, sampler, sample_processor, meta_policy
    except Exception as e:
        print(f"❌ FOMAML initialization failed: {e}")
        return None, None, None, None, None

def test_support_query_splitting():
    """Test support/query splitting functionality"""
    print("\nTesting support/query splitting...")
    
    # Create dummy paths data
    dummy_paths = {
        0: [{"observations": np.random.rand(10, 5), "actions": np.random.randint(0, 2, (10, 5))} for _ in range(10)],
        1: [{"observations": np.random.rand(10, 5), "actions": np.random.randint(0, 2, (10, 5))} for _ in range(8)]
    }
    
    # Create sampler for testing
    resource_cluster = Resources(
        mec_process_capable=(10.0 * 1024 * 1024),
        mobile_process_capable=(1.0 * 1024 * 1024),
        bandwidth_up=7.0, 
        bandwidth_dl=7.0
    )
    
    env = OffloadingEnvironment(
        resource_cluster=resource_cluster,
        batch_size=10,
        graph_number=10,
        graph_file_paths=[
            "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_1/random.20."
        ],
        time_major=False
    )
    
    meta_policy = MetaSeq2SeqPolicy(
        meta_batch_size=2,
        obs_dim=17, 
        encoder_units=64,
        decoder_units=64, 
        vocab_size=2
    )
    
    sampler = Seq2SeqMetaSampler(
        env=env,
        policy=meta_policy,
        rollouts_per_meta_task=1,
        meta_batch_size=2,
        max_path_length=100,
        parallel=False,
    )
    
    try:
        support_paths, query_paths = sampler.split_support_query(dummy_paths, support_ratio=0.7)
        
        # Check that splitting worked correctly
        assert len(support_paths) == 2, "Should have 2 tasks in support paths"
        assert len(query_paths) == 2, "Should have 2 tasks in query paths"
        
        # Check that support + query = original for each task
        for task_id in [0, 1]:
            original_len = len(dummy_paths[task_id])
            support_len = len(support_paths[task_id])
            query_len = len(query_paths[task_id])
            
            assert support_len + query_len == original_len, f"Task {task_id}: support + query != original"
            assert support_len > 0, f"Task {task_id}: support set should not be empty"
            assert query_len > 0, f"Task {task_id}: query set should not be empty"
        
        print("✅ Support/query splitting test passed!")
        return True
    except Exception as e:
        print(f"❌ Support/query splitting test failed: {e}")
        return False

def test_fomaml_training_step():
    """Test a single FOMAML training step"""
    print("\nTesting FOMAML training step...")
    
    algo, env, sampler, sample_processor, meta_policy = test_fomaml_initialization()
    
    if algo is None:
        print("❌ Cannot test training step - initialization failed")
        return False
    
    try:
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            
            # Sample some data
            task_ids = sampler.update_tasks()
            paths = sampler.obtain_samples(log=False, log_prefix='')
            
            # Split into support/query
            support_paths, query_paths = sampler.split_support_query(paths, support_ratio=0.7)
            
            # Process samples
            support_samples_data = sample_processor.process_samples(support_paths, log=False, log_prefix='')
            query_samples_data = sample_processor.process_samples(query_paths, log=False, log_prefix='')
            
            # Test inner loop (task adaptation)
            print("  Testing inner loop...")
            for task_id in range(min(2, len(support_samples_data))):
                policy_losses, value_losses = algo.adapt_task(
                    support_samples_data[task_id], task_id, batch_size=10)
                print(f"    Task {task_id}: Policy loss = {np.mean(policy_losses):.4f}, Value loss = {np.mean(value_losses):.4f}")
            
            # Test outer loop (meta-update)
            print("  Testing outer loop...")
            adapted_policies = ["adapted_policy_0", "adapted_policy_1"]
            query_losses = [0.5, 0.3]  # Dummy losses
            
            algo.meta_update(adapted_policies, query_losses)
            print("    Meta-update completed successfully")
            
        print("✅ FOMAML training step test passed!")
        return True
    except Exception as e:
        print(f"❌ FOMAML training step test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("=" * 50)
    print("FOMAML Implementation Tests")
    print("=" * 50)
    
    # Test 1: Initialization
    test_fomaml_initialization()
    
    # Test 2: Support/Query splitting
    test_support_query_splitting()
    
    # Test 3: Training step
    test_fomaml_training_step()
    
    print("\n" + "=" * 50)
    print("All tests completed!")
    print("=" * 50)

if __name__ == "__main__":
    main()
