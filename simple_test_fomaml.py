#!/usr/bin/env python3
"""
Simple test for FOMAML implementation
"""

import tensorflow as tf
import numpy as np
import sys
import os

# Clear any existing graph
tf.reset_default_graph()

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_fomaml_simple():
    """Simple test for FOMAML"""
    print("Testing FOMAML simple initialization...")
    
    try:
        from meta_algos.FOMAML import FOMAML
        from policies.meta_seq2seq_policy import MetaSeq2SeqPolicy
        from samplers.seq2seq_meta_sampler import Seq2SeqMetaSampler
        from samplers.seq2seq_meta_sampler_process import Seq2SeqMetaSamplerProcessor
        from baselines.vf_baseline import ValueFunctionBaseline
        from env.mec_offloaing_envs.offloading_env import Resources, OffloadingEnvironment
        
        # Create a simple environment for testing
        resource_cluster = Resources(
            mec_process_capable=(10.0 * 1024 * 1024),
            mobile_process_capable=(1.0 * 1024 * 1024),
            bandwidth_up=7.0, 
            bandwidth_dl=7.0
        )
        
        env = OffloadingEnvironment(
            resource_cluster=resource_cluster,
            batch_size=5,  # Very small batch for testing
            graph_number=5,
            graph_file_paths=[
                "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_1/random.20."
            ],
            time_major=False
        )
        
        # Create policy
        meta_policy = MetaSeq2SeqPolicy(
            meta_batch_size=2,  # Small batch for testing
            obs_dim=17, 
            encoder_units=32,  # Very small for testing
            decoder_units=32, 
            vocab_size=2
        )
        
        # Create sampler
        sampler = Seq2SeqMetaSampler(
            env=env,
            policy=meta_policy,
            rollouts_per_meta_task=1,
            meta_batch_size=2,
            max_path_length=50,  # Very small for testing
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
        
        # Test with session
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            print("✅ TensorFlow session initialized successfully!")
            
            # Test sampling
            task_ids = sampler.update_tasks()
            print(f"✅ Task sampling successful! Got {len(task_ids)} tasks")
            
            # Test path splitting
            paths = sampler.obtain_samples(log=False, log_prefix='')
            support_paths, query_paths = sampler.split_support_query(paths, support_ratio=0.7)
            print(f"✅ Path splitting successful! Support: {len(support_paths)}, Query: {len(query_paths)}")
            
        print("✅ All tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("Simple FOMAML Test")
    print("=" * 50)
    
    success = test_fomaml_simple()
    
    if success:
        print("\n🎉 All tests passed! FOMAML is working correctly.")
    else:
        print("\n💥 Tests failed! Please check the errors above.")
    
    print("=" * 50)
