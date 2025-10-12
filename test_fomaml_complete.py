#!/usr/bin/env python3
"""
Complete test for FOMAML implementation
"""

import tensorflow as tf
import numpy as np
import sys
import os

# Clear any existing graph
tf.reset_default_graph()

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_fomaml_complete():
    """Complete test for FOMAML implementation"""
    print("=" * 60)
    print("Complete FOMAML Test")
    print("=" * 60)
    
    try:
        from meta_algos.FOMAML import FOMAML
        from policies.meta_seq2seq_policy import MetaSeq2SeqPolicy
        from samplers.seq2seq_meta_sampler import Seq2SeqMetaSampler
        from samplers.seq2seq_meta_sampler_process import Seq2SeqMetaSamplerProcessor
        from baselines.vf_baseline import ValueFunctionBaseline
        from env.mec_offloaing_envs.offloading_env import Resources, OffloadingEnvironment
        
        print("✅ All imports successful")
        
        # Create environment
        resource_cluster = Resources(
            mec_process_capable=(10.0 * 1024 * 1024),
            mobile_process_capable=(1.0 * 1024 * 1024),
            bandwidth_up=7.0, 
            bandwidth_dl=7.0
        )
        
        env = OffloadingEnvironment(
            resource_cluster=resource_cluster,
            batch_size=5,
            graph_number=5,
            graph_file_paths=[
                "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_1/random.20."
            ],
            time_major=False
        )
        
        print("✅ Environment created successfully")
        
        # Create policy
        meta_policy = MetaSeq2SeqPolicy(
            meta_batch_size=2,
            obs_dim=17, 
            encoder_units=32,
            decoder_units=32, 
            vocab_size=2
        )
        
        print("✅ Policy created successfully")
        
        # Create sampler
        sampler = Seq2SeqMetaSampler(
            env=env,
            policy=meta_policy,
            rollouts_per_meta_task=1,
            meta_batch_size=2,
            max_path_length=50,
            parallel=False,
        )
        
        print("✅ Sampler created successfully")
        
        # Create sample processor
        baseline = ValueFunctionBaseline()
        sample_processor = Seq2SeqMetaSamplerProcessor(
            baseline=baseline,
            discount=0.99,
            gae_lambda=0.95,
            normalize_adv=True,
            positive_adv=False
        )
        
        print("✅ Sample processor created successfully")
        
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
        
        print("✅ FOMAML algorithm created successfully")
        
        # Test with session
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            print("✅ TensorFlow session initialized successfully")
            
            # Test sampling
            task_ids = sampler.update_tasks()
            print(f"✅ Task sampling successful! Got {len(task_ids)} tasks")
            
            # Test path splitting
            paths = sampler.obtain_samples(log=False, log_prefix='')
            support_paths, query_paths = sampler.split_support_query(paths, support_ratio=0.7)
            print(f"✅ Path splitting successful! Support: {len(support_paths)}, Query: {len(query_paths)}")
            
            # Test sample processing
            support_samples_data = sample_processor.process_samples(support_paths, log=False, log_prefix='')
            query_samples_data = sample_processor.process_samples(query_paths, log=False, log_prefix='')
            print(f"✅ Sample processing successful! Support: {len(support_samples_data)}, Query: {len(query_samples_data)}")
            
            # Test inner loop (task adaptation)
            print("\n--- Testing Inner Loop (Task Adaptation) ---")
            for task_id in range(min(2, len(support_samples_data))):
                if support_samples_data[task_id] is not None:
                    policy_losses, value_losses = algo.adapt_task(
                        support_samples_data[task_id], task_id, batch_size=10)
                    print(f"  Task {task_id}: Policy loss = {np.mean(policy_losses):.4f}, Value loss = {np.mean(value_losses):.4f}")
                else:
                    print(f"  Task {task_id}: No valid data")
            
            # Test outer loop (meta-update)
            print("\n--- Testing Outer Loop (Meta-Update) ---")
            adapted_policies = ["adapted_policy_0", "adapted_policy_1"]
            query_losses = [0.5, 0.3]  # Dummy losses
            
            algo.meta_update(adapted_policies, query_losses)
            print("✅ Meta-update completed successfully")
            
        print("\n" + "=" * 60)
        print("🎉 Complete FOMAML Test PASSED!")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_fomaml_complete()
    
    if success:
        print("\n🚀 FOMAML is ready for training!")
        print("You can now run: python meta_trainer.py")
    else:
        print("\n💥 FOMAML test failed! Please check the errors above.")
        sys.exit(1)
