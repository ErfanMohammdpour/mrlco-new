#!/usr/bin/env python3
"""
Test script to verify that the RL system is learning properly
"""

import sys
import os
import numpy as np
import tensorflow as tf

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from single_policy_trainer import SinglePolicyTrainer
from env.single_policy_offloading_env import SinglePolicyOffloadingEnvironment, Resources
from policies.meta_seq2seq_policy import Seq2SeqPolicy
from samplers.seq2seq_sampler import Seq2SeqSampler
from samplers.seq2seq_sampler_process import Seq2SeqSamplerProcessor
from baselines.vf_baseline import ValueFunctionBaseline
from single_policy_ppo import SinglePolicyPPO
from utils import logger

def test_learning():
    """Test if the RL system is learning properly"""
    
    print("=" * 60)
    print("TESTING RL LEARNING CAPABILITY")
    print("=" * 60)
    
    # Set TensorFlow logging level
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    
    # Configure logger
    logger.configure(dir="./test_logs", format_strs=['stdout', 'log', 'csv'])
    
    # Create map list (use 3 maps for testing)
    map_file_paths = [
        "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_1/random.20.",
        "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_2/random.20.",
        "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_3/random.20.",
    ]
    
    print(f"Using {len(map_file_paths)} maps for testing:")
    for i, path in enumerate(map_file_paths):
        print(f"  {i+1}. {path}")
    print()
    
    # Initialize MEC resource cluster
    print("Initializing MEC resource cluster...")
    resource_cluster = Resources(
        mec_process_capable=(10.0 * 1024 * 1024),
        mobile_process_capable=(1.0 * 1024 * 1024),
        bandwidth_up=7.0,
        bandwidth_dl=7.0
    )
    
    # Initialize environment
    print("Initializing environment...")
    env = SinglePolicyOffloadingEnvironment(
        resource_cluster=resource_cluster,
        batch_size=10,  # Small batch for testing
        graph_number=10,  # Small number of graphs
        graph_file_paths=map_file_paths,
        time_major=False
    )
    
    print(f"Environment initialized with {env.get_total_tasks()} tasks")
    
    # Initialize components
    print("Initializing training components...")
    
    # Value function baseline
    baseline = ValueFunctionBaseline()
    
    # Policy
    policy = Seq2SeqPolicy(
        obs_dim=17,
        encoder_units=64,  # Smaller for faster testing
        decoder_units=64,
        vocab_size=2
    )
    
    # Sampler
    sampler = Seq2SeqSampler(
        env=env,
        policy=policy,
        rollouts_per_task=1,
        max_path_length=100,  # Very short for testing
        parallel=False
    )
    
    # Sample processor
    sample_processor = Seq2SeqSamplerProcessor(
        baseline=baseline,
        discount=0.99,
        gae_lambda=0.95,
        normalize_adv=True,
        positive_adv=False
    )
    
    # PPO algorithm
    algo = SinglePolicyPPO(
        policy=policy,
        sampler=sampler,
        sampler_process=sample_processor,
        lr=1e-3,  # Higher learning rate for testing
        num_grad_steps=4,
        clip_value=0.2,
        vf_coef=0.5,
        max_grad_norm=0.5
    )
    
    # Trainer
    trainer = SinglePolicyTrainer(
        algo=algo,
        env=env,
        sampler=sampler,
        sample_processor=sample_processor,
        policy=policy,
        n_itr=5,  # Only 5 iterations for testing
        greedy_finish_time=0.0,
        start_itr=0,
        batch_size=10,
        save_interval=10
    )
    
    print("Starting training test...")
    print("=" * 60)
    
    # Run training
    try:
        avg_ret, avg_loss, avg_latencies = trainer.train()
        
        print("=" * 60)
        print("TRAINING TEST COMPLETED")
        print("=" * 60)
        print(f"Final average reward: {avg_ret[-1] if avg_ret else 'N/A'}")
        print(f"Final average loss: {avg_loss[-1] if avg_loss else 'N/A'}")
        print(f"Final average latency: {avg_latencies[-1] if avg_latencies else 'N/A'}")
        
        # Check if learning occurred
        if len(avg_ret) > 1:
            reward_improvement = avg_ret[-1] - avg_ret[0]
            print(f"Reward improvement: {reward_improvement:.4f}")
            
            if reward_improvement > 0:
                print("✅ LEARNING DETECTED: Reward improved over time!")
            else:
                print("❌ NO LEARNING: Reward did not improve")
        
        if len(avg_loss) > 1:
            loss_change = avg_loss[-1] - avg_loss[0]
            print(f"Loss change: {loss_change:.4f}")
            
            if loss_change < 0:
                print("✅ LEARNING DETECTED: Loss decreased over time!")
            else:
                print("❌ NO LEARNING: Loss did not decrease")
        
        return True
        
    except Exception as e:
        print(f"❌ Training test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_learning()
    if success:
        print("\n🎉 Test completed successfully!")
    else:
        print("\n💥 Test failed!")
        sys.exit(1)
