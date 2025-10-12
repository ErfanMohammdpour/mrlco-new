#!/usr/bin/env python3
"""
Pre-training Script for DRL Fine-tuning

This script pre-trains the policy on 22 maps and saves the weights for later fine-tuning.
The pre-trained weights can then be loaded and fine-tuned on specific maps.

Usage:
    python pretrain_on_maps.py --maps 22 --iterations 1000 --save_interval 100
"""

import tensorflow as tf
import numpy as np
import os
import sys
import argparse
from datetime import datetime

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from single_policy_trainer import SinglePolicyTrainer
from env.single_policy_offloading_env import SinglePolicyOffloadingEnvironment, Resources
from policies.meta_seq2seq_policy import Seq2SeqPolicy
from samplers.seq2seq_sampler import Seq2SeqSampler
from samplers.seq2seq_sampler_process import Seq2SeqSamplerProcessor
from baselines.vf_baseline import ValueFunctionBaseline
from single_policy_ppo import SinglePolicyPPO
from weight_manager import WeightManager
from utils import logger

def create_map_list(num_maps=22):
    """
    Create a list of map file paths for pre-training
    
    Args:
        num_maps: Number of maps to use for pre-training
        
    Returns:
        list: List of map file paths
    """
    # Base map file paths (you can modify this list based on your available maps)
    base_maps = [
        "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_1/random.20.",
        "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_2/random.20.",
        "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_3/random.20.",
        "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_4/random.20.",
        "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_5/random.20.",
        "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_6/random.20.",
        "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_7/random.20.",
        "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_8/random.20.",
        "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_9/random.20.",
        "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_13/random.20.",
        "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_14/random.20.",
        "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_15/random.20.",
        "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_16/random.20.",
        "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_17/random.20.",
        "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_18/random.20.",
        "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_19/random.20.",
        "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_20/random.20.",
        "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_21/random.20.",
        "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_22/random.20.",
        "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_23/random.20.",
        "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_24/random.20.",
        "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_25/random.20.",
    ]
    
    # Return the requested number of maps
    return base_maps[:num_maps]

def pretrain_policy(num_maps=22, iterations=1000, batch_size=100, learning_rate=5e-4, save_interval=100):
    """
    Pre-train the policy on multiple maps
    
    Args:
        num_maps: Number of maps to use for pre-training
        iterations: Number of training iterations
        batch_size: Batch size for training
        learning_rate: Learning rate for PPO
        save_interval: Save weights every N iterations
        
    Returns:
        tuple: (policy, weight_path, metadata_path)
    """
    
    print("=" * 80)
    print("PRE-TRAINING DRL POLICY ON MULTIPLE MAPS")
    print("=" * 80)
    print(f"Number of maps: {num_maps}")
    print(f"Training iterations: {iterations}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Save interval: {save_interval}")
    print()
    
    # Set TensorFlow logging level
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    
    # Configure logger
    log_dir = f"./pretraining_logs/maps_{num_maps}_iter_{iterations}"
    logger.configure(dir=log_dir, format_strs=['stdout', 'log', 'csv'])
    
    # Initialize weight manager
    weight_manager = WeightManager()
    
    # Create map list
    map_file_paths = create_map_list(num_maps)
    print(f"Using {len(map_file_paths)} maps for pre-training:")
    for i, path in enumerate(map_file_paths):
        print(f"  {i+1:2d}. {path}")
    print()
    
    # Initialize MEC resource cluster
    print("Initializing MEC resource cluster...")
    resource_cluster = Resources(
        mec_process_capable=(10.0 * 1024 * 1024),
        mobile_process_capable=(1.0 * 1024 * 1024),
        bandwidth_up=7.0,
        bandwidth_dl=7.0
    )
    
    # Initialize environment with all maps
    print("Initializing environment with all maps...")
    env = SinglePolicyOffloadingEnvironment(
        resource_cluster=resource_cluster,
        batch_size=batch_size,
        graph_number=100,  # Number of graphs per map
        graph_file_paths=map_file_paths,
        time_major=False
    )
    
    # Calculate baseline solutions
    print("Calculating baseline solutions...")
    action, greedy_finish_time = env.greedy_solution()
    print(f"Average greedy solution latency: {np.mean(greedy_finish_time):.4f}")
    
    finish_time = env.get_all_mec_execute_time()
    print(f"Average all-remote solution latency: {np.mean(finish_time):.4f}")
    
    finish_time = env.get_all_locally_execute_time()
    print(f"Average all-local solution latency: {np.mean(finish_time):.4f}")
    print()
    
    # Initialize components
    print("Initializing training components...")
    
    # Value function baseline
    baseline = ValueFunctionBaseline()
    
    # Single policy
    policy = Seq2SeqPolicy(
        obs_dim=17,
        encoder_units=128,
        decoder_units=128,
        vocab_size=2
    )
    
    # Sampler - optimized for better performance
    sampler = Seq2SeqSampler(
        env=env,
        policy=policy,
        rollouts_per_task=10,  # More rollouts per task for better efficiency
        max_path_length=1000,  # Reduced from 20000 for faster training
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
        lr=learning_rate,
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
        n_itr=iterations,
        greedy_finish_time=greedy_finish_time,
        start_itr=0,
        batch_size=1000,
        save_interval=save_interval
    )
    
    print("Starting pre-training...")
    print("=" * 80)
    
    # Start training
    with tf.compat.v1.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        try:
            # Train the policy
            avg_ret, avg_loss, avg_latencies = trainer.train()
            
            # Save final pre-trained weights
            print("\nSaving final pre-trained weights...")
            weight_path, metadata_path = weight_manager.save_pretrained_weights(
                policy=policy,
                iteration=iterations,
                map_count=num_maps,
                additional_info={
                    'final_avg_reward': avg_ret[-1] if avg_ret else 0.0,
                    'final_avg_loss': avg_loss[-1] if avg_loss else 0.0,
                    'final_avg_latency': avg_latencies[-1] if avg_latencies else 0.0,
                    'learning_rate': learning_rate,
                    'batch_size': batch_size
                }
            )
            
            print("=" * 80)
            print("PRE-TRAINING COMPLETED SUCCESSFULLY!")
            print("=" * 80)
            print(f"Final average reward: {avg_ret[-1]:.4f}")
            print(f"Final average loss: {avg_loss[-1]:.4f}")
            print(f"Final average latency: {avg_latencies[-1]:.4f}")
            print(f"Pre-trained weights saved to: {weight_path}")
            print(f"Metadata saved to: {metadata_path}")
            print(f"Logs saved to: {log_dir}")
            
            return policy, weight_path, metadata_path
            
        except KeyboardInterrupt:
            print("\nPre-training interrupted by user.")
            print("Saving current weights...")
            weight_path, metadata_path = weight_manager.save_pretrained_weights(
                policy=policy,
                iteration=trainer.start_itr,
                map_count=num_maps,
                additional_info={'status': 'interrupted'}
            )
            print(f"Weights saved to: {weight_path}")
            return policy, weight_path, metadata_path
            
        except Exception as e:
            print(f"\nPre-training failed with error: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None, None

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Pre-train DRL policy on multiple maps')
    parser.add_argument('--maps', type=int, default=22, help='Number of maps to use for pre-training')
    parser.add_argument('--iterations', type=int, default=1000, help='Number of training iterations')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='Learning rate for PPO')
    parser.add_argument('--save_interval', type=int, default=100, help='Save weights every N iterations')
    
    args = parser.parse_args()
    
    # Pre-train the policy
    policy, weight_path, metadata_path = pretrain_policy(
        num_maps=args.maps,
        iterations=args.iterations,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        save_interval=args.save_interval
    )
    
    if policy is not None:
        print("\n🎉 Pre-training completed successfully!")
        print(f"You can now use the weights for fine-tuning:")
        print(f"  python finetune_on_map.py --map_id 1 --weights {weight_path}")
        return 0
    else:
        print("\n❌ Pre-training failed!")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
