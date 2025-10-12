#!/usr/bin/env python3
"""
Fine-tuning Script for DRL on Specific Maps

This script loads pre-trained weights and fine-tunes the policy on a specific map
for a limited number of steps (e.g., 20 steps).

Usage:
    python finetune_on_map.py --map_id 1 --weights path/to/pretrained.ckpt --steps 20
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

def create_single_map_environment(map_id, resource_cluster):
    """
    Create environment for a specific map
    
    Args:
        map_id: ID of the specific map
        resource_cluster: MEC resource cluster
        
    Returns:
        SinglePolicyOffloadingEnvironment: Environment for the specific map
    """
    # Map file paths (you can modify this based on your available maps)
    map_files = {
        1: "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_1/random.20.",
        2: "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_2/random.20.",
        3: "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_3/random.20.",
        4: "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_4/random.20.",
        5: "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_5/random.20.",
        6: "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_6/random.20.",
        7: "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_7/random.20.",
        8: "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_8/random.20.",
        9: "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_9/random.20.",
        10: "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_10/random.20.",
        11: "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_11/random.20.",
        12: "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_12/random.20.",
        13: "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_13/random.20.",
        14: "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_14/random.20.",
        15: "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_15/random.20.",
        16: "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_16/random.20.",
        17: "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_17/random.20.",
        18: "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_18/random.20.",
        19: "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_19/random.20.",
        20: "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_20/random.20.",
        21: "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_21/random.20.",
        22: "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_22/random.20.",
        23: "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_23/random.20.",
        24: "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_24/random.20.",
        25: "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_25/random.20.",
    }
    
    if map_id not in map_files:
        raise ValueError(f"Map ID {map_id} not found. Available maps: {list(map_files.keys())}")
    
    map_file_path = map_files[map_id]
    
    # Create environment for specific map
    env = SinglePolicyOffloadingEnvironment(
        resource_cluster=resource_cluster,
        batch_size=50,  # Smaller batch for fine-tuning
        graph_number=50,  # Fewer graphs for fine-tuning
        graph_file_paths=[map_file_path],
        time_major=False
    )
    
    return env, map_file_path

def finetune_policy(map_id, weight_path, steps=20, learning_rate=1e-4, batch_size=50):
    """
    Fine-tune the policy on a specific map
    
    Args:
        map_id: ID of the specific map
        weight_path: Path to pre-trained weights
        steps: Number of fine-tuning steps
        learning_rate: Learning rate for fine-tuning (usually lower than pre-training)
        batch_size: Batch size for fine-tuning
        
    Returns:
        tuple: (policy, weight_path, metadata_path)
    """
    
    print("=" * 80)
    print("FINE-TUNING DRL POLICY ON SPECIFIC MAP")
    print("=" * 80)
    print(f"Map ID: {map_id}")
    print(f"Pre-trained weights: {weight_path}")
    print(f"Fine-tuning steps: {steps}")
    print(f"Learning rate: {learning_rate}")
    print(f"Batch size: {batch_size}")
    print()
    
    # Set TensorFlow logging level
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    
    # Configure logger
    log_dir = f"./finetuning_logs/map_{map_id}_steps_{steps}"
    logger.configure(dir=log_dir, format_strs=['stdout', 'log', 'csv'])
    
    # Initialize weight manager
    weight_manager = WeightManager()
    
    # Initialize MEC resource cluster
    print("Initializing MEC resource cluster...")
    resource_cluster = Resources(
        mec_process_capable=(10.0 * 1024 * 1024),
        mobile_process_capable=(1.0 * 1024 * 1024),
        bandwidth_up=7.0,
        bandwidth_dl=7.0
    )
    
    # Create environment for specific map
    print(f"Creating environment for map {map_id}...")
    env, map_file_path = create_single_map_environment(map_id, resource_cluster)
    print(f"Map file: {map_file_path}")
    
    # Calculate baseline solutions for this specific map
    print("Calculating baseline solutions for this map...")
    action, greedy_finish_time = env.greedy_solution()
    print(f"Greedy solution latency: {np.mean(greedy_finish_time):.4f}")
    
    finish_time = env.get_all_mec_execute_time()
    print(f"All-remote solution latency: {np.mean(finish_time):.4f}")
    
    finish_time = env.get_all_locally_execute_time()
    print(f"All-local solution latency: {np.mean(finish_time):.4f}")
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
    
    # Load pre-trained weights
    print(f"Loading pre-trained weights from: {weight_path}")
    if not weight_manager.load_weights(policy, weight_path):
        print("❌ Failed to load pre-trained weights!")
        return None, None, None
    
    # Sampler
    sampler = Seq2SeqSampler(
        env=env,
        policy=policy,
        rollouts_per_task=1,
        max_path_length=20000,
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
    
    # PPO algorithm with lower learning rate for fine-tuning
    algo = SinglePolicyPPO(
        policy=policy,
        sampler=sampler,
        sampler_process=sample_processor,
        lr=learning_rate,  # Lower learning rate for fine-tuning
        num_grad_steps=2,  # Fewer gradient steps for fine-tuning
        clip_value=0.1,    # Smaller clipping for fine-tuning
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
        n_itr=steps,
        greedy_finish_time=greedy_finish_time,
        start_itr=0,
        batch_size=batch_size,
        save_interval=steps  # Save at the end
    )
    
    print("Starting fine-tuning...")
    print("=" * 80)
    
    # Start fine-tuning
    with tf.compat.v1.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        try:
            # Fine-tune the policy
            avg_ret, avg_loss, avg_latencies = trainer.train()
            
            # Save fine-tuned weights
            print("\nSaving fine-tuned weights...")
            weight_path, metadata_path = weight_manager.save_finetuned_weights(
                policy=policy,
                map_id=map_id,
                iteration=steps,
                additional_info={
                    'pretrained_weights': weight_path,
                    'final_avg_reward': avg_ret[-1] if avg_ret else 0.0,
                    'final_avg_loss': avg_loss[-1] if avg_loss else 0.0,
                    'final_avg_latency': avg_latencies[-1] if avg_latencies else 0.0,
                    'learning_rate': learning_rate,
                    'batch_size': batch_size,
                    'map_file': map_file_path
                }
            )
            
            print("=" * 80)
            print("FINE-TUNING COMPLETED SUCCESSFULLY!")
            print("=" * 80)
            print(f"Map ID: {map_id}")
            print(f"Fine-tuning steps: {steps}")
            print(f"Final average reward: {avg_ret[-1]:.4f}")
            print(f"Final average loss: {avg_loss[-1]:.4f}")
            print(f"Final average latency: {avg_latencies[-1]:.4f}")
            print(f"Fine-tuned weights saved to: {weight_path}")
            print(f"Metadata saved to: {metadata_path}")
            print(f"Logs saved to: {log_dir}")
            
            return policy, weight_path, metadata_path
            
        except KeyboardInterrupt:
            print("\nFine-tuning interrupted by user.")
            print("Saving current weights...")
            weight_path, metadata_path = weight_manager.save_finetuned_weights(
                policy=policy,
                map_id=map_id,
                iteration=trainer.start_itr,
                additional_info={'status': 'interrupted'}
            )
            print(f"Weights saved to: {weight_path}")
            return policy, weight_path, metadata_path
            
        except Exception as e:
            print(f"\nFine-tuning failed with error: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None, None

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Fine-tune DRL policy on specific map')
    parser.add_argument('--map_id', type=int, required=True, help='ID of the specific map to fine-tune on')
    parser.add_argument('--weights', type=str, required=True, help='Path to pre-trained weights')
    parser.add_argument('--steps', type=int, default=20, help='Number of fine-tuning steps')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for fine-tuning')
    parser.add_argument('--batch_size', type=int, default=50, help='Batch size for fine-tuning')
    
    args = parser.parse_args()
    
    # Check if weight file exists
    if not os.path.exists(args.weights):
        print(f"❌ Weight file not found: {args.weights}")
        return 1
    
    # Fine-tune the policy
    policy, weight_path, metadata_path = finetune_policy(
        map_id=args.map_id,
        weight_path=args.weights,
        steps=args.steps,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size
    )
    
    if policy is not None:
        print("\n🎉 Fine-tuning completed successfully!")
        print(f"You can now evaluate the fine-tuned policy:")
        print(f"  python evaluate_policy.py --weights {weight_path}")
        return 0
    else:
        print("\n❌ Fine-tuning failed!")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
