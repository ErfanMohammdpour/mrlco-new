#!/usr/bin/env python3
"""
Evaluation Script for DRL Policy

This script loads trained weights (pre-trained or fine-tuned) and evaluates
the policy performance on specific maps.

Usage:
    python evaluate_policy.py --weights path/to/weights.ckpt --map_id 1 --episodes 10
"""

import tensorflow as tf
import numpy as np
import os
import sys
import argparse
from datetime import datetime

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from env.single_policy_offloading_env import SinglePolicyOffloadingEnvironment, Resources
from policies.meta_seq2seq_policy import Seq2SeqPolicy
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
        batch_size=1,  # Single episode evaluation
        graph_number=1,  # Single graph for evaluation
        graph_file_paths=[map_file_path],
        time_major=False
    )
    
    return env, map_file_path

def evaluate_policy(weight_path, map_id, episodes=10, render=False):
    """
    Evaluate the policy on a specific map
    
    Args:
        weight_path: Path to trained weights
        map_id: ID of the specific map
        episodes: Number of episodes to evaluate
        render: Whether to render the environment
        
    Returns:
        dict: Evaluation results
    """
    
    print("=" * 80)
    print("EVALUATING DRL POLICY")
    print("=" * 80)
    print(f"Weight file: {weight_path}")
    print(f"Map ID: {map_id}")
    print(f"Episodes: {episodes}")
    print(f"Render: {render}")
    print()
    
    # Set TensorFlow logging level
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    
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
    greedy_latency = np.mean(greedy_finish_time)
    print(f"Greedy solution latency: {greedy_latency:.4f}")
    
    finish_time = env.get_all_mec_execute_time()
    all_remote_latency = np.mean(finish_time)
    print(f"All-remote solution latency: {all_remote_latency:.4f}")
    
    finish_time = env.get_all_locally_execute_time()
    all_local_latency = np.mean(finish_time)
    print(f"All-local solution latency: {all_local_latency:.4f}")
    print()
    
    # Initialize policy
    print("Initializing policy...")
    policy = Seq2SeqPolicy(
        obs_dim=17,
        encoder_units=128,
        decoder_units=128,
        vocab_size=2
    )
    
    # Load trained weights
    print(f"Loading weights from: {weight_path}")
    if not weight_manager.load_weights(policy, weight_path):
        print("❌ Failed to load weights!")
        return None
    
    # Load metadata if available
    metadata_path = weight_path.replace('.ckpt', '_metadata.pkl')
    metadata = None
    if os.path.exists(metadata_path):
        metadata = weight_manager.load_metadata(metadata_path)
        if metadata:
            print(f"Loaded metadata: {metadata}")
    
    print("Starting evaluation...")
    print("=" * 80)
    
    # Evaluation results
    results = {
        'episodes': [],
        'rewards': [],
        'latencies': [],
        'actions': [],
        'greedy_latency': greedy_latency,
        'all_remote_latency': all_remote_latency,
        'all_local_latency': all_local_latency,
        'metadata': metadata
    }
    
    # Start evaluation
    with tf.compat.v1.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        try:
            for episode in range(episodes):
                print(f"Episode {episode + 1}/{episodes}")
                
                # Reset environment
                obs = env.reset()
                if render:
                    print(f"  Initial observation shape: {obs.shape}")
                
                # Run episode
                episode_rewards = []
                episode_latencies = []
                episode_actions = []
                
                done = False
                step = 0
                
                while not done and step < 1000:  # Max steps per episode
                    # Get action from policy
                    actions, logits, values = policy.get_actions(obs)
                    
                    # Take action
                    next_obs, rewards, done, infos = env.step(actions)
                    
                    # Store results
                    episode_rewards.append(np.mean(rewards))
                    episode_latencies.append(np.mean(infos))
                    episode_actions.append(actions[0] if len(actions) > 0 else [])
                    
                    if render:
                        print(f"    Step {step}: Action={actions[0] if len(actions) > 0 else 'N/A'}, "
                              f"Reward={np.mean(rewards):.4f}, Latency={np.mean(infos):.4f}")
                    
                    obs = next_obs
                    step += 1
                
                # Store episode results
                episode_reward = np.sum(episode_rewards)
                episode_latency = np.mean(episode_latencies)
                
                results['episodes'].append(episode + 1)
                results['rewards'].append(episode_reward)
                results['latencies'].append(episode_latency)
                results['actions'].append(episode_actions)
                
                print(f"  Episode {episode + 1} completed:")
                print(f"    Total reward: {episode_reward:.4f}")
                print(f"    Average latency: {episode_latency:.4f}")
                print(f"    Steps: {step}")
                print()
            
            # Calculate summary statistics
            avg_reward = np.mean(results['rewards'])
            std_reward = np.std(results['rewards'])
            avg_latency = np.mean(results['latencies'])
            std_latency = np.std(results['latencies'])
            
            # Calculate performance metrics
            reward_improvement = (avg_reward - (-1.0)) / 1.0 * 100  # Assuming reward range [-1, 0]
            latency_improvement = (greedy_latency - avg_latency) / greedy_latency * 100
            
            print("=" * 80)
            print("EVALUATION RESULTS")
            print("=" * 80)
            print(f"Map ID: {map_id}")
            print(f"Episodes evaluated: {episodes}")
            print(f"Average reward: {avg_reward:.4f} ± {std_reward:.4f}")
            print(f"Average latency: {avg_latency:.4f} ± {std_latency:.4f}")
            print()
            print("Baseline comparisons:")
            print(f"  Greedy solution latency: {greedy_latency:.4f}")
            print(f"  All-remote solution latency: {all_remote_latency:.4f}")
            print(f"  All-local solution latency: {all_local_latency:.4f}")
            print()
            print("Performance improvements:")
            print(f"  Reward improvement: {reward_improvement:.2f}%")
            print(f"  Latency improvement over greedy: {latency_improvement:.2f}%")
            print()
            
            # Save results
            results['summary'] = {
                'avg_reward': avg_reward,
                'std_reward': std_reward,
                'avg_latency': avg_latency,
                'std_latency': std_latency,
                'reward_improvement': reward_improvement,
                'latency_improvement': latency_improvement,
                'greedy_latency': greedy_latency,
                'all_remote_latency': all_remote_latency,
                'all_local_latency': all_local_latency
            }
            
            return results
            
        except Exception as e:
            print(f"\nEvaluation failed with error: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Evaluate DRL policy on specific map')
    parser.add_argument('--weights', type=str, required=True, help='Path to trained weights')
    parser.add_argument('--map_id', type=int, required=True, help='ID of the specific map to evaluate on')
    parser.add_argument('--episodes', type=int, default=10, help='Number of episodes to evaluate')
    parser.add_argument('--render', action='store_true', help='Whether to render the environment')
    
    args = parser.parse_args()
    
    # Check if weight file exists
    if not os.path.exists(args.weights):
        print(f"❌ Weight file not found: {args.weights}")
        return 1
    
    # Evaluate the policy
    results = evaluate_policy(
        weight_path=args.weights,
        map_id=args.map_id,
        episodes=args.episodes,
        render=args.render
    )
    
    if results is not None:
        print("\n🎉 Evaluation completed successfully!")
        return 0
    else:
        print("\n❌ Evaluation failed!")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
