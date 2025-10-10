"""
Simple Deep RL Evaluation Script
Loads trained model and prints/saves results
"""

import tensorflow as tf
import numpy as np
import os
import time
import joblib
from deep_rl_offloading import DeepRLOffloadingAgent
from env.mec_offloaing_envs.offloading_env import Resources, OffloadingEnvironment


def main():
    """Simple evaluation function"""
    
    print("="*60)
    print("SIMPLE DEEP RL EVALUATION")
    print("="*60)
    
    # Configuration
    n_eval_tasks = 50
    max_episode_length = 50
    
    # Check if specific model path is provided
    import sys
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        if not os.path.exists(model_path):
            print(f"❌ Model file not found: {model_path}")
            return
        print(f"Loading specified model: {model_path}")
    else:
        # Find the latest model
        model_dir = "./simple_model/"
        if not os.path.exists(model_dir):
            print("❌ Model directory not found: ./simple_model/")
            return
        
        # Find the latest checkpoint
        checkpoints = [f for f in os.listdir(model_dir) if f.endswith('.ckpt')]
        if not checkpoints:
            print("❌ No model checkpoints found in ./simple_model/")
            return
        
        # Get the latest checkpoint
        latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]) if 'episode' in x else 0)
        model_path = os.path.join(model_dir, latest_checkpoint)
        print(f"Loading latest model: {model_path}")
    
    # Create environment
    print("Creating environment...")
    resource_cluster = Resources(
        mec_process_capable=(10.0 * 1024 * 1024),
        mobile_process_capable=(1.0 * 1024 * 1024),
        bandwidth_up=7.0,
        bandwidth_dl=7.0
    )
    
    env = OffloadingEnvironment(
        resource_cluster=resource_cluster,
        batch_size=1,
        graph_number=100,
        graph_file_paths=[
                "./env/mec_offloaing_envs/data/dags/offloading_random_15/offloading_random_15.20.",
        ],
        time_major=False
    )
    
    # Create agent
    print("Creating Deep RL agent...")
    agent = DeepRLOffloadingAgent(
        obs_dim=17,
        action_dim=2,
        encoder_units=128,
        decoder_units=128,
        learning_rate=3e-4,
        gamma=0.99,
        tau=0.005,
        epsilon_start=0.0,  # No exploration during evaluation
        epsilon_end=0.0,
        epsilon_decay=1.0,
        buffer_size=1000,
        batch_size=16,
        num_layers=1  # Use single layer to avoid tuple state issue
    )
    
    # Evaluation results
    eval_rewards = []
    eval_latencies = []
    eval_lengths = []
    eval_task_ids = []
    
    print(f"Evaluating on {n_eval_tasks} tasks...")
    
    # Run evaluation
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        # Initialize target networks
        agent.initialize_target_networks()
        
        agent.load_model(model_path)
        print("✅ Model loaded successfully!")
        
        # Sample random tasks for evaluation
        task_ids = env.sample_tasks(n_eval_tasks)
        
        for i, task_id in enumerate(task_ids):
            env.set_task(task_id)
            obs = env.reset()
            sequence_length = np.array([obs.shape[1]] * obs.shape[0], dtype=np.int32)
            
            episode_reward = 0
            episode_length = 0
            episode_latency = 0
            
            # Run episode
            for step in range(max_episode_length):
                # Get action (greedy, no exploration)
                actions = agent.get_action(obs, sequence_length, training=False)
                
                # Take step in environment
                next_obs, rewards, done, info = env.step(actions)
                
                episode_reward += np.sum(rewards)
                episode_length += 1
                episode_latency = info[0] if isinstance(info, list) else info
                
                obs = next_obs
                
                if done:
                    break
            
            # Store results
            eval_rewards.append(episode_reward)
            eval_latencies.append(episode_latency)
            eval_lengths.append(episode_length)
            eval_task_ids.append(task_id)
            
            # Print result for each step
            print(f"Task {i+1:2d}/{n_eval_tasks} (ID: {task_id:2d}): "
                  f"Reward={episode_reward:8.6f}, "
                  f"Latency={episode_latency:8.6f}, "
                  f"Length={episode_length:2d}")
    
    # Calculate statistics
    reward_mean = np.mean(eval_rewards)
    reward_std = np.std(eval_rewards)
    latency_mean = np.mean(eval_latencies)
    latency_std = np.std(eval_latencies)
    length_mean = np.mean(eval_lengths)
    length_std = np.std(eval_lengths)
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Model: {model_path}")
    print(f"Tasks evaluated: {n_eval_tasks}")
    print(f"Evaluation time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("-"*60)
    print("PERFORMANCE METRICS:")
    print(f"  Average Reward: {reward_mean:.6f} ± {reward_std:.6f}")
    print(f"  Average Latency: {latency_mean:.6f} ± {latency_std:.6f}")
    print(f"  Average Length: {length_mean:.2f} ± {length_std:.2f}")
    print(f"  Min Reward: {np.min(eval_rewards):.6f}")
    print(f"  Max Reward: {np.max(eval_rewards):.6f}")
    print(f"  Min Latency: {np.min(eval_latencies):.6f}")
    print(f"  Max Latency: {np.max(eval_latencies):.6f}")
    print("="*60)
    
    # Save results
    eval_results = {
        'model_path': model_path,
        'evaluation_time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'n_tasks': n_eval_tasks,
        'rewards': eval_rewards,
        'latencies': eval_latencies,
        'lengths': eval_lengths,
        'task_ids': eval_task_ids,
        'statistics': {
            'reward_mean': reward_mean,
            'reward_std': reward_std,
            'latency_mean': latency_mean,
            'latency_std': latency_std,
            'length_mean': length_mean,
            'length_std': length_std,
            'reward_min': np.min(eval_rewards),
            'reward_max': np.max(eval_rewards),
            'latency_min': np.min(eval_latencies),
            'latency_max': np.max(eval_latencies)
        }
    }
    
    # Save to file
    os.makedirs("./simple_evaluation_results/", exist_ok=True)
    results_file = f"./simple_evaluation_results/evaluation_{int(time.time())}.pkl"
    joblib.dump(eval_results, results_file)
    print(f"Results saved to: {results_file}")
    
    # Also save as text file for easy reading
    text_file = f"./simple_evaluation_results/evaluation_{int(time.time())}.txt"
    with open(text_file, 'w') as f:
        f.write("DEEP RL EVALUATION RESULTS\n")
        f.write("="*60 + "\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Tasks evaluated: {n_eval_tasks}\n")
        f.write(f"Evaluation time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("-"*60 + "\n")
        f.write("PERFORMANCE METRICS:\n")
        f.write(f"  Average Reward: {reward_mean:.6f} ± {reward_std:.6f}\n")
        f.write(f"  Average Latency: {latency_mean:.6f} ± {latency_std:.6f}\n")
        f.write(f"  Average Length: {length_mean:.2f} ± {length_std:.2f}\n")
        f.write(f"  Min Reward: {np.min(eval_rewards):.6f}\n")
        f.write(f"  Max Reward: {np.max(eval_rewards):.6f}\n")
        f.write(f"  Min Latency: {np.min(eval_latencies):.6f}\n")
        f.write(f"  Max Latency: {np.max(eval_latencies):.6f}\n")
        f.write("="*60 + "\n")
        f.write("\nDETAILED RESULTS:\n")
        f.write("Task_ID\tReward\t\tLatency\t\tLength\n")
        f.write("-"*60 + "\n")
        for i, (task_id, reward, latency, length) in enumerate(zip(eval_task_ids, eval_rewards, eval_latencies, eval_lengths)):
            f.write(f"{task_id}\t\t{reward:.6f}\t{latency:.6f}\t{length}\n")
    
    print(f"Text results saved to: {text_file}")
    print("✅ Evaluation completed!")


if __name__ == "__main__":
    main()
