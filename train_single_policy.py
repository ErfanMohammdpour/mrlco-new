#!/usr/bin/env python3
"""
Single-Policy RL Training Script for Task Offloading

This script trains a single policy using PPO for the task offloading problem
in Mobile Edge Computing (MEC) environments.

Usage:
    python train_single_policy.py

The script will:
1. Initialize the MEC environment with task graphs
2. Create a single Seq2Seq policy
3. Train using PPO algorithm
4. Save the trained model
5. Generate training reports
"""

import tensorflow as tf
import numpy as np
import os
import sys

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from single_policy_trainer import SinglePolicyTrainer
from env.single_policy_offloading_env import SinglePolicyOffloadingEnvironment, Resources
from policies.meta_seq2seq_policy import Seq2SeqPolicy
from samplers.seq2seq_sampler import Seq2SeqSampler
from samplers.seq2seq_sampler_process import Seq2SeqSamplerProcessor
from baselines.vf_baseline import ValueFunctionBaseline
from single_policy_ppo import SinglePolicyPPO
from utils import logger

def main():
    """Main training function for single-policy RL"""
    
    # Set TensorFlow logging level
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    
    # Configure logger
    logger.configure(dir="./single_policy_offloading_log/", format_strs=['stdout', 'log', 'csv'])
    
    print("=" * 80)
    print("SINGLE-POLICY RL TRAINING FOR TASK OFFLOADING")
    print("=" * 80)
    
    # Training hyperparameters
    BATCH_SIZE = 100  # Number of trajectories per iteration
    ROLLOUTS_PER_TASK = 1  # Number of rollouts per task
    MAX_PATH_LENGTH = 20000  # Maximum episode length
    N_ITERATIONS = 3500  # Number of training iterations
    LEARNING_RATE = 5e-4  # Learning rate for PPO
    SAVE_INTERVAL = 100  # Save model every N iterations
    
    print(f"Training Configuration:")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Rollouts per Task: {ROLLOUTS_PER_TASK}")
    print(f"  Max Path Length: {MAX_PATH_LENGTH}")
    print(f"  Iterations: {N_ITERATIONS}")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print(f"  Save Interval: {SAVE_INTERVAL}")
    print()
    
    # Initialize MEC resource cluster
    print("Initializing MEC resource cluster...")
    resource_cluster = Resources(
        mec_process_capable=(10.0 * 1024 * 1024),  # 10 MB/s processing capacity
        mobile_process_capable=(1.0 * 1024 * 1024),  # 1 MB/s processing capacity
        bandwidth_up=7.0,  # 7 Mbps upload bandwidth
        bandwidth_dl=7.0   # 7 Mbps download bandwidth
    )
    
    # Define task graph file paths
    graph_file_paths = [
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
    
    # Initialize single-policy environment
    print("Initializing single-policy offloading environment...")
    env = SinglePolicyOffloadingEnvironment(
        resource_cluster=resource_cluster,
        batch_size=BATCH_SIZE,
        graph_number=100,
        graph_file_paths=graph_file_paths,
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
    
    # Single policy (not meta-policy)
    policy = Seq2SeqPolicy(
        obs_dim=17,  # Observation dimension
        encoder_units=128,  # Encoder hidden units
        decoder_units=128,  # Decoder hidden units
        vocab_size=2  # Action vocabulary size (0=local, 1=remote)
    )
    
    # Sampler for single-policy RL
    sampler = Seq2SeqSampler(
        env=env,
        policy=policy,
        rollouts_per_task=ROLLOUTS_PER_TASK,
        max_path_length=MAX_PATH_LENGTH,
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
    
    # PPO algorithm for single-policy RL
    algo = SinglePolicyPPO(
        policy=policy,
        sampler=sampler,
        sampler_process=sample_processor,
        lr=LEARNING_RATE,
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
        n_itr=N_ITERATIONS,
        greedy_finish_time=greedy_finish_time,
        start_itr=0,
        batch_size=1000,
        save_interval=SAVE_INTERVAL
    )
    
    print("Starting training...")
    print("=" * 80)
    
    # Create model save directory
    os.makedirs("./single_policy_model", exist_ok=True)
    
    # Start training
    with tf.compat.v1.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        try:
            avg_ret, avg_loss, avg_latencies = trainer.train()
            
            print("=" * 80)
            print("TRAINING COMPLETED SUCCESSFULLY!")
            print("=" * 80)
            print(f"Final average reward: {avg_ret[-1]:.4f}")
            print(f"Final average loss: {avg_loss[-1]:.4f}")
            print(f"Final average latency: {avg_latencies[-1]:.4f}")
            print(f"Model saved to: ./single_policy_model/")
            print(f"Logs saved to: ./single_policy_offloading_log/")
            
        except KeyboardInterrupt:
            print("\nTraining interrupted by user.")
            print("Saving current model...")
            policy.save_variables(save_path="./single_policy_model/single_policy_interrupted.ckpt")
            print("Model saved to: ./single_policy_model/single_policy_interrupted.ckpt")
            
        except Exception as e:
            print(f"\nTraining failed with error: {str(e)}")
            import traceback
            traceback.print_exc()
            return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
