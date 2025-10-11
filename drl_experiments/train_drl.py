"""
Main training script for DRL PPO experiments.
"""

import os
import sys
import argparse
import numpy as np
import tensorflow as tf
import csv
import time
from collections import defaultdict

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.mec_offloaing_envs.offloading_env import Resources, OffloadingEnvironment
from .configs import *
from .policy import DRLPolicy
from .rollout import collect_rollouts, batch_rollouts
from .fast_rollout import collect_fast_rollouts, batch_fast_rollouts
from .ppo import train_one_epoch, compute_makespan
from .gae import compute_gae, normalize_advantages


def set_seed(seed):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    tf.set_random_seed(seed)


def create_environment():
    """Create OffloadingEnvironment with training maps."""
    resource_cluster = Resources(
        mec_process_capable=(10.0 * 1024 * 1024),
        mobile_process_capable=(1.0 * 1024 * 1024),
        bandwidth_up=7.0,
        bandwidth_dl=7.0
    )
    
    env = OffloadingEnvironment(
        resource_cluster=resource_cluster,
        batch_size=1,  # Single rollout at a time
        graph_number=100,
        graph_file_paths=TRAIN_GRAPH_PATHS,
        time_major=time_major
    )
    
    return env


def log_metrics(epoch, metrics, csv_writer, start_time):
    """Log training metrics to CSV and stdout."""
    elapsed_time = time.time() - start_time
    
    # Print to stdout
    print(f"Epoch {epoch:3d}: "
          f"Policy Loss={metrics['policy_loss']:.4f}, "
          f"Value Loss={metrics['value_loss']:.4f}, "
          f"Entropy={metrics['entropy']:.4f}, "
          f"KL={metrics['approx_kl']:.4f}, "
          f"Clip Frac={metrics['clip_frac']:.4f}, "
          f"Time={elapsed_time:.1f}s")
    
    # Write to CSV
    csv_writer.writerow({
        'epoch': epoch,
        'policy_loss': metrics['policy_loss'],
        'value_loss': metrics['value_loss'],
        'entropy': metrics['entropy'],
        'approx_kl': metrics['approx_kl'],
        'clip_frac': metrics['clip_frac'],
        'elapsed_time': elapsed_time
    })


def save_checkpoint(policy, epoch, output_dir):
    """Save policy checkpoint."""
    checkpoint_path = os.path.join(output_dir, f"ckpt_epoch_{epoch}.ckpt")
    
    # Create checkpoint directory
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    # Save variables
    sess = tf.get_default_session()
    variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=policy.scope_name)
    ps = sess.run(variables)
    save_dict = {v.name: value for v, value in zip(variables, ps)}
    
    import joblib
    joblib.dump(save_dict, checkpoint_path)
    
    print(f"Checkpoint saved: {checkpoint_path}")


def main():
    """Main training loop."""
    parser = argparse.ArgumentParser(description='DRL PPO Training')
    parser.add_argument('--config', type=str, default='drl_experiments.configs',
                       help='Configuration module')
    parser.add_argument('--tasks_per_epoch', type=int, default=tasks_per_epoch,
                       help='Number of tasks per epoch')
    parser.add_argument('--rollouts_per_task', type=int, default=rollouts_per_task,
                       help='Number of rollouts per task')
    parser.add_argument('--ppo_epochs', type=int, default=ppo_epochs,
                       help='PPO epochs per batch')
    parser.add_argument('--minibatch_size', type=int, default=minibatch_size,
                       help='Minibatch size')
    parser.add_argument('--lr', type=float, default=lr,
                       help='Learning rate')
    parser.add_argument('--seed', type=int, default=seed,
                       help='Random seed')
    parser.add_argument('--num_epochs', type=int, default=num_epochs,
                       help='Number of training epochs')
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create environment
    print("Creating environment...")
    env = create_environment()
    print(f"Environment created with {len(TRAIN_TASK_IDS)} training maps")
    
    # Create policy
    print("Creating DRL policy...")
    policy = DRLPolicy(
        obs_dim=17,
        action_dim=2,
        encoder_units=encoder_units,
        decoder_units=decoder_units,
        num_layers=num_layers
    )
    
    # Create optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=args.lr)
    
    # Create CSV logger
    csv_path = os.path.join(output_dir, "training_log.csv")
    csv_file = open(csv_path, 'w', newline='')
    fieldnames = ['epoch', 'policy_loss', 'value_loss', 'entropy', 'approx_kl', 'clip_frac', 'elapsed_time']
    csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    csv_writer.writeheader()
    
    # Training configuration
    config = {
        'gamma': gamma,
        'gae_lambda': gae_lambda,
        'clip_ratio': clip_ratio,
        'vf_coef': vf_coef,
        'ent_coef': ent_coef,
        'max_grad_norm': max_grad_norm,
        'ppo_epochs': args.ppo_epochs,
        'minibatch_size': args.minibatch_size
    }
    
    print(f"Starting training for {args.num_epochs} epochs...")
    print(f"Tasks per epoch: {args.tasks_per_epoch}")
    print(f"Rollouts per task: {args.rollouts_per_task}")
    print(f"PPO epochs: {args.ppo_epochs}")
    print(f"Minibatch size: {args.minibatch_size}")
    print(f"Learning rate: {args.lr}")
    
    start_time = time.time()
    
    # Training loop
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for epoch in range(1, args.num_epochs + 1):
            epoch_start_time = time.time()
            
            # Sample tasks for this epoch
            sampled_tasks = np.random.choice(TRAIN_TASK_IDS, 
                                           size=min(args.tasks_per_epoch, len(TRAIN_TASK_IDS)),
                                           replace=False)
            
            print(f"Epoch {epoch}: Sampling tasks {sampled_tasks}")
            
            # Collect rollouts (back to real rollout for quality)
            all_rollouts = []
            for task_id in sampled_tasks:
                task_rollouts = collect_rollouts(env, policy, [task_id], args.rollouts_per_task)
                all_rollouts.extend(task_rollouts)
            
            if not all_rollouts:
                print(f"Warning: No rollouts collected in epoch {epoch}")
                continue
            
            # Batch rollouts (back to real batching)
            batch_data = batch_rollouts(all_rollouts)
            
            if batch_data is None:
                print(f"Warning: Failed to batch rollouts in epoch {epoch}")
                continue
            
            # Compute makespan metrics
            makespans = []
            for rollout in all_rollouts:
                makespan = compute_makespan(rollout['rewards'])
                makespans.append(makespan)
            
            avg_makespan = np.mean(makespans)
            std_makespan = np.std(makespans)
            
            print(f"Epoch {epoch}: Collected {len(all_rollouts)} rollouts, "
                  f"Avg Makespan: {avg_makespan:.4f} ± {std_makespan:.4f}")
            
            # Train policy
            try:
                metrics = train_one_epoch(policy, optimizer, batch_data, config)
                
                # Log metrics
                log_metrics(epoch, metrics, csv_writer, start_time)
                csv_file.flush()
                
            except Exception as e:
                print(f"Error in training epoch {epoch}: {e}")
                continue
            
            # Save checkpoint
            if epoch % save_interval == 0:
                save_checkpoint(policy, epoch, output_dir)
        
        # Save final checkpoint
        save_checkpoint(policy, args.num_epochs, output_dir)
    
    csv_file.close()
    
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.1f} seconds")
    print(f"Training log saved: {csv_path}")
    print(f"Checkpoints saved in: {output_dir}")


if __name__ == "__main__":
    main()
