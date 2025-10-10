"""
Evaluation script for DRL PPO experiments.
Supports both zero-shot evaluation and fine-tuning.
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
from .ppo import train_one_epoch, compute_makespan


def set_seed(seed):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    tf.set_random_seed(seed)


def create_eval_environment():
    """Create OffloadingEnvironment with evaluation maps."""
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
        graph_file_paths=EVAL_GRAPH_PATHS,
        time_major=time_major
    )
    
    return env


def load_policy(checkpoint_path, scope_name="drl_policy"):
    """Load policy from checkpoint."""
    policy = DRLPolicy(
        obs_dim=17,
        action_dim=2,
        encoder_units=encoder_units,
        decoder_units=decoder_units,
        num_layers=num_layers,
        scope_name=scope_name
    )
    
    # Load checkpoint
    sess = tf.get_default_session()
    variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope_name)
    
    import joblib
    loaded_params = joblib.load(checkpoint_path)
    
    restores = []
    for v in variables:
        if v.name in loaded_params:
            restores.append(v.assign(loaded_params[v.name]))
    
    sess.run(restores)
    
    return policy


def evaluate_zero_shot(policy, env, eval_task_ids, num_rollouts=10):
    """Evaluate policy without fine-tuning."""
    print("Running zero-shot evaluation...")
    
    results = {}
    
    for task_id in eval_task_ids:
        print(f"Evaluating task {task_id}...")
        
        # Collect rollouts
        rollouts = collect_rollouts(env, policy, [task_id], num_rollouts)
        
        if not rollouts:
            print(f"Warning: No rollouts collected for task {task_id}")
            continue
        
        # Compute metrics
        makespans = []
        rewards = []
        
        for rollout in rollouts:
            makespan = compute_makespan(rollout['rewards'])
            total_reward = np.sum(rollout['rewards'])
            
            makespans.append(makespan)
            rewards.append(total_reward)
        
        results[task_id] = {
            'makespan_mean': np.mean(makespans),
            'makespan_std': np.std(makespans),
            'reward_mean': np.mean(rewards),
            'reward_std': np.std(rewards),
            'num_rollouts': len(rollouts)
        }
        
        print(f"Task {task_id}: Makespan={np.mean(makespans):.4f}±{np.std(makespans):.4f}, "
              f"Reward={np.mean(rewards):.4f}±{np.std(rewards):.4f}")
    
    return results


def evaluate_finetune(policy, env, eval_task_ids, finetune_steps=20, num_rollouts=10):
    """Evaluate policy with fine-tuning."""
    print(f"Running fine-tuning evaluation with {finetune_steps} steps...")
    
    # Create optimizer for fine-tuning
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    
    # Training configuration
    config = {
        'gamma': gamma,
        'gae_lambda': gae_lambda,
        'clip_ratio': clip_ratio,
        'vf_coef': vf_coef,
        'ent_coef': ent_coef,
        'max_grad_norm': max_grad_norm,
        'ppo_epochs': 1,  # Single epoch per step
        'minibatch_size': minibatch_size
    }
    
    results = {}
    
    for task_id in eval_task_ids:
        print(f"Fine-tuning on task {task_id}...")
        
        # Store metrics over fine-tuning steps
        step_metrics = {
            'step': [],
            'makespan_mean': [],
            'makespan_std': [],
            'reward_mean': [],
            'reward_std': [],
            'policy_loss': [],
            'value_loss': []
        }
        
        # Initial evaluation
        rollouts = collect_rollouts(env, policy, [task_id], num_rollouts)
        if rollouts:
            makespans = [compute_makespan(r['rewards']) for r in rollouts]
            rewards = [np.sum(r['rewards']) for r in rollouts]
            
            step_metrics['step'].append(0)
            step_metrics['makespan_mean'].append(np.mean(makespans))
            step_metrics['makespan_std'].append(np.std(makespans))
            step_metrics['reward_mean'].append(np.mean(rewards))
            step_metrics['reward_std'].append(np.std(rewards))
            step_metrics['policy_loss'].append(0.0)
            step_metrics['value_loss'].append(0.0)
        
        # Fine-tuning steps
        for step in range(1, finetune_steps + 1):
            # Collect rollouts for training
            train_rollouts = collect_rollouts(env, policy, [task_id], rollouts_per_task)
            
            if train_rollouts:
                # Batch rollouts
                batch_data = batch_rollouts(train_rollouts)
                
                if batch_data is not None:
                    # Train policy
                    try:
                        metrics = train_one_epoch(policy, optimizer, batch_data, config)
                        
                        # Evaluate after training
                        eval_rollouts = collect_rollouts(env, policy, [task_id], num_rollouts)
                        
                        if eval_rollouts:
                            makespans = [compute_makespan(r['rewards']) for r in eval_rollouts]
                            rewards = [np.sum(r['rewards']) for r in eval_rollouts]
                            
                            step_metrics['step'].append(step)
                            step_metrics['makespan_mean'].append(np.mean(makespans))
                            step_metrics['makespan_std'].append(np.std(makespans))
                            step_metrics['reward_mean'].append(np.mean(rewards))
                            step_metrics['reward_std'].append(np.std(rewards))
                            step_metrics['policy_loss'].append(metrics['policy_loss'])
                            step_metrics['value_loss'].append(metrics['value_loss'])
                            
                            print(f"Step {step}: Makespan={np.mean(makespans):.4f}±{np.std(makespans):.4f}, "
                                  f"Reward={np.mean(rewards):.4f}±{np.std(rewards):.4f}")
                    
                    except Exception as e:
                        print(f"Error in fine-tuning step {step}: {e}")
                        continue
        
        results[task_id] = step_metrics
        
        # Print final results
        if step_metrics['makespan_mean']:
            final_makespan = step_metrics['makespan_mean'][-1]
            final_reward = step_metrics['reward_mean'][-1]
            print(f"Task {task_id} final: Makespan={final_makespan:.4f}, Reward={final_reward:.4f}")
    
    return results


def save_results(results, mode, output_path):
    """Save evaluation results to CSV."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', newline='') as csvfile:
        if mode == 'zero_shot':
            fieldnames = ['map_id', 'mode', 'makespan_mean', 'makespan_std', 
                         'reward_mean', 'reward_std', 'num_rollouts']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for task_id, metrics in results.items():
                writer.writerow({
                    'map_id': task_id,
                    'mode': mode,
                    'makespan_mean': metrics['makespan_mean'],
                    'makespan_std': metrics['makespan_std'],
                    'reward_mean': metrics['reward_mean'],
                    'reward_std': metrics['reward_std'],
                    'num_rollouts': metrics['num_rollouts']
                })
        
        elif mode == 'finetune':
            fieldnames = ['map_id', 'mode', 'step', 'makespan_mean', 'makespan_std',
                         'reward_mean', 'reward_std', 'policy_loss', 'value_loss']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for task_id, step_metrics in results.items():
                for i in range(len(step_metrics['step'])):
                    writer.writerow({
                        'map_id': task_id,
                        'mode': mode,
                        'step': step_metrics['step'][i],
                        'makespan_mean': step_metrics['makespan_mean'][i],
                        'makespan_std': step_metrics['makespan_std'][i],
                        'reward_mean': step_metrics['reward_mean'][i],
                        'reward_std': step_metrics['reward_std'][i],
                        'policy_loss': step_metrics['policy_loss'][i],
                        'value_loss': step_metrics['value_loss'][i]
                    })


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='DRL PPO Evaluation')
    parser.add_argument('--ckpt', type=str, required=True,
                       help='Path to checkpoint file')
    parser.add_argument('--mode', type=str, choices=['zero_shot', 'finetune'], 
                       default='zero_shot', help='Evaluation mode')
    parser.add_argument('--finetune_steps', type=int, default=20,
                       help='Number of fine-tuning steps')
    parser.add_argument('--num_rollouts', type=int, default=10,
                       help='Number of evaluation rollouts')
    parser.add_argument('--eval_tasks', type=int, nargs='+', default=EVAL_TASK_IDS,
                       help='Task IDs to evaluate')
    parser.add_argument('--seed', type=int, default=seed,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Create evaluation environment
    print("Creating evaluation environment...")
    env = create_eval_environment()
    print(f"Environment created with {len(args.eval_tasks)} evaluation tasks")
    
    # Load policy
    print(f"Loading policy from {args.ckpt}...")
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        policy = load_policy(args.ckpt)
        print("Policy loaded successfully!")
        
        # Run evaluation
        if args.mode == 'zero_shot':
            results = evaluate_zero_shot(policy, env, args.eval_tasks, args.num_rollouts)
            output_path = os.path.join(eval_dir, "eval_results_zero_shot.csv")
        elif args.mode == 'finetune':
            results = evaluate_finetune(policy, env, args.eval_tasks, 
                                      args.finetune_steps, args.num_rollouts)
            output_path = os.path.join(eval_dir, "eval_results_finetune.csv")
        
        # Save results
        save_results(results, args.mode, output_path)
        print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
