"""
Ultra-fast rollout collection for testing.
"""

import numpy as np
import tensorflow as tf


def collect_fast_rollout(env, policy, task_id, max_steps=5):
    """
    Collect a single rollout with minimal computation.
    """
    # Set task and reset environment
    env.set_task(task_id)
    obs = env.reset()
    
    # Ensure obs is in correct format [B=1, T, F]
    if len(obs.shape) == 2:
        obs = obs[np.newaxis, :, :]
    
    batch_size, seq_len, feat_dim = obs.shape
    
    # Limit sequence length for speed
    seq_len = min(seq_len, max_steps)
    
    # Generate random actions (for speed testing)
    actions = np.random.randint(0, 2, size=seq_len)
    log_probs = np.random.uniform(-2, 0, size=seq_len)
    values = np.random.uniform(-10, 10, size=seq_len)
    
    # Execute actions
    _, rewards, done, info = env.step([actions])
    
    # Extract rewards
    if isinstance(rewards, list):
        rewards = rewards[0]
    rewards = np.array(rewards)
    if len(rewards.shape) > 1:
        rewards = rewards.flatten()
    
    # Ensure same length
    min_len = min(len(actions), len(rewards))
    actions = actions[:min_len]
    log_probs = log_probs[:min_len]
    values = values[:min_len]
    rewards = rewards[:min_len]
    
    rollout_dict = {
        'obs': obs[0, :min_len, :],
        'actions': actions,
        'log_probs': log_probs,
        'values': values,
        'rewards': rewards,
        'length': min_len
    }
    
    return rollout_dict


def collect_fast_rollouts(env, policy, task_ids, rollouts_per_task):
    """
    Collect multiple fast rollouts.
    """
    all_rollouts = []
    
    for task_id in task_ids:
        for _ in range(rollouts_per_task):
            try:
                rollout = collect_fast_rollout(env, policy, task_id)
                all_rollouts.append(rollout)
            except Exception as e:
                print(f"Warning: Failed to collect rollout for task {task_id}: {e}")
                continue
    
    return all_rollouts


def batch_fast_rollouts(rollouts):
    """
    Convert rollouts to batched format.
    """
    if not rollouts:
        return None
    
    # Concatenate all rollouts
    all_obs = np.concatenate([r['obs'] for r in rollouts], axis=0)
    all_actions = np.concatenate([r['actions'] for r in rollouts], axis=0)
    all_log_probs = np.concatenate([r['log_probs'] for r in rollouts], axis=0)
    all_values = np.concatenate([r['values'] for r in rollouts], axis=0)
    all_rewards = np.concatenate([r['rewards'] for r in rollouts], axis=0)
    
    batch_data = {
        'obs': all_obs,
        'actions': all_actions,
        'log_probs': all_log_probs,
        'values': all_values,
        'rewards': all_rewards
    }
    
    return batch_data
