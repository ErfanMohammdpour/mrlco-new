"""
Rollout collection utilities for DRL PPO training.
Handles autoregressive action generation and environment interaction.
"""

import numpy as np
import tensorflow as tf


def collect_rollout(env, policy, task_id, max_steps=50):
    """
    Collect a single rollout for a given task.
    
    Args:
        env: OffloadingEnvironment instance
        policy: DRLPolicy instance
        task_id: task/map ID to collect rollout for
        max_steps: maximum number of steps in rollout
        
    Returns:
        rollout_dict: dictionary containing rollout data
            - obs: observations [T, F]
            - actions: action sequence [T]
            - log_probs: log probabilities [T]
            - rewards: step rewards [T]
            - values: value estimates [T]
            - length: sequence length
    """
    # Set task and reset environment
    env.set_task(task_id)
    obs = env.reset()  # Shape: [B=1, T, F] or [T, F]
    
    # Ensure obs is in correct format [B=1, T, F]
    if len(obs.shape) == 2:
        obs = obs[np.newaxis, :, :]  # Add batch dimension
    
    batch_size, seq_len, feat_dim = obs.shape
    
    # Initialize rollout storage
    actions = []
    log_probs = []
    values = []
    
    # Autoregressive action generation
    sess = tf.get_default_session()
    
    for t in range(seq_len):
        # Get current observation slice
        current_obs = obs[:, :t+1, :]  # [B=1, t+1, F]
        
        # Sample action for timestep t
        action, log_prob, value = policy.sample_action(current_obs, t)
        
        # Evaluate tensors to get actual values
        action_val, log_prob_val, value_val = sess.run([action, log_prob, value])
        
        actions.append(action_val[0])  # Remove batch dimension
        log_probs.append(log_prob_val[0])
        values.append(value_val[0])
    
    # Convert to arrays
    actions = np.array(actions)  # [T]
    log_probs = np.array(log_probs)  # [T]
    values = np.array(values)  # [T]
    
    # Execute full action sequence in environment
    env.set_task(task_id)  # Ensure correct task
    _, rewards, done, info = env.step([actions])  # env.step expects list of action sequences
    
    # Extract reward sequence (should be length T)
    if isinstance(rewards, list):
        rewards = rewards[0]  # Take first (and only) reward sequence
    
    rewards = np.array(rewards)  # Could be [T] or [1, T]
    
    # Ensure rewards is 1D
    if len(rewards.shape) > 1:
        rewards = rewards.flatten()  # [T]
    
    # Ensure all arrays have same length
    min_len = min(len(actions), len(rewards), len(values))
    actions = actions[:min_len]
    log_probs = log_probs[:min_len]
    values = values[:min_len]
    rewards = rewards[:min_len]
    
    rollout_dict = {
        'obs': obs[0],  # Remove batch dimension: [T, F]
        'actions': actions,  # [T]
        'log_probs': log_probs,  # [T]
        'rewards': rewards,  # [T]
        'values': values,  # [T]
        'length': min_len
    }
    
    return rollout_dict


def collect_rollouts(env, policy, task_ids, rollouts_per_task=3):
    """
    Collect multiple rollouts for multiple tasks.
    
    Args:
        env: OffloadingEnvironment instance
        policy: DRLPolicy instance
        task_ids: list of task IDs to collect rollouts for
        rollouts_per_task: number of rollouts per task
        
    Returns:
        all_rollouts: list of rollout dictionaries
    """
    all_rollouts = []
    
    for task_id in task_ids:
        for _ in range(rollouts_per_task):
            try:
                rollout = collect_rollout(env, policy, task_id)
                all_rollouts.append(rollout)
            except Exception as e:
                print(f"Warning: Failed to collect rollout for task {task_id}: {e}")
                continue
    
    return all_rollouts


def batch_rollouts(rollouts):
    """
    Convert list of rollouts into batched format for PPO training.
    
    Args:
        rollouts: list of rollout dictionaries
        
    Returns:
        batched_data: dictionary containing batched rollout data
    """
    if not rollouts:
        return None
    
    # Find maximum sequence length
    max_len = max(rollout['length'] for rollout in rollouts)
    batch_size = len(rollouts)
    feat_dim = rollouts[0]['obs'].shape[1]
    
    # Initialize batched arrays
    obs_batch = np.zeros((batch_size, max_len, feat_dim))
    actions_batch = np.zeros((batch_size, max_len), dtype=np.int32)
    log_probs_batch = np.zeros((batch_size, max_len))
    rewards_batch = np.zeros((batch_size, max_len))
    values_batch = np.zeros((batch_size, max_len))
    masks_batch = np.zeros((batch_size, max_len), dtype=np.bool)
    
    # Fill batched arrays
    for i, rollout in enumerate(rollouts):
        length = rollout['length']
        
        obs_batch[i, :length] = rollout['obs'][:length]
        actions_batch[i, :length] = rollout['actions'][:length]
        log_probs_batch[i, :length] = rollout['log_probs'][:length]
        rewards_batch[i, :length] = rollout['rewards'][:length]
        values_batch[i, :length] = rollout['values'][:length]
        masks_batch[i, :length] = True
    
    batched_data = {
        'obs': obs_batch,  # [B, T, F]
        'actions': actions_batch,  # [B, T]
        'log_probs': log_probs_batch,  # [B, T]
        'rewards': rewards_batch,  # [B, T]
        'values': values_batch,  # [B, T]
        'masks': masks_batch,  # [B, T]
        'lengths': [rollout['length'] for rollout in rollouts]
    }
    
    return batched_data
