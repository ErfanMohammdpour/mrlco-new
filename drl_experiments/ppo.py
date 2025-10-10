"""
Proximal Policy Optimization (PPO) implementation for DRL training.
"""

import numpy as np
import tensorflow as tf
from .gae import compute_gae, normalize_advantages


def compute_ppo_loss(old_log_probs, new_log_probs, advantages, returns, values, 
                    clip_ratio=0.2, vf_coef=0.5, ent_coef=0.02):
    """
    Compute PPO loss components.
    
    Args:
        old_log_probs: old policy log probabilities [B, T]
        new_log_probs: new policy log probabilities [B, T]
        advantages: computed advantages [B, T]
        returns: computed returns [B, T]
        values: value estimates [B, T]
        clip_ratio: PPO clipping ratio
        vf_coef: value function coefficient
        ent_coef: entropy coefficient
        
    Returns:
        loss_dict: dictionary containing loss components and metrics
    """
    # Compute probability ratio
    ratio = tf.exp(new_log_probs - old_log_probs)
    
    # Policy loss with clipping
    surr1 = ratio * advantages
    surr2 = tf.clip_by_value(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages
    policy_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))
    
    # Value function loss
    value_loss = 0.5 * tf.reduce_mean(tf.square(returns - values))
    
    # Entropy bonus
    entropy = -tf.reduce_mean(new_log_probs)
    
    # Total loss
    total_loss = policy_loss + vf_coef * value_loss - ent_coef * entropy
    
    # Metrics
    approx_kl = tf.reduce_mean(old_log_probs - new_log_probs)
    clip_frac = tf.reduce_mean(tf.cast(tf.greater(tf.abs(ratio - 1.0), clip_ratio), tf.float32))
    
    loss_dict = {
        'policy_loss': policy_loss,
        'value_loss': value_loss,
        'entropy': entropy,
        'total_loss': total_loss,
        'approx_kl': approx_kl,
        'clip_frac': clip_frac
    }
    
    return loss_dict


def train_one_epoch(policy, optimizer, batch_data, config):
    """
    Train policy for one epoch using PPO.
    
    Args:
        policy: DRLPolicy instance
        optimizer: TensorFlow optimizer
        batch_data: batched rollout data
        config: configuration dictionary
        
    Returns:
        metrics: dictionary containing training metrics
    """
    obs = batch_data['obs']  # [B, T, F]
    actions = batch_data['actions']  # [B, T]
    old_log_probs = batch_data['log_probs']  # [B, T]
    rewards = batch_data['rewards']  # [B, T]
    old_values = batch_data['values']  # [B, T]
    masks = batch_data['masks']  # [B, T]
    
    # Compute GAE advantages and returns
    advantages_list = []
    returns_list = []
    
    batch_size = obs.shape[0]
    for i in range(batch_size):
        # Extract valid sequence for this trajectory
        valid_length = int(np.sum(masks[i]))
        reward_seq = rewards[i, :valid_length]
        value_seq = old_values[i, :valid_length]
        
        # Compute GAE for this trajectory
        adv, ret = compute_gae([reward_seq], [value_seq], 
                              gamma=config['gamma'], 
                              lam=config['gae_lambda'])
        
        # Pad back to original length
        padded_adv = np.zeros_like(rewards[i])
        padded_ret = np.zeros_like(rewards[i])
        padded_adv[:valid_length] = adv[0]
        padded_ret[:valid_length] = ret[0]
        
        advantages_list.append(padded_adv)
        returns_list.append(padded_ret)
    
    # Normalize advantages
    advantages = normalize_advantages(advantages_list)
    advantages = np.stack(advantages)  # [B, T]
    returns = np.stack(returns_list)  # [B, T]
    
    # Flatten for minibatch training
    flat_obs = obs.reshape(-1, obs.shape[-1])  # [B*T, F]
    flat_actions = actions.flatten()  # [B*T]
    flat_old_log_probs = old_log_probs.flatten()  # [B*T]
    flat_advantages = advantages.flatten()  # [B*T]
    flat_returns = returns.flatten()  # [B*T]
    flat_old_values = old_values.flatten()  # [B*T]
    flat_masks = masks.flatten()  # [B*T]
    
    # Create dataset for minibatch training
    dataset_size = len(flat_obs)
    indices = np.arange(dataset_size)
    
    epoch_metrics = {
        'policy_loss': [],
        'value_loss': [],
        'entropy': [],
        'approx_kl': [],
        'clip_frac': []
    }
    
    # Minibatch training
    minibatch_size = config['minibatch_size']
    num_minibatches = dataset_size // minibatch_size
    
    for _ in range(config['ppo_epochs']):
        # Shuffle indices
        np.random.shuffle(indices)
        
        for i in range(num_minibatches):
            start_idx = i * minibatch_size
            end_idx = start_idx + minibatch_size
            
            batch_indices = indices[start_idx:end_idx]
            
            # Extract minibatch
            mb_obs = flat_obs[batch_indices]
            mb_actions = flat_actions[batch_indices]
            mb_old_log_probs = flat_old_log_probs[batch_indices]
            mb_advantages = flat_advantages[batch_indices]
            mb_returns = flat_returns[batch_indices]
            mb_old_values = flat_old_values[batch_indices]
            mb_masks = flat_masks[batch_indices]
            
            # Reshape for policy evaluation
            mb_obs = mb_obs.reshape(-1, 1, mb_obs.shape[-1])  # [mb_size, 1, F]
            
            # Evaluate current policy
            new_log_probs, entropy, new_values = policy.evaluate_actions(
                mb_obs, mb_actions.reshape(-1, 1)
            )
            
            # Reshape outputs
            new_log_probs = new_log_probs.flatten()
            entropy = entropy.flatten()
            new_values = new_values.flatten()
            
            # Compute PPO loss
            loss_dict = compute_ppo_loss(
                mb_old_log_probs, new_log_probs, mb_advantages, 
                mb_returns, new_values,
                clip_ratio=config['clip_ratio'],
                vf_coef=config['vf_coef'],
                ent_coef=config['ent_coef']
            )
            
            # Apply gradients
            gradients = tf.gradients(loss_dict['total_loss'], policy.trainable_variables)
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, config['max_grad_norm'])
            
            optimizer.apply_gradients(zip(clipped_gradients, policy.trainable_variables))
            
            # Store metrics
            for key in epoch_metrics:
                epoch_metrics[key].append(loss_dict[key].numpy())
    
    # Average metrics across minibatches
    avg_metrics = {}
    for key in epoch_metrics:
        avg_metrics[key] = np.mean(epoch_metrics[key])
    
    return avg_metrics


def compute_makespan(rewards):
    """
    Compute makespan from reward sequence.
    Makespan is typically the negative of the final reward or cumulative reward.
    
    Args:
        rewards: reward sequence [T]
        
    Returns:
        makespan: computed makespan
    """
    # For this environment, makespan is typically the cumulative reward
    # (negative because rewards are latency deltas)
    return -np.sum(rewards)
