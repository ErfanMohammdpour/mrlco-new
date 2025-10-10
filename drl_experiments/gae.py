"""
Generalized Advantage Estimation (GAE) computation for batched sequences.
"""

import numpy as np
import tensorflow as tf


def compute_gae(rewards, values, gamma=0.99, lam=0.95):
    """
    Compute GAE advantages and returns for batched sequences.
    
    Args:
        rewards: list of arrays, each of shape (T_i,) - step rewards for each trajectory
        values: list of arrays, each of shape (T_i,) - value estimates for each trajectory  
        gamma: discount factor
        lam: GAE lambda parameter
        
    Returns:
        advantages: list of arrays, each of shape (T_i,) - computed advantages
        returns: list of arrays, each of shape (T_i,) - computed returns
    """
    advantages = []
    returns = []
    
    for reward_seq, value_seq in zip(rewards, values):
        T = len(reward_seq)
        
        # Initialize arrays
        adv = np.zeros(T)
        ret = np.zeros(T)
        
        # Compute advantages using GAE
        gae = 0
        for t in reversed(range(T)):
            if t == T - 1:
                # Last timestep: no bootstrap
                delta = reward_seq[t] - value_seq[t]
            else:
                # Bootstrap with next value
                delta = reward_seq[t] + gamma * value_seq[t + 1] - value_seq[t]
            
            gae = delta + gamma * lam * gae
            adv[t] = gae
            
        # Compute returns from advantages
        ret = adv + value_seq
        
        advantages.append(adv)
        returns.append(ret)
    
    return advantages, returns


def compute_gae_tf(rewards, values, gamma=0.99, lam=0.95):
    """
    TensorFlow version of GAE computation for batched sequences.
    
    Args:
        rewards: tensor of shape (batch_size, max_seq_len) - padded reward sequences
        values: tensor of shape (batch_size, max_seq_len) - padded value sequences
        gamma: discount factor
        lam: GAE lambda parameter
        
    Returns:
        advantages: tensor of shape (batch_size, max_seq_len) - computed advantages
        returns: tensor of shape (batch_size, max_seq_len) - computed returns
    """
    batch_size = tf.shape(rewards)[0]
    max_seq_len = tf.shape(rewards)[1]
    
    # Initialize output tensors
    advantages = tf.zeros_like(rewards)
    returns = tf.zeros_like(rewards)
    
    # Process each sequence in the batch
    def compute_sequence_gae(i):
        reward_seq = rewards[i]
        value_seq = values[i]
        
        # Compute GAE for this sequence
        gae = tf.zeros([])
        adv_seq = tf.zeros_like(reward_seq)
        
        # Reverse iteration through time steps
        for t in reversed(range(max_seq_len)):
            if t == max_seq_len - 1:
                # Last timestep: no bootstrap
                delta = reward_seq[t] - value_seq[t]
            else:
                # Bootstrap with next value
                delta = reward_seq[t] + gamma * value_seq[t + 1] - value_seq[t]
            
            gae = delta + gamma * lam * gae
            adv_seq = tf.concat([adv_seq[:t], [gae], adv_seq[t+1:]], axis=0)
        
        # Compute returns
        ret_seq = adv_seq + value_seq
        
        return adv_seq, ret_seq
    
    # Apply to all sequences in batch
    advantages_list = []
    returns_list = []
    
    for i in range(batch_size):
        adv, ret = compute_sequence_gae(i)
        advantages_list.append(adv)
        returns_list.append(ret)
    
    advantages = tf.stack(advantages_list)
    returns = tf.stack(returns_list)
    
    return advantages, returns


def normalize_advantages(advantages, epsilon=1e-8):
    """
    Normalize advantages across the entire batch.
    
    Args:
        advantages: list of arrays - computed advantages for each trajectory
        epsilon: small constant for numerical stability
        
    Returns:
        normalized_advantages: list of arrays - normalized advantages
    """
    # Flatten all advantages
    all_adv = np.concatenate(advantages)
    
    # Compute normalization statistics
    adv_mean = np.mean(all_adv)
    adv_std = np.std(all_adv)
    
    # Normalize each trajectory
    normalized_advantages = []
    for adv in advantages:
        normalized_adv = (adv - adv_mean) / (adv_std + epsilon)
        normalized_advantages.append(normalized_adv)
    
    return normalized_advantages
