import numpy as np
import tensorflow as tf
from utils import logger
from collections import OrderedDict
import multiprocessing
from samplers.base import SampleProcessor


class Seq2SeqMetaSamplerProcessorOptimized(SampleProcessor):
    """
    Optimized sample processor with GPU-friendly operations and reduced CPU overhead
    
    Key optimizations:
    - Vectorized advantage computation
    - TF operations for GPU acceleration
    - Batch processing of paths
    - Reduced numpy-to-tensor conversions
    """

    def __init__(self,
                 baseline,
                 discount=0.99,
                 gae_lambda=1,
                 normalize_adv=False,
                 positive_adv=False,
                 use_tf_functions=True):
        
        self.baseline = baseline
        self.discount = discount
        self.gae_lambda = gae_lambda
        self.normalize_adv = normalize_adv
        self.positive_adv = positive_adv
        self.use_tf_functions = use_tf_functions

    @tf.function
    def _compute_advantages_tf(self, rewards, values, discount, gae_lambda):
        """TF function for computing advantages on GPU"""
        # Compute discounted returns
        returns = tf.TensorArray(dtype=tf.float32, size=tf.shape(rewards)[0])
        advantages = tf.TensorArray(dtype=tf.float32, size=tf.shape(rewards)[0])
        
        # Initialize last values
        last_return = 0.0
        last_value = 0.0
        last_advantage = 0.0
        
        # Reverse iterate for advantage computation
        for t in tf.range(tf.shape(rewards)[0] - 1, -1, -1):
            returns = returns.write(t, rewards[t] + discount * last_return)
            td_error = rewards[t] + discount * last_value - values[t]
            advantages = advantages.write(t, td_error + discount * gae_lambda * last_advantage)
            
            last_return = returns.read(t)
            last_value = values[t]
            last_advantage = advantages.read(t)
        
        return returns.stack(), advantages.stack()

    def process_samples(self, paths, log=False, log_prefix=''):
        """
        Optimized sample processing with vectorized operations
        
        Args:
            paths (dict): Dictionary of paths collected for each meta-task
            log (str): Whether to log statistics
            log_prefix (str): Prefix for logging
        
        Returns:
            samples_data (dict): Processed data ready for training
        """
        samples_data = OrderedDict()
        all_path_baselines = []

        # Process each meta-task's paths
        for meta_task_id, meta_paths in paths.items():
            samples_data[meta_task_id] = self._process_meta_task_batch(meta_paths)
            
        # Fit baseline with all paths if needed
        if log == 'all' or log == 'post':
            logger.logkv(log_prefix + "AveragePolicyStd", 
                        np.mean([np.std(samples_data[id]['actions']) for id in samples_data.keys()]))
            logger.logkv(log_prefix + "AverageReturn", 
                        np.mean([np.mean(np.sum(samples_data[id]['rewards'], axis=-1)) for id in samples_data.keys()]))

        return samples_data

    def _process_meta_task_batch(self, paths):
        """Process all paths for a meta task using vectorized operations"""
        if not paths:
            return {}
            
        # Stack all path data for vectorized processing
        observations = []
        actions = []
        logits = []
        rewards = []
        values = []
        finish_times = []
        
        for path in paths:
            observations.append(path['observations'])
            actions.append(path['actions'])
            logits.append(path['logits'])
            rewards.append(path['rewards'])
            values.append(path['values'])
            finish_times.append(path['finish_time'])
        
        # Convert to numpy arrays for efficient processing
        observations = np.array(observations, dtype=np.float32)
        actions = np.array(actions, dtype=np.int32)
        logits = np.array(logits, dtype=np.float32)
        rewards = np.array(rewards, dtype=np.float32)
        values = np.array(values, dtype=np.float32)
        finish_times = np.array(finish_times, dtype=np.float32)
        
        # Compute returns and advantages
        if self.use_tf_functions:
            # Use TF operations for GPU acceleration
            returns, advantages = self._compute_batch_advantages_tf(rewards, values)
        else:
            # Fallback to numpy
            returns, advantages = self._compute_batch_advantages_np(rewards, values)
        
        # Normalize advantages if requested
        if self.normalize_adv:
            advantages = self._normalize_advantages(advantages)
        
        if self.positive_adv:
            advantages = np.maximum(advantages, 0)
        
        # Prepare decoder inputs (shifted actions)
        decoder_inputs = self._prepare_decoder_inputs_vectorized(actions)
        
        # Get sequence lengths
        decoder_full_length = np.full(actions.shape[0], actions.shape[1], dtype=np.int32)
        
        return {
            'observations': observations,
            'actions': actions,
            'decoder_inputs': decoder_inputs,
            'decoder_full_length': decoder_full_length,
            'logits': logits,
            'values': values,
            'returns': returns,
            'advantages': advantages,
            'rewards': rewards,
            'finish_time': finish_times
        }

    def _compute_batch_advantages_tf(self, rewards, values):
        """Compute advantages for a batch using TF operations"""
        batch_size = rewards.shape[0]
        returns_list = []
        advantages_list = []
        
        # Process each trajectory in the batch
        for i in range(batch_size):
            if self.use_tf_functions:
                ret, adv = self._compute_advantages_tf(
                    tf.constant(rewards[i]), 
                    tf.constant(values[i]),
                    tf.constant(self.discount),
                    tf.constant(self.gae_lambda)
                )
                returns_list.append(ret.numpy())
                advantages_list.append(adv.numpy())
            else:
                ret, adv = self._compute_advantages_np_single(rewards[i], values[i])
                returns_list.append(ret)
                advantages_list.append(adv)
        
        return np.array(returns_list), np.array(advantages_list)

    def _compute_batch_advantages_np(self, rewards, values):
        """Compute advantages for a batch using numpy (fallback)"""
        batch_size = rewards.shape[0]
        returns = np.zeros_like(rewards)
        advantages = np.zeros_like(rewards)
        
        for i in range(batch_size):
            returns[i], advantages[i] = self._compute_advantages_np_single(rewards[i], values[i])
        
        return returns, advantages

    def _compute_advantages_np_single(self, rewards, values):
        """Compute advantages for a single trajectory"""
        T = len(rewards)
        returns = np.zeros(T)
        advantages = np.zeros(T)
        
        last_return = 0
        last_value = 0
        last_advantage = 0
        
        for t in reversed(range(T)):
            returns[t] = rewards[t] + self.discount * last_return
            td_error = rewards[t] + self.discount * last_value - values[t]
            advantages[t] = td_error + self.discount * self.gae_lambda * last_advantage
            
            last_return = returns[t]
            last_value = values[t]
            last_advantage = advantages[t]
        
        return returns, advantages

    def _normalize_advantages(self, advantages):
        """Normalize advantages with numerical stability"""
        mean = np.mean(advantages)
        std = np.std(advantages)
        return (advantages - mean) / (std + 1e-8)

    def _prepare_decoder_inputs_vectorized(self, actions):
        """Prepare decoder inputs (shifted actions) using vectorized operations"""
        batch_size, seq_len = actions.shape
        decoder_inputs = np.zeros_like(actions)
        
        # Set first token to start token (0)
        decoder_inputs[:, 0] = 0
        
        # Shift actions by one position
        if seq_len > 1:
            decoder_inputs[:, 1:] = actions[:, :-1]
        
        return decoder_inputs