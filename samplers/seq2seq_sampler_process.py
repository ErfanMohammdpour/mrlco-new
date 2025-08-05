"""
Single trajectory sampler processor for evaluation.
This is a simplified version of the meta sampler processor.
"""
import numpy as np
import tensorflow as tf
from utils import utils


class Seq2SeSamplerProcessor(object):
    """
    Processes samples from a single task (non-meta) sampler.
    """
    def __init__(self, baseline, discount=0.99, gae_lambda=0.95, normalize_adv=True,
                 positive_adv=False, env=None):
        self.baseline = baseline
        self.discount = discount
        self.gae_lambda = gae_lambda
        self.normalize_adv = normalize_adv
        self.positive_adv = positive_adv
        self.env = env

    def process_samples(self, paths, log=False, log_prefix=''):
        """
        Process samples from a single task.
        
        Args:
            paths: List of trajectory dictionaries
            log: Whether to log statistics
            log_prefix: Prefix for logging
            
        Returns:
            Dictionary of processed samples
        """
        # Concatenate paths
        observations = np.concatenate([path["observations"] for path in paths])
        actions = np.concatenate([path["actions"] for path in paths])
        rewards = np.concatenate([path["rewards"] for path in paths])
        agent_infos = utils.concat_tensor_dict_list([path["agent_infos"] for path in paths])
        env_infos = utils.concat_tensor_dict_list([path["env_infos"] for path in paths])
        
        # Fit baseline if provided
        if self.baseline is not None:
            self.baseline.fit(paths)
            
        # Compute returns and advantages
        all_returns = []
        all_advantages = []
        
        for path in paths:
            path_rewards = path["rewards"]
            path_length = len(path_rewards)
            
            # Compute returns
            returns = utils.discount_cumsum(path_rewards, self.discount)
            
            # Compute advantages
            if self.baseline is not None:
                path_baselines = self.baseline.predict(path)
                deltas = path_rewards[:-1] + self.discount * path_baselines[1:] - path_baselines[:-1]
                deltas = np.append(deltas, path_rewards[-1] - path_baselines[-1])
                advantages = utils.discount_cumsum(deltas, self.discount * self.gae_lambda)
            else:
                advantages = returns
                
            all_returns.append(returns)
            all_advantages.append(advantages)
            
        returns = np.concatenate(all_returns)
        advantages = np.concatenate(all_advantages)
        
        # Normalize advantages
        if self.normalize_adv:
            advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
            
        # Make advantages positive if requested
        if self.positive_adv:
            advantages = np.exp(advantages)
            
        # Create decoder inputs (shifted actions)
        decoder_inputs = []
        decoder_full_lengths = []
        
        for path in paths:
            path_actions = path["actions"]
            path_length = len(path_actions)
            
            # Create decoder inputs (prepend with start token)
            start_token = 0  # Assuming 0 is the start token
            decoder_input = np.concatenate([[start_token], path_actions[:-1]])
            decoder_inputs.append(decoder_input)
            decoder_full_lengths.append(path_length)
            
        decoder_inputs = np.concatenate(decoder_inputs)
        decoder_full_lengths = np.array(decoder_full_lengths)
        
        # Create samples dictionary
        samples_data = {
            "observations": observations,
            "actions": actions,
            "decoder_inputs": decoder_inputs,
            "decoder_full_length": decoder_full_lengths,
            "advantages": advantages,
            "returns": returns,
            "rewards": rewards,
            "agent_infos": agent_infos,
            "env_infos": env_infos
        }
        
        if log:
            # Log statistics
            logger = utils.get_logger()
            logger.record_tabular(log_prefix + "AverageReturn", np.mean([sum(path["rewards"]) for path in paths]))
            logger.record_tabular(log_prefix + "MaxReturn", np.max([sum(path["rewards"]) for path in paths]))
            logger.record_tabular(log_prefix + "MinReturn", np.min([sum(path["rewards"]) for path in paths]))
            logger.record_tabular(log_prefix + "NumTrajs", len(paths))
            
        return samples_data