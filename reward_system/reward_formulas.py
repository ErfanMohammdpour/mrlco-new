"""
Reward Formula Implementations

Contains 5 reward function implementations:
1. Linear Reward (default)
2. Logarithmic Reward
3. Exponential Reward
4. Temporal Difference Reward
5. Adaptive Difficulty Reward
"""

import numpy as np
from reward_system.base_reward import BaseReward


class LinearReward(BaseReward):
    """
    Formula 1: Linear Reward (Default)
    
    Simple linear normalization: R = -(cost - min_time) / (max_time - min_time)
    """
    
    def compute(self, cost, max_time, min_time, **kwargs):
        """
        Compute linear reward.
        
        Args:
            cost: Incremental latency
            max_time: Maximum possible latency
            min_time: Minimum possible latency
        
        Returns:
            Reward in [-1, 0] range
        """
        self.validate_inputs(cost, max_time, min_time)
        
        cost_range = max_time - min_time
        if cost_range < 1e-8:
            return np.zeros_like(cost) if isinstance(cost, (list, np.ndarray)) else 0.0
        
        return -(cost - min_time) / cost_range
    
    def get_name(self):
        return "linear"
    
    def get_params(self):
        return {}

class LogarithmicReward(BaseReward):
    def compute(self, cost, max_time, min_time, log_base=10.0, epsilon=1e-6, **kwargs):
        self.validate_inputs(cost, max_time, min_time)

        cost = np.asarray(cost, dtype=np.float64)

        cost_range = max_time - min_time
        if cost_range <= 0:
            return np.zeros_like(cost)

        normalized_cost = cost / cost_range         
        normalized_cost = np.clip(normalized_cost, 0.0, 1.0 - epsilon)

        if log_base == 'e' or log_base == np.e:
            log_func = np.log
            base_log_eps = -np.log(epsilon)
        else:
            log_base_val = np.float64(log_base)
            log_func = lambda x: np.log(x) / np.log(log_base_val)
            base_log_eps = -log_func(epsilon)


        log_reward = -log_func(1.0 - normalized_cost + epsilon) 


        log_reward = log_reward / base_log_eps

        reward = -log_reward

        return reward
    def get_name(self):
        return "logarithmic"
    
    def get_params(self):
        return {'log_base': 10.0, 'epsilon': 1e-6}

# class LogarithmicReward(BaseReward):
#     """
#     Formula 2: Logarithmic Reward
    
#     Provides better gradient signals near optimal solutions.
#     R = -log(1 - c_norm + ε) / log(base)
#     """
    
#     def compute(self, cost, max_time, min_time, log_base=10.0, epsilon=1e-6, **kwargs):
#         """
#         Compute logarithmic reward.
        
#         Args:
#             cost: Incremental latency
#             max_time: Maximum possible latency
#             min_time: Minimum possible latency
#             log_base: Logarithm base (default: 10.0)
#             epsilon: Small constant for numerical stability (default: 1e-6)
        
#         Returns:
#             Reward in approximately [-1, 0] range
#         """
#         self.validate_inputs(cost, max_time, min_time)
        
#         # Handle array inputs
#         is_array = isinstance(cost, (list, np.ndarray))
#         if is_array:
#             cost = np.array(cost, dtype=np.float64)
#         else:
#             cost = np.float64(cost)
        
#         # Normalize cost
#         cost_range = max_time - min_time
#         if cost_range < epsilon:
#             return np.zeros_like(cost) if is_array else 0.0
        
#         normalized_cost = (cost - min_time) / cost_range
#         normalized_cost = np.clip(normalized_cost, epsilon, 1.0 - epsilon)
        
#         # Logarithmic transformation
#         if log_base == 'e' or log_base == np.e:
#             log_func = np.log
#         else:
#             log_base_val = np.float64(log_base)
#             log_func = lambda x: np.log(x) / np.log(log_base_val)
        
#         # Compute logarithmic reward
#         log_reward = -log_func(1.0 - normalized_cost + epsilon)
        
#         # Normalize to [-1, 0] range
#         log_reward_max = -log_func(epsilon)  # Best case
#         log_reward_worst = -log_func(2.0 * epsilon)  # Worst case
        
#         normalized_reward = (log_reward - log_reward_max) / (log_reward_worst - log_reward_max)
        
#         return normalized_reward
    
#     def get_name(self):
#         return "logarithmic"
    
#     def get_params(self):
#         return {'log_base': 10.0, 'epsilon': 1e-6}


# class ExponentialReward(BaseReward):
#     """
#     Formula 3: Exponential Reward
    
#     Creates sharp distinctions between good and bad solutions.
#     R = -(1 - exp(-T · c_norm^p))
#     """
    
#     def compute(self, cost, max_time, min_time, temperature=2.0, exponent=2.0, **kwargs):
#         """
#         Compute exponential reward.
        
#         Args:
#             cost: Incremental latency
#             max_time: Maximum possible latency
#             min_time: Minimum possible latency
#             temperature: Temperature parameter (default: 2.0)
#             exponent: Exponent for exponential scaling (default: 2.0)
        
#         Returns:
#             Reward in [-1, 0] range
#         """
#         self.validate_inputs(cost, max_time, min_time)
        
#         # Normalize cost
#         cost_range = max_time - min_time
#         if cost_range < 1e-8:
#             return np.zeros_like(cost) if isinstance(cost, (list, np.ndarray)) else 0.0
        
#         normalized_cost = (cost - min_time) / cost_range
#         normalized_cost = np.clip(normalized_cost, 0, 1)
        
#         # Exponential scaling
#         exp_reward = np.exp(-temperature * (normalized_cost ** exponent))
        
#         # Normalize to [-1, 0] range
#         reward = -(1 - exp_reward)
        
#         return reward
    
#     def get_name(self):
#         return "exponential"
    
#     def get_params(self):
#         return {'temperature': 2.0, 'exponent': 2.0}


# class TemporalDifferenceReward(BaseReward):
#     """
#     Formula 4: Temporal Difference Reward
    
#     Rewards based on improvement rate over time.
#     R = (1-w) · R_base + w · improvement_rate
#     """
    
#     def compute(self, cost, max_time, min_time, previous_cost=None, improvement_weight=0.4, **kwargs):
#         """
#         Compute temporal difference reward.
        
#         Args:
#             cost: Current incremental latency
#             max_time: Maximum possible latency
#             min_time: Minimum possible latency
#             previous_cost: Cost from previous step/episode (optional)
#             improvement_weight: Weight for improvement component (default: 0.4)
        
#         Returns:
#             Reward value
#         """
#         self.validate_inputs(cost, max_time, min_time)
        
#         # Base reward (linear)
#         cost_range = max_time - min_time
#         if cost_range < 1e-8:
#             return np.zeros_like(cost) if isinstance(cost, (list, np.ndarray)) else 0.0
        
#         base_reward = -(cost - min_time) / cost_range
        
#         # Improvement component
#         if previous_cost is not None:
#             cost_diff = previous_cost - cost  # Positive = improvement
#             normalized_improvement = cost_diff / cost_range
#             improvement_reward = normalized_improvement * 2.0  # Amplify improvement signal
            
#             # Combine
#             reward = (1 - improvement_weight) * base_reward + improvement_weight * improvement_reward
#         else:
#             reward = base_reward
        
#         return np.clip(reward, -1, 1)
    
#     def get_name(self):
#         return "temporal_difference"
    
#     def get_params(self):
#         return {'improvement_weight': 0.4}


# class AdaptiveDifficultyReward(BaseReward):
#     """
#     Formula 5: Adaptive Difficulty Reward
    
#     Dynamically adjusts reward sensitivity based on task difficulty.
#     R = -(1 - exp(-T_adaptive · c_norm))
#     """
    
#     def compute(self, cost, max_time, min_time, base_temperature=2.0, **kwargs):
#         """
#         Compute adaptive difficulty reward.
        
#         Args:
#             cost: Incremental latency
#             max_time: Maximum possible latency
#             min_time: Minimum possible latency
#             base_temperature: Base temperature parameter (default: 2.0)
        
#         Returns:
#             Reward in [-1, 0] range
#         """
#         self.validate_inputs(cost, max_time, min_time)
        
#         # Normalize cost
#         cost_range = max_time - min_time
#         if cost_range < 1e-8:
#             return np.zeros_like(cost) if isinstance(cost, (list, np.ndarray)) else 0.0
        
#         normalized_cost = (cost - min_time) / cost_range
#         normalized_cost = np.clip(normalized_cost, 0, 1)
        
#         # Estimate task difficulty
#         difficulty = cost_range / (max_time + 1e-8)
        
#         # Current performance level
#         performance_level = normalized_cost
        
#         # Adaptive temperature
#         adaptive_temp = base_temperature * (1 + difficulty * (1 - performance_level))
        
#         # Exponential reward with adaptive temperature
#         exp_reward = np.exp(-adaptive_temp * normalized_cost)
#         reward = -(1 - exp_reward)
        
#         return reward
    
#     def get_name(self):
#         return "adaptive_difficulty"
    
#     def get_params(self):
#         return {'base_temperature': 2.0}

class ExponentialReward(BaseReward):
    def compute(self, cost, max_time, min_time, temperature=2.0, exponent=2.0, **kwargs):
        self.validate_inputs(cost, max_time, min_time)
        cost = np.asarray(cost, dtype=np.float32)
        max_time = float(max_time)
        if max_time <= 0.0:
            return np.zeros_like(cost)
        norm_cost = np.clip(cost / max_time, 0.0, 1.0)
        exponent = float(exponent)
        exponent = max(exponent, 1.0)
        temperature = float(temperature)
        temperature = max(temperature, 1e-8)
        z = norm_cost ** exponent
        exp_term = np.exp(-temperature * z)
        denom = 1.0 - np.exp(-temperature)
        if abs(denom) < 1e-8:
            reward = -z
        else:
            reward = -(1.0 - exp_term) / denom
        reward = np.clip(reward, -1.0, 0.0)
        return reward

    def get_name(self):
        return "exponential"

    def get_params(self):
        return {"temperature": 2.0, "exponent": 2.0}


class TemporalDifferenceReward(BaseReward):
    def compute(self, cost, max_time, min_time, previous_cost=None, improvement_weight=0.4, **kwargs):
        self.validate_inputs(cost, max_time, min_time)
        cost = np.asarray(cost, dtype=np.float32)
        max_time = float(max_time)
        if max_time <= 0.0:
            base_reward = -np.zeros_like(cost)
        else:
            norm_cost = np.clip(cost / max_time, 0.0, 1.0)
            base_reward = -norm_cost
        improvement_component = 0.0
        if previous_cost is not None and max_time > 0.0:
            prev = np.asarray(previous_cost, dtype=np.float32)
            cur_total = float(np.sum(np.clip(cost, 0.0, None)))
            prev_total = float(np.sum(np.clip(prev, 0.0, None)))
            improvement = (prev_total - cur_total) / max_time
            improvement = np.clip(improvement, -1.0, 1.0)
            improvement_component = improvement
        w = float(improvement_weight)
        w = np.clip(w, 0.0, 1.0)
        reward = (1.0 - w) * base_reward + w * improvement_component
        reward = np.clip(reward, -1.0, 0.0)
        return reward

    def get_name(self):
        return "temporal_difference"

    def get_params(self):
        return {"improvement_weight": 0.4}


class AdaptiveDifficultyReward(BaseReward):
    def compute(self, cost, max_time, min_time, base_temperature=2.0, **kwargs):
        self.validate_inputs(cost, max_time, min_time)
        cost = np.asarray(cost, dtype=np.float32)
        max_time = float(max_time)
        if max_time <= 0.0:
            return np.zeros_like(cost)
        norm_cost = np.clip(cost / max_time, 0.0, 1.0)
        cost_range = max_time - float(min_time)
        if max_time <= 0.0:
            difficulty = 1.0
        else:
            difficulty = cost_range / (max_time + 1e-8)
        difficulty = float(np.clip(difficulty, 0.0, 1.0))
        performance = 1.0 - norm_cost
        base_temperature = float(base_temperature)
        base_temperature = max(base_temperature, 1e-8)
        adaptive_temp = base_temperature * (1.0 + difficulty * (1.0 - performance))
        exp_term = np.exp(-adaptive_temp * norm_cost)
        denom = 1.0 - np.exp(-adaptive_temp)
        denom = np.where(np.abs(denom) < 1e-8, 1.0, denom)
        reward = -(1.0 - exp_term) / denom
        reward = np.clip(reward, -1.0, 0.0)
        return reward

    def get_name(self):
        return "adaptive_difficulty"

    def get_params(self):
        return {"base_temperature": 2.0}
        
class GreedyNormalizedReward(BaseReward):
    """
    Formula 6: Greedy Normalized Reward
    
    Rewards based on normalized gap from greedy solution.
    R = -gap, where gap = max(0, min(1, (episode_time - greedy_time) / greedy_time))
    """
    
    def compute(self, cost, max_time, min_time, greedy_time=None, episode_time=None, **kwargs):
        """
        Compute greedy normalized reward.
        
        Args:
            cost: Incremental latency (not used, kept for interface compatibility)
            max_time: Maximum possible latency (not used, kept for interface compatibility)
            min_time: Minimum possible latency (not used, kept for interface compatibility)
            greedy_time: Greedy solution finish time (required)
            episode_time: Episode finish time (required)
        
        Returns:
            Reward in [-1, 0] range, broadcast as constant over all timesteps
        """
        if greedy_time is None or episode_time is None:
            # Fallback to linear reward if greedy_time or episode_time not provided
            cost_range = max_time - min_time
            if cost_range < 1e-8:
                return np.zeros_like(cost) if isinstance(cost, (list, np.ndarray)) else 0.0
            return -(cost - min_time) / cost_range
        
        # Compute gap: normalized difference from greedy solution
        if greedy_time <= 0:
            gap = 1.0  # Worst case if greedy_time is invalid
        else:
            gap = (episode_time - greedy_time) / greedy_time
            gap = max(0.0, min(1.0, gap))  # Clip to [0, 1]
        
        # Reward is negative gap, broadcast as constant
        reward = -gap
        
        # Broadcast as constant over all timesteps (if cost is array-like)
        if isinstance(cost, (list, np.ndarray)):
            return np.ones_like(cost, dtype=np.float64) * reward
        else:
            return reward
    
    def get_name(self):
        return "greedy_normalized"
    
    def get_params(self):
        return {}


# Register all rewards
from reward_system.reward_registry import RewardRegistry

RewardRegistry.register('linear', LinearReward)
RewardRegistry.register('logarithmic', LogarithmicReward)
RewardRegistry.register('exponential', ExponentialReward)
RewardRegistry.register('temporal_difference', TemporalDifferenceReward)
RewardRegistry.register('adaptive_difficulty', AdaptiveDifficultyReward)
RewardRegistry.register('greedy_normalized', GreedyNormalizedReward)

