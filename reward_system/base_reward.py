"""
Base Reward Class

Abstract base class for all reward function implementations.
"""

import numpy as np


class BaseReward:
    """
    Base class for reward functions.
    
    All reward functions must inherit from this class and implement
    the compute() method.
    """
    
    def compute(self, cost, max_time, min_time, **kwargs):
        """
        Compute reward value.
        
        Args:
            cost: Incremental latency (current cost)
            max_time: Maximum possible latency
            min_time: Minimum possible latency
            **kwargs: Additional parameters specific to reward function
        
        Returns:
            Reward value (typically in [-1, 0] range)
        """
        raise NotImplementedError("Subclasses must implement compute() method")
    
    def validate_inputs(self, cost, max_time, min_time):
        """
        Validate input parameters.
        
        Args:
            cost: Cost value
            max_time: Maximum time
            min_time: Minimum time
        
        Raises:
            ValueError: If inputs are invalid
        """
        if max_time < min_time:
            raise ValueError(f"max_time ({max_time}) must be >= min_time ({min_time})")
        
        if np.any(np.isnan(cost)) or np.any(np.isinf(cost)):
            raise ValueError("cost contains NaN or Inf values")
    
    def get_name(self):
        """
        Get the name of this reward function.
        
        Returns:
            String name of the reward function
        """
        return self.__class__.__name__.lower().replace('reward', '')
    
    def get_params(self):
        """
        Get default parameters for this reward function.
        
        Returns:
            Dictionary of default parameters
        """
        return {}

