"""
Base Reward Interface

Defines the base class for all reward functions.
"""

import numpy as np


class BaseReward:
    """
    Base class for all reward functions.
    
    All reward functions must inherit from this class and implement
    the compute() method.
    """
    
    def compute(self, cost, max_time, min_time, **kwargs):
        """
        Compute reward from cost.
        
        Args:
            cost: Incremental latency (scalar or array)
            max_time: Maximum possible latency
            min_time: Minimum possible latency
            **kwargs: Additional parameters specific to reward function
        
        Returns:
            Reward value(s) - typically in range [-1, 0]
        """
        raise NotImplementedError("Subclasses must implement compute() method")
    
    def get_name(self):
        """
        Return reward function name.
        
        Returns:
            String name of the reward function
        """
        raise NotImplementedError("Subclasses must implement get_name() method")
    
    def get_params(self):
        """
        Return default parameters for this reward function.
        
        Returns:
            Dictionary of default parameters
        """
        return {}
    
    def validate_inputs(self, cost, max_time, min_time):
        """
        Validate input parameters.
        
        Args:
            cost: Cost value(s)
            max_time: Maximum time
            min_time: Minimum time
        
        Returns:
            True if valid, raises ValueError otherwise
        """
        if max_time <= min_time:
            raise ValueError(f"max_time ({max_time}) must be greater than min_time ({min_time})")
        
        if isinstance(cost, (list, np.ndarray)):
            cost_array = np.array(cost)
            if np.any(cost_array < 0):
                raise ValueError("Cost values must be non-negative")
        else:
            if cost < 0:
                raise ValueError("Cost value must be non-negative")
        
        return True

