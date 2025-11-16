"""
Reward Registry

Central registry for all reward functions.
"""


class RewardRegistry:
    """
    Registry for managing reward functions.
    
    Provides centralized access to all reward implementations.
    """
    
    _rewards = {}
    _default_reward = 'linear'
    
    @classmethod
    def register(cls, name, reward_class):
        """
        Register a reward function.
        
        Args:
            name: String name of the reward function
            reward_class: Reward class (subclass of BaseReward)
        """
        cls._rewards[name] = reward_class
    
    @classmethod
    def get(cls, name):
        """
        Get reward function instance by name.
        
        Args:
            name: String name of the reward function
        
        Returns:
            Reward function instance, or None if not found
        """
        reward_class = cls._rewards.get(name)
        if reward_class is None:
            return None
        return reward_class()
    
    @classmethod
    def list_all(cls):
        """
        List all registered reward names.
        
        Returns:
            List of reward function names
        """
        return list(cls._rewards.keys())
    
    @classmethod
    def get_default(cls):
        """
        Get default reward function instance.
        
        Returns:
            Default reward function instance
        """
        return cls.get(cls._default_reward)
    
    @classmethod
    def set_default(cls, name):
        """
        Set default reward function.
        
        Args:
            name: Name of reward function to set as default
        """
        if name not in cls._rewards:
            raise ValueError(f"Reward '{name}' not registered")
        cls._default_reward = name
    
    @classmethod
    def is_registered(cls, name):
        """
        Check if a reward function is registered.
        
        Args:
            name: Reward function name
        
        Returns:
            True if registered, False otherwise
        """
        return name in cls._rewards

