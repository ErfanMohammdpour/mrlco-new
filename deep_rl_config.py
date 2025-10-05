"""
Configuration file for Deep RL Offloading System
Contains hyperparameters and settings for easy tuning
"""

import os

class DeepRLConfig:
    """Configuration class for Deep RL offloading system"""
    
    # Environment Configuration
    ENV_CONFIG = {
        'mec_process_capable': 10.0 * 1024 * 1024,  # MEC server processing capacity
        'mobile_process_capable': 1.0 * 1024 * 1024,  # Mobile device processing capacity
        'bandwidth_up': 7.0,  # Uplink bandwidth (Mbps)
        'bandwidth_dl': 7.0,  # Downlink bandwidth (Mbps)
        'batch_size': 1,  # Single task per episode
        'graph_number': 100,  # Number of task graphs
        'time_major': False
    }
    
    # Task Graph Paths
    TASK_PATHS = [
        "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_1/random.20.",
        "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_2/random.20.",
        "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_3/random.20.",
        "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_5/random.20.",
        "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_6/random.20.",
        "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_7/random.20.",
        "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_9/random.20.",
        "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_10/random.20.",
        "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_11/random.20.",
        "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_13/random.20.",
    ]
    
    # Agent Configuration
    AGENT_CONFIG = {
        'obs_dim': 17,  # Observation dimension (task features)
        'action_dim': 2,  # Action dimension (local=0, offload=1)
        'encoder_units': 128,  # Graph2Seq encoder hidden units
        'decoder_units': 128,  # LSTM decoder hidden units
        'learning_rate': 3e-4,  # Learning rate
        'gamma': 0.99,  # Discount factor
        'tau': 0.005,  # Soft update parameter for target networks
    }
    
    # Exploration Configuration
    EXPLORATION_CONFIG = {
        'epsilon_start': 1.0,  # Initial exploration rate
        'epsilon_end': 0.01,  # Final exploration rate
        'epsilon_decay': 0.995,  # Exploration decay rate
        'exploration_strategy': 'epsilon_greedy',  # 'epsilon_greedy', 'ucb', 'thompson'
    }
    
    # Experience Replay Configuration
    REPLAY_CONFIG = {
        'buffer_size': 100000,  # Replay buffer size
        'batch_size': 64,  # Batch size for training
        'update_frequency': 4,  # Update frequency (every N steps)
        'target_update_frequency': 100,  # Target network update frequency
    }
    
    # Training Configuration
    TRAINING_CONFIG = {
        'n_episodes': 2000,  # Number of training episodes
        'max_episode_length': 50,  # Maximum episode length
        'save_interval': 200,  # Model save interval
        'eval_interval': 100,  # Evaluation interval
        'log_interval': 10,  # Logging interval
        'warmup_episodes': 100,  # Episodes before training starts
    }
    
    # Logging Configuration
    LOGGING_CONFIG = {
        'log_dir': './deep_rl_offloading_log/',
        'log_formats': ['stdout', 'log', 'csv'],
        'tensorboard_log_dir': './tensorboard_logs/',
        'save_models': True,
        'save_replay_buffer': False,
    }
    
    # Evaluation Configuration
    EVAL_CONFIG = {
        'n_eval_episodes': 50,  # Number of evaluation episodes
        'eval_frequency': 100,  # Evaluation frequency
        'test_tasks': 20,  # Number of test tasks for comparison
    }
    
    # Graph2Seq Encoder Configuration
    GRAPH2SEQ_CONFIG = {
        'num_layers': 2,  # Number of GCN layers
        'is_bidirectional': True,  # Use bidirectional encoding
        'aggregator_type': 'mean',  # 'mean', 'max', 'gated_mean'
        'dropout_rate': 0.1,  # Dropout rate
        'use_attention': True,  # Use attention mechanism
    }
    
    # Reward Shaping Configuration
    REWARD_CONFIG = {
        'use_dense_rewards': True,  # Use dense rewards instead of sparse
        'reward_scale': 1.0,  # Reward scaling factor
        'penalty_for_invalid_actions': -0.1,  # Penalty for invalid actions
        'bonus_for_efficiency': 0.1,  # Bonus for efficient solutions
    }
    
    # Model Architecture Variants
    ARCHITECTURE_VARIANTS = {
        'actor_critic': {
            'type': 'actor_critic',
            'description': 'Standard Actor-Critic with Graph2Seq encoder'
        },
        'dqn': {
            'type': 'dqn',
            'description': 'Deep Q-Network with Graph2Seq encoder'
        },
        'ppo': {
            'type': 'ppo',
            'description': 'Proximal Policy Optimization with Graph2Seq encoder'
        },
        'sac': {
            'type': 'sac',
            'description': 'Soft Actor-Critic with Graph2Seq encoder'
        }
    }
    
    # Hyperparameter Search Space
    HYPERPARAMETER_SEARCH = {
        'learning_rate': [1e-4, 3e-4, 1e-3, 3e-3],
        'gamma': [0.95, 0.99, 0.995],
        'tau': [0.001, 0.005, 0.01],
        'epsilon_decay': [0.99, 0.995, 0.999],
        'batch_size': [32, 64, 128],
        'buffer_size': [50000, 100000, 200000],
        'encoder_units': [64, 128, 256],
        'decoder_units': [64, 128, 256],
    }
    
    @classmethod
    def get_config(cls, config_type='default'):
        """Get configuration for specific type"""
        
        if config_type == 'default':
            return {
                'env': cls.ENV_CONFIG,
                'agent': cls.AGENT_CONFIG,
                'exploration': cls.EXPLORATION_CONFIG,
                'replay': cls.REPLAY_CONFIG,
                'training': cls.TRAINING_CONFIG,
                'logging': cls.LOGGING_CONFIG,
                'eval': cls.EVAL_CONFIG,
                'graph2seq': cls.GRAPH2SEQ_CONFIG,
                'reward': cls.REWARD_CONFIG,
            }
        
        elif config_type == 'fast_training':
            # Configuration for quick training/testing
            config = cls.get_config('default')
            config['training']['n_episodes'] = 500
            config['training']['max_episode_length'] = 20
            config['training']['eval_interval'] = 50
            config['training']['log_interval'] = 5
            config['replay']['buffer_size'] = 10000
            config['replay']['batch_size'] = 32
            return config
        
        elif config_type == 'production':
            # Configuration for production training
            config = cls.get_config('default')
            config['training']['n_episodes'] = 5000
            config['training']['max_episode_length'] = 100
            config['replay']['buffer_size'] = 500000
            config['replay']['batch_size'] = 128
            config['agent']['learning_rate'] = 1e-4
            return config
        
        elif config_type == 'hyperparameter_search':
            # Configuration for hyperparameter search
            config = cls.get_config('default')
            config['training']['n_episodes'] = 1000
            config['training']['eval_interval'] = 200
            return config
        
        else:
            raise ValueError(f"Unknown config type: {config_type}")
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories"""
        directories = [
            cls.LOGGING_CONFIG['log_dir'],
            cls.LOGGING_CONFIG['tensorboard_log_dir'],
            './deep_rl_model/',
            './comparison_log/',
            './reports/',
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    @classmethod
    def validate_config(cls, config):
        """Validate configuration parameters"""
        errors = []
        
        # Check required parameters
        required_sections = ['env', 'agent', 'training', 'replay']
        for section in required_sections:
            if section not in config:
                errors.append(f"Missing required section: {section}")
        
        # Check parameter ranges
        if 'agent' in config:
            if config['agent']['learning_rate'] <= 0 or config['agent']['learning_rate'] > 1:
                errors.append("Learning rate must be between 0 and 1")
            
            if config['agent']['gamma'] <= 0 or config['agent']['gamma'] >= 1:
                errors.append("Gamma must be between 0 and 1")
        
        if 'replay' in config:
            if config['replay']['batch_size'] > config['replay']['buffer_size']:
                errors.append("Batch size cannot be larger than buffer size")
        
        if errors:
            raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")
        
        return True


# Example usage and testing
if __name__ == "__main__":
    # Test configuration
    config = DeepRLConfig.get_config('default')
    DeepRLConfig.validate_config(config)
    DeepRLConfig.create_directories()
    
    print("Configuration loaded successfully!")
    print(f"Training episodes: {config['training']['n_episodes']}")
    print(f"Learning rate: {config['agent']['learning_rate']}")
    print(f"Buffer size: {config['replay']['buffer_size']}")
