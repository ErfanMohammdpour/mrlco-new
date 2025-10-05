# Deep RL Offloading System

This document describes the Deep RL implementation for task offloading, which replaces the original Meta-RL approach with a single-policy Deep RL system.

## Overview

The Deep RL system uses an **Actor-Critic architecture** with a **Graph2Seq encoder** to learn optimal offloading decisions. This approach simplifies the meta-learning complexity while maintaining the benefits of graph-based representation learning.

## Key Components

### 1. Deep RL Agent (`deep_rl_offloading.py`)
- **Actor Network**: Policy network using Graph2Seq encoder + LSTM decoder
- **Critic Network**: Value function network using Graph2Seq encoder + value head
- **Experience Replay**: Buffer for storing and sampling experiences
- **Target Networks**: Stable learning with target network updates

### 2. Training System (`deep_rl_trainer.py`)
- **Episode-based Training**: Single policy learns from multiple tasks
- **Exploration Strategy**: ε-greedy exploration with decay
- **Evaluation**: Regular evaluation on test tasks
- **Automated Reporting**: Comprehensive training metrics and visualizations

### 3. Configuration System (`deep_rl_config.py`)
- **Hyperparameter Management**: Centralized configuration
- **Multiple Presets**: Default, fast training, production configurations
- **Validation**: Parameter validation and range checking

### 4. Hyperparameter Tuning (`hyperparameter_tuning.py`)
- **Grid Search**: Exhaustive search over parameter combinations
- **Random Search**: Random sampling from parameter space
- **Bayesian Optimization**: (Future implementation)
- **Results Analysis**: Automated reporting and best configuration selection

## Architecture Comparison

### Meta-RL (Original)
```
Meta-Policy (Core Policy)
├── Task-Specific Policy 1
├── Task-Specific Policy 2
├── ...
└── Task-Specific Policy N

Training: Two-level optimization
├── Inner Loop: Update task-specific policies
└── Outer Loop: Update meta-policy
```

### Deep RL (New)
```
Single Policy (Actor-Critic)
├── Actor: Graph2Seq Encoder + LSTM Decoder
└── Critic: Graph2Seq Encoder + Value Head

Training: Single-level optimization
└── Experience Replay + Policy Gradient
```

## Key Advantages

1. **Simplified Architecture**: Single policy instead of multiple task-specific policies
2. **Better Sample Efficiency**: Experience replay allows learning from past experiences
3. **Stable Learning**: Target networks and experience replay improve stability
4. **Easier Hyperparameter Tuning**: Fewer parameters to tune
5. **Faster Training**: No meta-learning overhead

## Usage

### Basic Training

```python
from deep_rl_trainer import main
main()  # Run with default configuration
```

### Custom Configuration

```python
from deep_rl_config import DeepRLConfig
from deep_rl_offloading import DeepRLOffloadingAgent
from deep_rl_trainer import DeepRLTrainer

# Get custom configuration
config = DeepRLConfig.get_config('production')

# Create agent with custom parameters
agent = DeepRLOffloadingAgent(
    learning_rate=config['agent']['learning_rate'],
    gamma=config['agent']['gamma'],
    # ... other parameters
)

# Train with custom settings
trainer = DeepRLTrainer(
    agent=agent,
    env=env,
    n_episodes=config['training']['n_episodes'],
    # ... other settings
)
```

### Hyperparameter Tuning

```python
from hyperparameter_tuning import HyperparameterTuner

# Create tuner
tuner = HyperparameterTuner(
    search_type='random',  # or 'grid'
    n_trials=50,
    n_episodes=500
)

# Run tuning
best_config, best_score = tuner.run_tuning()
```

### Comparison with Meta-RL

```python
from compare_meta_vs_deep_rl import MetaRLvsDeepRLComparison

# Create comparison
comparison = MetaRLvsDeepRLComparison(env, test_tasks=20)

# Run evaluations
comparison.evaluate_greedy_baseline()
comparison.evaluate_meta_rl(meta_policy_path='./meta_model_final.ckpt')
comparison.evaluate_deep_rl(deep_rl_path='./deep_rl_model_final.ckpt')

# Generate report
stats = comparison.generate_comparison_report()
```

## Configuration Options

### Agent Parameters
- `learning_rate`: Learning rate for optimization (default: 3e-4)
- `gamma`: Discount factor for future rewards (default: 0.99)
- `tau`: Soft update parameter for target networks (default: 0.005)
- `encoder_units`: Graph2Seq encoder hidden units (default: 128)
- `decoder_units`: LSTM decoder hidden units (default: 128)

### Exploration Parameters
- `epsilon_start`: Initial exploration rate (default: 1.0)
- `epsilon_end`: Final exploration rate (default: 0.01)
- `epsilon_decay`: Exploration decay rate (default: 0.995)

### Training Parameters
- `n_episodes`: Number of training episodes (default: 2000)
- `max_episode_length`: Maximum episode length (default: 50)
- `batch_size`: Batch size for training (default: 64)
- `buffer_size`: Experience replay buffer size (default: 100000)

## File Structure

```
mrlco-new/
├── deep_rl_offloading.py      # Main Deep RL agent implementation
├── deep_rl_trainer.py         # Training loop and evaluation
├── deep_rl_config.py          # Configuration management
├── hyperparameter_tuning.py   # Hyperparameter optimization
├── compare_meta_vs_deep_rl.py # Comparison with Meta-RL
├── DEEP_RL_README.md          # This documentation
└── deep_rl_model/             # Saved models directory
    ├── checkpoint_episode_200.ckpt
    ├── checkpoint_episode_400.ckpt
    └── ...
```

## Training Process

1. **Environment Setup**: Initialize offloading environment with task graphs
2. **Agent Creation**: Create Deep RL agent with specified parameters
3. **Episode Loop**: 
   - Sample random task
   - Run episode with current policy
   - Store experiences in replay buffer
   - Update agent if buffer has enough samples
4. **Evaluation**: Regular evaluation on test tasks
5. **Reporting**: Generate training reports and visualizations

## Evaluation Metrics

- **Average Reward**: Mean reward over evaluation episodes
- **Average Latency**: Mean task completion latency
- **Convergence Speed**: Episodes to reach stable performance
- **Sample Efficiency**: Performance vs. number of samples used
- **Generalization**: Performance on unseen tasks

## Expected Performance

Based on the architecture design, the Deep RL system should achieve:

1. **Faster Convergence**: Single-policy learning converges faster than meta-learning
2. **Better Sample Efficiency**: Experience replay improves learning efficiency
3. **Stable Learning**: Target networks and experience replay reduce variance
4. **Good Generalization**: Graph2Seq encoder captures task structure well

## Troubleshooting

### Common Issues

1. **Memory Issues**: Reduce `buffer_size` or `batch_size`
2. **Slow Training**: Increase `update_frequency` or reduce `max_episode_length`
3. **Poor Performance**: Try different `learning_rate` or `epsilon_decay`
4. **Unstable Learning**: Increase `target_update_frequency` or reduce `learning_rate`

### Debug Mode

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Use fast training configuration
config = DeepRLConfig.get_config('fast_training')
```

## Future Improvements

1. **Advanced Exploration**: Implement UCB, Thompson sampling, or curiosity-driven exploration
2. **Multi-Objective Learning**: Optimize for both latency and energy consumption
3. **Transfer Learning**: Pre-train on synthetic tasks, fine-tune on real tasks
4. **Distributed Training**: Parallel training across multiple environments
5. **Online Learning**: Continuous learning from new tasks without retraining

## References

- Original Meta-RL Paper: "Fast Adaptive Task Offloading in Edge Computing Based on Meta Reinforcement Learning"
- Graph2Seq: "Graph-to-Sequence Learning using Gated Graph Neural Networks"
- Actor-Critic: "Actor-Critic Algorithms" (Sutton & Barto)
- Experience Replay: "Human-level control through deep reinforcement learning" (DQN paper)
