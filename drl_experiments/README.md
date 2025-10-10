# DRL Experiments

This directory contains implementations for vanilla Deep Reinforcement Learning (PPO) experiments on the task offloading problem. The code is designed to work alongside the existing Meta-RL (MRLCO) codebase without breaking existing functionality.

## Overview

The DRL experiments implement a single policy that learns to solve task offloading problems across multiple maps using Proximal Policy Optimization (PPO). This provides a baseline comparison to the Meta-RL approach.

### Key Features

- **Autoregressive Policy**: Generates action sequences step-by-step for task offloading
- **Multi-task Training**: Trains on 22 maps simultaneously using the same data split as MRLCO
- **PPO Implementation**: Standard PPO with GAE, clipping, and entropy regularization
- **Ready-mask Support**: Handles task dependencies through masking
- **Fine-tuning Evaluation**: Supports both zero-shot and fine-tuned evaluation
- **Comprehensive Logging**: CSV logs with detailed metrics

## File Structure

```
drl_experiments/
├── __init__.py              # Package initialization
├── configs.py              # Configuration and hyperparameters
├── policy.py                # DRL policy implementation
├── rollout.py               # Rollout collection utilities
├── gae.py                   # Generalized Advantage Estimation
├── ppo.py                   # PPO implementation
├── train_drl.py             # Main training script
├── eval_drl.py              # Evaluation script
├── attentive_stats_agg.py   # Optional attentive aggregator
└── README.md                # This file
```

## Data Splits

- **Training**: 22 maps (same as MRLCO) - `offload_random20_1` through `offload_random20_22`
- **Evaluation**: 3 unseen maps - `offload_random20_21`, `offload_random20_2`, `offload_random20_25`

This ensures fair comparison with MRLCO by using identical training data.

## Usage

### Training

```bash
# Basic training with default parameters
python -m drl_experiments.train_drl

# Custom parameters
python -m drl_experiments.train_drl \
    --tasks_per_epoch 10 \
    --rollouts_per_task 3 \
    --ppo_epochs 4 \
    --minibatch_size 2048 \
    --lr 3e-4 \
    --num_epochs 100 \
    --seed 42
```

### Evaluation

```bash
# Zero-shot evaluation
python -m drl_experiments.eval_drl \
    --ckpt drl_runs/ckpt_epoch_100.ckpt \
    --mode zero_shot \
    --num_rollouts 10

# Fine-tuning evaluation
python -m drl_experiments.eval_drl \
    --ckpt drl_runs/ckpt_epoch_100.ckpt \
    --mode finetune \
    --finetune_steps 20 \
    --num_rollouts 10
```

## Configuration

Key hyperparameters in `configs.py`:

```python
# Training configuration
tasks_per_epoch = 10          # Tasks sampled per epoch
rollouts_per_task = 3         # Rollouts per task
ppo_epochs = 4                # PPO epochs per batch
minibatch_size = 2048         # Minibatch size

# PPO hyperparameters
gamma = 0.99                  # Discount factor
gae_lambda = 0.95            # GAE lambda
clip_ratio = 0.2             # PPO clipping ratio
ent_coef = 0.02              # Entropy coefficient
vf_coef = 0.5                # Value function coefficient
lr = 3e-4                    # Learning rate
```

## Architecture

### Policy Network

- **Encoder**: Graph2Seq encoder (reuses existing implementation)
- **Decoder**: Autoregressive LSTM with attention mechanism
- **Heads**: 
  - Policy head: Binary classification (local=0, MEC=1)
  - Value head: Scalar value estimation

### Training Process

1. **Epoch Loop**: Sample `tasks_per_epoch` tasks
2. **Rollout Collection**: Collect `rollouts_per_task` rollouts per task
3. **Batching**: Concatenate all rollouts into PPO batch
4. **GAE Computation**: Compute advantages and returns
5. **PPO Updates**: Perform `ppo_epochs` of minibatch updates

## Output Files

### Training Outputs

- `drl_runs/training_log.csv`: Training metrics per epoch
- `drl_runs/ckpt_epoch_X.ckpt`: Model checkpoints

### Evaluation Outputs

- `drl_eval/eval_results_zero_shot.csv`: Zero-shot evaluation results
- `drl_eval/eval_results_finetune.csv`: Fine-tuning evaluation results

## Metrics

### Training Metrics

- Policy Loss: PPO policy loss with clipping
- Value Loss: Mean squared error for value function
- Entropy: Policy entropy (exploration measure)
- Approximate KL: KL divergence between old and new policies
- Clip Fraction: Fraction of clipped policy updates

### Evaluation Metrics

- Makespan: Total completion time (negative cumulative reward)
- Reward: Cumulative reward per episode
- Statistics: Mean ± standard deviation across rollouts

## Comparison with MRLCO

| Aspect | DRL | MRLCO |
|--------|-----|-------|
| **Approach** | Single policy across all tasks | Meta-learning with task adaptation |
| **Training Data** | Same 22 maps | Same 22 maps |
| **Evaluation** | 3 unseen maps | 3 unseen maps |
| **Fine-tuning** | 20 PPO steps | Few-shot adaptation |
| **Architecture** | Graph2Seq + PPO | Graph2Seq + Meta-RL |

## Troubleshooting

### Common Issues

1. **Shape Mismatches**: Ensure `time_major=False` in configs
2. **Memory Issues**: Reduce `minibatch_size` or `rollouts_per_task`
3. **Training Instability**: Adjust `clip_ratio` or `lr`
4. **Poor Performance**: Increase `ppo_epochs` or `num_epochs`

### Debugging

Enable verbose logging by modifying the print statements in the training loop. Check that:
- Rollouts are being collected successfully
- GAE computation is working correctly
- PPO updates are reducing loss

## Dependencies

- TensorFlow 1.x (compatible with existing codebase)
- NumPy
- Joblib (for checkpoint saving)
- Standard library modules (csv, time, collections)

## Future Extensions

- Multi-head attention in decoder
- Curriculum learning across map difficulties
- Transfer learning between different graph types
- Integration with other RL algorithms (SAC, TD3)
