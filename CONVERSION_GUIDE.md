# Meta-RL to Single-Policy RL Conversion Guide

This document explains how to convert the Meta-RL implementation to single-policy RL for task offloading in Mobile Edge Computing (MEC).

## Overview

The original codebase implements Meta-RL (MRLCO algorithm) for learning to offload tasks in MEC environments. This guide shows how to convert it to standard single-policy RL using PPO.

## Key Differences

### Meta-RL Architecture (Original)
- **Multiple Policies**: `MetaSeq2SeqPolicy` with `meta_batch_size` separate policies
- **Two-Level Optimization**: Inner task adaptation + outer meta-policy update
- **Meta-Batch Sampling**: Sample from multiple tasks simultaneously
- **MRLCO Algorithm**: Meta-learning with parameter synchronization

### Single-Policy RL Architecture (Converted)
- **Single Policy**: One `Seq2SeqPolicy` for all tasks
- **Single-Level Optimization**: Direct policy updates using PPO
- **Standard Sampling**: Sample from one task at a time or mixed sampling
- **PPO Algorithm**: Standard policy gradient with clipping

## File Structure Changes

### New Files Created
```
single_policy_trainer.py          # Single-policy trainer
single_policy_ppo.py             # PPO algorithm for single-policy RL
env/single_policy_offloading_env.py  # Single-policy environment wrapper
samplers/seq2seq_sampler.py      # Single-policy sampler
samplers/seq2seq_sampler_process.py  # Single-policy sample processor
train_single_policy.py           # Main training script
```

### Modified Files
```
samplers/vectorized_env_executor.py  # Added single-policy executors
```

## Key Changes Made

### 1. Policy Architecture
**Before (Meta-RL):**
```python
# Multiple policies for different tasks
meta_policy = MetaSeq2SeqPolicy(meta_batch_size=10, ...)
# Each task has its own policy
for i in range(meta_batch_size):
    meta_policies.append(Seq2SeqPolicy(...))
```

**After (Single-Policy RL):**
```python
# Single policy for all tasks
policy = Seq2SeqPolicy(obs_dim=17, encoder_units=128, ...)
```

### 2. Training Loop
**Before (Meta-RL):**
```python
# Two-level optimization
for itr in range(n_itr):
    # Sample from multiple tasks
    task_ids = sampler.update_tasks()
    paths = sampler.obtain_samples()
    
    # Inner policy updates for each task
    policy_losses, value_losses = algo.UpdatePPOTarget(samples_data)
    
    # Resample after inner updates
    new_paths = sampler.obtain_samples()
    
    # Outer meta-policy update
    algo.UpdateMetaPolicy()
```

**After (Single-Policy RL):**
```python
# Single-level optimization
for itr in range(n_itr):
    # Sample trajectories
    paths = sampler.obtain_samples()
    
    # Process samples
    samples_data = sampler_processor.process_samples(paths)
    
    # Direct policy update
    policy_losses, value_losses = algo.UpdatePPOTarget(samples_data)
```

### 3. Sampling Strategy
**Before (Meta-RL):**
```python
# Sample from multiple tasks simultaneously
class Seq2SeqMetaSampler:
    def obtain_samples(self):
        # Sample from meta_batch_size tasks
        for i in range(self.meta_batch_size):
            # Sample from task i
```

**After (Single-Policy RL):**
```python
# Sample from single task or mixed sampling
class Seq2SeqSampler:
    def obtain_samples(self):
        # Sample from current task
        # Task selection handled by environment
```

### 4. Environment Interface
**Before (Meta-RL):**
```python
# Meta-environment with task sampling
class OffloadingEnvironment(MetaEnv):
    def sample_tasks(self, n_tasks):
        return np.random.choice(self.total_task, n_tasks)
    
    def set_task(self, task):
        self.task_id = task
```

**After (Single-Policy RL):**
```python
# Single-policy environment wrapper
class SinglePolicyOffloadingEnvironment(OffloadingEnvironment):
    def reset(self):
        # Randomly select task for single-policy RL
        self.current_task_id = np.random.randint(0, self.total_task)
        self.set_task(self.current_task_id)
        return self.get_observation()
```

## Usage

### Running Single-Policy RL Training
```bash
python train_single_policy.py
```

### Key Parameters
- `BATCH_SIZE`: Number of trajectories per iteration (default: 100)
- `N_ITERATIONS`: Number of training iterations (default: 3500)
- `LEARNING_RATE`: PPO learning rate (default: 5e-4)
- `ROLLOUTS_PER_TASK`: Number of rollouts per task (default: 1)

## Expected Benefits

### Advantages of Single-Policy RL
1. **Simpler Architecture**: Easier to understand and debug
2. **Faster Training**: No meta-learning overhead
3. **Better Generalization**: Single policy learns across all tasks
4. **Easier Hyperparameter Tuning**: Fewer parameters to tune
5. **More Stable Training**: No meta-learning instability

### Potential Drawbacks
1. **Less Task-Specific Adaptation**: No task-specific fine-tuning
2. **Slower Convergence**: May take longer to learn task-specific strategies
3. **Less Sample Efficient**: No meta-learning sample efficiency benefits

## Performance Comparison

### Training Time
- **Meta-RL**: ~2x slower due to inner/outer optimization
- **Single-Policy RL**: Faster, direct optimization

### Sample Efficiency
- **Meta-RL**: Better for few-shot learning scenarios
- **Single-Policy RL**: Better for standard RL scenarios

### Memory Usage
- **Meta-RL**: Higher memory usage (multiple policies)
- **Single-Policy RL**: Lower memory usage (single policy)

## Migration Checklist

- [x] Create single-policy trainer
- [x] Create single-policy PPO algorithm
- [x] Create single-policy environment wrapper
- [x] Create single-policy sampler and processor
- [x] Create main training script
- [x] Add single-policy environment executors
- [x] Test the conversion

## Troubleshooting

### Common Issues
1. **Import Errors**: Make sure all new files are in the correct directories
2. **Shape Mismatches**: Single-policy RL expects different data shapes
3. **Memory Issues**: Single-policy RL uses less memory but may need different batch sizes

### Debug Tips
1. Start with small batch sizes and iterations
2. Check that the environment is properly wrapped
3. Verify that the policy is using the correct interface
4. Monitor training logs for convergence

## Future Improvements

1. **Curriculum Learning**: Gradually increase task difficulty
2. **Multi-Task Learning**: Learn multiple related tasks simultaneously
3. **Transfer Learning**: Pre-train on simple tasks, fine-tune on complex ones
4. **Hierarchical RL**: Use high-level and low-level policies
