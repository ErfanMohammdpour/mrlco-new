# FOMAML Implementation for Task Offloading

This document describes the First-Order Model-Agnostic Meta-Learning (FOMAML) implementation that replaces the original Reptile-based MRLCO approach.

## Overview

FOMAML is a first-order approximation of MAML that avoids the computational complexity of second-order gradients while maintaining most of the performance benefits of full MAML.

## Key Changes from MRLCO

### 1. Algorithm Architecture
- **MRLCO (Original)**: Uses Reptile-style parameter averaging
- **FOMAML (New)**: Uses explicit gradient-based meta-updates

### 2. Training Process
- **MRLCO**: Sample → PPO Update → Parameter Averaging
- **FOMAML**: Sample → Split Support/Query → Inner Loop → Outer Loop

### 3. Key Components

#### Inner Loop (Task Adaptation)
- Adapts model to each task using support set
- Performs multiple gradient descent steps
- Uses PPO loss for policy updates

#### Outer Loop (Meta-Update)
- Evaluates adapted models on query sets
- Computes meta-gradients using first-order approximation
- Updates meta-model parameters

## Files Modified

### 1. `meta_algos/FOMAML.py` (New)
- Complete FOMAML implementation
- Inner loop: `adapt_task()`
- Outer loop: `meta_update()`
- First-order gradient approximation

### 2. `meta_trainer.py`
- Updated training loop for FOMAML
- Support/query splitting
- Inner and outer loop execution
- Enhanced logging

### 3. `samplers/seq2seq_meta_sampler.py`
- Added `split_support_query()` method
- Splits data into support (70%) and query (30%) sets

### 4. `test_fomaml.py` (New)
- Comprehensive test suite
- Tests initialization, splitting, and training steps

## Usage

### Running FOMAML Training
```bash
python meta_trainer.py
```

### Running Tests
```bash
python test_fomaml.py
```

## Hyperparameters

### FOMAML Specific
- `inner_lr`: Learning rate for inner loop (task adaptation)
- `outer_lr`: Learning rate for outer loop (meta-update)
- `num_inner_grad_steps`: Number of gradient steps in inner loop
- `support_ratio`: Ratio of data used for support set (default: 0.7)

### Example Configuration
```python
algo = FOMAML(
    policy=meta_policy,
    meta_sampler=sampler,
    meta_sampler_process=sample_processor,
    inner_lr=5e-4,      # Inner loop learning rate
    outer_lr=5e-4,      # Outer loop learning rate
    meta_batch_size=10,
    num_inner_grad_steps=1,
    clip_value=0.2
)
```

## Expected Benefits

### 1. Better Task Adaptation
- More principled approach to meta-learning
- Better generalization to new tasks
- Faster adaptation with fewer samples

### 2. Improved Performance
- Better convergence compared to Reptile
- More stable training dynamics
- Better handling of task diversity

### 3. Theoretical Foundation
- Based on solid meta-learning theory
- First-order approximation of MAML
- Well-studied algorithm with proven results

## Monitoring Training

### Key Metrics
- **Inner Loop Losses**: Policy and value losses during task adaptation
- **Meta-Loss**: Loss on query sets after adaptation
- **Average Reward**: Performance on query sets
- **Average Latency**: Task completion time

### Logging
- Logs are saved to `./meta_offloading20_log_fomaml/`
- Models are saved to `./meta_model_fomaml/`
- Enhanced reporting with meta-learning specific metrics

## Troubleshooting

### Common Issues
1. **Memory Issues**: Reduce `meta_batch_size` or `num_inner_grad_steps`
2. **Convergence Issues**: Adjust `inner_lr` and `outer_lr`
3. **Data Splitting Issues**: Check `support_ratio` parameter

### Debug Mode
Enable debug logging by setting:
```python
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.DEBUG)
```

## Comparison with Original MRLCO

| Aspect | MRLCO | FOMAML |
|--------|-------|--------|
| **Meta-Update** | Parameter averaging | Gradient-based |
| **Computational Cost** | Low | Medium |
| **Convergence** | Slower | Faster |
| **Task Adaptation** | Implicit | Explicit |
| **Theoretical Foundation** | Weak | Strong |

## Future Improvements

1. **Full MAML**: Upgrade to second-order gradients
2. **MAML++**: Add learning rate scheduling and gradient normalization
3. **Meta-SGD**: Learn learning rates automatically
4. **Multi-Step**: Increase `num_inner_grad_steps` for better adaptation

## References

- [Model-Agnostic Meta-Learning (MAML)](https://arxiv.org/abs/1703.03400)
- [First-Order MAML](https://arxiv.org/abs/1803.02999)
- [MAML++](https://arxiv.org/abs/1810.09502)
