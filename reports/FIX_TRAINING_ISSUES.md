# Fix for Training Issues with Graph2Seq Integration

## Identified Problems

1. **Near-Zero Loss Values**: Policy losses are in the range of 1e-9 to 1e-10, indicating:
   - Likelihood ratios are very close to 1.0
   - No meaningful policy updates are happening
   - Gradients are essentially zero

2. **High Latency**: Average latency (~1284ms) is much higher than greedy baseline (~822ms)

3. **No Learning Progress**: Rewards remain flat around -9.0 with no improvement over 500 iterations

## Root Causes

1. **Learning Rate Too Small**: Inner learning rate of 5e-4 is insufficient for meaningful updates
2. **Potential Gradient Vanishing**: Graph2Seq encoder may be causing gradient flow issues
3. **State Dimension Mismatch**: Fixed with state projection, but may still affect learning

## Recommended Fixes

### 1. Increase Learning Rates
```python
# In meta_trainer.py, line 209-210
algo = MRLCO(policy=meta_policy,
             meta_sampler=sampler,
             meta_sampler_process=sample_processor,
             inner_lr=1e-3,  # Increased from 5e-4
             outer_lr=1e-3,  # Increased from 5e-4
             meta_batch_size=META_BATCH_SIZE,
             num_inner_grad_steps=1,
             clip_value=0.3)
```

### 2. Add Gradient Clipping
```python
# In MRLCO.py, after line 93
grads_and_var = self.inner_optimizer.compute_gradients(self.total_loss[i], params)
# Add gradient clipping
grads_and_var = [(tf.clip_by_norm(grad, 5.0), var) for grad, var in grads_and_var if grad is not None]
```

### 3. Initialize Graph2Seq with Smaller Weights
```python
# In graph2seq_modules/inits.py
def glorot(shape, name=None):
    """Glorot & Bengio (AISTATS 2010) init."""
    init_range = np.sqrt(6.0/(shape[0]+shape[1]))
    initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    # Scale down for better initial gradients
    initial = initial * 0.1  # Add this line
    return tf.Variable(initial, name=name)
```

### 4. Add Learning Rate Warmup
```python
# In MRLCO.py constructor
self.global_step = tf.Variable(0, trainable=False)
warmup_steps = 100
self.inner_lr_warmup = tf.train.polynomial_decay(
    learning_rate=self.inner_lr * 0.1,
    global_step=self.global_step,
    decay_steps=warmup_steps,
    end_learning_rate=self.inner_lr,
    power=1.0
)
```

### 5. Monitor and Debug
Add comprehensive logging to track:
- Actual gradient norms
- Likelihood ratio statistics
- Parameter update magnitudes

## Quick Test

To verify fixes are working:
1. Loss values should be in range 0.01-1.0 (not 1e-9)
2. Likelihood ratios should vary from 1.0 (std > 0.1)
3. Rewards should show improvement within 50 iterations
4. Latency should decrease as policy improves

## Implementation Priority

1. **High Priority**: Increase learning rates (immediate fix)
2. **High Priority**: Add gradient clipping (stability)
3. **Medium Priority**: Initialize with smaller weights
4. **Low Priority**: Add warmup schedule