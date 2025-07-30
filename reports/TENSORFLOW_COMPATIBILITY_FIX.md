# TensorFlow Graph Mode Compatibility Fix

## Issue
The error occurred when running `meta_trainer.py`:
```
tensorflow.python.framework.errors_impl.OperatorNotAllowedInGraphError: using a `tf.Tensor` as a Python `bool` is not allowed in Graph execution
```

This happened at line 54 in `graph2seq_encoder.py`:
```python
window_size = min(5, seq_len)  # seq_len is a tensor, can't use Python min()
```

## Root Cause
TensorFlow 1.x operates in graph mode by default, where tensor values are not known until runtime. Python operations like `min()`, `range()`, and conditionals cannot be used directly on tensors.

## Fixes Applied

### 1. Fixed `sequence_to_graph` method
Replaced Python operations with TensorFlow operations:
- `min(5, seq_len)` → `tf.minimum(5, seq_len)` 
- Used `tf.range()`, `tf.tile()`, `tf.expand_dims()` for tensor operations
- Removed Python loops in favor of tensor broadcasting

### 2. Fixed adjacency info construction
- Created proper adjacency matrix where each node connects to all nodes in its sequence
- Used tensor operations to handle dynamic batch sizes
- Ensured adjacency info has correct shape `[total_nodes, seq_len]`

### 3. Fixed neighbor length tensors
- Replaced `tf.constant(seq_len, shape=[...])` with `tf.fill([batch_size * seq_len], seq_len)`
- This properly creates a tensor filled with the sequence length value

## Updated Code Structure

The `sequence_to_graph` method now:
1. Creates a fully-connected adjacency pattern for each sequence
2. Uses batch offsets to separate sequences in different batches
3. Returns properly shaped tensors for the UniformNeighborSampler

## Verification

Created test scripts to verify the fix:
- `test_graph2seq_fix.py` - Tests full policy creation and forward pass
- `test_tensor_operations.py` - Tests individual tensor operations

Both confirm that:
- ✓ No more graph mode errors
- ✓ Correct tensor shapes maintained
- ✓ Adjacency structure is correct
- ✓ Forward pass works properly

## Usage

The fix is transparent to users. Simply run:
```python
python meta_trainer.py
```

The Graph2Seq encoder will now work correctly in TensorFlow's graph mode.