# GAT Encoder Implementation for MRLCO

## Overview

This document describes the implementation of Graph Attention Network (GAT) encoder for the MRLCO meta-reinforcement learning system. The GAT encoder provides an alternative to the existing Graph2Seq and LSTM encoders, offering improved representation learning through attention mechanisms.

## Features

### Core GAT Architecture
- **Multi-head Attention**: Configurable number of attention heads for richer representations
- **Layer Stacking**: Support for multiple GAT layers with activation functions
- **Attention Mechanism**: LeakyReLU-based attention with softmax normalization
- **Head Aggregation**: Choice between concatenation and averaging of attention heads
- **Dropout Regularization**: Training-time dropout for improved generalization

### Integration Features
- **BaseEncoder Interface**: Implements the standard encoder interface for seamless integration
- **TensorFlow 1.x Compatibility**: Fully compatible with existing TF 1.x codebase
- **Variable Scoping**: Proper variable scoping for parameter sharing in meta-learning
- **Decoder Compatibility**: Outputs compatible with existing attention and decoder mechanisms

## Architecture Details

### Graph Attention Mechanism

The GAT encoder implements the attention mechanism as described in "Graph Attention Networks" (Velickovic et al., 2018):

1. **Linear Transformation**: Apply learned weight matrix W to node features
   ```
   h' = W * h
   ```

2. **Attention Coefficients**: Compute pairwise attention scores
   ```
   e_ij = LeakyReLU(a^T [W*h_i || W*h_j])
   ```

3. **Attention Weights**: Normalize with softmax
   ```
   α_ij = softmax_j(e_ij)
   ```

4. **Feature Aggregation**: Weighted sum of neighbor features
   ```
   h'_i = σ(Σ_j α_ij * W*h_j)
   ```

### Multi-head Attention

- Multiple attention heads capture different aspects of relationships
- Heads can be concatenated (default) or averaged
- Final dimension: `hidden_dim * num_heads` (concat) or `hidden_dim` (average)

### Graph Construction

For sequence-to-sequence compatibility, the encoder treats input sequences as fully connected graphs where each position connects to all other positions, enabling the model to learn arbitrary dependencies.

## Usage

### Basic Usage

```python
# Use GAT encoder with default parameters
policy = Seq2SeqPolicy(obs_dim, encoder_units, decoder_units, vocab_size, encoder_type='gat')

# For meta-learning
meta_policy = MetaSeq2SeqPolicy(meta_batch_size, obs_dim, encoder_units, decoder_units, vocab_size, encoder_type='gat')
```

### Custom Configuration

```python
# Create policy with custom hyperparameters
hparams = tf.contrib.training.HParams(
    encoder_type='gat',
    num_heads=4,           # Number of attention heads
    concat=False,          # Average heads instead of concatenating
    dropout=0.2,           # Dropout rate
    # ... other parameters
)

policy = Seq2SeqPolicy(obs_dim, encoder_units, decoder_units, vocab_size, encoder_type='gat')
```

### Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_heads` | 8 | Number of attention heads per layer |
| `num_layers` | 2 | Number of GAT layers |
| `concat` | True | Whether to concatenate (True) or average (False) attention heads |
| `dropout` | 0.1 | Dropout rate for training |
| `hidden_dim` | encoder_units | Hidden dimension per attention head |

## Implementation Details

### File Structure
```
policies/
├── base_encoder.py        # Abstract base class
├── gat_encoder.py         # GAT encoder implementation
├── graph2seq_encoder.py   # Existing Graph2Seq encoder
├── rnn_encoder.py         # Existing RNN encoder
└── meta_seq2seq_policy.py # Main policy with encoder selection
```

### Key Classes and Methods

#### GATEncoder Class
```python
class GATEncoder(BaseEncoder):
    def encode(self, encoder_inputs):
        """Main encoding method returning (outputs, state)"""
    
    def get_output_dim(self):
        """Returns output feature dimension"""
    
    def _gat_layer(self, ...):
        """Applies single GAT layer with multiple heads"""
    
    def _attention_head(self, ...):
        """Computes single attention head"""
```

### Output Shapes

- **Input**: `[batch_size, num_nodes, input_dim]`
- **Encoder Outputs**: `[batch_size, num_nodes, output_dim]`
  - `output_dim = hidden_dim * num_heads` (if concat=True)
  - `output_dim = hidden_dim` (if concat=False)
- **Encoder State**: LSTM-compatible state tuple for decoder initialization

### Variable Scoping

The GAT encoder creates variables under the following hierarchy:
```
encoder/
├── gat_layer_0/
│   ├── attention_head_0/
│   │   ├── weight_matrix
│   │   └── attention_vector
│   └── attention_head_1/
│       ├── weight_matrix
│       └── attention_vector
└── state_projection/
    └── state_dense
```

## Compatibility

### Backward Compatibility
- Default encoder remains `graph2seq`
- Existing training scripts work unchanged
- All existing encoder types (`graph2seq`, `lstm`) continue to work

### Framework Compatibility
- Compatible with PPO training (`ppo_offloading.py`)
- Compatible with MRLCO meta-learning (`MRLCO.py`)
- Works with existing samplers and evaluators

### TensorFlow Compatibility
- Uses TensorFlow 1.x operations (`tf.get_variable`, `tf.variable_scope`)
- Compatible with existing TF 1.x computational graph
- No dependency on TensorFlow 2.x features

## Testing

Run the verification tests to ensure proper implementation:

```bash
python test_gat_encoder.py
```

The test suite verifies:
- Interface compliance with BaseEncoder
- GAT architecture components
- Integration with Seq2SeqNetwork
- Output shape compatibility
- Variable scope configuration
- TensorFlow 1.x compatibility
- Documentation quality

## Performance Considerations

### Memory Usage
- Multi-head attention increases memory requirements
- Fully connected graphs require O(n²) attention computation
- Consider reducing `num_heads` for large sequence lengths

### Training Stability
- LeakyReLU activation helps with gradient flow
- Dropout prevents overfitting
- Proper initialization with Glorot uniform

### Computational Complexity
- Time complexity: O(n² * d * h) where n=sequence length, d=hidden dim, h=num heads
- Space complexity: O(n² * h) for attention matrices

## Future Extensions

### Potential Improvements
1. **Sparse Attention**: Support for sparse adjacency matrices
2. **Edge Features**: Incorporation of edge attributes
3. **Positional Encoding**: Additional positional information
4. **Adaptive Heads**: Dynamic number of attention heads

### Integration Opportunities
1. **Hierarchical Attention**: Multi-level attention mechanisms
2. **Temporal GAT**: Time-aware graph attention
3. **Multi-modal**: Support for different input modalities

## References

1. Veličković, P., Cucurull, G., Casanova, A., Romero, A., Lio, P., & Bengio, Y. (2017). Graph attention networks. arXiv preprint arXiv:1710.10903.

2. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. Advances in neural information processing systems, 30.

## Contact

For questions or issues related to the GAT encoder implementation, please refer to the main MRLCO documentation or create an issue in the project repository.