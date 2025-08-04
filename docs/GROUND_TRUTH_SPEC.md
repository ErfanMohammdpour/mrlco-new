# Ground-Truth Specification from TF-1.15 Baseline

## Overview
This document captures the exact behavior implemented in the TF-1.15 baseline repository at `/workspace/mrlco_master/mrlco-new/`.

## 1. Data & DAG Pipeline

### Data Loading (offloading_env.py)
- **Environment Class**: `OffloadingEnvironment` inherits from `MetaEnv`
- **Resource Configuration**: 
  - MEC processing: 10.0 * 1024 * 1024
  - Mobile processing: 1.0 * 1024 * 1024
  - Bandwidth up/down: 7.0 Mbps
- **Graph Data**:
  - Loaded from 19 different graph files in `./env/mec_offloaing_envs/data/meta_offloading_20/`
  - Each graph loaded with `OffloadingTaskGraph` class
  - Graph number: 100
  - Batch size: 100
- **Input Processing**:
  - `time_major=False` in baseline
  - Input shape: `[batch_size, sequence_length, 17]` (obs_dim=17)
  - Encoder embeddings created via fully connected layer

### Task Graph Structure
- Tasks prioritized using `prioritize_tasks()` method
- Each task has:
  - Processing data size
  - Transmission data size
  - Pre-task dependencies (DAG structure)
- Encoding includes ranking and cost information

## 2. Encoder (Graph in Encoder)

### Graph2Seq Encoder (graph2seq_encoder.py)
- **Input**: `[batch_size, seq_len, input_dim]`
- **Architecture**:
  - Converts sequences to graph representation
  - Uses `MeanAggregator` for graph convolution
  - 2 GCN layers (`sample_layer_size=2`)
  - Hidden dimension: 128 (`encoder_units`)
  - Bidirectional: False
  - Dropout: 0.1 in train mode, 0.0 otherwise
- **Output**:
  - encoder_outputs: `[batch_size, seq_len, 2*hidden_dim]` 
  - encoder_state: LSTM-compatible state tuple
  - Final state created via max pooling over sequence

### Encoder Configuration (meta_seq2seq_policy.py)
- Embeddings: Random uniform [-1.0, 1.0]
- Encoder embeddings: Fully connected layer (no activation)
- Scope: Uses tf.AUTO_REUSE

## 3. Decoder & Attention

### Decoder Architecture (meta_seq2seq_policy.py)
- **Three decoder modes**:
  1. **Training decoder**: Uses `TrainingHelper`
  2. **Sample decoder**: Uses `FixedSequenceLearningSampleEmbedingHelper`
  3. **Greedy decoder**: Uses `GreedyEmbeddingHelper`
- **Attention**: LuongAttention mechanism enabled (`is_attention=True`)
- **Cell Type**: LSTM (`unit_type="lstm"`)
- **Layers**: 2 layers, 0 residual layers
- **Hidden units**: 128 (`decoder_units`)
- **Output layer**: Dense layer without bias
- **Vocabulary**: size=2 (binary actions: local/remote)
- **Special tokens**: start_token=0, end_token=2

### Decoder Outputs
- Logits → softmax → policy (π)
- Q-values via dense layer
- Value function: `vf = Σ(π * Q)`
- Sample neglogp via softmax cross entropy

## 4. PPO & Meta Implementation

### MRLCO Algorithm (MRLCO.py)
- **Optimizer Configuration**:
  - Inner optimizer: Adam (lr=1e-3)
  - Outer optimizer: Adam (lr=1e-3) 
  - Gradient clipping: max_grad_norm=0.5
  - PPO clip value: 0.3
  - Value function coefficient: 0.5
- **Loss Computation**:
  - Likelihood ratio clipped between [1-clip, 1+clip]
  - Clipped surrogate objective
  - Value loss: max of clipped and unclipped squared error
  - Total loss: surrogate_obj + vf_coef * vf_loss
- **Meta-Learning**:
  - First-order approximation (Reptile-style)
  - Gradient: (core_params - meta_params) / inner_lr / num_steps / batch_size / update_numbers
  - Inner gradient steps: 1
  - Meta batch size: 10

### PPO Update Process
- Batch size for inner updates: 1000
- Advantages normalized
- GAE lambda: 0.95
- Discount: 0.99
- Multiple gradient steps on same batch

## 5. Meta Trainer / Meta Evaluator

### Training Loop (meta_trainer.py)
- **Iterations**: 1000
- **Sampling**: 
  - Update tasks each iteration
  - 1 rollout per meta task
  - Max path length: 20000
  - Sequential sampling (parallel=False)
- **Logging**:
  - Average greedy latency
  - Average task losses (policy loss)
  - Average value losses
  - Average reward and latency after outer update
- **Checkpointing**:
  - Save interval: 100 iterations
  - Save path: `./meta_model_inner_step1/`

### Evaluation Metrics
- Greedy solution baseline computed at start
- All-MEC and all-local baselines computed
- Rewards: Score function based on normalized latency
- Finish time tracked per task

## 6. Checkpoints & I/O

### Variable Saving (meta_seq2seq_policy.py)
- Uses joblib for serialization
- Saves as dictionary: {variable.name: value}
- Creates directory if needed
- Compatible with both list and dict formats for loading

### TensorFlow Configuration
- TF 1.x style with sessions
- tf.compat.v1 used throughout
- Global variables initializer at start
- Default session management

## 7. Run/Learn Behavior

### Execution Flow
1. Initialize environment and compute baselines
2. Create policy networks (core + meta policies per task)
3. For each iteration:
   - Sample tasks and obtain trajectories
   - Process samples (compute advantages, returns)
   - Inner policy update (PPO on each task)
   - Resample to evaluate one-step update
   - Outer policy update (meta-learning step)
   - Sync meta policies with core policy
4. Save final model

### Key Tensor Shapes
- Observations: `[batch_size, seq_len, 17]`
- Actions: `[batch_size, seq_len]` (integers 0 or 1)
- Logits: `[batch_size, seq_len, 2]`
- Values: `[batch_size, seq_len]`
- Advantages: `[batch_size, seq_len]`
- Returns: `[batch_size, seq_len]`

### Dependencies
- Uses tf.contrib (deprecated in TF2)
- Requires tf.compat.v1 session management
- No eager execution
- No distributed/GPU-specific code in baseline