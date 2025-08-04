# Phase 0: Understanding Report - MRLCO TensorFlow Migration

## Executive Summary

This report provides a comprehensive analysis of the MRLCO (Meta Reinforcement Learning for Computation Offloading) codebase to prepare for migration from TensorFlow 1.15 to TensorFlow 2.19.0. The system implements a meta-RL approach using PPO with a Graph2Seq encoder for DAG scheduling in mobile edge computing environments.

## 1. System & Dataflow Map

### 1.1 Overall Architecture

```
Data Flow:
DAG Files (.gv) → Environment → Graph2Seq Encoder → Attention Decoder → PPO Inner Loop → Meta Outer Loop → Checkpoints
```

### 1.2 Key Components and Tensor Shapes

1. **Environment Input**:
   - DAG graphs: ~20 nodes per graph
   - Node features: [batch_size, seq_len=20, feature_dim=17]
   - Features: [task_id, local_cost, up_cost, mec_cost, down_cost, 6_predecessors, 6_successors]

2. **Graph2Seq Encoder**:
   - Input: [batch_size, seq_len, 17] 
   - Graph conversion: Creates fully connected graph within each sequence
   - 2 GCN layers with hidden_dim=128
   - Output: [batch_size, seq_len, 256] (concat mode)
   - Final state: [batch_size, 128] (projected down from 256)

3. **Attention Decoder**:
   - Luong attention over encoder outputs
   - 2-layer LSTM with hidden_dim=128
   - Output logits: [batch_size, seq_len, vocab_size=2] (binary offloading decision)
   - Value function head integrated

4. **PPO/Meta Training**:
   - Meta batch size: 10 tasks
   - Inner batch size: 1000 samples
   - Rollouts per task: 1
   - Max path length: 20000 (though actual paths are ~20 steps)

## 2. Module Inventory

### 2.1 Core Modules

- **Environment**: `env/mec_offloaing_envs/`
  - `offloading_env.py`: Main environment class
  - `offloading_task_graph.py`: DAG parsing and representation
  - `data/`: Pre-generated DAG files

- **Policies**: `policies/`
  - `meta_seq2seq_policy.py`: Meta-policy wrapper managing multiple task policies
  - `graph2seq_encoder.py`: Graph2Seq encoder adapter
  - `graph2seq_modules/`: GCN aggregators, layers, samplers
  - `model_helper.py`: RNN cell builders (uses tf.contrib)

- **Algorithms**: `meta_algos/`
  - `MRLCO.py`: Meta-RL algorithm with first-order approximation
  - `ppo_offloading.py`: Standard PPO for evaluation

- **Samplers**: `samplers/`
  - `seq2seq_meta_sampler.py`: Meta-batch sampling
  - `seq2seq_meta_sampler_process.py`: Advantage computation with GAE

- **Training/Eval**: 
  - `meta_trainer.py`: Main training script
  - `meta_evaluator.py`: Evaluation script

## 3. TensorFlow 1.15 Execution Patterns

### 3.1 Session-based Execution
- All training uses `tf.Session()` context managers
- `sess.run()` calls throughout for training steps
- `tf.get_default_session()` used in policies and algorithms

### 3.2 Placeholders
Found in multiple modules:
- `MRLCO.py`: old_logits, advs, r placeholders for each task
- `ppo_offloading.py`: Similar placeholders for PPO updates
- `meta_seq2seq_policy.py`: obs, decoder_inputs, decoder_targets placeholders

### 3.3 Variable Scopes & Collections
- Extensive use of `tf.compat.v1.variable_scope`
- `tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)`
- Manual variable management for meta-learning parameter sync

### 3.4 tf.contrib Dependencies
Critical dependencies:
- `tf.contrib.seq2seq`: BasicDecoder, dynamic_decode, attention mechanisms
- `tf.contrib.rnn`: LSTM/GRU cells, MultiRNNCell
- `tf.contrib.layers`: fully_connected
- `tf.contrib.learn.ModeKeys`: For training mode detection

### 3.5 Control Flow
- Uses `tf.compat.v1.control_dependencies` implicitly
- Graph-based control flow in decoder helpers

## 4. Algorithm Specifics

### 4.1 PPO Implementation
- **Ratio computation**: Uses likelihood_ratio_sym from CategoricalPd
- **Clipping**: epsilon = 0.3 for meta, 0.2 for standard PPO
- **Value loss**: Clipped value function loss
- **Advantages**: GAE with lambda=0.95, normalized
- **Optimizers**: Adam with lr=1e-3 (increased from 5e-4)

### 4.2 Meta-Learning (MRLCO)
- **Inner updates**: 1 gradient step per task
- **Outer updates**: First-order approximation: (θ_core - θ_task) / (α * K * M)
- **Parameter sync**: Manual copying via assign operations
- **Gradient clipping**: max_norm=0.5

### 4.3 Graph2Seq Encoder
- Converts sequences to fully-connected graphs
- 2 GCN layers with MeanAggregator
- Concatenation mode (doubles hidden dim)
- Dropout=0.1 during training
- Final state via max pooling + projection

## 5. Checkpointing & I/O

### 5.1 Checkpoint Format
- Uses joblib for serialization
- Saves as dictionary: {variable.name: value}
- Path pattern: `./meta_model_inner_step1/meta_model_*.ckpt`

### 5.2 Logging
- Custom logger with CSV/stdout/log outputs
- Metrics: iteration, avg_reward, avg_loss, avg_latency, policy_losses, value_losses, greedy_latencies

## 6. Risk Hotspots

### 6.1 Critical Migration Risks

1. **tf.contrib.seq2seq**:
   - BasicDecoder, dynamic_decode have no direct TF2 equivalent
   - Custom helpers (FixedSequenceLearningSampleEmbedingHelper)
   - Attention wrappers tightly coupled with decoder

2. **Graph Collections**:
   - Manual variable collection for meta-learning
   - Implicit graph-based variable sharing

3. **Session-based Training Loop**:
   - Extensive sess.run() calls
   - Feed_dict patterns throughout

4. **IndexedSlices Gradients**:
   - Embedding lookups in Graph2Seq may produce IndexedSlices
   - Special handling needed for gradient aggregation

5. **Custom Distributions**:
   - CategoricalPd with custom likelihood_ratio_sym method

### 6.2 Moderate Risks

1. **RNN Cells**: tf.contrib.rnn cells need replacement
2. **Variable Scopes**: Name-based variable sharing
3. **Checkpoint Compatibility**: joblib format vs TF2 checkpoints

### 6.3 Low Risks

1. **Basic Operations**: Most math ops have direct equivalents
2. **Optimizer**: Adam optimizer easily portable
3. **Data Pipeline**: Simple numpy-based, no tf.data complexity

## 7. Recommendations

### 7.1 Migration Strategy
1. Start with tf.compat.v1 for initial mechanical migration
2. Focus on decoder/attention as highest risk component
3. Implement custom tf.keras.Model for seq2seq architecture
4. Gradually remove compat.v1 usage module by module

### 7.2 Testing Requirements
1. Unit tests for tensor shapes at each interface
2. Numerical parity tests for:
   - Graph2Seq encoder outputs
   - Attention scores
   - PPO ratio/advantage calculations
   - Meta-gradient approximations
3. End-to-end training curve comparison
4. Checkpoint round-trip validation

### 7.3 Key Invariants to Preserve
1. Exact Graph2Seq adjacency construction
2. Attention mechanism behavior
3. Advantage normalization
4. Meta-parameter update formula
5. Random seed handling for reproducibility

## Appendix: Module Dependencies

```
meta_trainer.py
├── env/offloading_env.py
├── policies/meta_seq2seq_policy.py
│   ├── graph2seq_encoder.py
│   │   └── graph2seq_modules/*
│   └── model_helper.py (tf.contrib.rnn)
├── samplers/seq2seq_meta_sampler.py
├── baselines/vf_baseline.py
└── meta_algos/MRLCO.py
    └── distributions/categorical_pd.py
```

This completes the Phase 0 Understanding Report. The codebase is ready for migration planning (Phase 1).