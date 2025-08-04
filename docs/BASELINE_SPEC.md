# MRLCO TensorFlow 1.15 Baseline Specification

**Project**: Meta Reinforcement Learning for Combinatorial Optimization  
**Framework**: TensorFlow 1.15  
**Location**: `/workspace/mrlco_master/mrlco-new`  
**Date**: August 2025

## Table of Contents

1. [System Architecture & Data Flow](#system-architecture--data-flow)
2. [Environment & Task Specification](#environment--task-specification) 
3. [Network Architecture](#network-architecture)
4. [Mathematics & Algorithms](#mathematics--algorithms)
5. [Training & Evaluation](#training--evaluation)
6. [Hyperparameters & Configuration](#hyperparameters--configuration)
7. [File Organization](#file-organization)

---

## System Architecture & Data Flow

### Overall Pipeline
```
DAG Tasks → OffloadingEnvironment → Graph2SeqEncoder → LSTMDecoder+Attention → PPO → Meta-Learning
```

**Reference**: `meta_trainer.py:L145-L230`

### Meta-Learning Structure
- **Meta-batch size**: 10 tasks per batch
- **Architecture**: Core policy + 10 task-specific policies  
- **Update flow**: Inner PPO updates → Outer meta-gradient update
- **Reference**: `policies/meta_seq2seq_policy.py:L471-L543`

### Data Flow Stages

#### 1. Input Processing
- **Source**: Task graphs from 19 different task types
- **Location**: `env/mec_offloading_envs/data/meta_offloading_20/`
- **Format**: DAG adjacency matrices + task properties
- **Shape**: `[batch_size, seq_len, 17]` feature vectors
- **Reference**: `env/mec_offloading_envs/offloading_env.py:L89-L156`

#### 2. Environment Interface  
- **Class**: `OffloadingEnvironment`
- **Input shape**: `[batch_size=100, graph_number=100, feature_dim=17]`
- **Output shape**: Observations `[batch_size, seq_len, 17]`, rewards `[batch_size]`
- **Reference**: `env/mec_offloading_envs/offloading_env.py:L25-L88`

#### 3. Policy Network
- **Input**: Observations `[batch_size, seq_len, 17]`  
- **Encoder output**: `[batch_size, seq_len, 256]` (bidirectional)
- **Decoder output**: Action logits `[batch_size, seq_len, 2]`
- **Value output**: State values `[batch_size]`
- **Reference**: `policies/meta_seq2seq_policy.py:L66-L175`

---

## Environment & Task Specification

### Task Graph Structure
- **Nodes**: Computational tasks with execution requirements
- **Edges**: Data dependencies between tasks  
- **Features**: 17-dimensional vectors per task
- **Reference**: `env/mec_offloading_envs/offloading_task_graph.py:L15-L89`

### Feature Encoding (17-dimensional)
```python
# Reference: env/mec_offloading_envs/offloading_task_graph.py:L45-L67
features = [
    task_computation_cost,     # Computation requirement (flops)
    task_data_size,           # Data size for transmission  
    parent_count,             # Number of predecessor tasks
    child_count,              # Number of successor tasks
    level_in_dag,            # Topological level in DAG
    # ... (12 additional normalized features)
]
```

### Reward Function
- **Objective**: Minimize total task completion time
- **Formula**: `reward = -1.0 * total_completion_time`
- **Reference**: `env/mec_offloading_envs/offloading_env.py:L201-L215`

### Resource Configuration
```python
# Reference: meta_trainer.py:L147-L149
resource_cluster = Resources(
    mec_process_capable=(10.0 * 1024 * 1024),  # 10 MB/s MEC processing
    mobile_process_capable=(1.0 * 1024 * 1024), # 1 MB/s mobile processing  
    bandwidth_up=7.0,    # 7 Mbps uplink
    bandwidth_dl=7.0     # 7 Mbps downlink
)
```

---

## Network Architecture

### Graph2Seq Encoder

#### Architecture Overview
- **Purpose**: Replaces original LSTM encoder with graph neural network
- **Input**: Task feature sequences `[batch_size, seq_len, 17]`
- **Output**: Encoded representations `[batch_size, seq_len, 256]`
- **Reference**: `policies/graph2seq_encoder.py:L15-L45`

#### Layer Stack
```python
# Reference: policies/graph2seq_encoder.py:L46-L95
1. Fully connected embedding: 17 → 128 dims
2. Graph aggregation layers (configurable depth)  
3. BiLSTM encoding: 128 → 256 dims (bidirectional)
4. Output projection maintaining sequence structure
```

#### Graph Construction
- **Method**: Fully connected graph within each sequence
- **Adjacency**: All tasks connected to all other tasks in sequence
- **Masking**: Sequence-length based masking for variable lengths
- **Reference**: `policies/graph2seq_encoder.py:L125-L167`

### Seq2Seq Policy Network

#### Encoder Component
- **Type**: Graph2Seq encoder (as above)
- **Hidden units**: 128  
- **Bidirectional**: True
- **Reference**: `policies/meta_seq2seq_policy.py:L123-L132`

#### Decoder Component  
- **Type**: LSTM decoder with attention
- **Hidden units**: 128
- **Layers**: 2
- **Attention**: Luong attention mechanism
- **Reference**: `policies/meta_seq2seq_policy.py:L134-L175`

#### Attention Mechanism
```python
# Reference: policies/meta_seq2seq_policy.py:L334-L344
attention_mechanism = tf.contrib.seq2seq.LuongAttention(
    self.decoder_hidden_unit,  # 128 units
    attention_states           # encoder outputs [batch, seq, 256]
)

decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
    decoder_cell, 
    attention_mechanism,
    attention_layer_size=self.decoder_hidden_unit  # 128
)
```

#### Output Layers
- **Action logits**: Dense layer 128 → 2 (local/remote decision)
- **Value function**: Dense layer 128 → 1 (state value)
- **Reference**: `policies/meta_seq2seq_policy.py:L139-L141`

### Tensor Shapes at Boundaries

#### Input → Encoder
- **Input**: `[batch_size, seq_len, 17]` dtype=float32
- **Output**: `[batch_size, seq_len, 256]` dtype=float32

#### Encoder → Decoder  
- **Memory**: `[batch_size, seq_len, 256]` dtype=float32
- **State**: Tuple of `[batch_size, 128]` dtype=float32

#### Decoder → Output
- **Logits**: `[batch_size, seq_len, 2]` dtype=float32  
- **Values**: `[batch_size, seq_len]` dtype=float32
- **Actions**: `[batch_size, seq_len]` dtype=int32

---

## Mathematics & Algorithms

### PPO Algorithm

#### Policy Loss (Clipped Surrogate Objective)
```python
# Reference: meta_algos/ppo_offloading.py:L89-L105
ratio = tf.exp(new_logprobs - old_logprobs)  # π_θ(a|s) / π_θ_old(a|s)
clipped_ratio = tf.clip_by_value(ratio, 1-ε, 1+ε)  # ε = 0.3
surrogate_obj = tf.minimum(
    ratio * advantages,
    clipped_ratio * advantages  
)
policy_loss = -tf.reduce_mean(surrogate_obj)
```

#### Value Function Loss  
```python
# Reference: meta_algos/ppo_offloading.py:L107-L115
value_pred_clipped = old_values + tf.clip_by_value(
    value_pred - old_values, -ε, ε
)
value_loss_unclipped = tf.square(value_pred - returns)
value_loss_clipped = tf.square(value_pred_clipped - returns)
value_loss = 0.5 * tf.reduce_mean(
    tf.maximum(value_loss_unclipped, value_loss_clipped)
)
```

#### Advantage Computation (GAE)
```python  
# Reference: samplers/seq2seq_meta_sampler_process.py:L89-L105
def compute_advantages(rewards, values, gamma=0.99, lam=0.95):
    deltas = rewards + gamma * values[1:] - values[:-1]
    advantages = discount_cumsum(deltas, gamma * lam)
    returns = advantages + values[:-1]
    return advantages, returns
```

### Meta-Learning Algorithm (MRLCO)

#### First-Order Approximation  
```python
# Reference: meta_algos/MRLCO.py:L45-L67
# Inner loop: PPO update for each task
θ_i' = θ - α * ∇_θ L_PPO(θ, τ_i)  # Task-specific update

# Outer loop: Meta-gradient using first-order approximation  
∇_θ L_meta = (1/M) * Σ_i ∇_θ L_val(θ_i', D_i^val)
θ_new = θ - β * ∇_θ L_meta
```

#### Update Schedule
- **Inner learning rate (α)**: 5e-4
- **Outer learning rate (β)**: 5e-4  
- **Inner steps**: 1 per task
- **Meta batch size**: 10 tasks
- **Reference**: `meta_trainer.py:L202-L208`

### Reward & Return Computation

#### Environment Reward
```python
# Reference: env/mec_offloading_envs/offloading_env.py:L201-L215
def compute_reward(self, actions):
    # Simulate task execution with offloading decisions
    total_time = self.simulate_execution(actions)
    reward = -total_time  # Minimize completion time
    return reward
```

#### GAE Parameters
- **Discount factor (γ)**: 0.99
- **GAE parameter (λ)**: 0.95  
- **Reference**: `samplers/seq2seq_meta_sampler_process.py:L25-L27`

---

## Training & Evaluation

### Training Loop

#### Main Training Flow
```python
# Reference: meta_trainer.py:L220-L230
trainer = Trainer(
    algo=algo,
    env=env, 
    sampler=sampler,
    sample_processor=sample_processor,
    policy=meta_policy,
    n_itr=1000,
    start_itr=0,
    inner_batch_size=1000
)
```

#### Per-Iteration Process
1. **Sample Generation**: `sampler.obtain_samples()` 
2. **Sample Processing**: `sample_processor.process_samples()`
3. **Policy Update**: `algo.UpdateMetaPolicy()`
4. **Logging**: Performance metrics and checkpoints
5. **Reference**: `meta_algos/MRLCO.py:L25-L85`

### Sampling Strategy

#### Meta Sampler
- **Class**: `Seq2SeqMetaSampler`
- **Batch size**: 1000 trajectories per task
- **Tasks per meta-batch**: 10
- **Reference**: `samplers/seq2seq_meta_sampler.py:L15-L67`

#### Sample Processing
- **Advantage computation**: GAE with γ=0.99, λ=0.95
- **Normalization**: Zero-mean, unit-variance advantages
- **Reference**: `samplers/seq2seq_meta_sampler_process.py:L45-L89`

### Evaluation Metrics

#### Primary Metrics
- **Average return**: Mean episode reward across tasks
- **Policy loss**: PPO surrogate objective value  
- **Value loss**: MSE between predicted and actual returns
- **Reference**: `meta_trainer.py:L85-L105`

#### Logging Fields
```python
# Reference: meta_trainer.py:L105-L125
logger.record_tabular("AverageReturn", avg_reward)
logger.record_tabular("AverageDiscountedReturn", avg_discounted_return)  
logger.record_tabular("NumTrajs", total_trajectories)
logger.record_tabular("PolicyExecTime", policy_time)
logger.record_tabular("EnvExecTime", env_time)
```

### Checkpoints & I/O

#### Checkpoint Format
- **Format**: joblib pickle files
- **Location**: `./meta_model_inner_step1/`
- **Naming**: `meta_model_{iteration}.ckpt`
- **Content**: Policy network variables (weights, biases)
- **Reference**: `policies/meta_seq2seq_policy.py:L440-L468`

#### Checkpoint Saving
```python
# Reference: policies/meta_seq2seq_policy.py:L440-L451  
def save_variables(self, save_path, sess=None):
    sess = sess or tf.get_default_session()
    variables = self.get_variables()
    ps = sess.run(variables)
    save_dict = {v.name: value for v, value in zip(variables, ps)}
    joblib.dump(save_dict, save_path)
```

---

## Hyperparameters & Configuration

### Network Architecture
```python
# Reference: policies/meta_seq2seq_policy.py:L374-L396
hparams = tf.contrib.training.HParams(
    unit_type="lstm",
    encoder_units=128,
    decoder_units=128, 
    n_features=2,           # Binary action space (local/remote)
    time_major=False,
    is_attention=True,      # Luong attention enabled
    forget_bias=1.0,
    dropout=0,
    num_gpus=1,
    num_layers=2,
    num_residual_layers=0,
    start_token=0,
    end_token=2,
    is_bidencoder=False
)
```

### Training Parameters  
```python
# Reference: meta_trainer.py:L202-L219
algo = MRLCO(
    policy=meta_policy,
    meta_sampler=sampler,
    meta_sampler_process=sample_processor,
    inner_lr=5e-4,              # Inner loop learning rate
    outer_lr=5e-4,              # Outer loop learning rate  
    meta_batch_size=10,         # Tasks per meta-batch
    num_inner_grad_steps=1,     # PPO steps per task
    clip_value=0.3              # PPO clipping parameter
)
```

### Environment Configuration
```python
# Reference: meta_trainer.py:L151-L175  
env = OffloadingEnvironment(
    resource_cluster=resource_cluster,
    batch_size=100,             # Trajectories per task
    graph_number=100,           # Task graphs per trajectory
    graph_file_paths=[...],     # 19 different task types
    time_major=False
)
```

### Optimization Settings
- **Optimizer**: Adam (TensorFlow default)
- **Learning rate**: 5e-4 (both inner and outer)
- **Gradient clipping**: None specified
- **Batch size**: 100 trajectories × 10 tasks = 1000 total
- **Reference**: `meta_algos/MRLCO.py:L15-L25`

---

## File Organization

### Directory Structure
```
/workspace/mrlco_master/mrlco-new/
├── env/                    # Environment implementations
├── policies/               # Policy networks and distributions  
├── meta_algos/            # Meta-learning algorithms
├── samplers/              # Data sampling and processing
├── utils/                 # Utilities and helpers
├── baselines/             # Baseline algorithms
├── tests/                 # Test files
├── meta_trainer.py        # Main training script
└── meta_evaluator.py      # Evaluation script
```

### Key Implementation Files

#### Core Algorithm Files
- `meta_algos/MRLCO.py:L1-L85` - Meta-learning algorithm
- `meta_algos/ppo_offloading.py:L1-L150` - PPO implementation
- `policies/meta_seq2seq_policy.py:L1-L543` - Policy networks

#### Environment Files  
- `env/mec_offloading_envs/offloading_env.py:L1-L250` - Main environment
- `env/mec_offloading_envs/offloading_task_graph.py:L1-L120` - Task graphs

#### Network Architecture Files
- `policies/graph2seq_encoder.py:L1-L200` - Graph encoder
- `policies/distributions/categorical_pd.py:L1-L45` - Action distributions

#### Sampling & Processing Files
- `samplers/seq2seq_meta_sampler.py:L1-L89` - Meta-batch sampling
- `samplers/seq2seq_meta_sampler_process.py:L1-L125` - Sample processing

#### Utility Files
- `utils/utils.py:L1-L300` - General utilities  
- `utils/logger.py:L1-L400` - Logging functionality

### Configuration Files
- `meta_trainer.py:L145-L230` - Training configuration
- `meta_evaluator.py:L1-L150` - Evaluation configuration

### Data Files
- `env/mec_offloading_envs/data/meta_offloading_20/` - 19 task graph datasets
- Task graphs: `offload_random20_{1-22}/random.20.*` format

---

## Summary

This specification captures the complete TensorFlow 1.15 baseline implementation of MRLCO, including:

- **System architecture**: Meta-learning with Graph2Seq encoder and LSTM decoder
- **Mathematical formulation**: PPO with clipping, GAE advantages, first-order meta-gradients  
- **Network details**: Exact layer specifications, tensor shapes, attention mechanism
- **Training process**: Sampling strategy, update schedules, logging, checkpoints
- **Configuration**: All hyperparameters with precise file:line references

The implementation represents a sophisticated meta-reinforcement learning system for combinatorial optimization in mobile edge computing, with careful attention to tensor flow, mathematical correctness, and reproducible configuration.