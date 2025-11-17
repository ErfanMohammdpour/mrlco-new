# Technical Overview: MRLCO Meta-Reinforcement Learning for Task Offloading

## Executive Summary

This project implements **MRLCO (Meta-Reinforcement Learning for Computation Offloading)**, a meta-learning system for optimizing task offloading decisions in Mobile Edge Computing (MEC) environments. The system uses **PPO (Proximal Policy Optimization)** within a **meta-learning framework** to quickly adapt to new task graphs.

---

## 1. High-Level Architecture

### 1.1 System Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Meta-Learning Pipeline                     │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐ │
│  │   Trainer    │───▶│   Sampler    │───▶│  Environment  │ │
│  │ (meta_trainer)│   │(meta_sampler)│   │ (offloading)   │ │
│  └──────┬───────┘    └──────┬───────┘    └──────┬─────────┘ │
│         │                   │                   │            │
│         ▼                   ▼                   ▼            │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐ │
│  │   MRLCO      │    │  Processor   │    │ Task Graphs  │ │
│  │  Algorithm   │    │(sample_proc)  │    │   (DAGs)      │ │
│  └──────┬───────┘    └──────┬───────┘    └───────────────┘ │
│         │                   │                               │
│         ▼                   ▼                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │         MetaSeq2SeqPolicy (Graph2Seq Encoder)       │   │
│  │  ┌──────────────┐         ┌──────────────┐          │   │
│  │  │   Encoder    │────────▶│   Decoder    │          │   │
│  │  │ (Graph2Seq)  │         │  (LSTM+Attn)  │          │   │
│  │  └──────────────┘         └──────────────┘          │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Meta-Learning Framework

The system implements a **two-level optimization**:

1. **Inner Loop (Task-Specific)**: Fast adaptation to individual task graphs using PPO
2. **Outer Loop (Meta-Learning)**: Updates shared meta-policy using Reptile-style gradient aggregation

---

## 2. Module-by-Module Breakdown

### 2.1 Environment (`env/mec_offloaing_envs/offloading_env.py`)

**Purpose**: Simulates MEC task offloading environment

**Key Components**:
- `Resources`: Models MEC server and mobile device capabilities
  - `mec_process_capable`: MEC computation capacity (10.0 * 1024 * 1024)
  - `mobile_process_capable`: Mobile computation capacity (1.0 * 1024 * 1024)
  - `bandwidth_up/dl`: Wireless bandwidth (7.0 Mbps)

- `OffloadingEnvironment`: Main environment class
  - **Observation Space**: `[batch_size, seq_len, 17]` - Task features
  - **Action Space**: Binary `{0, 1}` - 0=local, 1=offload to MEC
  - **Reward**: Normalized latency score (see Section 3.3)

**Key Methods**:
- `get_scheduling_cost_step_by_step()`: Computes latency for a scheduling plan
- `get_reward_batch_step_by_step()`: Computes rewards from latency
- `score_func()`: Normalizes cost to reward: `-(cost - min_time) / (max_time - min_time)`

**Reward Calculation Flow**:
```
Action Sequence → Scheduling Plan → Latency Calculation → Normalized Reward
```

### 2.2 Policy Network (`policies/meta_seq2seq_policy.py`)

**Purpose**: Encodes task graphs and generates offloading decisions

**Architecture**:

```
Input: [batch_size, seq_len, obs_dim=17]
  │
  ├─▶ Encoder Embedding (FC layer) → [batch_size, seq_len, encoder_units=128]
  │
  ├─▶ Graph2Seq Encoder (see Section 2.3)
  │     │
  │     ├─▶ Sequence → Graph Conversion
  │     ├─▶ Graph Convolution Layers (2 layers, MeanAggregator)
  │     └─▶ Output: [batch_size, seq_len, 2*encoder_units] or [4*encoder_units] if bidirectional
  │
  └─▶ Decoder (LSTM with Attention)
        │
        ├─▶ LuongAttention Mechanism
        ├─▶ LSTM Cell (2 layers, decoder_units=128)
        └─▶ Output Projection → [batch_size, seq_len, vocab_size=2]
```

**Key Classes**:
- `Seq2SeqNetwork`: Core network architecture
- `Seq2SeqPolicy`: Single-task policy wrapper
- `MetaSeq2SeqPolicy`: Meta-learning wrapper (manages `meta_batch_size` task-specific policies)

**Value Function**:
- Uses Q-value head: `q = Dense(decoder_logits, vocab_size)`
- Value function: `vf = sum(pi * q, axis=-1)` where `pi = softmax(decoder_logits)`

### 2.3 Graph2Seq Encoder (`policies/graph2seq_encoder.py`)

**Purpose**: Converts sequence inputs to graph representation and encodes using Graph Convolutional Networks

**Architecture**:

```
Sequence Input [B, N, D]
  │
  ├─▶ Reshape to Graph: [B*N, D]
  │
  ├─▶ Create Adjacency: Fully-connected within each sequence
  │
  ├─▶ Graph Convolution Layer 1:
  │     ├─▶ MeanAggregator (neighbor aggregation)
  │     └─▶ Output: [B*N, hidden_dim]
  │
  ├─▶ Graph Convolution Layer 2:
  │     ├─▶ MeanAggregator (with concatenation)
  │     └─▶ Output: [B*N, 2*hidden_dim]
  │
  └─▶ Reshape & Pooling:
        ├─▶ Reshape: [B, N, 2*hidden_dim]
        ├─▶ Attention Pooling (mean + max + attention)
        └─▶ Final State: [B, state_dim] (LSTM-compatible)
```

**Key Features**:
- **Attention Mechanism**: Uses attention-weighted pooling for final state
- **Bidirectional Support**: Optional forward/backward graph convolution
- **Compatibility**: Outputs match original RNN encoder interface

### 2.4 Attention Mechanism

**Location**: `policies/meta_seq2seq_policy.py`, lines 323-344

**Type**: **Luong Attention** (additive attention)

**Implementation**:
```python
attention_mechanism = tf.contrib.seq2seq.LuongAttention(
    self.decoder_hidden_unit,  # attention_dim = 128
    attention_states  # encoder_outputs: [batch, seq_len, encoder_dim]
)

decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
    decoder_cell,
    attention_mechanism,
    attention_layer_size=self.decoder_hidden_unit
)
```

**How It Works**:
1. Encoder outputs: `[batch_size, seq_len, encoder_dim]`
2. At each decoder step, attention computes:
   - Attention scores: `score(h_decoder, h_encoder)`
   - Attention weights: `softmax(scores)`
   - Context vector: `sum(weights * encoder_outputs)`
3. Decoder uses context vector + current hidden state

### 2.5 PPO Algorithm (`meta_algos/MRLCO.py`)

**Purpose**: Implements Proximal Policy Optimization with meta-learning

**PPO Loss Components**:

1. **Policy Loss (Clipped Surrogate Objective)**:
   ```python
   likelihood_ratio = π_new(a|s) / π_old(a|s)
   clipped_obj = min(lr * adv, clip(lr, 1-ε, 1+ε) * adv)
   policy_loss = -mean(clipped_obj)
   ```

2. **Value Loss (Clipped Value Function)**:
   ```python
   vpred_clipped = v_old + clip(v_new - v_old, -ε, +ε)
   vf_loss = 0.5 * mean(max((v_new - r)², (vpred_clipped - r)²))
   ```

3. **Total Loss**:
   ```python
   total_loss = policy_loss + vf_coef * vf_loss
   ```

**Meta-Learning Update**:
- **Inner Update**: Per-task PPO updates (multiple gradient steps)
- **Outer Update**: Aggregates gradients across tasks:
  ```python
  grad = (core_params - task_params) / (inner_lr * num_inner_steps * meta_batch_size)
  ```

**Key Parameters**:
- `clip_value = 0.2`: PPO clipping parameter
- `vf_coef = 0.5`: Value function loss coefficient
- `inner_lr = 5e-4`: Inner loop learning rate
- `outer_lr = 5e-4`: Outer loop learning rate
- `num_inner_grad_steps = 1`: Number of inner PPO updates

### 2.6 Sampling (`samplers/seq2seq_meta_sampler.py`)

**Purpose**: Collects trajectories from multiple tasks

**Process**:
1. Sample `meta_batch_size` tasks
2. For each task, collect `rollouts_per_meta_task` trajectories
3. Policy generates action sequences for all tasks in parallel
4. Environment computes rewards and finish times

**Output Structure**:
```python
paths = {
    0: [path1, path2, ...],  # Task 0 trajectories
    1: [path1, path2, ...],  # Task 1 trajectories
    ...
}
```

Each path contains:
- `observations`: `[seq_len, obs_dim]`
- `actions`: `[seq_len]`
- `rewards`: `[seq_len]`
- `logits`: `[seq_len, vocab_size]`
- `values`: `[seq_len]`
- `finish_time`: scalar

### 2.7 Sample Processing (`samplers/seq2seq_meta_sampler_process.py`)

**Purpose**: Processes trajectories for PPO training

**Steps**:
1. **Compute Returns**: Discounted cumulative rewards
   ```python
   returns = discount_cumsum(rewards, discount=0.99)
   ```

2. **Fit Baseline**: Uses value function predictions as baseline

3. **Compute Advantages**: GAE (Generalized Advantage Estimation)
   ```python
   deltas = rewards + discount * V(s_{t+1}) - V(s_t)
   advantages = discount_cumsum(deltas, discount * gae_lambda)
   ```

4. **Normalize Advantages**: Optional normalization (zero mean, unit std)

**Output**: `samples_data` dictionary with:
- `observations`: `[batch_size * seq_len, obs_dim]`
- `actions`: `[batch_size * seq_len]`
- `rewards`: `[batch_size * seq_len]`
- `returns`: `[batch_size * seq_len]`
- `advantages`: `[batch_size * seq_len]`
- `values`: `[batch_size * seq_len]`
- `logits`: `[batch_size * seq_len, vocab_size]`
- `finish_time`: `[batch_size]`

---

## 3. Training Pipeline

### 3.1 Training Iteration Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    Training Iteration                        │
└─────────────────────────────────────────────────────────────┘

1. Sample Tasks
   ├─▶ env.sample_tasks(meta_batch_size=10)
   └─▶ Returns: [task_id_0, task_id_1, ..., task_id_9]

2. Collect Trajectories
   ├─▶ sampler.obtain_samples()
   ├─▶ For each task:
   │     ├─▶ Policy: get_actions(observations) → actions, logits, values
   │     └─▶ Env: step(actions) → rewards, finish_times
   └─▶ Returns: paths (dict of trajectories)

3. Process Samples
   ├─▶ processor.process_samples(paths)
   ├─▶ Compute returns, advantages, baselines
   └─▶ Returns: samples_data (per-task dictionaries)

4. Inner Policy Update (PPO)
   ├─▶ algo.UpdatePPOTarget(samples_data, batch_size=1000)
   ├─▶ For each task:
   │     ├─▶ Split into batches
   │     ├─▶ For num_inner_grad_steps:
   │     │     ├─▶ Compute PPO loss
   │     │     └─▶ Apply gradients to task-specific policy
   │     └─▶ Returns: policy_losses, value_losses
   └─▶ Task-specific policies updated

5. Resample (Evaluate Updated Policies)
   ├─▶ sampler.obtain_samples() (with updated policies)
   └─▶ Returns: new_paths

6. Outer Policy Update (Meta-Learning)
   ├─▶ algo.UpdateMetaPolicy()
   ├─▶ Compute gradients: (core_params - task_params) / (inner_lr * steps * batch_size)
   ├─▶ Update core_policy
   └─▶ Sync: task_policies ← core_policy

7. Logging
   ├─▶ Log rewards, latencies, losses
   └─▶ Save model (every save_interval iterations)
```

### 3.2 Tensor Flow: Training Iteration

**Step 1: Sampling**
```
Observations: [meta_batch_size, batch_size, seq_len, obs_dim=17]
  │
  ├─▶ Policy Forward Pass
  │     ├─▶ Encoder: [B, N, 17] → [B, N, 256] (if bidirectional)
  │     ├─▶ Decoder: [B, N, 256] → [B, N, 2]
  │     └─▶ Sample Actions: [B, N]
  │
  └─▶ Actions: [meta_batch_size, batch_size, seq_len]
```

**Step 2: Reward Computation**
```
Actions: [meta_batch_size, batch_size, seq_len]
  │
  ├─▶ Environment Step
  │     ├─▶ Scheduling Cost Calculation
  │     ├─▶ Latency: [seq_len] per trajectory
  │     └─▶ Reward: score_func(latency) → [seq_len]
  │
  └─▶ Rewards: [meta_batch_size, batch_size, seq_len]
```

**Step 3: Sample Processing**
```
Paths: List of dicts
  │
  ├─▶ Stack & Flatten
  │     ├─▶ Observations: [B*N, obs_dim]
  │     ├─▶ Actions: [B*N]
  │     ├─▶ Rewards: [B*N]
  │     ├─▶ Returns: [B*N] (discounted)
  │     ├─▶ Advantages: [B*N] (GAE)
  │     └─▶ Values: [B*N]
  │
  └─▶ samples_data: Dict with flattened arrays
```

**Step 4: PPO Update**
```
samples_data (per task)
  │
  ├─▶ Batch Split: [B*N] → batches of size 1000
  │
  ├─▶ For each batch:
  │     ├─▶ Feed to PPO graph:
  │     │     ├─▶ obs: [batch_size, seq_len, obs_dim]
  │     │     ├─▶ actions: [batch_size, seq_len]
  │     │     ├─▶ old_logits: [batch_size, seq_len, vocab_size]
  │     │     ├─▶ old_v: [batch_size, seq_len]
  │     │     ├─▶ advs: [batch_size, seq_len]
  │     │     └─▶ returns: [batch_size, seq_len]
  │     │
  │     ├─▶ Compute Loss:
  │     │     ├─▶ new_logits: [batch_size, seq_len, vocab_size]
  │     │     ├─▶ vpred: [batch_size, seq_len]
  │     │     ├─▶ policy_loss: scalar
  │     │     └─▶ value_loss: scalar
  │     │
  │     └─▶ Apply Gradients
  │
  └─▶ Task-specific policy updated
```

**Step 5: Meta Update**
```
Task-specific policies (after inner updates)
  │
  ├─▶ Compute Gradients:
  │     ├─▶ For each parameter:
  │     │     grad = (core_param - task_param) / (inner_lr * num_steps * meta_batch_size)
  │     │
  │     └─▶ Aggregate across tasks
  │
  ├─▶ Update Core Policy:
  │     └─▶ core_params ← core_params - outer_lr * aggregated_grads
  │
  └─▶ Sync: task_policies ← core_policy
```

### 3.3 Reward Structure

**Current Implementation**:
- **Objective**: Minimize task completion latency
- **Reward Function**: `score_func(cost, max_time, min_time) = -(cost - min_time) / (max_time - min_time)`
- **Reward Range**: `[-1, 0]` (normalized)
  - `-1`: Worst case (max_time)
  - `0`: Best case (min_time)

**Reward Computation**:
1. For each task in sequence:
   - Compute finish time based on action (local vs. offload)
   - Account for dependencies and resource availability
   - Compute incremental latency: `delta_makespan = max(finish_time, current_FT) - current_FT`
2. Normalize using task-specific `max_time` and `min_time`
3. Return step-wise rewards: `[seq_len]` per trajectory

---

## 4. Inference Pipeline

### 4.1 Inference Flow

```
Task Graph Input
  │
  ├─▶ Encode Task Graph
  │     └─▶ Observations: [batch_size, seq_len, obs_dim]
  │
  ├─▶ Policy Forward Pass
  │     ├─▶ Encoder: [B, N, 17] → [B, N, encoder_dim]
  │     ├─▶ Decoder (Greedy): [B, N, encoder_dim] → [B, N, 2]
  │     └─▶ Actions: argmax(logits) → [B, N]
  │
  └─▶ Execute Actions → Compute Latency
```

### 4.2 Tensor Dimensions: Inference

```
Input:
  observations: [batch_size, seq_len, obs_dim=17]
  decoder_full_length: [batch_size] (all = seq_len)

Encoder:
  encoder_inputs: [batch_size, seq_len, 17]
  encoder_embeddings: [batch_size, seq_len, 128]
  encoder_outputs: [batch_size, seq_len, 256] (if bidirectional)
  encoder_state: LSTMStateTuple(c=[batch_size, 128], h=[batch_size, 128])

Decoder (Greedy):
  decoder_inputs: [batch_size, seq_len] (shifted actions)
  decoder_logits: [batch_size, seq_len, 2]
  decoder_prediction: [batch_size, seq_len] (argmax)

Output:
  actions: [batch_size, seq_len] (binary: 0 or 1)
```

---

## 5. Key Tensor Shapes Summary

### 5.1 Policy Network

| Component | Shape | Description |
|-----------|-------|-------------|
| `encoder_inputs` | `[B, N, 17]` | Task features |
| `encoder_embeddings` | `[B, N, 128]` | Embedded features |
| `encoder_outputs` | `[B, N, 256]` | Graph2Seq encoder output |
| `encoder_state` | `LSTMStateTuple([B, 128], [B, 128])` | Encoder final state |
| `decoder_logits` | `[B, N, 2]` | Action logits |
| `decoder_prediction` | `[B, N]` | Sampled actions |
| `vf` (value function) | `[B, N]` | Value predictions |

### 5.2 Training Data

| Component | Shape | Description |
|-----------|-------|-------------|
| `observations` | `[B*N, 17]` | Flattened observations |
| `actions` | `[B*N]` | Flattened actions |
| `rewards` | `[B*N]` | Step-wise rewards |
| `returns` | `[B*N]` | Discounted returns |
| `advantages` | `[B*N]` | GAE advantages |
| `values` | `[B*N]` | Value function predictions |
| `logits` | `[B*N, 2]` | Action logits |
| `finish_time` | `[B]` | Task completion times |

Where:
- `B` = batch_size (number of trajectories per task)
- `N` = seq_len (number of tasks in graph, typically 20)
- `meta_batch_size` = 10 (number of tasks in meta-batch)

---

## 6. Dataset and Data Loading

### 6.1 Task Graph Format

- **Format**: GraphViz `.gv` files
- **Structure**: DAG (Directed Acyclic Graph)
- **Node Attributes**:
  - `size`: Processing data size
  - `expect_size`: Transmission data size
- **Edge Attributes**:
  - `size`: Data dependency size

### 6.2 Data Loading Process

1. **Load Graph Files**: Parse `.gv` files using `pydotplus`
2. **Encode Tasks**: Convert to feature vectors (17 dimensions)
   - Task features: `[task_id, local_cost, uplink_cost, mec_cost, downlink_cost]`
   - Predecessor indices: `[6 indices, padded with -1]`
   - Successor indices: `[6 indices, padded with -1]`
3. **Prioritize Tasks**: HEFT (Heterogeneous Earliest Finish Time) ranking
4. **Batch Creation**: Group graphs into batches

### 6.3 Feature Encoding

**17-Dimensional Feature Vector**:
```
[task_id, local_cost, uplink_cost, mec_cost, downlink_cost,
 pred_0, pred_1, pred_2, pred_3, pred_4, pred_5,
 succ_0, succ_1, succ_2, succ_3, succ_4, succ_5]
```

Where:
- `local_cost = processing_data_size / mobile_process_capable`
- `uplink_cost = up_transmission_cost(processing_data_size)`
- `mec_cost = processing_data_size / mec_process_capable`
- `downlink_cost = dl_transmission_cost(transmission_data_size)`

---

## 7. Important Design Decisions

### 7.1 Meta-Learning Strategy

- **Algorithm**: Reptile-style first-order meta-learning
- **Rationale**: Simpler than MAML, works well with PPO
- **Update Rule**: `grad = (core - task) / (lr * steps * batch_size)`

### 7.2 Graph Encoding

- **Choice**: Graph2Seq encoder (GCN-based) instead of RNN
- **Rationale**: Better captures graph structure and dependencies
- **Implementation**: Converts sequence to fully-connected graph, applies GCN layers

### 7.3 Attention Mechanism

- **Type**: Luong Attention (additive)
- **Location**: Decoder attention over encoder outputs
- **Purpose**: Allows decoder to focus on relevant encoder states

### 7.4 Reward Design

- **Normalization**: Task-specific min/max normalization
- **Range**: `[-1, 0]` for stability
- **Computation**: Step-wise incremental latency rewards

### 7.5 Value Function

- **Architecture**: Q-value head (state-action values)
- **Value Estimation**: `V(s) = sum(π(a|s) * Q(s,a))`
- **Baseline**: Uses value function predictions (not learned separately)

---

## 8. File Structure and Key Locations

```
mrlco-new/
├── meta_trainer.py              # Main training script
├── meta_evaluator.py             # Evaluation script (fine-tuning)
├── meta_algos/
│   ├── MRLCO.py                  # Meta-learning PPO algorithm
│   └── ppo_offloading.py        # Standard PPO (for evaluation)
├── policies/
│   ├── meta_seq2seq_policy.py   # Policy network (encoder + decoder)
│   ├── graph2seq_encoder.py     # Graph2Seq encoder implementation
│   └── graph2seq_modules/       # GCN components (aggregators, etc.)
├── env/
│   └── mec_offloaing_envs/
│       ├── offloading_env.py    # Environment and reward computation
│       └── offloading_task_graph.py  # Task graph parsing
├── samplers/
│   ├── seq2seq_meta_sampler.py  # Trajectory collection
│   └── seq2seq_meta_sampler_process.py  # Sample processing (GAE, etc.)
└── baselines/
    └── vf_baseline.py            # Value function baseline
```

---

## 9. Training Hyperparameters

**From `meta_trainer.py`**:
- `META_BATCH_SIZE = 10`: Number of tasks per meta-batch
- `batch_size = 100`: Number of trajectories per task
- `encoder_units = 128`: Encoder hidden dimension
- `decoder_units = 128`: Decoder hidden dimension
- `inner_lr = 5e-4`: Inner loop learning rate
- `outer_lr = 5e-4`: Outer loop learning rate
- `num_inner_grad_steps = 1`: Inner PPO updates per iteration
- `clip_value = 0.2`: PPO clipping parameter
- `inner_batch_size = 1000`: Batch size for PPO updates
- `n_itr = 3500`: Number of training iterations
- `discount = 0.99`: Reward discount factor
- `gae_lambda = 0.95`: GAE lambda parameter

---

## 10. Summary

This system implements a sophisticated meta-reinforcement learning framework for task offloading optimization:

1. **Architecture**: Graph2Seq encoder + LSTM decoder with attention
2. **Algorithm**: PPO with meta-learning (Reptile-style)
3. **Objective**: Minimize task completion latency
4. **Training**: Two-level optimization (inner: task-specific, outer: meta-learning)
5. **Reward**: Normalized latency score `[-1, 0]`

The system is designed to quickly adapt to new task graphs with minimal fine-tuning, making it suitable for dynamic MEC environments.


