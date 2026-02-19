# EVIDENCE MAP - MARGO Complete Technical Documentation

## 1. Architecture Details (from Code Analysis)

### 1.1 Graph2Seq Encoder (`policies/graph2seq_encoder.py`)

**Class:** `Graph2SeqEncoderAdapter`

| Component | Implementation | Lines |
|-----------|----------------|-------|
| Input dimension | `input_dim` (20) | L20-21 |
| Hidden dimension | `hidden_dim` (128) | L22 |
| GNN layers | `num_layers` (2) | L23 |
| Sample layer size | `sample_layer_size = 2` | L28 |
| Concatenation | `concat = True` | L29 |
| Dropout | `0.1` (train) / `0.0` (eval) | L30 |

**Sequence-to-Graph Conversion (Lines 36-82):**
```python
def sequence_to_graph(self, sequence_inputs):
    # Creates fully-connected graph within each sequence
    # Shape: [batch_size, seq_len, feature_dim] → graph representation
    fw_adj_info = tf.reshape(batch_adj, [total_nodes, seq_len])
    feature_info = tf.reshape(sequence_inputs, [total_nodes, feature_dim])
```

**Mean Aggregation (Lines 124-148):**
```python
fw_aggregator = MeanAggregator(
    dim_mul * self.hidden_dim, 
    self.hidden_dim, 
    concat=self.concat,  # True → output dim = 2*hidden_dim
    mode=self.mode
)
fw_hidden = fw_aggregator((fw_hidden, neigh_vec_hidden, fw_sampled_neighbors_len))
```

**Triple Readout (Lines 181-191) - NOVEL:**
```python
# 1. Attention Pooling
attn_logits = tf.layers.dense(encoder_outputs, 1, activation=None)
attn_weights = tf.nn.softmax(attn_logits, axis=1)
attn_pool = tf.reduce_sum(encoder_outputs * attn_weights, axis=1)

# 2. Mean Pooling
mean_pool = tf.reduce_mean(encoder_outputs, axis=1)

# 3. Max Pooling
max_pool = tf.reduce_max(encoder_outputs, axis=1)

# Concatenate and project
final_state = tf.layers.dense(
    tf.concat([mean_pool, max_pool, attn_pool], axis=-1),
    units=(4 * self.hidden_dim) if self.bidirectional else (2 * self.hidden_dim),
    activation=tf.tanh
)
```

### 1.2 Mean Aggregator (`policies/graph2seq_modules/aggregators.py`)

**Class:** `MeanAggregator` (Lines 72-133)

```python
class MeanAggregator(Layer):
    def __init__(self, input_dim, output_dim, neigh_input_dim=None,
            dropout=0, bias=True, act=tf.nn.relu,
            name=None, concat=False, mode="train"):
        # Weights initialization
        self.vars['neigh_weights'] = glorot([neigh_input_dim, output_dim])
        self.vars['self_weights'] = glorot([input_dim, output_dim])
        if self.bias:
            self.vars['bias'] = zeros([self.output_dim])

    def _call(self, inputs):
        self_vecs, neigh_vecs, neigh_len = inputs
        
        # Mean aggregation
        neigh_means = tf.reduce_mean(neigh_vecs, axis=1)
        
        # Transform
        from_neighs = tf.matmul(neigh_means, self.vars['neigh_weights'])
        from_self = tf.matmul(self_vecs, self.vars["self_weights"])
        
        # Concatenate or add
        if not self.concat:
            output = tf.add_n([from_self, from_neighs])
        else:
            output = tf.concat([from_self, from_neighs], axis=1)
        
        return self.act(output + self.vars['bias'])
```

### 1.3 Seq2Seq Policy (`policies/meta_seq2seq_policy.py`)

**Class:** `Seq2SeqNetwork` (Lines 64-368)

**Key Components:**
```python
# Hyperparameters (Lines 381-397)
hparams = tf.contrib.training.HParams(
    unit_type="lstm",
    encoder_units=128,
    decoder_units=128,
    n_features=3,  # vocab_size = 3 actions
    time_major=False,
    is_attention=True,  # Luong attention
    num_layers=2,
    start_token=0,
    end_token=2,
    is_bidencoder=False
)
```

**Luong Attention (Lines 334-339):**
```python
attention_mechanism = tf.contrib.seq2seq.LuongAttention(
    self.decoder_hidden_unit, attention_states)

decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
    decoder_cell, attention_mechanism,
    attention_layer_size=self.decoder_hidden_unit)
```

**Value Function (Lines 136-138):**
```python
self.pi = tf.nn.softmax(self.decoder_logits)
self.q = tf.layers.dense(self.decoder_logits, self.n_features, activation=None)
self.vf = tf.reduce_sum(self.pi * self.q, axis=-1)
```

### 1.4 MRLCO Meta-Learning (`meta_algos/MRLCO.py`)

**Class:** `MRLCO` (Lines 6-202)

**Hyperparameters:**
```python
def __init__(self, policy, meta_batch_size,
             outer_lr=1e-4,      # Actually 5e-4 in trainer
             inner_lr=0.1,      # Actually 5e-4 in trainer
             num_inner_grad_steps=4,  # Actually 1 in trainer
             clip_value=0.2,
             vf_coef=0.5,
             max_grad_norm=0.5):
```

**PPO Clipped Objective (Lines 73-81):**
```python
likelihood_ratio = self.policy.distribution.likelihood_ratio_sym(
    self.actions[i], self.old_logits[i], self.new_logits[i])

clipped_obj = tf.minimum(
    likelihood_ratio * self.advs[i],
    tf.clip_by_value(likelihood_ratio, 1.0 - self.clip_value, 1.0 + self.clip_value) * self.advs[i]
)
self.surr_obj.append(-tf.reduce_mean(clipped_obj))
```

**Value Function Loss (Lines 83-87):**
```python
vpredclipped = self.vpred[i] + tf.clip_by_value(
    self.vpred[i] - self.old_v[i], -self.clip_value, self.clip_value)
vf_losses1 = tf.square(self.vpred[i] - self.r[i])
vf_losses2 = tf.square(vpredclipped - self.r[i])
self.vf_loss.append(0.5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2)))
```

**Reptile Meta-Update (Lines 116-137):**
```python
def UpdateMetaPolicy(self):
    for i in range(self.meta_batch_size):
        params = sess.run(params_symbol)
        core_params = sess.run(core_params_symble)
        
        # First-order approximation gradient
        for i, core_var, meta_var in zip(itertools.count(), core_params, params):
            grads = (core_var - meta_var) / self.inner_lr / self.num_inner_grad_steps / self.meta_batch_size
            update_feed_dict[self.grads_placeholders[i]] = grads
        
        sess.run(self._outer_train, feed_dict=update_feed_dict)
    
    # Synchronize task policies with updated meta-policy
    self.policy.async_parameters()
```

### 1.5 Environment (`env/mec_offloaing_envs/offloading_env.py`)

**Resource Configuration (Lines 23-56):**
```python
class Resources:
    def __init__(self, mec_process_capable,
                 mobile_process_capable, bandwidth_up=7.0, bandwidth_dl=7.0,
                 v2v_process_capable=None, v2v_bandwidth=5.0,
                 use_energy=False, energy_config=None):
        
        self.mec_process_capable = 10.0 * 1024 * 1024  # 10 MIPS
        self.mobile_process_capable = 1.0 * 1024 * 1024  # 1 MIPS
        self.v2v_process_capable = 1.0 * 1024 * 1024  # Same as UE
        
        self.bandwidth_up = 7.0  # Mbps
        self.bandwidth_dl = 7.0  # Mbps
        self.v2v_bandwidth = 5.0  # Mbps (lower than MEC)
```

**Energy Configuration (Lines 97-112):**
```python
def _default_energy_config(self):
    return {
        'rho': 1.0,           # Local computation energy coefficient
        'f_l': 1.0,           # Local CPU frequency (normalized)
        'zeta': 2.0,          # CPU frequency exponent
        'ptx': 0.1,           # MEC transmission power (Watts)
        'prx': 0.05,          # MEC reception power (Watts)
        'ptx_v2v': 0.06,      # V2V TX power (40% lower than MEC)
        'prx_v2v': 0.03,      # V2V RX power (40% lower than MEC)
        'rho_v2v': 0.7,       # V2V computation coefficient (30% lower)
        'f_v2v': 1.0,         # V2V CPU frequency
        'latency_weight': 0.5,
        'energy_weight': 0.5,
    }
```

**Half-Duplex V2V Channel (Lines 519-583):**
```python
# V2V scheduling with half-duplex constraint
elif x == 2:
    # Step 1: V2V uplink starts when channel available and predecessors complete
    v2v_ul_start_time = max(v2v_channel_available_time,
                            max([max(FT_locally[j], FT_wr[j], FT_v2v_dl[j]) 
                                 for j in task_graph.pre_task_sets[i]]))
    
    # Step 2: Uplink transmission
    T_v2v_ul[i] = self.resource_cluster.v2v_transmission_cost(task.processing_data_size)
    FT_v2v_ul[i] = v2v_ul_start_time + T_v2v_ul[i]
    v2v_channel_available_time = FT_v2v_ul[i]  # Channel freed
    
    # Step 3: V2V execution on helper
    v2v_exec_start_time = max(v2v_available_time, FT_v2v_ul[i])
    FT_v2v_exec[i] = v2v_exec_start_time + exec_time
    
    # Step 4: V2V downlink (half-duplex: must wait for channel)
    v2v_dl_start_time = max(FT_v2v_exec[i], v2v_channel_available_time)
    FT_v2v_dl[i] = v2v_dl_start_time + T_v2v_dl[i]
```

### 1.6 Task Feature Encoding (`env/mec_offloaing_envs/offloading_task_graph.py`)

**20-Dimensional Feature Vector (Lines 220-260):**
```python
def encode_point_sequence_with_cost(self, resource_cluster):
    for i in range(self.task_number):
        task = self.task_list[i]
        
        # Timing costs (8 dimensions)
        local_process_cost = task.processing_data_size / resource_cluster.mobile_process_capable
        up_link_cost = resource_cluster.up_transmission_cost(task.processing_data_size)
        mec_process_cost = task.processing_data_size / resource_cluster.mec_process_capable
        down_link_cost = resource_cluster.dl_transmission_cost(task.transmission_data_size)
        
        # V2V costs (NEW)
        v2v_ul_cost = resource_cluster.v2v_transmission_cost(task.processing_data_size)
        v2v_helper_cost = task.processing_data_size / resource_cluster.v2v_process_capable
        v2v_dl_cost = resource_cluster.v2v_transmission_cost(task.transmission_data_size)

        task_embeding_vector = [i, local_process_cost, up_link_cost,
                                mec_process_cost, down_link_cost,
                                v2v_ul_cost, v2v_helper_cost, v2v_dl_cost]  # 8 dims

        pre_task_index_set = [...]   # 6 dims (padded with -1)
        succs_task_index_set = [...] # 6 dims (padded with -1)
        
        point_vector = task_embeding_vector + pre_task_index_set + succs_task_index_set  # 20 dims
```

**HEFT Prioritization (Lines 302-332):**
```python
def prioritize_tasks(self, resource_cluster):
    # Compute minimum execution time per task
    w = [0] * self.task_number
    for i, task in enumerate(self.task_list):
        t_locally = task.processing_data_size / resource_cluster.mobile_process_capable
        t_mec = up_transmission + mec_execution + down_transmission
        t_v2v = v2v_up + v2v_execution + v2v_down
        w[i] = min(t_locally, t_mec, t_v2v)
    
    # Compute upward rank recursively
    def rank(task_index):
        if len(self.succ_task_sets[task_index]) == 0:
            return w[task_index]
        else:
            return w[task_index] + max(rank(j) for j in self.succ_task_sets[task_index])
    
    # Sort by descending rank
    sort = np.argsort(rank_dict)[::-1]
    self.prioritize_sequence = sort
    return sort
```

---

## 2. Training Configuration (`meta_trainer.py`)

```python
# Line 201-314
META_BATCH_SIZE = 10

USE_ENERGY = True
ENERGY_CONFIG = {
    'use_energy': USE_ENERGY,
    'energy_weight': 0.5,
    'latency_weight': 0.5,
    'rho': 1.0,
    'f_l': 1.0,
    'zeta': 2.0,
    'ptx': 0.1,
    'prx': 0.05,
    'ptx_v2v': 0.06,
    'prx_v2v': 0.03,
    'rho_v2v': 0.7,
    'f_v2v': 1.0,
}

resource_cluster = Resources(
    mec_process_capable=(10.0 * 1024 * 1024),  # 10 MIPS
    mobile_process_capable=(1.0 * 1024 * 1024), # 1 MIPS
    bandwidth_up=7.0, bandwidth_dl=7.0,
    v2v_process_capable=(1.0 * 1024 * 1024),   # Same as UE
    v2v_bandwidth=5.0,                          # Lower than MEC
    use_energy=USE_ENERGY,
    energy_config=ENERGY_CONFIG
)

meta_policy = MetaSeq2SeqPolicy(
    meta_batch_size=META_BATCH_SIZE,
    obs_dim=20,
    encoder_units=128,
    decoder_units=128,
    vocab_size=3  # 3 actions: Local, MEC, V2V
)

algo = MRLCO(
    policy=meta_policy,
    inner_lr=5e-4,
    outer_lr=5e-4,
    meta_batch_size=META_BATCH_SIZE,
    num_inner_grad_steps=1,
    clip_value=0.2
)

sample_processor = Seq2SeqMetaSamplerProcessor(
    discount=0.99,
    gae_lambda=0.95,
    normalize_adv=True
)

trainer = Trainer(n_itr=3500)
```

---

## 3. Experimental Results Summary

### From `results/mrlco-compare/`

| Task | Method | Latency (ms) | Energy (J) |
|------|--------|--------------|------------|
| 1 | Greedy | 795 | 875 |
| 1 | MRLCO | 638 | 620 |
| 1 | **MARGO** | **634** | **595** |
| 2 | Greedy | 810 | 890 |
| 2 | MRLCO | 668 | 659 |
| 2 | **MARGO** | **661** | **633** |

### Improvement Summary

| Comparison | Latency | Energy |
|------------|---------|--------|
| MARGO vs MRLCO (Task 1) | -0.63% | -4.03% |
| MARGO vs MRLCO (Task 2) | -1.05% | -3.94% |
| MARGO vs Greedy (Task 1) | -20.3% | -32.0% |
| MARGO vs Greedy (Task 2) | -18.4% | -28.9% |

---

*This evidence map provides complete traceability from paper claims to source code.*
