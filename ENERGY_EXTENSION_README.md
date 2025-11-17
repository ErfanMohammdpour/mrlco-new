# Extending the Project to Control Energy (Optional Feature)

## Overview

The MRLCO system is designed to optimize **task completion latency** in Mobile Edge Computing (MEC) environments. This document describes how to extend the system to also optimize **energy consumption** as an optional objective, while maintaining full backward compatibility with the original latency-only optimization.

---

## Current Latency Handling

### How Latency is Currently Optimized

The system optimizes latency through the following mechanism:

1. **Reward Function**: The reward is computed as a normalized latency score:
   ```python
   latency_score = -(cost - min_time) / (max_time - min_time)
   ```
   Where:
   - `cost`: Actual task completion time
   - `min_time`: Theoretical minimum time (all tasks offloaded optimally)
   - `max_time`: Theoretical maximum time (worst-case scenario)

2. **Reward Range**: The reward is normalized to `[-1, 0]`:
   - `-1`: Worst case (maximum latency)
   - `0`: Best case (minimum latency)

3. **Training Objective**: The PPO algorithm maximizes cumulative rewards, which directly minimizes latency.

### Where Latency is Computed

- **Environment**: `env/mec_offloaing_envs/offloading_env.py`
  - `get_scheduling_cost_step_by_step()`: Computes step-wise latency
  - `get_reward_batch_step_by_step()`: Converts latency to rewards

- **Reward Structure**: Step-wise rewards `[seq_len]` per trajectory, where each reward corresponds to the incremental latency contribution of each task.

---

## Adding Energy as an Optional Extension

### Overview

The energy extension allows the model to optimize both **latency** and **energy consumption** simultaneously. The extension is controlled by a single boolean flag `use_energy`, ensuring complete backward compatibility.

### Energy Model

The energy consumption model includes two components:

1. **Local Execution Energy**:
   ```
   Energy_local = T_l × ρ × (f_l ^ ζ)
   ```
   Where:
   - `T_l`: Local execution time
   - `ρ`: Computation energy coefficient
   - `f_l`: Local CPU frequency
   - `ζ`: CPU frequency exponent (typically 2-3)

2. **Offloading Energy** (Transmission):
   ```
   Energy_offload = T_ul × P_tx + T_dl × P_rx
   ```
   Where:
   - `T_ul`: Uplink transmission time
   - `T_dl`: Downlink transmission time
   - `P_tx`: Transmission power (Watts)
   - `P_rx`: Reception power (Watts)

---

## Implementation Steps

### Step 1: Modify the Model

**File**: `env/mec_offloaing_envs/offloading_env.py`

#### 1.1 Update `Resources` Class

Add energy parameters to the `Resources` class:

```python
class Resources(object):
    def __init__(self, mec_process_capable, mobile_process_capable,
                 bandwidth_up=7.0, bandwidth_dl=7.0,
                 use_energy=False, energy_config=None):
        # ... existing initialization ...
        
        self.use_energy = use_energy
        if energy_config is None:
            energy_config = self._default_energy_config()
        self.energy_config = energy_config
        
        # Store energy weights if enabled
        if self.use_energy:
            self.latency_weight = energy_config.get('latency_weight', 0.5)
            self.energy_weight = energy_config.get('energy_weight', 0.5)
    
    def _default_energy_config(self):
        return {
            'rho': 1.0,
            'f_l': 1.0,
            'zeta': 2.0,
            'ptx': 0.1,
            'prx': 0.05,
            'latency_weight': 0.5,
            'energy_weight': 0.5,
        }
    
    def compute_local_energy(self, execution_time):
        """Compute energy for local execution"""
        if not self.use_energy:
            return 0.0
        return execution_time * self.energy_config['rho'] * \
               (self.energy_config['f_l'] ** self.energy_config['zeta'])
    
    def compute_transmission_energy(self, uplink_time, downlink_time):
        """Compute energy for transmission"""
        if not self.use_energy:
            return 0.0
        return (uplink_time * self.energy_config['ptx'] + 
                downlink_time * self.energy_config['prx'])
```

#### 1.2 Update `get_scheduling_cost_step_by_step()`

Modify the method to compute energy alongside latency:

```python
def get_scheduling_cost_step_by_step(self, plan, task_graph):
    # ... existing latency computation code ...
    
    return_latency = []
    return_energy = []  # NEW: Energy per step
    
    for item in plan:
        i = item[0]
        task = task_graph.task_list[i]
        x = item[1]
        
        # ... existing latency computation ...
        
        # NEW: Compute energy consumption
        if self.use_energy:
            if x == 0:  # Local execution
                energy_consumption = self.resource_cluster.compute_local_energy(T_l[i])
            else:  # Offloading
                energy_consumption = self.resource_cluster.compute_transmission_energy(
                    T_ul[i], T_dl[i])
            return_energy.append(energy_consumption)
        else:
            return_energy.append(0.0)
        
        return_latency.append(delta_make_span)
    
    # Return based on energy flag
    if self.use_energy:
        return return_latency, current_FT, return_energy
    else:
        return return_latency, current_FT
```

#### 1.3 Add Energy Bounds Calculation

Add a helper method to compute energy bounds:

```python
def _compute_energy_bounds(self, task_graph, max_time, min_time):
    """Compute theoretical min/max energy consumption"""
    if not self.use_energy:
        return 0.0, 0.0
    
    # Max energy: All tasks executed locally
    max_energy = sum([
        self.resource_cluster.compute_local_energy(
            task.processing_data_size / self.resource_cluster.mobile_process_capable
        ) for task in task_graph.task_list
    ])
    
    # Min energy: All tasks offloaded (minimal transmission)
    min_energy = sum([
        self.resource_cluster.compute_transmission_energy(
            self.resource_cluster.up_transmission_cost(task.processing_data_size),
            self.resource_cluster.dl_transmission_cost(task.transmission_data_size)
        ) for task in task_graph.task_list
    ])
    
    return max_energy, min_energy
```

#### 1.4 Update `get_reward_batch_step_by_step()`

Modify to combine latency and energy rewards:

```python
def get_reward_batch_step_by_step(self, action_sequence_batch, task_graph_batch,
                                  max_running_time_batch, min_running_time_batch):
    target_batch = []
    task_finish_time_batch = []
    energy_batch = []  # NEW: Energy batch
    
    for i in range(len(action_sequence_batch)):
        max_running_time = max_running_time_batch[i]
        min_running_time = min_running_time_batch[i]
        
        task_graph = task_graph_batch[i]
        self.resource_cluster.reset()
        plan = action_sequence_batch[i]
        
        if self.use_energy:
            # Get latency and energy
            cost, task_finish_time, energy = self.get_scheduling_cost_step_by_step(
                plan, task_graph)
            
            # Compute energy bounds
            max_energy, min_energy = self._compute_energy_bounds(
                task_graph, max_running_time, min_running_time)
            
            # Normalize energy
            energy_score = self.score_func(energy, max_energy, min_energy)
            
            # Normalize latency
            latency_score = self.score_func(cost, max_running_time, min_running_time)
            
            # Combine rewards
            combined_score = (self.resource_cluster.latency_weight * latency_score + 
                            self.resource_cluster.energy_weight * energy_score)
            
            target_batch.append(combined_score)
            energy_batch.append(energy)
        else:
            # Original behavior
            cost, task_finish_time = self.get_scheduling_cost_step_by_step(
                plan, task_graph)
            latency = self.score_func(cost, max_running_time, min_running_time)
            target_batch.append(latency)
            energy_batch.append([])  # Empty for backward compatibility
        
        task_finish_time_batch.append(task_finish_time)
    
    target_batch = np.array(target_batch)
    
    if self.use_energy:
        return target_batch, task_finish_time_batch, energy_batch
    else:
        return target_batch, task_finish_time_batch
```

### Step 2: Update Reward Computation

The reward function automatically handles energy when enabled. No changes needed to the reward computation logic - it's handled in Step 1.4 above.

### Step 3: Update Dataset/Logging

**File**: `samplers/seq2seq_meta_sampler.py`

Update path collection to handle energy:

```python
def obtain_samples(self, log=False, log_prefix=''):
    # ... existing code ...
    
    for idx, observation, action, logit, reward, value, done, env_info in zip(...):
        # Handle energy in env_info
        if self.env.use_energy:
            if isinstance(env_info, tuple) and len(env_info) == 2:
                task_finish_times, energy_info = env_info
            else:
                task_finish_times = env_info
                energy_info = None
        else:
            task_finish_times = env_info
            energy_info = None
        
        # Store energy in path if enabled
        for single_ob, single_ac, single_logit, single_reward, single_value, single_task_finish_time \
                in zip(observation, action, logit, reward, value, task_finish_times):
            running_paths[idx]["observations"] = single_ob
            running_paths[idx]["actions"] = single_ac
            running_paths[idx]["logits"] = single_logit
            running_paths[idx]["rewards"] = single_reward
            running_paths[idx]["finish_time"] = single_task_finish_time
            running_paths[idx]["values"] = single_value
            
            # NEW: Store energy if enabled
            if self.env.use_energy and energy_info is not None:
                running_paths[idx]["energy"] = energy_info
            
            # ... rest of existing code ...
```

**File**: `samplers/seq2seq_meta_sampler_process.py`

Update sample processing:

```python
def _append_path_data(self, paths):
    observations = np.array([path["observations"] for path in paths])
    actions = np.array([path["actions"] for path in paths])
    logits = np.array([path["logits"] for path in paths])
    rewards = np.array([path["rewards"] for path in paths])
    returns = np.array([path["returns"] for path in paths])
    values = np.array([path["values"] for path in paths])
    advantages = np.array([path["advantages"] for path in paths])
    finish_time = np.array([path["finish_time"] for path in paths])
    
    # NEW: Handle energy if present
    if 'energy' in paths[0] and paths[0]['energy'] is not None:
        energy = np.array([path["energy"] for path in paths])
        return observations, actions, logits, rewards, returns, values, advantages, finish_time, energy
    else:
        return observations, actions, logits, rewards, returns, values, advantages, finish_time
```

### Step 4: Update Config

**File**: `meta_trainer.py`

Add energy configuration:

```python
if __name__ == "__main__":
    # ... existing imports ...
    
    # ========== ENERGY CONFIGURATION ==========
    USE_ENERGY = False  # Set to True to enable energy optimization
    
    ENERGY_CONFIG = {
        'use_energy': USE_ENERGY,
        'energy_weight': 0.5,      # Weight for energy in combined reward
        'latency_weight': 0.5,      # Weight for latency in combined reward
        'rho': 1.0,                # Computation energy coefficient
        'f_l': 1.0,                # Local CPU frequency (normalized)
        'zeta': 2.0,               # CPU frequency exponent
        'ptx': 0.1,                # Transmission power (Watts)
        'prx': 0.05,               # Reception power (Watts)
        'normalize_energy': True,   # Whether to normalize energy rewards
    }
    # ==========================================
    
    resource_cluster = Resources(
        mec_process_capable=(10.0 * 1024 * 1024),
        mobile_process_capable=(1.0 * 1024 * 1024),
        bandwidth_up=7.0,
        bandwidth_dl=7.0,
        use_energy=USE_ENERGY,      # NEW: Energy flag
        energy_config=ENERGY_CONFIG  # NEW: Energy configuration
    )
    
    # ... rest of existing code ...
    
    # Update logging if energy is enabled
    if USE_ENERGY:
        avg_energy = np.mean([np.sum(samples_data[i]['energy']) 
                             for i in range(5)])
        logger.logkv('Average energy', avg_energy)
```

**File**: `meta_evaluator.py`

Apply the same changes as in `meta_trainer.py`.

### Step 5: Update Training/Evaluation

No changes needed to the training loop itself - the modifications above handle everything automatically. The training loop will:

- Use combined rewards when `use_energy=True`
- Use latency-only rewards when `use_energy=False`
- Log energy metrics when enabled

---

## Usage Instructions

### Running in Legacy Mode (Latency Only)

To run the system exactly as before (latency-only optimization):

```python
USE_ENERGY = False

resource_cluster = Resources(
    mec_process_capable=(10.0 * 1024 * 1024),
    mobile_process_capable=(1.0 * 1024 * 1024),
    bandwidth_up=7.0,
    bandwidth_dl=7.0,
    use_energy=USE_ENERGY  # False = original behavior
)
```

**Behavior**:
- System behaves identically to original implementation
- No energy computation occurs
- Rewards are based solely on latency
- Zero performance overhead

### Running with Energy Optimization

To enable energy optimization:

```python
USE_ENERGY = True

ENERGY_CONFIG = {
    'use_energy': True,
    'energy_weight': 0.5,      # 50% weight for energy
    'latency_weight': 0.5,      # 50% weight for latency
    'rho': 1.0,
    'f_l': 1.0,
    'zeta': 2.0,
    'ptx': 0.1,
    'prx': 0.05,
    'normalize_energy': True,
}

resource_cluster = Resources(
    mec_process_capable=(10.0 * 1024 * 1024),
    mobile_process_capable=(1.0 * 1024 * 1024),
    bandwidth_up=7.0,
    bandwidth_dl=7.0,
    use_energy=USE_ENERGY,
    energy_config=ENERGY_CONFIG
)
```

**Behavior**:
- System optimizes both latency and energy
- Rewards combine latency and energy scores
- Energy metrics are logged during training
- Model learns to balance latency vs. energy trade-offs

### Tuning Energy vs. Latency Trade-off

Adjust the weights to prioritize latency or energy:

```python
# Prioritize latency
ENERGY_CONFIG = {
    'energy_weight': 0.2,      # 20% weight for energy
    'latency_weight': 0.8,      # 80% weight for latency
    # ... other parameters ...
}

# Prioritize energy
ENERGY_CONFIG = {
    'energy_weight': 0.8,      # 80% weight for energy
    'latency_weight': 0.2,      # 20% weight for latency
    # ... other parameters ...
}

# Equal weighting (default)
ENERGY_CONFIG = {
    'energy_weight': 0.5,      # 50% weight for energy
    'latency_weight': 0.5,      # 50% weight for latency
    # ... other parameters ...
}
```

---

## Tensor Shape Changes

### When `use_energy = False`

**No changes** - all tensors remain identical:
- `rewards`: `[batch_size, seq_len]`
- `samples_data`: Same structure as original

### When `use_energy = True`

**New optional tensors**:
- `energy`: `[batch_size, seq_len]` - Energy consumption per step
- `samples_data['energy']`: `[batch_size * seq_len]` - Flattened energy (optional)

**Modified tensors**:
- `rewards`: Now combines latency and energy (shape unchanged: `[batch_size, seq_len]`)

---

## Backward Compatibility Guarantee

The implementation ensures **100% backward compatibility**:

1. **Identical Behavior**: When `use_energy=False`, the system behaves exactly as the original
2. **Zero Overhead**: No energy computation occurs when disabled
3. **Same Tensor Shapes**: All tensors match original shapes when disabled
4. **Same Training Dynamics**: Loss curves and convergence behavior are identical
5. **Model Compatibility**: Existing checkpoints/models work without modification

---

## Testing the Extension

### Verify Backward Compatibility

1. **Reward Comparison**:
   ```python
   # Run with use_energy=False
   rewards_legacy = train_with_energy(use_energy=False)
   
   # Run with use_energy=True, energy_weight=0
   rewards_energy = train_with_energy(use_energy=True, energy_weight=0)
   
   # Verify rewards are identical
   assert np.allclose(rewards_legacy, rewards_energy)
   ```

2. **Tensor Shape Verification**:
   ```python
   # Verify shapes match when disabled
   assert samples_data_legacy.keys() == samples_data_energy.keys()
   assert samples_data_legacy['rewards'].shape == samples_data_energy['rewards'].shape
   ```

### Verify Energy Functionality

1. **Energy Computation**:
   ```python
   # Verify energy values are non-zero when enabled
   assert np.any(energy_values > 0) when use_energy=True
   ```

2. **Combined Rewards**:
   ```python
   # Verify rewards combine latency and energy correctly
   combined_reward = latency_weight * latency_score + energy_weight * energy_score
   assert np.allclose(rewards, combined_reward)
   ```

---

## Summary

The energy extension provides a clean, backward-compatible way to optimize both latency and energy:

- **Single Flag Control**: `use_energy` boolean controls all energy logic
- **Zero Overhead**: When disabled, no energy computation occurs
- **Additive Design**: Energy added alongside latency, not replacing it
- **Configurable Weighting**: Users can tune latency vs. energy trade-off
- **Full Compatibility**: Original behavior preserved when flag is False

To use the extension, simply set `use_energy=True` and configure the energy parameters. The system will automatically optimize both objectives while maintaining full compatibility with the original latency-only mode.


