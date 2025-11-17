# Energy Extension Design: Adding Optional Energy Control to MRLCO

## Overview

This document describes the design for adding **energy optimization** as an optional objective to the MRLCO system, while maintaining **100% backward compatibility** with the existing latency-only optimization.

---

## 1. Design Principles

### 1.1 Backward Compatibility Requirement

**Critical**: The system MUST behave identically to the original implementation when `use_energy = False`.

- **Zero modifications** to existing behavior when flag is disabled
- **No performance degradation** in latency-only mode
- **Identical tensor shapes** when energy is disabled
- **Same training dynamics** and convergence behavior

### 1.2 Extension Strategy

- **Single Boolean Flag**: `use_energy = True/False`
- **Conditional Logic**: All energy-related code behind `if use_energy:` checks
- **Additive Changes**: Energy components added alongside (not replacing) latency
- **Configurable Weighting**: Energy weight can be tuned independently

---

## 2. Energy Model

### 2.1 Energy Consumption Components

Based on the commented code in `offloading_env.py`, energy consumption includes:

1. **Local Execution Energy**:
   ```python
   energy_local = T_l * rho * (f_l ** zeta)
   ```
   Where:
   - `T_l`: Local execution time
   - `rho`: Energy coefficient for computation
   - `f_l`: Local CPU frequency
   - `zeta`: CPU frequency exponent (typically 2-3)

2. **Offloading Energy** (Transmission):
   ```python
   energy_offload = T_ul * ptx + T_dl * prx
   ```
   Where:
   - `T_ul`: Uplink transmission time
   - `T_dl`: Downlink transmission time
   - `ptx`: Transmission power (uplink)
   - `prx`: Reception power (downlink)

### 2.2 Energy Parameters

**New Configuration Parameters**:
```python
energy_config = {
    'use_energy': False,  # Master flag
    'energy_weight': 0.5,  # Weight for energy in combined reward
    'latency_weight': 0.5,  # Weight for latency in combined reward
    'rho': 1.0,  # Computation energy coefficient
    'f_l': 1.0,  # Local CPU frequency (normalized)
    'zeta': 2.0,  # CPU frequency exponent
    'ptx': 0.1,  # Transmission power (Watts)
    'prx': 0.05,  # Reception power (Watts)
    'normalize_energy': True,  # Whether to normalize energy rewards
}
```

---

## 3. Integration Points

### 3.1 Files to Modify

#### **File 1: `env/mec_offloaing_envs/offloading_env.py`**

**Changes**:
1. Add energy parameters to `Resources` class
2. Modify `get_scheduling_cost_step_by_step()` to compute energy
3. Modify `get_reward_batch_step_by_step()` to include energy in rewards
4. Add energy normalization similar to latency normalization

**Key Modifications**:

```python
class Resources(object):
    def __init__(self, ..., use_energy=False, energy_config=None):
        # ... existing code ...
        self.use_energy = use_energy
        if energy_config is None:
            energy_config = self._default_energy_config()
        self.energy_config = energy_config

    def _default_energy_config(self):
        return {
            'rho': 1.0,
            'f_l': 1.0,
            'zeta': 2.0,
            'ptx': 0.1,
            'prx': 0.05,
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

**Modify `get_scheduling_cost_step_by_step()`**:
```python
def get_scheduling_cost_step_by_step(self, plan, task_graph):
    # ... existing latency computation ...
    
    return_latency = []
    return_energy = []  # NEW: Energy per step
    
    for item in plan:
        # ... existing latency computation ...
        
        if self.use_energy:
            if x == 0:  # Local execution
                energy_consumption = self.resource_cluster.compute_local_energy(T_l[i])
            else:  # Offloading
                energy_consumption = self.resource_cluster.compute_transmission_energy(
                    T_ul[i], T_dl[i])
            return_energy.append(energy_consumption)
        else:
            return_energy.append(0.0)
    
    if self.use_energy:
        return return_latency, current_FT, return_energy
    else:
        return return_latency, current_FT
```

**Modify `get_reward_batch_step_by_step()`**:
```python
def get_reward_batch_step_by_step(self, action_sequence_batch, task_graph_batch,
                                  max_running_time_batch, min_running_time_batch):
    target_batch = []
    task_finish_time_batch = []
    energy_batch = []  # NEW: Energy batch
    
    for i in range(len(action_sequence_batch)):
        # ... existing code ...
        
        if self.use_energy:
            cost, task_finish_time, energy = self.get_scheduling_cost_step_by_step(
                plan, task_graph)
            
            # Compute energy bounds for normalization
            max_energy, min_energy = self._compute_energy_bounds(
                task_graph, max_running_time, min_running_time)
            
            # Normalize energy
            energy_score = self.score_func(energy, max_energy, min_energy)
            
            # Combine latency and energy rewards
            latency_score = self.score_func(cost, max_running_time, min_running_time)
            
            combined_score = (self.latency_weight * latency_score + 
                            self.energy_weight * energy_score)
            
            target_batch.append(combined_score)
            energy_batch.append(energy)
        else:
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

#### **File 2: `samplers/seq2seq_meta_sampler.py`**

**Changes**: Handle energy in path collection

```python
def obtain_samples(self, log=False, log_prefix=''):
    # ... existing code ...
    
    for idx, observation, action, logit, reward, value, done, env_info in zip(...):
        # env_info may contain energy information
        if self.env.use_energy:
            task_finish_times, energy_info = env_info
        else:
            task_finish_times = env_info
            energy_info = None
        
        # Store energy in path if enabled
        if self.env.use_energy:
            running_paths[idx]["energy"] = energy_info
        # ... rest of existing code ...
```

#### **File 3: `samplers/seq2seq_meta_sampler_process.py`**

**Changes**: Process energy data in sample processing

```python
def _append_path_data(self, paths):
    # ... existing code ...
    
    if self.env.use_energy and 'energy' in paths[0]:
        energy = np.array([path["energy"] for path in paths])
    else:
        energy = None
    
    if energy is not None:
        return observations, actions, logits, rewards, returns, values, advantages, finish_time, energy
    else:
        return observations, actions, logits, rewards, returns, values, advantages, finish_time
```

#### **File 4: `meta_trainer.py`**

**Changes**: Add energy configuration and logging

```python
if __name__ == "__main__":
    # ... existing code ...
    
    # Energy configuration
    USE_ENERGY = False  # Set to True to enable energy optimization
    ENERGY_CONFIG = {
        'use_energy': USE_ENERGY,
        'energy_weight': 0.5,
        'latency_weight': 0.5,
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
    
    env = OffloadingEnvironment(
        resource_cluster=resource_cluster,
        # ... existing parameters ...
    )
    
    # ... rest of training code ...
    
    # Logging updates
    if USE_ENERGY:
        logger.logkv('Average energy', avg_energy)
```

#### **File 5: `meta_evaluator.py`**

**Changes**: Similar to trainer, add energy configuration

---

## 4. Tensor Shape Changes

### 4.1 When `use_energy = False`

**No changes** - all tensors remain identical:
- `rewards`: `[batch_size, seq_len]`
- `samples_data`: Same structure as before

### 4.2 When `use_energy = True`

**New tensors**:
- `energy`: `[batch_size, seq_len]` - Energy consumption per step
- `rewards`: `[batch_size, seq_len]` - Combined latency + energy rewards
- `samples_data['energy']`: `[batch_size * seq_len]` - Flattened energy (optional)

**Modified tensors**:
- `rewards`: Now combines latency and energy (shape unchanged)

---

## 5. Reward Function Changes

### 5.1 Current Reward (Latency Only)

```python
latency_score = -(cost - min_time) / (max_time - min_time)
reward = latency_score
```

### 5.2 New Reward (Latency + Energy)

```python
if use_energy:
    latency_score = -(cost - min_time) / (max_time - min_time)
    energy_score = -(energy - min_energy) / (max_energy - min_energy)
    
    reward = (latency_weight * latency_score + 
              energy_weight * energy_score)
else:
    reward = latency_score  # Original behavior
```

### 5.3 Energy Bounds Calculation

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

---

## 6. Implementation Steps

### Step 1: Modify `Resources` Class
- Add `use_energy` flag and `energy_config`
- Add energy computation methods
- Ensure backward compatibility (return 0.0 when disabled)

### Step 2: Modify `OffloadingEnvironment`
- Update `get_scheduling_cost_step_by_step()` to compute energy
- Update `get_reward_batch_step_by_step()` to combine rewards
- Add energy bounds calculation
- Ensure conditional returns match original when disabled

### Step 3: Update Sampling
- Modify `seq2seq_meta_sampler.py` to handle energy in paths
- Update `seq2seq_meta_sampler_process.py` to process energy data
- Ensure paths work with/without energy

### Step 4: Update Training Scripts
- Add energy configuration to `meta_trainer.py`
- Add energy logging
- Update `meta_evaluator.py` similarly

### Step 5: Testing
- Test with `use_energy=False` - verify identical behavior
- Test with `use_energy=True` - verify energy computation
- Verify tensor shapes match expectations

---

## 7. Backward Compatibility Checklist

- [ ] When `use_energy=False`, `get_scheduling_cost_step_by_step()` returns same tuple as before
- [ ] When `use_energy=False`, `get_reward_batch_step_by_step()` returns same tuple as before
- [ ] When `use_energy=False`, no energy computation occurs (zero overhead)
- [ ] When `use_energy=False`, reward values are identical to original
- [ ] When `use_energy=False`, tensor shapes are unchanged
- [ ] When `use_energy=False`, training dynamics are identical
- [ ] Existing checkpoints/models work without modification

---

## 8. Configuration Example

### 8.1 Legacy Mode (Latency Only)

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

### 8.2 Energy Mode (Latency + Energy)

```python
USE_ENERGY = True

ENERGY_CONFIG = {
    'use_energy': True,
    'energy_weight': 0.5,
    'latency_weight': 0.5,
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

---

## 9. Testing Strategy

### 9.1 Backward Compatibility Tests

1. **Reward Comparison Test**:
   - Run same task with `use_energy=False` and `use_energy=True` (with `energy_weight=0`)
   - Verify rewards are identical

2. **Tensor Shape Test**:
   - Verify all tensor shapes match when `use_energy=False`
   - Verify no new tensors are created when disabled

3. **Training Dynamics Test**:
   - Train for 100 iterations with `use_energy=False`
   - Compare loss curves, rewards, latencies with original

### 9.2 Energy Functionality Tests

1. **Energy Computation Test**:
   - Verify energy values are non-zero when `use_energy=True`
   - Verify energy computation matches expected formulas

2. **Combined Reward Test**:
   - Verify combined rewards are weighted correctly
   - Verify energy normalization works

3. **End-to-End Test**:
   - Train with `use_energy=True`
   - Verify model learns to optimize both latency and energy

---

## 10. Summary

This design provides a clean, backward-compatible extension for energy optimization:

1. **Single Flag Control**: `use_energy` boolean controls all energy logic
2. **Zero Overhead**: When disabled, no energy computation occurs
3. **Additive Design**: Energy added alongside latency, not replacing it
4. **Configurable Weighting**: Users can tune latency vs. energy trade-off
5. **Full Compatibility**: Original behavior preserved when flag is False

The implementation follows a minimal-invasive approach, ensuring the existing system remains unchanged while providing a powerful extension for energy-aware optimization.


