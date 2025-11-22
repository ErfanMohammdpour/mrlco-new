# V2V Energy Impact on Reward System
## How V2V Energy Affects Reward Calculation

**Date**: Analysis  
**Status**: ✅ **REWARD SYSTEM CORRECTLY HANDLES V2V ENERGY**

---

## Overview

The reward system combines **latency** and **energy** into a single reward signal. With V2V energy implementation, the system now correctly accounts for:
1. **V2V transmission energy** (separate parameters from MEC)
2. **V2V computation energy** (on helper vehicle, less than local)
3. **Energy bounds** that consider V2V as an offloading option

---

## Reward Calculation Flow

### Step 1: Calculate Latency and Energy

```python
# In get_scheduling_cost_step_by_step()
cost, task_finish_time, energy = self.get_scheduling_cost_step_by_step(plan, task_graph)
```

**Energy per task**:
- **Action 0 (Local)**: `compute_local_energy(T_l)` → Only computation energy
- **Action 1 (MEC)**: `compute_transmission_energy(T_ul, T_dl)` → Only transmission energy
- **Action 2 (V2V)**: `compute_v2v_transmission_energy(T_v2v_ul, T_v2v_dl) + compute_v2v_energy(T_v2v_exec)` → Transmission + Computation

**Result**: `energy` is a list of energy values, one per task in the plan.

---

### Step 2: Calculate Energy Bounds

```python
# In get_reward_batch_step_by_step()
max_energy, min_energy = self._compute_energy_bounds(
    task_graph, max_running_time, min_running_time)
```

**Max Energy** (worst case):
- All tasks executed locally
- Formula: `sum([compute_local_energy(...) for each task])`

**Min Energy** (best case):
- All tasks offloaded to the **most energy-efficient option** (MEC or V2V)
- Formula: `sum([min(mec_energy, v2v_energy) for each task])`
- Where:
  - `mec_energy` = MEC transmission energy only
  - `v2v_energy` = V2V transmission energy + V2V computation energy

**Key Impact**: V2V energy is now considered in the min energy calculation, which affects normalization.

---

### Step 3: Normalize Energy

```python
# Sum energy to get total energy consumption
total_energy = np.sum(energy) if isinstance(energy, (list, np.ndarray)) else energy

# Normalize energy (handle edge case where max == min)
if max_energy > min_energy:
    total_energy_score = self.score_func(total_energy, max_energy, min_energy)
    # Distribute energy score proportionally across steps
    if len(energy) > 0 and total_energy > 0:
        energy_proportions = np.array(energy) / total_energy
        energy_score = total_energy_score * energy_proportions
    elif len(energy) > 0:
        # If total_energy is 0, distribute score equally
        energy_score = np.full_like(energy, total_energy_score / len(energy), dtype=float)
    else:
        energy_score = np.array([total_energy_score])
else:
    # If no variation, set to zero
    energy_score = np.zeros_like(energy)
```

**Normalization Formula**:
```python
total_energy_score = -(total_energy - min_energy) / (max_energy - min_energy)
```

**Impact of V2V**:
- If V2V is more energy-efficient than MEC for some tasks, `min_energy` decreases
- This makes the normalization range larger (max_energy - min_energy increases)
- Lower total energy → Higher energy score → Better reward

---

### Step 4: Normalize Latency

```python
# Normalize latency - cost is incremental latencies, normalize element-wise
latency_score = self.score_func(cost, max_running_time, min_running_time)
```

**Formula**:
```python
latency_score = -(cost - min_time) / (max_time - min_time)
```

**Note**: Latency normalization is independent of energy.

---

### Step 5: Combine Latency and Energy Rewards

```python
# Combine rewards
combined_score = (self.resource_cluster.latency_weight * latency_score + 
                self.resource_cluster.energy_weight * energy_score)
```

**Combined Reward Formula**:
```python
reward = latency_weight * latency_score + energy_weight * energy_score
```

**Default Weights**:
- `latency_weight`: 0.5
- `energy_weight`: 0.5

**Impact of V2V Energy**:
- V2V actions now contribute **both latency and energy** to the reward
- Energy component includes **transmission + computation**
- Lower V2V energy → Higher energy score → Better combined reward

---

## V2V Energy Impact on Reward

### 1. **Energy Bounds Calculation**

**Before V2V Energy**:
- Min energy: Only MEC transmission energy considered
- Max energy: All local execution (unchanged)

**After V2V Energy**:
- Min energy: `min(mec_energy, v2v_energy)` for each task
- Max energy: All local execution (unchanged)

**Impact**:
- If V2V is more energy-efficient: `min_energy` decreases → Larger normalization range → More sensitive energy rewards
- If V2V is less energy-efficient: `min_energy` stays same (MEC) → No change

---

### 2. **Energy Normalization**

**Formula**:
```python
energy_score = -(total_energy - min_energy) / (max_energy - min_energy)
```

**V2V Impact**:
- **Lower V2V energy** → Lower `total_energy` → Higher `energy_score` → Better reward
- **V2V includes computation** → May be higher than MEC-only → Lower `energy_score` → Worse reward (if computation cost is high)

**Example**:
```
Task energy:
- Local: 10.0 Joules
- MEC: 2.0 Joules (transmission only)
- V2V: 3.0 Joules (transmission: 1.5 + computation: 1.5)

If V2V chosen:
- total_energy includes 3.0 Joules
- Compared to min_energy (2.0 from MEC)
- Energy score = -(3.0 - 2.0) / (10.0 - 2.0) = -0.125
```

---

### 3. **Reward Combination**

**Combined Reward**:
```python
reward = 0.5 * latency_score + 0.5 * energy_score
```

**V2V Impact**:
- V2V actions contribute to **both** latency and energy components
- Policy learns to balance:
  - **Latency**: V2V may have different latency than MEC
  - **Energy**: V2V has transmission + computation (may be different from MEC)

**Example Scenario**:
```
Task Options:
- Local: latency=5.0, energy=10.0
- MEC: latency=3.0, energy=2.0
- V2V: latency=2.5, energy=3.0

Normalized scores (assuming max_latency=5.0, min_latency=2.5, max_energy=10.0, min_energy=2.0):
- Local: latency_score=-0.5, energy_score=-1.0 → reward=-0.75
- MEC: latency_score=-0.2, energy_score=0.0 → reward=-0.1
- V2V: latency_score=0.0, energy_score=-0.125 → reward=-0.0625 (BEST!)

V2V wins because better latency compensates for slightly higher energy.
```

---

## Key Points

### 1. **Energy Bounds Consider V2V**
- `min_energy` uses `min(mec_energy, v2v_energy)` per task
- This ensures fair normalization across all offloading options

### 2. **V2V Energy Includes Both Components**
- Transmission energy (uses `ptx_v2v`, `prx_v2v`)
- Computation energy (uses `rho_v2v`, `f_v2v`)
- Total V2V energy = transmission + computation

### 3. **Reward System Balances Latency and Energy**
- Default: 50% latency, 50% energy
- Policy learns optimal trade-off
- V2V may be chosen if it provides better combined score

### 4. **Normalization Range**
- Larger range (max_energy - min_energy) → More sensitive rewards
- V2V inclusion may increase range if V2V is more efficient
- This helps distinguish between energy-efficient and inefficient actions

---

## Reward System Behavior

### When V2V is More Energy-Efficient:
- `min_energy` decreases (uses V2V instead of MEC)
- Normalization range increases
- Energy scores become more sensitive
- V2V actions get better energy scores → Better combined rewards

### When V2V is Less Energy-Efficient:
- `min_energy` stays same (uses MEC)
- Normalization range unchanged
- V2V actions get worse energy scores
- Policy may still choose V2V if latency benefit compensates

### When Energy is Disabled (`use_energy=False`):
- Only latency considered
- Reward = `latency_score` only
- No energy impact

---

## Configuration Impact

### Energy Weights:
```python
'latency_weight': 0.5,  # Weight for latency
'energy_weight': 0.5,   # Weight for energy
```

**Impact**:
- Higher `energy_weight` → More emphasis on energy efficiency
- Lower `energy_weight` → More emphasis on latency
- V2V may be preferred or avoided based on weights

### V2V Energy Parameters:
```python
'ptx_v2v': 0.06,   # V2V transmission power (lower than MEC 0.1)
'prx_v2v': 0.03,   # V2V reception power (lower than MEC 0.05)
'rho_v2v': 0.7,    # V2V computation coefficient (70% of local)
```

**Impact**:
- Lower `ptx_v2v`, `prx_v2v` → Lower V2V transmission energy → Better energy scores
- Lower `rho_v2v` → Lower V2V computation energy → Better energy scores
- These parameters directly affect V2V energy → Reward → Policy learning

---

## Summary

✅ **V2V Energy Correctly Integrated into Reward System**

**Key Impacts**:
1. ✅ Energy bounds consider V2V as offloading option
2. ✅ V2V energy (transmission + computation) included in reward calculation
3. ✅ Normalization accounts for V2V energy range
4. ✅ Combined reward balances latency and energy for all 3 actions
5. ✅ Policy learns optimal trade-off between latency and energy

**Result**: The reward system now correctly guides the policy to learn energy-efficient V2V offloading decisions that balance both latency and energy consumption.

