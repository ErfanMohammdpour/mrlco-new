# Greedy Solution Energy Update

## Summary

Energy computation has been added to the `greedy_solution()` method. When `use_energy=True`, the greedy solution now computes and returns energy consumption alongside latency.

---

## Changes Made

### 1. **`env/mec_offloaing_envs/offloading_env.py`** - `greedy_solution()` Method

**Added**:
- Energy tracking variables (`total_energy`, `T_l`, `T_ul`, `T_dl`)
- Energy computation for local execution: `compute_local_energy(T_l[i])`
- Energy computation for offloading: `compute_transmission_energy(T_ul[i], T_dl[i])`
- Energy batch tracking (`energy_batchs`)
- Conditional return: Returns energy when `use_energy=True`

**Return Format**:
- **When `use_energy=False`**: `(result_plan, finish_time_batchs)` - Original format
- **When `use_energy=True`**: `(result_plan, finish_time_batchs, energy_batchs)` - Includes energy

### 2. **`greedy_solution_for_current_task()` Method**

**Updated**:
- Handles conditional return from `greedy_solution()`
- Returns energy when enabled
- Maintains backward compatibility

**Return Format**:
- **When `use_energy=False`**: `(plan, finish_times)` - Original format
- **When `use_energy=True`**: `(plan, finish_times, energy)` - Includes energy

### 3. **`meta_trainer.py`** - Greedy Solution Call

**Updated**:
- Handles conditional return from `greedy_solution()`
- Prints energy statistics when enabled
- Flattens nested lists for proper averaging

**Output**:
```python
if use_energy:
    print("avg greedy solution latency: ", ...)
    print("avg greedy solution energy: ", ...)
else:
    print("avg greedy solution: ", ...)
```

### 4. **`meta_evaluator.py`** - Greedy Solution Call

**Updated**:
- Handles conditional return from `greedy_solution()`
- Prints energy statistics when enabled
- Flattens nested lists for proper averaging

---

## Energy Computation in Greedy Solution

### Local Execution Energy
```python
if action == 0:  # Local execution
    energy = compute_local_energy(T_l[i])
    total_energy += energy
```

### Offloading Energy
```python
if action == 1:  # Offloading
    energy = compute_transmission_energy(T_ul[i], T_dl[i])
    total_energy += energy
```

### Energy Tracking
- Energy is accumulated per task graph
- Total energy per task graph is stored in `energy_plan`
- Energy batch structure matches finish time batch structure

---

## Data Structure

### Return Structure

**When `use_energy=False`** (Original):
```python
result_plan = [
    [plan_batch_1, plan_batch_2, ...],  # Plans for batch 1
    [plan_batch_1, plan_batch_2, ...],  # Plans for batch 2
    ...
]

finish_time_batchs = [
    [time_1, time_2, ...],  # Finish times for batch 1
    [time_1, time_2, ...],  # Finish times for batch 2
    ...
]
```

**When `use_energy=True`** (With Energy):
```python
result_plan = [
    [plan_batch_1, plan_batch_2, ...],  # Plans for batch 1
    ...
]

finish_time_batchs = [
    [time_1, time_2, ...],  # Finish times for batch 1
    ...
]

energy_batchs = [
    [energy_1, energy_2, ...],  # Energy for batch 1
    ...
]
```

---

## Usage Examples

### Example 1: Using Greedy Solution with Energy

```python
# Enable energy
USE_ENERGY = True
resource_cluster = Resources(..., use_energy=USE_ENERGY, ...)
env = OffloadingEnvironment(resource_cluster=resource_cluster, ...)

# Get greedy solution
greedy_result = env.greedy_solution()
if env.resource_cluster.use_energy:
    action, finish_time, energy = greedy_result
    print(f"Average latency: {np.mean([item for sublist in finish_time for item in sublist])}")
    print(f"Average energy: {np.mean([item for sublist in energy for item in sublist])}")
else:
    action, finish_time = greedy_result
    print(f"Average latency: {np.mean([item for sublist in finish_time for item in sublist])}")
```

### Example 2: Using Greedy Solution for Current Task

```python
env.set_task(0)
result = env.greedy_solution_for_current_task()

if env.resource_cluster.use_energy:
    plan, finish_times, energy = result
    print(f"Energy consumption: {energy}")
else:
    plan, finish_times = result
    print(f"Finish times: {finish_times}")
```

---

## Backward Compatibility

✅ **100% Backward Compatible**:
- When `use_energy=False`, returns original format
- No energy computation occurs when disabled
- All existing code works without modification
- Zero performance overhead when disabled

---

## Console Output

### When Energy Enabled

```
avg greedy solution latency: X.XXXXXX
avg greedy solution energy: X.XXXXXX
```

### When Energy Disabled

```
avg greedy solution: X.XXXXXX
```

---

## Verification

To verify energy computation in greedy solution:

1. **Set `USE_ENERGY = True`** in training script
2. **Run training** - check console output for energy values
3. **Verify energy values** are non-zero and reasonable
4. **Compare** with energy values from training iterations

---

## Summary

✅ Energy computation added to `greedy_solution()`
✅ Energy returned when `use_energy=True`
✅ Backward compatible when `use_energy=False`
✅ All call sites updated to handle energy
✅ Energy statistics printed when enabled
✅ No breaking changes to existing code

The greedy solution now provides energy consumption metrics when energy optimization is enabled, allowing comparison between greedy baseline and learned policy energy consumption.

