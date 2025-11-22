# Training Information Reporting Status
## Excel and Plot Generation for Energy and V2V Metrics

**Date**: Analysis  
**Status**: ✅ **PARTIALLY IMPLEMENTED** - Energy metrics are saved, but V2V-specific breakdowns are not tracked separately

---

## Current Implementation Status

### ✅ **What IS Currently Saved:**

#### 1. **Excel Files (.xlsx)**
- **Location**: `training_reports/YYYYMMDD_HHMMSS/data.xlsx`
- **Content**: All training metrics in Excel format
- **Metrics Included**:
  - `average_reward` - Average reward per iteration
  - `average_loss` - Average loss per iteration
  - `average_latency` - Average latency per iteration
  - `policy_losses` - Policy loss per iteration
  - `value_losses` - Value function loss per iteration
  - `greedy_latencies` - Greedy solution latencies
  - `average_energy` - **Average energy consumption per iteration** ✅
  - `greedy_energy` - **Greedy solution energy consumption** ✅ (in `meta_evaluator.py`)

#### 2. **Plots (.png)**
- **Location**: `training_reports/YYYYMMDD_HHMMSS/*.png`
- **Generated Plots**:
  - `average_reward.png` - Reward over iterations
  - `average_loss.png` - Loss over iterations
  - `average_latency.png` - Latency over iterations
  - `policy_losses.png` - Policy loss over iterations
  - `value_losses.png` - Value loss over iterations
  - `greedy_latencies.png` - Greedy latency over iterations
  - `average_energy.png` - **Energy consumption over iterations** ✅
  - `greedy_energy.png` - **Greedy energy over iterations** ✅ (in `meta_evaluator.py`)

#### 3. **CSV Files (.csv)**
- **Location**: `training_reports/YYYYMMDD_HHMMSS/data.csv`
- **Content**: Same as Excel, in CSV format

#### 4. **JSON Files (.json)**
- **Location**: `training_reports/YYYYMMDD_HHMMSS/data.json`
- **Content**: All metrics in JSON format

#### 5. **Summary Statistics (.txt)**
- **Location**: `training_reports/YYYYMMDD_HHMMSS/summary.txt`
- **Content**: Mean, Min, Max, Std Dev for all metrics

---

## Current Energy Reporting

### In `meta_trainer.py`:
```python
additional_metrics = {
    'policy_losses': policy_losses_all,
    'value_losses': value_losses_all,
    'greedy_latencies': greedy_latencies_all,
    'average_energy': avg_energies  # ✅ Total energy (includes Local + MEC + V2V)
}
```

### In `meta_evaluator.py`:
```python
additional_metrics = {
    'policy_losses': avg_pg_loss,
    'value_losses': avg_vf_loss,
    'greedy_latencies': avg_greedy_latencies,
    'average_energy': avg_energies,      # ✅ Policy energy
    'greedy_energy': avg_greedy_energies  # ✅ Greedy energy
}
```

---

## ❌ **What is NOT Currently Tracked (V2V-Specific Breakdowns):**

### Missing Metrics:
1. **V2V Energy Breakdown**:
   - `v2v_transmission_energy` - Energy for V2V transmission only
   - `v2v_computation_energy` - Energy for V2V computation on helper vehicle
   - `total_v2v_energy` - Total V2V energy (transmission + computation)

2. **Action-Specific Energy**:
   - `local_energy` - Energy for local execution only
   - `mec_energy` - Energy for MEC offloading only
   - `v2v_energy` - Energy for V2V offloading only

3. **Action Distribution**:
   - `action_distribution` - Percentage of tasks offloaded to Local/MEC/V2V
   - `v2v_usage_rate` - Percentage of tasks using V2V

4. **V2V-Specific Metrics**:
   - `v2v_latency` - Average latency for V2V tasks
   - `v2v_vs_mec_energy_ratio` - Comparison of V2V vs MEC energy efficiency

---

## What Gets Saved After Training

### ✅ **Automatically Generated:**

1. **Excel File** (`data.xlsx`):
   - All metrics in columns
   - Formatted with headers
   - Auto-adjusted column widths
   - Can be opened in Excel/LibreOffice

2. **Plots** (`.png` files):
   - One plot per metric
   - Includes trend lines
   - Color-coded (energy = red, latency = teal)
   - High resolution (150 DPI)

3. **CSV File** (`data.csv`):
   - Same data as Excel
   - Can be imported into any tool

4. **JSON File** (`data.json`):
   - Machine-readable format
   - Includes all metrics

5. **Summary Statistics** (`summary.txt`):
   - Mean, Min, Max, Std Dev for each metric

---

## Example Report Structure

```
training_reports/
└── 20241201_143022/
    ├── data.xlsx          ✅ Excel file with all metrics
    ├── data.csv           ✅ CSV file
    ├── data.json          ✅ JSON file
    ├── summary.txt        ✅ Summary statistics
    ├── average_reward.png ✅ Plot
    ├── average_loss.png   ✅ Plot
    ├── average_latency.png ✅ Plot
    ├── average_energy.png ✅ Plot (Energy over iterations)
    ├── greedy_energy.png  ✅ Plot (Greedy energy, if in evaluator)
    └── ... (other plots)
```

---

## Current Energy Tracking

### Energy Calculation Includes:
- ✅ **Local Energy**: `T_l * rho * (f_l ^ zeta)`
- ✅ **MEC Energy**: `T_ul * ptx + T_dl * prx`
- ✅ **V2V Energy**: `(T_v2v_ul * ptx_v2v + T_v2v_dl * prx_v2v) + (T_v2v_exec * rho_v2v * (f_v2v ^ zeta))`

### Energy Reporting:
- ✅ Total energy per iteration is tracked
- ✅ Energy is included in Excel export
- ✅ Energy plots are generated
- ❌ **V2V energy is NOT separated from total energy**
- ❌ **No breakdown by action type (Local/MEC/V2V)**

---

## Recommendations for Enhanced V2V Reporting

### Option 1: Add V2V-Specific Breakdowns

**Track separately**:
- `local_energy` - Energy for local tasks
- `mec_energy` - Energy for MEC tasks
- `v2v_energy` - Energy for V2V tasks
- `v2v_transmission_energy` - V2V transmission only
- `v2v_computation_energy` - V2V computation only

**Benefits**:
- See energy breakdown by action type
- Compare V2V vs MEC energy efficiency
- Analyze V2V transmission vs computation costs

### Option 2: Add Action Distribution Metrics

**Track**:
- `action_distribution` - Count/percentage of each action
- `v2v_usage_rate` - Percentage of tasks using V2V
- `mec_usage_rate` - Percentage of tasks using MEC
- `local_usage_rate` - Percentage of tasks executed locally

**Benefits**:
- Understand policy behavior
- See how often V2V is chosen
- Compare with greedy solution action choices

### Option 3: Add Comparative Metrics

**Track**:
- `v2v_vs_mec_energy_ratio` - V2V energy / MEC energy
- `v2v_vs_local_energy_ratio` - V2V energy / Local energy
- `energy_savings_vs_local` - Energy saved vs all-local
- `energy_savings_vs_greedy` - Energy saved vs greedy

**Benefits**:
- Quantify V2V energy efficiency
- Compare policy performance
- Measure energy optimization success

---

## Summary

### ✅ **Currently Working:**
- Excel export with energy metrics ✅
- Plot generation for energy ✅
- CSV/JSON export ✅
- Summary statistics ✅
- Energy tracking in both trainer and evaluator ✅

### ❌ **Missing:**
- V2V-specific energy breakdowns ❌
- Action-specific energy tracking ❌
- V2V usage statistics ❌
- Comparative metrics (V2V vs MEC vs Local) ❌

### 📊 **What Gets Saved:**
- **Excel**: ✅ Yes (all metrics including total energy)
- **Plots**: ✅ Yes (energy plots generated)
- **V2V Breakdown**: ❌ No (only total energy tracked)

---

## Conclusion

**Current Status**: The V2V project **DOES save energy information** to Excel and plots, but it saves **total energy** only. V2V-specific breakdowns (transmission vs computation, V2V vs MEC vs Local) are **NOT currently tracked separately**.

**To get V2V-specific metrics**, you would need to:
1. Modify energy calculation to track action-specific energy
2. Add V2V breakdown metrics to `additional_metrics`
3. The automated reporting system will automatically include them in Excel/plots

The infrastructure is in place - you just need to track the additional metrics!

