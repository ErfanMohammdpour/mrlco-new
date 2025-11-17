# Energy Logging Enhancements - Plots and Excel Export

## Summary

Energy metrics have been integrated into the automated reporting system, including plots and Excel export functionality.

---

## Changes Made

### 1. **meta_trainer.py** - Energy Tracking

**Added**:
- `avg_energies = []` list to track energy per iteration
- Energy values appended to list during training when `use_energy=True`
- Energy metrics added to `additional_metrics` dictionary for reporting

**Key Code**:
```python
# Track energy metrics
avg_energies = []  # Track energy metrics for reporting

# During training iteration:
if self.env.resource_cluster.use_energy:
    # ... compute energy ...
    avg_energies.append(avg_energy)

# In report generation:
if self.env.resource_cluster.use_energy and len(avg_energies) > 0:
    energy_values = [e for e in avg_energies if e is not None]
    if len(energy_values) > 0:
        additional_metrics['average_energy'] = energy_values
```

### 2. **automated_reporting.py** - Enhanced Reporting

#### A. Excel Export Support

**Added**:
- Optional Excel (.xlsx) export using `openpyxl` library
- Formatted Excel files with styled headers and auto-adjusted column widths
- Falls back to CSV if `openpyxl` not available (CSV can be opened in Excel)

**Features**:
- Styled headers (blue background, white bold text)
- Auto-adjusted column widths
- Numeric formatting for metrics
- Professional Excel workbook format

**Installation** (optional):
```bash
pip install openpyxl
```

#### B. Enhanced Plot Generation

**Added**:
- Color-coded plots for different metric types:
  - **Energy**: Red (#FF6B6B)
  - **Latency**: Teal (#4ECDC4)
  - **Reward**: Light Green (#95E1D3)
  - **Loss**: Pink (#F38181)
- Special formatting for energy plots:
  - Title includes "(Energy Optimization)" label
  - Y-axis labeled as "Energy Consumption"
  - Dark red trend line for energy metrics

**Plot Features**:
- High-resolution plots (150 DPI)
- Trend lines with equations
- Grid for readability
- Professional styling

---

## Output Files Generated

When energy is enabled (`USE_ENERGY = True`), the following files are generated:

### 1. **data.csv**
- CSV file with all metrics including energy
- Can be opened directly in Excel
- Columns: `iteration`, `average_reward`, `average_loss`, `average_latency`, `average_energy`, `policy_losses`, `value_losses`, `greedy_latencies`

### 2. **data.xlsx** (if openpyxl installed)
- Excel workbook with formatted data
- Styled headers and formatted numbers
- Professional appearance

### 3. **data.json**
- JSON format with all metrics
- Useful for programmatic access

### 4. **summary.txt**
- Text summary with statistics:
  - Mean, Min, Max, Std Dev, Count
  - Includes energy statistics when available

### 5. **average_energy.png**
- Plot showing energy consumption over iterations
- Red line with trend analysis
- Title: "Average Energy Over Iterations (Energy Optimization)"

### 6. **Other metric plots**
- `average_reward.png`
- `average_loss.png`
- `average_latency.png`
- `policy_losses.png`
- `value_losses.png`
- `greedy_latencies.png`

---

## Usage

### Enable Energy Logging

1. Set `USE_ENERGY = True` in `meta_trainer.py`:
```python
USE_ENERGY = True

ENERGY_CONFIG = {
    'use_energy': True,
    'energy_weight': 0.5,
    'latency_weight': 0.5,
    # ... other parameters ...
}
```

2. Run training:
```bash
python meta_trainer.py
```

3. After training completes, check the `training_reports/` directory:
```
training_reports/
└── YYYYMMDD_HHMMSS/
    ├── data.csv          # CSV with energy column
    ├── data.xlsx         # Excel file (if openpyxl installed)
    ├── data.json         # JSON format
    ├── summary.txt       # Statistics including energy
    ├── average_energy.png    # Energy plot
    ├── average_reward.png
    ├── average_latency.png
    └── ... (other plots)
```

### Install Excel Support (Optional)

To enable native Excel (.xlsx) export:
```bash
pip install openpyxl
```

If `openpyxl` is not installed, CSV files are still generated and can be opened directly in Excel.

---

## Energy Plot Details

### Plot Characteristics

- **Color**: Red (#FF6B6B) - distinguishes energy from other metrics
- **Title**: "Average Energy Over Iterations (Energy Optimization)"
- **Y-axis**: "Energy Consumption"
- **Trend Line**: Dark red dashed line with equation
- **Resolution**: 150 DPI for high-quality output

### Example Plot Features

- Line plot with markers at regular intervals
- Grid for easy reading
- Trend line showing overall direction
- Legend with metric name and trend equation

---

## Excel File Format

### Header Row
- Blue background (#366092)
- White bold text
- Centered alignment

### Data Rows
- Numeric formatting (6 decimal places)
- Auto-adjusted column widths
- Clean, professional appearance

### Columns Included
- `iteration`: Iteration number
- `average_reward`: Combined reward (latency + energy)
- `average_loss`: Policy loss
- `average_latency`: Task completion latency
- `average_energy`: Energy consumption (when enabled)
- `policy_losses`: Policy loss values
- `value_losses`: Value function loss values
- `greedy_latencies`: Greedy baseline latencies

---

## Backward Compatibility

- **When `USE_ENERGY = False`**: 
  - No energy column in CSV/Excel
  - No energy plot generated
  - All other functionality unchanged

- **When `USE_ENERGY = True`**:
  - Energy automatically included in all exports
  - Energy plot automatically generated
  - No changes needed to existing code

---

## Summary Statistics

The `summary.txt` file includes energy statistics when available:

```
AVERAGE_ENERGY:
  Mean:     X.XXXXXX
  Min:      X.XXXXXX
  Max:      X.XXXXXX
  Std Dev:  X.XXXXXX
  Count:    N
```

---

## Notes

1. **Excel Export**: Requires `openpyxl` library. If not installed, CSV files work perfectly in Excel.

2. **Energy Tracking**: Energy is only tracked and logged when `USE_ENERGY = True`.

3. **Plot Colors**: Energy plots use red color to distinguish from latency (teal) and rewards (green).

4. **File Location**: All reports are saved in `training_reports/` directory with timestamped folders.

5. **Automatic**: Energy logging is automatic when enabled - no additional code changes needed.

---

## Example Output Structure

```
training_reports/
└── 20240101_120000/
    ├── data.csv              # All metrics including energy
    ├── data.xlsx             # Excel format (if openpyxl installed)
    ├── data.json             # JSON format
    ├── summary.txt           # Statistics
    ├── average_energy.png    # ⭐ Energy plot
    ├── average_reward.png
    ├── average_latency.png
    ├── average_loss.png
    ├── policy_losses.png
    ├── value_losses.png
    └── greedy_latencies.png
```

---

## Verification

To verify energy logging is working:

1. Check console output during training:
   ```
   Average energy, X.XXXXXX
   ```

2. After training, check report directory:
   - `average_energy.png` should exist
   - `data.csv` should have `average_energy` column
   - `summary.txt` should include `AVERAGE_ENERGY` section

3. Open `data.csv` or `data.xlsx` in Excel to verify energy column

---

## Conclusion

Energy metrics are now fully integrated into the reporting system:
- ✅ Tracked during training
- ✅ Included in CSV/Excel exports
- ✅ Plotted with special formatting
- ✅ Included in summary statistics
- ✅ Backward compatible (only when enabled)

The system automatically handles energy logging when `USE_ENERGY = True` with no additional configuration needed.


