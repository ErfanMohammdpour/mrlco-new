# MARGO Paper Figures Guide

## Overview of Experimental Results

This document provides a comprehensive guide to all experimental figures in the `results/` directory, explaining how to use them in the paper.

---

## 1. MRLCO Comparison Figures (`results/mrlco-compare/`)

These figures are **CRITICAL** for the paper as they show head-to-head comparison between MARGO and MRLCO.

### Task Distribution 1 (`mrlco-compare/1/`)

| File | Description | Key Observations |
|------|-------------|------------------|
| `task1-latency-with-greedy.jpg` | 3-way latency comparison (MARGO vs MRLCO vs Greedy) | MARGO (orange): ~634ms, MRLCO (blue): ~638ms, Greedy (green): ~795ms |
| `task1-energy-with-greedy.jpg` | 3-way energy comparison (MARGO vs MRLCO vs Greedy) | MARGO (orange): ~595J, MRLCO (blue): ~620J, Greedy (green): ~875J |
| `task1-latency.jpg` | 2-way latency (MARGO vs MRLCO) | Shows ~0.6% improvement |
| `task1-energy.jpg` | 2-way energy (MARGO vs MRLCO) | Shows ~4% improvement |

**Numerical Results (Task 1):**
- **MARGO Latency**: 634 ms (final, iteration 20)
- **MRLCO Latency**: 638 ms (final, iteration 20)
- **Improvement**: 0.63% (4 ms reduction)
- **MARGO Energy**: 595 J
- **MRLCO Energy**: 620 J
- **Improvement**: 4.03% (25 J reduction)
- **Greedy Latency**: 795 ms
- **Greedy Energy**: 875 J

### Task Distribution 2 (`mrlco-compare/2/`)

| File | Description | Key Observations |
|------|-------------|------------------|
| `task2-latency-with-greedy.jpg` | 3-way latency comparison | MARGO: ~661ms, MRLCO: ~668ms, Greedy: ~810ms |
| `task2-energy-with-greedy.jpg` | 3-way energy comparison | MARGO: ~633J, MRLCO: ~659J, Greedy: ~890J |
| `task2-latency.jpg` | 2-way latency comparison | Shows consistent improvement |
| `task2-energy.jpg` | 2-way energy comparison | Shows ~3.9% improvement |

**Numerical Results (Task 2):**
- **MARGO Latency**: 661 ms
- **MRLCO Latency**: 668 ms
- **Improvement**: 1.05% (7 ms reduction)
- **MARGO Energy**: 633 J
- **MRLCO Energy**: 659 J
- **Improvement**: 3.94% (26 J reduction)
- **Greedy Latency**: 810 ms
- **Greedy Energy**: 890 J

---

## 2. MARGO Training Dynamics (`results/` root)

These figures show MARGO's training with V2V and energy optimization enabled.

### Task 1 Training Metrics

| File | Description | Key Observations |
|------|-------------|------------------|
| `task1-average-latency.jpg` | Latency over 100 iterations | Decreases from ~590ms to ~575ms (-2.5%) |
| `task1-average-energy.jpg` | Energy over 100 iterations | Fluctuates around ~655J (exploration of V2V vs MEC) |
| `task1-average-reward.jpg` | Reward over 100 iterations | Increases from -1.622 to -1.585 (+2.3%) |
| `task1-average-loss.jpg` | Total loss convergence | Converges toward zero |
| `task1-policy-loses.jpg` | Policy loss convergence | From -0.0036 to -0.0008 |
| `task1-value-loses.png` | Value loss (if available) | Shows baseline estimation quality |
| `task1-greedy-latency.jpg` | Greedy baseline latency | Constant at ~699.3 ms |
| `task1-greedy-energy.jpg` | Greedy baseline energy | Constant at ~919.6 J |

**Improvement Over Greedy (Task 1):**
- Latency: 574.9ms vs 699.3ms = **17.8% improvement**
- Energy: 657J vs 919.6J = **28.5% improvement**

### Task 2 Training Metrics

| File | Description | Key Observations |
|------|-------------|------------------|
| `task2-average-latency.jpg` | Latency over 100 iterations | Dramatic decrease from ~682ms to ~604ms |
| `task2-average-energy.jpg` | Energy over 100 iterations | Around ~682J with exploration |
| `task2-average-reward.jpg` | Reward over 100 iterations | Improves from -1.80 to -1.61 (+10.6%) |
| `task2-policy-losses.jpg` | Policy loss | From -0.005 to -0.001 |
| `task2-greedy-latency.jpg` | Greedy baseline | Constant at ~706.32 ms |

**Improvement Over Greedy (Task 2):**
- Latency: 604ms vs 706.3ms = **14.5% improvement**
- Energy: ~683J vs ~920J (estimated) = **~26% improvement**

---

## 3. Recommended Figures for Paper

### Figure 1: Three-Way Comparison (MARGO vs MRLCO vs Greedy)
**Files to use:** `mrlco-compare/1/task1-latency-with-greedy.jpg`, `mrlco-compare/1/task1-energy-with-greedy.jpg`
**Purpose:** Shows the dramatic advantage of learning-based methods over Greedy, and MARGO's superiority over MRLCO.
**Placement:** Main Results section

### Figure 2: Training Dynamics
**Files to use:** `task1-average-latency.jpg`, `task1-average-energy.jpg`, `task1-average-reward.jpg`, `task1-average-loss.jpg`
**Purpose:** Shows convergence behavior and joint optimization.
**Placement:** Training Analysis section (as 2x2 subfigure)

### Figure 3: Task 2 Validation
**Files to use:** `mrlco-compare/2/task2-latency-with-greedy.jpg`, `mrlco-compare/2/task2-energy-with-greedy.jpg`
**Purpose:** Validates generalization across task distributions.
**Placement:** Results section (as 1x2 subfigure)

### Figure 4: Convergence Analysis
**Files to use:** `task1-policy-loses.jpg`, `task2-policy-losses.jpg`
**Purpose:** Shows stable PPO convergence.
**Placement:** Appendix or supplementary

---

## 4. Summary Statistics for Tables

### Table: Main Comparison (For Paper)

| Method | Task 1 Latency | Task 1 Energy | Task 2 Latency | Task 2 Energy |
|--------|----------------|---------------|----------------|---------------|
| Greedy | 795 ms | 875 J | 810 ms | 890 J |
| MRLCO | 638 ms | 620 J | 668 ms | 659 J |
| **MARGO** | **634 ms** | **595 J** | **661 ms** | **633 J** |
| MARGO vs MRLCO | -0.63% | -4.03% | -1.05% | -3.94% |
| MARGO vs Greedy | -20.3% | -32.0% | -18.4% | -28.9% |

### Table: Training Metrics Summary

| Metric | Initial (Iter 0) | Final (Iter 100) | Change |
|--------|------------------|------------------|--------|
| Latency (Task 1) | 589.7 ms | 574.9 ms | -2.5% |
| Energy (Task 1) | ~655 J | ~657 J | Stable (exploration) |
| Reward (Task 1) | -1.622 | -1.585 | +2.3% |
| Latency (Task 2) | 682 ms | 604 ms | -11.4% |
| Reward (Task 2) | -1.80 | -1.61 | +10.6% |

---

## 5. Key Insights from Figures

### Why Energy Improvement > Latency Improvement?
The ~4% energy improvement vs ~1% latency improvement is explained by:
1. **V2V Action**: V2V transmission power (0.06W) is 40% lower than MEC (0.1W)
2. **V2V Computation**: Energy coefficient reduced by 30% for V2V
3. **Trade-off Exploration**: Energy fluctuation shows active exploration of V2V vs MEC

### Why Latency Improvement is Modest but Consistent?
1. Both MARGO and MRLCO use similar Reptile meta-learning
2. The main latency gain comes from better DAG structure understanding via GNN
3. GNN captures critical paths that LSTM misses

### Why Results are Consistent Across Tasks?
1. Meta-learning ensures generalization
2. Graph2Seq encoder learns transferable task dependency patterns
3. Triple readout captures diverse graph properties

---

## 6. LaTeX Figure Inclusion Code

```latex
% Example for main comparison figure
\begin{figure*}[t]
\centering
\begin{subfigure}{0.48\textwidth}
    \includegraphics[width=\linewidth]{../results/mrlco-compare/1/task1-latency-with-greedy.jpg}
    \caption{Task 1: Latency Comparison}
\end{subfigure}
\hfill
\begin{subfigure}{0.48\textwidth}
    \includegraphics[width=\linewidth]{../results/mrlco-compare/1/task1-energy-with-greedy.jpg}
    \caption{Task 1: Energy Comparison}
\end{subfigure}
\caption{Three-way comparison between MARGO, MRLCO, and Greedy baseline.}
\end{figure*}
```

---

## 7. Notes for Paper Writing

1. **Always cite MRLCO when comparing**: "Compared to the state-of-the-art MRLCO baseline~\cite{wang2020fast}..."

2. **Explain energy fluctuation**: "The energy fluctuation in training reflects the model's exploration of the V2V action space, which provides significant energy savings at a modest latency cost."

3. **Highlight V2V contribution**: "The 4% energy improvement is directly attributable to the V2V cooperative computing capability, which MRLCO's binary action space cannot exploit."

4. **Be precise about metrics**: Always specify whether you're reporting final values (iteration 20 for MRLCO comparison) or training progression (100 iterations for MARGO dynamics).

