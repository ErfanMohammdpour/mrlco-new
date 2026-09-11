# Experimental Analysis Report

## Overview
This report summarizes the quantitative analysis of the final experimental data files for the MARGO paper. The analysis covers the convergence and stability metrics over 100 training iterations, as well as a detailed node-level policy behavior analysis at iteration 100.

## 1. Performance and Stability Summary
The following metrics were computed from the iteration logs (`MARGO1.csv`, `MARGO2.csv`, `MRLCO1.csv`, `MRLCO2.csv`). The "Last-20" metrics represent the average over iterations 81-100, providing a reliable measure of converged behavior.

| Task / Method | Final Lat. | Last-20 Lat. | Lat. Impr. | Final Energy | Last-20 Energy | Energy Impr. | Lat. Std. | Energy Std. | Comp. Impr. |
|---|---|---|---|---|---|---|---|---|---|
| T1 / MARGO | 574.90 | 575.40 | 17.71% | 659.30 | 659.01 | 28.34% | 0.40 | 0.38 | 23.03% |
| T1 / MRLCO-style | 616.31 | 617.16 | 22.41% | 626.80 | 629.80 | 27.96% | 1.83 | 3.77 | 25.18% |
| T2 / MARGO | 603.89 | 605.16 | 14.32% | 682.97 | 682.69 | 27.04% | 0.94 | 0.43 | 20.68% |
| T2 / MRLCO-style | 713.49 | 652.89 | 19.29% | 572.84 | 646.13 | 27.45% | 29.13 | 33.40 | 23.37% |

*Note: Improvements are computed relative to the greedy baseline recorded within each corresponding run. Since greedy references differ between MARGO and MRLCO-style experiments, direct absolute method-to-method comparisons should be interpreted together with convergence stability metrics.*

## 2. Node-level Policy Behavior
The node-level decisions were analyzed from `iteration_100_detailed_MARGO_1.xlsx` and `iteration_100_detailed_MARGO_2.xlsx`. Each file contains 2,000 decisions (100 DAGs × 20 nodes).

| Task / Action | Decisions | Share | Avg Latency | Avg Energy | Avg Depth | Avg Pred. | Avg Succ. |
|---|---|---|---|---|---|---|---|
| T1 Local | 698 | 34.90% | 67.68 | 67.68 | 0.76 | 2.46 | 1.71 |
| T1 MEC | 1167 | 58.35% | 127.94 | 10.38 | 0.51 | 1.67 | 2.41 |
| T1 V2V | 135 | 6.75% | 195.08 | 49.57 | 1.04 | 3.36 | 0.77 |
| T2 Local | 687 | 34.35% | 70.74 | 70.74 | 0.78 | 2.04 | 1.38 |
| T2 MEC | 1166 | 58.30% | 128.03 | 10.42 | 0.54 | 1.34 | 1.96 |
| T2 V2V | 147 | 7.35% | 202.34 | 51.71 | 1.01 | 2.30 | 0.52 |

## 3. Structure-conditioned Action Distribution
The action distribution was further broken down by structural properties of the tasks (depth and number of successors).

| Task | Structural Group | Nodes | Local | MEC | V2V |
|---|---|---|---|---|---|
| T1 | Depth 0 | 955 | 28.06% | 70.26% | 1.68% |
| T1 | Depth 1 | 831 | 40.07% | 48.01% | 11.91% |
| T1 | Depth >=2 | 214 | 45.33% | 45.33% | 9.35% |
| T1 | Exit nodes | 692 | 40.03% | 48.12% | 11.85% |
| T1 | Successors 1-2 | 514 | 39.49% | 53.70% | 6.81% |
| T1 | Successors >=3 | 794 | 27.46% | 70.28% | 2.27% |
| T2 | Depth 0 | 918 | 26.80% | 70.81% | 2.40% |
| T2 | Depth 1 | 865 | 40.46% | 47.63% | 11.91% |
| T2 | Depth >=2 | 217 | 41.94% | 47.93% | 10.14% |
| T2 | Exit nodes | 830 | 39.04% | 48.07% | 12.89% |
| T2 | Successors 1-2 | 576 | 35.76% | 59.03% | 5.21% |
| T2 | Successors >=3 | 594 | 26.43% | 71.89% | 1.68% |

## 4. Graph-level Summary
| Task | Avg Makespan | Avg Energy | Avg Local Actions | Avg MEC Actions | Avg V2V Actions | V2V Range |
|---|---|---|---|---|---|---|
| T1 | 1044.05 | 660.41 | 6.98 | 11.67 | 1.35 | 0-2 |
| T2 | 1053.72 | 683.48 | 6.87 | 11.66 | 1.47 | 1-2 |

## Interpretation
The analysis confirms that MARGO successfully learns a graph-aware scheduling policy. It reduces latency and energy relative to its own greedy baseline and exhibits very low standard deviations in the last 20 iterations, indicating stable convergence. The node-level analysis reveals that MARGO strategically assigns tasks to execution endpoints based on their structural properties, such as assigning high-successor tasks to MEC and reserving V2V for exit or low-successor tasks to avoid half-duplex bottlenecks.
