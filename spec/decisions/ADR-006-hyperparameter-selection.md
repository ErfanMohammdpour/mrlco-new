# ADR-006: Validation-Only Hyperparameter Selection

Status: Accepted  
Decision date: 2026-09-03

## Decision

Do not copy legacy `inner_lr` / `outer_lr` / `num_inner_grad_steps` into the v0.1 protocol as if they were already correct.

Legacy defects that MUST NOT be frozen as semantics:

- `outer_lr=5e-4` drove sequential Adam on the core, not Reptile interpolation `theta + eps * mean_delta`
- `num_inner_grad_steps=1` counted outer loops over minibatches, not one optimizer apply
- `batch_size=10` mixed trajectory, graph, and decoder-position units

v0.1 therefore freezes:

1. Optimizer *form* and clip/entropy/GAE constants
2. Unit names (`trajectories`, `distributions`, `optimizer_steps`)
3. A closed candidate grid and a validation-only selection rule

Final numeric `inner_learning_rate`, `outer_step_size`, and `k_steps` are recorded after selection on `validation_support` / `validation_query` only.

## Selection protocol (normative)

```yaml
data_role: validation_support_and_query_only
meta_test_use_for_selection: false
selection_metric: validation_query_composite_objective   # J_report; MUST minimize
metric_direction: minimize
tie_tolerance_abs: 1.0e-6

seeds: [0, 1, 2]
n_seeds: 3
weight_initialization:
  shared_across_candidates: true
  init_seed: 0
rollout_seed_policy: per_run_seed_from_seeds_list

aggregation:
  # equal weight per validation_query graph (5 dists × 80 = 400 graphs)
  within_seed: mean_over_validation_query_graphs(J_report)
  across_seeds: mean_over_seeds(within_seed)
  # NOT used for v0.1: mean_over_distributions(mean_over_graphs)

checkpoint_selection:
  within_run: best validation_query_composite_objective over validation_interval checks
  fixed_budget_outer_iterations: 3500
  early_stopping: none_in_v0.1_fixed_budget

meta_train_distribution_sampling:
  within_meta_batch: without_replacement   # sample 5 distinct meta_train dists
  across_outer_iterations: with_replacement_reshuffle

candidate_grid:
  inner_learning_rate: [1.0e-4, 3.0e-4, 5.0e-4, 1.0e-3]
  outer_step_size: [0.05, 0.1, 0.25, 0.5]
  k_steps: [1, 5, 10]

tie_break_order:
  1. lower selection_metric (after seed aggregation), within tie_tolerance_abs
  2. lower compute_budget = k_steps * support_graphs_per_meta_task * trajectories_per_support_graph
  3. lower inner_learning_rate
  4. lower outer_step_size
  5. lower k_steps
```

Evidence artifact after selection MUST record the winning triple, seed-wise metrics, and the exact checkpoint outer-iteration index.

## Why

Hyperparameters chosen on meta-test would invalidate the frozen split.
Blind legacy copy would freeze a known-wrong outer update.
Incomplete tie-break / aggregation rules would allow two labs to pick different winners from the same grid.
