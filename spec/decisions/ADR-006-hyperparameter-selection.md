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

## Selection rule

```yaml
data_role: validation_support_and_query_only
selection_metric: validation_query_composite_objective
tie_break: lower_compute_budget
meta_test_use_for_selection: false
```

Compute budget for tie-break:

`k_steps * support_graphs_per_meta_task * trajectories_per_support_graph`

## Why

Hyperparameters chosen on meta-test would invalidate the frozen split.
Blind legacy copy would freeze a known-wrong outer update.
