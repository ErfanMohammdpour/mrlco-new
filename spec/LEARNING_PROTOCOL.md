# Learning Protocol

Status: Frozen form + closed selection grid (`ADR-006`)  
Related ADR: `ADR-006-hyperparameter-selection.md`

## 1. Environment-learning interface

`MARGO-SPEC-v0.1` uses fixed-length autoregressive planning:

- decoder emits one action token per task
- environment consumes whole action plan after decoding
- one environment call returns whole-schedule metrics and shaped rewards

The implementation MUST NOT claim online observation of updated resource state after each decoder token.

## 2. PPO contract

Inner adaptation MUST define all of the following explicitly:

- `support_graphs_per_meta_task`
- `trajectories_per_support_graph`
- `inner_optimizer`
- `inner_learning_rate`
- exact meaning of `K`
- `ppo_batch_size_trajectories`
- batch shuffling policy
- value clipping equation
- entropy coefficient or explicit absence of entropy term

For v0.1 form (frozen now):

- `K` MUST mean count of Adam optimizer apply steps, not a legacy epoch label and not a minibatch-pass counter
- `ppo_inner_optimizer = adam` with `beta1=0.9`, `beta2=0.999`, `epsilon=1.0e-8`
- `policy_clip_epsilon = 0.2`
- `value_clip_epsilon = 0.2`
- `entropy_coefficient = 0.0` (`entropy_claim_allowed: false`)
- clipped value target MUST be:
  `v_old + clip(v_new - v_old, -eps, eps)`
- mini-batches MUST be shuffled before each adaptation epoch
- inner optimizer state MUST be fresh for each meta-task
- `gradient_clip_norm = 0.5`
- `discount_gamma = 0.99`
- `gae_lambda = 0.95`
- `advantage_normalization = true`

Batch unit: `ppo_batch_size_trajectories` is a count of full-plan trajectories, not decoder positions.

## 3. Reptile-style outer update contract

All meta-task adaptations in one meta-batch MUST start from the same `theta0`.

Canonical update:

```text
theta0 = core weights
for task in meta_batch:
    task_weights = copy(theta0)
    task_weights = adapt(task_weights, support_data, exact_K_steps, fresh_optimizer_state)
mean_delta = mean(task_weights - theta0)
core_weights = theta0 + outer_step_size * mean_delta
```

Required properties:

- order invariance up to numerical tolerance
- one outer application per meta-batch
- no sequential Adam on the core in place of interpolation
- no scaling by number of mini-batches unless justified by derivation and tests

`outer_step_size` is the interpolation coefficient `eps` in the formula above.
It is NOT the legacy `outer_optimizer` Adam learning rate.

## 4. Policy synchronization contract

- before first sampling pass, task policies MUST equal core policy
- after every outer update, task policies MUST be synchronized from updated core policy
- no task slot may retain stale optimizer moments from a previous distribution assignment

## 5. Evaluation contract

- support and query sets MUST be disjoint
- adaptation uses support only
- reported metrics use query only
- zero-shot (`K=0`) and selected `K` MUST both be reported
- meta-test MUST NOT be used to choose hyperparameters

## 6. Frozen structural budgets

- `meta_batch_size_distributions = 5` (sampled from `meta_train` only)
- `support_graphs_per_meta_task = 20`
- `trajectories_per_support_graph = 1`
- `ppo_batch_size_trajectories = 20`
- `outer_iterations = 3500` (fixed compute budget, not a convergence claim)
- `validation_interval = 50`
- `checkpoint_selection_metric = validation_query_composite_objective`
- `early_stopping_rule = none_in_v0.1_fixed_budget`

`total_optimizer_steps_per_meta_task = k_steps` after selection.

## 7. Numeric values selected later

These remain in a closed grid. They are not pending-undefined; they are pending-evidence:

- `inner_learning_rate`
- `outer_step_size`
- `k_steps`

Grid and rule: `frozen_experiment.yaml` → `hyperparameter_selection`.

## 8. Legacy reference (NOT v0.1 semantics)

```yaml
legacy_reference:
  inner_learning_rate: 5.0e-4
  legacy_outer_adam_learning_rate: 5.0e-4
  declared_inner_grad_steps: 1
  ppo_batch_size: 10
  meta_batch_size: 10
  training_iterations: 3500
  discount: 0.99
  gae_lambda: 0.95
  policy_clip_epsilon: 0.2
  value_clip_epsilon: 0.2
```

Do not treat `declared_inner_grad_steps: 1` as one optimizer step.
Do not treat `legacy_outer_adam_learning_rate` as Reptile `outer_step_size`.

## 9. Minimum logged fields per run

- seed
- exact train/validation/test split version
- support graph count
- support trajectory count
- query graph count
- inner optimizer step count (`k_steps`)
- outer update count
- entropy coefficient or `0`
- value clip epsilon
- `ppo_batch_size_trajectories`
- `meta_batch_size_distributions`
- selected `inner_learning_rate` and `outer_step_size`
- whether values came from the frozen validation grid

## 10. Prohibited ambiguity

The following phrases MUST NOT appear without exact numeric definition:

- "one-step adaptation"
- "fast adaptation"
- "K=1"
- "few-shot adaptation"
- "stable convergence"

Each MUST map to a concrete optimizer-step budget and dataset role.
