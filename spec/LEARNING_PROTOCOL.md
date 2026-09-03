# Learning Protocol

Status: Frozen (`ADR-006` fixed literature-derived defaults)
Related ADR: `ADR-006-hyperparameter-selection.md`

## 1. Environment-learning interface

`MARGO-SPEC-v0.1` uses fixed-length autoregressive planning:

- decoder emits one action token per task
- environment consumes whole action plan after decoding
- one environment call returns whole-schedule metrics
- token-level shaped rewards are computed **post-hoc** by telescoping provisional schedules with completion policy `all_UE` (see `OBJECTIVE_AND_ENERGY.md` §6)

The implementation MUST NOT claim online observation of updated resource state after each decoder token.
The implementation MUST NOT invent a different `delta_latency_t` / `delta_energy_t` definition without a new spec version.

## 2. PPO contract

Inner adaptation MUST define all of the following explicitly:

- `support_graphs_per_meta_task`
- `trajectories_per_support_graph`
- `inner_optimizer`
- `inner_learning_rate`
- exact meaning of `K` / `k_steps`
- `ppo_batch_size_trajectories`
- batch shuffling policy
- value clipping equation
- entropy coefficient or explicit absence of entropy term

For v0.1 (frozen):

- `k_steps = 3` MUST mean count of Adam optimizer apply steps, not a legacy epoch label and not a minibatch-pass counter
- `ppo_inner_optimizer = adam` with `beta1=0.9`, `beta2=0.999`, `epsilon=1.0e-8`
- `inner_learning_rate = 5.0e-4`
- `policy_clip_epsilon = 0.2`
- `value_clip_epsilon = 0.2`
- `vf_coef = 0.5`
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

## 3. Outer update contract (`mrlco_first_order_mean_pseudogradient`)

All meta-task adaptations in one meta-batch MUST start from the same `theta0`.

Canonical update (minimization / `apply_gradients` convention):

```text
theta0 = core parameters
for task i in meta_batch:
    theta_i = copy(theta0)
    theta_i = adapt(theta_i, Adam(alpha=5e-4), optimizer_steps=3, fresh_optimizer_state=true)
mean_pseudogradient = mean( (theta0 - theta_i) / (alpha * k_steps) )
theta = outer_adam.apply(mean_pseudogradient)   # beta = 5e-4; state persistent
```

Required properties:

- order invariance up to numerical tolerance
- exactly one outer Adam application per meta-batch
- outer Adam optimizer state persistent across outer iterations
- no sequential per-task outer Adam on the core
- no Reptile interpolation `theta <- theta + eps * mean(theta_i - theta)`
- `beta` MUST NOT be written as `outer_step_size`

## 4. Policy synchronization contract

- before first sampling pass, task policies MUST equal core policy
- after every outer update, task policies MUST be synchronized from updated core policy
- no task slot may retain stale optimizer moments from a previous distribution assignment

## 5. Evaluation contract

- support and query sets MUST be disjoint
- adaptation uses support only
- reported metrics use query only
- zero-shot (`k_steps=0`) and frozen `k_steps=3` MUST both be reported
- meta-test MUST NOT be used to choose hyperparameters
- multi-seed runs after Phase 1–3 repair are **evaluation**, not tuning

## 6. Frozen structural budgets

- `meta_batch_size_distributions = 10` (sampled from `meta_train` only; MR-LCO 25-dist topology setting)
- `support_graphs_per_meta_task = 20`
- `trajectories_per_support_graph = 1` → `20` trajectories per meta-task (MR-LCO Table 2)
- `ppo_batch_size_trajectories = 20`
- `outer_iterations = 3500` (fixed compute budget, not a convergence claim)
- `validation_interval = 50`
- `checkpoint_selection_metric = validation_query_composite_objective` (for logging / checkpointing only; not for hyperparameter search)
- `early_stopping_rule = none_in_v0.1_fixed_budget`

`total_optimizer_steps_per_meta_task = k_steps = 3`.

## 7. Hyperparameter provenance

```yaml
hyperparameter_provenance:
  policy: fixed_literature_derived_defaults
  source: MR-LCO Table 2
  source_arxiv: "2008.02033v5"
  optimization_claim: false
  validation_tuning_performed: false
```

Manuscript / logs MUST NOT claim these values are optimized or best for MARGO.

Architecture sizes (MR-LCO `2×256` LSTM etc.) are **not** part of this freeze; deferred to Phase 2.

## 8. Legacy pitfalls (NOT v0.1)

Do not treat repository `declared_inner_grad_steps: 1` as one optimizer step.
Do not treat outer `β=5e-4` as Reptile `outer_step_size`.
Do not run sequential outer Adam once per task inside a meta-batch.

## 9. Minimum logged fields per run

- seed
- exact train/validation/test split version
- support graph count
- support trajectory count
- query graph count
- `k_steps`
- outer update count
- entropy coefficient or `0`
- value clip epsilon
- `ppo_batch_size_trajectories`
- `meta_batch_size_distributions`
- `inner_learning_rate`, outer Adam `learning_rate`
- `outer_update_method`
- `hyperparameter_provenance.policy`

## 10. Prohibited ambiguity

The following phrases MUST NOT appear without exact numeric definition:

- "one-step adaptation"
- "fast adaptation"
- "K=1"
- "few-shot adaptation"
- "stable convergence"
- "optimized hyperparameters" / "best hyperparameters" (forbidden for v0.1 defaults)

Each MUST map to a concrete optimizer-step budget and dataset role.
