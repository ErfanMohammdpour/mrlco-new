# ADR-006: Fixed Literature-Derived Learning Hyperparameters

Status: Accepted
Decision date: 2026-09-03

## Decision

Freeze MARGO v0.1 learning hyperparameters as **fixed literature-derived defaults** from MR-LCO Table 2 ([arXiv:2008.02033v5](https://arxiv.org/abs/2008.02033)), not as values claimed to be optimal for MARGO.

Frozen numeric values:

| Parameter | Symbol | Value |
| --- | --- | ---: |
| Inner learning rate | `α` | `5.0e-4` |
| Outer Adam learning rate | `β` | `5.0e-4` |
| Inner optimizer steps | `k_steps` / `m` | `3` |
| Discount | `γ` | `0.99` |
| GAE | `λ` | `0.95` |
| PPO clip | `ε` | `0.2` |
| Value-loss coefficient | `c1` / `vf_coef` | `0.5` |
| Entropy coefficient | — | `0.0` |
| Trajectories per meta-task | — | `20` |
| Meta-batch size (25-dist topology) | `B` | `10` |
| Inner / outer optimizer | — | Adam (`β1=0.9`, `β2=0.999`, `ε=1e-8`) |

Claims:

- `optimization_claim: false`
- `validation_tuning_performed: false`
- meta-test MUST remain untouched for hyperparameter choice (none is performed)

## Outer update (NOT Reptile)

MR-LCO `β` is an **outer Adam learning rate**, not a Reptile interpolation coefficient.

Canonical v0.1 outer method: `mrlco_first_order_mean_pseudogradient`

```text
theta0 = core parameters

for each meta-task i in meta_batch:
    theta_i = copy(theta0)
    theta_i = adapt(
        theta_i,
        inner_optimizer=Adam(alpha=5e-4),
        optimizer_steps=3,
        fresh_optimizer_state=true
    )

# minimization / apply_gradients convention:
mean_pseudogradient = mean( (theta0 - theta_i) / (alpha * k_steps) )
theta = outer_adam.apply(mean_pseudogradient)   # lr = beta = 5e-4
```

Equivalent ascent form uses `(theta_i - theta0) / (alpha * k_steps)`.

Required properties:

- one outer Adam application per meta-batch
- outer Adam state persistent across outer iterations
- all tasks start from the same `theta0`
- task-order invariant up to numerical tolerance
- no sequential per-task outer Adam on the core
- no Reptile `theta <- theta + eps * mean(theta_i - theta)`

## What is intentionally NOT copied from MR-LCO

Encoder/decoder architecture sizes (`2×256` LSTM, LayerNorm, Tanh) are **not** frozen here.
MARGO uses a graph encoder, three actions, and a latency-energy objective; architecture remains MARGO-specific and is deferred to Phase 2.

## Supersedes

This ADR supersedes the previous “validation-only candidate grid” decision for v0.1.
No `inner_learning_rate_candidates` / `outer_step_size_candidates` / `k_steps_candidates` grid is used.
No `hyperparameter_selection_evidence.json` artifact is required for Phase 0 closure.

## Why

Running a validation grid on a broken simulator/encoder/learning loop would freeze wrong numbers under a false “selection” claim.
Literature-derived fixed defaults are publication-safe when provenance and non-optimality are explicit.
Using Reptile `outer_step_size=5e-4` would misrepresent MR-LCO’s outer Adam update.
