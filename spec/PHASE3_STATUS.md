# Phase 3 — Learning loop

Status: IN PROGRESS

Branch: `phase3-learning` from tag `phase2-freeze-v0.1` (`a8ced73b32b1b565fdd1d0330a724282127f376b`).

**Do not move or rewrite `phase1-freeze-v0.1` or `phase2-freeze-v0.1`.**

Phase 3 does **not** imply paper results, multi-seed evaluation as a claim, or GPU training.

Until Phase 3 closes: **no scientific GPU training** and no new paper figures. Unit tests and CPU smoke tests are allowed.

## Scope

In:

- PPO value clip `v_old + clip(v_new - v_old, -eps, eps)`
- `k_steps = 3` means three Adam `apply_gradients` calls, not a minibatch-pass counter
- fresh inner Adam state per meta-task / task slot
- outer update = `mrlco_first_order_mean_pseudogradient`: shared `theta0`, mean `(theta0-theta_i)/(alpha k)`, **one** outer Adam apply
- mini-batch shuffle before each inner epoch
- `ppo_batch_size_trajectories = 20`, `support_graphs_per_meta_task = 20`
- trainer loads exactly the 15 `meta_train` distributions from `spec/split_loader.py`
- evaluator adapts on held-out **support** (20) and reports **query** (80)
- evaluator reports zero-shot `k_steps=0` then independent `k_steps=3` from the loaded core
- trainer syncs task slots from core at the **start** of every outer iteration
- validation every 50 outer iters: copy core → scratch policy, adapt on `validation_support`, metric on `validation_query`, never mutate core
- checkpoint metric = `validation_query_composite_objective` (mean query return; higher is better)
- LEARNING_PROTOCOL §9 log fields
- default `CUDA_VISIBLE_DEVICES=""` unless `MARGO_ALLOW_GPU=1` after Phase 3 close

Out until this status is CLOSED:

- scientific GPU training / paper figures
- claiming optimized hyperparameters
- TF 1.15 CPU smoke of the full inner/outer learning loop
- multi-seed evaluation campaign

## Notes

- Phase 2 encoder remains frozen. Do not reopen DAG packing or stats.
- Outer `β=5e-4` is Adam lr, not Reptile `outer_step_size`.
- Training still loads 100 graphs per train dist into memory, then samples 20 graph indices per meta-task.
- Integer `env.set_task(n)` clears `graph_indices` and leaks the full 100-graph pool. Held-out code must pass `{dist_index, graph_indices}`.

## Gate

```bash
python3 spec/phase3_gate.py
```

Must print `Phase 3 learning: PASS`. This is **not** Phase 3 closure.
