# 500-iteration pilot: iteration-count audit (Gate-21 follow-up)

Question: does changing `outer_iterations` from 3500 to 500 change ANY optimization
schedule? Audited by grep over the primary path
(`meta_trainer.py`, `meta_algos/*.py`, `spec/learning_ops.py`, `spec/eval_protocol.py`,
`spec/objective_contract.py`, `spec/constraints_config.py`,
`scheduler/constraints.py`, `spec/automotive_training/*`).

| mechanism | depends on total iterations? | evidence |
|---|---|---|
| outer loop bound | **yes, and that is the only thing that should** | `meta_trainer.py:374` `for itr in range(self.start_itr, self.n_itr)` |
| inner LR (5e-4) / outer LR (5e-4) | no | constants at `meta_trainer.py:912/913`; `frozen_experiment.yaml:learning` has no schedule/decay field |
| LR decay / annealing / cosine / step schedule | **none exists** | no `exponential_decay`/`polynomial_decay`/`lr_schedule`/`anneal` match anywhere on the path |
| epsilon-greedy / exploration schedule | **none exists** | entropy coefficient constant `0.0` (`ppo_offloading.py:35`, enforced), no epsilon term |
| critic warmup | no | `critic_warmup_iters=0`; the parameter is only accepted for `learning_mode="kl_bc_ppo"`, which the automotive primary does not use |
| dual ascent (lambda) | **no schedule**, but the number of ascent steps equals the number of iterations | `automotive_constraints.AutomotiveDualController`: fixed `dual_lr=0.05`, one projected ascent step per outer iteration |
| validation cadence | no | `validation_interval=50` and `meta_trainer.py:94-95` hard-requires 50 → 0,50,...,450 for a 500-iteration run |
| checkpoint scoring | no | lexicographic rule `automotive_lexicographic_v1`; no iteration term, only fewer evaluations (10 vs 70) |
| save interval / final checkpoint | no | `save_interval=100` → 0,100,...,400 plus `meta_model_final.ckpt` at the end of `train()` |
| early stopping | no | `early_stopping_rule: none_in_v0.1_fixed_budget` |
| inner PPO step count / meta-batch / support budget | no | frozen constants 3 / 10 / 20, asserted at construction |

## Reported consequences of 3500 -> 500 (explicit, not silent)

1. The number of outer meta-updates drops from 3500 to 500.
2. The dual variables therefore receive 500 ascent steps instead of 3500. There is no
   dual schedule to compress, but the **terminal lambda** (and hence the cumulative
   constraint pressure) will differ from a 3500-step run. This is reported explicitly.
3. Validation runs 10 times (itr 0,50,...,450) instead of 70; the last validation is at
   iteration 450 because 499 is not a multiple of 50. A final checkpoint is still
   written at the end of training.
4. Nothing else changes: dataset, split, observation schema, normalization, scheduler
   axes, MC runtime, constraints, sampler budgets, PPO hyperparameters, checkpoint rule
   and the meta-test guard are identical — enforced mechanically by
   `validate_pilot_config.py`, which fails on any unexpected difference between
   `frozen_automotive_primary.yaml` and `frozen_automotive_pilot_500.yaml`.
