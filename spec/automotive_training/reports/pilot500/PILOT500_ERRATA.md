# ERRATA - 5 seeds x 500 iterations pilot (2026-10-02)

Status of this bundle:

```
ENGINEERING_PASS
SCIENTIFIC_PERFORMANCE_SUPERSEDED
```

Read this before using any number in `summary.json`, `validation_table.csv` or
`seed_*/logs/progress.csv`. **No raw file in this bundle was modified**; the errata is
additive.

## Defect 1 (critical): validation never evaluated the trained core

In `spec/automotive_training/automotive_primary.py@62f49e99` the evaluator guarded the
core -> scratch weight copy with

```python
assign = getattr(self.policy, "assign_trainable", None)
if assign is not None:
    assign(self.source_policy, self.policy)
```

`policies/meta_seq2seq_policy.py::Seq2SeqPolicy` has **no** `assign_trainable` method, so
the copy never happened. Consequences:

* k=0 measured a `validation_policy` that was never synchronised with the trained core;
* the k=3 adaptation ran on that same long-lived scratch policy, so adaptation
  **accumulated across validations** (iteration 50 adapted a policy already adapted at
  iteration 0, and so on);
* the lexicographic `best_val` checkpoint was therefore selected with a metric computed
  from a stale/accumulated scratch policy.

The correct implementation already existed (`meta_algos/variable_io.assign_trainable`,
used by `meta_algos/held_out_eval.py`). Fix: explicit, verified
`_sync_from_core()` at the start of **every** evaluation, plus a hard failure if the sync
is imperfect and an assertion that the adaptation never mutates the core (Gate B).

## Defect 2 (critical): the Lagrangian multipliers never reached the rollouts

`broadcast_constraint_lambdas()` wrote the new multipliers to the trainer's original env
only, while `MetaIterativeEnvExecutor` runs every rollout on `copy.deepcopy(env)` clones.
Proof in this bundle: for all five seeds, `constraint/penalty` is **0.00000000 for all
500 iterations** while `constraint/mean_violation_*` is non-zero for 500/500 - the duals
moved (`lambda_HIGH` up to 0.94) but the policy never received the penalty.

Reading: the pilot was effectively **latency-only training with externally tracked
duals**, not Lagrangian-constrained training. Fix: the executors (iterative and parallel)
now implement `set_constraint_lambdas`, the trainer broadcasts to every clone and raises
if no environment accepted the multipliers (Gate C).

## Defect 3 (latent): the penalty was applied twice

`AutomotiveEnv.step()` (non-single-distribution path) applied
`rewards[-1] -= penalty / D_G` and then applied the same penalty again through
`_telemetry()`. Dormant while the clones kept lambda=0, but it would have activated the
moment defect 2 was fixed. Fixed to a single application (Gate D1) with a blocking test
that also fails if the factor-of-two version ever returns.

## Defect 4: the dual batch was never reset

`AutomotiveDualController.observe()` appends to an internal batch and nothing cleared it
after `dual_step()`, so the ascent at iteration t used the mean over **all** violations
since iteration 0 (the end-of-run report shows `batch_size = 100000 = 500 x 200`). Fixed:
`reset_batch()` after every dual step, so `lambda_{t+1} = [lambda_t + eta * mean(V_t)]_+`
(Gate D2).

## Reporting defects (Gate E)

* `git_sha` was reported as `unknown` (the container has no `git` binary); now resolved
  from `.git/HEAD` with a `code_dirty` flag.
* `method_id` was the same constant for every budget; now derived from the run kind.
* sampler counters reported `support_calls = 1, query_calls = 999` because only the very
  first call of the run was labelled support; roles now follow the trainer's `log` flag
  and the contract (`support_calls == query_calls == iterations`) is enforced.
* `high_task_tardiness_rate` / `medium_task_tardiness_rate` were graph-level incidence
  rates with task-level names; renamed to
  `graph_high_tardiness_incidence_rate` / `graph_medium_tardiness_incidence_rate`, and
  true task-level rates plus task counts are now reported.
* `firm_task_miss_rate` divided by `20 * graphs`; the denominator is now the number of
  HIGH + MEDIUM tasks, with `firm_task_miss_count` reported separately.
* k=0 latency was not logged at all (only its discounted return, equal to
  `-latency`); it is now a first-class logged metric.

## What in this bundle remains valid

* the dataset, scheduler, MC runtime, energy/telemetry contract and the training
  plumbing are unaffected by these defects;
* `seed_*/logs/progress.csv` still contains the true optimisation trace of each run
  (it is what the policy actually optimised: latency-only in practice);
* the core checkpoint weights on the server remain usable for a **post-hoc corrected
  evaluation** (Gate G), clearly labelled as "latency-only training, constraints not
  active in the reward, corrected evaluator".

## What must not be used

* any k0/k3 latency, violation rate or checkpoint-selection claim from this bundle as a
  measurement of the trained policy;
* the comparison against all-MEC / greedy / HEFT v2 in `ANALYSIS.md` section 7 (the
  reference numbers themselves are correct, but the model side was mis-measured).
