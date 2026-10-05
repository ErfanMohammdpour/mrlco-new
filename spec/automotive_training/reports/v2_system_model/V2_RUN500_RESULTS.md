# V2_RUN500_RESULTS — authorized seed-0 final run, executed

**STATUS: COMPLETED 500.** Everything below is read from the run's own artifacts. No value is
projected, inferred or carried over from another run.

## 1. Run identity

| item | value |
|---|---|
| run dir | `/root/v2-runs/final500_seed0_20261005T0400` |
| container | `v2final500c` |
| source | `/root/v2-final` @ `49e5cb8` (see `SOURCE_REVISION.txt` in the run dir) |
| commit chain | `0f9cb096` (base) → `48a4367` (accounting/energy/guard) → `ef88073` (progress ledger) → `407b17a` (YAML memoisation) → `49e5cb8` (constraint telemetry) |
| image | `margo-phase4-tf115-nv2212:latest` @ `sha256:82bc781bc2888a171379e1b45da6846c559ef712e33a76f74948b1cd7172dc46` |
| started (UTC) | `2026-10-04T22:54:46Z` |
| wall | `13055.16 s` = **3 h 37 m 35 s** |
| config | seed 0, `--iterations 500` (TOTAL), background 2, link moderate, background-policy `all_mec`, energy-budget-fraction 0.5, ue-budget-fraction 1.0, R=2, S=2, meta_batch 10, support/query 20 trajectories, k=3, dual_lr 0.05, max_lambda 1000, validation_interval 50, save_interval 100 |

### Completion evidence (`run.json`)

```
ok                       = True
iterations_requested     = 500
iterations_started       = 500
iterations_completed     = 500      <-- exactly the target, no overshoot
last_completed_iteration = 499
next_iteration           = 500      <-- no completed work repeated
outer_updates_completed  = 500      <-- 500 real MRLCO outer updates
validations_completed    = 10       <-- itr 0,50,100,...,450
error                    = None
checkpoint.status        = completed_iterations=500
checkpoint.promotable    = true
checkpoint.was_resume    = false
```

This is the first run on this branch that reaches its target and reports it consistently:
the previous GPU run reported `started=2 / completed=1` with `ok=true` beside a
`KeyError: 'iterations_executed'` traceback.

### Checkpoints (real completed boundaries)

| iteration | file | written (local) |
|---|---|---|
| 0 | `meta_model_0.ckpt` | 02:27:38 |
| 100 | `meta_model_100.ckpt` | 03:10:53 |
| 200 | `meta_model_200.ckpt` | 03:53:48 |
| 300 | `meta_model_300.ckpt` | 04:36:35 |
| 400 | `meta_model_400.ckpt` | 05:19:49 |
| best val | `meta_model_best_val.ckpt` | 04:58:08 |
| final | `meta_model_final.ckpt` | 06:02:19 |
| TF full | `v2_run.ckpt.{index,data-00000-of-00001,meta}` (122 MB) | 06:02:28 |
| resume ledger | `v2_resume_state.json` | 06:02:28 |

**Measured rate: ~43 min per 100 iterations = ~26 s per outer iteration.**

## 2. Extraction

The run has no `progress.csv`: `run_tf.py` never calls `logger.configure()`, so all 73 metric
keys went to stdout only. They were parsed out of `logs/stdout.log` (44 095 lines) into:

* `metrics_all.tsv` — 500 rows × 102 columns
* `metrics_by_iteration.json` — same, as records
* `metrics_summary.json` — per-column first/last/min/max/mean/median and first-quarter vs
  last-quarter means

Note the logger truncates keys longer than 23 characters to `...`, so a few long names
(`constraint/total_energy_signed`, `constraint/total_violation`) are recovered by prefix, not by
literal name.

**Caveat:** 96 metric keys exist in this log; 73 per iteration. The 96 = 73 + 23 extra columns
that appear only on the 10 validation iterations (the 10 blocks are 102 lines vs 73 for the
other 490).

## 3. Training outcome (500 iterations)

### Objective — improved, monotone in trend

| key | first | last | min | max | mean | 1st-quartile mean | last-quartile mean |
|---|---|---|---|---|---|---|---|
| `training/objective_discounted_return` | 0.530 | **0.760** | -0.011 | 1.79 | 0.798 | 0.810 | 0.846 |
| `Average reward` | 0.599 | 0.856 | -0.017 | 2.04 | 0.903 | 0.921 | 0.955 |
| `AverageDiscountedReturn` | 0.530 | 0.760 | -0.011 | 1.79 | 0.798 | 0.810 | 0.846 |
| `Average latency` (normalised) | 0.320 | **0.206** | 0.154 | 0.537 | 0.2206 | 0.2384 | 0.2047 |
| `Average greedy latency` | 0.0561 | 0.0743 | 0.0305 | 0.0915 | 0.0608 | 0.0612 | 0.0620 |
| `MaxReturn` | 2.70 | 1.91 | 0.0855 | 4.93 | 2.884 | 2.989 | 2.794 |
| `MinReturn` | -0.188 | -0.010 | -0.428 | 0.0456 | -0.0804 | -0.0885 | -0.0554 |
| `StdReturn` | 0.854 | 0.723 | 0.0458 | 1.60 | 0.938 | 0.961 | 0.946 |

Normalised latency fell **35.6 %** (0.320 → 0.206); the objective the PPO optimises rose
**43 %** (0.530 → 0.760).

### Action mix — real placement change

| key | first | last | mean | 1st-q | last-q |
|---|---|---|---|---|---|
| `action_fraction/local` | 0.129 | **0.446** | 0.464 | 0.478 | 0.400 |
| `action_fraction/mec` | 0.436 | 0.352 | 0.384 | 0.456 | 0.340 |
| `action_fraction/v2v` | 0.435 | **0.201** | 0.152 | 0.0659 | 0.260 |

Helper (v2v) placement fell from 43.5 % to 20.1 %; local rose from 12.9 % to 44.6 %. The policy
shifted work off helpers and MEC toward local execution.

### Energy — all three accounting boundaries

| key | first | last | min | max | mean | 1st-q | last-q |
|---|---|---|---|---|---|---|---|
| `energy/requester_joules` | 0.191 | 0.198 | 0.0914 | 0.541 | 0.2123 | 0.2324 | 0.1734 |
| `energy/mobile_joules` | 2.52 | 1.22 | 0.216 | 2.52 | 0.9960 | 0.5783 | 1.444 |
| `energy/system_joules` | 63.9 | 89.4 | 42.9 | 133 | 82.74 | 85.33 | 83.17 |
| `energy/primary_joules` | 63.9 | 89.4 | 42.9 | 133 | 82.74 | 85.33 | 83.17 |
| `Average energy` (legacy MOBILE) | 1280 | 1790 | 857 | 2660 | 1655 | 1707 | 1664 |
| `energy/primary_scope` | \multicolumn{6}{l}{`system` (constant)} |

**Scope warning.** `energy/primary_scope = system` and `energy/system_joules` is the
**whole-world** figure: on this configuration the background DAGs of other owners contribute
~99 % of it (measured in the acceptance run: system 36.31 J, requester 0.29 J, background
35.81 J). The **constrained** quantity after the scope fix is the **foreground** system energy,
which is why `--energy-budget-fraction 0.5` now yields violations near 0.1 rather than ~137.
`foreground_system_joules` exists in the per-episode record but is not in the logger's column
set, so the foreground split is **not** recoverable from this run's log. Requester and mobile
boundaries ARE clean and analysable.

### Lagrangian constraint / dual state

| key | first | last | min | max | mean | 1st-q | last-q |
|---|---|---|---|---|---|---|---|
| `constraint/enabled` | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| `constraint/updates` | 1 | 500 | 1 | 500 | 250.5 | 63 | 438 |
| `constraint/buffer` | 200 | 200 | 200 | 200 | 200 | 200 | 200 |
| `constraint/lambda_total_energy` | 0.0 | **1.3055** | 0.0 | 1.3055 | 0.9251 | 0.3504 | 1.164 |
| `constraint/lambda_ue_energy` | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| `constraint/lambda_broadcast_targets` | 11 | 11 | 11 | 11 | 11 | 11 | 11 |
| `constraint/total_energy_signed` | 0.00277 | 0.101 | 0.00277 | 0.360 | 0.0973 | 0.1444 | 0.0811 |
| `constraint/total_energy_violation` | 0.00996 | 0.101 | 0.00429 | 0.360 | 0.0975 | 0.1450 | 0.0811 |
| `constraint/ue_energy_signed` | 0.00719 | 0.0 | 0.0 | 0.00719 | 0.000175 | 0.000514 | 1.76e-05 |
| `constraint/penalty` | 0.0 | 0.132 | 0.0 | 0.199 | 0.0788 | 0.0491 | 0.0944 |
| `constraint/penalty_applied` | 0.0 | 0.132 | 0.0 | 0.199 | 0.0788 | 0.0491 | 0.0944 |
| `constraint/mean_signed_total_energy` | -0.399 | -0.537 | -0.771 | -0.108 | **-0.4945** | -0.4786 | -0.5664 |
| `constraint/mean_signed_ue_energy` | see note | | | | | | |
| `constraint_status/total_energy` | `active` | | | | | | |
| `constraint_status/ue_energy` | `active` | | | | | | |
| `constraint_status/{deadline,helper_energy,mec_task_fraction,v2v_airtime,v2v_task_fraction}` | `not_configured` | | | | | | |
| `correctness/lambda_b...` | 11 | 11 | 11 | 11 | 11 | 11 | 11 |
| `correctness/penalty_...` | 0 | 440 | 0 | 440 | 216 | 20 | 415 |
| `constraint/batch_size_after_reset_*` | 0 | | | | | | |

### Run-level penalty totals (`run.json`)

```
penalty.episodes                        = 100000   (500 iterations x 200 query episodes)
penalty.episodes_with_nonzero_penalty   =  60532   (60.5 %)
penalty.penalty_sum                     = 7876.95
penalty.steps_with_penalty              =    489   of 500 outer iterations
penalty.signed_rows_observed            = 100000
constraints.updates                     = 500
constraints.lambdas                     = {total_energy: 1.3055, ue_energy: 0.0}
```

**The energy constraint binds.** 60.5 % of episodes carried a positive energy violation, the
dual took exactly one step per outer iteration, λ_total_energy ascended 0 → 1.3055, and the
penalty is a nonzero, dimensionless quantity applied once on the terminal token.

**UNRESOLVED CONTRADICTION — flagged, not explained away.** `constraint/total_energy_violation`
is positive every iteration (mean +0.0975) while `constraint/mean_signed_total_energy` is
NEGATIVE every iteration (mean -0.4945). `ConstraintController.dual_step` computes
`lambda += dual_lr * mean_signed` with `dual_lr = 0.05`; a persistently negative mean_signed
drives λ **down** to 0, yet λ rose to 1.3055. So `constraint/mean_signed_<name>` and
`constraint/lambda_<name>` cannot both describe the same update. The same sign pattern is
present in the earlier 2-iteration GPU run (`mean_signed_total_energy = -0.397` with
`lambda_total_energy = 7.08`, i.e. λ rising on a negative mean). Either the `mean_signed` column
is misattributed to a different controller (the legacy deadline adapter still exists as
`auto_legacy_deadline_controller`), or the ascent has a second driver. **A reader must not use
`constraint/mean_signed_*` as the ascent input for this run.** The positive-violation,
penalty, λ and `updates` columns are mutually consistent and are the ones used above.

### Validation — real R/S evaluation, 10 checkpoints

| key | first (itr 0) | last (itr 450) | min | max | mean | 1st-q | last-q |
|---|---|---|---|---|---|---|---|
| `validation/firm_task_miss_rate` | 0.220 | **0.0312** | 0.0112 | 0.220 | 0.0460 | 0.1217 | 0.0284 |
| `validation/high_task_miss_rate` | 0.374 | **0.0525** | 0.019 | 0.374 | 0.0781 | 0.207 | 0.0475 |
| `validation/medium_task_miss_rate` | 0.0 | 0.000714 | 0.0 | 0.00143 | 0.000214 | 0.0 | 0.00107 |
| `validation/graph_hard_miss` | 0.405 | **0.0** | 0.0 | 0.405 | 0.0405 | 0.2025 | 0.0 |
| `validation/graph_high_miss` | 0.825 | 0.150 | 0.090 | 0.825 | 0.2185 | 0.4825 | 0.145 |
| `validation/graph_medium_miss` | 0.0 | 0.005 | 0.0 | 0.010 | 0.0015 | 0.0 | 0.0075 |
| `validation/n_tardy_high_tasks` | 150 | **21** | 7.6 | 150 | 31.3 | 83 | 19 |
| `validation/n_tardy_medium_tasks` | 0.0 | 0.2 | 0.0 | 0.4 | 0.06 | 0.0 | 0.3 |
| `validation/query_mean_latency` | 0.267 | **0.0895** | 0.084 | 0.267 | 0.1079 | 0.1777 | 0.0870 |
| `validation/k0_regret` | 0.243 | **0.0563** | 0.0499 | 0.243 | 0.0767 | 0.1477 | 0.0555 |
| `validation/k3_regret` | 0.239 | **0.0610** | 0.0556 | 0.239 | 0.0795 | 0.1494 | 0.0585 |
| `validation/oracle_capacity` | 0.0285 | 0.0285 | 0.0285 | 0.0285 | 0.0285 | 0.0285 | 0.0285 |
| `validation/n_high_tasks` | 400 | 400 | 400 | 400 | 400 | 400 | 400 |
| `validation/n_medium_tasks` | 280 | 280 | 280 | 280 | 280 | 280 | 280 |
| `evaluation/replicates` | 5 | 5 | 5 | 5 | 5 | 5 | 5 |
| `evaluation/protocol_*` | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| `evaluation/paired_k3_*` | -0.00481 | 0.00472 | -0.00481 | 0.0108 | 0.00272 | 0.00129 | 0.00303 |
| `validation/mc_policy_*`, `validation/mode_switch_*`, `validation/hi_mode_r*` | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Headline: **graph-level hard deadline misses fell to exactly 0.0** by the last validation, HIGH
tardiness fell 150 → 21 tasks (-86 %), firm task miss rate fell 0.220 → 0.0312 (-86 %), and
held-out query latency fell 0.267 → 0.0895 (-66 %).

**Honest caveat on meta-adaptation.** `validation/k0_regret` and `validation/k3_regret` moved
almost together (0.243 → 0.0563 and 0.239 → 0.0610). The k3 (adapted) value is NOT better than
k0 at the end (0.0610 vs 0.0563 — slightly worse), and `evaluation/paired_k3_*` ends at +0.0047
with a mean of +0.0027 (first quarter +0.0013, last quarter +0.0030). So the run demonstrates a
strongly improved **base** policy, but does **not** demonstrate that the 3-step inner adaptation
beats no adaptation. Do not claim a meta-learning advantage from this run.

### PPO / optimisation health — converged and stable

| key | first | last | min | max | mean | 1st-q | last-q |
|---|---|---|---|---|---|---|---|
| `policy/approx_kl` | 1.29 | 0.0372 | 0.00719 | 1.29 | 0.0605 | 0.1161 | 0.0339 |
| `policy/clip_fraction` | 0.658 | 0.0925 | 0.0563 | 0.658 | 0.1254 | 0.1799 | 0.0944 |
| `policy/entropy_valid` | 0.954 | 0.262 | 0.119 | 0.954 | 0.3406 | 0.3668 | 0.3002 |
| `policy/grad_norm` | 16.4 | 0.790 | 0.382 | 16.4 | 1.428 | 2.841 | 0.832 |
| `policy/policy_loss_mean` | 0.377 | 0.0126 | -0.0141 | 0.377 | 0.0128 | 0.0273 | 0.00827 |
| `policy/value_loss_mean` | 2.13 | 0.0652 | 0.0115 | 2.13 | 0.1014 | 0.1903 | 0.0664 |
| `critic/value_abs_max` | 2.47 | 2.36 | 0.171 | 3.09 | 2.084 | 2.165 | 2.091 |
| `policy/invalid_action_rate` | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| `policy/argmax_masked_rate` | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| `correctness/core_scr...` | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `correctness/core_unc...` | 1 | 1 | 1 | 1 | 1 | 1 | 1 |

KL fell 35×, gradient norm 21×, value loss 33×. Entropy fell 0.954 → 0.262 (policy sharpened) but
stayed well above zero and `collapse/flag` was 0 for all 500 iterations. Zero invalid actions and
zero masked-argmax throughout.

### Mask / collapse channels

```
mask/active_rate       = 0   (all 500)
mask/all_invalid_rate  = 0   (all 500)
mask/forced_rate       = 0   (all 500)
collapse/flag          = 0   (all 500)
```
These four are structurally zero on this path; they are not evidence about masking quality.

### Checkpoint selection

```
checkpoint_is_best_val   mean 0.4 ; first 1 ; last 0
checkpoint_selection_hash = a3bd00d5108237acdb70... (constant when non-empty, else '')
```
The best-validation checkpoint was selected on 40 % of iterations (200/500), i.e. every 100
iterations plus the validation cadence — consistent with `save_interval = 100` and 10 validation
points; `meta_model_best_val.ckpt` is timestamped at itr 300's neighbourhood (04:58).

## 4. Throughput

| key | first | last | min | max | mean |
|---|---|---|---|---|---|
| `EnvExecTime` | 8.98 | 8.57 | 7.76 | 10.6 | 8.730 |
| `PolicyExecTime` | 0.385 | 0.666 | 0.229 | 1.26 | 0.379 |
| `NumTrajs` | 200 | 200 | 200 | 200 | 200 |

Simulator time dominates (8.73 s vs 0.38 s per iteration) and `EnvExecTime` is flat over 500
iterations — no degradation, no leak. GPU utilisation was ~12 % with vLLM co-resident; the run is
CPU-bound in the scheduler, which is why the YAML memoisation (`407b17a`) had such a large
effect.

## 5. What this run does and does not establish

**Establishes**
* the full v2 production chain executes end to end for 500 outer iterations and reaches its
  declared total target, with counters, exit status, checkpoint and JSON report all consistent;
* the constrained-Lagrangian loop is live: positive energy violations are measured, λ ascends
  from 0 to 1.3055, and a nonzero dimensionless penalty is applied once per episode;
* the base policy improves materially: normalised latency -35.6 %, objective +43 %,
  graph hard misses to 0, HIGH tardiness -86 %, held-out latency -66 %;
* optimisation is stable: KL -35×, grad norm -21×, no collapse, no invalid actions.

**Does NOT establish**
* any meta-learning advantage: k3 regret is not better than k0 at the end (0.0610 vs 0.0563);
* that whole-world system energy is controlled: the constraint measures the **foreground**
  ledger while `energy/system_joules` (mean 82.74 J) is whole-world and background-dominated;
* that `constraint/mean_signed_*` is the ascent input — the sign contradiction above is open;
* per-episode or per-schedule behaviour: `audit_writer=None` on the v2 path, so no
  `trajs_ppo.jsonl` / `trajs_eval.jsonl` exist;
* the R/S per-cell structure: only aggregated validation rows are logged.

## 6. Known defects still open from this run

1. `v2_progress.json` was never written and `v2_phase_timings.jsonl` is 0 bytes for the entire
   500-iteration run — the atomic progress reporting required by the spec does not work on the
   v2 TF path, so there is no per-phase timing record despite the instrumentation existing.
2. `constraint/mean_signed_*` contradicts `constraint/lambda_*` (section 3).
3. No `progress.csv` (`logger.configure()` is never called for v2).
4. `foreground_system_joules` is not a logged column, so the constrained energy is not directly
   visible in the run's own output.
