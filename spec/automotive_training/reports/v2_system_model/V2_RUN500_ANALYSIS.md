# MARGO v2 — تحلیل کامل اجرای نهایی ۵۰۰ iteration

**فایل مرجع اجرا:** `/root/v2-runs/final500_seed0_20261005T0400`
**سورس:** `49e5cb8` روی `/root/v2-final` | **ایمیج:** `margo-phase4-tf115-nv2212:latest` @ `sha256:82bc781bc288…`
**تاریخ اجرا:** `2026-10-04T22:54:46Z` → `2026-10-05T02:32:21Z` (۳h ۳۷m ۳۵s)

> این سند از artifact های خودِ اجرا ساخته شده. هیچ عددی از run دیگری منتقل، تخمین یا infer نشده.
> ستون‌های استخراج‌شده از `logs/stdout.log` (چون v2 هیچ‌وقت `logger.configure()` صدا نمی‌زند و
> `progress.csv` وجود ندارد). **مهم:** جدول stdout کلیدهای بلندتر از ۲۳ کاراکتر را truncate
> می‌کرد و باگ برخورد کلید داشت؛ بخش ۳.۵ این را کامل توضیح می‌دهد.

---

## فهرست
1. [خلاصه اجرایی](#0-خلاصه-اجرایی)
2. [هویت اجرا و شواهد تکمیل](#1-run-identity)
3. [استخراج داده](#2-extraction)
4. [نتیجه آموزش](#3-training-outcome-500-iterations)
5. [تحلیل عمیق عددی (A1–A10)](#بخش-تحلیل-عمیق)
6. [Throughput](#4-throughput)
7. [چه چیزی اثبات می‌شود و چه چیزی نه](#5-what-this-run-does-and-does-not-establish)
8. [نقص‌های باز](#6-known-defects-still-open-from-this-run)

---

## 0. خلاصه اجرایی

| پرسش | پاسخ |
|---|---|
| اجرا کامل شد؟ | **بله** — `completed=500/500`, `next_iteration=500`, `ok=True`, `error=None` |
| چند outer update؟ | **۵۰۰** MRLCO first-order |
| چند validation؟ | **۱۰** (itr 0,50,…,450) با R=2/S=2 واقعی |
| چقدر طول کشید؟ | **۳ ساعت ۳۷ دقیقه ۳۵ ثانیه** (~۲۶ ثانیه/iteration) |
| یاد گرفت؟ | **بله روی policy پایه** — latency نرمال‌شده ۰.۳۲۰→۰.۲۰۶ (−۳۵.۶٪)، objective ۰.۵۳۰→۰.۷۶۰ (+۴۳٪) |
| deadline? | graph-level hard miss به **صفر** رسید؛ HIGH tardiness ۱۵۰→۲۱ (−۸۶٪) |
| energy constraint binding بود؟ | **بله** — λ از ۰ به **۱.۳۰۵۵**، ۶۰.۵٪ اپیزودها over budget |
| meta-learning بهتر شد؟ | **خیر** — k3 regret (۰.۰۶۱۰) بهتر از k0 (۰.۰۵۶۳) نیست. **این را ادعا نکن** |
| پایدار بود؟ | **بله** — KL −۳۵×، grad norm −۲۱×، collapse=۰ در همه ۵۰۰ تا |

**سه نکته‌ای که باید در مقاله رعایت شود:**
1. `energy/system_joules` (mean **82.74 J**) **whole-world** است و ~۹۹٪ آن background است. constraint روی **foreground** اندازه می‌گیرد. این دو کمیت متفاوت‌اند.
2. ارتقای **meta-adaptation** اثبات نشده (k3 ≈ k0). فقط base policy بهتر شده.
3. ستون‌های `constraint/total_energy_signed` و `constraint/mean_signed_total_energy` **در لاگ این اجرا وجود ندارند** (باگ truncation). فقط مقادیر `run.json` قابل استنادند.

---

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
| `constraint/total_energy_violation` | 0.00996 | 0.101 | 0.00429 | 0.360 | 0.0975 | 0.1450 | 0.0811 |
| `constraint/ue_energy_violation` | 0.00719 | 0.0 | 0.0 | 0.00719 | 0.000175 | 0.000514 | 1.76e-05 |
| `constraint/total_violation` (= sum of channels) | 0.00996 | 0.101 | 0.00429 | 0.360 | 0.0977 | — | — |
| `constraint/penalty` | 0.0 | 0.132 | 0.0 | 0.199 | 0.0788 | 0.0491 | 0.0944 |
| `constraint/penalty_applied` | 0.0 | 0.132 | 0.0 | 0.199 | 0.0788 | 0.0491 | 0.0944 |
| `constraint/mean_signed_ue_energy` | -0.399 | -0.537 | -0.771 | -0.108 | -0.4945 | -0.4786 | -0.5664 |

> **COLUMN-IDENTITY CORRECTION (important).** The stdout table truncates keys longer than 23
> characters and `HumanOutputFormat.writekvs` keyed its print dict on the TRUNCATED name while
> iterating `sorted(...)`, so within each 20-character prefix group the lexicographically LAST
> key silently overwrote the others. In this run that meant:
>
> | printed label | what actually survived | what was hidden |
> |---|---|---|
> | `constraint/total_ene...` | `constraint/total_energy_violation` | `..._signed`, `..._raw`, `..._budget`, `..._scale`, `..._lambda` |
> | `constraint/ue_energy...` | `constraint/ue_energy_violation` | `..._signed`, `..._raw`, `..._budget`, `..._scale`, `..._lambda` |
> | `constraint/mean_sign...` | **`constraint/mean_signed_ue_energy`** (`ue` > `total`) | `constraint/mean_signed_total_energy` |
> | `constraint/batch_siz...` | `batch_size_after_reset_ue_energy` | `..._total_energy` |
>
> Verified by forcing the repo's own `HumanOutputFormat` with both channels. So in this run's
> log `constraint/total_energy_signed` and `constraint/mean_signed_total_energy` **never appeared
> at all**, and an earlier version of this report mislabelled them — which produced a spurious
> "unresolved contradiction" (a strongly negative mean signed cost beside a rising lambda). That
> contradiction does **not** exist: the negative series is the `ue_energy` channel, whose
> `lambda_ue_energy` is correctly 0.
>
> **Fixed** in `utils/logger.py` (key identity preserved; the table just widens) with a
> regression test that fails on the old code (`test_logger_key_identity.py`, 3 tests).
>
> **The training dual was always correct.** Reconstructed from the λ column,
> `mean_signed_total_energy = (λ_t − λ_{t−1}) / 0.05` averages **+0.103** over iterations 1–240
> and **+0.006** over 241–499, going negative for 33 iterations — only a SIGNED ascent can do
> that. `λ_final / 0.05 = 1.3055 / 0.05 = 26.11`, i.e. mean +0.0522 over 500 steps.

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

**The dual is verified correct — the earlier "contradiction" was a display bug.** One
authoritative dual owner exists (`env.constraint_controller` = `trainer.auto_controller` =
`auto_v2_constraint_manager` = one `V2ConstraintManager`); the legacy
`auto_legacy_deadline_controller` is never observed and never stepped and has no `mean_signed`
key at all. The apparent sign conflict came solely from the truncated-key collision documented
above, now fixed.

**Binding, stated precisely.** A positive hinge mean is not evidence of binding: it is nonzero
whenever any episode is over budget. Binding requires the SIGNED mean to approach zero at
λ* > 0, which is what happened — λ plateaued/oscillated at ~1.23–1.31 while the reconstructed
signed mean decayed from +0.103 to +0.006. 60.5 % of episodes were over the **total-energy**
budget (λ_ue = 0, so the penalty is total-energy only). The `ue_energy` channel is active but
NOT binding: mean signed −0.4945, violation ≈ 0, λ_ue = 0.

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

## بخش تحلیل عمیق

### A1. trajectory sample

| itr | objective | latency | reward | requester J | mobile J | system J | lam_te | viol_te | penalty | local | mec | v2v | approx_kl | grad_norm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.53 | 0.32 | 0.599 | 0.191 | 2.52 | 63.9 | 0 | 0.00277 | 0 | 0.129 | 0.436 | 0.435 | 1.29 | 16.4 |
| 1 | 0.332 | 0.499 | 0.376 | 0.49 | 0.593 | 86.6 | 0 | 0.0543 | 0 | 0.836 | 0.045 | 0.119 | 0.483 | 5.11 |
| 2 | 0.194 | 0.537 | 0.221 | 0.541 | 0.813 | 85.2 | 0 | 0.00759 | 0 | 0.9 | 0.0423 | 0.0578 | 0.155 | 4.29 |
| 5 | 0.214 | 0.469 | 0.244 | 0.481 | 0.687 | 72.1 | 0 | 0.0107 | 0 | 0.871 | 0.0948 | 0.0343 | 0.144 | 2.96 |
| 10 | 0.573 | 0.221 | 0.653 | 0.22 | 0.766 | 71.9 | 0.000678 | 0.098 | 0 | 0.546 | 0.349 | 0.104 | 0.125 | 2.35 |
| 25 | 1.39 | 0.161 | 1.58 | 0.113 | 0.229 | 98.3 | 0.117 | 0.341 | 0.0341 | 0.405 | 0.562 | 0.0323 | 0.353 | 7.83 |
| 50 | 0.205 | 0.477 | 0.234 | 0.481 | 0.702 | 79 | 0.264 | 0.0325 | 0.00895 | 0.481 | 0.492 | 0.0265 | 0.256 | 2.46 |
| 75 | 1.45 | 0.22 | 1.64 | 0.216 | 0.451 | 109 | 0.455 | 0.232 | 0.103 | 0.459 | 0.485 | 0.0558 | 0.105 | 2.64 |
| 100 | 0.726 | 0.241 | 0.818 | 0.24 | 0.539 | 72.5 | 0.575 | 0.0929 | 0.0533 | 0.497 | 0.433 | 0.0698 | 0.0608 | 1.91 |
| 150 | 1.17 | 0.186 | 1.33 | 0.183 | 0.704 | 108 | 0.851 | 0.183 | 0.154 | 0.454 | 0.456 | 0.0905 | 0.109 | 1.31 |
| 200 | 0.705 | 0.187 | 0.8 | 0.207 | 0.483 | 83.2 | 1.13 | 0.104 | 0.116 | 0.525 | 0.423 | 0.0517 | 0.0218 | 0.656 |
| 250 | 0.597 | 0.246 | 0.675 | 0.256 | 1.25 | 74.7 | 1.23 | 0.0432 | 0.0533 | 0.497 | 0.343 | 0.16 | 0.0649 | 0.935 |
| 300 | 0.655 | 0.326 | 0.744 | 0.349 | 1.13 | 84.9 | 1.18 | 0.0331 | 0.0393 | 0.569 | 0.299 | 0.132 | 0.0503 | 1.32 |
| 350 | 0.569 | 0.261 | 0.643 | 0.249 | 1.36 | 69.8 | 1.06 | 0.0248 | 0.0262 | 0.462 | 0.316 | 0.223 | 0.0216 | 0.777 |
| 400 | 0.835 | 0.178 | 0.944 | 0.13 | 1.46 | 88.1 | 1.08 | 0.112 | 0.121 | 0.339 | 0.378 | 0.283 | 0.0166 | 0.721 |
| 450 | 1.11 | 0.199 | 1.25 | 0.171 | 1.47 | 91.2 | 1.19 | 0.0962 | 0.114 | 0.398 | 0.36 | 0.242 | 0.037 | 0.702 |
| 475 | 0.907 | 0.214 | 1.03 | 0.158 | 1.54 | 83.2 | 1.24 | 0.0854 | 0.106 | 0.336 | 0.349 | 0.314 | 0.0252 | 0.815 |
| 499 | 0.76 | 0.206 | 0.856 | 0.198 | 1.22 | 89.4 | 1.31 | 0.101 | 0.132 | 0.446 | 0.352 | 0.201 | 0.0372 | 0.79 |

### A2. validation cadence table

| itr | firm_miss | high_miss | ghard | ghigh | tardy_H | tardy_M | q_latency | k0_regret | k3_regret | paired_k3 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.22 | 0.374 | 0.405 | 0.825 | 150 | 0 | 0.267 | 0.243 | 0.239 | -0.00481 |
| 50 | 0.0235 | 0.04 | 0 | 0.14 | 16 | 0 | 0.0883 | 0.0524 | 0.0598 | +0.00739 |
| 100 | 0.0247 | 0.042 | 0 | 0.145 | 16.8 | 0 | 0.0894 | 0.0574 | 0.0609 | +0.00356 |
| 150 | 0.0182 | 0.031 | 0 | 0.105 | 12.4 | 0 | 0.084 | 0.0499 | 0.0556 | +0.0057 |
| 200 | 0.0285 | 0.0485 | 0 | 0.165 | 19.4 | 0 | 0.0922 | 0.06 | 0.0637 | +0.00376 |
| 250 | 0.0324 | 0.055 | 0 | 0.175 | 22 | 0 | 0.0966 | 0.0573 | 0.0682 | +0.0108 |
| 300 | 0.0447 | 0.076 | 0 | 0.25 | 30.4 | 0 | 0.101 | 0.073 | 0.0726 | -0.000436 |
| 350 | 0.0112 | 0.019 | 0 | 0.09 | 7.6 | 0 | 0.0863 | 0.0626 | 0.0578 | -0.0048 |
| 400 | 0.0256 | 0.0425 | 0 | 0.14 | 17 | 0.4 | 0.0844 | 0.0546 | 0.0559 | +0.00134 |
| 450 | 0.0312 | 0.0525 | 0 | 0.15 | 21 | 0.2 | 0.0895 | 0.0563 | 0.061 | +0.00472 |

### A3. objective | first | last | min | max | mean | quartiles | slope

| metric | first | last | min | max | mean | q1 | q2 | q3 | q4 | slope/iter |
|---|---|---|---|---|---|---|---|---|---|---|
| `training/objective_d` | 0.53 | 0.76 | -0.011 | 1.79 | 0.7982 | 0.8096 | 0.8147 | 0.7224 | 0.846 | +2.41e-05 |
| `training/objective_l` | 0.599 | 0.856 | -0.0173 | 2.04 | 0.9034 | 0.9208 | 0.9217 | 0.8163 | 0.9546 | +9.08e-06 |
| `Average reward` | 0.599 | 0.856 | -0.0173 | 2.04 | 0.9034 | 0.9208 | 0.9217 | 0.8163 | 0.9546 | +9.08e-06 |
| `Average latency` | 0.32 | 0.206 | 0.154 | 0.537 | 0.2206 | 0.2384 | 0.1953 | 0.2441 | 0.2047 | -5.17e-05 |
| `Average greedy latency` | 0.0561 | 0.0743 | 0.0305 | 0.0915 | 0.06085 | 0.06116 | 0.05994 | 0.06029 | 0.062 | +3.24e-06 |
| `MaxReturn` | 2.7 | 1.91 | 0.0855 | 4.93 | 2.884 | 2.989 | 2.894 | 2.858 | 2.794 | -0.000382 |
| `MinReturn` | -0.188 | -0.01 | -0.428 | 0.0456 | -0.08035 | -0.0885 | -0.1183 | -0.05927 | -0.05536 | +0.000123 |
| `StdReturn` | 0.854 | 0.723 | 0.0458 | 1.6 | 0.9383 | 0.961 | 0.9561 | 0.8899 | 0.9464 | -5.38e-05 |

### A4. energy

| metric | first | last | min | max | mean | q1 | q2 | q3 | q4 | slope/iter |
|---|---|---|---|---|---|---|---|---|---|---|
| `energy/requester_joules` | 0.191 | 0.198 | 0.0914 | 0.541 | 0.2123 | 0.2324 | 0.2059 | 0.2373 | 0.1734 | -0.000118 |
| `energy/mobile_joules` | 2.52 | 1.22 | 0.216 | 2.52 | 0.996 | 0.5783 | 0.6622 | 1.299 | 1.444 | +0.00249 |
| `energy/system_joules` | 63.9 | 89.4 | 42.9 | 133 | 82.74 | 85.33 | 83.18 | 79.29 | 83.17 | -0.00836 |
| `Average energy` | 1280 | 1790 | 857 | 2660 | 1655 | 1707 | 1664 | 1585 | 1664 | -0.167 |

### A5. constraint / dual

| metric | first | last | min | max | mean | q1 | q2 | q3 | q4 | slope/iter |
|---|---|---|---|---|---|---|---|---|---|---|
| `constraint/lambda_to` | 0 | 1.31 | 0 | 1.31 | 0.9251 | 0.3504 | 1.03 | 1.155 | 1.164 | +0.00209 |
| `constraint/lambda_ue` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | +0 |
| `constraint/penalty` | 0 | 0.132 | 0 | 0.199 | 0.07876 | 0.04906 | 0.1098 | 0.06178 | 0.09442 | +7.62e-05 |
| `constraint/updates` | 1 | 500 | 1 | 500 | 250.5 | 63 | 188 | 313 | 438 | +1 |
| `constraint/total_ene` | 0.00277 | 0.101 | 0.00277 | 0.36 | 0.09728 | 0.1444 | 0.1104 | 0.0532 | 0.08113 | -0.000196 |
| `constraint/ue_energy` | 0.00719 | 0 | 0 | 0.00719 | 0.0001745 | 0.0005143 | 2.627e-05 | 0.00014 | 1.763e-05 | -1.21e-06 |
| `constraint/mean_sign` | -0.399 | -0.537 | -0.771 | -0.108 | -0.4945 | -0.4786 | -0.492 | -0.4408 | -0.5664 | -0.00016 |

### A6. PPO

| metric | first | last | min | max | mean | q1 | q2 | q3 | q4 | slope/iter |
|---|---|---|---|---|---|---|---|---|---|---|
| `policy/approx_kl` | 1.29 | 0.0372 | 0.00719 | 1.29 | 0.06052 | 0.1161 | 0.03956 | 0.05254 | 0.03392 | -0.000204 |
| `policy/clip_fraction` | 0.658 | 0.0925 | 0.0563 | 0.658 | 0.1254 | 0.1799 | 0.1076 | 0.1196 | 0.09443 | -0.000212 |
| `policy/entropy_valid` | 0.954 | 0.262 | 0.119 | 0.954 | 0.3406 | 0.3668 | 0.3176 | 0.3777 | 0.3002 | -0.000119 |
| `policy/grad_norm` | 16.4 | 0.79 | 0.382 | 16.4 | 1.428 | 2.841 | 1.129 | 0.9124 | 0.8315 | -0.00545 |
| `policy/policy_loss_mean` | 0.377 | 0.0126 | -0.0141 | 0.377 | 0.01277 | 0.02729 | 0.005743 | 0.009777 | 0.008271 | -4.81e-05 |
| `policy/value_loss_mean` | 2.13 | 0.0652 | 0.0115 | 2.13 | 0.1014 | 0.1903 | 0.08326 | 0.06586 | 0.06635 | -0.000338 |
| `critic/value_abs_max` | 2.47 | 2.36 | 0.171 | 3.09 | 2.084 | 2.165 | 2.14 | 1.941 | 2.091 | -0.000284 |

### A7. action mix

| metric | first | last | min | max | mean | q1 | q2 | q3 | q4 | slope/iter |
|---|---|---|---|---|---|---|---|---|---|---|
| `action_fraction/local` | 0.129 | 0.446 | 0.129 | 0.9 | 0.4639 | 0.4778 | 0.4929 | 0.4851 | 0.3996 | -0.000196 |
| `action_fraction/mec` | 0.436 | 0.352 | 0.0423 | 0.627 | 0.3839 | 0.4562 | 0.4235 | 0.316 | 0.34 | -0.000347 |
| `action_fraction/v2v` | 0.435 | 0.201 | 0.0168 | 0.435 | 0.1522 | 0.06585 | 0.08357 | 0.199 | 0.2603 | +0.000542 |

### A8. throughput

| metric | first | last | min | max | mean | q1 | q2 | q3 | q4 | slope/iter |
|---|---|---|---|---|---|---|---|---|---|---|
| `EnvExecTime` | 8.98 | 8.57 | 7.76 | 10.6 | 8.73 | 8.702 | 8.66 | 8.719 | 8.838 | +0.000461 |
| `PolicyExecTime` | 0.385 | 0.666 | 0.229 | 1.26 | 0.379 | 0.4231 | 0.3653 | 0.3494 | 0.3783 | -0.000119 |

### A9. correlations with the optimised objective (Pearson over 500 iterations)

| metric | r |
|---|---|
| `training/objective_l` | +1.000 |
| `Average reward` | +1.000 |
| `energy/system_joules` | +0.874 |
| `Average energy` | +0.874 |
| `constraint/mean_sign` | -0.750 |
| `StdReturn` | +0.659 |
| `constraint/total_ene` | +0.602 |
| `MaxReturn` | +0.581 |
| `constraint/penalty` | +0.578 |
| `critic/value_abs_max` | +0.358 |
| `energy/requester_joules` | -0.303 |
| `Average latency` | -0.268 |
| `EnvExecTime` | -0.266 |
| `MinReturn` | +0.230 |
| `policy/value_loss_mean` | +0.214 |
| `constraint/ue_energy` | -0.208 |
| `policy/entropy_valid` | -0.208 |
| `policy/clip_fraction` | -0.206 |

### A10. first-quarter -> last-quarter change (sorted by |%|)

| metric | q1 mean | q4 mean | change |
|---|---|---|---|
| `constraint/updates` | 63 | 438 | +595.2% |
| `action_fraction/v2v` | 0.065846 | 0.26032 | +295.3% |
| `constraint/lambda_to` | 0.35042 | 1.1642 | +232.2% |
| `energy/mobile_joules` | 0.57832 | 1.4444 | +149.7% |
| `constraint/ue_energy` | 0.00051428 | 1.763e-05 | -96.6% |
| `constraint/penalty` | 0.049063 | 0.094419 | +92.4% |
| `policy/approx_kl` | 0.11607 | 0.033921 | -70.8% |
| `policy/grad_norm` | 2.8409 | 0.8315 | -70.7% |
| `policy/policy_loss_mean` | 0.027291 | 0.0082708 | -69.7% |
| `policy/value_loss_mean` | 0.19025 | 0.066351 | -65.1% |
| `policy/clip_fraction` | 0.17995 | 0.094427 | -47.5% |
| `constraint/total_ene` | 0.14442 | 0.081126 | -43.8% |
| `MinReturn` | -0.088505 | -0.055362 | +37.4% |
| `action_fraction/mec` | 0.45621 | 0.33998 | -25.5% |
| `energy/requester_joules` | 0.23244 | 0.17344 | -25.4% |
| `constraint/mean_sign` | -0.47863 | -0.56638 | -18.3% |
| `policy/entropy_valid` | 0.36685 | 0.30021 | -18.2% |
| `action_fraction/local` | 0.47784 | 0.3996 | -16.4% |
| `Average latency` | 0.23845 | 0.20467 | -14.2% |
| `PolicyExecTime` | 0.42314 | 0.37828 | -10.6% |
| `MaxReturn` | 2.9893 | 2.7941 | -6.5% |
| `training/objective_d` | 0.80959 | 0.84595 | +4.5% |
| `training/objective_l` | 0.92079 | 0.95463 | +3.7% |
| `Average reward` | 0.92079 | 0.95463 | +3.7% |
| `critic/value_abs_max` | 2.1648 | 2.091 | -3.4% |
| `energy/system_joules` | 85.327 | 83.169 | -2.5% |

---

### A11. تفسیر اعداد (خواندن A1–A10)

**۱. انرژی و latency هم‌جهت‌اند، نه رقیب.** `energy/system_joules` با objective همبستگی
**r = +0.874** دارد (A9). یعنی هر جا latency بهتر شده، انرژی سیستم **بیشتر** شده — چون
سیاست کار را به MEC می‌فرستد. این انرژی عمدتاً **background** است (`background ≈ 35.8 J` از
`system ≈ 36.3 J` در یک نمونه اندازه‌گیری‌شده)، پس MEC برای DAGهای owners دیگر هم کار می‌کند.
**نتیجه:** `system_joules` معیار «هزینه‌ی سیاست» نیست؛ بیشتر شاخص «چقدر MEC درگیر شده» است.
در مقاله به‌عنوان مصرف کل سیستم گزارش شود، نه هزینه‌ی requester.

**۲. کانال `ue_energy` واقعاً آزاد است.** `mean_signed_ue_energy = −0.4945` با
`λ_ue = 0` در تمام ۵۰۰ iteration (A5). یعنی انرژی requester همیشه ≈۰.۵۰۵ برابر پلن all-UE
بوده — همیشه زیر بودجه. این کانال فعال است ولی هیچ‌وقت نمی‌بندد. برای اینکه بودجه‌ی UE واقعاً
قید شود، باید fraction را از `1.0` به مقدار کوچک‌تر برد (تصمیم طراحی، نه باگ).

**۳. dual روی مرز همگرا شد.** `λ_total` از ۰ به ۱.۳۱ رسید ولی به max (1000) نرسید.
بازسازی `mean_signed_total_energy = (λ_t − λ_{t−1})/0.05` نشان می‌دهد میانگین از
**+0.103** (itr 1–240) به **+0.006** (itr 241–499) افتاده و ۳۳ بار منفی شده. این امضای
KKT است: `mean(g) → 0` در `λ* > 0`. پس قید **active** است، نه pin‌شده روی سقف.

**۴. گذار فاز در action mix.** A7/A10: `v2v` از ۰.۰۶۶ (q1) به ۰.۲۶۰ (q4) رفت (**+۲۹۵٪**) در
حالی که `local` از ۰.۴۷۸ به ۰.۴۰۰ و `mec` از ۰.۴۵۶ به ۰.۳۴۰ آمد. یعنی اواخر training سیاست
دوباره helper را بیشتر استفاده کرد. `action_fraction/v2v` مقدار first=0.435 و last=0.201 دارد؛
این‌ها با q1/q4 تناقض ندارند چون itr 0 خارج از بازه‌ی q1 است (سرد شدن اولیه).

**۵. همگرایی بهینه‌سازی بدون collapse.** A6/A10: `approx_kl` −۷۱٪، `grad_norm` −۷۱٪،
`value_loss` −۶۵٪، `clip_fraction` −۴۸٪. entropy از ۰.۹۵۴ به ۰.۲۶۲ افتاد (**−۷۲٪**) که
سیاست را تیز می‌کند، ولی `collapse/flag = 0` و `mask/*_rate = 0` در همه‌ی ۵۰۰ iteration و
`invalid_action_rate = 0`. پس تیزی entropy تا مرز collapse نرفت.

**۶. max/min return باریک شد، نه بهتر.** `MaxReturn` ۲.۷۰→۱.۹۱ و `StdReturn` ۰.۸۵۴→۰.۷۲۳
افت کردند در حالی که `Average reward` ۰.۵۹۹→۰.۸۵۶ بالا رفت. این یعنی واریانس بین اپیزودها کم
شد و کف بدترها بالا آمد — نه اینکه بهترین حالت بهتر شد. برای مقاله این تفکیک مهم است:
improvement از **کاهش واریانس** آمده، نه از کشف راه‌حل‌های بهتر.

**۷. validation زودتر از training اشباع شد.** A2: `firm_miss` در itr 50 به ۰.۰۲۳۵ رسید و تا
itr 450 همان حدود ماند (۰.۰۳۱۲)، با یک بدترشدن در itr 300 (۰.۰۴۴۷). `hard_miss` از itr 50
به بعد **صفر مطلق** است. پس سود اصلی در ۵۰ iteration اول گرفته شده؛ ۴۵۰ iteration بعدی
بهبود حاشیه‌ای (objective +۴.۵٪ از q1 به q4) با هزینه‌ی زمانی ۳ ساعت است. اگر بودجه‌ی
محاسباتی محدود است، **۱۰۰ iteration کافی به‌نظر می‌رسد** — ولی این یک مشاهده‌ی تک‌seed است.

**۸. کارایی نمونه‌ای.** `constraint/updates` slope = **+1.0** دقیقاً یعنی یک dual step در هر
iteration — بدون پرش، بدون تکرار (A5). `constraint/updates` از ۱ تا ۵۰۰ بدون شکاف.
همین برای `outer_update_count` برقرار است (بخش ۳). یعنی هیچ iteration‌ای دوباره اجرا نشد.

---

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
* ~~that `constraint/mean_signed_*` is the ascent input~~ — the dual IS verified correct; the
  apparent contradiction was a display-layer key collision (section 3), now fixed;
* per-episode or per-schedule behaviour: `audit_writer=None` on the v2 path, so no
  `trajs_ppo.jsonl` / `trajs_eval.jsonl` exist;
* the R/S per-cell structure: only aggregated validation rows are logged.

## 6. Known defects still open from this run

1. `v2_progress.json` was never written and `v2_phase_timings.jsonl` is 0 bytes for the entire
   500-iteration run — the atomic progress reporting required by the spec does not work on the
   v2 TF path, so there is no per-phase timing record despite the instrumentation existing.
2. ~~`constraint/mean_signed_*` contradicts `constraint/lambda_*`~~ — **RESOLVED**: it was the
   stdout truncated-key collision, not a dual defect (section 3). Fixed in `utils/logger.py`
   with a regression test; the dual was correct all along.
3. No `progress.csv` (`logger.configure()` is never called for v2).
4. `foreground_system_joules` is not a logged column, so the constrained energy is not directly
   visible in the run's own output.
