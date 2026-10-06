# V2_PARAMETER_CATALOG — چه چیزی برای نتیجه‌گیری در دسترس است

مرجع: اجرای `500`-iteration روی `/root/v2-runs/final500_seed0_20261005T0400`.
همهٔ اعداد از artifact های خود اجرا استخراج شده‌اند.

> **هشدار مقدماتی.** جدول `stdout` کلیدهای بلندتر از ۲۳ کاراکتر را با
> `s[:20] + "..."` کوتاه می‌کند (`utils/logger.py::_truncate`). تا کامیت `7d176f9` دیکشنری
> چاپ با کلید **کوتاه‌شده** ساخته می‌شد، پس در هر گروه با پیشوند ۲۰ کاراکتری مشترک فقط
> کلید **آخر به ترتیب الفبا** زنده می‌ماند. یعنی در لاگ این اجرا **۲۵ کلید از ۱۰۲ کلید
> وجود ندارند** و بعضی ستون‌ها متعلق به کلید دیگری هستند. بخش ۵ فهرست دقیق را می‌دهد.

---

## 1. خلاصه — چه دسته‌هایی در دسترس است

| دسته | تعداد کلید | فرکانس | وضعیت |
|---|---|---|---|
| Objective / reward | 2 | هر iteration | ✅ کامل |
| Latency / return | 9 | هر iteration | ✅ کامل |
| Action mix | 3 | هر iteration | ✅ کامل |
| Energy (سه boundary) | 6 | هر iteration | ✅ کامل |
| Constraint / dual | 13 | هر iteration | ⚠️ بخشی از دست رفته |
| Constraint status | 6 | هر iteration | ✅ کامل |
| PPO / optimization | 9 | هر iteration | ✅ کامل |
| Mask / collapse | 4 | هر iteration | ⚠️ ساختاری صفر |
| Validation (R/S) | 20 | ۱۰ بار (itr 0,50,…,450) | ✅ کامل |
| Evaluation / protocol | 3 | ۱۰ بار | ✅ کامل |
| Checkpoint selection | 8 | ۱۰ بار | ⚠️ برخورد کلید |
| Protocol / scalars | 18 | هر iteration | ✅ کامل |
| **جمع** | **۱۰۲** | | |

به‌علاوه `run.json` بیست‌ودو کلید در `evidence` دارد (§4).

---

## 2. Objective و reward

| کلید (log) | معنی | نوع |
|---|---|---|
| `training/objective_discounted_return` | معیار اصلی PPO (با `SHAPING_DISCOUNT=0.99`) | سود |
| `training/objective_legacy_undiscounted_sum` | مجموع بدون تخفیف — **هرگز معیار انتخاب نیست** | سود |
| `Average reward,` | برابر `objective_legacy` | سود |
| `AverageDiscountedReturn` / `AverageReturn` | همان دو مقدار | سود |
| `MaxReturn` / `MinReturn` / `StdReturn` | پراکندگی بین اپیزودها | توزیع |

## 3. Latency، throughput و بنچمارک

| کلید | معنی |
|---|---|
| `Average latency` | latency نرمال‌شدهٔ training rollout |
| `Average greedy latency` | latency پلن greedy روی همان world |
| `EnvExecTime` | ثانیهٔ شبیه‌ساز در هر iteration |
| `PolicyExecTime` | ثانیهٔ شبکهٔ عصبی در هر iteration |
| `NumTrajs` | تعداد trajectory (۲۰۰) |
| `Itr` | شمارهٔ iteration |

## 4. انرژی — سه boundary + scope

| کلید | boundary | نوع |
|---|---|---|
| `energy/requester_joules` | **requester** (foreground UE) | J |
| `energy/mobile_joules` | **mobile** (requester + helpers) | J |
| `energy/system_joules` | **system** (کل world، شامل background) | J |
| `energy/primary_joules` | برابر `system` چون `primary_scope=system` | J |
| `energy/primary_scope` | ثابت `system` | رشته |
| `Average energy,` | معیار **legacy MOBILE** — صریحاً برچسب‌دار | J |
| `energy/average_energ...` | برچسب scope برای معیار legacy (`mobile`) | رشته |

**در `run.json`:** `last_rollout.{system,requester,mobile,background}_joules` و
`unmodeled_components`.

**مهم:** `system_joules` **whole-world** است؛ ~۹۹٪ آن background است (اندازه‌گیری‌شده).
`requester_joules` **foreground-only** است. این دو کمیت متفاوت‌اند و نباید مخلوط شوند.

## 5. Constraint و dual — و آنچه از دست رفته

### 5.1 کلیدهایی که واقعاً در لاگ هستند

| برچسب چاپ‌شده | مقدار واقعی |
|---|---|
| `constraint/lambda_to...` | `constraint/lambda_total_energy` ✅ |
| `constraint/lambda_ue...` | `constraint/lambda_ue_energy` ✅ |
| `constraint/total_ene...` | **`constraint/total_energy_violation`** ⚠️ |
| `constraint/ue_energy...` | **`constraint/ue_energy_violation`** ⚠️ |
| `constraint/mean_sign...` | **`constraint/mean_signed_ue_energy`** ⚠️ |
| `constraint/total_vio...` | `constraint/total_violation` ✅ |
| `constraint/penalty` | `constraint/penalty` ✅ |
| `constraint/penalty_a...` | `constraint/penalty_applied` ✅ |
| `constraint/updates` / `buffer` / `enabled` | ✅ |
| `constraint/batch_siz...` | `batch_size_after_reset_ue_energy` ⚠️ |
| `constraint/lambda_br...` | `constraint/lambda_broadcast_targets` ✅ |
| `constraint_status/{to,ue,de,he,me,v2}...` | ۶ کانال ✅ |

### 5.2 کلیدهایی که **در لاگ نیستند** (باگ truncation، رفع‌شده در `7d176f9`)

```
constraint/total_energy_signed        constraint/ue_energy_signed
constraint/total_energy_raw           constraint/ue_energy_raw
constraint/total_energy_budget        constraint/ue_energy_budget
constraint/total_energy_scale         constraint/ue_energy_scale
constraint/total_energy_lambda        constraint/ue_energy_lambda
constraint/mean_signed_total_energy
constraint/batch_size_after_reset_total_energy
checkpoint_selection_metric_name      checkpoint_selection_scalar
checkpoint_selection_source           checkpoint_selection_uses_logged_objective
checkpoint_is_best_val_lexicographic
validation/objective_discounted_return_{k0,k3}
validation/objective_legacy_undiscounted_sum_{k0,k3}
```

**پیامد:** M، B، S، g و **λ فعال در هر rollout** برای هیچ‌کدام از دو کانال قابل استخراج
نیست. تنها منبع قابل استناد `run.json` و `ckpt/v2_resume_state.json` است:
```
λ_total_energy = 1.305534390180372    λ_ue_energy = 0.0    updates = 500
```

### 5.3 `run.json` → `evidence.penalty`
```
episodes                       = 100000   (500 × 200)
episodes_with_nonzero_penalty  =  60532   (60.5%)
penalty_sum                    = 7876.95
steps_with_penalty             =    489 / 500   (تعریف: iteration، نه step)
signed_rows_observed           = 100000
```

## 6. Validation (R/S) — ۱۰ نقطه، کامل

| کلید | معنی | نوع |
|---|---|---|
| `validation/firm_task_miss_rate` | نرخ از دست رفتن deadline firm | ٪ |
| `validation/high_task_miss_rate` | تسک‌های HIGH دیرشده | ٪ |
| `validation/medium_task_miss_rate` | تسک‌های MEDIUM دیرشده | ٪ |
| `validation/graph_hard_miss` | نقض hard deadline در سطح graph | ٪ |
| `validation/graph_high_miss` | miss سطح graph برای HIGH | ٪ |
| `validation/graph_medium_miss` | miss سطح graph برای MEDIUM | ٪ |
| `validation/n_tardy_high_tasks` | **تعداد** تسک HIGH دیرشده | شمارش |
| `validation/n_tardy_medium_tasks` | تعداد تسک MEDIUM دیرشده | شمارش |
| `validation/n_high_tasks` | مخرج HIGH (۴۰۰) | شمارش |
| `validation/n_medium_tasks` | مخرج MEDIUM (۲۸۰) | شمارش |
| `validation/query_mean_latency` | latency روی query set | s |
| `validation/k0_regret` | رگرت بدون انطباق | s |
| `validation/k3_regret` | رگرت با ۳ مرحله انطباق | s |
| `validation/oracle_ca...` | ظرفیت اوراکل پنل | s |
| `validation/query_graphs` | ۴۰ | شمارش |
| `validation/rollouts_k0` | ۲۰۰ | شمارش |
| `validation/mc_policy_...` | نقض سیاست Monte-Carlo | شمارش |
| `validation/mode_swit...` | نرخ تغییر mode | ٪ |
| `validation/hi_mode_r...` | نرخ mode HIGH | ٪ |
| `validation/objective...` | معیار objective | — |

## 7. Evaluation و protocol

| کلید | معنی |
|---|---|
| `evaluation/replicates` | ۵ (SELECT_REPLICATES) |
| `evaluation/protocol_...` | ۱ (protocol فعال) |
| `evaluation/paired_k3...` | Δ(k3−k0) زوجی |
| `validation_query_composite_objective` | معیار ترکیبی k3 |
| `validation_query_composite_objective_k0` | همان برای k0 |
| `validation_query_mean_latency_k0` / `_k3` | latency k0/k3 |
| `correctness/core_scr...` | امضای core |
| `correctness/core_unc...` | core بدون تغییر (۱ = تأیید) |
| `correctness/lambda_b...` | تعداد هدف broadcast |
| `correctness/penalty_...` | شمارندهٔ پنالتی |

## 8. PPO / optimization

| کلید | معنی | سلامت |
|---|---|---|
| `policy/approx_kl` | فاصلهٔ KL سیاست | 1.29 → 0.037 |
| `policy/clip_fraction` | نرخ clip شدن | 0.658 → 0.0925 |
| `policy/entropy_valid` | آنتروپی سیاست | 0.954 → 0.262 |
| `policy/grad_norm` | نرم گرادیان | 16.4 → 0.79 |
| `policy/policy_loss_mean` | loss سیاست | 0.377 → 0.0126 |
| `policy/value_loss_mean` | loss value function | 2.13 → 0.0652 |
| `policy/invalid_action_rate` | اکشن نامعتبر | ۰ (همیشه) |
| `policy/argmax_masked_rate` | argmax روی ماسک | ۰ (همیشه) |
| `critic/value_abs_max` | بیشینهٔ |value| | 2.47 → 2.36 |
| `entropy_coefficient` | ضریب آنتروپی | ۰ |
| `kl_bc_mean` | KL نسبت به BC | — |
| `value_clip_epsilon` | ۰.۲ | — |

## 9. Action mix و mask

| کلید | معنی |
|---|---|
| `action_fraction/local` | سهم تصمیم‌های UE |
| `action_fraction/mec` | سهم تصمیم‌های MEC |
| `action_fraction/v2v` | سهم تصمیم‌های HELPER |
| `mask/active_rate` | نرخ ماسک فعال (۰ — ساختاری) |
| `mask/all_invalid_rate` | (۰) |
| `mask/forced_rate` | (۰) |
| `collapse/flag` | هشدار فروپاشی (۰ در ۵۰۰) |

## 10. Hyperparameter و protocol (ثابت، برای اصالت)

```
seed=0 | k_steps=3 | inner_learning_rate=5e-4 | outer_learning_rate=5e-4
meta_batch_size=10 | ppo_batch_size_trajectories=20 | support_trajectories=20
support_graphs_per_meta_task=20 | query_graph_count=80 | value_clip_epsilon=0.2
inner_update_mode=publication | outer_update_method=mrlco_first_order_meta
hyperparameter_provenance=fixed_literature_derived
split_role=meta_train_support | split_version=MARGO-SPLIT-v1
```

## 11. `run.json` — کلیدهای `evidence`

```
checkpoint          {path, promotable, restored_from, resume_state_path, status, was_resume}
config              کل کانفیگ resolved (۱۹ کلید)
constraints         {names, lambdas, updates, spec_sha256, status}
device              {requested, cuda_visible_devices}
error               None = موفق
iteration_measurements  {mean_system_joules, mean_requester_joules, mean_violation}
iterations_requested / started / completed / executed
last_completed_iteration / next_iteration
outer_updates_completed / validations_completed
lambda_broadcast    {targets, values}         ← ۱۱ = ۱ env + ۱۰ clone
penalty             {episodes, episodes_with_nonzero_penalty, penalty_sum,
                     steps_with_penalty, signed_rows_observed}
last_rollout        {makespan_s, world_makespan_s, system/requester/mobile/
                     background_joules, constraint_penalty, constraint_violations,
                     constraint_signed, n_tasks, n_high/medium/helper/mec_tasks,
                     world_id, world_fingerprint_sha256, unmodeled_components,
                     energy_constraint}
measured_env        مسیر telemetry
sampler_counters    شمارنده‌های sampler
trained_core_norms  {n_tensors: 30, total_norm, all_finite}
progress            {path, exists}
wall_seconds        13055.16
```

## 12. چه چیزی **نیست** — و چطور اضافه می‌شود

| کمبود | چرا | راه‌حل |
|---|---|---|
| per-episode breakdown | `audit_writer=None` در v2 stack | وصل کردن `audit_writer` → `trajs_ppo.jsonl` / `trajs_eval.jsonl` |
| تفکیک foreground/background انرژی | `foreground_system_joules` در record هست ولی جزو `CSV_COLUMNS` نیست | افزودن به `CSV_COLUMNS` |
| `progress.csv` | `logger.configure()` هرگز صدا زده نمی‌شود | صدا زدن `logger.configure(dir=...)` در `run_tf.py` |
| R/S خام per-cell | evaluator خروجی indexed ندارد | افزودن dump indexed |
| phase timings | `v2_phase_timings.jsonl` صفر بایت (باگ حل‌نشده) | دیباگ `_iteration_completed` |
| baseline در لاگ آموزش | `all_MEC` هرگز لاگ نمی‌شود | افزودن به loop |
| M/B/S/g/λ per channel | برخورد کلید (رفع شد در `7d176f9`) | اجرای مجدد — یا خواندن از `run.json` |
| per-node observation | کانال‌های v2 per-graph تکرار می‌شوند | افزودن per-node channels |

## 13. غیر از این اجرا — داده‌های ارزیابی جداگانه

از checkpoint های همین اجرا (`eval_trained.json`, `eval_final.json`, `eval_ksweep.json`):

| مورد | مقدار |
|---|---|
| 40 query graph × 5 replicate | paired |
| سیاست k=0 (itr 350 / 499) | 0.091083 / 0.080600 s |
| سیاست k=1,2,3,5 | 0.086529 / 0.085048 / 0.085807 / 0.090450 s |
| `all_UE` | 0.279134 |
| `all_MEC` | 0.026452 |
| `all_HELPER` | 0.177473 |
| uniform-random | 0.172051 |
| `greedy_coordinate_descent_MC` | 0.035061 |
| `heft_reference_v2` | 0.032665 |
| `heft_nominal_s` | 0.029881 |
| `candidate_oracle_s` | 0.026115 |
| برد/باخت per-graph در برابر all_MEC | **۰ / ۴۰** |
| نرخ نقض hard deadline | ۰ |
| نقض MC policy | ۰ |
