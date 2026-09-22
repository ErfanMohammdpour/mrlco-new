# ②B — Objective، کانال‌های constraint، انتخاب چک‌پوینت lexicographic، و mask API

**تاریخ:** ۲۰۲۶-۰۹-۲۱ · **وضعیت:** ②A.1 + ②B پیاده و تست‌شده · کل تست‌های غیر-TF: **۲۹۹ passed, 5 skipped**
**تصویب‌شده:** «②B را شروع کن؛ ولی قبلش earliest-consumer را primary نگذار. NO PPO/Lagrangian در ②B.»

---

## ۰. ②A.1 — دو اصلاحی که قبل از ②B خواستی

### ۰٫۱ مبنای ددلاین: all-consumers-ready (blocker علمی رفع شد)
```
F_ready(i) = max_{j ∈ succ(i)} delivery(i → j)        (برای sink: بازگشت به UE)
```
- `first_available` (زودترین مصرف‌کننده) **فقط diagnostic** است و دیگر مبنای miss نیست.
- مبنای اصلی حالا `all_consumers_ready` است و در `ScheduleResult.deadline_basis` صریحاً ثبت می‌شود.
- `last_delivery` به‌عنوان alias سازگاری حفظ شد.
- **تست مثال خودت:** `A → B` (روی MEC، آماده در `finish`) و `A → C` (روی UE، آماده بعد از دانلینک کند) با ددلاین بین آن دو: با معیار earliest **pass** می‌شد، با معیار all-consumers **MISS** است. (`test_deadline_between_the_two_arrivals_is_a_miss`) — این همان چیزی است که mask و دیتاست آینده به آن وابسته‌اند، پس همین‌جا قفل شد.

### ۰٫۲ جدا کردن criticality از weight
| قبل | بعد |
|---|---|
| `criticality: float` (عملاً وزن penalty) | `criticality_class: "low"\|"medium"\|"high"` (**مفهوم داده**) + `tardiness_weight: float` (**ضریب penalty**) |

تست شده که `criticality_class` **هیچ اثر عددی ندارد** و فقط `tardiness_weight` در soft tardiness ضرب می‌شود. مپینگ آیندهٔ generator (`HIGH→hard`, `MEDIUM→firm`, `LOW→soft`) بدون قاطی‌کردن این دو ممکن است.

قواعد ددلاین هم صریحاً **diagnostic** علامت خوردند (`DIAGNOSTIC_ONLY = True` + docstring: «final benchmark از `D_G → EFT/LFT → subdeadlines` با feasibility-aware generation می‌آید») و rule `per_task_slack_of_avail` به `per_task_slack_of_ready` تغییر نام داد (+ `per_task_slack_of_first_avail` فقط برای مقایسهٔ تشخیصی).

---

## ۱. ②B-1 — Objective و کانال‌های cost (فقط محاسبه و log)

```
L_norm      = L / L_ref                              L_ref = مرجع ثابت per-graph (پیش‌فرض all-UE)
T_soft_norm = Σ_{i∈S} w_i · [F_ready_i − d_i]⁺ / d_i
J           = L_norm + β_s · T_soft_norm

c_E = [ E_system / B_E − 1 ]⁺        (cap شده)
c_H = #hard_miss / max(1, N_H)
c_F = #firm_miss / max(1, N_F)       با بودجهٔ c_F ≤ ε_F
```
- **هیچ min-max کامپوزیت ۰٫۵/۰٫۵ در مسیر نیست** ⟹ J در latency یکنوا است و اشباع نمی‌کند.
- `ObjectiveSpec`: `latency_ref ∈ {all_ue, all_mec, fixed}` · `β_s` · `energy_budget_j` یا `energy_budget_frac_of_all_ue` · `ε_H`, `ε_F` · `cost_cap`.
- لاگ‌ها: `objective/J`, `objective/latency_norm`, `objective/soft_tardiness_norm`, `objective/c_E/c_H/c_F`, `objective/feasible`, `objective/total_violation`.
- **نه Lagrangian، نه gradient** — طبق تصویب فقط محاسبه و log.

فایل: `env/mec_offloaing_envs/scheduler/objective.py`

## ۲. ②B-2 — انتخاب چک‌پوینت lexicographic
```
feasible(π) := c_H ≤ ε_H  ∧  c_E = 0  ∧  c_F ≤ ε_F
π* = argmin_{π ∈ feasible} J(π)
```
- متریک جدید: `lexicographic_feasible_then_J` (نام در `SELECTION_METRIC_NAME`).
- اگر **هیچ** چک‌پوینتی شدنی نبود: برنده اعلام **نمی‌شود**؛ فقط کوچک‌ترین نقض نرمال‌شده به‌عنوان debug گزارش می‌شود و در پیام صریح نوشته می‌شود «NOT a scientific winner».
- تست کلیدی: چک‌پوینتی با `J` بهتر ولی نقض انرژی، به چک‌پوینت شدنی با `J` بدتر **می‌بازد** — و همین تست نشان می‌دهد چرا gate باید قبل از مقایسهٔ اسکالر بیاید.

سیم‌کشی: `spec/objective_selection.py` (آداپتور metric→PlanObjective) + `meta_trainer.py`:
- `build_frozen_primary_stack(..., objective_mode="off"|"log_only"|"lexicographic", objective_spec=...)`
- پیش‌فرض `off` ⟹ مسیر legacy و انتخاب قبلی دست‌نخورده.
- در `_run_validation` متریک قدیمی هنوز لاگ می‌شود (برای مقایسه) و برندهٔ lexicographic جداگانه: `checkpoint_is_best_val_lexicographic`.
- ⚠️ این بخش TF است و اینجا اجرا نمی‌شود؛ کامپایل و منطقش تست شده، ولی **روی kish باید با یک ران ۱-تکراری تأیید شود**.

## ۳. ②B-3 — mask API فقط (بدون lookahead)
فایل: `env/mec_offloaing_envs/scheduler/feasibility.py`

```python
feasible_actions(ctx, resources, actions=(0,1,2), workload_bytes=...) -> list[ActionFeasibility]
mask_vector(feas) -> [bool, bool, bool]
optimistic_lower_bound_ready(action, ctx, resources, ...) -> float
```
**قاعدهٔ صدا بودن (soundness) که تأکید کردی:**
| حالت | نتیجه |
|---|---|
| optimistic LB > deadline | **mask مجاز است** (اثبات ناامکان‌شدنی) |
| heuristic نتوانست پلن شدنی بسازد | **mask ممنوع** — دلیل ثبت می‌شود: `heuristic_only_not_used` |

LB خوش‌بینانه = `parents_ready + transfer_in + workload/rate + delivery` با فرض **رقابت صفر** (هیچ صف CPU/کانال). هر زمان‌بندی واقعی ≥ این کران است، پس `LB > d` اثبات است.
تست‌ها:
1. بی‌ددلاین ⇒ هیچ mask.
2. ددلاین ناممکن ⇒ هر سه action mask با دلیل `lb_exceeds_deadline`.
3. **آزمون صدا بودن علیه شبیه‌ساز:** هر action ای که mask شده، در شبیه‌ساز واقعاً miss می‌شود.
4. **آزمون کران:** `LB ≤ availability` واقعی برای هر سه tier (با تلورانس float).
5. `heuristic_failure_must_not_mask(False) == "heuristic_only_not_used"`.

full lookahead (fastest-feasible suffix) عمداً پیاده **نشد** — کار ③ است.

## ۴. موارد باز و نکات مهم

1. **`B_E` نسبت به `E_ue` در مدل فیزیکی ضعیف است:** چون compute MEC مرتبه‌ها بزرگ‌تر از انرژی all-UE است، budget `frac_of_all_ue` خیلی زود شلیک می‌کند. پیشنهاد: `B_E` را از **پنل کاندید** بگیر (`min/max` روی {all-UE, greedy, 2-opt} × slack) یا مطلق. (این را حین تست کشف کردم؛ در ③ تصمیم بگیریم.)
2. **ددلاین روی دیتاست فعلی وجود ندارد** ⟹ کانال‌های `c_H/c_F/T_soft` صفر می‌مانند تا دیتاست deadline-aware ساخته شود؛ کد آماده است.
3. **صداقت دربارهٔ محدودهٔ تست:** همهٔ ماژول‌های جدید pure و تست‌شده‌اند؛ فقط سیم‌کشی TF (دو خط در `meta_trainer`) اینجا اجرا نشده و نیاز به تأیید ۱-تکراری روی kish دارد.
4. **نرخ رادیو دست نخورد** (طبق تصحیح خودت: ۱۰ MHz پهنای‌باند است نه throughput؛ `R = B·log2(1+SINR)` یا `R = B·η`). این به فاز مستقل **RADIO_MODEL_V1** (پس از ③) منتقل شد و در آنجا `bandwidth_hz` / `spectral_efficiency` / SINR جدا می‌شوند.

## ۵. ترتیب تأییدشده از اینجا
`③` optimistic deadline LB + fastest-feasible suffix + mask واقعی + potential اصلاح‌شده → `③.5` RADIO_MODEL_V1 → `④` spec cleanup → `⑤` effective-rank → `⑥` decoder 256 → دیتاست جدید → Constrained PPO + meta-RL.

---

# پیوست — رفع دو blocker قبل از ③ (۲۰۲۶-۰۹-۲۱)

کل تست‌های غیر-TF: **۳۱۱ passed, 5 skipped**.

## B1. LB حالا واقعاً optimistic و sound است
`feasibility.py` بازنویسی شد به قرارداد تأییدشده:

```
ParentInput(source_location, source_ready_s = parent COMPUTE finish, bytes)
A_{p→i}(a) = t_p^source + transfer_lb(loc_p, loc_a, bytes_p)
S_i^LB(a) = max_p A_{p→i}(a)          # انتقال‌ها موازی، نه جمع
F_i^LB(a) = S_i^LB(a) + C_i / f_a
R_i^LB(a) = F_i^LB(a)                                  اگر non-sink
          = F_i^LB(a) + transfer_lb(loc_a, UE, out)    اگر sink
```
سه اصلاح مشخص:
1. **بدون delivery فانتوم برای non-sink.** موفق‌ها می‌توانند co-located باشند (مثلاً همه روی MEC ⇒ دانلینک صفر)، پس هر مقدار مثبت = over-estimate ⇒ false mask. تست: `test_non_sink_has_no_delivery_term`.
2. **parent-by-parent و موازی.** `parents_ready_s` قبلی و مجموع bytes حذف شد. تست: دو parent با bytes بزرگ همان کران یک parent را می‌دهند و از حالت سری کمترند (`test_parents_transfer_in_parallel_not_serialized`).
3. **`source_ready_s` = finish محاسبهٔ parent، نه all-consumers-ready آن.** تست: انتقال MEC→MEC صفر است و کران دقیقاً finish پدر می‌شود.

`transfer_lb` = **جمع** هاپ‌های مسیر (زمان واقعی بدون صف) ⇒ sound و tight تر از نادیده‌گرفتن هاپ‌ها. تست دوهاپ MEC→HELPER (MEC_DL + V2V) گذاشته شد.

**تست صدا بودن قوی‌تر** (دقیقاً همان گزاره‌ای که خواستی): برای هر action ماسک‌شده، **همهٔ** جایگذاری‌های ممکن successorها (۳×۳) شبیه‌سازی می‌شود و در هیچ‌کدام مقدار `all_consumers_ready` زیر ددلاین نمی‌آید — یعنی mask نسبت به کل خانوادهٔ ادامه‌ها معتبر است، نه فقط یک suffix آزمایشی. به‌علاوه `test_lower_bound_never_exceeds_sampled_schedules` تضمین می‌کند `LB ≤ availability` واقعی برای همهٔ نمونه‌ها.

## B2. validation حالا per-graph است، نه ratio-of-means
`spec/objective_selection.py` بازنویسی شد:
```
J_val            = mean_g J_g
soft_tard_norm   = mean_g soft_tard_norm_g
c_H              = Σ hard_miss_g / Σ N_hard,g          (باید 0)
c_F              = Σ firm_miss_g / Σ N_firm,g          (≤ ε_F)
c_E_max          = max_g c_E,g                         (قید per-episode سخت)
energy_viol_rate = mean_g 1[c_E,g > 0]                 (باید 0)
```
- **آداپتور ratio-of-means حذف شد** (`objective_from_validation` / `summarize_references` پاک شدند). تنها ورودی مجاز `objective_from_plans([(ScheduleResult, ReferenceRanges), ...])` است.
- تست `test_grid_mean_differs_from_ratio_of_means` بایاس را قفل می‌کند.
- **تست سناریوی خودت:** گراف A بالای بودجه، گراف B خیلی زیر بودجه ⇒ `mean(E) ≤ mean(B)` (میانگین‌ها سالم به‌نظر می‌رسند) ولی `energy_violation_rate = 0.5`، `c_E_max > 0`، و `feasible = False`.
- **counts از n_graphs بازسازی نمی‌شوند:** تست با ۲ گراف × ۲ تسک hard ⇒ `hard_task_total = 4` (نه 2).
- سیم‌کشی TF به per-graph منتقل شد: اگر evaluator کلید `validation_per_graph_plans` را ندهد، `objective/unavailable=1` لاگ می‌شود و در حالت `lexicographic` **هیچ برنده‌ای انتخاب نمی‌شود** (fallback خاموش به کامپوزیت قدیمی ممنوع).

## B3. hardening ها
- `ObjectiveSpec`: **XOR** بین `energy_budget_j` و `energy_budget_frac_of_all_ue`؛ بودجه باید **> ۰** باشد (صفر ⇒ خطا).
- `hard_miss_epsilon` در primary **روی صفر freeze** شد: هر مقدار ≠۰ خطا می‌دهد با پیام صریح؛ شکل احتمالاتی فقط با نام دیگری (`chance_hard_epsilon`) که فعلاً `NotImplementedError` است.
- `c_E_raw` (نقض uncapped) در کنار `c_E` (cap شده) ذخیره و لاگ می‌شود: cap فقط برای PPO، نه برای گزارش. تست: `c_E_raw > cost_cap` و `c_E == cost_cap`.

## B4. B_E از پنل کاندید (support برای ③)
`candidate_panel_budget(panel, rho)`:
```
E_min  = min_s E_s
E_fast = Eِ argmin_s T_s
B_E(ρ) = E_min + ρ·(E_fast − E_min),   ρ ∈ [0,1]
```
تست: نقاط انتهایی، یکنوایی در ρ، اعتبارسنجی. **این هنوز بودجهٔ نهایی نیست** — بودجهٔ مشترک energy+deadline بعداً همراه dataset generator از یک anchor schedule به‌طور هم‌زمان شدنی ساخته می‌شود.

## B5. ساختار ③ (قفل‌شده، پیاده‌سازی بعد)
```
current action
   ├── Sound optimistic DAG LB (تنها مرجع mask)
   │      LB > hard deadline ?  yes → MASK   ·   no → باز بماند
   └── Constructive fastest-feasible suffix (فقط potential/reward)
          found → Ĵ(s) از پروفایل جست‌وجوشده
          failed → NO MASK
```
- potential فقط objective اصلی را شکل می‌دهد: `Ĵ(s) = L̂(s)/L_ref + β_s·T̂_soft(s)` و `r_t = Ĵ(s_{t-1}) − γ·Ĵ(s_t)`؛ **energy و firm داخل J نمی‌روند** (کانال مستقل می‌مانند)؛ و در حالت terminal باید `Ĵ(s_T) = J_actual` برقرار باشد.
- ترتیب: ③ → RADIO_MODEL_V1 → **دوباره اجرای همهٔ تست‌های soundness/LB/suffix/Pareto ③** (چون تغییر rate رادیو مستقیماً deadline feasibility و potential را عوض می‌کند).
