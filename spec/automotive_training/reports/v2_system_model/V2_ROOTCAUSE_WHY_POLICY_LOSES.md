# V2_ROOTCAUSE — چرا سیاست آموزش‌دیده از یک باسلاین تک‌اکشنی می‌بازد

**سؤال:** مدل ۵۰۰-iteration با latency 0.0806 ثانیه در برابر `all_MEC` با 0.0299 ثانیه
باخت (۲.۶۹× در ارزیابی، ۳.۰۵× روی probe مستقیم من) و در ۰ از ۴۰ graph برد. چرا؟

**روش:** خواندن استاتیک کد در mirror (HEAD `49e5cb8`) + probe های pure-Python با همان
scheduler واقعی (بدون TF). هیچ فایلی تغییر نکرد. Docker محلی و joblib روی `kish-ai` موجود
نبود، پس بازرسی مستقیم وزن‌ها **UNVERIFIED** است.

---

## 0. اعداد مرجع (probe خودم، واقعی)

روی ۴۰ graph اعتبارسنجی، `REALIZATION_SEEDS[0]`, epoch 0, background 2, link moderate:

| policy | mean latency (s) | نسبت به all-MEC |
|---|---|---|
| `all_UE` | 0.279134 | ۱۰.۵۵× |
| `all_HELPER` | 0.177473 | ۶.۷۱× |
| **uniform-random** (۳ اکشن، ۴۰ نمونه) | **0.172051** | ۶.۵۰× |
| **سیاست آموزش‌دیده** (itr 499) | **0.080600** | **۳.۰۵×** |
| `greedy_coordinate_descent_MC` | 0.035061 | ۱.۳۳× |
| **`all_MEC`** | **0.026452** | **۱.۰۰×** |
| `candidate_oracle_s` (پنل) | 0.026115 | ۰.۹۹× |

### دو واقعیت تعیین‌کننده که کل مسئله را بازتعریف می‌کنند

**الف) سیاست واقعاً یاد گرفته — بد نیست، فقط بهینه نیست.** سیاست از random **۲.۱۳×**
بهتر است (0.1721 → 0.0806). پس gradient وجود دارد و کار می‌کند.

**ب) `all_MEC` تقریباً بهینه است و سقف بهبود ناچیز است.** روی ۴۰ graph:
```
all-MEC خودش oracle پنل است روی   24/40 graph
greedy_cd برنده است روی            16/40 graph
بهترین سود ممکن از mixing (میانگین) = 1.32%   (0.026452 → 0.026115)
بهترین سود ممکن (بدترین حالت)        = ~5%
برنامه‌های greedy خودشان 67% اکشن MEC دارند (539 از 800 تصمیم)
```
یعنی «همیشه MEC» عملاً بهینه است. **مدل باید خیلی دقیق باشد تا ۱.۳٪ را بگیرد —
و به‌جای آن ۲۰۵٪ بدتر است.** پس این یک شکست «کمی» نیست، یک شکست «کیفی» است.

---

## H1 — reward نمی‌تواند plan خوب را از بد تشخیص دهد
### VERDICT: **REFUTED**

`_telescoping` (spec/automotive_training/v2/env.py:586-602):
```python
previous = self._schedule_slot(slot, [0] * n, validate=False).makespan_s
for k in range(n):
    prefix = [int(a) for a in actions[:k + 1]] + [0] * (n - k - 1)
    current = self._schedule_slot(slot, prefix, validate=False).makespan_s
    rewards.append(-(current - previous) / l_scale)
    previous = current
```
مجموع دقیقاً telescoping می‌شود به `-(L_final - L_0)/D_G`، پس فرض ساختاری H1 **درست** است.

اما کمّیت آن **علیه** H1 است. اندازه‌گیری روی ۱۲ graph واقعی:
```
|total return| per episode = |L_final - L_0|/D_G :  min 0.295  max 1.805  mean 0.910
توکن‌های با reward غیرصفر:  min 15  max 17  از 20   (میانگین 79%)
توکن‌های صفر-سیگنال: 4.2 از 20
```
با اکشن هدف `all-MEC`، **۷۹٪ توکن‌ها reward غیرصفر می‌گیرند**. مسیر prefix-only
(`previous` از `[0]*n` شروع می‌شود) یعنی هر تصمیم `k` از مسیر all-UE منشعب می‌شود و اثرش
اندازه‌گیری می‌شود. پاداش dense است، نه sparse.

نسبت سیگنال به جریمه هم بد نیست: `penalty/|return| = 0.0788/0.798 = 0.099` (۱۰٪).

**نتیجه:** H1 مکانیزم را درست توصیف می‌کند ولی علت شکست نیست.

---

## H2 — آموزش روی میانگین ۱۰ توزیع، بدون هیچ تخصص‌سازی
### VERDICT: **SUPPORTED — علت غالب**

### H2.1 معماری: یک core مشترک، پارامتر هر task دور ریخته می‌شود
`policies/meta_seq2seq_policy.py:1037-1054` یک `Seq2SeqPolicy` جدا برای هر task می‌سازد،
و `async_parameters` (`:1096-1099`) با `assign_old_eq_new_tasks[i]` **کل** core را روی هر
۱۰ task کپی می‌کند:
```python
self.assign_old_eq_new_tasks.append(
    U.function([], [], updates=[tf.compat.v1.assign(oldv, newv)
                                for (oldv, newv) in
                                zipsame(self.meta_policies[i].get_variables(),
                                        self.core_policy.get_variables())]))
```
**هیچ task embedding یا per-task head وجود ندارد.** هویت یک task فقط وزن‌های adapted آن است —
و آن وزن‌ها هر outer step **دور ریخته می‌شوند**.

### H2.2 meta-objective فقط کیفیت «پس از انطباق» را می‌بیند
ترتیب در `meta_trainer.py` (تأییدشده با grep):
```
388  self.algo.sync_task_policies_from_core()      # core → هر ۱۰ task
397  paths      = self.sampler.obtain_samples(log=False)   # SUPPORT rollout
459  algo.UpdatePPOTarget(samples_data)            # ۳ گام inner PPO → taskها adapted می‌شوند
528  new_paths  = self.sampler.obtain_samples(log=True)    # QUERY rollout  ← با policyهای adapted
540  self.algo.UpdateMetaPolicy()                   # بیرونی
```
هیچ `sync_task_policies_from_core` بین ۴۵۹ و ۵۲۸ نیست. پس **core هیچ سیگنالی از رفتار
پیش-انطباق خودش نمی‌گیرد**؛ فقط از «چقدر انطباق کمک کرد». این Reptile-style است، نه MAML:
MAML نیاز دارد که loss روی `theta0` هم ارزیابی شود.

### H2.3 به‌روزرسانی بیرونی = میانگین جابه‌جایی‌های متناقض
`meta_algos/MRLCO.py:256-267`:
```python
grads = mean_pseudogradient(theta0, adapted, self.inner_lr, self.num_inner_grad_steps)
```
`spec/learning_ops.py:16-30`:
```python
scale = float(alpha) * float(k_steps)                      # 5e-4 * 3 = 1.5e-3
acc += (theta0 - theta_i[j]) / scale
grads.append(acc / float(n))                               # میانگین ساده روی ۱۰ task
```
با `max_grad_norm=0.5` در inner loop (`MRLCO.py:29`)، جابه‌جایی هر task اساساً **گرادیان
clip-شده** در نقطهٔ adapted است. میانگین ۱۰ گرادیان متناقض یک جهت ضعیف می‌دهد که هیچ
task به‌تنهایی نمی‌خواست. `meta_batch_size` روی ۱۰ **قفل** است (`MRLCO.py:41-42` خطا می‌دهد).

### H2.4 تنها مکانیزم تخصص‌سازی = ۳ گام inner
در آزمون، تنها چیزی که می‌تواند یک graph خاص را تشخیص دهد، ۳ گام PPO روی core است.
شواهد میدانی که این کافی نیست:
```
در خود run:  k3 regret 0.0610  vs  k0 regret 0.0563   ← k3 بدتر
ارزیابی من:  itr350  k3-k0 = -0.004524 s (3/5 برد)
             itr499  k3-k0 = +0.002151 s (1/5 برد)    ← علامت عوض می‌شود
```
### H2.5 شاهد رفتاری: ترکیب اکشن یک سازش است
از لاگ خود run (`action_fraction/*`):
```
              itr 0     itr 499     میانگین     آنچه oracle می‌خواهد
local          0.129     0.446      0.464       ~0.16  (235/800)
mec            0.436     0.352      0.384       ~0.67  (539/800)
v2v            0.435     0.201      0.152       ~0.03  (26/800)
```
سیاست MEC را از ۰.۴۳۶ به ۰.۳۵۲ **کم** کرد در حالی که می‌بایست به ~۰.۶۷ **زیاد** کند،
و local را به ۰.۴۴۶ رساند. این امضای یک core است که به سمت یک «سازش میانگین‌گیر»
کشیده شده — دقیقاً همان چیزی که میانگین‌گیری روی taskهای متناقض تولید می‌کند.

---

## H3 — فضای اکشن/decoder نمی‌تواند plan درست را بیان کند
### VERDICT: **REFUTED (به‌عنوان محدودیت بیان)**

`all-MEC` اکشن ثابت `[1]*20` است و کاملاً در فضای ۳^۲۰ وجود دارد. توزیع مشاهده‌شده
mec=0.352 نشان می‌دهد decoder می‌تواند و می‌کند اکشن MEC تولید کند؛ فقط جرم احتمال را
روی آن متمرکز نمی‌کند. decoder auto-regressive است (`decoder_targets` هدف،
`decoder_inputs` ورودی teacher-forced) با `memory_layer` و attention روی خروجی encoder.

**هیچ نشانه‌ای از emit یکنواخت برای همه graphها نیست:** سیگنچر ۷۰-بعدی هر ۴۰ graph
**یکتاست** (40/40 در دقت 1e-4)، و ۲۸ ستون ویژگی std>0.1 دارند. پس decoder ورودی
متمایزکننده دارد. این محدودیت **بیان** نیست، شکست **یادگیری/شرطی‌سازی** است.

---

## H4 — مشاهده اطلاعات لازم برای انتخاب را ندارد
### VERDICT: **REFUTED**

مشاهده واقعی را ساختم: `env.reset()` → تنسور `(40, 20, 109)`.

context گراف‌سطح (`context_fields.py`, ۳۰ فیلد) **۲۳ از ۳۰ ثابت** است:
```
CONSTANT: est_ul, est_dl, est_v2v, conf_*, helper_busy_s, epsilon_class,
          criticality_high/medium_share, mec_workers, est_age_*, epsilon_node_*,
          energy_budget_ratio, lambda_*, contact_slack, background_dags
VARYING : helper_contact_remaining_s, est_wait_{mec,ue,helper}_s,
          est_energy_{ue,mec,helper}_j
```
**اما این context گراف‌سطح است و به همهٔ ۲۰ node broadcast می‌شود — تعیین‌کننده نیست.**
آنچه تعیین‌کننده است بلوک ۷۰-بعدی feature هر node است:
```
۲۸ feature با std>0.1: compute_workload_bytes, task_output_bytes, external_input_bytes,
   incoming/outgoing_edge_bytes, indegree, outdegree, decoder_index, depth,
   is_root, is_sink, criticality_{low,medium,high}, tardiness_weight_scaled,
   return_hop_over_min_lb, lb_ue_log1p, feasible_ue, lb_mec_log1p, feasible_mec,
   lb_helper_log1p, feasible_helper, mc_* (9 فیلد mixed-criticality)
 ۴ feature ضعیف (1e-3<std<0.1)
۳۸ feature ~ثابت
```
یعنی **per-node workload + per-action lower bound + per-node feasibility** همه حاضرند.
node mask هم همه ۱ است (۲۰ node فعال، بدون padding).

**نتیجه:** H4 رد می‌شود. اطلاعات کافی موجود است؛ مسئله این است که یک core مشترک آن را
به یک قاعدهٔ شرطی per-graph تبدیل نمی‌کند.

---

## H5 — بهینه‌سازی منحط است
### VERDICT: **PARTIAL** — نرمال‌سازی سراسری رد، بودجهٔ ۳ گام تأیید

**بخشی که رد می‌شود:** ادعای «نرمال‌سازی advantage سیگنال بین‌گراف را نابود می‌کند».
`normalize_advantages` **درون** `_compute_samples_data` صدا زده می‌شود، که در حلقهٔ
per-meta-task اجرا می‌شود (`samplers/seq2seq_meta_sampler_process.py:35-42`):
```python
for meta_task, paths in paths_meta_batch.items():
    samples_data, paths = self._compute_samples_data(paths)   # ← نرمال‌سازی اینجا، per-task
```
پس نرمال‌سازی روی ۲۰ path همان task است، نه روی ۲۰۰. GAE هم استاندارد است
(`samplers/base.py:150-162`). پس H5 به این شکل رد می‌شود.

**بخشی که تأیید می‌شود:** بودجهٔ انطباق. `num_inner_grad_steps` روی ۳ **قفل** است
(`MRLCO.py:34-35` خطا می‌دهد اگر ≠۳). سیاست باید یک core تولید کند که با ۳ گام PPO به
رفتار بهینهٔ هر graph برسد. `policy_loss_mean` نهایی 0.0126 و `value_loss_mean` 0.0652
نشان می‌دهد inner PPO **همگرا شده** — یعنی ۳ گام کافی بوده که به یک حل مسئلهٔ *آسان*
برسد، و آن حل آسان همان «سازش میانگین‌گیر» است. `entropy_valid` 0.954→0.262 یعنی سیاست
تیز شده، اما روی یک قاعدهٔ متوسط، نه روی `[1]*20`.

---

## H6 — قید انرژی در جهت اشتباه می‌کشد
### VERDICT: **REFUTED (در جهت)**

پس از فیکس scope، اندازه‌گیری روی ledger **foreground** با لنگر `refs.E_mec` است
(`env/mec_offloaing_envs/scheduler/constraints.py:359-392`) و
`--energy-budget-fraction 0.5` یعنی `B = 0.5·E_mec`.

استدلال H6 درست است که budget نصف all-MEC است، پس MEC «نقض» می‌کند. **ولی جهت گرادیان
برعکس H6 است.** قید فقط **signed مثبت** را جریمه می‌کند، با وزن `λ_total = 1.3055`:
```
d(reward)/d(normalised energy) = -λ = -1.3055
```
یعنی اگر MEC یک واحد انرژی سیستم (نرمال‌شده به E_mec) بیشتر مصرف کند ولی بیش از
۱.۳ واحد latency نرمال‌شده بخرد، Lagrangian **MEC را ترجیح می‌دهد**. و می‌خرد: از
all-UE به all-MEC، latency از 0.279 به 0.026 می‌آید (−90٪) در برابر انرژی نرمال‌شده‌ای که
در حد O(1) است. پس Lagrangian به سمت MEC هل می‌دهد، نه دور از آن.

شاهد رفتاری هم خلاف H6 است: ۶۰.۵٪ اپیزودها
violation داشتند — یعنی سیاست **نقض می‌کند** ولی سود latency را نمی‌گیرد. `.446 local`
یک سازش است که **نه قید را ارضا می‌کند** (۶۰.۵٪ نقض) و **نه latency را بهینه** (۳× بدتر).
این نشانهٔ شکست تصمیم‌گیری است، نه یک قید که در جهت اشتباه می‌کشد.

**نکتهٔ مهم:** قید **مانع** بهینه شدن نیست؛ فقط مسئله را سخت‌تر می‌کند.

---

## H7 — نقص پیاده‌سازی مشخص
### VERDICT: **PARTIAL** — یک نقص واقعی (بازنویسی اکشن) با اثر متوسط

### H7.1 نقص تأییدشده: action بازنویسی می‌شود ولی decoder اکشن *sampled* را شرط می‌کند
`v2/shared_scheduler.py:532-586`:
```python
location = {0: UE, 1: MEC, 2: HELPER}[_plan_action(dag.dag_id, spec.task_id)]
if location == HELPER:
    if dag.helper_id is None or state is None:
        location = UE          # <-- بازنویسی به local
    else:
        if not admissible(state, ..., margin=contact_margin):
            location = UE      # <-- بازنویسی به local
            helper_rejected = True
...
effective_loc[key] = location
```
پس اکشن‌های UE و MEC **همیشه بدون تغییر** اجرا می‌شوند، ولی **HELPER می‌تواند بی‌صدا به UE
تبدیل شود**. در run اکشن v2v حدود ۳۵٪ در itr 0 و ۲۰٪ در انتها بود.

مشکل: بازخورد خودهم‌بسته (`decoder_inputs`) از اکشن **sampled** ساخته می‌شود، نه از
`effective_loc`. یعنی سیاست ۲۰ تصمیم را بر اساس جای‌گیری‌هایی می‌گیرد که **واقعاً اجرا
نشدند**. اگر HELPER غیرقابل‌قبول به UE تبدیل شود، nodeهای بعدی روی یک جهان خیالی شرط
می‌شوند و \(\log\pi(a_{k+1}\mid a_{\le k})\) روی همان خیال بنا می‌شود.

**اثر:** متوسط، چون تنها مسیر HELPER را می‌زند و فقط وقتی helper غیرقابل‌قبول باشد. در
run با `reliability=False` (گیت خاموش) مسیر `admissible()` اصلی‌ترین ماشه است.

### H7.2 نقص تأییدشده: `background_policy="all_ue"` کرش می‌کند
`v2/shared_scheduler.py:618`:
```python
rate = float(compute.ue_cpu_bytes_per_s[dag.owner])
```
`ue_cpu_bytes_per_s` یک tuple با یک ورودی (owner 0) است، پس برای DAG با `owner >= 1`
(`IndexError`). روی مسیر اجراشده پیش‌فرض `all_mec` است، پس **اثر صفر روی این run** دارد،
ولی یعنی sweep روی background policy ممکن نیست.

### H7.3 چیزهایی که بررسی و سالم یافت شدند
- اکشن‌های UE/MEC بدون بازنویسی، بدون admission/fallback اجرا می‌شوند.
- `plan_from_actions` (`v2/world.py:120-127`) ترتیب decoder را حفظ می‌کند (بدون جابه‌جایی).
- `_slot_of_graph` بر اساس هویت dataset graph است، نه object identity (`v2/env.py:507-527`).
- `reset_inner_optimizer(task_id)` قبل از هر انطباق فراخوانی می‌شود (`MRLCO.py:363`).
- `sync_task_policies_from_core` در ابتدای هر iteration (`meta_trainer.py:388`).
- ماسک: `mask/active_rate = mask/all_invalid_rate = mask/forced_rate = 0` و
  `policy/invalid_action_rate = 0` در همه ۵۰۰ iteration → shield خاموش/بی‌اثر، تصمیم‌ها
  آزادانه و معتبر بودند.
- `argmax_masked_rate = 0` → آرگ‌مکس هرگز با ماسک برخورد نکرده.

---

# WHAT EXPLAINS THE 2.69x GAP

با اعداد مرجع: سیاست 0.0806 با بودجهٔ attainable oracle 0.0261. یعنی از
«random = 0.1721» تا «oracle = 0.0261» مسیر ۰.۱۴۶۰ را فقط ۶۲.۷٪ طی کرده
(`(0.1721-0.0806)/(0.1721-0.0261)`)، در حالی که all-MEC ۹۹.۷٪ آن را طی می‌کند.

| رتبه | علت | سهم تخمینی از شکاف | شاهد |
|---|---|---|---|
| **۱** | **H2 — نبود مکانیزم تخصص‌سازی: یک core مشترک برای ۱۰ توزیع، پارامتر هر task دور ریخته می‌شود، meta-objective فقط پس-انطباق را می‌بیند، و gradient بیرونی میانگین جابه‌جایی‌های متناقض است.** | **~۱.۸–۲.۲ از ۳.۰۵×** | `async_parameters` کل core را کپی می‌کند (`meta_seq2seq_policy.py:1096`)؛ ترتیب support→adapt→query (`meta_trainer.py:397/459/528`) بدون sync میانی؛ `mean_pseudogradient` میانگین می‌گیرد (`learning_ops.py:16-30`)؛ ترکیب اکشن سازشی `.446/.352/.201` در برابر oracle `.16/.67/.03`؛ `meta_batch_size=10` قفل |
| **۲** | **H5 — بودجهٔ ۳ گام انطباق برای نصب یک قاعدهٔ per-graph روی core سازشی** | **~۰.۳–۰.۵×** | `num_inner_grad_steps` قفل ۳ (`MRLCO.py:34-35`)؛ k3 regret بدتر از k0 در خود run (0.0610 vs 0.0563)؛ علامت k3−k0 بین checkpointها عوض می‌شود |
| **۳** | **H7.1 — بازنویسی HELPER→UE با شرطی‌سازی روی اکشن sampled** | **~۰.۱–۰.۳×** | `shared_scheduler.py:532-586`؛ ۲۰–۳۵٪ اکشن‌ها v2v هستند |
| ۴ | H1 — شکل پاداش (telescoping prefix-only) | ~۰.۰۵–۰.۱× (سیگنال را ضعیف می‌کند، نه sparse) | ۷۹٪ توکن غیرصفر؛ اما پاداش فقط تابع makespan نهایی است |
| ۵ | H6 — قید انرژی | ~۰.۰ (جهت درست، مسئله را سخت‌تر می‌کند) | λ=1.3 به نفع MEC |
| ۶ | H3 / H4 — بیان و مشاهده | **۰** (ردشده) | ۲۸ feature قوی؛ ۴۰/۴۰ graph متمایز؛ `[1]*20` در فضای اکشن |
| — | H7.2 (`all_ue` crash) | **۰** روی این run | پیش‌فرض `all_mec` |

**جمع:** H2 ≈ ۶۰–۷۰٪، H5 ≈ ۱۰–۱۵٪، H7.1 ≈ ۳–۱۰٪، H1 ≈ ۲–۳٪. این‌ها تخمین‌اند و
**marginal** هستند — هرکدام به‌تنهایی قابل ابطال نیستند بدون یک ablation. تنها عدد
**hard** این است: policy ۰.۰۸۰۶، oracle ۰.۰۲۶۱، all-MEC ۰.۰۲۶۴۵، random ۰.۱۷۲۱.

---

# WHAT TO CHANGE FIRST

## یک تغییر با بالاترین ارزش

**در `policies/meta_seq2seq_policy.py::async_parameters` / `meta_trainer.py`، core را
«آگاه از رفتار پیش-انطباق خودش» کن.**

دقیقاً: rollout پشتیبان را **قبل از** `UpdatePPOTarget` جمع کن (یا یک rollout اضافی روی
`theta0` بگیر) و از آن در به‌روزرسانی بیرونی استفاده کن، تا core سیگنال بگیرد که
**بدون انطباق** چه می‌کند. الان ترتیب
`sync → support rollout → adapt → query rollout → outer`
یعنی core فقط «کیفیت انطباق‌پذیری» را بهینه می‌کند و می‌تواند یک `theta0` بی‌کیفیت داشته
باشد که ۳ گام آن را به یک سازش می‌رساند.

**چرا این اول؟** چون تنها علتی است که هم (الف) مکانیزمی است که می‌تواند یک قاعدهٔ per-graph
را با یک core مشترک تولید کند، و هم (ب) با شاهد رفتاری سازش اکشن‌ها و بی‌اثری k3 مطابقت
دارد. H5 (بودجهٔ ۳ گام) و H7.1 (بازنویسی) هر دو در حضور H2 حل نمی‌شوند.

## آزمون تصمیم‌گیر (قبل از هر تغییر بزرگ)

یک ablation ارزان که H2 و H5 را از هم جدا می‌کند:

1. **`k=0` روی ۴۰ graph با `meta_model_final.ckpt`** → من دارم: **0.080600**.
2. **همان ولی با انطباق عمیق‌تر** (اجازهٔ k=10 یا k=20 به‌جای ۳، فقط برای تشخیص، نه برای
   نتیجهٔ نهایی). اگر k=20 به ~۰.۰۳۰ نزدیک شود → **H5** علت است (بودجهٔ انطباق).
   اگر k=20 هم ~۰.۰۸ بماند → **H2** علت است (core هیچ‌وقت یاد نگرفته چه کند).
3. **کنترل:** یک بار همهٔ ۱۰ توزیع را یکسان کن (یک graph) و همان بودجه را آموزش بده.
   اگر آن مدل all-MEC را یاد بگیرد، ناتوانی در تخصص‌سازی روی چند توزیع تأیید می‌شود.

## تغییر دوم (ارزان و مستقل)

**`decoder_inputs` را از `effective_loc` بساز، نه از اکشن sampled.** یعنی بازخورد
خودهم‌بسته ترتیبی باید جای‌گیریِ **واقعاً اجرا‌شده** را ببیند. مسیر:
`v2/shared_scheduler.py` `effective_loc` → `V2ScheduleResult` → `env.step` →
اکشن ذخیره‌شده در path. این هم H7.1 را می‌بندد و هم likelihood را با رفتار واقعی هم‌راستا
می‌کند.

## آزمایش کنترل که باید قبل از هر ادعایی اجرا شود

`all_MEC` را به‌عنوان **باسلاین اجباری در همان حلقهٔ لاگ** بگذار (الان فقط در پنل ارزیابی
هست، و `Average greedy latency` چیزی متفاوت است). در run، `Average latency` به ۰.۲۰۶ رسید
در حالی که all-MEC ۰.۰۲۶۵ است — این تناقض ۸× هیچ‌وقت در لاگ آموزش دیده نمی‌شود.

---

## محدوده و چیزهای UNVERIFIED

- **بازرسی مستقیم وزن‌ها: UNVERIFIED.** قالب joblib را شکافتم (یک `NumpyArrayWrapper`،
  ناحیهٔ متادیتا دقیقاً `data[:213]`، `dtype=f4`, `order=C`، shape متغیر اول `(3,128)`،
  بدنه یک stream صاف float32 با ۸۴۸٬۸۷۱ عنصر) ولی بدون joblib جدول shape هر متغیر
  قابل رمزگشایی نیست. Docker محلی daemon ندارد و joblib روی `kish-ai` نصب نیست.
  در نتیجه نمی‌توانم توزیع logits را مستقیماً بخوانم.
- فقط **یک seed (0)** و **دو checkpoint**. ablationهای پیشنهادی اجرا نشده‌اند.
- سهم‌های درصدی در «WHAT EXPLAINS» **تخمین** هستند، نه اندازه‌گیری. تنها اعداد hard:
  policy 0.080600 / oracle 0.026115 / all-MEC 0.026452 / random 0.172051.
- probe ها روی `REALIZATION_SEEDS[0]`، epoch 0 اجرا شدند (همان realization ارزیابی).
