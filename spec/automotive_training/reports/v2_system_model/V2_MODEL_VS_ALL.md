# V2_MODEL_VS_ALL — جدول مقایسهٔ مدل ما با همهٔ روش‌ها

**منبع:** `V2HeldOutEvaluator` روی split ارزیابی (۴۰ graph × ۵ realization paired).
همهٔ اعداد از **یک منبع** هستند: `eval_trained.json` (itr 350) و `eval_final.json` (itr 499)
و `eval_ksweep.json`. seeds: `[1000, 1017, 1034, 1051, 1068]`، decode `deterministic`.

> **چرا این جدول با probe های قبلی فرق دارد.** probe های من (`fullcmp.py`, `pairedcmp.py`)
> با `V2AutomotiveEnv._schedule_slot` روی محیطی که خودم ساختم اجرا شدند و اعداد مطلق
> **۲.۵× بالاتر** از evaluator دادند (`all_MEC`: 0.0756 در برابر 0.0299) — یعنی آن محیط
> با محیط evaluator یکی نبود. **نسبت‌ها سازگار بودند** (`all_UE/all_MEC ≈ 3.6` در هر دو)،
> ولی اعداد مطلق probe **معتبر نیستند**. جدول زیر فقط از evaluator است.
> (آلارم قبلی من دربارهٔ «baseline با scheduler v1» **اشتباه بود**: `stack.py:609`
> `baseline_panel` را برای v2 override می‌کند و `set_v1_schedule_trap(True)` را arm می‌کند.)

---

## جدول اصلی

| candidate | latency (s) | نسبت به مدل | نوع |
|---|---|---|---|
| `candidate_oracle_s` | **0.028452** | ۰.۳۵× | latency-oracle پنل |
| `heft_nominal_s` | 0.029881 | ۰.۳۷× | HEFT اسمی (بدون اجرا) |
| **`all_MEC`** | **0.029931** | ۰.۳۷× | تک‌action |
| `greedy_coordinate_descent_MC` | 0.031239 | ۰.۳۹× | جست‌وجوی حریصانه |
| `heft_reference_v2` | 0.032665 | ۰.۴۱× | HEFT روی v2 |
| **مدل ما — k=3** (adapted) | **0.082751** | ۱.۰۳× | |
| **مدل ما — k=0** (base) | **0.080600** | **۱.۰۰×** | |
| `all_HELPER` | 0.202054 | ۲.۵۱× | تک‌action |
| `all_UE` | 0.319593 | ۳.۹۷× | تک‌action |

`best_val` (itr 350) برای comparison: k=0 = 0.091083، k=3 = 0.086559.
باسلاین‌ها **در هر دو ارزیابی یکسان‌اند** (realization-keyed).

## نسبت‌های headline (checkpoint نهایی)

```
مدل / all_MEC          = 2.69x
مدل / oracle پنل       = 2.83x
مدل / HEFT             = 2.47x
all_MEC / oracle       = 1.052x   ->  شکاف latency فقط ۵.۲۰٪
regret                 = 0.052148 s
```

## منحنی k (روی `best_val`، همان evaluator)

| k | latency (s) | نسبت به all_MEC |
|---|---|---|
| 0 | 0.091083 | 3.04× |
| 1 | 0.086529 | 2.89× |
| **2** | **0.085048** | **2.84×** ← بهترین |
| 3 | 0.085807 | 2.87× |
| 5 | 0.090450 | 3.02× |

انطباق در ~۶٪ اشباع می‌شود و از k=2 به k=5 **بدتر** می‌شود.

## سلامت قید — مدل نهایی

| معیار | مقدار | verdict |
|---|---|---|
| `graph_hard_violation_rate` | **0.0** | ✅ هیچ نقض hard deadline |
| `mc_policy_violation_count` | **0** | ✅ |
| `mean_dropped_task_count` | **0** | ✅ هیچ تسکی drop نشد |
| `mean_graph_violation_s` | **0.0** | ✅ |
| `mode_switch_rate` | 0.0 | ✅ |
| `n_tardy_high_tasks` | **3.4** از ۴۰۰ | ۰.۸۵٪ |
| `n_tardy_medium_tasks` | **0.0** از ۲۸۰ | ۰٪ |
| `mean_high_tardiness_s` | 0.000375 | ناچیز |
| `mean_medium_tardiness_s` | 0.0 | ✅ |

**مدل از نظر قید سخت **بی‌عیب** است.** مشکل عملکرد latency است، نه نقض deadline.

---

## چه چیزی جا افتاده — و چطور کامل شود

| معیار | وضعیت | علت |
|---|---|---|
| **latency** | ✅ کامل | evaluator |
| **miss / tardiness** | ✅ کامل | evaluator |
| **hard / dropped** | ✅ کامل | evaluator |
| **انرژی** | ❌ | probe من با evaluator هم‌راستا نشد؛ باید `baseline_panel` **خودِ v2** را برای انرژی گسترش داد (`stack.py:609`)، نه probe جدا |
| **ER/EE** | ❌ | منتظر انرژی |
| **action mix باسلاین‌ها** | ❌ | پنل فقط makespan ذخیره می‌کند |
| **p95 latency** | ❌ | evaluator فقط mean/std/min/max میدهد |

**کار درست برای کامل کردن:** افزودن `energy` و `tardiness` به dict پنل در
`spec/automotive_training/v2/stack.py::V2HeldOutEvaluator.baseline_panel` — همانجا که
`result` و `ledger` **قبلاً در دست** هستند. یک تغییر کوچک، و آن‌وقت جدول کامل از یک منبع
با realization یکی در می‌آید.

---

## محدودهٔ ادعا

* یک seed آموزش (۰)، دو checkpoint، یک decoding (`deterministic`).
* `candidate_oracle_s` یک **latency-oracle** است (best-of-panel روی makespan)، نه oracle
  چندمعیاره — در probe های من `hard > 0` داشت چون انرژی و miss را بهینه نمی‌کند.
* `heft_nominal_s` عدد **اسمی HEFT** است، نه اجرای واقعی روی v2.
* انرژی **در این جدول نیست** — و بدون آن، جدول ناقص است، چون
  `all_MEC` گران‌ترین گزینه از نظر انرژی است و `all_UE` ارزان‌ترین.
* **پس این جدول را به‌عنوان جدول نهایی مقاله استفاده نکن** تا انرژی اضافه شود.
