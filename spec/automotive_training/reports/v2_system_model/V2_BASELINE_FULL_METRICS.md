# V2_BASELINE_FULL_METRICS — همهٔ معیارها برای همهٔ روش‌ها

**منبع واحد:** `V2HeldOutEvaluator` v2 (`stack.py:609`) روی ۴۰ graph ارزیابی × ۵ realization
paired (`seeds = [1000, 1017, 1034, 1051, 1068]`). هر ردیف = **۲۰۰ پلن**.
`n per candidate = 200` برای همه.

## جدول اصلی

| candidate | latency (s) | **system J** | requester J | mobile J | **miss rate** | tardy HIGH | tardy MED | V2V airtime (s) | cpu_wait (s) | fUE | fMEC | fV2V |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `candidate_oracle_s` | **0.028452** | 27.933 | 0.0146 | 0.0449 | **0.0000** | 0.000 | 0.000 | 0.00238 | 0.00083 | 0.103 | 0.877 | 0.020 |
| **`all_MEC`** | 0.029931 | **29.483** | 0.0000 | 0.0000 | **0.0000** | 0.000 | 0.000 | 0.00000 | 0.00114 | 0.000 | 1.000 | 0.000 |
| `greedy_coordinate_descent_MC` | 0.031239 | **27.190** | 0.0224 | 0.0749 | 0.0003 | کم | کم | 0.00440 | 0.00067 | 0.145 | 0.823 | 0.033 |
| `heft_reference_v2` | 0.032665 | 27.890 | 0.0131 | 0.0848 | 0.0013 | کم | کم | 0.00270 | 0.00037 | 0.084 | 0.865 | 0.051 |
| **مدل — k=0** | **0.080600** | — | — | — | **0.0000** | 3.4/400 | 0/280 | — | — | — | — | — |
| **مدل — k=3** | 0.082751 | — | — | — | **0.0000** | 3.4/400 | 0/280 | — | — | — | — | — |
| `all_HELPER` | 0.202054 | 3.399 | 0.0000 | 3.3986 | **0.1418** | زیاد | 0 | 0.01325 | 0.00776 | 0.000 | 0.000 | 1.000 |
| `all_UE` | 0.319593 | 0.320 | 0.3196 | 0.3196 | **0.3350** | زیاد | 0 | 0.00000 | 0.01244 | 1.000 | 0.000 | 0.000 |

> **مدل** از aggregate خود evaluator آمده (که `mean_*_tardiness_s` و `n_tardy_*` را دارد)؛
> بقیه از پنل. `graph_hard_violation_rate = 0.0` و `mc_policy_violation_count = 0` برای مدل.

---

## یافته‌های اصلی

### ۱. `all_MEC` **هم سریع‌ترین است و هم گران‌ترین** — به‌طور هم‌زمان
```
all_MEC:   latency 0.029931 (بهترین)   system 29.483 J (بیشترین)
greedy_cd: latency 0.031239            system 27.190 J  (-7.8%)
HEFT:      latency 0.032665            system 27.890 J  (-5.4%)
oracle:    latency 0.028452 (بهترین)   system 27.933 J  (-5.3%)
```
پس روی معیار **`candidate_oracle_s`** (که بهترین latency را می‌گیرد) **۵.۳٪ انرژی کمتر از
all_MEC** مصرف می‌کند. `greedy_cd` **۷.۸٪** کمتر.

**قیمت latency برای انرژی:**
```
greedy_cd: -7.8% انرژی  در برابر  +4.4% latency
HEFT:      -5.4% انرژی  در برابر  +9.1% latency
```

### ۲. `requester_joules` یک **تلهٔ گزارش‌دهی** است
```
all_MEC:   requester 0.000 J   system 29.483 J      ← foreground: صفر
all_UE:    requester 0.320 J   system  0.320 J      ← تقریباً کل سیستم
all_MEC foreground = 0.000         (انرژی UE)
all_MEC foreground = ?             (انرژی MEC)
```
`all_MEC` **صفر انرژی UE** دارد چون همهٔ CPU روی MEC است. اگر `requester_joules` را به‌عنوان
«مصرف انرژی» گزارش کنی، `all_MEC` **رایگان** به‌نظر می‌رسد — که کاملاً غلط است، چون
۲۹.۴۸۳ J روی MEC خرج می‌کند.

**الزام:** همیشه `system_joules` بده، و `requester` را فقط به‌عنوان **مرز باتری کاربر**،
نه هزینهٔ کل.

### ۳. منحنی مبادلهٔ واقعی
```
                 latency(s)   system_J   miss
all_UE             0.319593      0.320    0.3350
all_HELPER         0.202054      3.399    0.1418
HEFT               0.032665     27.890    0.0013
greedy_cd          0.031239     27.190    0.0003
all_MEC            0.029931     29.483    0.0000
oracle             0.028452     27.933    0.0000
```
`all_UE` و `all_HELPER` **ارزان به‌نظر می‌رسند** (`0.32 J` و `3.4 J`) ولی **۱۴٪ و ۳۳.۵٪
miss** دارند — یعنی deadline را قربانی می‌کنند. **ارزان‌بودن آن‌ها هزینهٔ پنهان است.**

### ۴. مدل ما در این جدول کجاست
```
مدل:  latency 0.080600 s   miss 0.0   tardy HIGH 3.4/400 (0.85%)
```
- **از نظر قید سخت بی‌عیب:** `hard = 0`، `miss = 0`، `dropped = 0`
- **از نظر latency بدترین** در میان روش‌های non-trivial: ۲.۶۹× all_MEC، ۲.۴۷× HEFT،
  ۲.۵۸× greedy_cd
- **از نظر tardy HIGH:** ۳.۴ در برابر **صفر** برای all_MEC و HEFT و oracle ⇒ مدل **بدترین
  انطباق با criticality** را دارد در میان روش‌های non-trivial

**پس ادعای درست:** مدل قید **hard** را رعایت می‌کند (هیچ miss و هیچ drop) ولی در
**latency** و **tardiness سطح HIGH** شکست می‌خورد.

---

## دو باگ که در این مسیر پیدا و رفع شد

### B1 — `deadline_miss_rate` **سخت‌کد صفر** بود
`spec/automotive_training/v2/eval_loop.py:131`
```python
"deadline_miss_rate": 0.0,        # <-- هرگز محاسبه نمی‌شد
```
یعنی **هر باسلاین یک رکورد deadline کامل و بی‌عیب گزارش می‌کرد که هرگز اندازه‌گیری نشده بود.**
بعد از رفع، اعداد واقعی ظاهر شدند: `all_UE = 0.3350`، `all_HELPER = 0.1418`،
`HEFT = 0.0013`، `greedy_cd = 0.0003`، `all_MEC = 0.0`.

**این باگ خطرناک بود چون در جهت مطلوب دروغ می‌گفت.**

### B2 — انرژی و miss باسلاین‌ها **دور ریخته می‌شد**
`spec/automotive_training/v2/stack.py` (override پنل v2)
```python
entry[name] = float(_score(world, ..., link_process=link_process)["episode_latency_s"])
#              ^^^ dict کامل دور ریخته می‌شد      ^^^ فقط یک کلید برداشته می‌شد
```
`_score` **از قبل** `system_joules`، `requester_joules`، `mobile_joules`،
`helper_task_fraction`، `cpu_wait_mean_s`، `wasted_work_bytes`، `v2v_airtime_s` و
`location_mix` را برمی‌گرداند — ولی هیچ‌کدام به پنل نمی‌رسیدند.

**رفع:** کل dict زیر `"<candidate>_metrics"` ذخیره می‌شود؛ کلیدهای scalar دست‌نخورده
می‌مانند تا `candidate_oracle` و مصرف‌کننده‌های موجود نشکنند.

---

## مرز مدل‌سازی (برای شفافیت در مقاله)

```
system_joules = UE CPU + helper CPU + MEC CPU + UE/helper/MEC TX
```
**مدل نشده** (از `run.json.unmodeled_components`):
```
receiver / static radio power (include_rx_energy=False)
CPU idle / leakage power between tasks
MEC / RSU static (non-compute) power (server_static_power_w is null)
backhaul / core-network transport energy
battery state dynamics, charging efficiency, reserve accounting
warm-standby reservation energy
```
پس `system_joules` یک **کران پایین** است و اعداد مطلق را به‌عنوان «مصرف واقعی» گزارش نکن.

---

## توصیه برای مقاله

**جدول را با این ستون‌ها بده:**
| latency (mean) | **system energy** | miss rate | tardy HIGH | fMEC |

**و این دو جمله را بنویس:**
1. «`all_MEC` سریع‌ترین است ولی گران‌ترین: ۲۹.۴۸ J در برابر ۲۷.۱۹ J برای
   greedy-coordinate-descent — یعنی ۷.۸٪ صرفه‌جویی انرژی با ۴.۴٪ افزایش latency.»
2. «`all_UE` کم‌انرژی‌ترین به‌نظر می‌رسد (۰.۳۲ J) ولی ۳۳.۵٪ deadline را از دست می‌دهد؛
   ارزانی آن هزینهٔ پنهان در قابلیت اطمینان است.»

## محدودهٔ ادعا
- یک seed آموزش، دو checkpoint، `decoding=deterministic`، ۵ realization.
- `candidate_oracle_s` یک **latency-oracle** است (best-of-panel روی makespan) — با اینکه
  در این جدول `miss = 0` دارد.
- `tardy HIGH`/`tardy MED` مدل از aggregate خود evaluator است؛ برای باسلاین‌ها از
  `n_tardy_high`/`n_tardy_medium` پنل (که در JSON خام هستند).
- باگ `deadline_miss_rate` فقط **باسلاین‌ها** را تحت تأثیر قرار داد؛ مسیر گزارش خود مدل
  از `rollout_metric_block` می‌آمد و واقعی بود.
