# ②A — Deadline semantics و measurement (بدون هیچ تغییری در PPO/reward)

**تاریخ:** ۲۰۲۶-۰۹-۲۱ · **وضعیت:** پیاده و تست‌شده · کل تست‌های غیر-TF: **۲۷۰ passed, 5 skipped**
**تصویب‌شده:** «اول فقط scheduler و metrics؛ هنوز PPO و Lagrangian وارد نشود.»

---

## ۱. قاعدهٔ اندازه‌گیری (همان چیزی که تأکید کردی)

```
miss  ⟺  F_available > deadline_s          (نه: finish > deadline_s)
```

- `F_available` = **زمانی که خروجی تسک واقعاً برای مصرف‌کننده قابل استفاده است** = زودترین رسیدن به اجراکنندهٔ یک successor؛ و برای sinkها = زمان برگشت خروجی به UE.
- `last_delivery` هم ثبت می‌شود (دیرترین تحویل به همهٔ مصرف‌کننده‌ها) برای گزارش، ولی **مبنای miss نیست**.
- مثال خودت به‌صورت تست پیاده شد: تسک روی MEC محاسبه‌اش ساعت ۱ تمام می‌شود، دانلینک تا ۵٫۵ طول می‌کشد، ددلاین ۳ است ⟹ **miss**؛ اگر معیار `finish` بود، pass می‌شد. هر دو حالت در تست قفل شده‌اند.

## ۲. تغییرات schema

### `CanonicalTask` (model.py)
| فیلد | معنا |
|---|---|
| `cycles_per_bit: float \| None` | شدت محاسباتی per-task؛ `None` ⇒ مقدار global همان اپیزود (cleanup C2) |
| `deadline_s: float \| None` | ددلاین |
| `criticality: float = 1.0` | وزن تسک در soft tardiness |
| `deadline_type: "none"\|"soft"\|"firm"\|"hard"` | کانال ددلاین — **جدا از هم** |

متدها: `compute_cycles(global)`، `effective_cycles_per_bit(global)`، `has_deadline`.

### `TaskExecutionRecord`
`finish_available`، `last_delivery`، `deadline_s`، `deadline_type`، `criticality`، `tardiness_s`، و پراپرتی‌های `availability_seconds` و `missed`.

### `ScheduleResult`
`soft_tardiness_s` · `soft_tardiness_normalized` · `firm_miss_count/rate` · `hard_miss_count/rate` · `hard_feasible` · `mean_tardiness_s` · `max_tardiness_s` + متد `deadline_metrics()`.

## ۳. قواعد تخصیص ددلاین (چون دیتاست ددلاین ندارد)
`scheduler/deadlines.py` — قطعی، بدون تصادف، بدون تغییر دیتاست:

| rule | فرمول |
|---|---|
| `none` (پیش‌فرض) | هیچ ددلاینی؛ نتیجهٔ زمان‌بندی **بی‌تغییر** |
| `uniform_of_all_mec` | `d_i = factor · L_allMEC` |
| `uniform_of_all_ue` | `d_i = factor · L_allUE` |
| `per_task_slack_of_ref` | `d_i = factor · finish_i(plan مرجع)` |
| `per_task_slack_of_avail` | `d_i = factor · F_available_i(plan مرجع)` ← هم‌معنا با قاعدهٔ اندازه‌گیری |

به‌علاوه `with_deadlines(dag, deadlines_s, ...)` برای نسخهٔ pure (بدون mutation) و `criticality_by_task` برای وزن‌دهی per-task.

## ۴. تست‌ها (`test_deadline_semantics.py` — ۱۳ تست)
- `deadline_type="none"` ⇒ همهٔ متریک‌ها صفر و `hard_feasible=True`؛ و **T/E دقیقاً برابر همان گراف بدون فیلدهای ②A** (measurement افزودنی است، نه تغییردهنده).
- **مثال تأییدشدهٔ تو**: compute زودتر، DL دیرتر ⇒ miss؛ و همان پلن با معیار `finish` pass می‌شد.
- رسیدن به ددلاین ⇒ `tardiness=0`.
- soft tardiness با `criticality` خطی است و `soft_tardiness_normalized = Σ w·t/ d`.
- کانال‌های firm و hard جدا شمرده می‌شوند (`firm_miss_count=1` ولی `hard_miss_count=0`).
- `F_available ≥ finish` همیشه (برای هر سه tier).
- sink روی UE ⇒ `F_available == finish` (هاپ برگشت ندارد).
- `max ≥ mean` tardiness.
- قواعد: `none` بی‌اثر · `uniform` = `factor·L_mec` · `per_task` مقادیر متمایز می‌دهد · rule/factor نامعتبر ⇒ خطا.

## ۵. آنچه عمداً انجام **نشد** (ورود به ②B)
- J = latency + β_s·soft_tardiness: هنوز نه.
- `energy_cost`، `hard_deadline_cost`، `firm_deadline_cost` و **hard feasibility mask**: هنوز نه.
- Lagrangian / dual ascent: هنوز نه (لایهٔ constraint از قبل هست ولی روی مسیر جدید فعال نشده).
- هیچ تغییری در reward، PPO، GAE و انتخاب چک‌پوینت **در این مرحله** داده نشد؛ عدد `checkpoint_selection_metric` هنوز قدیمی است و در ②B عوض می‌شود.

## ۶. مورد بازِ باقی‌مانده که باید بدانی
نرخ‌های **رادیو** در `physical_v1` هنوز از جدول legacy می‌آیند (MEC UL/DL ‏۹۱۷٬۵۰۴ B/s از ۷ Mib/s، V2V ‏۶۵۵٬۳۶۰ B/s از ۵ Mib/s)، چون در قرارداد ① فقط *انرژی* رادیو (TX-only) عوض شد. مرجع ادبیات برای پهنای‌باند ۱۰ MHz است (Liu) که معادل ۱٫۲۵ MB/s می‌شود. اگر بخواهی در ②B یا ③ پهنای‌باند را هم فیزیکی کنم، بگو — ولی چون روی makespan اثر می‌گذارد، جدا و با گزارش قبل/بعد انجامش می‌دهم.
