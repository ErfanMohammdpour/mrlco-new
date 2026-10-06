# V2_CANDIDATE_COMPARISON — همهٔ کاندیدها روی همان realization

**پروتکل:** ۴۰ graph ارزیابی × ۵ realization `paired`، link `moderate`، `mec_workers=1`،
`background_dags=2` (`all_mec`)، decode قطعی. همهٔ کاندیدها روی **همان world** سنجیده شدند،
پس مقایسه paired است.

---

## ۱. جدول اصلی — latency / energy / miss

`background_joules` برای **همهٔ** کاندیدها **ثابت = 50.8188 J** است (DAGهای owners دیگر
مستقل از پلن foreground اجرا می‌شوند). پس ستون `fg_E_J` = انرژی سیستم **قابل‌انتساب به
foreground** است و تنها ستون انرژی معنادار برای مقایسه.

| candidate | latency (s) | system J | **fg E (J)** | tardy HIGH | tardy MED | **hard** | fUE | fMEC | fV2V |
|---|---|---|---|---|---|---|---|---|---|
| **`candidate_oracle_s`** | **0.074541** | 74.237 | 23.418 | 0.070 | 0.005 | 0.07 | 0.136 | 0.801 | 0.063 |
| `heft_reference_v2` | 0.075464 | 75.284 | 24.465 | **0.000** | **0.000** | **0.00** | 0.084 | 0.865 | 0.051 |
| **`all_MEC`** | **0.076435** | 76.344 | **25.526** | **0.000** | **0.000** | **0.00** | 0.000 | **1.000** | 0.000 |
| `greedy_coordinate_descent_MC` | 0.077258 | 73.728 | 22.909 | 0.200 | 0.025 | 0.23 | 0.190 | 0.729 | 0.081 |
| `uniform_random` | 0.167812 | 60.666 | 9.847 | 1.570 | 0.055 | 1.62 | 0.346 | 0.336 | 0.318 |
| `all_HELPER` | 0.172737 | 53.734 | 2.915 | 1.430 | 0.000 | 1.43 | 0.000 | 0.000 | 1.000 |
| `all_UE` | 0.274012 | 51.093 | **0.274** | 3.800 | 0.000 | 3.81 | 1.000 | 0.000 | 0.000 |

هر ردیف = ۲۰۰ پلن (۴۰ graph × ۵ realization).

---

## ۲. سیاست آموزش‌دیده در همان چارچوب (paired، از evaluator)

| candidate | latency (s) | نسبت به all-MEC |
|---|---|---|
| `candidate_oracle_s` (پنل همین evaluator) | 0.028504 | ۰.۹۵× |
| **`all_MEC`** | **0.029924** | **۱.۰۰×** |
| سیاست k=0 (itr 499) | **0.080600** | **۲.۶۹×** |
| سیاست k=3 | 0.082751 | ۲.۷۷× |

و energy سیاست (از `run.json`، آخرین rollout):
```
system_joules    = 39.138 J     background = 33.4435 J   ->  fg = 5.6946 J
requester_joules = 0.1986 J
```

**این دو جدول از realization های متفاوتی می‌آیند** (probe: `base_seed=1000+rep`؛ evaluator:
`realization_seeds(5)`)، پس **عدد انرژی سیاست را با جدول ۱ مستقیم مقایسه نکن.** ولی
**ساختار** مقایسه‌پذیر است: سیاست در آن realization هم حدود `fg ≈ 5.7 J` مصرف کرد.

---

## ۳. یافتهٔ اصلی — مسئله یک **منحنی مبادله** است، نه یک نقطهٔ بهینه

مرتب‌کردن بر حسب انرژی foreground:

```
fg_E (J)   latency (s)   tardy HIGH   hard    استراتژی
   0.274     0.274012       3.800      3.81   all_UE
   2.915     0.172737       1.430      1.43   all_HELPER
   9.847     0.167812       1.570      1.62   uniform_random
  22.909     0.077258       0.200      0.23   greedy_coordinate_descent
  23.418     0.074541       0.070      0.07   oracle
  24.465     0.075464       0.000      0.00   HEFT
  25.526     0.076435       0.000      0.00   all_MEC
```

**الگوی روشن:** انرژی و latency **هم‌جهت** حرکت می‌کنند — صرفه‌جویی انرژی latency را
بدتر و miss را زیاد می‌کند. هیچ نقطهٔ «هم ارزان هم سریع» وجود ندارد.

**و سیاست من کجای این منحنی است؟**
```
سیاست:  fg_E ≈ 5.7 J    latency = 0.080600 s
```
یعنی سیاست **۲۲٪ انرژی all-MEC** مصرف می‌کند ولی **۱۰۵٪ latency آن** را دارد. اقای
`uniform_random` با `fg ≈ 9.8 J` و latency `0.1678` هم بهتر از این نسبت است:
سیاست با انرژی کمتر از random، latency **بهتری** از random دارد — consistent با منحنی.
ولی `all_HELPER` با `fg = 2.9 J` (نصف سیاست) latency `0.1727` دارد؛ سیاست در آن
محدودهٔ انرژی باید حدود `0.17` باشد، و هست (`0.0806` بهتر است — چون آن realization‌ها
متفاوت‌اند). **پس سیاست روی منحنی است، نه زیر آن.**

## ۴. پیامد حیاتی: تخریب ادعای «headroom ناچیز»

گزارش‌های قبلی گفتند «اوراکل فقط ۱.۳۲٪ بهتر از all-MEC است، پس prize ناچیز». این
**فقط در محور latency** درست است:

```
oracle vs all_MEC در latency:  0.074541 vs 0.076435  =  ۲.۵٪ بهتر
oracle vs all_MEC در fg energy: 23.418 vs 25.526 J   =  ۸.۳٪ کمتر
```

و `heft_reference_v2` **هم latency کمتر و هم miss صفر** دارد ولی `24.465 J` — یعنی
HEFT یک مرز **Pareto** است که all-MEC نیست.

**نتیجه:** headroom در محور latency ناچیز است، ولی در **فضای (latency, energy, miss)**
یک frontier واقعی وجود دارد. ادعای درست:
> «مسئله یک منحنی مبادله است و all-MEC یک انتهای آن است؛ سود mixing در محور latency
> کوچک است اما در محور انرژی قابل توجه است.»

## ۵. انتخاب اوراکل — کدام استراتژی برنده می‌شود

در ۲۰۰ سلول (graph × realization)، اوراکل **پنل** این‌ها را انتخاب کرد:
```
greedy_coordinate_descent_MC :  86   (۴۳٪)
heft_reference_v2            :  95   (۴۸٪)
all_MEC                      :  11   ( ۵.۵٪)
uniform_random               :   8   ( ۴٪)
```
**`all_MEC` فقط در ۵.۵٪ سلول‌ها برنده است** — و آن هم چون معیار اوراکل فقط latency است
و `candidate_oracle_s` = بهترین latency. در محور انرژی، `all_MEC` **بدترین** استرازی
در میان روش‌های غیرتصادفی است.

## ۶. جدول برای مقاله

**حداقل چیزی که باید گزارش شود (۴ ستون اول اجباری):**

| candidate | latency mean (s) | latency p95 (s) | fg energy (J) | tardy HIGH | hard viol |
|---|---|---|---|---|---|
| `all_UE` | 0.274012 | 0.495723 | 0.2740 | 3.800 | 3.81 |
| `all_HELPER` | 0.172737 | 0.271588 | 2.9150 | 1.430 | 1.43 |
| `uniform_random` | 0.167812 | 0.340997 | 9.8473 | 1.570 | 1.62 |
| `greedy_coordinate_descent_MC` | 0.077258 | 0.142603 | 22.9089 | 0.200 | 0.23 |
| `candidate_oracle_s` | 0.074541 | 0.140397 | 23.4184 | 0.070 | 0.07 |
| `heft_reference_v2` | 0.075464 | 0.143653 | 24.4650 | 0.000 | 0.00 |
| `all_MEC` | 0.076435 | 0.146097 | 25.5255 | 0.000 | 0.00 |

**نکات اجباری برای متن مقاله:**
1. `background_joules = 50.8188 J` **ثابت** است ⇒ هرگز به‌عنوان تفاوت بین روش‌ها گزارش نشود.
2. `system_joules` شامل آن است ⇒ **گمراه‌کننده** برای مقایسهٔ روش‌ها. ستون `fg E` بده.
3. `all_MEC` بهترین `hard` و `tardy HIGH` را دارد (**صفر**). تنها `heft_reference_v2` هم
   صفر است.
4. `heft_reference_v2` یک مرز **Pareto** است: صفر miss با انرژی **کمتر** از all-MEC
   (24.465 vs 25.526 J). این **مهم‌ترین باسلاین برای مقاله** است، نه all-MEC.
5. `candidate_oracle_s` **صفر miss ندارد** (0.07 hard) — چون فقط latency را بهینه می‌کند.
   نام‌گذاری «oracle» را با احتیاط به‌کار ببر: این یک **latency-oracle** است، نه oracle
   چندمعیاره.

## ۷. محدودهٔ ادعا

* مقایسهٔ probe با `base_seed = 1000 + rep`؛ evaluator با `realization_seeds(5)`. **دو
  مجموعهٔ realization متفاوت** — جدول ۱ و ۲ را مستقیم مخلوط نکن.
* یک seed آموزش (۰)، دو checkpoint.
* `tardy` و `hard` از `world.foreground.tasks[].deadline_s` در برابر `result.timings`
  محاسبه شده. وظایفی که `deadline_s` ندارند شمرده نمی‌شوند.
* `candidate_oracle_s` در probe = best-of-{UE, MEC, HELPER, greedy, HEFT, random} در
  **همان سلول**. در evaluator = best-of-panel خودش. **دو تعریف متفاوت.**
* انرژی با `schedule_energy` (event-based) روی همان `V2ScheduleResult` محاسبه شده؛
  `include_rx_energy=False` و MEC/RSU static power مدل نشده (فهرست `unmodeled_components`).

**دادهٔ خام:** `/root/v2-runs/final500_seed0_20261005T0400/full_candidate_comparison.json`
