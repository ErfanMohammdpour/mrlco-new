# پروژه A — مدل‌سازی و ارزیابی آفلودینگ DAG در MEC با کمک V2V

**دانشگاه:** KNTU  
**درس:** [نام درس را وارد کنید]  
**دانشجو:** [نام و شماره دانشجویی]  
**استاد:** [نام استاد]  
**تاریخ:** [تاریخ ارائه]

---

## ۱. خلاصه پروژه (Abstract)

این پروژه یک **چارچوب شبیه‌سازی و ارزیابی** برای آفلودینگ برنامه‌های **DAG** در محیط **Mobile Edge Computing (MEC)** با کمک **Vehicle-to-Vehicle (V2V)** ارائه می‌دهد. تمرکز اصلی روی **مدل‌سازی واقع‌گرایانه سیستم** (شامل محدودیت **نیم‌دوبلکس** کانال V2V)، **هدف مشترک تأخیر–انرژی**، و **تحلیل رفتار زمان‌بندی** است.

برخلاف کارهایی که فقط Local/MEC را در نظر می‌گیرند، این پروژه فضای تصمیم **سه‌گانه** {Local, MEC, V2V} را formalize می‌کند و سیاست‌های مختلف (Greedy، HEFT، و سیاست یادگرفته) را روی **یک شبیه‌ساز واحد** مقایسه می‌کند.

**خروجی‌های مورد انتظار:**
- مدل ریاضی سیستم + شبیه‌ساز قابل اجرا
- benchmark شامل ۱,۹۰۰ گراف DAG
- نتایج مقایسه‌ای latency/energy و تحلیل ساختاری تصمیم‌ها
- گزارش ۱۰–۱۲ صفحه + پرزنت ۱۵–۱۸ دقیقه

---

## ۲. تعریف مسئله (مستقل)

### ۲.۱ انگیزه
اپلیکیشن‌های خودرویی (مثلاً perception مشترک، analytics بلادرنگ) workload وابسته به هم دارند. هر task باید روی **UE محلی**، **سرور MEC**، یا **خودروی همسایه (V2V helper)** اجرا شود. تصمیم‌ها باید هم **تأخیر کل (makespan)** و هم **مصرف انرژی** را در نظر بگیرند.

### ۲.۲ سؤال تحقیق این پروژه
> در محیط MEC+V2V با محدودیت نیم‌دوبلکس و وابستگی DAG، چگونه می‌توان **هزینه تأخیر و انرژی** را مدل کرد، **سیاست‌های زمان‌بندی** را به‌صورت عادلانه مقایسه کرد، و **رفتار ساختاری** (عمق task، تعداد successor) را تحلیل نمود؟

### ۲.۳ فرضیات سیستم
| موجودیت | توضیح |
|---------|--------|
| UE | پردازنده محلی با ظرفیت \(f_l\) |
| MEC | سرور لبه با uplink/downlink جدا |
| V2V helper | خودروی همسایه با کانال **نیم‌دوبلکس** |
| DAG | \(N=20\) task، یال = وابستگی داده |

---

## ۳. نوآوری‌ها و سهم علمی (۴ مورد — متوازن با پروژه B)

1. **فرمول‌بندی آفلودینگ DAG با فضای تصمیم سه‌گانه** Local/MEC/V2V (نه binary)
2. **مدل زمان‌بندی V2V نیم‌دوبلکس** — صف کانال مشترک + صف پردازش helper
3. **مدل انرژی ترکیبی** — DVFS محلی + انتقال MEC/V2V + compute روی helper
4. **چارچوب benchmark و تحلیل ساختاری** — ۱۹ توزیع، ۱,۹۰۰ گراف، breakdown بر اساس depth/successor

> **نکته:** این پروژه «شبیه‌ساز + تحلیل» است؛ یادگیری عمیق در پروژه B پوشش داده می‌شود. شما سیاست learned را فقط به‌عنوان **یکی از سیاست‌های قابل ارزیابی** معرفی می‌کنید.

---

## ۴. مدل ریاضی (هسته فنی)

### ۴.۱ ویژگی task (۲۰ بعد)
\[
\mathbf{x}_i = [i,\; T_i^{loc}, T_i^{up}, T_i^{mec}, T_i^{dl},\; T_i^{vup}, T_i^{vexe}, T_i^{vdl},\; \mathbf{p}_i,\; \mathbf{s}_i]
\]
- \(\mathbf{p}_i\): predecessor indices (padding)
- \(\mathbf{s}_i\): successor indices (padding)

### ۴.۲ تأخیر
- Local: \(T_i^{loc} = w_i / f_l\)
- MEC: uplink → remote exec → downlink
- V2V: uplink → exec on helper → downlink (با محدودیت نیم‌دوبلکس)

### ۴.۳ انرژی (پارامترهای کد)
از `meta_trainer.py` → `ENERGY_CONFIG`:
- \(\lambda_{lat} = 0.5\), \(\lambda_{ene} = 0.5\)
- \(\rho=1.0\), \(\zeta=2.0\), \(p_{tx}=0.1\), \(p_{rx}=0.05\)
- V2V: \(p_{tx}^{v2v}=0.06\), \(p_{rx}^{v2v}=0.03\), \(\rho_{v2v}=0.7\)

### ۴.۴ پاداش/امتیاز ترکیبی
\[
r_i = \lambda_{lat}\, r_i^{lat} + \lambda_{ene}\, r_i^{ene}
\]
هر جزء نرمال‌شده نسبت به bounds نظری گراف.

### ۴.۵ Greedy baseline (پیاده‌سازی شده)
در هر گام HEFT-priority، حالت با **کمترین finish time** انتخاب می‌شود.  
کد: `env/mec_offloaing_envs/offloading_env.py` → `greedy_solution()`

---

## ۵. پیاده‌سازی (ماژول‌های تحت مالکیت)

| فایل | نقش |
|------|-----|
| `env/mec_offloaing_envs/offloading_task_graph.py` | پارس `.gv`، رتبه HEFT، feature encoding |
| `env/mec_offloaing_envs/offloading_env.py` | شبیه‌ساز، reward، greedy، V2V scheduling |
| `env/mec_offloaing_envs/data/meta_offloading_20/` | ۱,۹۰۰ گراف benchmark |
| `calculate_opt_solution.py` | مرجع optimal برای توزیع‌های کوچک |
| `paper/analyze.py` | تحلیل node-level و ساختاری (بخش ۳ گزارش شما) |

### اجرای پایه
```bash
# از ریشه repo
python meta_trainer.py   # فقط برای گرفتن greedy baseline و لاگ انرژی
python calculate_opt_solution.py   # optional: bound بهینه
python paper/analyze.py   # جداول رفتار ساختاری (با CSV/XLSX موجود)
```

---

## ۶. آزمایش‌ها (۵ آزمایش — هم‌تراز با پروژه B)

| # | آزمایش | وضعیت | خروجی |
|---|--------|--------|--------|
| E1 | **Greedy vs All-MEC vs All-Local** | از env موجود | جدول latency/energy |
| E2 | **HEFT baseline** | **باید اضافه شود** | مقایسه با Greedy |
| E3 | **حساسیت \(\lambda_{ene}\)** (0.3, 0.5, 0.7) | **باید اضافه شود** | نمودار Pareto |
| E4 | **حساسیت پهنای باند V2V** (3, 5, 7 Mbps) | **باید اضافه شود** | جدول makespan |
| E5 | **تحلیل ساختاری action** (depth, successor) | **موجود** در `analysis_tables/` | نمودار stacked bar |

### نتایج موجود برای ارجاع (سیاست learned — فقط بخش ارزیابی)
از `paper/analysis_tables/main_performance_stability.csv` (Greedy ref: T1≈809 lat, T2≈792 lat):

| Task | Method | Last-20 Lat. | Lat. Impr. | Last-20 Energy | Comp. Impr. |
|------|--------|-------------|------------|----------------|-------------|
| T1 (dist 10) | Learned policy | 575.40 | 17.71% | 659.01 | 23.03% |
| T2 (dist 12) | Learned policy | 605.16 | 14.32% | 682.69 | 20.68% |

### رفتار ساختاری (E5 — از گزارش تحلیل)
| Task | Action | Share | Avg Depth |
|------|--------|-------|-----------|
| T1 | Local / MEC / V2V | 34.9% / 58.4% / 6.8% | — |
| T1 depth≥2 | — | V2V بیشتر روی exit/low-successor | 1.04 |

**تفسیر برای پرزنت:** taskهای با successor زیاد → MEC؛ taskهای exit → بیشتر Local/V2V برای اجتناب از bottleneck نیم‌دوبلکس.

---

## ۷. کارهای اضافه (الزامی — ~۱ هفته)

### ۷.۱ پیاده‌سازی HEFT (~۲ روز)
- الگوریتم HEFT کلاسیک روی همان DAG و cost model
- فایل پیشنهادی: `baselines/heft_scheduler.py`
- API: `heft_plan(task_graph) -> plan, makespan, energy`

### ۷.۲ اسکریپت Pareto (~۲ روز)
- `scripts/sensitivity_lambda.py` — سه λ، خروجی CSV + plot
- `scripts/sensitivity_v2v_bandwidth.py` — سه bandwidth

### ۷.۳ Case Study (~۱ روز)
- یک DAG نمونه (۲۰ node): timeline Gantt برای ۳ حالت Local/MEC/V2V
- اسلاید یا appendix

### ۷.۴ جدول notation + یک figure سیستم
- از `figures/fig_system_model.png`, `fig_joint_objective.png`

---

## ۸. ساختار گزارش (۱۰–۱۲ صفحه)

| فصل | عنوان | صفحات |
|-----|--------|-------|
| ۱ | مقدمه و انگیزه | ۱ |
| ۲ | مرور ادبیات (MEC, DAG scheduling, V2V) | ۱.۵ |
| ۳ | مدل سیستم و فرمول‌بندی | ۲.۵ |
| ۴ | شبیه‌ساز و baselineها | ۲ |
| ۵ | طراحی benchmark (۱۹×۱۰۰) | ۱ |
| ۶ | نتایج آزمایش (E1–E5) | ۲.۵ |
| ۷ | بحث و محدودیت‌ها | ۱ |
| ۸ | نتیجه‌گیری | ۰.۵ |
| — | منابع | — |

---

## ۹. پرزنت — اسکریپت اسلاید به اسلاید (~۱۶ دقیقه)

| اسلاید | زمان | محتوا | چه بگویید |
|--------|------|--------|-----------|
| 1 | 1′ | عنوان، نام، KNTU | «پروژه من روی **مدل و ارزیابی** آفلودینگ DAG در MEC+V2V است.» |
| 2 | 1.5′ | انگیزه خودرو/Edge | فشار latency + battery |
| 3 | 1.5′ | چرا DAG؟ | precedence → NP-hard |
| 4 | 2′ | `fig_system_model` | UE, MEC, V2V helper |
| 5 | 2′ | نیم‌دوبلکس V2V | کانال مشترک؛ uplink/downlink serial |
| 6 | 1.5′ | مدل انرژی + λ | trade-off |
| 7 | 1.5′ | Greedy + HEFT | دو baseline |
| 8 | 1′ | Benchmark 1900 graphs | 19 distributions |
| 9 | 2′ | نتایج E1–E3 | جداول/نمودار |
| 10 | 1.5′ | تحلیل ساختاری E5 | depth/successor |
| 11 | 1′ | محدودیت‌ها | synthetic, single helper |
| 12 | 0.5′ | Q&A | — |

### در پرزنت **نگویید**
- جزئیات Graph2Seq، Reptile، PPO clip
- «ما meta-RL invent کردیم»

### اگر پرسیدند ML کجاست؟
> «شبیه‌ساز من **سیاست‌agnostیک** است. سیاست learned یک black-box روی همین env ارزیابی شده؛ طراحی یادگیری خارج از scope این پروژه است.»

---

## ۱۰. سوالات دفاع (Q&A)

| سوال احتمالی | پاسخ کوتاه |
|--------------|------------|
| چرا V2V؟ | ظرفیت compute نزدیک + مصرف توان کمتر از cellular دور |
| نیم‌دوبلکس چیست؟ | uplink و downlink همزمان روی یک کانال ممکن نیست |
| Greedy چرا ضعیف است؟ | myopic — فقط finish time فوری، نه بار کانال آینده |
| HEFT چه فرقی با Greedy دارد� | HEFT rank-based global heuristic برای DAG heterogeneous |
| λ=0.5 چرا؟ | trade-off متعادل؛ E3 حساسیت را نشان می‌دهد |
| دیتاست واقعی است؟ | synthetic benchmark با fatness/density متنوع — محدودیت صریح |

---

## ۱۱. مرز با پروژه B (برای جلوگیری از overlap)

| موضوع | پروژه A (شما) | پروژه B (هم‌تیمی) |
|--------|---------------|-------------------|
| مدل V2V/انرژی | **عمیق** | ۱ اسلاید |
| Graph2Seq / Meta-RL | ۱ خط | **عمیق** |
| Greedy/HEFT/Pareto | **عمیق** | فقط عدد baseline |
| Ablation encoder | — | **عمیق** |
| تحلیل ساختاری scheduling | **عمیق** | خلاصه action % |

---

## ۱۲. منابع پیشنهادی

- Mao et al., MEC survey
- Topcuoglu et al., HEFT
- Karagiannis et al., vehicular networking
- Wang et al., MRLCO (فقط برای context — نه focus)

---

## ۱۳. چک‌لیست قبل از تحویل

- [ ] HEFT پیاده و در جدول E2
- [ ] نمودار Pareto (E3)
- [ ] حساسیت bandwidth (E4)
- [ ] گزارش ۱۰+ صفحه با notation یکدست
- [ ] پرزنت ۱۵–۱۸ دقیقه rehearse شده
- [ ] عنوان پروژه روی cover متفاوت از پروژه B
- [ ] محدودیت‌ها explicit

---

*نسخه: 1.0 — سند پروژه A (سیستم و زمان‌بندی)*
