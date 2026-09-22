# ENERGY_MODEL_V1 — ① freeze artifact

**تاریخ:** ۲۰۲۶-۰۹-۲۱ · **وضعیت:** پیاده‌شده، تست‌شده، و audit شده · **قبل از ② منجمد می‌شود**
**دامنهٔ audit:** dist 1، ۱۰ گراف، خانواده‌های {all_UE, all_MEC, all_HELPER, greedy_from_mec, 2-opt}، scope = `system`
**خروجی خام:** `reports/energy/pareto_<preset>_d1.json` · **ابزار:** `spec/energy_pareto_audit.py` · **تست‌ها:** `test_energy_model.py` (۱۲ تست)

---

## ۱. قرارداد پیاده‌شده (دقیقاً همان چیزی که تصویب شد)

```
C_i        = D_i · cycles_per_bit              # cycles (ویژگی تسک)
T_cpu(i,x) = C_i / f_x                         # s
E_cpu(i,x) = κ_x · C_i · f_x²                  # J   (== (κ_x f_x³)·T_cpu)

total_requester_joules = E_UE
total_mobile_joules    = E_UE + E_helper
total_system_joules    = E_UE + E_helper + E_MEC_compute + E_MEC_tx     ← primary (energy_scope=system)
```

| مورد | تصمیم |
|---|---|
| مدل | `physical_v1` (primary) · `legacy` کاملاً دست‌نخورده و bit-exact |
| سوئیچ | `ResourceConfig.from_frozen_yaml(model="legacy"|"physical_v1"|None)`؛ پیش‌فرض کد = `legacy`، پس هیچ ران/ابزار قبلی تغییر نمی‌کند |
| نرخ CPU | در physical از `f/(8·ξ)` مشتق می‌شود تا `C/f == duration` دقیقاً برقرار باشد (واحدها سازگار) |
| رادیو | **TX-only** در primary (`include_rx_energy=false`) چون هیچ `p_rx` تأییدشده‌ای در ادبیات نیست؛ RX فقط sensitivity با پارامتر صریح |
| `rho_v2v=0.7` | فقط در `legacy`؛ در physical حذف شده و جای آن `κ_helper` مستقل نشسته |
| `κ=0` برای MEC | در physical **خطا** می‌دهد (`ValueError`) — همان تقارنی که این مدل برای رفعش ساخته شد |
| واحدها | SI: cycles, cycles/s, bytes, bytes/s, s, W, J — تبدیل بیت↔بایت صریح (`C = D·8·ξ`) |
| provenance | هر پارامتر در `spec/frozen_experiment.yaml` فیلد `source` دارد |

### پارامترهای primary (با منبع)
| tier | f | κ | منبع | توان ضمنی `P=κf³` | نرخ مشتق‌شده |
|---|---|---|---|---|---|
| UE | ۱٫۰ GHz | ۱e-۲۷ | داخل بازهٔ تأییدشده (κ ۱e-۲۸…۱e-۲۶، f ۰٫۲–۲٫۵); κ=۱e-۲۷ = tier خودرو در Gu25/Liang24؛ **از Liu قرض گرفته نشده** | ۱٫۰ W | ۰٫۴۲ MB/s |
| helper | ۱٫۵ GHz | ۵e-۲۷ | Liu 2023, DCN 9(6), Table 1 | ۱۶٫۹ W | ۰٫۶۳ MB/s |
| MEC | ۱۰ GHz | ۱e-۲۷ | Liu 2023, DCN 9(6), Table 1 (+ فرکانس double-sourced با Zhao arXiv:1807.02311) | **۱۰۰۰ W** ⚠️ | ۴٫۱۷ MB/s |
| رادیو | — | — | UE TX ۱ W = ۳۰dBm (Liu) · MEC TX ۳٫۱۶۲ W = ۳۵dBm (Zhao، provenance جدا) · helper TX ۱ W (فرض) | — | — |

⚠️ `P_MEC = 1 kW` همان چیزی است که review هشدار داد: مدل `κf³` فقط توان **dynamic** است و با ۵۰–۱۰۰W کل نود مقایسه نمی‌شود. طبق تصویب، مدل canonical primary ماند و مدل static/idle فقط در schema آماده است (`server_static_power_w`, `server_utilization_model: false`).

---

## ۲. نتیجهٔ audit — آیا هندسه عوض شد؟

### ۲٫۱ primary (Liu calibration، scope=system، ξ=۳۰۰)
| خانواده | T (s) | E_system | E_requester | E_mobile |
|---|---|---|---|---|
| all_UE | ۳۵۴۱٫۴ | **۳٫۵۴e۳** | ۳٫۵۴e۳ | ۳٫۵۴e۳ |
| all_HELPER | ۲۵۵۹٫۹ | ۴٫۰۶e۴ | ۴۱۳ | ۴٫۰۶e۴ |
| all_MEC | ۶۲۲٫۲ | ۳٫۵۵e۵ | ۲۹۵ | ۲۹۵ |
| greedy_from_mec | ۵۴۱٫۸ | ۳٫۱۴e۵ | ۶۱۱ | ۱٫۸۴e۳ |
| 2-opt | **۵۲۰٫۷** | ۳٫۰۴e۵ | ۷۳۱ | ۲٫۷۲e۳ |

**نتیجهٔ کلیدی:** `all_MEC` در **۰/۱۰** گراف روی جبههٔ Pareto است و در **۰/۱۰** گراف هیچ خانواده‌ای را dominate نمی‌کند. دو پلن ترکیبی (`greedy`, `2-opt`) هم سریع‌ترند و هم کم‌مصرف‌تر از all-MEC خالص.
⟹ **شرط توقف ① برآورده شد**: دوران «MEC همه را در هر دو بُعد می‌برد» تمام شده و یک trade-off سه‌گانهٔ واقعی داریم:
`all_UE` (کندترین، ارزان‌ترین) ↔ `all_HELPER` (میانه) ↔ `mixed/2-opt` (سریع‌ترین، گران‌ترین ولی ارزان‌تر از all-MEC).

### ۲٫۲ جدول حساسیت — کیفیت هندسه به calibration وابسته است
| preset | تغییر | P_MEC | T ALL-MEC | E_sys ALL-MEC | E_sys 2-opt | جبههٔ Pareto | MEC dominator؟ |
|---|---|---|---|---|---|---|---|
| **primary** | — | ۱۰۰۰ W | ۶۲۲ | ۳٫۵۵e۵ | ۳٫۰۵e۵ | UE, HELPER, twopt (+greedy ۵/۱۰) | نه |
| `zhao` | κ_MEC=۱e-۲۸ | ۱۰۰ W | ۶۲۲ | ۳٫۶۳e۴ | ۳٫۳۸e۴ | UE, twopt (+greedy ۵/۱۰) | نه |
| `f6` | f_MEC=۶ GHz | ۲۱۶ W | ۸۱۷ | ۱٫۲۸e۵ | ۱٫۰۵e۵ | UE, HELPER, twopt | نه |
| `f4` | f_MEC=۴ GHz | ۶۴ W | ۱۰۷۱ | ۵٫۷۶e۴ | ۴٫۶۴e۴ | UE, HELPER, twopt (+greedy ۴/۱۰) | نه |
| `xi1000` | ξ=۱۰۰۰ | ۱۰۰۰ W | ۱۳۴۴ | ۱٫۱۸e۶ | ۱٫۰۳e۶ | UE, HELPER, twopt (+greedy ۳/۱۰) | نه |
| `rx` | RX روشن | ۱۰۰۰ W | ۶۲۲ | ۳٫۵۵e۵ | ۳٫۰۵e۵ | مانند primary | نه |

مشاهدات مهم برای ② و مقاله:
1. **هیچ preset ای all-MEC را به dominator برنمی‌گرداند.** حتی مطلوب‌ترین حالت برای MEC (`zhao`, ۱۰۰W) هم all-MEC را از جبهه بیرون می‌گذارد، چون یک پلن ترکیبی می‌تواند سرعت MEC را برای بخشی از DAG بخرد و هزینهٔ انرژی‌اش را فقط برای همان بخش بپردازد.
2. **`all_HELPER` در `zhao` از جبهه حذف می‌شود** (MEC ارزان‌تر و سریع‌تر از helper می‌شود): یعنی «V2V چه زمانی ارزش دارد» یک سؤال calibration-محور است — همین برای ② مهم است.
3. **ترتیب انرژی بین tiers در primary:** `E_UE/MiB = ۲٫۵۲ J` · `E_helper/MiB = ۲۸٫۳ J` · `E_MEC/MiB = ۲۵۲ J` (به‌ازای هر مگابایت، با ξ=۳۰۰). یعنی MEC ۱۰۰ برابر UE و ۹ برابر helper به‌ازای هر بایت گران است — که همان **معکوس** مدل قبلی (MEC ≈ رایگان) است.
4. **`E_requester`** (مرز A) برعکس رفتار می‌کند: all-MEC فقط ۲۹۵ J از باتری خودرو می‌گیرد در برابر ۳۵۴۱ J لوکال — یعنی اگر مرز requester انتخاب شود، MEC دوباره جذاب می‌شود. **پس مرز انرژی، قطب‌نمای نتیجه است و باید در مقاله صریح اعلام شود.**

---

## ۳. تست‌های پذیرش (همه pass)
`env/mec_offloaing_envs/scheduler/tests/test_energy_model.py` — ۱۲ تست:
- `E = κCf²` و `E = (κf³)(C/f)` برای هر سه tier تا `places=9` برابرند
- `T = C/f` با نرخ مشتق‌شده برابر است
- `E_system = E_mobile + E_MEC_compute + E_MEC_tx` و `E_req ≤ E_mob ≤ E_sys`
- **legacy bit-exact** روی گراف واقعی (T و E با اعداد مرجع MARGO-SPEC-v0.1)
- تغییر scope فقط aggregation را عوض می‌کند (زمان‌بندی ثابت)
- MEC TX فقط در system (هرگز داخل mobile)
- `include_rx_energy=False` ⇒ صفر بودن هر سه مؤلفهٔ RX
- `include_rx_energy=True` بدون `p_rx` ⇒ خطا (عدد اختراع نمی‌شود)
- `κ_MEC=0` در physical ⇒ خطا؛ دسترسی فقط از راه `legacy`
- مونوتونی: `C↑⇒E↑` · `κ↑⇒E↑` · `f↑⇒T↓ و E↑` · `ξ` خطی
- کل سوئیت غیر-TF: **۲۵۳ passed, 5 skipped**

---

## ۴. چیزهایی که ① تغییر نداد
- مسیر default همه‌چیز روی `legacy` است؛ هیچ اعداد قبلی، تست قدیمی یا ابزار audit بازتولیدش را از دست نداد.
- reward/j/deadline دست نخورد (آن‌ها ② هستند). فقط aggregation سه‌مرزی اضافه شد.
- `rho_v2v=0.7` و مدل توان قدیمی برای reproduction باقی است.

## ۵. کار باقی‌ماندهٔ همین مرحله (قبل از ②)
1. **کالیبراسیون UE مستقل:** κ_UE=۱e-۲۷ داخل بازهٔ تأییدشده انتخاب شد ولی منبعش «انتخاب در بازه» است، نه یک مقالهٔ واحد. اگر می‌خواهی محکم‌تر شود: یک sweep کوچک κ_UE ∈ {۱e-۲۸, ۱e-۲۷, ۱e-۲۶} به جدول حساسیت اضافه کن (اسکریپت آماده است).
2. **مدل static/idle سرور:** schema آماده است (`server_static_power_w`, `server_utilization_model`) ولی مقدار ندارد؛ طبق تصویب در primary وارد نشد.
3. **تصمیم دربارهٔ ξ:** ۳۰۰ پیش‌فرض است؛ `xi1000` نشان می‌دهد فقط مقیاس عوض می‌شود و رتبه‌بندی جبهه تقریباً ثابت می‌ماند.
4. **`E_requester` به‌عنوان مرز دوم:** جدول نشان می‌دهد نتیجهٔ policy شدیداً به مرز وابسته است؛ در ② باید هر سه مرز همزمان لاگ شوند (ساختارش هست).

> **برای رفتن به ② آماده است** چون شرط توقف برآورده شد: هیچ dominance باقی نمانده و همهٔ invariantهای قرارداد ① pass شده‌اند.

---

# پیوست — چهار cleanup تصویب‌شده + sweep κ_UE (۲۰۲۶-۰۹-۲۱)

## C1. تناقض scope در config حذف شد
- **authoritative:** `energy.accounting_primary_scope: system`
- `energy.objective_weights` حالا `status: legacy_only` دارد (چون در فرمولاسیون مقید، انرژی یک کانال constraint است، نه جملهٔ وزنی هدف).
- **اعتبارسنجی سخت در کد:** `validate_frozen_energy_scope(doc, spec)` اگر `accounting_primary_scope` با `energy_model.energy_scope` اختلاف داشته باشد **ValueError** می‌دهد (نه warning). تست شد: `mobile` در برابر `system` ⇒ خطا.
- تابع کمکی `objective_weights_status(doc)` وضعیت را برمی‌گرداند (`legacy_only`).

## C2. `cycles_per_bit` آینده‌نگر شد
- `CanonicalTask` حالا `cycles_per_bit: float | None` دارد + متد `compute_cycles(global)` و `effective_cycles_per_bit(global)` که fallback به مقدار global می‌دهند.
- `ResourceConfig.cpu_rate_for_task(loc, task)` و `compute_energy_joules(..., cycles_per_bit=None)` مسیر per-task را پشتیبانی می‌کنند؛ موتور از آن استفاده می‌کند.
- تست: تسکی با `cycles_per_bit=1200` روی همان گراف، دقیقاً `C·8·1200/f` زمان می‌برد (override کار می‌کند).

## C3. تست end-to-end زمان‌بندی (مهم‌ترین cleanup)
`TestSchedulerTimingUsesDerivedRate` — ۴ تست:
1. برای **هر تسک** در هر سه پلن خالص: `finish − start == C_i / f_tier` تا `places=9` (با `C_i = W_i·8·ξ`).
2. makespan فیزیکی با makespan legacy در هر سه پلن **متفاوت** است ⟹ اثبات اینکه جدول نرخ قدیمی (۱/۱/۱۰ MB/s که هنوز در YAML هست) به‌کار نمی‌رود.
3. در legacy همان تسک‌ها دقیقاً `W/rate_legacy` طول می‌کشند (پس legacy دست‌نخورده).
4. override سطح‌تسک (C2) در زمان‌بندی اثر می‌گذارد.

## C4. checkpoint metric
در ②B عوض می‌شود: انتخاب چک‌پوینت از `validation_query_composite_objective` (کامپوزیت ۰٫۵L+۰٫۵E) جدا می‌شود و به `J = L_norm + β_s·soft_tardiness_norm` + گزارش جدا برای کانال‌های constraint منتقل می‌شود. **انجام نشده تا ②B (طبق تصویب).**

## C5. sweep κ_UE — نتیجه: جبهه qualitative تغییر نکرد ✅
| κ_UE | E_sys all_UE | جبههٔ Pareto | MEC dominator؟ |
|---|---|---|---|
| ۱e-۲۸ | ۳۵۴ | UE, HELPER, 2-opt (+greedy ۵/۱۰) | نه |
| **۱e-۲۷ (primary)** | ۳۵۴۱ | UE, HELPER, 2-opt (+greedy ۵/۱۰) | نه |
| ۱e-۲۶ | ۳٫۵۴e۴ | UE, HELPER, 2-opt (+greedy ۵/۱۰) | نه |

فقط مقیاس انرژی `all_UE` عوض می‌شود (خطی با κ_UE)؛ **عضویت جبهه و حکم «MEC dominator نیست» در هر سه حالت ثابت است** ⟹ کالیبراسیون UE نقطهٔ شکنندهٔ نتیجه نیست. خروجی خام: `reports/energy/pareto_ue28_d1.json` و `pareto_ue26_d1.json`.

## جمع‌بندی وضعیت
- کل تست‌های غیر-TF: **۲۷۰ passed, 5 skipped** (قبلاً ۲۵۳).
- ① بسته و منجمد؛ ②A هم در همین نشست انجام شد (گزارش جدا: `DEADLINE_SEMANTICS.md`).
