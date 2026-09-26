# MARGO — تاریخ کامل باگ‌ها، مسیر توسعه و قراردادهای نهایی

**مخزن:** `MARGO_BASELINE/mrlco-new` · **شاخه:** `phase4-eval` · **HEAD در زمان تدوین:** `90cfea6`
**روش گردآوری:** `git log/show/diff` روی همهٔ شاخه‌ها، خواندن کد در HEAD، و اسناد `docs/` و `reports/v0.3-audit/`. هر ادعا با `file:line` یا مسیر artifact مستند شده است. موارد تأییدنشده صریحاً `UNVERIFIED` علامت خورده‌اند.

---

## ۰. یک واقعیت ساختاری که کل تاریخ را توضیح می‌دهد

این مخزن **دو ریشهٔ مستقل** دارد:

| ریشه | تاریخ | توضیح |
|---|---|---|
| `451af97` | 2020-07-25 | upstream `linkpark1904/metarl-offloading` (MRLCO اصلی) |
| `37acb69` | 2026-02-19 | «Initial commit» فورک: ۳۵۴۲ فایل / +۲۶۹٬۶۷۶ خط |

`git merge-base --is-ancestor e7ac45f 37acb69` → **NO**؛ یعنی دو تاریخِ نامرتبط‌اند. مسیر `main` = `37acb69 → 7cb45fb → 84ee463 → f2ae7e2 → 53fe08d` و همهٔ شاخه‌های فاز از `53fe08d` جدا شده‌اند.

**نتیجهٔ مهم:** بازنویسی V2V + Graph2Seq + `obs_dim=20` + `vocab_size=3` که باگ‌های کلاسیک MARGO را **ساخت**، هیچ commitی در این مخزن ندارد؛ به‌صورت squash‌شده وارد شده است. پس آن باگ‌ها را نباید به هیچ sha نسبت داد — با auditing همان وضعیت واردشده کشف شدند.

---

## ۱. نقشهٔ زمانی

| فاز | بازهٔ تاریخ | موضوع | نمونه shas |
|---|---|---|---|
| A | 2020-07 → 2020-09 | upstream MRLCO | `6139985`, `a053fce`, `0b12de2`, `1ec1f2a` |
| B | 2025-07-28/29 | فورک (ریشهٔ دوم) | `c770ed5`, `f96eab9`, `e7ac45f` |
| C | 2026-02-19 | ایمپورت squash‌شده | `37acb69`, `53fe08d` |
| D | 2026-09-03/04 | فاز ۰–۳: موتور کانونیکال، obs، پروتکل یادگیری | `5bcd53b`, `937e4c5`, `7058b88`, `f2453b7`, `c0cbecc`, `367c10e`, `fdfc98b`, `f246006` |
| E | 2026-09-12 | بازنویسی v0.3 | `e6cf007` |
| F | 2026-09-21/22 | v0.4: ددلاین + obs v3 + رابط masked-PPO | `8a3d391`, `fc9ebce`, `08d74d8`, `c504017` |
| G | 2026-09-22/23 | ران‌های mask-sanity + watchdog | `7f8f5d2`, `974f39f`, `91ca2a1`, `a9bae0f`, `7fa31f2`, `a92b8a8` |
| H | 2026-09-24/25 | ددلاین: regime dataset + verdict شیلد | `de70131`, `78efd56`, `f831e69`, `92554c5`, `fada17e` |
| I | 2026-09-25 | مهاجرت scope انرژی (E1–E4.2) + تشخیص‌ها | `2963d49`, `b407de9`, `55b42f5`, `8139d32`, `e41b498`, `827d80c`, `01888d0`, `7f5b63a` |
| J | 2026-09-25/26 | Pilot A، ارزیاب چک‌پوینت، watchdog درون‌حلقه‌ای | `fe90a11`, `64a15db`, `d3ff0c8`, `7b5e8f2`, `e3fd3cd`, `efaffbc`, `ff608dd`, `62c4538` |
| K | 2026-09-26/27 | قرارداد هدف v1 + صداقت انتخاب + شاهد فیزیک | `c644fc5`, `f0c0cc8`, `cf7efbf`, `a1badce`, `d74c447`, `56cc4a2`, `37a59fe`, `dd0bc5c`, `90cfea6` |

تعداد commitها: ۱۲۳ روی شاخهٔ فعلی، ۱۷۱ در کل مخزن؛ ۶۱ commit با پیام `fix`.

---

## ۲. کاتالوگ کامل باگ‌ها

### فاز A — upstream MRLCO (۲۰۲۰)

| # | sha | ناحیه | باگ (symptom) | ریشه | فیکس |
|---|---|---|---|---|---|
| A1 | `6139985` | arity محیط | `ValueError: too many values to unpack` — فراخوان‌ها `(cost, energy)` باز می‌کردند ولی تابع یک مقدار برمی‌گرداند | بازنویسی `get_all_*_execute_time` به یک خروجی، بدون به‌روزرسانی همهٔ فراخوان‌ها | ناقص: `a053fce` فقط `__init__` را درست کرد؛ نسخهٔ `meta_evaluator.py` تا `e7ac45f` (۲۰۲۵) خراب ماند |
| A2 | `a053fce` | arity محیط | ساخت env کرش می‌کرد | همان | `self.local_exe_time` / `self.mec_exe_time` |
| A3 | `0b12de2` | تکرار سیمولاتور | دو کپی واگرا از زمان‌بند در مخزن (۶۸۹ خط) با همان typo | copy-paste | حذف کپی؛ parser و sampler به منبع واحد وصل شدند |
| A4 | `d32df14` | فیزیک | API انرژی/نرخ دست‌خوش تغییر شد (`Ploss`, `up_transmission_cost(data,distance)→(data)`, حذف `lambda_t/lambda_e`) | ساده‌سازی، نه باگ‌فیکس | −۹۴ خط |
| A5 | `1ec1f2a`,`a3b7a4b`,… | TF2 | حذف `tf.variable_scope/placeholder/get_collection/assign/Session` | مهاجرت TF1.15→TF2 | `tf.compat.v1.*` |
| A6 | `736ef83`,`7420022`,… | کد مرده | ~۱۶۰۰ خط مرده | مسیرهای رهاشده | حذف؛ `ppo_reptile.py`→`MRLCO.py` |

**A1 کهن‌الگوی این مخزن است:** تغییر arity در یک call site فیکس می‌شود و پنج سال در جای دیگر خراب می‌ماند.

نکته: در آن era `end_token=2` **درست** بود، چون `vocab_size=2` بود (`451af97`, `adaee65`). باگ‌های V2V/MEC-wait/clique هنوز وجود ندارند.

### فاز B — فورک ۲۰۲۵ (ریشهٔ دوم)

| # | sha | ناحیه | باگ | ریشه | فیکس |
|---|---|---|---|---|---|
| B1 | `c770ed5` | reporting | کرش در ساخت گزارش | `plt.style.use('seaborn-v0_8-darkgrid')` فقط در matplotlib ≥3.6 | try/fallback |
| B2 | `f96eab9` | evaluator | مسیر چک‌پوینت hard-coded و ناموجود | مسیر کهنه | `./meta_model_inner_step1/meta_model_final.ckpt` |
| B3 | `e7ac45f` | arity evaluator | `finish_time, energy_cost = env.get_all_mec_execute_time()` → خطای unpack | همان A1 که هرگز به evaluator منتقل نشده بود | unpack تک‌مقداری |
| B4 | `9fac1a1`,`7ecfc85`,`5b013cf` | trainer | commitهای «fix bug» که فقط `n_itr` را ۵۰۰→۳۰۰→۲۰→۱۰۰۰ می‌چرخانند | **باگ قابل‌شناسایی در diff نیست** | — |
| B5 | `c770ed5` | hyperparams | `inner_lr/outer_lr` 5e-4→1e-3 + glorot×0.5 | tuning، نه fix | — |
| B6 | `d70b165`…`f33c511` | encoder | Graph2Seq اضافه شد ولی `obs_dim=17`/`vocab_size=2` ماند | کار ویژگی (۷ commit) | — |
| B7 | latent | PPO | **value clip روی مقدار جدید لنگر می‌انداخت**: `vpredclipped = vpred + clip(vpred − old_v, ±ε)` | خطای کلاسیک کپی PPO | فیکس در `fdfc98b` (۲۰۲۶-۰۹-۰۴) |
| B8 | latent | PPO logging | `vf_loss`/`pg_loss` در طول K step جمع می‌شد و بدون reset تقسیم بر K می‌شد | مقدار اولیه بیرون حلقه | فیکس در `f246006` |

### فاز C — ایمپورت ۲۰۲۶-۰۲-۱۹

فقط README/.gitignore/assets؛ **هیچ کد یا باگی فیکس نشد**. همین وضعیت است که دو سند audit توصیف می‌کنند: `docs/CODE_AUDIT.md` می‌گوید «کد فعلی: Graph2Seq, obs_dim=20, vocab_size=3, V2V half-duplex, clip=0.2, inner_batch_size=10, n_itr=3500» در برابر origin «LSTM, obs_dim=17, vocab_size=2, بدون V2V, clip=0.3, inner_batch_size=1000».

### فاز D — کمپین اول فیکس واقعی (۲۰۲۶-۰۹-۰۳/۰۴)

| # | sha | ناحیه | باگ | ریشه | فیکس |
|---|---|---|---|---|---|
| D1 | `5bcd53b` | audit | ۸ باگ منطقی + ۸ نقص تکمیلی برای اولین بار catalog شد | وضعیت ایمپورت‌شده هرگز audit نشده بود | اسناد `docs/CODE_AUDIT.md`, `docs/LOGIC_AND_LEARNING_BUGS.md` |
| D2 | `b49b60f`,`5c3029e`,`937e4c5` | زمان‌بند | **باگ ۱: MEC منتظر predecessorِ V2V نمی‌ماند** — `ws_start_time = max(ws_avaliable_time, max([max(FT_locally[j], FT_ws[j]) ...]))` بدون `FT_v2v_dl[j]`/`FT_wr[j]` | شاخهٔ MEC از MRLCO دوسطحی کپی شده و با افزودن action=2 به‌روز نشده بود | موتور کانونیکال `scheduler/` با جدول route ۳×۳ و تقویم انتقال صریح |
| D3 | `7058b88` | انرژی | **باگ ۶: کران‌های انرژی غلط** — `min_energy` = انرژی انتقالی همه (all-MEC) با اینکه `ptx_v2v=0.06 < ptx=0.1`؛ پس پلن V2V-محور انرژی *مثبت* می‌گرفت؛ ضمناً MEC compute = 0 ولی helper compute شارژ می‌شد | conflate شدن «پلن» و «مرز» | حذف bounds؛ `compute_reference_ranges` با سه پلن pure؛ `E = total_mobile = UE + HELPER` (ADR-001) |
| D4 | `f2453b7` | مقیاس reward | **باگ ۷: نرمال‌سازی latency با کران‌های تک‌تسکی روی deltaهای makespan** → امتیاز ≪ −۱ یا مثبتِ نادرست | `score_func`/`calculate_max_min_runningcost` از origin دوسطحی | `telescoping_token_rewards`: `r_t = −(w_L·ΔL/L_scale + w_E·ΔE/E_scale)` |
| D5 | `c0cbecc` | گراف | **باگ ۴: adjacency یک clique کامل ۲۰ نودی بود** (خودِ نود هم وصل)؛ یال‌های `.gv` هرگز به `fw_adj_info` نمی‌رسید | جایگزینی LSTM→GCN بدون سیم‌کشی DAG | جداول همسایه بر اساس **decoder index**، `MAX_NEIGH=19`، خطای صریح روی overflow |
| D6 | `c0cbecc` | encoder | **باگ ۵: ویژگی‌های pred/succ شناسهٔ خام task بودند نه موقعیت HEFT؛ فقط `range(0,i)`؛ سقف ۶ با truncate خاموش** | انکودر ۱۷ ویژگی بعد از تغییر ترتیب ردیف‌ها دست‌نخورده مانده بود | `MAX_NEIGH=19`، خطا جای truncate، یال‌ها از `edge_set` |
| D7 | `c0cbecc` | numerics | **نقص #11: میانگین همسایه، slotهای PAD را می‌شمرد** | aggregator با padding به‌روز نشده بود | `tf.sequence_mask(neigh_len)` + تقسیم بر تعداد معتبر |
| D8 | `367c10e` | encoder | dropout زنده بود (`0.1`) و aggregation پیشین می‌توانست خاموش بماند (`bidirectional` پیش‌فرض false) | مسیر نیمه‌سیم‌کشی‌شده | `ENCODER_DROPOUT=0`، `bidirectional=True` اجباری، ردیف PAD صفر |
| D9 | `367c10e` | بازتولید | هش فایل‌های spec بین CRLF و LF متفاوت بود | هش روی بایت خام | هش بعد از canonical LF + تست مقایسهٔ بازتولید stats |
| D10 | `fdfc98b` | PPO | **نقص #9: value-clip روی مقدار جدید** (همان B7) | خطای کلاسیک PPO | `old_v + clip(vpred − old_v, ±ε)` + تست منبع |
| D11 | `fdfc98b` | trainer | **نقص #10: `for i in range(5)`** فقط ۵ از ۱۰ policy متا را لاگ/میانگین می‌کرد | عدد ثابت | `range(len(new_samples_data))` |
| D12 | `fdfc98b`/`810e91d` | متا | **باگ ۳: گرادیان متا ~۱۰۰× کوچک** (`inner_batch_size=10` در برابر ۱۰۰۰) و تقسیم بر `inner_lr` با اینکه inner **Adam** است | مخرج شامل `inner_lr` و `update_numbers` | `mean_pseudogradient(theta0, adapted, inner_lr, k_steps)` + تغییر پروتکل به outer Adam |
| D13 | `fdfc98b` | متا | **باگ ۸: «Reptile» در واقع ۱۰ apply متوالی Adam روی core متحرک بود** | خواندن `θ` داخل حلقه | خواندن همهٔ `θ'_i`، یک `mean_pseudogradient`، **یک** apply، سپس sync |
| D14 | `f246006` | split/leak | **نشت query: `set_task` عددی، اسلایس ارزیابی را پاک می‌کرد و کل استخر ۱۰۰ گرافی را در validation افشا می‌کرد** | ایندکس‌گذاری عددی به‌جای قرارداد `{dist_index, graph_indices}` | گارد `spec/eval_protocol.require_sliced_task` + `split_loader` |
| D15 | `f246006` | پروتکل PPO | `k_steps` به‌عنوان Adam-apply شمرده نمی‌شد؛ core در validation تغییر می‌کرد | mismatch پروتکل/متریک | شمارش apply، فریز core در validation، reset accumulatorها |
| D16 | `f246006` | importها | `mpi4py` اجباری بود؛ `tf.get_logger` در بعضی buildها نبود | وابستگی اختیاری = اجباری | import اختیاری |
| D17 | `69d3396` | گیت فاز۰ | «closure» فقط در prose ادعا می‌شد | بدون اسکریپت گیت | `spec/phase0_gate.py` + سخت‌سازی oracleها |
| D18 | `39ea0ed` | provenance | گیت فاز ۴ در ایمیج بدون `git` کرش می‌کرد | shell بدون گارد | تحمل نبود git |
| D19 | `464827e`,`30aa491` | ایمیج GPU | تگ `nvidia/cuda:10.0` حذف شده بود → build شکست | base image از بین رفته | بازسازی روی `tensorflow/tensorflow:1.15.5-gpu-py3` |
| D20 | `1046c28` | driver | driver فاز ۴ بدون deps میزبان import نمی‌شد | PYTHONPATH/mpi4py/matplotlib | اصلاح campaign + logger |

**سه‌گانهٔ فیزیک (D2/D3/D4):** «عدد گزارش‌شده دروغ است و policy برای آن پاداش می‌گیرد».
**سه‌گانهٔ بازنمایی (D5/D6/D7):** «شبکه ادعا می‌کند ساختار گراف را می‌بیند ولی نمی‌بیند».
**سه‌گانهٔ متا (D12/D13):** مقیاس، ترتیب، و معنای apply.

### فاز E — بازنویسی v0.3 (`e6cf007`، ۷۴ فایل / +۲۰٬۹۶۴ / −۱۲۲)

| # | ناحیه | باگ | ریشه | فیکس |
|---|---|---|---|---|
| E1 | decoder | **باگ ۲: `end_token=2` خودِ اکشن V2V بود** → با `vocab_size=3` دنباله در اولین V2V قطع می‌شد؛ مسیر greedy خراب بود هرچند مسیر sample (که با `sequence_length` تمام می‌شود) آن را در آموزش می‌پوشاند | sentinel از era دوسطحی مانده بود | `end_token=int(vocab_size)` (=۳) با کامنت صریح |
| E2 | process pool | `multiprocessing` از `fork` بعد از init TF/CUDA استفاده می‌کرد | start method پیش‌فرض | `get_context("spawn")` + `CUDA_VISIBLE_DEVICES=""` |
| E3 | encoder | `dropout=0.1` هرگز به `MeanAggregator` نمی‌رسید | پارامتر adapter منتقل نمی‌شد | dropout صریحاً ۰ |

### فاز F — v0.4: ددلاین + obs v3 + رابط masked-PPO

| # | sha | ناحیه | باگ | ریشه | فیکس |
|---|---|---|---|---|---|
| F1 | `8a3d391` | obs | **policy از نظر ساختاری ددلاین را نمی‌دید** (v1/v2 هیچ ویژگی ددلاینی ندارند؛ تغییر ددلاین، observation را بیت‌به‌بیت یکسان می‌گذاشت) | schema قدیمی | obs v3: ۱۶ ویژگی جدید، `FEATURE_DIM 15→31`، `PACKED_DIM 54→70` |
| F2 | `8a3d391` | معنای miss | miss روی **پایان compute** سنجیده می‌شد؛ تسکی که compute‌اش تمام شده ولی داده نرسیده «پاس» می‌شد | مبنای غلط | miss ⟺ `all_consumers_ready > d` (max مصرف‌کننده + hop برگشت sink) + `deadline_basis` روی نتیجه |
| F3 | `8a3d391` | objective | `criticality: float` هم‌زمان *مفهوم* و *وزن جریمه* بود | یک فیلد، دو نقش | تفکیک به `criticality_class` + `tardiness_weight` |
| F4 | `8a3d391` | کران | کران پایین استاتیک برای non-sink ترم «تحویل» خیالی داشت و والدها را سریالی می‌کرد | کران قبل از قواعد residency نوشته شده بود | max موازی والدها، `source_ready_s` = پایان compute والد، بدون تحویل برای non-sink |
| F5 | `08d74d8`→`c504017` | **critic (شدیدترین نقص runtime تاریخ)** | `vf` حدود ۱e۸–۱e۹ می‌شد، value clip بی‌معنا، `vf_loss` منفجر (مقیاس گرادیان ~۱e۱۶) | `q = dense(masked_logits)` — یک لایهٔ dense، `−1e9` را به `q` اکشن‌های معتبر هم می‌رساند | `q = dense(raw_logits)`، `vf = Σ π·q`؛ smoke `max|vf| < 1e3` + بازسازی numpy |
| F6 | `c504017` | mask | ددلاین soft/firm بی‌صدا hard-shield می‌شدند | `feasible_by_deadline` برای *هر* نوع ددلاین «proof» برمی‌گرداند و مستقیم شیلد می‌شد | شیلد فقط روی `deadline_is_hard` گیت شد |
| F7 | `c504017` | قرارداد mask | حالت runtime بی‌صدا به static تنزل می‌کرد | fallback | `observation_mask(mode="runtime")` خطا می‌دهد |
| F8 | `c504017` | تشخیص | `decoder_prediction` (teacher-forced) می‌توانست اکشن ممنوع‌شده را گزارش کند | argmax روی logits بدون mask | روی logits ماسک‌شده |
| F9 | `08d74d8` | سیم‌کشی mask | ماسک فقط `pi` را می‌گرفت و rollout بدون شیلد می‌ماند؛ broadcast ماسک ۳بعدی روی گام‌های `[B,A]` می‌شکست | ماسک به helper نمی‌رسید | ماسک به همهٔ decoderها و helper + ذخیره در batch + خطا در update اگر نباشد |
| F10 | `08d74d8` | numerics | masked `-inf` در `softmax_cross_entropy_with_logits_v2` → NaN و گرادیان NaN | sentinel `-inf` | `NEG_LARGE = −1e9`، dead-end guard، entropy روی اکشن‌های معتبر |
| F11 | `91ca2a1` | shape sampler | `ValueError: actions shape (1000,20) != logits shape (1000,)` | ایندکس‌گذاری دوگانهٔ ماسک/raw | `masking.select_task_batch` با ایندکس تسک + گارد ragged-stack |
| F12 | `7f8f5d2` | CSV | `policy/invalid_action_rate` عدد **۱.۰۹۶** و `policy/argmax_masked_rate` رشتهٔ `validation_query_composite_objective` را می‌خواند | header از set-difference، بازنویسی بدون truncate، و نقل‌نکردن کلید دارای کاما | header به ترتیب درج، truncate و pad هر ردیف، کوت کردن فیلدهای دارای separator |
| F13 | `974f39f` | CSV | بار دوم: header ۴۷ فیلد در برابر ردیف ۴۳ | همان، کامل بسته نشده بود | quoting + چک تعداد فیلد در validator |
| F14 | `c93ddf8` | CSV | smoke یک‌iteration روی کیش **header ۶۴ فیلدی و ردیف اول ۱۲۸ فیلدی** نوشت | دو writer مشترک | `CSVOutputFormat` ردیف‌ها را در حافظه بافر و header+همهٔ ردیف‌ها را بازنویسی می‌کند |
| F15 | `a636a6a` | متریک | drift ~۵e-۴ در entropy که شبیه باگ متریک بود | sample decoder توکن بعدی خود را می‌کشد؛ `sess.run` دوم مسیر دیگری می‌رود | cross-check با `last_sample_pi` ضبط‌شده در rollout |
| F16 | `c0c83c3` | probe | `dump_encoder_activations` با scope `'dump_encoder'` ساخته می‌شد و restore بی‌صدا شکست می‌خورد | scope اشتباه | `scope_name='encoder'` + گزارش `missing_in_ckpt` |
| F17 | `34fd49d` | smoke | سه باگ در اولین ران واقعی TF (fetch بدون feed، `IndexedSlices`، ماسک float32) | harness هرگز اجرا نشده بود | `sample_feed()` مشترک، densify گرادیان‌ها، bool کردن ماسک |
| F18 | `1f7edbd`→`c208685` | ابزار audit | `branch_watch` گزارش `new_commits: []` می‌داد و marker را جلو می‌برد (از دست رفتن دائمی commit) | `ls-remote` شیء را fetch نمی‌کند | fetch به `refs/remotes/...` قبل از log؛ خطای fetch fatal |
| F19 | `c208685` | ابزار | `gpu_run_masked runtime … \|\| true` خطاها را سبز می‌کرد | بلعیدن خطا | حذف `\|\| true` |
| F20 | `c208685` | صحت smoke | «PPO-shaped update» رول‌اوت نمونه‌برداری‌شده را replay نمی‌کرد (init دوباره وزن‌ها را پاک می‌کرد) | باگ harness | فقط init کردن slotهای Adam؛ جفت‌کردن گرادیان‌ها |
| F21 | `60eebfd` | ابزار | SHA ناشناخته قبلی، `merge-base` را غیرقابل‌تصمیم می‌کرد و «not rewritten» خوانده می‌شد | مقایسهٔ undecidable | خطای صریح + `check=True` |
| F22 | `c208685` | گیت | smoke فقط *نوع* exception را assert می‌کرد | assertion ضعیف | assert زیررشتهٔ پیام |

### فاز G — ران‌های mask-sanity و watchdog

| # | sha | باگ | فیکس |
|---|---|---|---|
| G1 | `a9bae0f` | `mask_metrics` ماسک NaN را می‌پذیرفت: `NaN.astype(bool)→True` → گزارش **نادرست ولی باورپذیر** | رد NaN/inf/غیرباینری قبل از `astype(bool)` |
| G2 | `a9bae0f` | target قدیمی `par500` به run dir v0.1 می‌نوشت، off/static را جدا نمی‌کرد، provenance نداشت | entrypoint اختصاصی `spec/mask_sanity.py` با preflight کشنده + اعتبارسنجی CSV پس از ران |
| G3 | `a92b8a8` | هر دو watchdog چند ثانیه بعد از لانچ خارج می‌شدند («هنوز CSV نیست» = تخلف)؛ و `docker ps --format {{.Command}}` **ENTRYPOINT** را نشان می‌دهد نه آرگومان‌های پایتون | معافیت startup + خواندن از `.Config.Cmd` |
| G4 | `7fa31f2` | `docker run` معمولی با قطع ssh می‌مرد | launcher `setsid`+`nohup` |
| G5 | `be2bc09` | آستانهٔ stall ۳۰ دقیقه کمتر از سنگین‌ترین iteration (۲۲.۵ دقیقه) بود | ۴۵ دقیقه |
| G6 | `d543030` | تست‌های CSV از کلیدهای **مصنوعی** استفاده می‌کردند — دقیقاً همان‌جایی که F12 زنده ماند | replay هر ۴۵ کلید واقعی trainer |
| G7 | `c790939` | تست‌ها وقتی TF در `sys.modules` نبود stub نصب می‌کردند | فقط وقتی `find_spec` واقعی پیدا نشود |
| G8 | `df92985` | `test_eas_adapt` با `obs_dim=13` سخت‌کد | استفاده از `PACKED_DIM` |

**G1 از نوع «عدد غلطِ باورپذیر» است:** NaN ماسک، یک شیلد کامل را بی‌صدا گزارش می‌کرد.

### فاز H — ددلاین: dataset و verdict شیلد

| # | sha | باگ | ریشه | فیکس |
|---|---|---|---|---|
| H1 | `de70131`→`78efd56` | **ددلاین‌های مشتق از کران relaxation قابل زمان‌بندی نبودند** (اولین تست‌های fork/join و چند-sink همان‌جا شکست خوردند) | کران صفر-رقابت است ولی موتور روی ۶ منبع تک‌ظرفیتی non-preemptive زمان‌بندی می‌کند — «Feasibility of a relaxation is not feasibility» | `witness.py`: جست‌وجوی قطعی روی موتور واقعی + لنگر کردن ددلاین به ready زمان‌های witness + بازبینی `relaxed_lb_s` |
| H2 | `f831e69` | `active_rate > 0` نمی‌تواند یک رژیم را قضاوت کند | نبود فلگ مستقل | فلگ جداگانهٔ MEC-closure + تفکیک depth/sink/criticality |
| H3 | `92554c5` | **گیت همه‌جا با `mec_closed_rate = 0.0000` شکست می‌خورد** و `active_rate` ۰.۰۱۶–۰.۰۵۵ (زیر کف ۰.۰۵) | MEC در **۶۰۰/۶۰۰** تسک کمترین `ready_lb` را دارد؛ هر ددلاینی که MEC را ببندد، UE/HELPER را هم می‌بندد → ردیف all-invalid → dead-end guard ماسک را رها می‌کند | verdict صادقانه + سه گزینه (runtime prefix shield، تغییر ادعا، تغییر فیزیک) |
| H4 | `fada17e` | ادعای اولیه («MEC واقعاً بهترین نیست») **پس گرفته شد** | ماسک static queue-blind است و لایهٔ runtime هم آگاهی صف ندارد | retract + ثبت ترتیب مجاز |
| H5 | `9d06131` | ۱۵MB sidecar materialized در git | artifact مشتق | gitignore + بازتولید با `python -m spec.deadline_sweep --materialize` |

### فاز I — مهاجرت scope انرژی

| # | sha | باگ | ریشه | فیکس |
|---|---|---|---|---|
| I1 | `2963d49` | **روشن کردن energy فیزیکی، زمان‌بندی را هم عوض می‌کرد** | `ResourceConfig.physical` هم نرخ و هم انرژی را می‌چرخاند | تفکیک محورهای مستقل `timing_model`, `radio_timing_model`, `energy_model`, `radio_model`, `energy_scope` |
| I2 | `b407de9` | **نقص P0 باقی‌مانده از I1**: `static_bounds._cpu_rate` و `feasibility._cpu_rate` هنوز با `resources.physical` انتخاب می‌شدند | مهاجرت ناقص | هر دو به API فقط-timing واگذار شدند؛ تست invariance ضرب‌دری |
| I3 | `d9a960b` | **عدم تطابق scope، ثبت و باز گذاشته شد**: `objective.py` صورت‌حساب `total_system_joules` می‌خواند ولی بودجه/مرجع از `refs.E_ue` (mobile) می‌آمد | `refs.E_ue` نام *پلن* است نه *مرز* | مستند + assert در تست inventory |
| I4 | `55b42f5` | cache مراجع با `id(task_graph)` کلید می‌خورد (ناپایدار در workerها، کور به پارامترها) | هویت شیء به‌جای محتوا | `scheduling_graph_fingerprint` + `reference_ranges_cache_key` |
| I5 | `8139d32` | خطر «زمان‌بندی دوم» توسط telemetry | محاسبه از مسیر دیگر | telemetry صرفاً post-processing همان `ScheduleResult` |
| I6 | `e41b498` | **در `parallel=True` هزینه‌های قید هرگز به trainer نمی‌رسید** (`constraint/updates=0, buffer=0`) | telemetry از worker عبور نمی‌کرد | raw/budget/signed/violation/penalty سوار همان کانال telemetry |
| I7 | `b407de9`+`57dfc5b` | کانال انرژی objective مرزها را قاطی می‌کرد | هر consumer scope خودش را انتخاب می‌کرد | SYSTEM یکدست + `require_reference_scope` |
| I8 | `827d80c` | **انرژی داخل reward primary بود** | قرارداد از publication ارث رسیده بود | reward primary صرفاً latency (`J_t = L_t/L_scale`)؛ publication فقط opt-in |
| I9 | `7faf12d` | در `latency_only` وزن‌های Resources به شاخهٔ publication می‌افتادند | شاخه مدیریت نشده | وزن‌ها صریحاً `(1.0, 0.0)` |
| I10 | `ad436e3` | `Average energy` بین mobile/system مبهم بود | ستون بدون برچسب | pin به MOBILE + `energy/average_energy_scope` |
| I11 | `9f7dedb` | smoke با `resources declare no usable energy_scope (got '')` می‌مرد | `scheduler_config=None` | پاس دادن config resolved + `strict_scheduler_config=True` |
| I12 | `01888d0` | **اعداد held-out با قرارداد legacy محاسبه می‌شدند در حالی که train/val با config resolved بودند** | دومین محل ساخت | evaluator از `evaluator_scheduler_config()` مصرف می‌کند |
| I14 | `da60ada` | یک ردیف pure-plan هم `budget_status=not_configured` و هم `total_energy_status=active` داشت | دو spec در یک ردیف | `pure_plan_evidence_v2` با سناریوهای جدا |
| I15 | `bb77453` | `decoder_order_audit` مستقیم `result.energy.total_system_joules` می‌خواند | دور زدن accessor | `energy_scalar(scope=system)` |
| I16 | `7f5b63a` | تست‌های upstream-parity به کلون `/tmp` اشاره می‌کردند که در ایمیج نیست | وابستگی بیرونی | fixture vendored `spec/fixtures/upstream_metarl/` (sha256 `41c9b90c…`) |
| I17 | `acbbeee` | audit counterfactual هم prefix و هم suffix را عوض می‌کرد | مقایسهٔ کنترل‌نشده | prefix ثابت، suffix پلن پایه |
| I19 | `cef6772` | واگرایی canonical از upstream (خارجی ۳۲.۵۱MB در برابر ۸.۳۹MB، برگشت خروجی کامل در برابر فقط sink، ۴ تقویم در برابر ۳) | تفاوت مدل‌سازی، **نه باگ کد ما** | به‌عنوان factor نام‌گذاری‌شده ثبت شد |
| I20 | `bb77453`/`5dca3f2` | کران استاتیک queue-blind است و **۱ از ۱۰** موقعیت decoder را در `wide_shallow` غلط رتبه می‌دهد (Spearman ۰.۵۳۵) | کران رقابت را نمی‌بیند | اندازه‌گیری و گزارش به‌عنوان OPEN |

**I1/I2 یک نقص زوجی‌اند** و گارد دائمی‌شان تست cross-product است: یک پلن تحت (timing legacy, energy legacy) و (timing legacy, energy physical) باید makespan/bounds/mask/obs **بیت‌به‌بیت یکسان** بدهد.
**I4/I5/I12 یک ریشه دارند:** *محاسبهٔ دوباره* به‌جای *حمل کردن*.

### فاز J — Pilot و ارزیابی

| # | sha | ناحیه | باگ | فیکس |
|---|---|---|---|---|
| J1 | `fe90a11` | ابزار | اسکالرهای KL/clip/grad در CSV **نبودند** | ساخت این opها در گراف هر task داخل `MRLCO` + تجمیع در outer update |
| J2 | `64a15db` | گزارش | گزارش Pilot A **latency آموزش را به‌جای validation** برچسب زده بود | تصحیح: validation itr0 `k0=982.8035s`, `k3=893.3548s`, all-MEC 630.2798, Greedy 626.6017 |
| J3 | `c1cfbfa` | پروتکل | معیار PASS غیرقابل‌اندازه‌گیری: `validation_interval=50` > ۲۵ iteration → فقط یک نقطه (init) | verdict INCONCLUSIVE + تمدید به ۴۰ |
| J4 | `d3ff0c8` | env ارزیاب | ارزیابی چک‌پوینت فوراً می‌مرد: `resource_vec only valid for obs v2/v3` | `configure_obs_env()` قبل از import سیاست |
| J5 | `7b5e8f2` | runner طولانی | همان کلاس خطا در `pilot_long.py` | همان فیکس + ثبت در evidence |
| J6 | `7b5e8f2` | classification | `v == v` NaN را می‌گرفت ولی **Inf را نه**؛ ارزیابی iteration آخر هم وارد trajectory نمی‌شد | `math.isfinite` + افزودن ردیف نهایی |
| J7 | `7b5e8f2` | ارزیاب | چاپ `results['deterministic_k0']` که کلیدش حذف شده بود → `KeyError` **بعد از** نوشتن JSON | چاپ کلیدهای واقعی |
| J8 | `7b5e8f2` | evidence | چهار JSON ارزیابی **خالی** در مخزن (شبیه evidence ولی تهی) | حذف + README با اعداد بازیابی‌شده |
| J9 | `e3fd3cd` | مسیر container | target ارزیابی مسیرهای میزبان را دست‌نخورده پاس می‌داد؛ container فقط `/work` را می‌بیند → JSON در `/tmp` محوشد | بازنویسی مسیرها به `/work` |
| J10 | `e3fd3cd` | probe | probe تعیّن **بعد از** حلقهٔ label اجرا می‌شد (k3 سیاست scratch را جابه‌جا کرده بود) → **non-determinism کاذب** | probe تازه قبل از هر ارزیابی + probe بعد از adaptation با نام جدا |
| J11 | `90631c9` | تست | assertion target طولانی خیلی شل بود | سخت‌تر شد |
| J12 | `efaffbc` | parity | Pilot A با `parallel=True` و runner طولانی با `parallel=False` → مسیر تصادفی/سرعت متفاوت | `parallel=True` + ثبت در contract + تست هم‌ترازی |
| J13 | `ff608dd` | watchdog | runner طولانی فقط **پس از** ران متریک‌ها را می‌دید | `Trainer.pilot_watchdog` درون حلقه + `PilotInstabilityError` + ثبت `STOPPED_UNSTABLE` و خروج غیرصفر |
| J14 | `ff294eb` | تست | helper ردیف inf اشتباه بود | تصحیح |
| J15 | `62c4538` | evidence | verdict فقط در prose یک README بود | `spec/checkpoint_eval_compare.py` به‌عنوان merge خالص و بازاجراپذیر |
| J16-J18 | `6f68fce`,`7f0e46b`,`33e43b1` | probe فیزیک | shape مبهم `summarise`، ورودی ناسازگار `verdict`، و `plan length 0` چون `prioritize_sequence` تا اجرای `prioritize_tasks()` خالی است | `{tiers, n_samples}`، پذیرش هر دو shape، ترتیب HEFT per-config |

**J4/J5 یک باگ در دو entrypoint**: `MARGO_OBS_VERSION` باید **قبل از import سیاست** ست شود و هیچ‌چیز آن را الزام نمی‌کرد.

### فاز K — این نشست: قرارداد هدف، صداقت انتخاب، فیزیک مخلوط

| # | sha | ناحیه | باگ | ریشه | فیکس |
|---|---|---|---|---|---|
| K1 | `f0c0cc8` | objective/انتخاب | **اسکالر لاگ‌شده و انتخاب‌کنندهٔ `meta_model_best_val.ckpt` اشتباه بود**: `composite_query_objective` جمع **بی‌تخفیف** rewardهاست در حالی که PPO بازده تخفیف‌دار `Σγ^(t−1)r_t` را بهینه می‌کند. روی یک rollout: `0.6023961999171015` در برابر `0.555946990805303` | دو objective روی یک مسیر آموزش | `spec/objective_contract.py` (`objective_contract_v1`) + ثبت `validation/objective_discounted_return_k0\|k3` + sidecar `ckpt/meta_model_best_val.metric.json` |
| K2 | `cf7efbf` | انتخاب | (۱) `objective_mode="lexicographic"` **بی‌صدا** به اسکالر legacy برمی‌گشت؛ (۲) در `log_only` انتخاب همچنان روی composite قدیمی بود و این فقط در prose بود | fallback خاموش + منبع نامستند | حذف fallback؛ افزودن `checkpoint_selection_source`, `checkpoint_selection_uses_logged_objective`, `checkpoint_selection_scalar`؛ خواندن `composite` قبل از لاگ |
| K3 | `a1badce` | verdict ارزیابی | `checkpoint_eval_compare` کاندیداها را با **میانگین latency** می‌سنجید — functional متفاوت از objective آموزش | conflate متریک/هدف | ارزیابی روی `objective_k0/k3`؛ latency به‌عنوان همراه (`latency_gain_vs_true_init_seconds`)؛ fallback برچسب‌دار برای JSONهای قبل از قرارداد |
| K4 | `f0c0cc8` | کانال objective | کانال objective سطح‌پلن **تولیدکننده نداشت** (`objective/unavailable`) | نبود producer | env همان `ScheduleResult` و مرجع SYSTEM را به‌عنوان `validation_per_graph_plans` ثبت می‌کند (بدون زمان‌بندی دوم) |
| K5 | `f0c0cc8` | متریک validation | نسبتِ میانگین‌ها به‌جای میانگین per-graph (گراف A بالای بودجه، B خیلی زیر → خطا پنهان می‌شد) | میانگین‌گیری قبل از آستانه | حذف adapter؛ `objective_from_plans` تنها ورودی |
| K6 | `f0c0cc8` | شمارش | شمارش‌ها از `n_graphs` بازسازی می‌شد (`hard_task_total` ۲ به‌جای ۴) | تجمیع مشتق‌شده | تست قفل‌کننده |
| K7 | `c644fc5` | اسناد | ابعاد obs و روش مردهٔ `v0.2-cavia` ناسازگار | اسناد کهنه | pin + علامت مرده |
| K8 | `37a59fe`/`dd0bc5c` | فیزیک | **ژول گزارش‌شده ۱.۶۸–۲.۵۲ برابر `P(f)·T_scheduled` است** چون primary ترکیب `timing=legacy_frozen_rates` + `energy=physical_v1` است (اندازه‌گیری روی ۱۵ گراف واقعی) | دو محور بی‌صدا مخلوط | اندازه‌گیری + gate `require_energy_timing_consistency` + پریست هم‌فیزیک |
| K9 | `90cfea6` | evidence | — | — | اثبات `selection_scalar == objective_k3` مو‌به‌مو + sidecar هم‌ارز + تفاوت عددی با اسکالر قدیمی |

---

## ۳. کلاس‌های تکرارشوندهٔ باگ و ثابت محافظ

1. **عددی که باید *حمل* شود، *محاسبه* می‌شود.** evaluator کانفیگ خودش را می‌ساخت (`01888d0`)؛ telemetry می‌توانست دوباره زمان‌بندی کند (`8139d32`)؛ cache با `id()` کلید می‌خورد (`55b42f5`)؛ probe با `sess.run` دوم ۵e-۴ واگرا شد (`a636a6a`). **ثابت:** `ScheduleResult` (با fingerprint گراف و `scheduler_config_sha256`) تنها منبع عدد است و mismatch **خطا** می‌دهد.
2. **یک سوئیچ که بی‌صدا دو محور را می‌چرخاند.** `ResourceConfig.physical` (`2963d49`) و باقی‌مانده‌اش (`b407de9`). **ثابت:** تست invariance ضرب‌دری.
3. **کرانی که جای «زمان‌بندی دست‌یافتنی» به‌کار می‌رود.** ددلاین لنگرشده به relaxation (`78efd56`)، کران static queue-blind (`acbbeee`)، شیلد runtime بدون صف (`fada17e`). **ثابت:** گواهی با **witness replay** روی موتور واقعی + `relaxed_lb_s ≤ witness_makespan`.
4. **ناهم‌ترازی تعداد فیلد در CSV — سه بار.** (`7f8f5d2`, `974f39f`, `c93ddf8`). **ثابت:** بازنویسی header و **همهٔ** ردیف‌ها از حافظه، quoting فیلدهای دارای جداکننده، و `len(header) == len(row)`.
5. **گزاره‌ای که برای کیس اشتباه جواب می‌دهد.** `end_token=2` داخل `{0,1,2}` (`e6cf007`)؛ `feasible_*` به‌عنوان *proof* برای soft/firm و استفاده به‌عنوان *shield* (`c504017`)؛ `criticality` که هم‌زمان مفهوم و وزن بود (`8a3d391`). **ثابت:** هر query کیسِ خودش را نام می‌برد و caller آن را assert می‌کند (`deadline_is_hard`, `require_reference_scope`, `require_energy_timing_consistency`).
6. **fallback/truncate/default خاموش.** lexicographic → legacy (`cf7efbf`)؛ درجهٔ >۶ در observation truncate و pred فقط `range(0,i)` (`c0cbecc`)؛ `MARGO_OBS_VERSION` پیش‌فرض v1 (`d3ff0c8`, `7b5e8f2`)؛ mask در runtime بی‌صدا static (`c504017`)؛ SHA ناشناخته = «not rewritten» (`60eebfd`). **ثابت:** `.get()` با default یک باگ است؛ state ناموجود/ناهمخوان با مقادیر خطا می‌دهد.
7. **متریک‌هایی که در کپی‌های worker محاسبه می‌شوند، برای trainer نامرئی‌اند.** (`e41b498`, `efaffbc`). **ثابت:** هر چیزی که trainer لاگ می‌کند باید سوار کانال telemetry/`samples_data` باشد و تنظیم parallel باید assert شود.
8. **Evidenceی که ذخیره‌شده به‌نظر می‌رسد ولی نیست.** JSONهای خالی (`7b5e8f2`)، گم شدن JSON در `/tmp` (`e3fd3cd`)، verdict در prose (`62c4538`). **ثابت:** مقایسه یک تابع خالص بازاجراپذیر روی ورودی‌های commit‌شده + manifest با path/bytes/sha256.
9. **functional اشتباه بهینه/انتخاب/نمره‌دهی می‌شود.** composite بی‌تخفیف برای انتخاب (`f0c0cc8`)، latency برای نمره‌دهی (`a1badce`)، انرژی داخل reward latency-only (`827d80c`). **ثابت:** `objective_contract_v1` یک معیار را نام می‌برد و لاگ، انتخاب و ارزیابی همان را می‌خوانند.

---

## ۴. قراردادهای نهایی

### ۴.۱ Reward نهایی

مسیر: `offloading_env.get_reward_batch_step_by_step` (`env/mec_offloaing_envs/offloading_env.py:563`) → `scheduler/reward.py:113` `telescoping_token_rewards`.

ساخت مشترک همهٔ حالت‌ها:
- ترتیب decoder = `prioritize_sequence` (HEFT)؛ `N = 20`.
- پلن موقت `P_t = (a_1..a_t) + fill(UE=0)`، `FILL_UNASSIGNED = 0` (`reward.py:37`).
- هر `P_t` یک‌بار زمان‌بندی می‌شود (`:235`)؛ `L_t = result.makespan_seconds` (`:236`)؛ `E_t = energy_scalar(result, scope=SCOPE_MOBILE)` (`:237`).
- `P_0` از مرجع بدون time‌بندی: `L_0 = refs.L_ue`, `E_0 = refs.E_ue` (`:224-226`).
- `J_t` (`:250-256`)، `r_t = J_{t-1} − discount·J_t` (`:258-260`)، `discount = env.shaping_discount` (`offloading_env.py:621`).

| حالت | وضعیت | فرمول |
|---|---|---|
| `latency_only` | **PRIMARY** (E3.1) | `J_t = L_t / L_scale`, `L_scale = max(L_max−L_min, 1e-12)`؛ `r_t = (L_{t-1} − γL_t)/L_scale`؛ **بدون ترم انرژی** (`reward.py:180-183,192-194`؛ `energy_api.py:257-258`) |
| `publication` | LEGACY، opt-in صریح | `J_t = 0.5·L_t/L_scale + 0.5·E_t/E_scale` با `E_t` در مرز **MOBILE**؛ با `γ=1` دقیقاً `−(0.5ΔL/L_scale + 0.5ΔE/E_scale)` (`reward.py:198-206,237`) |
| `latency_over_all_mec` | تشخیصی | `J_t = L_t / max(refs.L_mec, 1e-12)` (`reward.py:195-197,250`) |

- `discount`/`shaping_discount`: پیش‌فرض تابع `1.0` (`reward.py:127`)؛ در builder primary `0.99` (`meta_trainer.py:713`) که با `discount=0.99` و `gae_lambda=0.95` گام PPO یکی است (`meta_trainer.py:901-903`) ⇒ اتحاد telescoping برقرار است.
- ترم قید روی reward: `attribution="terminal"` (پیش‌فرض) → `r_N −= Σλ_i·violation_i`؛ `"telescoped"` → پخش روی توکن‌ها (`reward.py:262-276`).
- **واگرایی entrypoint:** `build_frozen_primary_stack` پیش‌فرض `latency_only` است (`meta_trainer.py:708`) ولی driver فاز ۴ پیش‌فرض `publication` دارد (`spec/phase4_train_driver.py:158,266`) و wrapperهای n-iteration `latency_over_all_mec` سخت‌کد می‌کنند (`:408,428,…`). **پیش از استناد به هر عدد ۳۵۰۰-iteration این را روشن کنید.**

### ۴.۲ Objective نهایی و انتخاب چک‌پوینت — `objective_contract_v1`

| مورد | مقدار |
|---|---|
| schema | `objective_contract_v1` (`spec/objective_contract.py:40`) |
| reward mode اعلامی | `latency_only` (`:41`) |
| `γ` | `0.99` (`:42`) |
| `N` | `TASKS_PER_GRAPH = TOKENS_PER_PLAN = 20` (`:43-44`) |
| **معیار** (بالاتر بهتر) | `R = mean_trajectories Σ_t γ^(t-1) r_t` (`:87-113`) — همان چیزی که PPO بهینه می‌کند |
| کلید CSV | `validation/objective_discounted_return` (`meta_trainer.py:228`, `frozen_experiment.yaml:274`) |
| همراه‌ها (معیار نیستند) | `validation/objective_legacy_undiscounted_sum_*`, `query_mean_latency` |
| انتخاب چک‌پوینت | `selection_value(metric, k3)` روی **k3**؛ `composite > best_val_objective` (`meta_trainer.py:230,284-286`) |
| artifact انتخاب | `meta_model_best_val.ckpt` + sidecar `meta_model_best_val.metric.json` (`:293-311`) |
| محتوای sidecar | `{schema: best_val_metric_v1, contract, metric_name, csv_key, value, itr, higher_is_better: true, reward_mode, discount, companion_query_mean_latency_seconds}` |

اتحاد کلیدی: `Σ_t γ^(t-1) r_t = J_0 − γ^N J_N`. چون `J_0` ثابت هر گراف و `N` ثابت است، `mean(R) = mean(J_0) − γ^N·mean(J_N)` ⇒ رتبه‌بندی با `R` و با `J_N` یکسان است (`affine_relation`, `ranking_agrees`).

اثبات تجربی: `reports/v0.3-audit/objective_contract/objective_contract_smoke.json` — `selection_scalar == objective_k3 == 0.555946990805303`، `legacy_undiscounted_sum_k3 = 0.6023961999171015`، `latency_k3 = 897.5328s`، `passed: true` (۱۱ چک).
**کاوئات:** این artifact روی `training_code_sha = d74c4477…` ساخته شده، نه HEAD؛ و ران آرشیوی ۳۵۰۰-iteration (`spec/kish_log_archive/runs/margo_v0.1_primary/seed_0/logs/progress.csv`) ستون‌های قرارداد را **ندارد** و هنوز `checkpoint_selection_metric = validation_query_composite_objective` را ثبت می‌کند.

### ۴.۳ Energy نهایی

**قرارداد:** `C_i = D_i·cycles_per_bit`، `T_cpu = C/f`، `E_cpu = κ·C·f² = (κ f³)·T_cpu` (`energy_model.py:1-29`).

**مرزها** (`model.py:296-330`):
- `requester = UE` (compute + UL + DL + V2V TX/RX)
- `mobile = requester + helper` (compute + V2V TX/RX)
- `system = mobile + mec_compute_optional + mec_tx_optional` ← **مرز primary**

**legacy** (`resources.py:211-215`): `E_UE = T·ρ_ue·f_l^ζ`, `E_helper = T·ρ_helper·f_v2v^ζ`, `E_MEC = 0`؛ پارامترها: ρ_ue 1.0, f_l 1.0, ζ 2.0, ptx_mec 0.1, prx_mec 0.05, ptx_v2v 0.06, prx_v2v 0.03, ρ_helper 0.7, f_v2v 1.0 (`frozen_experiment.yaml:85-93`).

**physical_v1** (`frozen_experiment.yaml:136-177`, `energy_model.py:90-93`):

| tier | f | κ | P = κf³ |
|---|---|---|---|
| ue | 1.0e9 | 1.0e-27 | **1.0 W** |
| helper | 1.5e9 | 5.0e-27 | **16.875 W** |
| mec | 10.0e9 | 1.0e-27 | **1000 W** |

`cycles_per_bit = 300.0`، `include_rx_energy = false`، `ue_tx_w = 1.0`، `mec_tx_w = 3.162`، `helper_tx_w = 1.0`، `ue_rx_w/helper_rx_w = null`.
قواعد hop (`energy_model.py:297-316`): MEC_UL → `T·ue_tx_w`؛ MEC_DL → `mec_tx_optional += T·mec_tx_w` (RX فقط اگر فعال)؛ V2V از helper → `helper_v2v_tx += T·helper_tx_w`؛ V2V از UE → `ue_v2v_tx += T·ue_tx_w`.

**مراجع:** `ReferenceRanges` با `L_ue/L_mec/L_helper/E_ue/E_mec/E_helper`، `L_scale = max(L_max−L_min, 1e-12)`، `E_scale` مشابه (`energy_api.py:96-262`)؛ `schema_version = energy_reference_ranges_v2`؛ حالت‌ها `pure_location` و `candidate_panel` (پیش‌فرض primary: `candidate_panel`).
کلید cache (`energy_cache.py:105-137`) = sha256 روی: schema، fingerprint گراف (`scheduling_graph_v1`)، `reference_mode`، `energy_scope`، `panel_max_passes`، `panel_extra_sha256`، `scheduler_config_sha256`، `energy_model`، `radio_model`. فیلدهای ددلاین **عمداً حذف** شده‌اند.

**Telemetry (`energy_telemetry_v1`)**: `requester/mobile/system/primary_joules` + `primary_scope` + `scheduler_config_sha256` + `schema_version`؛ تجمیع وزن‌دار بر حسب episode؛ در حالت قید فعال، `constraint_<name>_raw/_budget/_signed/_violation` و `constraint_penalty_applied` هم سوار همان رکورد می‌شوند. مسیر: env → sampler → processor → `samples_data['energy_telemetry']` → trainer. ستون‌های CSV: `energy/{requester,mobile,system,primary}_joules`, `energy/primary_scope`؛ `Average energy,` به‌عنوان متریک legacy صریحاً MOBILE برچسب خورده.

**Objective سطح‌پلن** (`scheduler/objective.py`): `J = L/L_ref + β_soft·T_soft_norm` با `β_soft = 1.0`, `L_ref = L_ue`؛ کانال‌ها `c_H`, `c_F` (ε=0)، `c_E = min(max(0, E_system/B_E − 1), cost_cap=10.0)`؛ `feasible = hard_ok ∧ energy_ok ∧ firm_ok`.

**Constraints و dual** (`constraints.py`): mode `off|lagrangian` (پیش‌فرض off)، attribution `terminal|telescoped`، ۷ کانال (`ue_energy, helper_energy, total_energy, v2v_airtime, v2v_task_fraction, mec_task_fraction, deadline`)؛ `signed_i = (raw_i − budget_i)/scale_i`، `violation_i = max(0, signed_i)`، `penalty = Σλ_i·violation_i`؛ `λ ← min(1e3, max(0, λ + η·mean(signed)))` با `η = dual_lr = 0.05`، `max_lambda = 1e3`؛ `dual_lr = 0` ⇒ λ برای همیشه صفر می‌ماند.

**قطعی: حلقهٔ دوگانه در sampling باز است.** `observe()` فقط داخل `env.step` روی **کپی**ها اجرا می‌شود (`copy.deepcopy` در `vectorized_env_executor.py:24` یا pickle در `:111/:220`) و trainer روی کنترلر env اصلی `dual_step()` می‌زند (`meta_trainer.py:604`؛ buffer همیشه خالی → `constraints.py:533-535`) ⇒ `constraint/updates` شمارش می‌شود ولی λ صفر می‌ماند. خود trainer این را می‌گوید (`meta_trainer.py:607-609`). شاهد: `reward_A1_equals_off = ... = true` و `lagrangian_off_lambda_stays_zero = true` در `reports/v0.3-audit/constraint_smoke/constraint_smoke_evidence.json`.

### ۴.۴ فیزیک زمان در برابر انرژی (اندازه‌گیری‌شده)

قرارداد primary عمداً **مخلوط** است (`primary_config.py:19-25`): `timing_model = radio_timing_model = legacy_frozen_rates` ولی `energy_model = radio_model = physical_v1`, `energy_scope = system`.

نتیجه: نرخ مؤثر زمان‌بندی (بایت/ثانیه) با نرخ ضمنی مدل انرژی:

| tier | نرخ زمان‌بندی | نرخ فیزیکی (f/(8ξ)) | نسبت |
|---|---|---|---|
| ue | 1,048,576 | 416,666.67 | **2.5165824** |
| helper | 1,048,576 | 625,000 | **1.6777216** |
| mec | 10,485,760 | 4,166,666.67 | **2.5165824** |

پس `T_scheduled = D/R_sched` در حالی که مدت فیزیکی `C/f = D/R_phys` است؛ یعنی `E_workload = κCf² = P(f)·T_phys = P(f)·T_busy/ratio` — **ژول گزارش‌شده `ratio` برابر ژولِ ضمنی مدت زمان‌بندی‌شده است**.

شاهد روی ۱۵ گراف واقعی (`reports/v0.3-audit/energy_consistency/energy_consistency_evidence.json`, verdict `MIXED_PHYSICS_MEASURED`, خطای اتحاد `7.1e-15`):
- mixed: ue busy 510.73s → workload 1285.29 J در برابر duration-consistent 510.73 J (P=1 W) ⇒ توان ضمنی **2.5166 W**؛ helper 429.79s → 12168.08 J در برابر 7252.74 J ⇒ **28.31 W**؛ mec 48.27s → 121482.48 J در برابر 48272.80 J ⇒ **2516.58 W**.
- co-physical: نسبت‌ها دقیقاً **1.0** (ue 1.0 W، helper 16.875 W، mec 1000 W).
- هزینهٔ انتخاب فیزیک هم‌فاز: میانگین makespan از **922.25s به 1304.18s** ⇒ **+41.414%**.

API افزوده: `TierSpec.compute_joules_from_duration` (`energy_model.py:99-107`)، `energy_telemetry.duration_consistency` (`:105-154`) و `duration_consistency_kvs` (`:181-197`)، پریست `PHYSICAL_SCHEDULER_AXES` و `resolved_physical_scheduler_config` و gate `require_energy_timing_consistency` (`primary_config.py:32-91`).
**کاوئات:** gate هیچ فراخوان تولیدی ندارد (فقط تست و probe) و `duration_consistency_kvs` در CSV آموزش سیم‌کشی نشده است.

### ۴.۵ Observation نهایی

انتخاب نسخه با `MARGO_OBS_VERSION` (پیش‌فرض `v1`) که **قبل از اولین import `policies.graph2seq_encoder`** باید ست شود، چون آن ماژول `FEATURE_DIM/PACKED_DIM/FEATURE_NAMES/MAX_NEIGH` را در import bind می‌کند (`encoder_obs.py:145-151`؛ `policies/graph2seq_encoder.py:13-23,58-59`).

| نسخه | FEATURE_DIM | PACKED_DIM | محتوا |
|---|---|---|---|
| v1 | 11 | 50 | پایه |
| v2 | 15 | 54 | +۴ کانال منابع (`log_ul_bps`, `log_v2v_bps`, `log_mec_cpu`, `log_ue_cpu`) |
| v3 | 31 | 70 | +۱۶ کانال ددلاین/feasibility |

`PACKED_DIM = FEATURE_DIM + 2·19 + 1`؛ `MAX_TASKS=20`, `MAX_NEIGH=19`, `PAD_INDEX=−1`.

**v1 (۱۱ ویژگی، ترتیب دقیق):** 0 `compute_workload_bytes`, 1 `task_output_bytes`, 2 `external_input_bytes`, 3 `incoming_edge_bytes`, 4 `outgoing_edge_bytes`, 5 `indegree`, 6 `outdegree`, 7 `decoder_index`, 8 `depth`, 9 `is_root`, 10 `is_sink`.

**v3 ۱۶ ویژگی اضافه** (`encoder_obs.py:76-93`, `_deadline_block` `:437-491`):

| # | نام | فرمول | مقدار وقتی ددلاین ندارد |
|---|---|---|---|
| 15 | `has_deadline` | `1` اگر `deadline_s is not None and deadline_type != "none"` | **0** |
| 16-18 | `deadline_is_soft/firm/hard` | one-hot نوع | 0 |
| 19 | `slack_ratio_min_lb` | `clip((d − min_a ready_lb)/d, −1, +1)` | **0** |
| 20-22 | `criticality_low/medium/high` | one-hot کلاس | medium = **1** |
| 23 | `tardiness_weight_scaled` | `min(1, log1p(w)/log1p(10))` | **0.30103** (وزن پیش‌فرض ۱) |
| 24 | `return_hop_over_min_lb` | sink: `min(4, cheap_return/max(min_ready_lb,1e-9))`؛ غیر sink: 0 | 0 برای غیر-sink، محاسبه‌شده برای sink |
| 25/27/29 | `lb_{ue,mec,helper}_log1p` | `min(4, log1p(max(0,lb)/scale·10))` | محاسبه‌شده |
| 26/28/30 | `feasible_{ue,mec,helper}` | `ready_lb ≤ d` | **1** («رد نشده»، نه «قابل‌اجرا») |

نکتهٔ ریز مهم: `has_deadline` وقتی `deadline_type="none"` صفر می‌شود ولی `slack_ratio_min_lb` و `feasible_*` **صرف‌نظر از نوع** از `deadline_s` حساب می‌شوند (`encoder_obs.py:451-454,490`).

**Layout:** `obs[k] = concat(features[FEATURE_DIM], fw[19], bw[19], mask[1])`؛ ستون آخر **node-validity** است (در تولید همیشه ۱) و ربطی به mask اکشن ندارد؛ ماسک اکشن از کانال‌های `feasible_*` ساخته می‌شود (`masking.py:253-265`).
جداول همسایه: اندیس decoder، مرتب صعودی، `PAD_INDEX=−1`، درجهٔ >19 **خطا** (truncate خاموش ندارد).
**Standardization:** همه به‌جز `is_root`, `is_sink` و ۱۶ ویژگی v3؛ آمار فقط از `meta_train` و hash-pin شده به `dataset_manifest` (`e336074f…`) و `split_policy` (`1ae009a3…`) با canonical-LF. فایل‌ها: `spec/encoder_feature_stats{,_v2,_v3}.json` (v3 مشتق از v2 + ردیف‌های identity، بدون refit). `n_graphs = 1500` در v1 و `19500 = 1500 گراف × ۱۳ resource profile` در v2/v3 — این عدد **نشتی نیست**، همان ۱۳ profile متا-ترین است.

### ۴.۶ State و گراف نهایی

- `CanonicalTask` (`model.py:44-97`): `task_id`, `compute_workload_bytes`, `task_output_bytes`, `external_input_bytes`, `cycles_per_bit` (None→global)، `deadline_s`, `deadline_type`, `criticality_class`, `tardiness_weight`؛ `has_deadline` property.
- `CanonicalEdge(src, dst, edge_output_bytes)` بدون self-loop؛ `CanonicalDAG` با `ConflictingDuplicateEdgeError` برای یال تکراری با وزن متفاوت و یال‌های مرتب‌شده.
- پارسر `.gv` (`offloading_task_graph.py`): فقط `size`→compute، `expect_size`→output، `size` یال→بایت وابستگی؛ `alpha` **خوانده نمی‌شود**؛ شناسهٔ گره ۱-based → ۰-based.
- تبدیل: `external_input_bytes = processing_data_size` فقط برای root؛ یال‌ها از `edge_set` نه `dependency_matrix` (ماتریس duplicates را overwrite می‌کند).
- ترتیب decoder: HEFT upward-rank: `w[i] = min(t_local, t_mec, t_v2v)` و `rank(i) = w[i] + max_{j∈succ} rank(j)`، `argsort` نزولی (`offloading_task_graph.py:233-263`)؛ `adapter.validate_plan` توپولوژیک بودن را چک می‌کند.
- **تنها ورودی سیاست، تنسور obs است** (بدون گراف و بدون context جداگانه)؛ `resource_ctx_from_vec` کد مرده است؛ CAVIA/PEARL(z) در کد هست ولی در استک primary **خاموش** است.

### ۴.۷ Scheduler نهایی

- شش تقویم تک‌ظرفیتی non-preemptive: `UE_CPU, MEC_UL, MEC_CPU, MEC_DL, HELPER_CPU, V2V_CHANNEL` (`calendar.py:50-57`)؛ رزرو = earliest-fit با تلورانس `1e-15` و assert عدم‌همپوشانی `1e-12`.
- جدول route (`routes.py:9-19`): UE→MEC = `[MEC_UL]`؛ UE→HELPER = `[V2V]`؛ MEC→UE = `[MEC_DL]`؛ **MEC→HELPER = `[MEC_DL, V2V]`**؛ **HELPER→MEC = `[V2V, MEC_UL]`**؛ هم‌مکان = صفر hop؛ HELPER→HELPER = صفر.
- مدت انتقال = `bytes / hop_rate(hop)`؛ مدت compute = `bytes / cpu_rate_for_task`؛ `ready = max(ورود ورودی خارجی root، همهٔ ورودهای والدها)`.
- **makespan** = `max` روی sinkها از `finish + Σ hopهای برگشت به UE` (hop برگشت داخل makespan است؛ انرژی نه).
- مبنا: `first_available` = اولین تحویل؛ `all_consumers_ready` = آخرین تحویل (max مصرف‌کننده) و برای sink شامل برگشت به UE ⇒ **پایان compute هرگز «تمام‌شده» حساب نمی‌شود**.
- پارامترها (`frozen_experiment.yaml`): CPU نرخ‌ها 1048576 / 1048576 / 10485760 B/s؛ رادیو 917504 / 917504 / 655360 B/s؛ واحد «Mbps» عملاً **mebibit/s** (`·1024²/8`).
- کلیدهای `capacity/preemptive/duplex` و `include_return_transfer_in_makespan` و `*_mbps` در yaml **فقط سند هستند** (خوانده نمی‌شوند).

### ۴.۸ Decoder و Attention نهایی

Graph2Seq (meanagg, readout triple, 2 لایه) → LSTM دوسطحی با ۱۲۸ واحد → `LuongAttention(128, attention_states, scale=False)` → `Dense(3, use_bias=False)`.
- `q = dense(raw_logits, 3)` و `vf = Σ π·q` — **عمداً از logits بدون mask** تا `−1e9` شیلد به critic نشت نکند (فیکس F5).
- اکشن: slot k ↔ `decoder_order[k]`؛ مقدار ۰/۱/۲ = UE/MEC/HELPER.
- `start_token = 0`، `end_token = vocab_size = 3` و **غیرقابل‌تولید** (خروجی فقط ۳ یونیت دارد) ⇒ طول پلن همیشه ۲۰.
- ماسک: قبل از softmax با `NEG_LARGE = −1e9` (هرگز `−inf`)، dead-end guard، entropy روی support معتبر، و **همان ماسک ذخیره‌شده در رول‌اوت در update هم استفاده می‌شود** (بازمحاسبه نسبت را می‌شکند).
- نمونه‌برداری: `Categorical` روی logits ماسک‌شده؛ temperature پیش‌فرض ۱.۰، `top_p` خاموش؛ greedy = argmax روی logits ماسک‌شده.
- `time_major=False` سخت‌کد؛ encoder output یک‌بار ساخته و در همهٔ decoderها reuse می‌شود.

### ۴.۹ PPO و Meta-Learning نهایی

| hyperparameter | مقدار | منبع |
|---|---|---|
| `inner_learning_rate` | 5e-4 | `frozen_experiment.yaml:239` |
| `outer_learning_rate` | 5e-4 | `:245` |
| `k_steps` | 3 (= ۳ بار Adam apply) | `:240`؛ assert در `MRLCO.py:433-435` و `ppo_offloading.py:261-269` |
| `meta_batch_size` | 10 | `:254` |
| `support_graphs_per_meta_task` | 20 | `:255` |
| `trajectories_per_support_graph` | 1 (طبق قرارداد) | `:256` |
| `ppo_batch_size_trajectories` | 20 | `:253` |
| `policy_clip_epsilon` | 0.2 | `:259` |
| `value_clip_epsilon` | 0.2 | `:260` |
| `entropy_coefficient` | **0.0** و در گراف آموزش **وجود ندارد** | `:261`؛ `MRLCO.py` هیچ `entropy` ندارد |
| `discount_gamma` / `gae_lambda` | 0.99 / 0.95 | `:263-264` |
| `advantage_normalization` | true (over کل batch قبل از انتخاب ۲۰ ردیف) | `:265`؛ `seq2seq_meta_sampler_process.py:93-94` |
| `gradient_clip_norm` | 0.5 | `:266` |
| `vf_coef` | 0.5 | `:267` |
| Adam | β1 0.9, β2 0.999, ε 1e-8 | `:236-238` |
| `inner_optimizer_state` | fresh per meta task | `:268` |
| `outer_update_method` | `mrlco_first_order_mean_pseudogradient` | `:242` |
| outer apply | **یک** بار per meta-batch، Adam ماندگار | `MRLCO.py:239-267` |
| `validation_interval` | 50 | `:271` |
| `shaping_discount` | 0.99 | `meta_trainer.py:713` |
| `outer_iterations` | 3500 | `:270` |
| seeds | 0..4 (campaign) | `phase4_campaign.yaml:21` |

- درون‌حلقه: PPO روی policy هر task با logits ماسک‌شدهٔ ذخیره‌شده، clip 0.2، value clip `old_v + clip(v−old_v, ±0.2)`، `vf_loss = 0.5·max((v−r)²,(v_clip−r)²)`، loss = `surr + 0.5·vf`، **بدون ترم entropy**.
- بیرون: `grad = mean_i[(θ0 − θ_i)/(α·k)]` و یک `apply_gradients` با Adam بیرونی ماندگار.
- validation held-out: adapt روی ۲۰ پشتیبان، گزارش روی ۸۰ query، فقط `k ∈ {0,3}`، ۵ توزیع validation = {2, 6, 10, 16, 17}؛ meta-test = {7, 12, 14, 20, 23}.
- **واقعیت تعداد trajectory:** sampler با `10 × 1 × 20000` توکن ≈ **۱۰۰۰ ردیف در هر meta-task (≈۵۰ در هر گراف پشتیبان)** تولید می‌کند و بعد `select_support_rows(n, 20)` فقط ۲۰ ردیف تصادفی نگه می‌دارد — یعنی «۲۰ trajectory» قرارداد، تعداد *نگه‌داشته‌شده* است نه *تولیدشده*، و آن ۲۰ ردیف لزوماً ۲۰ گراف را پوشش نمی‌دهند.
- Advantage normalization روی **کل** ~۱۰۰۰ ردیف قبل از انتخاب ۲۰ ردیف محاسبه می‌شود.

### ۴.۱۰ V2V نهایی

- یک لینک منطقی روی تقویم `V2V_CHANNEL` در **هر دو جهت** ⇒ half-duplex به‌صورت emergent (کلید `duplex: half` در yaml خوانده نمی‌شود).
- نرخ: 655360 B/s = `5·1024²/8`؛ مسیر MEC↔HELPER دومرحله‌ای و از طریق UE است (بدون لینک مستقیم).
- انرژی: legacy → فرستنده `ptx_v2v_w=0.06`، گیرنده `prx_v2v_w=0.03`؛ physical → فرستنده توان خودش (`helper_tx_w=1.0` یا `ue_tx_w=1.0`)، و RX فقط اگر `include_rx_energy=true` (که نیست) ⇒ در primary **هیچ انرژی RX مدل نمی‌شود**.
- انتخاب HELPER فقط از طریق اکشن ۲ (بدون head جدا)؛ در حالت mask=off هیچ اکشنی ماسک نمی‌شود و در static فقط برای تسک‌های hard-deadline.
- فرض‌های صریح: پهنای‌باند/بهرهٔ V2V با `assumption: true` و توان TX helper «برابر UE» فرض شده؛ هیچ‌کدام منتشرشده نیستند.
- محدودیت‌ها: یک helper، تقویم ظرفیت ۱، helper compute و V2V می‌توانند هم‌پوشانی داشته باشند، انرژی helper در objective سطح‌پلن نیست (فقط کانال قید)، و V2V با وضعیت منابع mask نمی‌شود.

### ۴.۱۱ Mask / Shield نهایی

حالت‌ها: `off` (پیش‌فرض) / `static` / `runtime`.
- `static`: از سه کانال `feasible_*` ماسک می‌سازد و **فقط روی `deadline_is_hard`** گیت می‌کند ⇒ تسک بدون ددلاین یا soft/firm کاملاً باز می‌ماند.
- `runtime`: **اجرا نمی‌شود** — `observation_mask` خطا می‌دهد چون شیلد per-prefix به state تقویم نیاز دارد که در observation نیست.
- dead-end guard: اگر هیچ اکشنی feasible نبود، ماسک رها می‌شود و شمارش می‌شود.
- **مسئلهٔ wording:** کد `feasible_*` را «proof feasibility» می‌نامد (`feasibility.py:1-11`, `masking.py:16-19`, `encoder_obs.py:409-412`) در حالی که کران، **optimistic، queue-blind و prefix-blind** است. عبارت درست: «feasibility تحت relaxation خوش‌بینانه»، نه «proof».
- سیم‌کشی نهایی: در همهٔ ران‌های ثبت‌شده `mask_mode = off` است و preflight در `mask_sanity` رکورد `deadlines = 0` می‌دهد؛ یعنی static عملاً no-op بوده است.

### ۴.۱۲ Deadline و Criticality — چطور اضافه شد

**۱) مدل:** چهار فیلد روی `CanonicalTask` (`model.py:58-64`): `deadline_s` (None)، `deadline_type` ∈ {none, soft, firm, hard}، `criticality_class` ∈ {low, medium, high}، `tardiness_weight` (1.0). کد صریحاً می‌گوید کلاس از کانال ددلاین و از وزن جریمه **جدا** است (`model.py:37-40,60-63`). اگر `.gv` هیچ فیلدی نداشته باشد: همهٔ تسک‌ها `deadline_s=None`, `type="none"`, `class="medium"`, `w=1.0` (`adapter.py:44-47`).

**۲) معنای زمان:** miss روی `all_consumers_ready` سنجیده می‌شود: `tardiness = max(0, all_consumers_ready − d)`؛ soft → `Σ w·t/d`، firm → شمارش miss، hard → شمارش miss با ε=0 (`engine.py:230-251`, `objective.py:234-249`). `deadline_basis = "all_consumers_ready"` روی نتیجه ثبت می‌شود. سند `DEADLINE_SEMANTICS.md:14-15,45` **کهنه** است (مبنای قدیمی `F_available` را توصیف می‌کند).

**۳) تولید (sidecar v1):** `deadline_regime.py` + `witness.py` + `static_bounds.py`:

```
eft[i]      = min_a bounds.ready_lb[i][a]                # sink شامل hop برگشت
dur[i]      = eft[i] − max_p eft[p]
graph_lb    = max_i eft[i]
D_G         = kappa · graph_lb
LFT[i]      = D_G اگر بدون successor، وگرنه min(D_G, min_{j∈succ}(LFT[j] − dur[j]))
d_relaxed[i]= eft[i] + alpha·(LFT[i] − eft[i])
d_anchor[i] = eft[i] + alpha·(kappa·W[i] − eft[i])       # W = all_consumers_ready پلن witness
d_i         = d_anchor اگر anchor_ready_s داده شده باشد، وگرنه d_relaxed
floor       = max(1e-9, 1e-6·max(graph_lb,1e-9))         # clamp با شمارش
```

**۴) Witness:** `find_fastest_plan` (greedy از all-MEC + جست‌وجوی یک‌توکنی) → `generate_graph_deadlines(anchor_ready_s=...)` → stamp → `find_witness(seed_actions=fastest.actions)`. معیار witness: `hard_miss_count·1e6 + makespan`؛ verdict نهایی یک **replay واقعی** روی موتور است.

**۵) Schema sidecar (`deadline_regime_v1`):** `schema_version, generator_version, generator_commit, regime, kappa, alpha, deadline_type, cycles_per_bit, construction, ordering, criticality_policy, tardiness_weight_policy, seed, source_manifest_sha256, resources_sha256, provenance{energy_model,radio_model,energy_scope}, graphs{"<dist>/<file>.gv": {content_sha256, tasks{tid:{deadline_s,deadline_type,criticality_class,tardiness_weight}}, bounds{...}, witness{found,actions,makespan_s,hard_miss_count,method,evaluations,per_task{...}}}}`.

**۶) Criticality:** از عمق DAG با quantile ۲۵٪/۷۵٪ و sinkها همیشه high (`deadline_regime.py:375-388`)؛ وزن‌ها `{high:2.0, medium:1.0, low:0.5}` (`:391`). **این یک proxy ساختاری است، نه مدل mixed-criticality**: کد صریح می‌گوید (`model.py:37-40`, `deadline_regime.py:376`) و کلاس هیچ‌جا برای تصمیم زمان‌بندی خوانده نمی‌شود. توزیع در sidecar متا-ترین: high 14664 / medium 13478 / low 1858.

**۷) پارامترها:** `kappa` (مقیاس بودجه، required؛ committed 1.05)، `alpha` (درون‌یابی، committed 1.0)، `deadline_type`، `cycles_per_bit = 300.0`، `allow_infeasible=False`، `big_m=1e6`، `max_passes=4`، `max_pair_rounds=2/3`، `max_pairs=40/60`، `deadline_floor`، `DEFAULT_CLASS_WEIGHTS`، و جدول رژیم‌ها (`none`, `loose_hard(1.50,1.00)`, `medium_hard(1.10,0.50)`, `tight_hard(1.02,0.25)`, `infeasible_labelled(0.80,0.25,غیرقابل‌آموزش)`, `soft_mix`, `firm_mix`) در `spec/deadline_sweep.py:41-49`.

**۸) اعداد committed (`reports/v0.3-audit/deadline_sweep.json`، `limit=8/dist`):**

| κ | α | split | witness | active | MEC closed | graphs with MEC closure | passes |
|---|---|---|---|---|---|---|---|
| 1.05 | 1.0 | meta_train | 1.000 | 0.0421 | **0.0000** | 0 | false |
| 1.05 | 1.0 | validation | 1.000 | 0.0550 | **0.0000** | 0 | **true** |
| 1.05 | 1.0 | meta_test | 1.000 | 0.0163 | **0.0000** | 0 | false |
| 1.25 | 1.0 | (سه split) | 1.000 | 0.0088–0.0279 | **0.0000** | 0 | false |
| 1.10 | 0.5 | meta_train | 0.000 | 0 | 0 | 0 | خطا: هیچ گرافی witness نداد (۶۰/۶۰) |

- sidecarهای committed (`spec/deadline_regimes/loose_hard_*.json`): ۱۵۰۰/۵۰۰/۵۰۰ گراف، witness ۱۰۰٪، **همه با `method="anchored_seed"` و `evaluations=1`** ⇒ چون `alpha=1`، `d_i = κ·W_i` و همان پلن، witness خودش می‌شود: **نرخ witness دوری (circular) است**.
- **باگ گیت:** `checks["passes"]` در `deadline_gates.py:225-229` شرط `mec_closure_present` را که در `:223` محاسبه می‌شود **حذف کرده**؛ به همین دلیل split validation با `mec_closed_rate=0` «pass» می‌شود، در حالی که سند طراحی همان را fail می‌نامد و تست `test_deadline_gates.py:143-150` این رفتار را به‌عنوان «trainable but flagged» قفل کرده است.
- **مکانیزم:** روی ۶۰۰ تسک، MEC در **۶۰۰/۶۰۰** کمترین `ready_lb` را دارد (نسبت UE/MEC ۱.۱۲–۱۰، میانگین ۴.۸۶) ⇒ هر ددلاینی که MEC را ببندد، UE و HELPER را هم می‌بندد، ردیف all-invalid می‌شود و dead-end guard ماسک را رها می‌کند. پس شیلد static **نمی‌تواند** بگوید «MEC ممنوع، بقیه مجاز».
- **سیم‌کشی: هیچ‌کدام.** grep نشان می‌دهد `stamp_task_graph`/`build_regime_with_witness`/`DeadlineRegime` فقط در `spec/deadline_sweep.py` استفاده می‌شوند؛ `offloading_env.py:452` گراف را می‌سازد و بلافاصله در `:463` encode می‌کند (بدون hook) ⇒ **هیچ ران آموزشی تا امروز ددلاین نداشته** و `mask/active_rate = 0` در همهٔ CSVها. هیچ env var یا فلگ deadline هم وجود ندارد.
- `generator_commit` و `source_manifest_sha256` در هر سه sidecar **خالی** است (provenance ناقص)، و provenance آن‌ها `legacy/legacy/mobile` است.
- نقایص کد در همین ناحیه: `return_hop_over_min_lb` در عمل همیشه صفر است (چون min روی `ACTION_LOCATIONS` شامل UE است و `transfer_lower_bound(src==dst)=0`)؛ ۱۳ از ۱۶ کانال v3 روی دیتاست فعلی ثابت‌اند؛ و ترکیب «`deadline_s` داده‌شده + `deadline_type="none"`» باعث ناسازگاری feature/شیلد می‌شود.

---

## ۵. فرض‌ها و محدودیت‌های صریح

1. **دیتاست مصنوعی است**: ۳۴۰۰ فایل `.gv` = ۲۵۰۰ در `meta_offloading_20` (۲۵ توزیع × ۱۰۰) + ۹۰۰ در `meta_offloading_n` (۹ توزیع، n=10..50 که **هیچ ارجاع کدی ندارند** و با `MAX_TASKS=20` قابل مصرف نیستند). بازتولید بایت‌به‌بایت ممکن نیست: seed تولید ثبت نشده، daggen در مخزن نیست و headerها timestamp دارند. آنچه pin شده، **محتوای** فایل‌هاست (`raw_sha256` + `canonical_graph_sha256`).
2. **split ثابت و قابل بازبینی**: grid لاتین (`fat_i==dens_i`→validation، `fat_i==(dens_i+2)%5`→meta_test) و `stratified_sha256_rank_v1` با seed `7ccf0bc4…`؛ ۱۵/۵/۵ توزیع؛ ۲۰ support / ۸۰ query.
3. **ددلاین‌ها ساختگی‌اند**: هیچ SLA واقعی وجود ندارد؛ رژیم‌ها «synthetic deadline regime» هستند و باید همین‌طور نام‌گذاری شوند.
4. **criticality یک proxy ساختاری از عمق DAG** است، نه کلاس معنایی؛ هیچ مدل LO/HI، هیچ WCET چندحالته و هیچ رفتار overload وجود ندارد.
5. **الگوریتم متا**: `mean((θ0−θi)/(αk))` با Adam بیرونی (نه Reptile و نه FOMAML)؛ مقیاس `/(αk)` عملاً توسط RMS نرمال‌سازی Adam خنثی می‌شود.
6. **بازده آموزش = π** و سیاست با CAVIA/context خاموش است.
7. **فیزیک مخلوط**: زمان از جدول legacy و انرژی از مدل فیزیکی (بخش ۴.۴) ⇒ ژول گزارش‌شده ≠ `P·T` زمان‌بندی‌شده.
8. **انرژی در policy بی‌اثر است** در قرارداد فعلی (`latency_only`) و بودجهٔ objective (`1e12 J`) عملاً غیرمقیدکننده است.
9. **قید فعال هرگز اجرا نشده**: هیچ رانی `dual_lr > 0` نداشته و حلقهٔ dual در sampling باز است.
10. **تک‌seed**: همهٔ ران‌های ثبت‌شده seed 0 هستند؛ نویز بین دو ران ۵۰۰تایی: `AverageReturn` اختلاف میانگین ۱.۱۹٪، latency ۱.۰۹٪، energy ۲۶.۶٪ ⇒ مقایسه‌های <۵٪ بی‌معنا.
11. **ارزیابی held-out تعیّن‌پذیر نیست**: probe تازه `deterministic_k0_fresh = false` برای هر سه label ⇒ verdict `BLOCKED_CHECKPOINT_EVALUATION`.
12. **deadline/criticality/v2v-shield هیچ شاهد آموزشی ندارند** (بخش ۴.۱۲).
13. **پیش‌فرض‌های واگرا بین entrypointها**: `reward_mode` (`latency_only` در builder، `publication` در driver)، و `scheduler_config` (پاس‌داده‌شده و strict در پایلوت/ارزیاب، `None` در driver فاز ۴ ⇒ energy_model=None و scope خالی در آموزش).
14. **گزارش‌های تاریخی خالی/آلوده**: `extension_40` حاوی artifactهای ران ۵۰۰تایی است؛ `constraint_smoke_evidence.json` کلیدی دارد که اسکریپت کامیت‌شده تولیدش نمی‌کند؛ `pilot_a_evidence.json` کانفیگش hard-coded است؛ لاگ‌ها و sidecarها و `.ckpt`ها gitignore شده‌اند و در clone وجود ندارند.
15. **دوگانگی مستندات**: `OBJECTIVE_AND_ENERGY.md` هنوز reward قدیمی (۰.۵/۰.۵ با `total_mobile_joules`) را «frozen» می‌نامد؛ `MASKED_PPO_INTERFACE_6b.md §3.1` ابعاد `19/23/31` را می‌گوید که با ۵۰/۵۴/۷۰ نمی‌خواند؛ `FINAL_DATAFLOW.md`/`FINAL_README.md`/`FINAL_IMPLEMENTATION_PLAN.md`/`CURRENT_ARCHITECTURE.md` همچنان `MARGO-METHOD-v0.2-cavia` را حمل می‌کنند (ADR-007 آن را مرده اعلام کرده).

---

## ۶. وضعیت فعلی و شکاف‌ها

**آنچه واقعاً کار می‌کند و شاهد دارد:**
- مسیر latency-only / mask-off / constraints-off / بدون‌ددلاین: کامل سیم‌کشی‌شده و با دو ران واقعی GPU (۲۵ و ۴۰ iteration) و یک جفت ران ۵۰۰تایی مستند است.
- موتور کانونیکال، obs v1/v2/v3، جداول همسایه، standardization با hash-pin، PPO/meta با قرارداد مستند، telemetry سه‌مرزی، و instrumentation پایلوت.
- از این نشست: قرارداد واحد هدف + اثبات یک‌iteration، ارزیابی واقعی چک‌پوینت‌ها با verdict صادقانه، و اندازه‌گیری + گیت فیزیک.

**شکاف‌های باز (به ترتیب اهمیت):**
1. **ارزیابی تعیّن‌پذیر نیست** ⇒ هیچ verdict ارزیابی معتبر نیست تا k0/k3 seeded و paired شود (وگرنه اختلاف چند ثانیه‌ای بی‌معناست).
2. **فیزیک زمان/انرژی یکی نیست** و انتخاب هم‌فیزیک ۴۱.۴٪ makespan را عوض می‌کند ⇒ تصمیم رسمی لازم است (یا هر ادعای انرژی با کاوئات نسبت ۲.۵۲/۱.۶۸).
3. **حلقهٔ dual باز است** ⇒ قید فعال عملاً غیرقابل‌یادگیری با سیم‌کشی فعلی.
4. **گیت ددلاین MEC-closure را در `passes` ندارد** و هیچ رژیمی MEC را نمی‌بندد؛ و sidecar به loader وصل نیست.
5. **«۲۰ support trajectory» در واقع ~۱۰۰۰ تولید و ۲۰ انتخاب تصادفی است** (پوشش گراف تضمینی نیست) و advantage normalization روی کل استخر انجام می‌شود.
6. **`MARGO_TRAINING_CODE_SHA`/`MARGO_EVAL_CODE_SHA` هیچ‌جا ست نمی‌شوند** ⇒ ران طولانی `""` ثبت می‌کند (در merge جدید override می‌شود).
7. **ران ۱۰۰۰تایی هرگز اجرا نشده** و `best_val` در ران‌های <۵۰ iteration فقط مدل ۱-آپدیتی است.
8. **ران آرشیوی ۳۵۰۰-iteration زیر قرارداد جدید نیست** (ستون‌های objective را ندارد).
9. اسناد و evidence آلوده/کهنه (بند ۱۴ و ۱۵ بالا) و نبود `manifest` یکدست برای همهٔ artifactها.

---

## ۷. جدول پارامترهای تجمیعی (خلاصهٔ عملیاتی)

| گروه | پارامتر | مقدار |
|---|---|---|
| داده | توزیع‌ها (meta_train / validation / meta_test) | ۱۵ / ۵ / ۵ از ۲۵ |
| داده | support / query هر توزیع held-out | ۲۰ / ۸۰ |
| داده | `MAX_TASKS`, `MAX_NEIGH`, `PAD_INDEX` | ۲۰، ۱۹، −۱ |
| obs | v1/v2/v3 → FEATURE_DIM | ۱۱ / ۱۵ / ۳۱ |
| obs | v1/v2/v3 → PACKED_DIM | ۵۰ / ۵۴ / ۷۰ |
| obs | نسخه | `MARGO_OBS_VERSION=v3` در پایلوت/ارزیاب |
| reward | mode primary | `latency_only`؛ `J_t = L_t/L_scale` |
| reward | `shaping_discount` / PPO `γ` / `λ` | ۰.۹۹ / ۰.۹۹ / ۰.۹۵ |
| هدف | معیار | `validation/objective_discounted_return` (k3، بالاتر بهتر) |
| زمان‌بندی | تقویم‌ها | ۶ × ظرفیت ۱، non-preemptive |
| زمان‌بندی | نرخ CPU (UE/helper/MEC) | 1,048,576 / 1,048,576 / 10,485,760 B/s |
| زمان‌بندی | نرخ رادیو (UL/DL/V2V) | 917,504 / 917,504 / 655,360 B/s |
| انرژی | مدل | `physical_v1`, scope `system` |
| انرژی | f و κ (ue/helper/mec) | 1e9,1e-27 / 1.5e9,5e-27 / 1e10,1e-27 |
| انرژی | `cycles_per_bit`, `include_rx_energy` | ۳۰۰، false |
| انرژی | توان TX (ue/mec/helper) | ۱.۰ / ۳.۱۶۲ / ۱.۰ W |
| قید | `dual_lr`, `max_lambda` | ۰.۰۵، ۱e۳ (هرگز فعال نشده) |
| قید | بودجه‌های profile | mec≤0.50، ue≤0.60·E_ue، helper≤0.35·E_ue، v2v≤0.15، airtime≤0.60·L |
| آموزش | inner/outer lr | ۵e-4 / ۵e-4 |
| آموزش | `k_steps`, meta-batch, ppo-batch | ۳، ۱۰، ۲۰ |
| آموزش | clip policy/value, vf_coef, grad-clip | ۰.۲، ۰.۲، ۰.۵، ۰.۵ |
| آموزش | entropy | ۰.۰ (در گراف آموزش وجود ندارد) |
| آموزش | validation_interval / outer_iterations | ۵۰ / ۳۵۰۰ |
| ددلاین | `kappa`, `alpha` (committed) | ۱.۰۵، ۱.۰ (`loose_hard`) |
| ددلاین | انواع | none / soft / firm / hard |
| ددلاین | وزن کلاس‌ها | high ۲.۰ / medium ۱.۰ / low ۰.۵ |
| mask | حالت‌ها | off (پیش‌فرض) / static / runtime (غیرقابل‌اجرا) |
| mask | مقدار logit ماسک | −1e9 |

---

## ۸. پیوست

**SHAهای کلیدی:** `37acb69` (ایمپورت) · `53fe08d` (split/protocol) · `937e4c5`/`b49b60f` (موتور کانونیکال) · `c0cbecc`/`367c10e` (obs و encoder) · `fdfc98b`/`f246006` (PPO/meta) · `e6cf007` (v0.3) · `8a3d391` (ددلاین+obs v3) · `08d74d8`/`c504017` (شیلد و critic) · `de70131`/`78efd56`/`f831e69`/`92554c5` (ددلاین regime) · `2963d49`/`b407de9`/`8139d32`/`e41b498`/`827d80c`/`01888d0` (انرژی) · `fe90a11`/`1eaa438` (Pilot instrumentation) · `03419ff`/`64a15db` (P1/P2) · `ff608dd`/`62c4538`/`efaffbc`/`cf7efbf` (فیکس‌های پایلوت) · `f0c0cc8`/`a1badce`/`d74c447`/`56cc4a2`/`37a59fe`/`dd0bc5c`/`90cfea6` (این نشست).

**مسیرهای evidence:** `reports/v0.3-audit/` (DIAGNOSTIC_REPORT_PARTS_A_D.md، deadline_sweep.json، DEADLINE_SEMANTICS.md، mask_smoke/، queue_audit/، upstream_parity/، rank_audit/، energy_smoke/، constraint_smoke/، pilot/، checkpoint_eval/، objective_contract/، energy_consistency/) · `spec/kish_log_archive/` · `spec/deadline_regimes/*.json` · `spec/encoder_feature_stats*.json` · `docs/{CODE_AUDIT,LOGIC_AND_LEARNING_BUGS,FULL_TRAINING_PROCESS}.md`.

**دستورهای کلیدی:** `spec/kish_gpu.sh energy-tests` (سوئیت کامل؛ آخرین بار **۸۹۸ تست OK** روی ایمیج TF1.15) · `checkpoint-eval` / `checkpoint-eval-compare` · `energy-consistency` · `pilot-a-long-1000` (اجرا نشده) · `python -m spec.objective_contract_smoke --itr 1 --i-allow-gpu` · `python -m spec.deadline_sweep --materialize`.

**قواعد ثابت مخزن:** هر عدد از `ScheduleResult` با fingerprint؛ هر محور فیزیک جدا و assert‌شده؛ هر کانال بدون budget = `not_configured` صریح؛ هر claim انرژی باید یا هم‌فیزیک باشد یا نسبت ۲.۵۲/۱.۶۸ را کاوئات کند؛ هر verdict ارزیابی فقط با `objective_contract_v1` و ارزیابی seeded معنا دارد.
