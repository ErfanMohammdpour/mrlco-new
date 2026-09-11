# MARGO چیست و چرا زمان‌بند روی GPU اجرا نمی‌شود

این یادداشت توصیف هنجاری سیستم **منجمد** `MARGO-SPEC-v0.1` است، نه ادعای نتیجهٔ مقاله.
اعداد زمان‌بندی از learning probeٔ پنج‌تایی (`margo_v0.1_learning_probe`، seed 0، kish RTX 4090) آمده‌اند و `paper_result=false` هستند.
`readme.md` ریشهٔ مخزن در چند نقطه کهنه است (Reptile، ویژگی ۲۰بعدی، ۱۹ توزیع، درصدهای مقالهٔ قبلی). منبع حقیقت: `spec/` و تگ‌های freeze.

---

## 1. پروژه چیست

**MARGO** = Meta-learning with Attention-augmented gRaph-to-sequence for energy-aware task Offloading.

یک سیاست meta-RL یاد می‌گیرد که هر گره از یک DAG کاربردی را به یکی از سه محل اجرا بفرستد:

| اکشن | محل | معنی |
|------|------|------|
| `0` Local | `UE` | اجرای محلی روی خود خودرو |
| `1` MEC | `MEC` | آفلود به سرور لبه (RSU/BS) |
| `2` V2V | `HELPER` | آفلود به یک خودروی کمکی |

هدف مشترک تأخیر–انرژی با وزن ثابت `0.5 / 0.5`:

- تأخیر علمی: `makespan_seconds` (زمان تکمیل کل اپلیکیشن، با بازگشت خروجی sinkها به UE)
- انرژی علمی: `total_mobile_joules` = انرژی UE + انرژی compute هلپر + انرژی رادیوی V2V هلپر
- انرژی compute خود MEC در هدف اولیه نیست (فقط حسابداری اختیاری)

قرارداد انرژی: `spec/OBJECTIVE_AND_ENERGY.md` و `ADR-001`.
قرارداد زمان‌بندی: `spec/SCHEDULING_SEMANTICS.md`، `ADR-002` (محل داده)، `ADR-003` (ظرفیت منبع).

### 1.1 مسئلهٔ فیزیکی

یک اپلیکیشن ۲۰ تسکی با یال‌های وابستگی. تسک `j` تا وقتی خروجی `i` به **محل اجرای** `j` نرسیده شروع نمی‌شود. خروجی هر تسک همان‌جا می‌ماند که اجرا شده (`output_location = execution_location`).

منابع تک‌ظرفیت و غیرقابل‌پیش‌دستی:

- `UE_CPU`
- `MEC_UL` (آپلینک به MEC)
- `MEC_CPU`
- `MEC_DL` (دانلینک از MEC)
- `HELPER_CPU`
- `V2V_CHANNEL` (نیم‌دوبلکس: ارسال و دریافت هم‌زمان ممنوع)

مسیر ارتباط برای هر جفت محل پیش‌نیاز/پس‌نیاز در ماتریس `SCHEDULING_SEMANTICS.md` ثابت است؛ مثلاً MEC→HELPER یعنی `MEC_DL` سپس `V2V`.

این مدل **شبیه‌سازی رویداد گسسته** است، نه ضرب ماتریس.

### 1.2 داده و اسپلیت

۲۵ خانوادهٔ توزیع (`distribution_id`)، هر کدام ۱۰۰ گراف ۲۰تسکی.
اسپلیت لاتین منجمد `MARGO-SPLIT-v1` (`ADR-004`):

- `meta_train`: ۱۵ توزیع — تنها جایی که outer loop از آن سمپل می‌کند
- `validation`: `{2, 6, 10, 16, 17}` — فقط انتخاب چک‌پوینت، نه تیون
- `meta_test`: `{7, 12, 14, 20, 23}` — گزارش نهایی؛ تا پایان ترین دست‌نخورده

هر توزیع held-out به support (۲۰ گراف) و query (بقیه) جدا می‌شود. Adaptation روی support، متریک روی query.

### 1.3 سیاست (تنها بخشی که GPU می‌بیند)

ورودی: مشاهدهٔ packed با `obs_dim = 50` (`FEATURE_DIM + 2*MAX_NEIGH + 1`، `MAX_NEIGH = 19`). همسایه = predecessor **و** successor روی DAG کانونیکال، نه clique. تجمیع: masked mean دولایه. Dropout انکودر `0.0`. (`phase2-freeze-v0.1`)

سپس:

- Graph2Seq → embedding گره
- triple readout (attention + mean + max) → بردار گراف
- LSTM decoder با Luong attention روی **گره‌ها** → برای هر تسک logits سه اکشن

حدود ۸۲۴ هزار پارامتر trainable برای **یک** core policy. ده اسلات meta-task کپی همان بردارند، نه ده شبکهٔ جدید.

### 1.4 یادگیری (منجمد Phase 3)

این Reptile نیست.

هر outer iteration:

1. همگام‌سازی ده task-policy از core (`theta0`)
2. سمپل ۱۰ توزیع از ۱۵ توزیع `meta_train`
3. برای هر توزیع ۲۰ trajectory کامل (یک پلن برای هر گراف)
4. inner PPO: `k_steps = 3` یعنی **سه بار Adam apply**، optimizer تازه برای هر meta-task، clip سیاست و ارزش `0.2`
5. سمپل تازه با سیاست adapted (query/eval support)
6. outer: `mean((theta0 - theta_i) / (alpha * k))` سپس **یک** Adam روی core با `beta = 5e-4`

نام روش: `mrlco_first_order_mean_pseudogradient`.
بودجهٔ مقاله: `outer_iterations = 3500`. تشخیصی ۵ / ۲۰۰ / ۱۰۰۰ جداست و `paper_result=false`.

قرارداد: `spec/LEARNING_PROTOCOL.md`، تگ `phase3-freeze-v0.1`.

### 1.5 فازها

| فاز | تگ | معنی |
|-----|-----|------|
| 0 | `phase0-freeze-v0.1` | ADRها، اسپلیت، تعریف outer |
| 1 | `phase1-freeze-v0.1` | زمان‌بند کانونیکال + انرژی + پاداش telescoping |
| 2 | `phase2-freeze-v0.1` | انکودر DAG packed |
| 3 | `phase3-freeze-v0.1` | PPO clip، mean-PG، sync، eval روی اسپلیت |
| 4 | در جریان | کمپین ارزیابی؛ تا آرتیفکت خام نباشد رقم مقاله نیست |

---

## 2. دو ماشین جدا در یک outer iteration

یک iter معمولی روی kish (بدون validation) حدود **۱۰٫۴ دقیقه** است. این دو بلوک **یکسان‌اند**؛ دو بار ۴٫۷ دقیقهٔ جدا نیستند.

| بلوک | کجا | زمان تقریبی | چیست |
|------|------|-------------|------|
| `obtain_samples_ppo` | CPU تقریباً همه | ~۵٫۱ دقیقه | ۲۰۰ پلن → پاداش |
| از این: `PolicyExecTime` | GPU | ~۲۸s (~۰٫۵ دقیقه) | `get_actions` |
| از این: `EnvExecTime` | CPU | ~۲۸۰–۲۸۹s (~۴٫۷ دقیقه) | `env.step` + telescoping + `schedule()` |
| inner PPO `k=3` | GPU | چند ثانیه | سه Adam روی ۱۰ task |
| `obtain_samples_eval` | مثل اول | ~۵٫۲ دقیقه | دوباره ۲۰۰ پلن |
| outer Adam | GPU/CPU ناچیز | &lt;۱s | یک mean-PG |
| validation هر ۵۰ iter | عمدتاً CPU | ~۱۰٫۷ دقیقه اضافه | ۵ dist × k=0 و k=3 |

پس از حدود ده دقیقه، **نه دقیقه CPU زمان‌بند** است و **یک دقیقه GPU سیاست**.
موقع rollout، `nvidia-smi` utilization صفر است؛ VRAM فقط نشست TF (~۶۵۰ MiB) را نگه می‌دارد.

کد صریح است: `Seq2SeqMetaSampler(..., parallel=False)` → `MetaIterativeEnvExecutor` ده محیط را **پشت‌سرهم** `step` می‌کند، نه روی ده هسته.

---

## 3. قرارداد محیط: پلن کامل، نه MDP گام‌به‌گام

`SYSTEM_SPEC.md` §4: MARGO v0.1 یک **macro-action autoregressive planner** است.

1. Decoder کل دنبالهٔ طول `N=20` را بدون دیدن تقویم منابع بعد از هر توکن می‌سازد.
2. محیط **یک‌بار** پلن کامل را زمان‌بندی می‌کند.
3. پاداش توکن‌به‌توکن **بعد از واقعیت** از روی چند زمان‌بندی موقت ساخته می‌شود.

سیاست بعد از هر تسک، صف CPU/کانال را مشاهده نمی‌کند. بنابراین GPU در حلقهٔ تصمیم، تقویم را جلو نمی‌برد. تقویم فقط در موتور CPU بعد از اتمام پلن زنده می‌شود.

`env.step` در `offloading_env.py` دقیقاً همین است: `done = True` بعد از یک فراخوانی. یک «قدم محیط» = یک اپیزود کامل برای یک batch گراف.

---

## 4. موتور زمان‌بند چه می‌کند

ورودی: DAG کانونیکال + `decoder_order` + اکشن `0/1/2` برای هر تسک.
خروجی: `ScheduleResult` شامل `start`/`finish` هر تسک، بازه‌های منابع، hopهای انتقال، انرژی تجزیه، `makespan_seconds`.

الگوریتم در `scheduler/engine.py` و `scheduler/calendar.py`:

1. ترتیب توپولوژیک با heap (`heapq`) با گرهش decoder.
2. شش تقویم خالی. هر تقویم لیست بازه‌های `[start, end)` است.
3. برای هر تسک به ترتیب:
   - ورودی ریشه اگر لازم است از UE به محل اجرا منتقل شود (`route` + `reserve` روی کانال).
   - برای هر یال پیش‌نیاز: صبر تا `finish[pred]`، بعد hopهای ماتریس ارتباط، `reserve` روی `MEC_UL` / `MEC_DL` / `V2V_CHANNEL`.
   - `ready = max` روی همهٔ مسیرهای ورودی.
   - `reserve` روی CPU محل اجرا از لحظهٔ `ready`؛ **جستجوی شکاف**: اولین جایی که بازه با قبلی‌ها هم‌پوشانی ندارد.
   - انرژی hop و compute طبق ضریب‌های منجمد جمع شود.
4. sinkها باید خروجی را به UE برگردانند؛ makespan = آخرین بازگشت به UE.
5. ادعای عدم هم‌پوشانی روی هر منبع (`_assert_no_overlap`).

این حلقه به **state مرتب‌شونده** وابسته است. زمان شروع تسک `t+1` تابع finish و تقویم پرشدهٔ تسک‌های قبلی است. نمی‌توان ۲۰ تسک را به‌صورت SIMD مستقل روی GPU زد مگر مدل زمان‌بندی عوض شود.

حجم کار عددی کوچک است: ۲۰ گره، چند یال، شش لیست بازه. غالب زمان interpreter پایتون است (شاخه، heap، sort بازه‌ها، ساخت dataclass)، نه FLOP.

---

## 5. چرا telescoping کار را ۲۰ برابر می‌کند

پاداش آموزش در `OBJECTIVE_AND_ENERGY.md` §6:

برای پلن نهایی `a_1..a_N` و سیاست تکمیل `all_UE`:

- `P_0` = همه Local (مراجع `L_ue` / `E_ue`)
- `P_t` = پیشوند `a_1..a_t` به‌علاوهٔ بقیه Local
- هر `P_t` یک `schedule()` کامل → `(L_t, E_t)`
- `r_t = -(0.5 * (L_t-L_{t-1}) / L_scale + 0.5 * (E_t-E_{t-1}) / E_scale)`

خواستهٔ جبری: `sum_t r_t` با متریک نهایی نسبت به all-UE یکی است، ولی هر توکن گرادیان محلی دارد.

پیاده‌سازی در `telescoping_token_rewards`: حلقهٔ `for t in range(1, n+1)` و هر بار `schedule_via_adapter` از صفر. تقویم `P_t` را از `P_{t-1}` به‌صورت افزایشی جلو نمی‌برد (پلن موقت فرق دارد؛ تسک‌های پرشده Local ممکن است مسیر را عوض کنند).

پس **هر trajectory** ≈ ۲۰ بار زمان‌بندی کامل DAG بیست‌تسکی (به‌علاوهٔ مراجع all-UE/MEC/HELPER برای مقیاس، که جدا محاسبه می‌شوند).

یک `obtain_samples`:

- ۱۰ توزیع × ۲۰ گراف = ۲۰۰ trajectory
- ۲۰۰ × ۲۰ = **۴۰۰۰** فراخوانی `schedule_via_adapter`
- دو بار در هر outer iter (PPO + eval) → **۸۰۰۰** زمان‌بندی

با `EnvExecTime ≈ 283s` یعنی حدود **۷۰ ms** به ازای هر schedule. برای رویدادگسستهٔ پایتونی روی ۲۰ گره عدد معقول است. GPU این ۷۰ ms را به ۷ µs تبدیل نمی‌کند، چون کار GEMM نیست.

---

## 6. چرا این مرحله روی GPU اجرا نمی‌شود

### 6.1 جنس کار با معماری GPU نمی‌خواند

GPU برای کار **یکنواخت، داده-موازی، بدون شاخهٔ سنگین، با شدت حسابی بالا** ساخته شده: ضرب ماتریس، convolution، LSTM روی batch بزرگ.

زمان‌بند این است:

- کنترل‌فلو داده-وابسته (اکشن، وجود یال، خالی/پر بودن تقویم)
- جهش روی heap و لیست بازه
- mutation ترتیبی state
- N=۲۰ — حتی اگر هستهٔ CUDA نوشته شود، هزینهٔ launch از خود کار بیشتر است
- warp divergence: نخ‌های یک warp مسیر MEC می‌روند، بقیه Local/V2V

نتیجه: بازنویسی CUDA برای این اندازه معمولاً **کندتر** از پایتون روی یک هستهٔ خوب است، مگر هزاران DAG مستقل را در یک kernel دسته‌ای با تقویم باعرض ثابت شبیه‌سازی کنی — که دیگر همان `schedule()` منجمد نیست.

### 6.2 گراف TensorFlow فقط سیاست را دارد

نشست TF 1.15 شامل encoder / decoder / PPO / outer Adam است.
`schedule()` در گراف نیست. از `env.step` با اشیای پایتون (`CanonicalDAG`, `ResourceCalendar`) صدا می‌شود.

«بیاور روی GPU» یعنی یکی از این‌ها، نه سوییچ CUDA:

- کل موتور را به `tf.while_loop` / XLA ببری با تقویم tensorای — معنا عوض یا بایت-دقیق سخت
- یا CUDA جدا که TF از آن بی‌خبر است

هیچ‌کدام با تصویر فعلی `margo-phase4-tf115-nv2212` یکی نیست.

### 6.3 قرارداد علمی اجازهٔ تقریب GPU-friendly نمی‌دهد

Phase 1 موتور را **منبع حقیقت** کرده. پاداش، `J_report`، greedy، val/test همه باید همان `schedule()` را ببینند.

اگر برای سرعت یک surrogate عصبی یا فرمول بستهٔ بدون تقویم بگذاری:

- هم‌پوشانی منبع و نیم‌دوبلکس V2V غلط می‌شود
- مسیر دوهاپ MEC→HELPER غلط می‌شود
- telescoping دیگر همان `r_t` منجمد نیست

آن وقت اعداد Phase 4 با spec v0.1 قابل استناد نیستند. نیاز به spec version جدید است.

### 6.4 حتی موازی‌سازی روی GPU هم bottleneck را عوض نمی‌کند مگر مدل عوض شود

ده توزیع در یک iter از نظر DAG مستقل‌اند. این موازی‌سازی **CPU multiprocess** است (`MetaParallelEnvExecutor` در کد هست، `parallel=True` خاموش است)، نه GPU.

GPU در این طرح حداکثر batch کردن `get_actions` را دارد که **الان هم** یک forward روی ده مشاهده است (~۲۸s شامل overhead پایتون/TF، نه کمبود TFLOPS).

A100 / H100 همان ۸۰۰۰ `schedule()` پایتونی را اجرا نمی‌کنند. VRAM اضافه هم بی‌مصرف است (نشست ~۶۵۰ MiB).

### 6.5 وابستگی ترتیبی telescoping داخل یک پلن

`P_1` تا `P_20` از نظر پیشوند وابسته‌اند. می‌توان ۲۰ schedule را از نظر داده موازی کرد (هر `P_t` مستقل از اجرای `P_{t-1}` است، فقط پلن فرق دارد) — باز روی **CPU چند هسته‌ای** یا vectorized C، نه به‌خاطر Tensor Core.

الان حتی همان موازی‌سازی prefix هم نیست؛ حلقهٔ سریال پایتون است.

---

## 7. چه چیزی *واقعاً* روی GPU است

- forward سیاست: embedding، masked mean، LSTM، Luong
- inner PPO: clip surrogate، value clip، Adam روی ۱۰ کپی
- outer: ساخت mean-PG و یک `apply_gradients`

این‌ها روی RTX 4090 حدود یک دقیقه از ده دقیقه‌اند. تعویض به A100 حداکثر همان دقیقه را کمی کوتاه می‌کند.

---

## 8. اگر دیوار باید بیاید پایین — بدون خلط با GPU

مسیرهایی که **معنای v0.1 را نگه می‌دارند** (هنوز تصمیم پروتکل‌اند، پیاده نشده):

1. `parallel=True` برای ۱۰ dist در `obtain_samples` — سقف نظری حدود ۵× روی تکهٔ env اگر ده فرایند واقعاً موازی شوند و TF fork-safe باشد. ریسک مهندسی دارد.
2. پیاده‌سازی C/Numba **همان** `schedule()` با تست برابر بودن بایت‌به‌بایت با `engine.py`.
3. موازی کردن ۲۰ فراخوانی telescoping برای یک پلن (هر `P_t` جدا) روی چند هسته.

مسیرهایی که **spec را می‌شکنند** مگر ADR جدید:

- حذف telescoping و پاداش فقط ترمینال
- زمان‌بند تقریبی / شبکه به‌جای تقویم
- خاموش کردن انرژی یا V2V برای سرعت
- کم کردن `N` یا تعداد گراف support

GPU جدید در هیچ کدام از ردیف‌های نگه‌دارندهٔ معنا، عامل اصلی نیست.

---

## 9. جمع‌بندی یک خطی

MARGO سیاست Graph2Seq را با meta-PPO روی GPU یاد می‌دهد تا پلن Local/MEC/V2V برای DAG بیست‌تسکی بسازد.
جهان شبیه‌سازی‌شده یک زمان‌بند رویدادگسستهٔ تک‌ظرفیت با انرژی و V2V است که پاداش را با بیست بار زمان‌بندی مجدد همان پلن (telescoping) می‌سازد.
آن جهان پایتون ترتیبی است، در گراف TF نیست، و با معماری GPU نمی‌خواند.
کندی مشاهده‌شده ضعف 4090 نیست؛ هزینهٔ قرارداد علمی Phase 1 است.

---

## 10. ارجاع

- `spec/SYSTEM_SPEC.md`
- `spec/SCHEDULING_SEMANTICS.md`
- `spec/OBJECTIVE_AND_ENERGY.md` §6
- `spec/LEARNING_PROTOCOL.md`
- `env/mec_offloaing_envs/scheduler/engine.py`
- `env/mec_offloaing_envs/scheduler/calendar.py`
- `env/mec_offloaing_envs/scheduler/reward.py`
- `env/mec_offloaing_envs/offloading_env.py` (`step`, `get_reward_batch_step_by_step`)
- `samplers/seq2seq_meta_sampler.py` (`parallel=False`)
- `samplers/vectorized_env_executor.py` (`MetaIterativeEnvExecutor`)
- `meta_trainer.py` (دو `obtain_samples` در هر iter)
