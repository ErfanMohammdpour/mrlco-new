# MARGO — گزارش مقایسهٔ نسخهٔ فعلی با نسخهٔ روی گیت

**تاریخ گزارش:** ۲۰۲۶-۰۹-۲۱
**مخزن گیت (مرجع):** `https://github.com/MA-RGO/MARGO` — شاخهٔ `main` — کامیت `53fe08d` («Add result figures for README»)
**نسخهٔ فعلی:** `MARGO_BASELINE/mrlco-new` — شاخهٔ `phase4-eval` — کامیت `e6cf007` («feat(v0.3): Phase 1–5 results, spec ledger, and Persian progress report»)
**وضعیت درخت کار:** پاک (`git status` تمیز، هیچ تغییر commit‌نشده‌ای وجود ندارد)
**فاصله:** `0` کامیت عقب، `25` کامیت جلوتر از `origin/main`

---

## ۰. حکم کوتاه (Executive verdict)

1. **خانوادهٔ معماری شبکه حفظ شده است.** همان Graph2Seq (MeanAggregator روی predecessor/successor + خروجی ۲۵۶)، همان LSTM دو‌لایهٔ ۱۲۸ + Luong attention، همان فضای اکشن سه‌تایی `{0=Local, 1=MEC, 2=V2V}`، و همان اسکلت PPO-inner + outer first-order mean-PG (که در گیت «Reptile» نامیده می‌شد) در کد موجود است — اما چند جزئیات داخلی انکودر و کل پروتکل بهینه‌سازی/ارزیابی عوض شده (بخش ۴ و ۶).
2. **هر ۸ باگ مستندشدهٔ نسخهٔ گیت رفع شده است** (`docs/LOGIC_AND_LEARNING_BUGS.md` + `docs/CODE_AUDIT.md`) به‌علاوه ۶ ایراد دیگر که در مسیر پیدا شد (لنگر value-clip، `range(5)`، میانگین آلوده به padding، fork/CUDA، importهای شکننده، reset نشدن loss).
3. **اما نسخهٔ فعلی «گیت + فقط رفع باگ» نیست.** پنج چیز روی مسیر بحرانی عوض شده: (الف) قالب observation (`20` فیچر تخت → `50/54` با جدول همسایهٔ decoder-index)، (ب) اسپلیت داده (`19×100` → `25×100` با holdout ساختاری fat×density)، (ج) موتور زمان‌بندی + reward کانونیک جدید (`env/.../scheduler/`، ۲٫۱k خط غیرتست) که `greedy` را هم عوض کرده، (د) پروتکل بهینه‌سازی/ارزیابی (inner ۳ گام Adam روی ۲۰ گراف support، outer یک گام mean-PG، val اجباری، support/query تفکیک‌شده)، و (ه) روشِ نتایج (BC + best-of-k به‌جای meta-RL).
4. **حلقهٔ اصلی گیت (Reptile/PPO با ۳۵۰۰ تکرار) هنوز اجرا نشده است.** تنها ران قدیمی (در `results/ckpt_ours_final_3500/MARGO`) روی نسخهٔ باگ‌دار است و به گِردی (plateau) `~863` ثانیه رسیده در برابر `greedy ≈ 829` — یعنی سیاست در میانگین **بدتر** از greedy، و فقط در ۷۲۴ از ۳۵۰۰ تکرار بهتر. تمام ۷ کانفیگ تشخیصی PPO-on-θ در نسخهٔ فعلی هم به فروپاشی occupancy روی MEC رسیدند (ADR-007).
5. **هیچ عددی `paper_result` نیست.** meta-test باز نشده، فاز ۵ پارک شده، فاز X نزده، ADR-013 وجود ندارد، تگ `v0.3-freeze` زده نشده.
6. **پاسخ مستقیم به سؤال «آیا نسخه‌ای داریم که معماری‌اش همان گیت باشد ولی مشکلات رفع شده باشد؟»** → **نیمه‌بله**: «خانوادهٔ مدل و رابط‌ها» بله (با رفع باگ‌ها)، «ورودی/داده/شبیه‌ساز/پروتکل/روش» نه. جزئیات و سه گزینهٔ ادامه در بخش ۶ و ۷.

---

## ۱. مرجع مقایسه (چه چیزی با چه چیزی)

| مورد | نسخهٔ گیت | نسخهٔ فعلی |
|---|---|---|
| مسیر | `https://github.com/MA-RGO/MARGO` (remote `origin`) | `MARGO_BASELINE/mrlco-new` |
| شاخه | `main` | `phase4-eval` |
| کامیت | `53fe08d` | `e6cf007` |
| تعداد کامیت | — | `+25` |
| درخت کار | — | تمیز (هیچ تغییر commit‌نشده) |

نکته: شاخه‌های `phase0-learning-oracles` … `phase4-eval` روی remote دوم (`erfan` → `ErfanMohammdpour/mrlco-new`) هم push شده‌اند؛ روی `origin` (MA-RGO/MARGO) فقط `main` وجود دارد. یعنی **نسخهٔ فعلی هنوز به مخزن مرجع push نشده است.**

---

## ۲. حجم تغییرات

| نوع | تعداد فایل | خطوط |
|---|---|---|
| Added (جدید) | ۳۶۶ | ~۱۱۳٫۸k درج |
| Modified (ویرایش‌شده) | ۱۸ | ۲٫۰۱۵ درج / ۱٫۶۴۷ حذف |
| Deleted (حذف‌شده) | **۰** | — |

هیچ فایلی از درخت گیت حذف نشده است. کد پایتون:

| بخش | گیت (LOC) | فعلی (LOC) | تغییر |
|---|---|---|---|
| `env/` (شامل `scheduler/` جدید) | ۱٫۴۳۹ | ۷٫۵۶۳ | +۶٫۱۲۴ |
| `spec/` (کل پوشه جدید) | ۰ | ۱۷٫۲۶۳ | +۱۷٫۲۶۳ |
| `policies/` | ۱٫۷۳۱ | ۲٫۱۶۱ | +۴۳۰ |
| `meta_algos/` | ۳۳۰ | ۶۶۴ | +۳۳۴ |
| `samplers/` | ۹۹۱ | ۱٫۰۶۶ | +۷۵ |
| `utils/` | ۹۱۸ | ۹۲۱ | +۳ |
| `baselines/` | ۲۹۱ | ۲۹۱ | ۰ |
| فایل‌های ریشه (`meta_trainer.py`, `meta_evaluator.py`, …) | ۲٫۳۰۹ | ۲٫۳۸۳ | +۷۴ |
| **جمع** | **۸٫۰۰۹** | **~۳۲٫۵۷۶** | **~۴ برابر** |

به‌علاوه: ۲۴ فایل تست، **۲۲۰ تابع تست**، ۱۲ سند ADR، ۵ سند gate اجرایی (`phase0`…`phase4`)، ۲۴ سند markdown در `spec/`، ۱۵ fixture oracle، ۱۲۴ فایل لاگ/JSON آرشیو Kish، ۱۷ شکل، و یک گزارش فارسی mentor-facing (`reports/HWs-Template/main.tex`).

تفکیک ۳۶۶ فایل جدید: ۲۵۴ در `spec/`، ۳۷ در `env/` (پکیج `scheduler/` + تست‌ها)، ۲۸ در `reports/` (قالب LaTeX گزارش فارسی)، ۱۷ شکل، ۱۳ سند در `docs/`، ۸ در `paper/` (کلاس/بیب IEEE + جدول‌های تحلیل)، به‌علاوه `meta_algos/held_out_eval.py`، `meta_algos/variable_io.py`، `samplers/dummy_step_env.py`، `scripts/export_margo_figures.py`. بر حسب نوع: ۸۸ `.py`، ۸۲ `.json`، ۶۰ `.md`، ۲۸ `.log`، ۱۸ `.yaml`، ۱۷ `.pdf`، ۱۷ `.png`، ۱۶ `.txt`، ۱۵ `.csv`.

---

## ۳. باگ‌های نسخهٔ گیت و وضعیت رفع آن‌ها

مرجع: `docs/LOGIC_AND_LEARNING_BUGS.md` (۸ باگ) و `docs/CODE_AUDIT.md` (لیست ۸ موردی مکمل). در جریان بازنویسی، ۶ ایراد دیگر هم پیدا و رفع شد (ردیف‌های +۹ تا +۱۴).

| # | باگ در نسخهٔ گیت | اثر | وضعیت در نسخهٔ فعلی | شاهد |
|---|---|---|---|---|
| ۱ | **MEC منتظر اتمام predecessorِ V2V نمی‌ماند.** `ws_start_time = max(FT_locally[j], FT_ws[j])` بدون `FT_v2v_dl[j]`؛ همین حفره در `greedy_solution` | makespan و reward خوش‌بینانه برای الگوی V2V→MEC؛ سیاست یاد می‌گیرد این ترکیب «ارزان» است | **رفع** | موتور کانونیک `env/.../scheduler/engine.py` + جدول مسیر ۳×۳ (`routes.py`) + ADR-005 (canonicalization یال‌ها) |
| ۲ | **`end_token=2` خودش اکشن V2V است.** (میراث MRLCO با `vocab_size=2`) | `GreedyEmbeddingHelper` در اولین V2V دنباله را قطع می‌کند → پلن ناقص؛ تلهٔ ادعای greedy | **رفع** | `policies/meta_seq2seq_policy.py:623-624` → `end_token=int(vocab_size)` = ۳ |
| ۳ | **meta-gradient حدود ۱۰۰× کوچک‌تر از فرمول مبنا.** `inner_batch_size=10` در برابر `1000` → `update_numbers≈100` و تقسیم بر آن؛ به‌علاوه inner Adam است ولی بر `inner_lr` تقسیم می‌شود | گام outer بی‌اثر/پرانرژی‌نویز؛ adaptation بین taskها ضعیف | **رفع** | `meta_algos/MRLCO.py:232-243` — `mean_pseudogradient(...)` سپس یک گام outer روی میانگین |
| ۴ | **Graph2Seq یال DAG را نمی‌بیند؛ adjacency یک clique کامل است.** `tf.tile(seq_indices, [seq_len,1])` → هر نود به همه وصل، از جمله خودش | ادعای message-passing روی DAG پشتیبانی نمی‌شد؛ encoder عملاً مخلوط‌کنندهٔ سراسری بود | **رفع** | `policies/graph2seq_encoder.py:83-109, 145-146, 203-204` — unpack جدول‌های fw (successor) و bw (predecessor) از obs؛ نودهای `PAD` dummy |
| ۵ | **فیچر pred/succ با `task_id` خام است نه موقعیت HEFT**؛ pred فقط از `range(0,i)`؛ سقف ۶ همسایه با حذف بقیه | سیگنال ساختار گراف در ورودی گمراه‌کننده؛ ناسازگاری observation با `pre_task_sets` شبیه‌ساز | **رفع** | `env/.../scheduler/encoder_obs.py` — همسایه‌ها با **decoder-index**، `MAX_NEIGH=19`، درجهٔ >۱۹ خطا؛ منبع یال‌ها `edge_set` نه ماتریس (ADR-005) |
| ۶ | **کران انرژی غلط + حساب ناسازگار.** `min_energy` = تمام-MEC، در حالی که V2V توان ارسال کمتری دارد؛ محاسبهٔ helper جزو انرژی UE ولی محاسبهٔ MEC صفر | پاداش انرژی می‌تواند مثبت شود؛ سوگیری سیستماتیک به نفع MEC داخل گرادیان | **رفع** | ADR-001: `E = total_mobile_joules = UE + HELPER`؛ `_compute_energy_bounds` (۳۰ خط) حذف و به `scheduler/energy_api.compute_reference_ranges` با سه پلن خالص `all_UE / all_MEC / all_HELPER` منتقل شد؛ تست‌های `test_phase5_energy.py` |
| ۷ | **مقیاس latency غلط.** `max_time`/`min_time` هزینهٔ «یک تسک تنها» است ولی روی «دلتای makespan» اعمال می‌شد | دو مؤلفهٔ reward در یک بازه نبودند؛ advantage مخدوش | **رفع** | کامیت `f2453b7`: `score_func` ۶۰ خطی per-step حذف و `scheduler/reward.telescoping_token_rewards` جایش آمد: `r_t = -(w_L·ΔL/L_scale + w_E·ΔE/E_scale)`؛ امتیاز سطح-پلن |
| ۸ | **«Reptile» در واقع حلقهٔ Adam متوالی روی ۱۰ task است** (`θ` وسط حلقه عوض می‌شود) | جهت meta، میانگین جابه‌جایی نیست؛ meta-update از جنس دیگری | **رفع** | `meta_algos/MRLCO.py:232-243` — خواندن همهٔ `θ'_i`، `mean_pseudogradient(theta0, adapted, inner_lr, k_steps)`، سپس **یک** `apply_gradients` روی core و `sync_task_policies_from_core()` |
| +۹ | **لنگر value-clip در PPO غلط بود:** `vpredclipped = self.vpred[i] + clip(vpred - self.old_v[i], ...)` — یعنی clip حول ارزش **جدید** نه **قدیم** | surrogate ارزش بی‌معنا؛ نمونهٔ کلاسیک باگ PPO | **رفع** | `meta_algos/MRLCO.py:148` و `meta_algos/ppo_offloading.py:108` — `self.old_v[i] + clip(...)` |
| +۱۰ | **`for i in range(5)`** در `meta_trainer.py` — فقط ۵ از ۱۰ سیاست متا-batch جمع/لاگ می‌شد | بخشی از meta-batch در گزارش و میانگین نادیده | **رفع** | `meta_trainer.py:251/257/265` — `range(len(new_samples_data))` |
| +۱۱ | **میانگین همسایه‌ها اسلات‌های padding را هم می‌شمرد** (`tf.reduce_mean(neigh_vecs)`) | نودهای با درجهٔ کم، بردار مخلوط با صفر می‌گرفتند | **رفع** | `policies/graph2seq_modules/aggregators.py` — `tf.sequence_mask(neigh_len)` و تقسیم بر تعداد معتبر |
| +۱۲ | **`multiprocessing` با fork بعد از TF/CUDA** (کارگرها CUDA را می‌دیدند)؛ `mpi4py` اجباری؛ `tf.get_logger` در TF2 می‌شکست | crash/محیط آلوده در اجرای موازی | **رفع** | `samplers/vectorized_env_executor.py` — `get_context("spawn")` + `CUDA_VISIBLE_DEVICES=""`؛ `utils/logger.py` و `policies/model_helper.py` — importهای اختیاری |
| +۱۳ | `vf_loss`/`pg_loss` بدون reset بین گام‌های inner (فقط لاگ eval را خراب می‌کرد) | عدد eval دروغ | **رفع** | `meta_algos/ppo_offloading.py:189-193` — لیست per-call |
| +۱۴ | `dropout=0.1` به `MeanAggregator` پاس نمی‌شد | regularization خاموش | **رفع (شفاف‌سازی)** | `policies/graph2seq_encoder.py:78, 119, 369-370` — dropout صریحاً ۰ و مستند شده |

یعنی از منظر «رفع مشکلات موجود»، کاری که در نسخهٔ فعلی انجام شده **کامل و مستند** است: هر باگ با مکان دقیق، مکانیزم اثر، و ترتیب اصلاح ثبت شده.

---

## ۴. چیزهایی که «فقط رفع باگ» نبودند — تغییر ساختار

این‌ها همان چیزهایی هستند که با خواستهٔ «همان ساختار گیت، کمترین تغییر» نمی‌خوانند:

| تغییر | گیت | فعلی | چرا ساختاری است |
|---|---|---|---|
| **قالب observation** | ۲۰ فیچر تخت (`obs_dim=20`: id، زمان‌های local/MEC/V2V، ۶ pred id، ۶ succ id) | `obs ∈ R^{20×50}` (v1: ۱۱ فیچر + ۱۹ successor-index + ۱۹ predecessor-index + mask) و `R^{20×54}` (v2: ۱۵ فیچر برای محور منابع) | ورودی مدل عوض شده؛ شکست/تغییر ckpt؛ در ران‌ها `readout=mean` انتخاب شد هرچند پیش‌فرض کد `triple` مانده |
| **اسپلیت داده** | ۱۹ توزیع × ۱۰۰ گراف، meta-train ۱۵ / meta-test ۴ (توزیع `12` برای eval) | ۲۵ توزیع × ۱۰۰ گراف (۲۵۰۰ گراف)، `latin_grid_holdout_v1` روی محور fat×density: train ۱۵ / val ۵ / meta-test ۵ + support/query تفکیک‌شدهٔ ۲۰/۸۰ با SHA-256 | مقایسهٔ مستقیم با اعداد مقاله/گیت ممکن نیست |
| **موتور زمان‌بندی** | منطق شبیه‌ساز داخل `offloading_env.py` (۱۰۴۸ خط) | `offloading_env.py` به ۶۷۶ خط کاهش یافته و منطق به پکیج جدید `env/mec_offloaing_envs/scheduler/` (۱۱ ماژول + ۲۴ تست) منتقل شده | بازنویسی مسیر بحرانی فیزیک، نه پچ |
| **baseline حریص** | `greedy_solution()` داخل env | `scheduler.greedy.greedy_plan` (الگوریتم متفاوت) | عدد «greedy» در لاگ‌ها با گیت یکی نیست؛ مقایسهٔ مستقیم با اعداد قبلی معتبر نیست |
| **قرارداد نمونه‌گیری تسک** | `sample_tasks()` عدد برمی‌گرداند | دیکشنری `{"dist_index","graph_indices"}` + `_slice_current()` + support ۲۰ گراف | تغییر قرارداد env (SPI) |
| **کوپلینگ entry قدیمی** | `meta_trainer.py` خودکفا | `meta_trainer.py` اکنون `from spec.eval_protocol import ...` و `from spec.train_audit import ...` دارد | مسیر متای گیت به فریم‌ورک جدید وابسته شده |
| **جایگاه روش** | روش = Graph2Seq + PPO-inner + Reptile-outer | ADR-007: PPO-on-θ، Reptile-on-θ و CAVIA-on-z **از روش حذف** شدند و فقط baseline/ablation منفی‌اند؛ روش اعلام‌شده = **BC از معلم 2-opt + نمونه‌گیری best-of-k** | کد متا مانده، ولی چیزی که نتایج را ساخته مسیر دیگری است |
| **پوشهٔ `spec/`** | وجود ندارد | ۱۷k خط: ۱۲ ADR، ۱۵ gate، ۱۵ oracle اسباب‌بازی، manifest/validator، اسپلیت، اسکیمای دیتاست | لایهٔ جدید مستندسازی/اعتبارسنجی |

نکتهٔ مهم: `paper/MARGO_paper_final.tex` تنها **۹۳ درج/۹۳ حذف** دارد و با `--ignore-all-space` **صفر** تفاوت — یعنی متن مقاله دست‌نخورده و تغییر فقط whitespace است.

---

## ۵. نتایج

### ۵٫۱ نتیجهٔ نسخهٔ گیت (ران قدیمی، فقط برای مقایسهٔ پایه)

فایل: `results/ckpt_ours_final_3500/MARGO/meta_offloading20_log-inner_step1/progress.csv` — **توجه:** این مسیر بیرون از مخزن گیت است (در ریشهٔ workspace، پوشهٔ `results/` که در `.gitignore` است)؛ روی شاخهٔ `phase4-eval` نسخهٔ داخل مخزن فقط ۲ سطر دارد. اعداد زیر از خود فایل روی دیسک محاسبه شده‌اند (۳۵۰۰ تکرار، ۱۰٫۰۰۰ trajectory در هر تکرار):

| سنجه | مقدار |
|---|---|
| میانگین latency، ۱۰ تکرار اول | ۱٫۴۱۱٫۸ |
| میانگین latency، تکرار ۴۰۰–۶۰۰ | ۸۷۰٫۸ |
| میانگین latency، ۱۰۰۰ تکرار آخر | **۸۶۳٫۳** (گِرد شده؛ از تکرار ~۵۰۰ به بعد تخت) |
| بهترین تکرار | **۷۱۰٫۹** در تکرار ۱۳۵۹ (قلهٔ نمونه، پایدار نیست) |
| میانگین greedy مرجع (همان شبیه‌ساز) | **۸۲۹٫۰** |
| تعداد تکرارهایی که سیاست بهتر از greedy بود | **۷۲۴ / ۳۵۰۰** |

**نتیجه:** با بودجهٔ بلند (۳۵۰۰ تکرار) سیاست روی این شبیه‌ساز به‌طور میانگین **بدتر از greedy** است؛ و این عدد روی شبیه‌سازی است که خودش باگ‌های ۱/۶/۷ را داشت (makespan و reward خوش‌بینانه). پس این نتیجه هم علمی و هم فیزیکی قابل اتکا نیست.

### ۵٫۲ وضعیت فازها در نسخهٔ فعلی

| فاز | هدف | گیت | نتیجه |
|---|---|---|---|
| ۰ | prune + freeze روش | ثبت ADR-001…005، تگ `phase0-freeze-v0.1` | **PASS / CLOSED** |
| ۱ (شبیه‌ساز) | فیزیک کانونیک | تگ `phase1-freeze-v0.1` | **PASS / CLOSED** |
| ۱ (encoder) | آیا شکاف OOD نمایشی است؟ | `ΔT < ۱۰` برای gatv2 و dagformer | **هر دو منفی** (`−۱۱٫۷۶` و `−۸٫۵۵`) → شکاف نمایشی نیست، backbone = `meanagg` (ADR-010) |
| ۲ | best-of-k | `T_best_32 ≤ 510` | **PASS: ۵۰۹٫۱۲±۰٫۴۸** (۳ بذر) |
| ۳ | EAS-on-φ | query `≤ ۵۲۰` | **FAIL: ۵۷۷٫۷ → ۵۷۶٫۴** (ADR-011) |
| ۳B | EAS per-instance | `Δ ≥ ۵s` vs best-of-k | **FAIL: −۰٫۲s** |
| ۴ | محور منابع (obs v2) | BC/BOK: PASS یک‌طرفه؛ EAS/CTX: رد | **seed0 تمام**: BOK متناقض‌نما PASS، EAS `Δ−۰٫۴s` FAIL، CTX `Δ=۰٫۰` FAIL |
| ۵ | Pareto انرژی | فیزیک لوکال PASS، ران کامل | **پارک‌شده** (۱ از ۶۵ کانفیگ تمام شد) |
| X | کنترل PPO-500 از صفر | پیش‌بینی P1–P3 | **نزده** |
| ۶ | eval/paper (۵ بذر + meta-test) | منتظر ADR-013 + تگ `v0.3-freeze` | **بسته** |

### ۵٫۳ نردبان T روی validation (ثانیه، کمتر بهتر؛ obs v1 مگر ذکر شود)

| روش | T |
|---|---|
| 2-opt (معلم/سقف) | **۴۲۳٫۶** (train ۴۱۴٫۱) |
| Hamming-2 | ۴۲۷٫۳ |
| pair motif oracle | ۴۳۶٫۶ |
| pair search k50 (heuristic، روش نیست) | ۴۳۸٫۹ |
| **BOK k=64 (Phase 4، obs v2، seed0)** | **۴۴۴٫۵** |
| BOK k=32 (Phase 4، obs v2) | ۴۴۷٫۷ |
| greedy_from_mec | ۴۶۳٫۵ |
| BC greedy (Phase 4، obs v2) | ۴۹۳٫۸ |
| BOK k=64 (Phase 2) | ۵۰۱٫۵±۰٫۳۷ |
| BOK k=32 (Phase 2) | ۵۰۹٫۱±۰٫۴۸ |
| BC (Control A، obs v1) | ۵۷۵٫۱ |
| scheduled-sampling BC | ۵۷۴٫۱ |
| BC greedy (Phase 1، mean readout) | ۵۷۷٫۰ |
| BC-2opt greedy | ۵۸۱٫۸ |
| all-MEC | ۶۳۴٫۲ |

### ۵٫۴ best-of-k (آموزش نیست، فقط نمونه‌گیری + یک `schedule()`)

Phase 2 (n=500، ۳ بذر): k=1 → **۵۸۸٫۱۴**، k=8 → ۵۳۲٫۲۳، k=32 → ۵۰۹٫۱۲، k=64 → **۵۰۱٫۵۴**؛ هزینه ۰٫۱۷۵ ثانیه/گراف در k=64. oracle-within-samples فقط ۲٫۲٪ — یعنی نمونه‌گیری تقریباً هیچ‌وقت به 2-opt نمی‌رسد.

Phase 4 (obs v2، seed0): k=1 → ۴۹۳٫۸۲، k=8 → ۴۵۹٫۳۹، k=32 → ۴۴۷٫۶۷، k=64 → **۴۴۴٫۴۸**. نمونهٔ held-out `p_9`: ۴۱۸٫۴ → **۳۷۷٫۹** (معلم ۳۶۳٫۶).

### ۵٫۵ نتایج منفی (ارزش علمی اصلی تغییرات)

- **PPO-on-θ در ۷ کانفیگ occupancy را نابود می‌کند:** `par500` در تکرار ۲۱ به MEC≥۰٫۹۵ و در ۱۸۰ به ۰٫۹۹۸؛ `bc50` در تکرار ۹ از حوضچهٔ expert بیرون؛ `kl_bc_ppo` با β=۰٫۱ نتوانست نگه دارد (`kl_bc` ۱٫۱۸→۱۶)؛ `bc_fewshot` val را ۵۸۰→۶۵۷ برد؛ `binary_lat` (بدون V2V و انرژی) هم MEC ۴۵٪→۹۵٪.
- **CAVIA-on-z سه بار تخت:** ۵۸۴٫۵→۵۸۵٫۱ / ۵۸۴٫۵→۵۸۴٫۲ / ۵۷۲٫۷→۵۷۲٫۷.
- **EAS روی φ سطح-توزیع و سطح-نمونه: FAIL.**
- **CTX/z در محور منابع: `ΔT=۰`** (z استفاده نمی‌شود).
- **`oracle_dist_id` و `rewrite_mec` هم بی‌اثر/مضر** (`rewrite` از all-MEC: ۶۱۰٫۶ در برابر ۵۷۵).
- **triple readout رد شد:** mean ۵۷۷٫۰۲ در برابر triple ۵۸۸٫۱۴ (ADR-010).

### ۵٫۶ انرژی (Phase 5، پارک‌شده)

فیزیک لوکال PASS و smoke روی ۸ گراف: λ=1.00 → T=۳۳۳/E=۳۱۴؛ λ=0.75 → ۳۴۱/۲۴۵؛ λ=0.50 → ۴۳۰/۷۴؛ λ=0.25 و ۰٫۰۰ → ۴۶۸/۳۹. ران کامل متوقف شد (۱ از ۶۵) → **ادعای Pareto ممنوع**.

### ۵٫۷ آنچه مستندات صریحاً می‌گویند «ادعا نکن»

بهتر بودن از MRLCO/DAMRL · Graph2Seq با DAG-edge به‌عنوان نوآوری · PPO-inner + Reptile به‌عنوان روش · درصد نسبت به publication greedy · few-shot / EAS / CAVIA / PEARL-context · جبههٔ انرژی · عدد meta-test · `paper_result=true`.

### ۵٫۸ اعداد «مقالهٔ قدیمی» که هنوز در مخزن هستند (پروتکل متفاوت، آشتی‌نشده)

`paper/EVIDENCE_MAP.md` §۳ یک جدول قدیمی دارد (واحد در آن `ms` نوشته شده ولی در `reports/HWs-Template/main.tex` ثانیه است — تناقض واحدها):

| تسک | Greedy | MRLCO | MARGO |
|---|---|---|---|
| T1 latency | ۷۹۵ | ۶۳۸ | **۶۳۴** |
| T1 energy | ۸۷۵ | ۶۲۰ | **۵۹۵** |
| T2 latency | ۸۱۰ | ۶۶۸ | **۶۶۱** |
| T2 energy | ۸۹۰ | ۶۵۹ | **۶۳۳** |

MARGO در برابر MRLCO فقط **−۰٫۶٪ latency / −۴٪ energy** است؛ حکم خود worklog: «برای مقالهٔ Q1 کافی نیست». این اعداد با نردبان validation نسخهٔ فعلی **قابل مقایسه نیستند** (پروتکل، اسپلیت و واحد متفاوت).

`paper/analysis_tables/` (۱۰۰ تکرار، میانگین ۲۰ تکرار آخر): T1 MARGO → latency ۵۷۵٫۴ / energy ۶۵۹٫۰ (بهبود ۱۷٫۷٪ / ۲۸٫۳٪)، T2 MARGO → ۶۰۵٫۲ / ۶۸۲٫۷ (۱۴٫۳٪ / ۲۷٫۰٪)؛ اما بهبود نسبت به **greedy داخلی خودِ همان ران** است، نه مقایسهٔ روش-به-روش (خودِ گزارش هم همین را می‌گوید). توزیع اکشن: T1 → Local ۳۴٫۹٪ / MEC ۵۸٫۴٪ / V2V ۶٫۸٪.

### ۵٫۹ کیفیت شواهد (ایرادهایی که باید بدانی)

- **همهٔ ۷ ران ثبت‌شده `code_dirty=true` هستند**؛ دو ران `code_sha="unknown"`.
- `spec/PHASE4_DIAGNOSTIC_RESULTS.json` **JSON خراب** است (`json.load` در خط ۵۳۴ می‌شکند). فقط `PHASE4_WORKLOG.json` سالم است.
- بزرگ‌ترین ران متای کامل‌شده فقط **۱۸۱ تکرار** از ۵۰۰ است (`par500`، ۹٫۳۶ ساعت) و در **۱۵۹ از ۱۸۱** تکرار فلگ `action_collapse_MEC_*` خورده؛ entropy میانگین eval ۰٫۰۹۶.
- **هیچ ران متای ۳۵۰۰ تکراری ثبت‌شده‌ای در نسخهٔ فعلی وجود ندارد؛ frozen primary (seeds 0–4) شروع نشده.**
- تمام پوشه‌های `runs/` فقط `seed_0` هستند — هیچ `seed_1`/`seed_2` روی دیسک نیست (فقط BOK فاز ۲ و ablation انکودر ۳ بذری‌اند).
- baselineهای `baselines/` (linear/VF/zero) در ارزیابی‌های v0.3 **استفاده نشده‌اند**؛ در نتایج اثری از `linear`/`VF`/`PEARL`/`DAMRL` نیست.
- اسناد v0.2 قدیمی هنوز مانده‌اند: `spec/FINAL_README.md:3`، `FINAL_IMPLEMENTATION_PLAN.md:4`، `FINAL_DATAFLOW.md:1`، `CURRENT_ARCHITECTURE.md:7` هنوز `MARGO-METHOD-v0.2-cavia` را حمل می‌کنند در حالی که ADR-007 آن را مرده اعلام کرده.
- `spec/PHASE4_WORKLOG.md:157` می‌گوید «readout ablation هنوز pending» در حالی که تمام شده — سند کهنه.


---

## ۶. پاسخ صریح به سؤال اصلی

> «آیا الان نسخه‌ای داریم که معماری‌اش همان چیزی باشد که روی گیت است ولی مشکلات رفع شده باشد؟»

**تفکیک لایه‌ای:**

| لایه | همان گیت است؟ | توضیح |
|---|---|---|
| معماری شبکهٔ سیاست (Graph2Seq روی pred/succ، خروجی ۲۵۶؛ LSTM 2×۱۲۸ + Luong؛ vocab=3؛ `start=0`) | **تقریباً، با یک تغییر داخلی** | همان خانواده و همان رابط‌ها؛ اما جهت‌ها اکنون **اجباری دوطرفه** و ترکیب `fw_hidden + bw_hidden` است (در گیت فقط `fw` استفاده می‌شد و concat بود) و dense تعبیهٔ نود از سطح policy به انکودر منتقل شده؛ `end_token` هم ۳ شد |
| triple readout | **در کد بله، در ران‌ها نه** | کد `readout_type` دارد (پیش‌فرض `triple`)؛ نتیجهٔ ADR-010 می‌گوید `mean` بهتر است و ران‌ها با `mean` شده‌اند |
| اسکلت یادگیری متا (PPO-inner + outer first-order mean-PG، `n_itr=3500`, `K=3`, meta-batch ۱۰) | **بله، در کد** | `meta_trainer.build_frozen_primary_stack` باقی است و حتی نام الگوریتم هم اصلاح شده (`mrlco_first_order_mean_pseudogradient`، نه Reptile)؛ اما **هرگز اجرا نشده** (بزرگ‌ترین ران متا: ۱۸۱ تکرار) و ADR-007 آن را از «روش» حذف کرده |
| ورودی مدل (observation) | **نه** | `20` → `50`/`54` با همسایه‌های decoder-index |
| مرز داده/اسپلیت | **نه** | `19×100` → `25×100` + holdout ساختاری + support/query |
| شبیه‌ساز (مسیر بحرانی فیزیک) | **نه** | بازنویسی در `scheduler/` (۶٫۵k خط با تست) |
| روشی که نتایج را ساخته | **نه** | BC از 2-opt + best-of-k |

**نتیجهٔ یک‌خطی:** نسخهٔ فعلی = **«همان شبکه، با فیزیک و باگ‌های رفع‌شده، اما با قالب ورودی/داده/موتور جدید و روش یادگیری متفاوت»**. اگر معیار تو دقیقاً «درخت گیت + حداقل پچ برای رفع باگ» باشد، چنین نسخه‌ای **الان وجود ندارد** — این نسخه چیزی بیشتر از آن است. و اگر معیار «قابل‌دفاع بودن علمی» باشد، نسخهٔ فعلی جلوتر است ولی از روش گیت (meta-RL) فاصله گرفته و **حلقهٔ متای آن هنوز اعتبارسنجی نشده**.

---

## ۷. سه گزینه برای ادامه (با کمترین تغییر)

### گزینه A — «گیت + ۸ پچ حداقلی» (همان چیزی که خواستی)
فورک از `origin/main` و اعمال فقط این پچ‌ها، بدون `spec/`، بدون موتور جدید، بدون تغییر obs/split:

| # | فایل (در نسخهٔ گیت) | پچ |
|---|---|---|
| ۱ | `env/mec_offloaing_envs/offloading_env.py` (`get_scheduling_cost_step_by_step`, `greedy_solution`) | شروع uplink/compute MEC = `max(FT_locally[j], FT_wr[j], FT_v2v_dl[j])` |
| ۲ | `policies/meta_seq2seq_policy.py` (`HParams`) | `end_token=2` → `3` |
| ۳ | `meta_trainer.py` یا `meta_algos/MRLCO.py` | `inner_batch_size 10→1000` **یا** حذف `update_numbers` از مخرج |
| ۴ | `meta_algos/MRLCO.py` (`UpdateMetaPolicy`) | میانگین `(θ − θ'_i)` روی همهٔ taskها، سپس **یک** `apply_gradients` |
| ۵ | `policies/graph2seq_encoder.py` (`sequence_to_graph`) | adjacency = pred ∪ succ بعد از map کردن `task_id → موقعیت HEFT`؛ self-loop جدا |
| ۶ | `env/.../offloading_task_graph.py` (`encode_point_sequence_with_cost`) | pred/succ با موقعیت HEFT نه id؛ pred از `pre_task_sets` نه `range(0,i)`؛ درجهٔ >۶ خطا نه حذف |
| ۷ | `env/.../offloading_env.py` (`_compute_energy_bounds`, `compute_*_energy`) | `min` واقعی روی {all-local, all-MEC, all-V2V} + یک تعریف واحد انرژی (UE+helper) |
| ۸ | `offloading_env.py` (reward) | نرمال‌سازی مستقل latency و energy یا امتیاز سطح-پلن |
| ۹ | `meta_algos/MRLCO.py`, `meta_algos/ppo_offloading.py` | `vpredclipped = old_v + clip(vpred − old_v, ±ε)` (لنگر clip روی ارزش قدیم، نه جدید) |
| ۱۰ | `env/.../offloading_env.py`, `meta_trainer.py` | `range(5)` → `range(len(new_samples_data))`؛ ماسک کردن میانگین همسایه‌ها در `aggregators.py` |

هزینه: کم (۶ فایل). سود: دقیقاً معماری گیت. ریسک: باید همهٔ ران‌ها را از صفر تکرار کنی؛ و شانس موفقیت متا روی این شبیه‌ساز پایین است (شواهد ۷ کانفیگ تشخیصی).

### گزینه B — نسخهٔ فعلی + بازگرداندن مسیر متا (توصیهٔ من برای «کمترین تغییر مؤثر»)
نسخهٔ فعلی را نگه دار، اما:
1. مسیر `build_frozen_primary_stack` را با **obs v1** روی موتور اصلاح‌شده اجرا کن (اجرای همان ۳۵۰۰ تکرار گیت، ولی این بار فیزیک درست).
2. اگر فروپاشی MEC تکرار شد → همان نتیجهٔ منفی C2 را با مکانیزم مستند کن (ارزش انتشار دارد).
3. مسیر `spec/` و BC/best-of-k را به‌عنوان روش جدا نگه دار و دو روایت را قاطی نکن.
مزیت: هم «رفع باگ‌ها» حفظ می‌شود، هم روش گیت (Reptile/PPO) بالاخره یک ران معتبر می‌گیرد.

### گزینه C — نسخهٔ فعلی به‌عنوان روش نهایی (BC + best-of-k)
همان مسیر ADR-007/ADR-011: ادعای meta-RL را حذف کن، meta-test را باز کن و ۵ بذر را بزن. کم‌ریسک‌ترین از نظر «نتیجهٔ موجود»، ولی **دیگر «معماری گیت» نیست**.

---

## ۸. کارهای باقی‌مانده و ریسک‌ها

1. **meta-test باز نشده** (۵ توزیع `{7,12,14,20,23}`) و ADR-013 نوشته نشده؛ همچنین `release/v0.3/`، `paper/CLAIMS_MATRIX.md` و `spec/make_report.py` **وجود ندارند** (فاز ۶ شروع نشده).
2. **seed 1 و 2** برای BC/BOK محور منابع نزده → همهٔ اعداد Phase 4 تک‌بذر است.
3. **Phase X (`ppo500`)** به‌عنوان کنترل تصمیم ADR-007 نزده؛ بدون آن، حذف PPO-on-θ از روش فقط با ۷ کانفیگ کوتاه پشتیبانی می‌شود.
4. **Phase 5 ناتمام** → ادعای انرژی/Pareto فعلاً ممنوع؛ مقاله فعلاً فقط latency.
5. **نسخهٔ فعلی روی `origin` نیست**؛ روی remote شخصی push شده. برای «پیش بردن نسخهٔ گیت» باید مشخص کنی کدام remote مرجع است.
6. **پاک‌سازی درخت کار:** پوشهٔ `MARGO` (workspace) شامل کپی‌های متعدد مخزن است (`BASELINE/`, `MRLCO/`, `v2v/`, `ERMIA-FULLMAML/`, `results/ckpt_ours_final_3500/MARGO`) که برخی درخت کارشان هزاران فایل dirty دارد (عمدتاً فایل‌های CRLF و دادهٔ `.gv`). فقط `MARGO_BASELINE/mrlco-new` نسخهٔ جاری است. برای جلوگیری از اشتباه، بقیه را آرشیو/حذف کن.
7. **محیط اجرا:** روی این مک‌بوک، پایتون پیش‌فرض `3.14` است و TensorFlow نصب نیست؛ پروژه به **پایتون ۳٫۷ + TF 1.15 با `tf.contrib`** نیاز دارد. پس هیچ ران/تستی محلی ممکن نیست و باید از ایمیج‌های `spec/Dockerfile.tf115`، `Dockerfile.tf115-gpu` یا `Dockerfile.tf115-nv2212` روی `kish-ai` (RTX 4090) استفاده شود. این یعنی بازتولید نتایج فعلی فقط روی همان میزبان ممکن است.

---

## ۹. بازتولید (دستورهای کوتاه)

```bash
cd /Users/erfanmhp/Desktop/ErfanMhp/publications/MARGO/MARGO_BASELINE/mrlco-new

# چه چیزی تغییر کرده؟
git log --oneline --reverse origin/main..HEAD
git diff --stat origin/main HEAD
git diff --name-status origin/main HEAD

# فقط تغییرات کد (بدون data/log)
git diff --stat origin/main HEAD -- $(git diff --name-only --diff-filter=M origin/main HEAD)

# تست‌ها (CPU؛ تست‌های TF اسکیپ می‌شوند)
python3 -m pytest env/mec_offloaing_envs/scheduler/tests -q

# گیت فاز ۲ (encoder) - نیازمند TF 1.15 با tf.contrib و پایتون ۳٫۷
python spec/phase2_gate.py     # باید بزند: Phase 2 encoder: PASS
```

---

## ۱۰. جمع‌بندی نهایی

- **تغییرات:** ۲۵ کامیت، ۳۶۶ فایل جدید، ۱۸ فایل ویرایش، صفر حذف؛ ۸ باگ مستند + ۶ باگ اضافی رفع شده؛ ~۲۴k خط کد و اسناد جدید؛ ۲۲۰ تست؛ صفر قابلیت حذف‌شده.
- **نتایج:** فیزیک و باگ‌ها اصلاح شد؛ encoder ablation منفی؛ best-of-k واقعاً کار می‌کند (`۵۸۸ → ۵۰۱٫۵` با k=64 روی obs v1 و `۴۹۳٫۸ → ۴۴۴٫۵` روی obs v2)؛ PPO-on-θ/CAVIA/EAS/CTX همه شکست خوردند؛ هیچ عددی نهایی و meta-test‌شده نیست.
- **آیا «گیت + رفع باگ» داریم؟** خانوادهٔ شبکه و رابط‌ها بله (با تغییرات داخلی انکودر)؛ اما ورودی/داده/شبیه‌ساز/پروتکل/روش نه. برای رسیدن به خواسته‌ات دقیقاً، گزینهٔ A (پچ‌های حداقلی روی `origin/main`) لازم است؛ برای کمترین تغییرِ مؤثر، گزینهٔ B.

---

## پیوست الف — هر ۱۸ فایل ویرایش‌شده با تعداد خطوط

| فایل | درج | حذف | جنس تغییر |
|---|---:|---:|---|
| `env/mec_offloaing_envs/offloading_env.py` | ۱۱۸ | ۴۹۰ | انتقال شبیه‌ساز به `scheduler/` + رفع باگ ۱/۶/۷ + تغییر قرارداد `sample_tasks` |
| `env/mec_offloaing_envs/offloading_task_graph.py` | ۱۹ | ۸۶ | بازنشستگی انکودرهای obs قدیمی (۶ همسایهٔ بریده) و واگذاری به `scheduler/encoder_obs.py` (باگ ۵) |
| `policies/graph2seq_encoder.py` | ۳۴۵ | ۱۸۴ | رفع clique (باگ ۴) + خواندن adj از obs + دوطرفهٔ اجباری (`fw+bw`) + `_readout` ماسک‌دار + برج‌های `gatv2`/`dagformer` اختیاری |
| `policies/meta_seq2seq_policy.py` | ۲۹۳ | ۲۹ | `end_token=3` (باگ ۲)، پارامتر `encoder_type`/`readout_type`، ماژول‌های اختیاری CAVIA/z/oracle-dist/EAS/nucleus |
| `policies/graph2seq_modules/aggregators.py` | ۵ | ۳ | میانگین ماسک‌دار همسایه (رفع آلودگی padding) |
| `policies/model_helper.py` | ۴ | ۱ | import اختیاری `tf.get_logger` |
| `meta_algos/MRLCO.py` | ۲۹۵ | ۱۴۱ | رفع mean-pseudogradient (باگ ۳/۸) + رفع لنگر value-clip + حالت‌های `publication`/`kl_bc`/`vf_only` + elite support |
| `meta_algos/ppo_offloading.py` | ۱۷۳ | ۹۹ | رفع لنگر value-clip + reset شدن lossها + گاردهای بودجهٔ فریز + anchor KL به BC |
| `meta_trainer.py` | ۴۰۱ | ۱۵۵ | رفع `range(5)` + گاردهای پروتکل فریز (`PPO_BATCH=20`، `K=3`) + val اجباری + کوپلینگ به `spec/` |
| `meta_evaluator.py` | ۱۷۳ | ۳۴۵ | حذف ۲۴۶ خط ریاضی تکراری (واگذاری به `schedule_via_adapter`) + pipeline support/query |
| `samplers/vectorized_env_executor.py` | ۳۲ | ۵ | رفع fork بعد از CUDA: `get_context("spawn")` + `CUDA_VISIBLE_DEVICES=""` |
| `samplers/seq2seq_*_process.py` (۲ فایل) | ۸+۸ | ۲+۲ | شاخهٔ اختیاری POMO (پیش‌فرض خاموش) |
| `samplers/__init__.py` | ۱ | ۳ | hygiene ایمپورت (برای worker اسپاون) |
| `utils/logger.py` | ۵ | ۲ | `mpi4py` اختیاری |
| `.gitignore` | ۱۴ | ۰ | نادیده‌گرفتن run/ckpt/data |
| `readme.md` | ۲۸ | ۷ | پایتون ۳٫۷، TF 1.15 CPU، ممنوعیت GPU تا فاز ۳، دستور gate |
| `paper/MARGO_paper_final.tex` | ۹۳ | ۹۳ | **فقط whitespace** (با `--ignore-all-space` صفر تفاوت) |

جمع‌بندی حذف ۱٫۶۴۷ خط: ~۹۹۰ خط **جابه‌جا** شد (env→`scheduler/`، task_graph→`encoder_obs`، ارزیاب→`scheduler/`، حلقه‌ها→`spec/learning_ops.py`، لیست گراف‌ها→`spec/split_loader.py`)؛ ~۲۰۵ بازآرایی درون‌فایلی؛ ~۱۷۶ پیاده‌سازی قدیمی بازنشسته (obs بریده، کران انرژی، reward per-step)؛ ۹۳ whitespace. **قابلیت حذف‌شدهٔ بدون جانشین: صفر.**

## پیوست ب — ۲۵ کامیت جدید (به ترتیب)

```
5bcd53b feat(spec): freeze split and learning protocol
69d3396 fix(spec): harden phase 0 protocols and preflight gates
810e91d fix(spec): freeze MR-LCO literature learning defaults
937e4c5 feat(sim): add canonical production scheduling engine
b49b60f refactor(sim): route environment scheduling through canonical engine
5c3029e refactor(sim): migrate callers to single engine
7058b88 feat(sim): add canonical energy reporting API
f2453b7 feat(sim): add post-hoc telescoping token rewards
db3b36a feat(sim): freeze weights and add Phase 1 gate
b611f70 docs(sim): close Phase 1 hygiene before freeze tag
c0cbecc feat(encoder): implement canonical DAG observations
367c10e fix(encoder): harden phase 2 runtime and reproducibility
06b97e9 docs(encoder): record Python 3.7 + TF 1.15 CPU smoke evidence
4625f69 docs(encoder): record same-SHA TF 1.15 evidence for 06b97e9
a8ced73 docs(encoder): close Phase 2 encoder freeze
fdfc98b feat(learn): wire frozen PPO, outer mean-PG, and split
f246006 fix(learn): align eval, PPO, and val with protocol
f17d264 test(learn): add inner/outer TF smoke and split scenarios
0c77692 docs(learn): close Phase 3 learning freeze
e16080c feat(eval): add Phase 4 campaign harness with GPU locks
39ea0ed fix(eval): tolerate missing git in Phase 4 provenance
30aa491 feat(eval): add TF 1.15 GPU image for NVIDIA hosts
464827e fix(eval): build TF 1.15 GPU image from official CUDA 10 base
1046c28 fix(eval): make Phase 4 GPU driver importable without extra host deps
e6cf007 feat(v0.3): Phase 1–5 results, spec ledger, and Persian progress report
```

سه کامیت اول + پنج کامیت `sim` = رفع باگ و موتور کانونیک؛ `encoder` = obs کانونیک (تغییر ورودی)؛ `learn`/`eval` = فریم‌ورک جدید؛ `docs` = مستندسازی.
