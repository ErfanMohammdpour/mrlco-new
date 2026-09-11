# MARGO v0.3 — نتایج تا ۲۰۲۶-۰۹-۱۰

`paper_result=false` روی همهٔ ردیف‌ها. meta-test دست‌نخورده. `margo_v0.1_primary` (۳۵۰۰ iteration) شروع نشده.

منبع کاننیکال عددها: `spec/PHASE4_WORKLOG.md`. روایت: `spec/PHASE4_DIAGNOSTIC_LOG.md`. JSON ماشینی: `spec/PHASE4_WORKLOG.json`. خام Kish: `spec/kish_log_archive/`.

میزبان: kish-ai RTX 4090. ایمیج: `margo-phase4-tf115-nv2212`. واحد: **ثانیه** برای makespan، **ژول** برای انرژی. هیچ makespan را ms ننویس.

---

## حکم یک‌خطی

روش مقالهٔ v0.3 فعلاً این است، نه چیز دیگر:

**BC (encoder=`meanagg`، readout=`mean`، obs v2) + best-of-k inference.** بدون few-shot، بدون EAS، بدون CAVIA، بدون PPO-on-θ، بدون جستجوی `schedule()` سر تست.

سقف فیزیکی هنوز `h` + جستجو است (~۴۳۹)، نه LSTM یک‌گذر (~۵۷۵ روی OOD قدیمی؛ ~۴۹۴ با obs v2). انرژی Pareto پارک شده. Phase X (`ppo500`) و Phase 6 (meta-test / پنج بذر) نزده.

نمی‌توان ادعا کرد بهتر از MRLCO است: پروتکل فرق دارد، انرژی کامل نیست، meta-test نیست، بیشتر محور منبع فقط seed0 است.

---

## روش قفل‌شده و ADR

| ADR | حکم |
|---|---|
| ADR-001 | `E = UE + HELPER` (compute+radio). **MEC server compute خارج از objective.** |
| ADR-007 | PPO-on-θ و CAVIA-on-z از روش حذف. |
| ADR-008 | inference = π ± نمونه k≤۶۴ + یک `schedule()` هر نمونه. pair/2-opt معلم یا سقف‌اند، نه π. |
| ADR-009 | محور ۱ = خانواده DAG؛ محور ۲ = پروفایل منبع (Phase 4). |
| ADR-010 | شکاف OOD نمایشی نیست. backbone = **meanagg**. readout = **mean**. |
| ADR-011 | EAS روی φ سطح-توزیع **FAIL**. روش = BC صفرشات + best-of-k. |
| ADR-012 | محور منبع: obs v2 (FEATURE_DIM 15، PACKED_DIM 54). v1 hash فریز. |

هدر معماری: `MARGO-METHOD-v0.3`.

---

## Split فریز

- train ۱۵ dist: `{1,3,4,5,8,9,11,13,15,18,19,21,22,24,25}` × ۱۰۰ گراف
- val `{2,6,10,16,17}`
- meta-test `{7,12,14,20,23}` — **Phase 6 only**
- DAG بیست‌تسکی، `{0=Local, 1=MEC, 2=V2V}`، `end_token=3`
- محور منبع (Phase 4): گرید ۴۵؛ train ۱۳ / val held-out ۲ + frozen continuity / meta-test ۸ (نزده)

Control A (`bc_unseen`، obs v1): val greedy **۵۷۵.۱** / meta-test **۵۵۷**. Mix val ~۰.۲۱/۰.۷۵/۰.۰۴. `n_non` p50 **۴**.

---

## مرجع heuristic (validation مگر ذکر شود)

| مرجع | T (s) | نقش |
|---|---|---|
| all-MEC | val **۶۳۴** / train **۶۲۸** | کف تنک‌نبودن |
| publication greedy (از empty) | train ~**۵۹۴** | greedy ضعیف مقالهٔ قدیم |
| greedy_from_mec | val **۴۶۴** / train **۴۴۸** | heuristic قوی |
| 2-opt از greedy_from_mec | val **۴۲۴** / train **۴۱۴.۱** | سقف جستجو (معلم) |
| pair k50 (search، نه روش) | val **۴۳۸.۹** vs oracle **۴۳۶.۶** | سقف `h`+جستجو |

---

## وضعیت فازها

| فاز | فایل | وضعیت | گیت |
|---|---|---|---|
| 0 freeze spec | `PHASE0_PRUNE_AND_FREEZE.md` | **CLOSED** `phase0-freeze-v0.1` | PASS |
| 1 simulator (قدیم) | `PHASE1_STATUS.md` | **CLOSED** `phase1-freeze-v0.1` | PASS |
| 1 encoder (v0.3) | `PHASE1_ENCODER_ABLATION.md` | **تمام** | شکاف نمایشی نیست. meanagg+mean |
| 2 best-of-k | `PHASE2_BEST_OF_K_INFERENCE.md` | **تمام** seed {0,1,2} | `T_best_32≤510` PASS (**۵۰۹.۱۲**) |
| 3 EAS-on-φ | `PHASE3_EAS_SUPPORT_ADAPTATION.md` | **تمام، FAIL** | query ≤۵۲۰ FAIL. ADR-011 |
| 3B EAS per-instance | `PHASE3B_INSTANCE_EAS_LOCK.md` | **تمام، FAIL** | Δ vs bok **−۰.۲s** |
| X ppo500 | `PHASEX_PPO500_CONTROL.md` | **نزده** | بند Phase 6 |
| 4 resource axis | `PHASE4_SCENARIO_AXIS.md` | **seed0 تمام** | BC+BOK PASS؛ EAS/CTX FAIL. seed 1–2 نزده |
| 5 energy Pareto | `PHASE5_ENERGY_PARETO.md` | **پارک** | فیزیک لوکال PASS؛ معلم کامل نه |
| 6 eval/paper | `PHASE6_EVALUATION_RELEASE.md` | **بسته** | meta-test قفل تا ADR-013 |

---

## نردبان T روی validation (ثانیه، پایین‌تر بهتر)

محور DAG-only (obs v1، پروفایل فریز قدیمی) مگر ستون Phase 4:

| روش | T | یادداشت |
|---|---|---|
| 2-opt | **۴۲۳.۶** | سقف معلم |
| pair search k50 | **۴۳۸.۹** | heuristic؛ ADR-008 روش نیست |
| BOK k=64 (Phase 2، triple ckpt) | **۵۰۱.۵±۰.۳۷** | ۳ seed |
| BOK k=32 (Phase 2) | **۵۰۹.۱۲±۰.۴۸** | گیت ≤۵۱۰ |
| BOK k=64 (Phase 4 obs v2، seed0) | **۴۴۴.۵** | frozen profile |
| BOK k=32 (Phase 4 obs v2، seed0) | **۴۴۷.۷** | frozen |
| BC greedy Phase 4 obs v2 seed0 | **۴۹۳.۸** | frozen؛ vs Phase1 **۵۷۷.۰** (Δ −۸۳.۲) |
| BC greedy Phase 1 mean readout | **۵۷۷.۰** | seed0 |
| Control A `bc_unseen` | **۵۷۵** | obs v1 |
| BC-2opt greedy | **۵۸۱.۸** | معلم بهتر، OOD همان |
| rewrite all-MEC | **۶۱۰.۶** | `rewrite_hurts` |
| all-MEC | **۶۳۴** | |

---

## C2 — نتیجه منفی PPO-on-θ (هفت config + binary)

Occupancy expert تحت PPO روی θ خراب می‌شود. مهارت در برابر greedy_from_mec / 2-opt ساخته نمی‌شود.

| run | نتیجه |
|---|---|
| `diag_500_parallel` | MEC `max_action_frac≥0.95` iter **۲۱**؛ iter ۱۸۰ MEC **۰.۹۹۸** |
| `diag_latency_tmec` | T **۹۵۸→۷۹۶**؛ صفر پلن ۱–۲ غیر-MEC |
| `diag_pomo_tmec` | بدتر از lat50؛ elite `n_non` p50 **۹** |
| `diag_bc_greedy_tmec` | iter0 نزدیک expert؛ iter9 Local **۵۵٪** |
| `diag_kl_bc_ppo` | Local **۲۱٪→۷۴٪**؛ `kl_bc` **۱.۱۸→۱۶** |
| `diag_bc_fewshot` | val **۵۸۰→۶۵۷** |
| `diag_binary_lat` | ایزوله بدون V2V/انرژی. در worklog هنوز «running»؛ در `PHASEX` نقل: best T **۵۰۲** iter5 بعد **۵۳۵**، MEC **۴۵٪→۹۵٪**. JSON نهایی در این snapshot قفل نشده |

گیت باقی C2: Phase X `ppo500` پیش‌بینی P1–P3. اگر train greedy ≤۴۴۸، ADR-007 باز می‌شود.

IL در-توزیع کار می‌کند: `bc_continue` train greedy **۴۶۴** / token **۰.۹۶۸**. OOD T منتقل نمی‌شود.

---

## Phase 1 encoder (تمام)

Fixture Kish قبل از ادیت. رگرسیون atol 1e-5.

Val T، readout=triple، ۳ seed:

| encoder | n_params | mean±std | ΔT vs meanagg |
|---|---|---|---|
| **meanagg** | ۴۲۹۱۸۵ | **۵۸۸.۷۸±۱۱.۸۳** | — |
| gatv2 | ۶۲۶۸۱۷ (۱.۴۶×) | ۶۰۰.۵۴±۴.۲۹ | **−۱۱.۷۶** |
| dagformer | ۵۳۱۰۷۳ (۱.۲۴×) | ۵۹۷.۳۳±۲.۸۴ | **−۸.۵۵** |

گیت ΔT<۱۰ برای هر دو (هر دو منفی) → **«شکاف OOD نمایشی نیست»**.

Readout meanagg seed0 vs triple ۵۸۸.۱۴: **mean ۵۷۷.۰۲** (Δ −۱۱.۱۲) / attn ۵۸۳.۰۸ / max ۵۹۳.۹۴ / zero ۵۹۵.۰۰. Adopt **mean**.

Ckpt Phase 3: `meanagg_mean/seed_0`. Phase 2 bok روی **triple**.

---

## Phase 2 best-of-k (تمام، seed 0,1,2)

Ckpt: `meanagg_triple/seed_0` sha `01d566ac80e5fe58`. آموزش نیست.

Val n=500:

| k | evals | s0 | s1 | s2 | mean±std | s/graph |
|---|---|---|---|---|---|---|
| 1 | 1 | 588.1409 | 588.1409 | 588.1409 | **588.1409±0** | 0.0027 |
| 8 | 8 | 533.0132 | 532.5030 | 531.1619 | **532.23±0.96** | 0.022 |
| 32 | 32 | 508.9313 | 509.6677 | 508.7621 | **509.12±0.48** | 0.088 |
| 64 | 64 | 501.6941 | 501.7953 | 501.1182 | **501.54±0.37** | 0.175 |

Train `T_best_64` **۴۱۵.۳۷±۰.۰۹** (2-opt train ۴۱۴.۱). Temp k=32: 0.7→**۵۱۹.۸۹**؛ 1.0→۵۰۹.۱۲؛ 1.3→**۵۰۲.۲۸**. Dup k=64 val **۰.۷۰۳**. Oracle-within-samples k=64 **۰.۰۲۲** — تقریباً هیچ‌وقت 2-opt را نمی‌زند.

حکم: نمونه‌گیری mean OOD را ارزان می‌گیرد (۵۸۸→۵۰۹). به 2-opt ۴۲۴ نمی‌رسد.

---

## Phase 3 EAS (تمام، منفی)

Ckpt `meanagg_mean/seed_0`. φ=`lastlayer` = ۳۸۴ پارامتر.

Main lastlayer+pg_il seed0: query greedy **۵۷۷.۷→۵۷۶.۴**. گیت ≤۵۲۰ **FAIL**. Support best پایین می‌آید (جستجو روی support کار می‌کند). Mix سالم. L_IL تخت.

Ablation seed0 query T*: il ۵۷۷.۹؛ **pg ۵۷۱.۲** (بهترین، باز FAIL)؛ film ۵۷۷.۷ و `phi_l2=0` (مرده)؛ full ۵۸۰.۷ (بدتر).

Path B بودجه برابر B=32، n=500: bok **۴۹۸.۸** vs EAS-inst **۴۹۸.۶** (Δ −۰.۲). گیت ≥۵s **FAIL**.

سه CAVIA قبلی تخت: 584.5→585.1؛ 584.5→584.2؛ 572.7→572.7.

Oracle dist_id: ident **۵۷۵.۱**؛ train **۴۵۳.۷** vs ۴۶۴؛ val **۵۸۴.۲** vs A ۵۷۵. **`oracle_no_gain`**. φ سطح-توزیع نساز.

Rewrite از all-MEC: val **۶۱۰.۶** vs A ۵۷۵؛ `n_apply` val 0.004 / test **۰**. **`rewrite_hurts`**.

---

## Phase 4 محور منبع (seed0 تمام)

obs v2: FEATURE_DIM **۱۵** / PACKED_DIM **۵۴**. stats v2 n_graphs=۱۹۵۰۰. v1 sha دست‌نخورده `94e598759e4b02544bf216220a48225521abd6977419880c4a818631bb9c5e83`.

Unique-dir (با Phase 5 ids) **۴۵**.

### Expert 2-opt (CPU)

۱۳ پروفایل train ~۲۲ دقیقه. mix_mec با UL بالا می‌رود (UL3≈۰.۵۱–۰.۵۷ / UL11≈۰.۷۱–۰.۸۰). frozen twopt **۴۱۴.۱** train / val **۴۲۳.۶۲**. Held-out: `p_5_5_10` **۵۱۶.۹**؛ `p_9_5_10` **۳۶۳.۶**.

### BC profiles seed0 — PASS (یک‌طرفه)

۱۲۰ ep / ۱۹۵۰۰ گراف / ۱۵۲ دقیقه. Frozen greedy **۴۹۳.۸** vs Phase1 **۵۷۷.۰** (Δ −۸۳.۲). Continuity دوطرفه FAIL؛ **یک‌طرفه no-regression PASS**. Held-out: `p_5` ۶۲۱.۸/۵۱۶.۹ tok 0.670؛ `p_9` ۴۱۸.۴/۳۶۳.۶ tok 0.737. Mix با expert یکی. Ckpt sha `2a385073e71a0689`.

### EAS profiles seed0 — FAIL

T0 **۵۲۱.۱** → T* **۵۲۱.۵** (Δ **−۰.۴s**). همان الگوی ADR-011 روی محور منبع.

### CTX profiles seed0 — FAIL

T0 **۵۲۸.۲** = Tctx **۵۲۸.۲** (Δ **۰.۰**). z استفاده نمی‌شود. few-shot سناریو مرده.

### BOK profiles seed0 — PASS

آموزش نیست. Frozen n=500 vs 2-opt **۴۲۳.۶**:

| k | T_best | vs greedy | vs 2-opt |
|---|---|---|---|
| 1 | **۴۹۳.۸۲** | 0 | +۷۰.۲ |
| 8 | ۴۵۹.۳۹ | −۳۴.۴ | +۳۵.۸ |
| 32 | **۴۴۷.۶۷** | −۴۶.۲ | +۲۴.۰ |
| 64 | **۴۴۴.۴۸** | −۴۹.۳ | +۲۰.۹ |

Held-out: `p_5` greedy 621.8 → T64 **546.9** / expert 516.9؛ `p_9` 418.4 → **377.9** / 363.6. Mix frozen greedy L/M/V **۰.۲۲۶/۰.۷۰۳/۰.۰۷۲**. Temp k=32: 0.7→453.1، 1.0→447.7، 1.3→444.8.

seed 1 و 2 برای BC/BOK پروفایل **نزده**.

---

## Phase 5 انرژی (پارک ۲۰۲۶-۰۹-۱۰)

هدف: `J_λ = λ T_norm + (1−λ) E_norm`، λ ∈ {1.00, 0.75, 0.50, 0.25, 0.00}. λ=1 باید با لیبل latency Phase 4 بیت‌به‌بیت یکی باشد.

فیزیک لوکال PASS: 2-opt J_1 = 2-opt T روی toy؛ helper compute روی V2V؛ HV=6 روی سه نقطه؛ انرژی ۵ گراف 1e-6 J.

Kish CPU smoke seed0 **PASS**: λ=1.0 match Phase 4 (`n=8`). frozen_7_5_10 هشت گراف:

| λ | T | E | mix_mec |
|---|---|---|---|
| 1.00 | 333 | 314 | 0.775 |
| 0.75 | 341 | 245 | 0.831 |
| 0.50 | 430 | 74 | 0.969 |
| 0.25 / 0.00 | 468 | 39 | 1.000 |

روند T↑ E↓ با افت λ. λ پایین → all-MEC چون MEC compute در E نیست (ADR-001).

Full **STOPPED**. ۱ از ۶۵: `p_3_3_5` λ=1.00 T=718.9 E=736.4 match Phase 4 n=1500. وسط λ=0.75 کشته شد. npz جزئی مانده. **ادعای Pareto نکن.** BC/EAS انرژی نیست.

بدون Phase 5 کامل، Phase 6 فقط latency است.

---

## آنچه ادعا نکن

- بهتر از MRLCO / DAMRL
- Graph2Seq با DAG-edge به‌عنوان نوآوری (encoder نتیجهٔ ADR-010 است)
- PPO inner + Reptile به‌عنوان روش
- درصد نسبت به publication greedy
- few-shot / EAS / CAVIA / PEARL context
- انرژی Pareto / جبههٔ آموخته
- عدد meta-test به‌عنوان نتیجهٔ مقاله
- `paper_result=true`

---

## مانده (بدون ران در این نشست)

1. **Phase X** `margo_v0.3_diag_ppo500_control` — PPO از صفر، ۵۰۰ iter، انرژی خاموش. کد جاب هنوز نیست. بند Phase 6.
2. Phase 4 seed 1 و 2 برای BC/BOK پروفایل (std).
3. Resume Phase 5 اگر Pareto در مقاله لازم شد.
4. Phase 6 فقط بعد ADR-013 + تگ `v0.3-freeze` + باز شدن meta-test.

Artifact این گزارش: `spec/RESULTS_TO_DATE_2026-09-10.md`.
