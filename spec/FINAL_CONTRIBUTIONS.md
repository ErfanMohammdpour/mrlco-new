# سهم علمی v0.3 نسبت به DAMRL و MRLCO

روش: `MARGO-METHOD-v0.3`.  
**CAVIA-on-z از روش حذف شد** (ADR-007؛ فقط ablation منفی).  
شواهد جدول‌ها diagnostic Kishاند (`paper_result=false`) مگر خلافش نوشته شود.  
فاز ۱–۳ گیت‌خورده: encoder شکاف نمایشی نیست؛ bok `T_best_32≤510` PASS؛ EAS FAIL → روش = BC+best-of-k (ADR-011). فاز ۴ seed0 PASS برای BC/BOK. فاز ۵ پارک. فاز X و ۶ نزده. ادعای مقاله فقط بعد eval پنج‌بذر + meta-test (فاز ۶). Snapshot: `spec/RESULTS_TO_DATE_2026-09-10.md`.

---

## 1. چه چیزی را ادعا نکن

DAMRL (Liao et al., FGCS 2027) زودتر زده: GAT دوطرفه، PEARL `z` از گذار `(s,a,r,s')`، dual replay، تطبیق بدون ∇ تست.

MRLCO (Wang et al., TPDS 2021): Seq2Seq + inner PPO + first-order meta روی θ.

MeanAgg ما از GAT آن‌ها قوی‌تر نیست. فاز ۱: gatv2/dagformer **بدتر** از meanagg (ΔT −۱۱.۸ / −۸.۶). Encoder نوآوری نیست (ADR-010).

**CAVIA را روش ننویس.** سه ران query تخت: `cavia_frozen` 584.5→585.1؛ `cavia_strong` 584.5→584.2؛ `cavia_bccont` 572.7→572.7.

---

## 2. چهار سهم قابل دفاع (به شرط گیت)

### C1 — مسئله + پروتکل

پلن گسستهٔ ۲۰تایی `{UE,MEC,V2V}` با تقویم نیم‌دوطرفه، انرژی `total_mobile_joules`، holdout ساختاری fat×density، support/query جدا (20/80).

| شواهد آماده | گیت باقی |
|---|---|
| فیزیک `lat50` OK؛ `ROUTE_TABLE` 3×3؛ `latin_grid_holdout_v1`؛ محور منبع seed0 (ADR-012) BC mix با teacher | انرژی فاز ۵ (پارک)؛ ۵ seed فاز ۶ |

بدون فاز ۴ این فقط سیستم‌مدل است، نه meta قابل مقایسه با DAMRL/MRLCO.

### C2 — نتیجهٔ منفی با مکانیزم (هفت config)

meta-PPO روی θ occupancy را می‌کشد؛ در برابر heuristic قوی (greedy_from_mec / 2-opt) مهارت نمی‌سازد.

| run | عدد (`PHASE4_DIAGNOSTIC_LOG.md`) |
|---|---|
| `margo_v0.1_diag_500_parallel` | MEC `max_action_frac≥0.95` iter **21**؛ iter 180 MEC **0.998** |
| `margo_v0.1_diag_latency_tmec` | T **958→796**؛ صفر پلن ۱–۲ غیر-MEC |
| `margo_v0.1_diag_pomo_tmec` | elite `n_non` p50 **9** |
| `margo_v0.1_diag_bc_greedy_tmec` | iter 9 Local **55%** |
| `margo_v0.1_diag_kl_bc_ppo` | Local **21%→74%**؛ `kl_bc_mean` **1.18→16** |
| `margo_v0.1_diag_bc_fewshot` | val **580→657** |
| `margo_v0.2_diag_binary_lat` | ایزولهٔ بدون V2V/انرژی (§26). worklog هنوز running؛ نقل PHASEX: best 502@iter5 سپس 535، MEC 45%→95% |

گیت باقی: فاز X `ppo500` پیش‌بینی P1–P3. اگر P2 غلط شد (train greedy ≤448)، ADR-007 باز می‌شود.

### C3 — سیاست تقطیرشده از جستجو + best-of-k (few-shot حذف)

BC از لیبل 2-opt + نمونه‌گیری از π (ADR-008، k≤64). EAS روی φ **حذف از روش** (ADR-011).

| شواهد آماده | گیت باقی |
|---|---|
| encoder ADR-010 meanagg+mean؛ Phase2 `T_best_32` **509.12±0.48** PASS؛ Phase4 obs v2 frozen greedy **493.8** / T32 **447.7** / T64 **444.5** | seed 1–2 محور منبع؛ ۵ seed فاز ۶ |

فاز ۳ FAIL → C3 = صفرشات+best-of-k. ادعای few-shot حذف.

### C4 — سوئیت baseline قوی

publication greedy از empty ضعیف است (~594 train). مرجع صادق: greedy_from_mec val **464** / 2-opt **424** / all-MEC **634**.

| شواهد آماده | گیت باقی |
|---|---|
| `twopt_expert` train 414.1؛ Hamming-2 `frac_h2` **0.73** | HEFT در همان `schedule()` (فاز ۶)؛ MRLCO-style با clip/sync درست؛ ۵ seed |

---

## 3. سهم‌های کمکی / غیر ادعا

**Triple readout:** رد شد. mean بهتر از triple (۵۷۷ vs ۵۸۸ seed0). جزء معماری، نه contribution جدا.

**Pair/motif search:** سقف فیزیکی (`pair_seq` bestimp k20 val **447.9**). در مقاله baseline/oracle، **نه** جزء π (ADR-008).

**FiLM / z:** نامزد `adapt_subset` فاز ۳. **CAVIA-on-z حذف شد.**

**PEARL dual-buffer / CTX z:** فاز ۴ ساخته شد و ΔT=۰. روش نیست.

---

## 4. یک جمله برای مقدمه

> DAMRL تطبیق سناریوی پیوستهٔ گام‌به‌گام با PEARL بدون گرادیان است. MRLCO تطبیق Seq2Seq با PPO روی وزن سیاست است. رقبا با greedy ضعیف مقایسه می‌کنند. MARGO پلن گسستهٔ مقید به تقویم V2V را با Graph2Seq می‌سازد؛ نشان می‌دهد PPO روی θ occupancy تنک expert را خراب می‌کند؛ سیاست را از جستجوی 2-opt تقطیر می‌کند و با best-of-k نمونه‌گیری می‌کند. few-shot / EAS / CAVIA-on-z جزء روش نیستند (گیت فاز ۳ قرمز).

---

## 5. نقشهٔ شواهد → ادعا

| ادعا | شواهد آماده | شکاف / گیت |
|---|---|---|
| PPO-on-θ few-shot ضرر است | `bc_fewshot`, `kl_bc_ppo`, `par500`, `bc50` | فاز X بودجهٔ بلند |
| BC train کلون می‌کند، OOD نه | `bc_unseen`, `bc_2opt`, `bc_scheduled` | بسته (encoder کمکی نکرد) |
| نمایش `h` روی OOD برای جستجو کافی است | `pair_head`, `pair_seq` | نباید روش شود (ADR-008) |
| **CAVIA T query را بهتر نمی‌کند** | `cavia_frozen`, `cavia_strong`, `cavia_bccont` | بسته؛ روش نیست |
| EAS query را از k0 بهتر می‌کند | Phase3 577.7→576.4؛ PathB Δ−0.2؛ Phase4 Δ−0.4 | **رد**؛ روش نیست |
| best-of-k mean OOD | Phase2 T32 **509.12**؛ Phase4 T32 **447.67** | ۵ seed فاز ۶ |
| mean readout بهتر از triple | seed0 577.02 vs 588.14 | ADR-010 |
| 0.5/0.5 + V2V فیزیک درست | ADR-001 + `lat50` فیزیک OK | فاز ۵ Pareto پارک |
| obs v2 محور منبع | frozen greedy 493.8؛ mix با UL | seed 1–2 |
