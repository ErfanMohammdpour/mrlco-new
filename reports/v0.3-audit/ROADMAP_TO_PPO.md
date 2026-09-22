# نقشهٔ راه تا Constrained PPO — تصمیم‌ها، ترتیب، گیت‌ها

**تاریخ:** ۲۰۲۶-۰۹-۲۱ · **وضعیت:** برنامه (اجرا نشده) · **مبنای تصمیم:** repo @ `8a3d391` (push‌شده) · ۳۶۲ تست غیر-TF pass

---

## ۰. تصمیم ۱۲۸ → ۲۵۶: **فعلاً نه**

| پرسش | پاسخ |
|---|---|
| آیا encoder ۲۵۶ با decoder ۱۲۸ از نظر معماری درست است؟ | **بله.** `node_h=256` (حافظهٔ Luong) و `state=128` (خلاصه برای init) دو نقش متفاوت دارند؛ projection عمدی است نه bug. |
| آیا mismatch است؟ | **نه.** `readout_proj` → ۲۵۶ و `state_projection` → ۱۲۸ دقیقاً همان طراحی فریز است. |
| آیا evidence برای ۲۵۶ داریم؟ | **نه.** erank ساختاری: `node_h` ≈ ۲۹٫۴ از ۲۵۶ · `state128` ≈ ۲۸٫۸ از ۱۲۸. هیچ‌کدام نزدیک اشباع نیستند. |
| آیا ۱۲۸ قطعاً کافی است؟ | **هنوز ۱۰۰٪ نه** — چون probe با وزن تصادفی بود. شرط تجدیدنظر در بخش ۴. |
| گلوگاه فعلی کجاست؟ | **policy/masking**، نه capacity. |

**سه شرط تجدیدنظر (هر سه لازم):** (A) erank واقعی `state128` روی چک‌پوینت آموزش‌دیدهٔ v3 نزدیک اشباع باشد (>۱۰۰)، (B) A/B با seed/دیتاست/بودجهٔ ثابت بهبود معنادار بدهد، (C) شواهد underfit بودن decoder (مثلاً loss دیکودر بالا با encoder غنی). تا آن‌وقت `state_projection: 128 → 256` انجام نمی‌شود.

---

## ۱. یک تصحیح مهم به ترتیب پیشنهادی

پیشنهاد اول این بود که **اول** activation probe روی چک‌پوینت آموزش‌دیده اجرا شود. یک مشکل فنی دارد:

- چک‌پوینت‌های موجود (تنها چک‌پوینت‌های روی دیسک):
  - `runs/phase4/margo_v0.1_diag_500_parallel/seed_0/ckpt/meta_model_best_val.ckpt`
  - `runs/phase4/margo_v0.1_learning_probe/seed_0/ckpt/meta_model_{best_val,final,0}.ckpt`
- این ران‌ها با **obs v1** (packed ۵۰) آموزش دیده‌اند. شبکه‌ای که برای ۵۰ بعد ساخته شده، **نمی‌تواند** obs v3 (packed ۷۰) بخورد. پس «erank حالت آموزش‌دیده **با ددلاین**» تا اولین آموزش v3 وجود ندارد.

**اصلاح ترتیب:**
```
1) ⑥b (masked-softmax + dead-end guard + metrics + wiring)     ← blocker واقعی
2) ران تشخیصی کوتاه v3 (۳۰۰–۵۰۰ تکرار، بدون ددلاین)            ← sanity + metrics
3) activation probe روی چک‌پوینت v3 آموزش‌دیده                 ← عدد تصمیم ⑥
4) دیتاست deadline-aware
5) curriculum: no-deadline → soft → firm → hard
6) A/B: decoder 128 vs 256 (فقط اگر شرط‌های A/B/C)
```
در ضمن برای گرفتن یک **baseline آموزش‌دیدهٔ بدون ددلاین** می‌توان همین حالا (روی kish) این را زد:
```bash
python spec/dump_encoder_activations.py --obs-version v1 \
    --ckpt runs/phase4/margo_v0.1_diag_500_parallel/seed_0/ckpt \
    --graphs 120 --out acts_v1.npz
python spec/effective_rank_probe.py --mode activations --npz acts_v1.npz
```

**دو باگ که همین حالا در ابزار درست شد** (تا مرحلهٔ ۳ قابل اجرا باشد):
1. `dump_encoder_activations.py` انکودر را با `scope_name="dump_encoder"` می‌ساخت ⇒ نام متغیرها با چک‌پوینت policy (`encoder/...`) نمی‌خواند و restore بی‌صدا شکست می‌خورد. الان `scope_name="encoder"` + `Saver(var_list=encoder_vars)` + گزارش `missing_in_ckpt`.
2. فلگ `--obs-version` اضافه شد؛ اگر نسخهٔ obs با چک‌پوینت نخواند، ابزار صریح هشدار می‌دهد (PARTIAL) نه اینکه عدد بی‌معنا بدهد.

---

## ۲. گام بعد: ⑥b — بزرگ‌ترین blocker

### ۲٫۱ کد لازم (TF side)
| # | کار | جای دقیق | معیار پذیرش |
|---|---|---|---|
| 1 | `feasible[B,20,3]` به‌عنوان ورودی policy | `policies/meta_seq2seq_policy.py` (`Seq2SeqNetwork.create_decoder` و سه helper) | سه helper (train/sample/greedy) همه از توزیع ماسک‌شده نمونه می‌زنند |
| 2 | **masked logits**: `tf.where(feasible, logits, -1e9)` **قبل** از softmax | همان | هیچ `-inf` مطلق (NaN در backward) |
| 3 | entropy فقط روی کنش‌های مجاز | `samplers/base.py` + PPO entropy | `H_valid` لاگ شود، نه `H_all` |
| 4 | `old_logp` و `new_logp` هر دو ماسک‌شده | `meta_algos/ppo_offloading.py`, `meta_algos/MRLCO.py` | تست: با mask ثابت، ratio ≡ ۱ |
| 5 | **dead-end guard**: اگر هر سه ماسک ⇒ هیچ‌کدام ماسک نشود + `all_invalid_rate` لاگ | تابع ماسک قبل از policy | softmax روی مجموعهٔ خالی هرگز رخ ندهد |
| 6 | mask تابع **محض state** (rollout و update یکسان) | env → `suffix.feasible_action_mask(ctx)` | تست تکرارپذیری |
| 7 | wiring: env ماسک را از وضعیت پیشوند می‌سازد و در `env_infos` می‌فرستد | `offloading_env.step` | mask در `samples_data` دیده شود |

### ۲٫۲ سنجه‌های اجباری (Phase 2 metrics)
```
mask/active_rate        چند تصمیم ≥۱ کنش ماسک دارند        (باید کم باشد)
mask/forced_rate        چند تصمیم فقط یک کنش مجاز دارند     (~۰ ⇒ فرمولاسیون منحط نیست)
mask/all_invalid_rate   چند تصمیم همه ماسک بودند            ≥۰ و هر رخداد لاگ شود
policy/argmax_masked_rate   چند بار argmax سیاست (بدون ماسک) ماسک‌شده بود  ← شاهد یادگیری
policy/invalid_rate     کنش نامعتبر نمونه‌شده              باید ۰ باشد
policy/entropy_valid    آنتروپی روی کنش‌های مجاز
objective/hard_miss_total, objective/c_E_raw, objective/energy_violation_rate
```

### ۲٫۳ تست‌های ⑥b (قبل از هر ران جدی)
1. با mask ثابت: `ratio == 1` و `logp_new == logp_old`.
2. mask یک کنش ⇒ احتمال آن کنش صفر و entropy روی دو کنش باقی‌مانده.
3. همه ماسک ⇒ guard فعال، توزیع یکنواخت روی سه کنش، `all_invalid_rate += 1`.
4. کنش ماسک‌شده هرگز نمونه نمی‌شود (۱۰۰۰ نمونه).
5. mask از state بازتولید می‌شود (دو بار محاسبه ⇒ یکسان).

---

## ۳. گام ۳: ران تشخیصی کوتاه v3
```bash
MARGO_OBS_VERSION=v3 MARGO_CONSTRAINTS=spec/constraints.yaml \
  python -c "from spec.phase4_train_driver import run_primary_seed; run_primary_seed(0, allow_gpu=True)"
```
با `objective_mode="log_only"` و **بدون ددلاین** (دیتاست فعلی)، ۳۰۰–۵۰۰ تکرار:
- گیت عبور: `objective/*` لاگ می‌شود · `mask/*` معنادار است · `policy/invalid_rate == 0` · هیچ NaN · loss نزولی.
- اگر این پاس نشد، هیچ‌کدام از گام‌های بعدی معنا ندارد.

## ۴. گام ۴: دیتاست deadline-aware (اصلاح پیشنهاد)
پیشنهاد اول: `Loose 1.5×LB`, `Medium 1.05×LB`, `Tight 0.8×LB`.
**اصلاح:** `Tight = 0.8×LB` باعث می‌شود نمونه‌ها **ذاتاً ناسازگار** شوند (هیچ کنشی feasible نیست) ⇒ shield فقط `all_invalid` تولید می‌کند و policy چیزی یاد نمی‌گیرد. پیشنهاد:
| رژیم | ددلاین | تضمین |
|---|---|---|
| Loose | `1.50 × LB_min` | راحتی؛ چند کنش آزاد |
| Medium | `1.10 × LB_min` | trade-off واقعی |
| Tight | `1.02 × LB_min` | فقط کنش بهینهٔ LB زنده می‌ماند (سخت ولی شدنی) |
| Infeasible (کوچک، برچسب‌دار) | `0.8 × LB_min` | **فقط** برای تست dead-end guard، جدا گزارش شود، نه در آموزش |
به‌علاوه: `deadline_type` با profile: `HIGH→hard`, `MEDIUM→firm`, `LOW→soft` (کلاس و کانال جدا بمانند) و anchor = **static LB** (plan-independent، بدون leakage).

## ۵. گام ۵: curriculum
```
مرحلهٔ ۱  بدون ددلاین (فقط latency)          ← تأیید یادگیری پایه
مرحلهٔ ۲  soft (جریمه در J)
مرحلهٔ ۳  firm (کانال c_F + ε_F)
مرحلهٔ ۴  hard (shield + feasibility)
```
هر مرحله فقط وقتی جلو می‌رود که مرحلهٔ قبل پایدار باشد (KL، clip fraction، واریانس reward، `hard_miss_total`).

## ۶. گام ۶: A/B decoder (فقط با شرط‌های بخش ۰)
`Model A`: encoder 256 → state projection → 128 · `Model B`: حذف projection ⇒ state 256 (و در صورت تمایل variant D با `Dense 256→256`). ثابت: seed، دیتاست، تعداد گام، hyperparams. مقایسه: reward، `hard_miss_total`، energy، latency، sample efficiency، پایداری.

---

## ۷. ممنوع‌ها (تا اطلاع ثانوی)
❌ تغییر معماری encoder (Graph2Seq → Transformer) · ❌ افزایش hidden بدون evidence · ❌ افزودن جملهٔ جدید به reward · ❌ تغییر هم‌زمان obs + decoder + reward + PPO (attribution از بین می‌رود) · ❌ تغییر radio/energy physics تا پایان ⑥b.

---

## ۸. ترتیب کامیت‌ها
```
commit 1  chore(probe): fix encoder scope + obs-version in activation dump      [انجام‌شده، uncommitted]
commit 2  feat(mask): masked-softmax plumbing into policy (3 helpers)
commit 3  fix(ppo): mask-aware old/new logprob + entropy over valid actions
commit 4  feat(mask): dead-end guard + all_invalid_rate
commit 5  feat(obs): shield metrics logging (active/forced/argmax-masked/invalid)
commit 6  feat(dataset): deadline-aware generator (loose/medium/tight + labelled infeasible)
commit 7  feat(train): curriculum no-deadline -> soft -> firm -> hard
commit 8  exp(decoder): 128 vs 256 A/B (only if conditions A/B/C hold)
```
هر کامیت با تست‌های خودش؛ هیچ کامیتی دو مؤلفه را هم‌زمان عوض نمی‌کند.

---

## ۹. خلاصهٔ یک‌خطی
**گلوگاه فعلی capacity نیست، constraint integration است.** مسیر: ⑥b → ران تشخیصی v3 → activation probe v3 → دیتاست ددلاین‌دار → curriculum → و فقط بعد از پایداری، A/B عرض decoder.
