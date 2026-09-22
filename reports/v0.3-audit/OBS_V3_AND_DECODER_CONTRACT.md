# ⑤a — Observation v3 و قرارداد ورودی decoder

**تاریخ:** ۲۰۲۶-۰۹۲۱ · **وضعیت:** پیاده و تست‌شده · **کل تست‌های غیر-TF: ۳۵۳ passed, 5 skipped** (۱۵ تست جدید)
**فایل‌ها:** `scheduler/encoder_obs.py` (v3) · `scheduler/static_bounds.py` (جدید) · `spec/encoder_feature_stats_v3.json` · `spec/make_obs_v3_stats.py` · `spec/embedding_deadline_probe.py` · `tests/test_obs_v3.py`

---

## ۱. مسئله‌ای که ⑤a حل می‌کند
در بازبینی قبل از PPO روشن شد: obs فعلی **هیچ فیچر ددلاین/feasibility ندارد** (v1: ۱۱ فیچر، v2: ۱۵). یعنی سیاست ساختاری کور است و shield تنها منبع اطلاعات feasibility می‌شود. ⑤a این کوری را برمی‌دارد.

## ۲. obs v3 — ۱۶ فیچر جدید (FEATURE_DIM ۱۵→۳۱ · PACKED_DIM ۵۴→۷۰)

| گروه | فیچرها | چرا |
|---|---|---|
| حضور/نوع ددلاین | `has_deadline`, `deadline_is_soft/firm/hard` (one-hot) | سیاست باید نوع قید را ببیند |
| حاشیه | `slack_ratio_min_lb = clip((d − min_a LB_a)/d, −1, 1)` | **مهم‌ترین سیگنال**: «چقدر وقت دارم» |
| criticality | `criticality_low/medium/high` (one-hot) | مفهوم داده، جدا از penalty |
| penalty | `tardiness_weight_scaled = log1p(w)/log1p(10)` | ضریب soft tardiness |
| بازگشت sink | `return_hop_over_min_lb` | هزینهٔ اجتناب‌ناپذیر بازگشت به UE |
| **per-action** | `lb_ue/mec/helper_log1p` + `feasible_ue/mec/helper` | سیاست بفهمد **چرا** ماسک شد، نه اینکه گزینه غیب شود |

همهٔ ۱۶ فیچر **کراندار by construction** (0/1 یا ratio کلیپ‌شده) ⇒ **z-score نمی‌شوند**؛ ردیف‌های آماری‌شان identity است. به همین دلیل `encoder_feature_stats_v3.json` از v2 مشتق می‌شود (۱۵ ستون آماری کپی + ۱۶ ردیف identity) بدون هیچ آمار ساختگی؛ `spec/make_obs_v3_stats.py --check` هم staleness را می‌گیرد.

v1 و v2 **دست‌نخورده و بیت‌به‌بیت**: سوئیچ `set_obs_version("v1"|"v2"|"v3")` یا `MARGO_OBS_VERSION`. تست `test_v1_and_v2_are_untouched` ابعاد و مسیر legacy را قفل می‌کند.

## ۳. `static_bounds.py` — مرزهای استاتیک (مبنای فیچرها)
```
start_lb[pos][a] = max_p min_a' finish_lb[pred_pos][a']      # پیشینی‌ها آزاد، انتقال صفر
finish_lb[pos][a] = start_lb + C_task / f_a
ready_lb[pos][a]  = finish_lb                                    non-sink
                  = finish_lb + transfer_lb(a → UE, output)       sink
```
- همان relaxation لایهٔ PROOF (③) ولی **بدون prefix**: وابسته فقط به DAG + مدل منابع/رادیو + `cycles_per_bit` — **هرگز به policy**، پس leakage ندارد.
- تست `test_bounds_are_admissible_against_real_schedules`: برای هر ۲۷ ترکیب action، `min_ready_lb ≤ availability` واقعی.
- تست `test_faster_tier_has_smaller_bound`، `test_sink_pays_return_hop_only_for_remote_tiers`، و `test_feasibility_matches_the_runtime_mask_semantics` (پرچم feasibility در obs = همان قاعدهٔ mask زمان اجرا).

## ۴. تست وابستگی به ددلاین (خواستهٔ صریح تو)
| نسخه | نتیجه |
|---|---|
| v1 و v2 | دو state که فقط در ددلاین تفاوت دارند، obs **کاملاً یکسان** (`assert_allclose atol=0`) ⇒ **کوری مستند شد** |
| v3 | obs **متفاوت** است؛ `slack_ratio` یکنوا در ددلاین؛ با ددلاین تنگ حداقل یک action `feasible=0` می‌شود |

نیمهٔ دوم (embedding بعد از encoder) نیاز به TF دارد ⇒ `spec/embedding_deadline_probe.py` روی kish:
```bash
python spec/embedding_deadline_probe.py            # pre-training: encoder_outputs باید متفاوت باشند
python spec/embedding_deadline_probe.py --ckpt <dir>   # بعد از آموزش: توزیع action باید تغییر کند
```
خروجی `encoder_sees_deadline: true/false` می‌دهد و با کد خطا fail می‌شود اگر ددلاین به representation نرسیده باشد.

## ۵. قرارداد دقیق ورودی decoder (مبنای ⑥b)

```
PER TASK (decoder position k، هم‌تراز prioritize_sequence)
├─ features[k, 0:31]            obs v3 (۳۱ فیچر بالا، استانداردشده جز ۱۸ فیچر کراندار)
├─ fw_idx[k, 0:19]              successor decoder-indices  (PAD=-1)     ← داخل packed
├─ bw_idx[k, 0:19]              predecessor decoder-indices             ← داخل packed
└─ mask[k]                      node mask                             ← داخل packed
packed obs: [B, 20, 70]

SIDE TENSORS (جدا از packed، برای mask و attention)
├─ feasible[B, 20, 3]           bool/float — از static bounds (obs) و در rollout از shield زمان اجرا
├─ lb_ready[B, 20, 3]           float — همان ready_lb (برای تفسیر و advantage)
└─ slack[B, 20]                 float — slack_ratio

decoder
  logits = head(h)                      # [B, 20, 3]
  masked = where(feasible, logits, NEG)  # NEG = 1e9-، نه -inf مطلق
  π = softmax(masked); logπ = log_softmax(masked)
  entropy = -Σ_{a feasible} π log π      # فقط کنش‌های مجاز
```
**قواعد الزامی (از بازبینی):**
1. `old_logp` و `new_logp` هر دو از توزیع ماسک‌شده.
2. entropy فقط روی مجموعهٔ مجاز.
3. mask تابع محض state (rollout و update یکسان).
4. **dead-end guard:** اگر هر سه action ماسک شدند ⇒ هیچ‌کدام ماسک نشود (و hard-miss در کانال constraint ثبت شود). *هنوز پیاده نشده — بخشی از ⑥b است.*

## ۶. آنچه ⑤a عمداً انجام نداد
- هیچ خطی از policy/decoder تغییر نکرد (mask هنوز به شبکه وصل نیست) — کار ⑥b.
- PPO/Lagrangian/dual: هیچ.
- deadline برای دیتاست فعلی وجود ندارد، پس فیچرهای v3 در گراف‌های فعلی صفر می‌مانند؛ کد و تست آماده‌اند و پس از تولید دیتاست deadline-aware فعال می‌شوند.
- effective-rank probe (⑤b) دست نخورد؛ الان می‌توان روی **obs v3** اجرا شود، نه v1/v2.

## ۷. گام بعد
`⑤b effective-rank` روی obs v3 — سنجش rank مؤثر حالت ۲۵۶ و projection ۱۲۸ برای تصمیم ⑥ (decoder 256).
