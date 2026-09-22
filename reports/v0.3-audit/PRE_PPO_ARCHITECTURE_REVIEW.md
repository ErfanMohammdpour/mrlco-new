# تحلیل معماری قبل از PPO — پاسخ به چهار سؤال

**تاریخ:** ۲۰۲۶-۰۹-۲۱ · **دامنه:** ① انرژی · ② objective/constraint · ③ mask/suffix/potential · ④ رادیو
**وضعیت کد:** ۳۳۸ تست غیر-TF pass · `feasible_action_mask` ساخته شده ولی **هنوز به env/policy وصل نیست**

---

## ۰. حکم کوتاه

۱. **معماری درست است**، ولی **یک blocker** دارد: قالب observation **هیچ فیچر deadline/mask ندارد** (v1: ۱۱ فیچر، v2: ۴ فیچر منابع؛ بررسی شد). یعنی سیاست از نظر ساختاری **کور** است به ددلاین، و در آن حالت mask تنها منبع اطلاعات feasibility می‌شود — دقیقاً همان چیزی که نگرانش بودی. تا obs درست نشود، PPO فقط یاد می‌گیرد «به shield اعتماد کن»، نه «deadline-aware باش».

۲. **mask اثباتی تقلب نیست**، به شرط چهار چیز که در بخش ۱ فهرست شده. تفکیک کلیدی: *hard feasibility* را policy نباید یاد بگیرد (mask حذفش می‌کند)؛ *deadline-awareness بین گزینه‌های شدنی و ددلاین soft/firm* را **باید** یاد بگیرد و برای آن obs لازم است.

۳. **روش A (masked logits) درست است**، با سه جزئیات فنی که اگر رعایت نشوند gradient bias می‌دهد (بخش ۲). روش B مکمل است نه جانشین.

۴. **suffix heuristic** فقط برای (potential + baseline ارزیابی + اختیاری warm-start). هرگز oracle در حلقهٔ آموزش (بخش ۳).

۵. **ترتیب باید یک تغییر داشته باشد:** `obs v3` **قبل از** effective-rank probe، چون probe باید روی همان obsی اجرا شود که PPO می‌بیند (بخش ۴).

---

## ۱. سؤال ۱ — آیا mask تقلب است؟ آیا policy کور می‌شود؟

### ۱٫۱ این کار در ادبیات استاندارد است، ولی دو خانوادهٔ متفاوت‌اند
| خانواده | مکانیزم | شرط مشروعیت |
|---|---|---|
| **Invalid action masking** | logits عمل‌های نامعتبر صفر/‎−∞ می‌شود | همان توزیع ماسک‌شده در rollout **و** در update؛ log-prob ذخیره‌شده از توزیع ماسک‌شده |
| **Safety shielding** | یک لایهٔ بیرونی عمل‌های unsafe را حذف می‌کند | shield باید **sound** باشد: فقط چیزی را حذف کند که اثباتاً غیرممکن است |

mask ما دقیقاً از نوع دوم است و sound بودنش تست شده (`test_dag_lower_bound_is_admissible` + `test_masked_action_has_no_feasible_completion`). پس «حذف ناحیهٔ ناممکن» است، نه «تصمیم‌گیری جای policy».

⚠️ ارجاع‌های استاندارد این حوزه (invalid action masking؛ shielding) را از دانش عمومی می‌گویم: **قبل از استناد در مقاله venue را تأیید کن** — قبلاً در همین پروژه گرفتار ارجاع تأییدنشده شدیم.

### ۱٫۲ ولی نگرانی تو در یک نقطه **واقعی** است
دو چیز را باید از هم جدا کرد:

| نوع | چه کسی حل می‌کند | آیا policy باید یاد بگیرد؟ |
|---|---|---|
| ددلاین **hard** که اثباتاً نقض می‌شود | shield (mask) | نه — ممکن نیست، پس یادگیری لازم ندارد |
| ددلاین **hard/firm** که چند action شدنی دارد | policy | **بله** — باید بهترین گزینهٔ شدنی را انتخاب کند |
| ددلاین **soft** | جریمهٔ tardiness در J | **بله** — و gradient دارد |

ما امروز فقط مورد اول را داریم. مورد دوم و سوم نیاز به **فیچر ددلاین در observation** دارند، و آن وجود ندارد. یعنی:
- سیاست نمی‌تواند بفهمد «این تسک ددلاین سخت دارد، پس HELPER را انتخاب نکن حتی اگر سریع‌تر به‌نظر می‌رسد».
- سیاست نمی‌تواند tardiness را با latency معامله کند (ministrators: soft) چون تسک را با ددلاینش نمی‌بیند.
- نتیجه: رفتار می‌شود «logits می‌گوید MEC → shield می‌گوید نه → UE اجرا می‌شود» — همان سناریویی که خودت نوشتی.

### ۱٫۳ پنج شرط لازم برای اینکه mask «کمک» باشد نه «عصا»
1. mask فقط اثباتی (✅ داریم)
2. **obs شامل فیچر deadline و ماسک** (❌ نداریم → blocker)
3. soft-tardiness داخل J و قابل یادگیری (✅ داریم)
4. dead-end guard: اگر **همهٔ** actionها ماسک شدند، **هیچ‌کدام** ماسک نشود (❌ نداریم — باید اضافه شود؛ وگرنه shield بن‌بست می‌سازد و softmax روی مجموعهٔ خالی NaN می‌دهد)
5. اندازه‌گیری: `mask/active_rate`، `mask/forced_rate`، و **ablation بدون mask** به‌عنوان شاهد یادگیری (❌ نداریم)

**جواب صریح به «آیا policy هیچ‌وقت deadline awareness یاد نمی‌گیرد؟»** — با شرایط ۲+۳+۵ **یاد می‌گیرد**؛ بدون ۲، **نه**. پس blocker همان obs است.

---

## ۲. سؤال ۲ — decoder چطور mask را بگیرد؟

### ۲٫۱ روش A درست است (و کافی)، با این پیاده‌سازی دقیق
```python
logits   = policy_head(state)               # [B, 20, 3]
neg      = tf.float32.min / 1e9             # نه -inf مطلق (NaN در softmax/backward)
masked   = tf.where(feasible, logits, neg)
pi       = tf.nn.softmax(masked, axis=-1)
logp_a   = tf.nn.log_softmax(masked, axis=-1)  # gather کنش a از همین
```
سه جزئیاتی که اگر رعایت نشوند **bias** می‌دهد:
1. **Log-prob قدیم و جدید هر دو از توزیع ماسک‌شده.** اگر `old_logp` از توزیع ناماسک‌شده ذخیره شود، نسبت PPO غلط و gradient مغشوش می‌شود.
2. **Entropy فقط روی کنش‌های مجاز.** `-Σ_{a∈valid} p_a log p_a` (نرمال‌شده یا خام، ولی ثابت در طول ران). entropy روی کل ۳ عمل، مقدار را مصنوعی بالا می‌برد و bonus را خراب می‌کند.
3. **mask باید تابع محض state باشد.** اگر در rollout و در update (بعد از چند گام به‌روزرسانی) ماسک متفاوت باشد، importance ratio بی‌معنا می‌شود. پس mask را از state مشتق کن، نه از حافظهٔ جاری.

### ۲٫۲ روش B مکمل است، نه جایگزین
فیچرهای پیشنهادی برای **obs v3**:
| فیچر | چرا |
|---|---|
| `has_deadline`, one-hot `deadline_type ∈ {none,soft,firm,hard}` | سیاست باید نوع قید را ببیند |
| `slack_ratio = clip((d_i − LB_i)/max(d_i,ε), −1, +1)` | مهم‌ترین سیگنال: چقدر حاشیه دارد |
| `criticality_class` one-hot (low/med/high) | مفهوم داده، جدا از penalty |
| `tardiness_weight` | ضریب جریمه |
| **`LB_ready` نرمال‌شده برای هر سه action + `feasible[a]` (۳+۳)** | سیاست بفهمد «چرا ماسک شد»، نه اینکه ناگهان یک گزینه غیب شود |
| `is_sink` (هست) + `return_hop_bytes` | بازگشت به UE در ددلاین sink مؤثر است |

با این‌ها policy می‌تواند *پیش‌بینی* کند و generalize کند؛ mask نقش تور ایمنی می‌گیرد، نه منبع اطلاعات.

### ۲٫۳ جای mask در معماری فعلی
mask در `scheduler/feasibility.py` + `suffix.py` ساخته می‌شود ⇒ نیاز به دو کابل جدید:
- `encoder_obs` باید فیچرهای بالا را بسته‌بندی کند (obs v3، مثل v2 با سوئیچ نسخه، تا legacy بیت‌به‌بیت بماند).
- `meta_seq2seq_policy` باید یک تانسور `feasible[B,20,3]` بگیرد و در سه helper (train/sample/greedy) اعمال کند.
هیچ‌کدام امروز وجود ندارد؛ این بخشی از کار PPO است، نه ③.

---

## ۳. سؤال ۳ — heuristic suffix لازم است؟

| نقش | مجاز؟ | یادداشت |
|---|---|---|
| ساختن potential (`state_potential`) | ✅ | پیاده‌شده و تست‌شده؛ با γ=1 ترتیب پلن‌ها از ترمینال می‌آید |
| **baseline / ceiling** در ارزیابی | ✅ | مثل role قبلی pair-search: «سقف جست‌وجو»، نه «روش» |
| warm-start با BC (مسیر `bc_greedy_mec` موجود) | ✅ اختیاری | فقط initialization |
| **انتخاب/بازنویسی کنش در حلقهٔ آموزش** | ❌ | leakage؛ policy چیزی یاد نمی‌گیرد |
| **mask بر پایهٔ شکست heuristic** | ❌ | همان قاعدهٔ ③ که تست شده |

**ریسک پنهان که باید اعلام کنی:** potential بر پایهٔ یک heuristic است، پس shaping به کیفیت آن heuristic وابسته است. توصیه: `potential_gap = Ĵ(s_0) − J_actual` را لاگ کن، و یک ablation «با shaping / بدون shaping» بزن تا سهم آن مشخص باشد. اگر gap بزرگ شد، shaping گمراه‌کننده است.

---

## ۴. سؤال ۴ — ترتیب مراحل

ترتیب تأییدشدهٔ قبلی: ①→②→③→④→⑤ effective-rank→⑥ decoder 256→dataset→PPO.
**پیشنهاد من یک جابه‌جایی دارد:**

```
① energy ✅  ② objective/constraints ✅  ③ mask/suffix/potential ✅  ④ radio ✅
   ↓
⑤a  OBS v3  (deadline + criticality + tardiness_weight + LB/feasible per action)   ← blocker
⑤b  effective-rank probe  (روی obs v3 نهایی، نه v1/v2)
⑥   decoder 128→256 (+ حذف state projection)
⑥b  masked-softmax + entropy روی valid + dead-end guard + logprob سازگار
⑦   dataset deadline-aware  (D_G → EFT/LFT → subdeadlines، feasibility-aware)
⑧   constrained PPO + meta-RL  (cost critic + dual، یا Lagrangian)
```

دلیل جابه‌جایی:
- **probe روی obs v1/v2 بی‌معناست** اگر PPO با obs v3 اجرا شود؛ rank پایین ممکن است فقط نشان دهد «ددلاین در ورودی نیست».
- **dataset باید بداند obs چه می‌بیند**؛ وگرنه ددلاین‌های تولیدشده در ورودی دیده نمی‌شوند.

---

## ۵. چک‌لیست عددی قبل از PPO (این‌ها را همین حالا می‌توان طراحی کرد)

| سنجه | تعریف | هدف |
|---|---|---|
| `mask/active_rate` | نسبت تصمیم‌هایی که ≥۱ action ماسک دارند | باید **کم** باشد؛ بالا بودنش ⇒ ددلاین‌ها غیرواقعی‌اند |
| `mask/forced_rate` | تصمیم‌هایی که فقط **یک** action مجاز دارند | نزدیک صفر؛ بالا ⇒ فرمولاسیون منحط |
| `policy/argmax_masked_rate` | چقدر argmax سیاست **بدون** ماسک، ماسک‌شده بود | **نزولی** در طول آموزش = شاهد یادگیری deadline awareness |
| `eval/no_mask_hard_miss_rate` | نرخ نقض hard بدون shield | فاصلهٔ سیاست خام از صدا بودن |
| `potential_gap` | `Ĵ(s_0) − J_actual` | کیفیت heuristic |
| `objective/c_E_raw`, `hard_miss_total` | موجود | شدت واقعی نقض |

اگر `argmax_masked_rate` در طول آموزش ↓ نشود، یعنی سیاست فقط دارد به shield تکیه می‌کند و باید فیچر/جریمه تقویت شود — این معیار، تشخیص دقیق همان ترسی است که نوشتی.

---

## ۶. دفاع علمی در مقاله
- shield را به‌عنوان **sound safety filter** معرفی کن، با اثبات admissibility (تست‌های ③) — نه به‌عنوان contribution الگوریتمی.
- **اجباری:** گزارش نرخ فعال بودن shield + `forced_rate` + **ablation بدون shield**. بدون این‌ها داور می‌گوید policy آزاد نبوده.
- **صادقانه:** بگو hard deadline با shield تضمین می‌شود و deadline-awareness بین گزینه‌های شدنی از فیچرها و soft penalty یاد گرفته می‌شود. دو روایت را قاطی نکن.
- ارجاع‌های پیشنهادی (invalid-action-masking، shielding، CPO، PPO-Lagrangian، CMDP) را **قبل از سابمیت راستی‌آزمایی کن**؛ من این‌ها را صرفاً به‌عنوان جهت معرفی می‌کنم، نه نقل قول تأییدشده.

---

## ۷. جمع‌بندی: چهار تغییر لازم قبل از PPO
1. **obs v3** با فیچرهای ددلاین/criticality/penalty/ماسک — **blocker**، بدون آن mask به عصا تبدیل می‌شود.
2. **dead-end guard** در mask (همه ماسک ⇒ هیچ ماسک) — وگرنه NaN/بن‌بست.
3. **masked softmax** با entropy روی valid و logprob سازگار در rollout و update.
4. **ابزارهای سنجش** `mask/active_rate`, `mask/forced_rate`, `argmax_masked_rate`, `potential_gap` + ablation بدون shield.

با این چهار مورد، معماری فعلی هم علمی و هم قابل دفاع است و می‌توان با خیال راحت وارد ⑤ شد.
