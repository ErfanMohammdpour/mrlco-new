# ③ — Optimistic DAG lookahead، Suffix سازنده، mask اثباتی، و potential

**تاریخ:** ۲۰۲۶-۰۹-۲۱ · **وضعیت:** پیاده و تست‌شده · **لایهٔ pure scheduler — بدون TF/PPO/radio**
**کل تست‌های غیر-TF: ۳۲۶ passed, 5 skipped** (۱۵ تست جدید ③)
**فایل اصلی:** `env/mec_offloaing_envs/scheduler/suffix.py`

---

## ۰. دو مکانیزم جدا (هیچ‌وقت قاطی نمی‌شوند)

| مکانیزم | نقش | اختیار mask؟ |
|---|---|---|
| **PROOF**: `dag_lower_bound_ready/masks` | relaxation روی *همهٔ* ادامه‌ها: جای‌گذاری آزادِ تسک‌های تصمیم‌نگرفته، رقابت صفر منابع، هم‌مکانیِ فرضی برای انتقال‌ها | **بله** (تنها مرجع) |
| **CONSTRUCTIVE**: `construct_fastest_feasible_suffix` | یک suffix حریص قطعی → Ĵ(s) | **نه** — شکست heuristic هرگز mask نمی‌کند |

```
action
 ├─ LB(task)  → > deadline ? MASK
 ├─ DAG relaxation → hard descendant unreachable ? MASK
 └─ constructive suffix → found: Ĵ(s)   |   failed: NO MASK  (reason=heuristic_only_not_used)
```

## ۱. ③.1 — `SuffixContext` و `ObjectiveContext`
- `ObjectiveContext(latency_ref_s, beta_soft, energy_budget_j=None, deadlines={task: TaskDeadline})` — **کاملاً decoupled از energy model**؛ هیچ ارجاعی به مدل فیزیکی یا بودجهٔ پنل ندارد. تست اعتبارسنجی: `latency_ref_s > 0`، بودجه اگر داده شد `> 0`.
- `SuffixContext(dag, resources, order, objective, decisions, prefix_finish, index, cycles_per_bit)` — وضعیت جاری: تسک فعلی، تسک‌های باقی‌مانده، تصمیم‌های گرفته‌شده، finish پیشوند، sink بودن، bytes موفق‌ها، و `parent_inputs()` که همان API صدا ②B را برمی‌گرداند.

## ۲. ③.2 — suffix سازنده
`construct_fastest_feasible_suffix(ctx, action=None)`:
در هر تسک باقی‌مانده، هر سه placement با `finish_lower_bound` تخمین زده می‌شود، نامزدها فیلتر می‌شوند و **سریع‌ترین شدنی** انتخاب می‌شود؛ در پایان پلن کامل با موتور کانونیک `schedule()` اجرا و Ĵ از نتیجهٔ واقعی محاسبه می‌شود.
- `action` داده‌شده ⇒ lookahead مشروط به آن action (برای `evaluate_action`).
- `action=None` ⇒ `state_potential(ctx)`: Ĵ به‌عنوان **تابع وضعیت**، که برای اتحاد telescoping لازم است.
- **اصلاح مهم حین پیاده‌سازی:** ددلاین `soft` هرگز placement را رد نمی‌کند (soft = هزینه، نه قید). فقط `hard`/`firm` فیلتر می‌کنند. وگرنه یک penalty به constraint تبدیل می‌شد.

## ۳. ③.3 — ارزیابی action
`evaluate_action(ctx, action)` → `ActionSuffixEvaluation(action, location, mask, mask_reason, lb_ready_s, suffix, potential)`؛
`evaluate_all_actions(ctx)` و `best_by_potential(evals)` (کم‌ترین Ĵ بین actionهای مجاز؛ اگر Ĵ نداشتند، کم‌ترین LB).
`feasible_action_mask(ctx)` فقط از دو منبع اثباتی mask می‌دهد.

## ۴. ③.4 — mask اثباتی
`dag_lower_bound_masks(ctx, action)`: relaxation را از وضعیت جاری تا انتهای DAG پیش می‌برد و اگر **هر** ددلاین hard در آن تخفیف هم نقض شود ⇒ `mask=True` با دلیل `dag_lower_bound_infeasible`.
relaxation: تسک فعلی با action اجباری، بقیه با **ارزان‌ترین tier**، انتقال‌های وابستگی **صفر** (هم‌مکانی مجاز است)، رقابت منابع صفر، ولی sinkها **ارزان‌ترین هاپ بازگشت** را می‌پردازند.

## ۵. ③.5 — تست‌ها (۱۵ تست)
| تست | چه چیزی را قفل می‌کند |
|---|---|
| **Test 1** `test_later_deadline_decides_current_action` | ددلاین یک تسک **بعدی** تصمیم تسک فعلی را عوض می‌کند: MEC روی task0 ⇒ suffix found؛ UE روی task0 ⇒ **mask اثباتی** (`dag_lower_bound_infeasible`) |
| Test 1b | با ددلاین شل، هر سه action suffix دارند و Ĵها **متفاوت**اند؛ `best_by_potential` کم‌ترین Ĵ را می‌گیرد (نه لزوماً action=UE) |
| **Test 2** `test_heuristic_failure_alone_keeps_action_open` + `test_mask_reason_never_claims_proof_from_heuristic` | شکست heuristic هرگز به mask یا به دلیل «اثباتی» تبدیل نمی‌شود |
| **Test 3** `test_masked_action_has_no_feasible_completion` | برای هر action ماسک‌شده، **همهٔ ۹ ادامهٔ ممکن** شبیه‌سازی می‌شود و در هیچ‌کدام ددلاین hard برآورده نمی‌شود (همان گزارهٔ قوی که خواستی) |
| Test 3b `test_dag_lower_bound_is_admissible` | relaxation هرگز از availability واقعی هیچ ترکیبی بالاتر نمی‌زند (admissible) |
| Test 3c `test_mask_comes_only_from_proof_sources` | تطابق دلیل mask با منبع اثباتی |
| **Test 4** `test_terminal_potential_equals_actual_J` | `Ĵ(s_T) = J_actual` (برابری تا places=9) |
| **Test 4b** `test_shaped_reward_sign_and_telescoping` | با `γ=1`: `Σr = Ĵ(s_0) − Ĵ(s_T)` دقیق؛ `Ĵ(s_0)` برای دو پلن **یکسان**؛ اختلاف امتیاز دو پلن **فقط** از ترمینال می‌آید؛ پلن سریع‌تر امتیاز بهتر |
| Test 4c | شکل ماسک `[bool,bool,bool]` |

## ۶. نکتهٔ طراحی که حین تست کشف شد (مهم برای ④/PPO)
در ابتدا potential را «مشروط به action» حساب می‌کردم (`construct(..., action)`)، ولی آن‌گاه `Ĵ(s_0)` بین پلن‌های مختلف **یکسان نیست** و اتحاد telescoping معنا از دست می‌دهد. تفکیک درست:
- `state_potential(ctx)` → **تابع وضعیت** ⇒ ورودی ریوارد: `r_t = Ĵ(s_{t-1}) − γĴ(s_t)`
- `evaluate_action(...)` → lookahead مشروط به action ⇒ **فقط برای ارزیابی/انتخاب action**، نه برای ریوارد.

با این تفکیک، `Σr = Ĵ(s_0) − Ĵ(s_T) = −(J_actual − const)` و ترتیب پلن‌ها فقط توسط objective نهایی تعیین می‌شود (تست 4b همین را عددی قفل می‌کند).

## ۷. آنچه ③ عمداً انجام نداد
- ❌ energy وارد J نشد (کانال مستقل ماند)
- ❌ firm/hard داخل potential نرفتند
- ❌ PPO / Lagrangian / dual variable: هیچ
- ❌ radio rate: دست نخورد (`R = B·log2(1+SINR)` در RADIO_MODEL_V1)
- ❌ TF: هیچ خطی از trainer تغییر نکرد

## ۸. گام بعد
`④ RADIO_MODEL_V1`: جداسازی `bandwidth_hz`، `spectral_efficiency`/SINR، `effective_rate_bps` (و حذف فرض پنهان ۱ bit/s/Hz).
**سپس الزاماً همهٔ تست‌های soundness/LB/suffix/Pareto مرحلهٔ ③ دوباره اجرا شوند**، چون تغییر rate رادیو مستقیماً deadline feasibility و potential را جابه‌جا می‌کند.
