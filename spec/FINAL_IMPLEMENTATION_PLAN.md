# پیاده‌سازی روی کد فعلی — فازها و گیت

ریشه: `MARGO_BASELINE/mrlco-new`  
روش: `MARGO-METHOD-v0.2-cavia`  
نکن: بازنویسی تگ فریز، شروع 3500، Docker/TF روی لپ‌تاپ، `paper_result=true` قبل از پنج‌بذر.

---

## 0. نقشهٔ فایل

| قطعه | مسیر | کار فاز |
|---|---|---|
| Graph2Seq + readout | `policies/graph2seq_encoder.py` | دست نزن مگر ablation readout |
| LSTM+Luong | `policies/meta_seq2seq_policy.py` | FiLM روی `encoder_state` قبل `clone(cell_state=...)` |
| Context | **جدید** `policies/cavia_context.py` | `z`, `γ,β`, init |
| Inner CAVIA | **جدید** `meta_algos/cavia_offloading.py` | REINFORCE-on-z، Adam فقط `z` |
| حلقه | `meta_trainer.py` | مسیر جدید؛ `MRLCO.UpdateMetaPolicy` را صدا نزن |
| Driver | `spec/phase4_train_driver.py` + `phase4_campaign.py` | `margo_v0.2_diag_cavia_*` unique dir |
| Kish | `spec/kish_gpu.sh` | هدف `cavia0` / `cavia20` |
| فیزیک | `scheduler/engine.py` `reward.py` | دست نزن |
| اسپلیت | `spec/split_loader.py` | همان support/query |
| BC ckpt | `runs/phase4/margo_v0.1_diag_bc_2opt/seed_0/` | نقطهٔ شروع θ |
| جستجوی جفت | `spec/pair_*.py` | وارد eval روش نشود |

لایهٔ فریز `k_steps=3` PPO برای ablation منفی نگه دار؛ پیش‌فرض روش نشود.

---

## فاز ۰ — رگرسیون `z=0` ≡ BC

هدف: FiLM خام با `z=0` همان greedy T val **≈581.8**.

کار:

1. `Seq2SeqNetwork.create_decoder`: بعد از `encoder_state`، اگر placeholder `z` داده شد FiLM؛ وگرنه identity.
2. `γ = 1 + Dense(z)`, `β = Dense(z)`، وزن صفر.
3. Load `bc_2opt` `bc_core.ckpt` با همان UID/slot (`kl_bc_anchor` اگر scope عوض شد).
4. تست واحد CPU: یک گراف، logits `z=0` برابر شبکهٔ بدون FiLM (atol 1e-5).
5. گیت Kish کوتاه: unseen val n=500 greedy T در `[575, 590]`.

خروجی: `runs/phase4/margo_v0.2_diag_cavia_identity/`.  
اگر این نپاسد فاز ۱ شروع نشود.

---

## فاز ۱ — CAVIA روی val، encoder+decoder فریز

هدف: آیا ۲۰ گام روی `z` query را بهتر از k0 می‌کند **بدون** خراب کردن mix.

کار:

1. `cavia_context.py`: `z` متغیر، نه در `θ` سیاست.
2. Inner: ۲۰ گراف support، sample π، `A = T - mean_T_batch`،  
   `L = mean(stop_grad(A) * Σ_k logπ(a_k|h,z))`، Adam `z`.
3. `θ_enc`,`θ_dec` `trainable=False`.
4. Eval: val `{2,6,10,16,17}` و در همان run meta-test گزارش (انتخاب مدل فقط val).
5. لاگ per dist: T k0, T k5, T k20, mix, `n_non`, `||z||`, `Δz`.

گیت پاس (diagnostic):

```text
val T_k20 < val T_k0 − 15     # حاشیه؛ اگر 5–15 «ضعیف ولی زنده»
mix Local k20 < 0.28          # k3 PPO به ~0.30 رفت
n_non p50 ∈ [3, 6]
```

گیت رد: T بالا مثل 657 یا mix Local>0.35 → اول lr_z / baseline / sample vs greedy inner را عوض کن. برنگرد به pair-search به‌عنوان روش.

خروجی: `margo_v0.2_diag_cavia_frozen/seed_0/cavia_eval.json`.

آزمایش موازی ارزان (همان ckpt، بدون support):

```text
z = MLP کوچک از readout خود query   # متا نیست
```

اگر این ≈ CAVIA support: سهم متا ضعیف است؛ برو سراغ FiLM گراف نه few-shot. در `FINAL_CONTRIBUTIONS` C1 را پایین بیاور.

---

## فاز ۲ — outer روی FiLM / φ0

اگر فاز ۱ پاس:

- `Dense_γ, Dense_β` قابل آموزش
- outer Adam روی `L_query` بعد از inner z
- هنوز `θ_enc` فریز
- `θ_dec` پیش‌فرض فریز؛ ablation: LR `1e-5` روی لایهٔ logits

بودجه: ۵۰–۲۰۰ outer، meta-batch ۱۰، `paper_result=false`.  
ولیدیشن هر ۵۰: k0 و k20.

گیت: val k20 بهتر از فاز ۱ پایدار بماند؛ k0 نباید مثل PPO بد شود.

---

## فاز ۳ — ablation معماری و هدف

فقط اگر فاز ۲ زنده:

| ablation | سؤال |
|---|---|
| w/o z (k0 همیشه) | few-shot می‌ارزد؟ |
| inner PPO-on-θ k=3 از همین ckpt | شاهد منفی آماده |
| readout mean-only / max-only / triple | C3 |
| loss T در برابر 0.5/0.5 | C4 هدف |
| K∈{0,5,10,20} | منحنی تطبیق |

Pair refine در این فاز فقط ستون «heuristic ceiling» اگر لازم شد از JSON قبلی، نه اجرای دوبارهٔ اجباری.

---

## فاز ۴ — paper eval

شروط شروع (هر سه):

1. تأیید صریح در چت
2. `--i-allow-gpu` + `MARGO_ALLOW_GPU=1`
3. گیت فاز ۲ پاس روی val؛ meta-test برای انتخاب hparam استفاده نشده

سپس: ۵ بذر، meta-test `{7,12,14,20,23}`، k0 و k20، `paper_result` فقط روی این آرتیفکت.

3500 PPO فریز همچنان جدا و پیش‌فرض نیست.

---

## جزئیات فنی FiLM (فاز ۰)

در `create_decoder` بعد از ساخت `encoder_state` و قبل از `AttentionWrapper.clone`:

```text
# encoder_state: tuple(LSTMStateTuple, LSTMStateTuple)  هر کدام c,h = s[B,128]
s = encoder_state[0].h
gamma = 1.0 + film_gamma(z)     # Dense(32→128), kernel init zeros
beta  = film_beta(z)            # Dense(32→128), zeros
s_t = gamma * s + beta
new_state = tuple(LSTMStateTuple(c=s_t, h=s_t) for _ in range(2))
# سپس zero_state.clone(cell_state=new_state)
```

Placeholder: `z [None, 32]`؛ اگر `B` گراف یک خانواده یک z دارند: `tile`.  
`get_actions` باید `z` در `feed_dict` بگذارد. پیش‌فرض صفر.

REINFORCE: `logπ` از همان `sample_decoder_logits`؛ `stop_gradient` روی θ با `tf.stop_gradient` روی وزن‌ها یا `var_list=[z]`.

```text
opt.minimize(L, var_list=[z])   # نه get_trainable_variables سیاست
```

---

## ترتیب تست محلی / Kish

```text
python3 spec/phase4_gate.py          # نباید بشکند؛ method جدید را جدا اضافه کن
# لپ‌تاپ: فقط تست واحد FiLM identity (بدون TF GPU)

# Kish:
kish_gpu.sh cavia0     # فاز 0
kish_gpu.sh cavia20    # فاز 1
```

لاگ: `/opt/margo/logs/gpu_cavia0.log` و `gpu_cavia20.log`.  
unit systemd جدا (`margo-cavia0`)؛ جاب pairseq تمام است، GPU خالی.

---

## تصمیم‌های قفل‌شده این سند

- خروجی = پلن π؛ `schedule` فیزیک
- few-shot = CAVIA روی z نه PEARL و نه PPO-on-θ
- pretrain BC جدا (انجام شده)
- pair search روش نیست
- `end_token=3`, obs 50, h 256
- freeze tags و 3500 دست‌نخورده
