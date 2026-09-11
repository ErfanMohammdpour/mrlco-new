# جریان لحظه‌به‌لحظه — `MARGO-METHOD-v0.2-cavia`

قرارداد شکل: `B` = اندازهٔ بچ گراف. یک گراف: `B=1`. Support یک dist: معمولاً `B=20`.  
همهٔ اندیس‌ها 0-based مگر فایل `.gv`.

---

## مسیر ۰ — از فایل تا obs (هر گراف، موجود)

### تیک ۰.۱ پارس

```text
ورودی:  offload_random20_{d}/random.20.{i}.gv
خروجی: OffloadingTaskGraph
        tasks[tid]: compute_workload_bytes, task_output_bytes
        edges: (src, dst, edge_output_bytes)
```

### تیک ۰.۲ کانونیک

```text
ورودی:  OffloadingTaskGraph
خروجی: CanonicalDAG
        CanonicalTask(tid, compute_workload_bytes, task_output_bytes, external_input_bytes)
        CanonicalEdge(src, dst, edge_output_bytes)
external_input_bytes = processing_data_size اگر root وگرنه 0
```

تابع: `to_canonical_dag` در `scheduler/adapter.py`.

### تیک ۰.۳ HEFT

```text
ورودی:  CanonicalDAG + ResourceConfig
خروجی: prioritize_sequence  ∈ {0..19}^{20}   جایگشت task_id
```

`prioritize_sequence[k] = tid` یعنی توکن k مال آن تسک است.

### تیک ۰.۴ pack

```text
ورودی:  CanonicalDAG, prioritize_sequence, FeatureStats(meta_train)
خروجی: obs ∈ R^{20×50} float32
```

`encode_task_graph` / `encoder_obs.py`:

```text
برای k در 0..19:
  tid = order[k]
  feat[11] = zscore ویژگی‌ها (جز is_root, is_sink)
  fw[19]   = decoder-index successorها، پد -1
  bw[19]   = decoder-index predecessorها، پد -1
  mask     = 1
obs[k] = concat(feat, fw, bw, mask)
```

بچ محیط آموزش: اغلب `[100, 20, 50]` داخل `encoder_batchs` سپس زیرنمونهٔ ۲۰.

---

## مسیر ۱ — یک forward سیاست با `z` (هدف روش)

فرض `B` گراف، `obs [B,20,50]`, `z [B,32]` یا `z [32]` پخش‌شده روی بچ (یک z برای کل support یک خانواده).

### تیک ۱.۱ unpack + embed

```text
in:  obs [B,20,50]
out: feature_slice [B,20,11]
     fw_idx [B,20,19] int32
     bw_idx [B,20,19] int32
     node_mask [B,20]
     embed [B,20,128] = Dense(feature_slice)
```

### تیک ۱.۲ GNN

```text
هر جهت 2× MeanAggregator concat
fw_hidden [B,20,256]
bw_hidden [B,20,256]
h = ReLU(fw+bw) [B,20,256]
```

### تیک ۱.۳ triple readout

```text
attn_pool, mean_pool, max_pool : هر کدام [B,256]
u = concat → [B,768]
g = tanh(Dense) → [B,256]
s = Dense → [B,128]
```

### تیک ۱.۴ FiLM (ساخت)

```text
in:  s [B,128], z [B,32]
γ = Dense_γ(z) [B,128]     # init 0 + bias 1 از طریق γ = 1 + Linear(z)
β = Dense_β(z) [B,128]     # init 0
s̃ = γ ⊙ s + β            [B,128]
encoder_state = (LSTMStateTuple(c=s̃,h=s̃), LSTMStateTuple(c=s̃,h=s̃))
```

اگر `z=0` و init درست: `s̃ = s` ≡ ckpt BC.

### تیک ۱.۵ decode گام k (موجود + state فیلم)

`k = 0`:

```text
token_in = start_token = 0
e = Embed(0) [B,128]
```

هر k:

```text
LSTM layer0, layer1 روی e با state
Luong: score(query, h[:,j,:]) j=0..19 → α [B,20]
ctx = Σ_j α_j h_j [B,128]
logits_k [B,3]
```

نمونه:

```text
a_k ~ Categorical(logits_k)     # train / CAVIA inner
a_k = argmax(logits_k)          # گزارش query
```

`k=19` تمام. اگر greedy روی `end_token=3` ایستاد: `align_greedy_pred` پد MEC=1 تا طول ۲۰.

```text
out: actions [B,20] int32 ∈ {0,1,2}
     logits  [B,20,3]     (اگر sample/train)
```

### تیک ۱.۶ zip

برای گراف b:

```text
plan_b = list(zip(prioritize_sequence_b, actions[b]))
# [(tid, a), ...] طول 20
```

---

## مسیر ۲ — یک فراخوانی محیط (موجود)

### تیک ۲.۱ schedule

```text
in:  CanonicalDAG, decoder_order, actions[20], ResourceConfig
out: ScheduleResult
       makespan_seconds: float
       energy.total_mobile_joules: float
       task records, transfer records
```

داخل، برای هر `tid` در ترتیب توپو:

1. `loc = Location.from_action(a[rank[tid]])`
2. برای هر پدر `p`: `hops = route(residency[p], loc)`؛ رزرو کانال به ترتیب hops؛ حجم = `edge_output_bytes`
3. اگر root: hops از UE با `external_input_bytes`
4. رزرو CPU محل به مدت `compute_workload_bytes / rate(loc)`
5. `residency[tid] = loc`
6. بعد از همه: هر sink → UE با `task_output_bytes`

### تیک ۲.۲ تلسکوپ (اگر CAVIA از r_t استفاده کند)

```text
in:  plan کامل
out: rewards [20]
     makespans [21]   # L_0..L_N
     energies [21]
N بار schedule روی P_t  (پیشوند + fill UE)
```

هزینه: تا ۲۰ `schedule` اضافه per گراف. برای گیت اول CAVIA مجاز است فقط T نهایی (۱ schedule).

### تیک ۲.۳ `env.step` لایهٔ قدیم

```text
in:  actions [20, 20]   # بعضی اجراکننده‌ها (n_graph, n_token)
out: r [20], done=True, info(T,E)
مشاهدهٔ منبع بعد از توکن: وجود ندارد
```

---

## مسیر ۳ — Pretrain A (موجود)

یک ایپاک روی ۱۵۰۰ گراف meta_train:

```text
بچ 32:
  obs [32,20,50]
  targets [32,20] int  = لیبل 2-opt
  decoder_inputs = concat(0, targets[:, :-1])
  decoder_full_length = [20]*32

  forward model=train
  loss = mean CE(logits, targets)
  Adam θ_enc, θ_dec   lr=5e-4
```

Eval A (نه روش کامل):

```text
greedy z=0 (فیلم خاموش/ z صفر)
یک schedule per گراف
لاگ T_mean, mix, token_acc
```

عدد قفل diagnostic `bc_2opt` (نه مقاله): train T **435.6** / val **581.8** / test **560.8**.

---

## مسیر ۴ — یک گام inner CAVIA (ساخت)

خانوادهٔ `i`، support ۲۰ گراف. `θ` قفل.

```text
z: variable [32]  (نه per-graph)

برای k در 1..20:
  obs_S [20,20,50]          # همان ۲۰ گراف
  h,s = encoder_frozen(obs_S)
  s̃ = FiLM(s, broadcast(z))
  a = sample_decoder(h, s̃)  # [20,20]
  برای j=1..20:
      T_j, E_j = schedule(G_j, order_j, a[j])
  L = mean_j T_j            # گیت 1؛ بعداً J 0.5/0.5
  g = ∇_z L                 # TF: z باید در گراف باشد؛ schedule numpy است
```

**قطع گرادیان فیزیک:** `schedule` خارج TF است. پس `∇_z L` از T نمی‌آید مگر:

1. **امتیاز سیاست (پیشنهاد فاز ۱):**  
   `L_z = mean_j [ stop_grad(Â_j) * Σ_k log π(a_{jk}|h,z) ]`  
   با `Â_j = T_j - baseline` (میانگین بچ یا `vf` فریز).  
   این REINFORCE روی **z** است نه روی θ. θ `stop_gradient`.
2. یا تلسکوپ `Σ r_t logπ` همان شکل، فقط z در `logπ` است.

بدون (1) CAVIA به Adam عددی finite-diff روی z نیاز دارد (۲۰×۲ dim گران). پس فاز ۱ = REINFORCE-on-z.

```text
z ← Adam(z, ∇_z L_z, α=5e-4, β1=0.9, β2=0.999)
clip ||g|| اگر > 0.5 (همان gradient_clip_norm)
```

بعد از ۲۰ گام: `z*` ثابت. Query:

```text
obs_Q [n_q,20,50]
a = greedy π(h, FiLM(s,z*))
T,E = schedule یک‌بار per query graph
```

---

## مسیر ۵ — یک outer iteration متا (ساخت)

```text
نمونه 10 dist از 15 meta_train بدون جای‌گزینی
sync: θ از core (در فاز B اول θ اصلاً عوض نمی‌شود)

برای هر dist i موازی منطقی / سریال مثل الان:
  set_task(support_i)
  z_i ← 0
  20 گام CAVIA → z_i*
  set_task(query_slice_i)     # ۲۰ گراف تصادفی از ۸۰، یا همه در val
  L_i = mean query T یا J تحت z_i*

φ (Dense_γ, Dense_β، و اگر φ0 برای init z):
  g_φ = mean_i ∇_φ L_i
  φ ← Adam_outer(g_φ, 5e-4)   # state پایدار بین itr
```

اگر `θ` فریز کامل باشد، `∇_φ` از FiLM و از `logπ` نسبت به s̃ می‌آید. Encoder backward ندارد.

تعداد `schedule` تقریبی هر outer (۱۰ dist، sample یک‌بار per گراف per inner step):

```text
10 dist × 20 inner × 20 support = 4000 schedule
+ query 10 × ~20 = 200
≈ 4200   (کمتر از حلقهٔ فریز که ~20000 مسیر با تلسکوپ 20× بود)
```

---

## مسیر ۶ — meta-test رسمی (ساخت؛ عدد هنوز نیست)

برای dist در meta-test:

```text
k=0:
  z=0
  greedy query 80 → T0, E0, mix0

k=20:
  z=0
  for 20 steps: sample support 20, REINFORCE-on-z, Adam
  greedy query 80 → T20, E20, mix20
```

گیت علمی (نه مقاله): val، از ckpt `bc_2opt`:

```text
T0 باید ≈ 581.8   (رگرسیون)
T20 < T0 با حاشیهٔ پایدار؛ mix از ~0.20/0.75/0.05 به Local~0.30 نپرد
اگر T20 ≥ T0 مثل k3 PPO (657): CAVIA نپاسد
```

---

## مسیر ۷ — diagnostic جستجو (موجود، خارج روش)

فقط سقف:

```text
h فریز → pair 9-way CE → top-k → 9×schedule قبول اگر T↓
```

قفل `pair_seq` Kish:

| روش | val T | test T |
|---|---:|---:|
| BC greedy | 581.8 | 560.8 |
| scan k20 | 457.8 | 443.5 |
| bestimp k20 | 447.9 | 434.5 |
| bestimp k50 | 438.9 | 426.8 |
| 2-opt | 423.6 | 411.6 |

این اعداد سقف heuristicاند. در مسیر ۶ نباید ظاهر شوند مگر ستون baseline جدا.

---

## چک‌لیست شکل (فاز ۰)

| تانسور | شکل | dtype |
|---|---|---|
| obs | `[B,20,50]` | float32 |
| h | `[B,20,256]` | float32 |
| s | `[B,128]` | float32 |
| z | `[32]` یا `[B,32]` | float32 |
| γ,β | `[B,128]` | float32 |
| logits | `[B,20,3]` | float32 |
| actions | `[B,20]` | int32 |
| T,E | اسکالر per گراف | float64 |
