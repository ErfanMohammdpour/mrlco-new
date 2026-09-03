# باگ‌های منطقی و یادگیری — `MARGO_BASELINE/mrlco-new`

فقط همین درخت کد. مقایسه با مبدأ `MRLCO/metarl-offloading` فقط برای تشخیص ارث در برابر باگ تازه‌ی فورک.

این فایل **باگ نیست**: تعداد task در هر iteration، اسم `max_path_length`، فاصله‌ی save، اینکه لاگ فقط ۵ task از ۱۰ تا را میانگین می‌گیرد، اختلاف LR/K بین train و eval، یا اینکه سیستم train می‌شود و crash نمی‌کند. آن‌ها تنظیم آزمایش‌اند، نه خطای منطق یا گرادیان.

معیار ورود به این لیست:

- شبیه‌ساز زمان/انرژی را خلاف وابستگی DAG یا خلاف مدل منبع حساب می‌کند (عدد reported دروغ است، یا پاداش سیاست را گمراه می‌کند).
- سیگنال یادگیری (reward / advantage / meta-gradient) چیزی غیر از آنچه حلقه ادعا می‌کند می‌سازد.
- شبکه ادعا می‌کند ساختار گراف را می‌بیند ولی نمی‌بیند.
- مسیر decode/token با فضای اکشن برخورد می‌کند.

---

## ۱. MEC منتظر اتمام predecessorِ V2V نمی‌ماند

**نوع:** باگ منطق شبیه‌ساز + تقلب در makespan (پاداش خوش‌بینانه)

**کجا:** `env/mec_offloaing_envs/offloading_env.py` — `get_scheduling_cost_step_by_step` و همان الگو در `greedy_solution`

Local و V2V بعد از اضافه‌شدن اکشن ۲ درست وصله شدند: شروع کار باید بعد از `max(FT_locally, FT_wr, FT_v2v_dl)` predecessor باشد. نتیجه‌ی V2V وقتی به منبع برمی‌گردد که downlink تمام شود (`FT_v2v_dl`).

خط MEC از مبدأ MRLCO کپی شده و **همان دو مقدار قدیمی** را می‌گیرد:

```python
ws_start_time = max(ws_avaliable_time,
                    max([max(FT_locally[j], FT_ws[j]) for j in task_graph.pre_task_sets[i]]))
```

`FT_v2v_dl[j]` نیست. `FT_wr[j]` هم نیست.

شروع compute روی ابر هم فقط `FT_cloud[j]` predecessor را می‌بیند. اگر predecessor روی V2V بوده، `FT_cloud[j]` صفر است.

**سناریو:** یال `j → i`، سیاست `action[j]=2` (V2V) و `action[i]=1` (MEC).

واقعیت فیزیکی: داده‌ی `j` باید به UE برگردد، بعد uplinkِ MEC برای `i` بتواند شروع شود (یا حداقل وابستگی داده برقرار باشد).

آنچه کد می‌کند: `FT_locally[j]=0` و `FT_ws[j]=0` چون `j` اصلاً MEC/local نبود. پس `ws_start_time` فقط صف uplink MEC است. MECِ `i` می‌تواند **قبل از تمام‌شدن V2Vِ `j`** آپلود را شروع کند. Makespan کوتاه‌تر از واقعیت. Latency reward بهتر از واقعیت. سیاست یاد می‌گیرد ترکیب V2V→MEC «ارزان» است در حالی که در مدل درست باید صبر کند.

همین حفره در greedy هست (`~764`). پس baseline حریص هم برای همین الگو خوش‌بین است. مقایسه‌ی نسبی policy در برابر greedy **داخل همین شبیه‌ساز** سازگار است، ولی هر دو عدد latency برای الگوهای V2V→MEC **غلط**اند. اگر کسی این اعداد را makespan واقعی گزارش کند، تقلب unintentional است.

**ارث:** در MRLCO باینری، predecessor یا local بود (`FT_locally`) یا MEC (`FT_ws` برای کانال ارسال، `FT_cloud` برای ابر). حفره بعد از اکشن V2V ساخته شد، چون شاخه‌ی MEC به‌روز نشد.

**اثر روی یادگیری:** هر trajectory که V2V را قبل از MEC روی یال DAG بگذارد، advantage ساختگی مثبت می‌گیرد. توزیع اکشن (~۷٪ V2V در نتایج) ممکن است همین سوگیری را پنهان/تقویت کند.

**درست:** برای شروع uplink MEC همان `max(FT_locally[j], FT_wr[j], FT_v2v_dl[j])` که local/V2V دارند. برای شروع compute ابر، اگر وابستگی باید روی UE باشد، باید finish واقعی predecessor بیاید نه فقط `FT_cloud`.

---

## ۲. `end_token=2` الان خود اکشن V2V است

**نوع:** باگ فضای اکشن / decode (در مسیر فعلی train پنهان، روی greedy-decode فعال)

**کجا:** `policies/meta_seq2seq_policy.py` — `HParams.end_token=2`، `vocab_size=3`

مبدأ MRLCO: `vocab_size=2`، اکشن `{0 Local, 1 MEC}`، `end_token=2` **خارج** از مجموعه. Sentinel استاندارد seq2seq.

این فورک اکشن `2 = V2V` اضافه کرد و `end_token` را عوض نکرد.

`tf.contrib.seq2seq.GreedyEmbeddingHelper` وقتی `sample_id == end_token` می‌شود، decode را تمام می‌کند. اولین V2V در پلن = قطع دنباله. بقیه‌ی تسک‌ها یا تکرار آخرین توکن می‌مانند یا صفر. پلن ناقص.

مسیر فعلی `get_actions` از `FixedSequenceLearningSampleEmbedingHelper` استفاده می‌کند و finish را با `sequence_length` می‌بندد، نه با `end_token`. Train و eval فعلی **فعلاً** ۲۰ اکشن کامل می‌گیرند. گراف `greedy_decoder_prediction` داخل شبکه ساخته می‌شود؛ اگر کسی greedy decode را برای گزارش/infer روشن کند، شکسته است.

**اثر یادگیری مستقیم الان:** صفر، تا وقتی sample helper باشد.

**اثر واقعی:** تله. ادعای «greedy policy» در این گراف TF غلط است. هر ارزیابی با `GreedyEmbeddingHelper` V2V را با EOS قاطی می‌کند.

**درست:** `end_token` را مقداری خارج از `{0,1,2}` بگذار (مثلاً `3`) یا اصلاً به helper نده؛ طول ثابت ۲۰ کافی است.

---

## ۳. Meta-gradient Reptile حدود ۱۰۰ برابر کوچک‌تر از فرمول مبدأ — به‌علاوه Adam به‌جای SGD

**نوع:** باگ حلقه‌ی یادگیری (وزن‌ها حرکت می‌کنند، مقیاس/جهت meta غلط است)

**کجا:** `meta_trainer.py` `inner_batch_size=10`؛ `meta_algos/MRLCO.py` `UpdatePPOTargetPerTask` + `UpdateMetaPolicy`

داده‌ی inner تقریباً همان مبدأ است: حدود ۱۰۰۰ trajectory برای هر meta-task (۱۰ rollout × ۱۰۰ گراف).

تفاوت در خردکردن batch:

| | مبدأ MRLCO | این کد |
|---|---|---|
| `inner_batch_size` | `1000` | `10` |
| `batch_number` | `≈ 1` | `≈ 100` |
| `self.update_numbers` | `1` | `100` |

فرمول outer:

```python
grads = (core_var - meta_var) / inner_lr / num_inner_grad_steps / meta_batch_size / update_numbers
```

Reptile کلاسیک: جهت meta ≈ `(θ' - θ)` یعنی جابه‌جایی بعد از adaptation. تقسیم بر `inner_lr` وقتی درست است که inner **SGD** باشد (`θ' = θ - α ∇`). اینجا inner **Adam** است. جابه‌جایی `θ' - θ` دیگر `α ∇` نیست؛ moment و adaptive scale داخل‌اند. تقسیم دوباره بر `inner_lr` مقیاس را خراب می‌کند.

بعد همان بردار بر `update_numbers` تقسیم می‌شود. با `batch_size=10` این عامل `100` است. نسبت به مبدأ، همان داده، گام outer حدود **۱۰۰× کوچک‌تر** است.

لایه‌ی سوم: `UpdateMetaPolicy` روی ۱۰ task **پشت‌سرهم** `apply_gradients` می‌زند. بعد از task صفر، `core` جابه‌جا شده. گرادیان task یک می‌شود `(core_جدید - θ'_1)` نه `(core_اول - θ'_1)`. میانگین Reptile نیست؛ ۱۰ آپدیت Adam متوالی روی جهت‌های مختلف است، با moment مشترک outer Adam.

نتیجه:

- Inner زیاد به‌روز می‌شود (۱۰۰ minibatch Adam روی PPO، `old_logits` فریز از زمان sample — از نظر PPO یک epoch است، ولی نسبت به مبدأ که یک آپدیت روی کل batch بود، adaptation خیلی تهاجمی‌تر است).
- Outer همان جابه‌جایی را بر ۱۰۰ تقسیم می‌کند و با Adam دوباره نرم می‌کند.

سیستم «یاد می‌گیرد» به این معنی که loss/latency جابه‌جا می‌شود. آنچه Reptile باید باشد (میانگین جابه‌جایی inner روی taskها، یک گام meta) نیست. تعمیم بین توزیع گراف‌ها ضعیف‌تر و پرنویزتر از ادعای الگوریتم است.

**درست (یکی از این‌ها، نه همه با هم بدون فکر):**

- `inner_batch_size` را نزدیک مبدأ (`1000`) برگردان تا `update_numbers≈1`؛ یا
- `update_numbers` را از مخرج بردار؛ و
- inner را SGD کن اگر می‌خواهی تقسیم بر `α` معنی داشته باشد؛ و
- گرادیان ۱۰ task را **میانگین بگیر، بعد یک‌بار** به core اعمال کن.

---

## ۴. Graph2Seq یال DAG را نمی‌بیند — adjacency یک clique است

**نوع:** باگ منطق encoder / ادعای معماری در برابر آنچه واقعاً یاد گرفته می‌شود

**کجا:** `policies/graph2seq_encoder.py` — `sequence_to_graph`

کامنت می‌گوید گراف می‌سازد. پیاده:

```python
single_seq_adj = tf.tile(tf.expand_dims(seq_indices, 0), [seq_len, 1])
```

هر نود به همه‌ی ۲۰ موقعیت وصل است، از جمله خودش. `MeanAggregator` میانگین همسایه‌ها را می‌گیرد = میانگین **همه** نودهای دنباله. دو لایه‌ی GCN روی گراف کامل، معادل دو بار مخلوط‌کردن سراسری است، نه پیام روی predecessor/successor.

یال‌های `.gv` هیچ‌وقت به `fw_adj_info` نمی‌روند.

مبدأ MRLCO حداقل LSTM را روی ترتیب HEFT می‌راند (ترتیب تقریباً topo). این فورک LSTM را با «GCN» عوض کرد ولی ساختار گراف را نداد. Decoder هنوز Luong attention روی همان ۲۰ بردار مخلوط‌شده دارد.

`dropout=0.1` روی adapter ست می‌شود ولی به `MeanAggregator(...)` پاس نمی‌شود (پیش‌فرض `dropout=0`). حتی همان regularization هم خاموش است. این جزئی است نسبت به clique؛ ذکر می‌شود چون نشان می‌دهد مسیر Graph2Seq نصفه وصل شده.

**اثر یادگیری:** سیاست می‌تواند از فیچرهای ۲۰بعدی و ترتیب HEFT چیزی یاد بگیرد. نمی‌تواند «این نود به آن نود یال دارد» را از طریق GCN بفهمد. هر نتیجه‌ای که به Graph2Seq / message-passing روی DAG نسبت داده شود، از این کد پشتیبانی نمی‌شود.

**درست:** adjacency = pred ∪ succ بعد از map کردن task id → موقعیت در دنباله‌ی HEFT. self-loop جدا و کنترل‌شده. Dropout را واقعاً به aggregator بده.

---

## ۵. فیچر pred/succ با id خام است، نه موقعیت در دنباله‌ی HEFT

**نوع:** باگ ساخت observation (سیگنال ساختار گراف در ورودی خراب/گمراه‌کننده)

**کجا:** `env/mec_offloaing_envs/offloading_task_graph.py` — `encode_point_sequence_with_cost` سپس `encode_point_sequence_with_ranking_and_cost`

بردار هر تسک:

`[id, T_loc, T_up, T_mec, T_dl, T_vup, T_vexe, T_vdl] + 6 pred id + 6 succ id`

بعد ردیف‌ها با `prioritize_sequence` (HEFT) جابه‌جا می‌شوند. خود اعداد pred/succ **همان id اصلی ۰..۱۹** می‌مانند.

نود شماره‌ی `k` در tensor ورودی = تسک در رتبه‌ی HEFT `k`، نه لزوماً تسک با `id=k`. عدد `7` در اسلات predecessor یعنی «تسک با شناسه‌ی ۷»، نه «همسایه‌ی موقعیت ۷ در sequence». GCN هم که همسایه را از این اعداد نمی‌خواند (باگ ۴). پس این ۱۲ عدد فقط اسکالر اضافی‌اند؛ مدل باید یاد بگیرد آن‌ها شناسه‌اند نه مختصات. بعد از shuffle HEFT، هم‌ترازی id با موقعیت شکسته است.

علاوه: pred فقط از `range(0, i)` جمع می‌شود (فقط id کوچک‌تر). اگر در `.gv` یال از id بزرگ‌تر به کوچک‌تر باشد، **فیچر آن را نمی‌بیند** در حالی که `pre_task_sets` در شبیه‌ساز آن را می‌بیند. سیاست و env روی یک DAG واحد توافق کامل ندارند.

حداکثر ۶ همسایه؛ بقیه `-1.0`. اگر درجه‌ی گراف > ۶ باشد، یال‌ها در observation حذف می‌شوند.

**اثر:** حتی اگر کسی فردا adj را درست کند، تا id→position map نشود، فیچر همسایه با نود GCN جور درنمی‌آید.

---

## ۶. کران انرژی برای نرمال‌سازی پاداش غلط است — و حساب انرژی MEC در برابر V2V ناسازگار

**نوع:** باگ سیگنال reward (و سوگیری به نفع MEC)

**کجا:** `_compute_energy_bounds` و `compute_*_energy` در `offloading_env.py`

### ۶آ. `min_energy` = تمام‌MEC

```python
min_energy = sum(compute_transmission_energy(T_up, T_dl) for task)
max_energy = sum(compute_local_energy(...) for task)
```

V2V توان ارسال کمتری دارد (`ptx_v2v=0.06` در برابر `ptx=0.1`). پلن با V2V زیاد می‌تواند **انرژی کمتر از `min_energy`** داشته باشد. آن وقت

`score = -(E - min_E) / (max_E - min_E)`

مثبت می‌شود. Latency score معمولاً منفی است. ترکیب `0.5 lat + 0.5 ene` برای V2V می‌تواند از سمت انرژی **پاداش مثبت** بگیرد چون از کران «حداقل» رد شده — نه چون لزوماً بهینه‌ی انرژی است، چون کران دروغ است.

اگر انرژی از `max_energy` هم بالاتر برود (V2V = رادیو + محاسبه‌ی helper)، score < −۱. مقیاس reward از بازه‌ی طراحی خارج است. PPO/GAE با advantage نرمال‌شده کمی مقاوم است، ولی نسبت latency/energy داخل همان trajectory خراب می‌ماند.

### ۶ب. چه چیزی در انرژی شمرده می‌شود

- Local: فقط محاسبه‌ی UE.
- MEC: فقط رادیوی UE (`ptx`/`prx`). محاسبه‌ی سرور **صفر**.
- V2V: رادیو + **محاسبه‌ی خودرو کمکی** (`rho_v2v * t_exec`).

اگر هدف انرژی UE است: محاسبه‌ی helper نباید روی بودجه‌ی UE بیاید.
اگر هدف انرژی سیستم است: محاسبه‌ی MEC باید بیاید.

در هر دو تفسیر، MEC نسبت به V2V ارزان‌تر از واقعیت مدل می‌شود. سیاست برای انرژی به MEC کشیده می‌شود. با وزن `λ_ene=0.5` این سوگیری داخل گرادیان است، نه فقط لاگ.

پاداش انرژی یک اسکالر نرمال روی **کل گراف** است که بعد با نسبت انرژی هر گام پخش می‌شود. این خودش heuristic است؛ مشکل اصلی کران و حساب ناسازگار است نه پخش.

Greedy اکشن را فقط با finish time انتخاب می‌کند؛ انرژی را بعداً جمع می‌زند. پس greedy رقیب joint latency–energy نیست. مقایسه‌ی انرژی policy با انرژی greedy، بهینه‌سازی دو تابع هدف مختلف است. این تقلب متریک است اگر در جدول به‌عنوان baseline انرژی فروخته شود.

---

## ۷. نرمال latency روی دلتای makespan است، کران‌ها هزینه‌ی یک تسک تنها

**نوع:** باگ مقیاس reward

**کجا:** `calculate_max_min_runningcost` + `score_func` روی `cost` که دلتای makespan است

`max_time` = max(یک hop ارسال/اجرا روی **بزرگ‌ترین** data size).
`min_time` = اجرای MEC روی **کوچک‌ترین** data size.

`cost[t]` = افزایش makespan بعد از زمان‌بندی تسک `t` (ممکن است صفر باشد اگر کار پشت critical path پنهان شود، یا بزرگ باشد اگر صف half-duplex/CPU طول بکشد).

`score_func` همان `max_time`/`min_time` را روی هر دلتا اعمال می‌کند.

نتیجه:

- دلتای صف V2V می‌تواند از `max_time` بزند → score ≪ −۱.
- دلتای صفر → score مثبت (`min_time/(max-min)`).

انگیزه‌ی «کار را موازی پنهان کن» برای makespan بد نیست. مشکل این است که مقیاس بین گراف‌ها و بین latency و energy یکی نیست، و بعد از V2V (صف کانال مشترک) دم توزیع دلتا پهن‌تر از مدل باینری MRLCO است. Inner PPO روی همین اعداد GAE می‌سازد.

ارث از MRLCO است؛ با اکشن سوم و half-duplex اثرش بزرگ‌تر شده. یادگیری را خراب می‌کند چون دو جزء reward در یک بازه نیستند.

---

## ۸. Reptile ادعا می‌شود؛ حلقه Adam متوالی روی task است

**نوع:** باگ الگوریتم meta (مرتبط با ۳، جداگانه چون حتی با `update_numbers=1` می‌ماند)

**کجا:** `MRLCO.UpdateMetaPolicy`

شبه کد واقعی:

1. برای task `i`: `θ'_i` را از شبکه‌ی inner بخوان.
2. `θ` فعلی core را بخوان (بعد از آپدیت taskهای قبلی).
3. `g = (θ - θ'_i) / (α K M N_upd)` را به Adam outer بده.
4. برو task بعد.

Reptile کاغذ: `θ ← θ + ε · mean_i(θ'_i − θ)` با **همان θ** برای همه‌ی i.

اینجا θ وسط حلقه عوض می‌شود. Outer Adam moment را روی این ۱۰ جهت متوالی جمع می‌کند. `async_parameters` بعد از هر ۱۰ تا core را روی همه‌ی inner کپی می‌کند — این قسمت درست است.

حتی اگر باگ ۳ (عامل ۱۰۰) را درست کنی، تا میانگین قبل از یک `apply_gradients` ساخته نشود، meta-update همان Reptile نیست.

---

## جمع‌بندی اثر روی «چه چیزی واقعاً یاد گرفته می‌شود»

سیاست یک seq2seq است: مشاهده‌ی ثابت ۲۰×۲۰ (ترتیب HEFT + زمان‌های تکی + id همسایه)، decode خودبرگشت ۲۰ اکشن `{0,1,2}`، یک `env.step` کل پلن را شبیه‌سازی می‌کند.

آنچه باید یاد بگیرد: زمان‌بندی DAG با منابع اشتراکی و انرژی.

آنچه سیگنال می‌دهد:

1. شبیه‌ساز برای V2V→MEC زمان را کمتر از واقعیت می‌دهد → گرادیان آن الگو را دوست دارد.
2. Encoder گراف را نمی‌بیند → ظرفیت Graph2Seq برای DAG استفاده نمی‌شود؛ یادگیری از فیچر تخت + ترتیب HEFT است.
3. Meta-step خیلی کوچک و از جنس Adam-Reptile خراب است → adaptation بین taskها ضعیف‌تر از ادعای MAML/Reptile.
4. Reward انرژی MEC را ارزان و کران min را غلط می‌سازد → تعادل latency/energy داخل PPO کج است.
5. `end_token` روی greedy decode V2V را EOS می‌کند.

وزن PPO inner به‌روز می‌شود. Env اجرا می‌شود. نتایج latency می‌توانند از greedy و از MRLCO-style بهتر شوند **داخل همین شبیه‌ساز**. این نتایج فیزیک درست V2V→MEC را تضمین نمی‌کنند و Graph2Seq واقعی را ثابت نمی‌کنند.

---

## ترتیب اصلاح اگر قرار است کد عوض شود

1. MEC (و greedy MEC): wait روی `FT_v2v_dl` / finish واقعی predecessor — بدون این، هر train جدید هنوز روی makespan دروغ است.
2. `end_token` خارج از `{0,1,2}`.
3. Meta: یک بار میانگین `(θ - θ'_i)` روی taskها؛ `update_numbers` را از مخرج حذف یا `inner_batch_size` را به مقیاس یک آپدیت برگردان؛ inner SGD اگر فرمول `/ inner_lr` می‌ماند.
4. Adj واقعی DAG (id → index HEFT) + فیچر pred/succ با همان index.
5. کران انرژی: min واقعی = min(all-local, all-MEC, all-V2V) یا کران تحلیلی که V2V را شامل شود؛ حساب انرژی MEC/V2V را روی یک تعریف واحد (فقط UE یا کل سیستم) بیاور.

---

## عمداً این‌جا نیامده

| مورد | چرا باگ یادگیری/منطق این لیست نیست |
|---|---|
| `M=10` task در هر iter | تنظیم meta-batch |
| `max_path_length=20000` | بودجه‌ی sample؛ loop تمام می‌شود |
| لاگ میانگین ۵ task از ۱۰ | فقط گزارش |
| `K_train=1` در برابر `K_eval=3`، LR متفاوت، ۳۵۰۰ در برابر ۱۰۱ | پروتکل آزمایش، نه باگ گرادیان |
| Loss جمع‌شونده بدون reset وقتی `K=1` | وزن درست؛ با `K=3` فقط عدد لاگ eval دروغ است |
| `get_running_cost` یک‌بار صدا | در این درخت درست است |
| ایندکس greedy با `total_task=20` | لیست per-distribution است |
| Half-duplex V2V و فرمول انرژی زنده | پیاده شده‌اند |
| Encoder 256-D بعد concat در برابر decoder 128 | projection هست |
| PPO بدون entropy bonus | انتخاب الگوریتم |
| `ValueFunctionBaseline` = `path["values"]` | baseline همان vf شبکه در زمان sample |
| ۱۰× deepcopy دیتاست | هزینه حافظه |

اگر موردی از جدول بالا وزن را عوض کند یا makespan را دروغ کند، باید به لیست اصلی برگردد. الان نمی‌کند.
