# پروژه B — یادگیری فرا با Graph-to-Sequence برای تطبیق سریع سیاست آفلودینگ DAG

**دانشگاه:** KNTU  
**درس:** [نام درس را وارد کنید]  
**دانشجو:** [نام و شماره دانشجویی]  
**استاد:** [نام استاد]  
**تاریخ:** [تاریخ ارائه]

---

## ۱. خلاصه پروژه (Abstract)

این پروژه یک **چارچوب Meta-Reinforcement Learning** برای **تطبیق سریع** سیاست آفلودینگ task graph (DAG) ارائه می‌دهد. وقتی **توزیع workload** عوض می‌شود، RL معمولی نیاز به retrain سنگین دارد. این کار با **Graph2Seq encoder**، **triple readout**، و **Reptile + PPO** یک initialization مشترک یاد می‌گیرد که با چند گام fine-tune روی توزیع unseen بهبود می‌یابد.

تمرکز اصلی: **معماری یادگیری**، **پروتکل meta-train/test**، **مقایسه با baseline MRLCO-style**، و **ablation** اجزای encoder.

**خروجی‌های مورد انتظار:**
- pipeline آموزش meta-RL (`meta_trainer.py`) + ارزیابی (`meta_evaluator.py`)
- نتایج convergence و stability روی Task 1/2
- ablation w/o triple readout و w/o GNN
- گزارش ۱۰–۱۲ صفحه + پرزنت ۱۵–۱۸ دقیقه

---

## ۲. تعریف مسئله (مستقل)

### ۲.۱ انگیزه
در Edge/MEC، الگوی task (اندازه داده، چگالی یال، نسبت compute/comm) بین سناریوها فرق می‌کند. Policy یکبار train شده روی توزیع A روی توزیع B ضعیف عمل می‌کند.

### ۲.۲ سؤال تحقیق این پروژه
> چگونه می‌توان با **نمایش گرافی DAG** و **meta-RL** سیاست آفلودینگ را طوری آموزش داد که روی **توزیع‌های unseen** با **بودجه adaptation کم** (مثلاً ۱۰۰ iteration PPO) بهتر از baseline sequence-based converge کند؟

### ۲.۳ محدوده
- محیط شبیه‌سازی: platform استاندارد (V2V/energy) — **جزئیات radio در پروژه A**
- تصمیم: sequence of 20 ternary actions {Local, MEC, V2V}
- Meta-train: distributions 1–15 | Meta-test: 16–19

---

## ۳. نوآوری‌ها و سهم علمی (۴ مورد — متوازن با پروژه A)

1. **Graph2Seq encoder** با message passing روی neighborhood DAG (pred ∪ succ)
2. **Triple readout** (attention + mean + max) برای graph-level state
3. **Seq2Seq decoder** با Luong attention → 20 decision ternary
4. **Meta-training** PPO inner-loop + Reptile outer-loop (`MRLCO`) با reward latency+energy

> **نکته:** شما env را «black-box simulator with realistic costs» معرفی می‌کنید — معادلات V2V را deep dive نمی‌کنید.

---

## ۴. معماری (هسته فنی)

### ۴.۱ Pipeline
```
DAG features [B,20,20]
  → Graph2Seq encoder (2× MeanAggregator, H=128)
  → Triple readout → dense → LSTM init state
  → LSTM decoder + Luong attention (2 layers)
  → Categorical policy (3 actions) + value head
```

### ۴.۲ MDP (conceptual)
| جزء | تعریف |
|-----|--------|
| State | encoded graph + step index |
| Action | \(a_t \in \{0,1,2\}\) per task in HEFT order |
| Episode | one full 20-action schedule → env simulates once |
| Reward | \(\lambda_{lat} r^{lat} + \lambda_{ene} r^{ene}\) |

### ۴.۳ Meta-learning (Reptile)
\[
g_{meta} = \frac{1}{M \cdot K} \sum_{i=1}^{M} (\theta_{core} - \theta_i^{(K)})
\]
- \(M=10\) tasks per meta-batch
- \(K=1\) inner PPO step (training)
- 3,500 meta-iterations

### ۴.۴ ماژول‌های کد

| فایل | نقش |
|------|-----|
| `policies/graph2seq_encoder.py` | Encoder + triple readout |
| `policies/graph2seq_modules/` | GCN layers, aggregators |
| `policies/meta_seq2seq_policy.py` | Seq2Seq + MetaSeq2SeqPolicy |
| `meta_algos/MRLCO.py` | Reptile outer + PPO inner |
| `meta_algos/ppo_offloading.py` | PPO for evaluation fine-tune |
| `meta_trainer.py` | Meta-training entry |
| `meta_evaluator.py` | Adaptation on held-out dist |
| `samplers/seq2seq_meta_sampler*.py` | Rollout + GAE |

---

## ۵. Hyperparameters (از `meta_trainer.py`)

| Parameter | Value |
|-----------|-------|
| obs_dim | 20 |
| Hidden units | 128 |
| Actions | 3 |
| PPO ε | 0.2 |
| γ | 0.99 |
| GAE λ | 0.95 |
| λ_lat / λ_ene | 0.5 / 0.5 |
| Meta-batch M | 10 |
| Inner steps K | 1 |
| Meta iterations | 3,500 |
| Inner/outer LR | 5e-4 |

---

## ۶. آزمایش‌ها (۵ آزمایش — هم‌تراز با پروژه A)

| # | آزمایش | وضعیت | خروجی |
|---|--------|--------|--------|
| E1 | **MARGO vs MRLCO-style** (T1 dist10, T2 dist12) | **موجود** | latency, energy, composite |
| E2 | **Convergence curves** (100 iter adapt) | **موجود** | `fig_latency/energy_convergence_task*.pdf` |
| E3 | **Stability** (std last-20) | **موجود** | جدول stability |
| E4 | **Ablation: w/o triple readout** | **باید اضافه شود** | جدول vs full |
| E5 | **Ablation: LSTM encoder (w/o GNN)** | **باید اضافه شود** | جدول vs full |
| E6* | **Adaptation budget** (0, 25, 50, 100 steps) | **باید اضافه شود** | curve |

\* E6 اختیاری قوی — اگر وقت کم است E4+E5 کافی است.

### نتایج E1–E3 (موجود)

از `paper/analysis_tables/main_performance_stability.csv`:

| Task | Method | Last-20 Lat. | Lat. Std. | Comp. Impr. |
|------|--------|-------------|-----------|-------------|
| T1 | **MARGO (ours)** | **575.40** | **0.40** | 23.03% |
| T1 | MRLCO-style | 617.16 | 1.83 | 25.18% |
| T2 | **MARGO (ours)** | **605.16** | **0.94** | 20.68% |
| T2 | MRLCO-style | 652.89 | 29.13 | 23.37% |

**نکته دفاع:** MRLCO-style گاهی energy پایین‌تر ولی **latency بدتر و ناپایدارتر** (std بالا در T2).

### Protocol
- Meta-train: distributions **1–15** (1,500 graphs)
- Meta-test: **16–19** (400 graphs)
- Evaluation: 100 adaptation iterations, compare last-20 average
- Baselines: Greedy (fixed ref), MRLCO-style (seq encoder, binary Local/MEC, latency-focused)

---

## ۷. کارهای اضافه (الزامی — ~۱ هفته)

### ۷.۱ Ablation triple readout (~۲ روز)
- در `graph2seq_encoder.py`: flag `use_triple_readout=False` → فقط mean pool
- train کوتاه (500–1000 meta-iters) یا fine-tune از checkpoint
- جدول: full vs mean-only

### ۷.۲ Ablation GNN (~۲–۳ روز)
- switch به LSTM encoder legacy (یا `comprehensive_encoder_verification.py` as ref)
- همان protocol eval

### ۷.۳ Adaptation curve (~۲ روز)
- `scripts/adaptation_budget_study.py`
- latency @ iter {0, 10, 25, 50, 100} on dist 16 or 12

### ۷.۴ Action distribution slide
- از `structure_conditioned_actions.csv` — ۱ اسلاید «policy یاد گرفته structure-aware است»

---

## ۸. ساختار گزارش (۱۰–۱۲ صفحه)

| فصل | عنوان | صفحات |
|-----|--------|-------|
| ۱ | مقدمه — adaptation challenge | ۱ |
| ۲ | مرور (RL, meta-RL, GNN for graphs) | ۱.۵ |
| ۳ | فرمول MDP + meta-learning | ۱.۵ |
| ۴ | معماری Graph2Seq + decoder | ۲.۵ |
| ۵ | PPO + Reptile algorithm | ۱.۵ |
| ۶ | پروتکل آزمایش و hyperparams | ۱ |
| ۷ | نتایج E1–E6 + ablation | ۲.۵ |
| ۸ | بحث، complexity، محدودیت | ۱ |
| ۹ | نتیجه‌گیری | ۰.۵ |

---

## ۹. پرزنت — اسکریپت اسلاید به اسلاید (~۱۶ دقیقه)

| اسلاید | زمان | محتوا | چه بگویید |
|--------|------|--------|-----------|
| 1 | 1′ | عنوان، KNTU | «پروژه من **meta-RL + Graph2Seq** برای adaptation سریع روی DAG offloading است.» |
| 2 | 1.5′ | چرا adaptation؟ | توزیع task عوض می‌شود |
| 3 | 1.5′ | محدودیت MRLCO | seq encoder, binary, latency-only |
| 4 | 2′ | `fig_margo_architecture` | overview |
| 5 | 2′ | `fig_graph2seq_encoder` | neighborhood aggregation |
| 6 | 1.5′ | `fig_triple_readout` | سه pooling مکمل |
| 7 | 1.5′ | Decoder + 3 actions | Luong attention |
| 8 | 2′ | `fig_meta_reptile_training` | PPO inner, Reptile outer |
| 9 | 1′ | Dataset + meta split | 1900 graphs, 15/4 |
| 10 | 2′ | Results E1–E3 | latency + stability |
| 11 | 1.5′ | Ablation E4–E5 | w/o readout, w/o GNN |
| 12 | 1′ | Complexity N=20 | O(N²) decoder OK |
| 13 | 0.5′ | Q&A | — |

### در پرزنت **نگویید**
- جزئیات معادلات نیم‌دوبلکس V2V
- «ما simulator ساختیم» — بگویید «از simulator استاندارد MEC+V2V استفاده کردیم»

### اگر پرسیدند V2V چطور model شده؟
> «Platform evaluation شامل V2V و energy است؛ تمرکز contribution من **learning architecture** و **meta-training** است.»

---

## ۱۰. سوالات دفاع (Q&A)

| سوال | پاسخ |
|------|------|
| چرا Graph2Seq نه LSTM؟ | DAG structure در node features + neighborhood agg؛ MRLCO seq-only topology را dilute می‌کند |
| Triple readout چرا؟ | attn=critical tasks, mean=global load, max=bottleneck |
| Reptile vs MAML? | first-order، بدون Hessian — سریع‌تر برای scale ما |
| چرا episode one-shot 20 action? | Seq2Seq scheduling formulation از MRLCO lineage |
| std پایین یعنی چی؟ | convergence stable — مخصوصاً vs MRLCO T2 std=29 |
| Ablation چه نشان می‌دهد? | هر component سهم measurable دارد |

---

## ۱۱. مرز با پروژه A

| موضوع | پروژه A | پروژه B (شما) |
|--------|---------|---------------|
| V2V/energy equations | عمیق | ۱ اسلاید |
| Graph2Seq/Meta-RL | ۱ خط | **عمیق** |
| Greedy/HEFT/Pareto | **عمیق** | baseline numbers only |
| Ablation | — | **عمیق** |
| Structural scheduling analysis | **عمیق** | action % summary |

---

## ۱۱. اجرای آزمایش

```bash
# Meta-train (long)
python meta_trainer.py

# Fine-tune eval on held-out distribution
python meta_evaluator.py

# Analysis tables
python paper/analyze.py

# Encoder sanity
python comprehensive_encoder_verification.py
```

**Checkpoints:** `./meta_model_inner_step1/meta_model_final.ckpt`  
**Logs:** `./meta_offloading20_log-inner_step1/`, `./meta_evaluate_ppo_log/`

---

## ۱۲. Figures برای اسلاید

| Figure | Path |
|--------|------|
| Architecture | `figures/fig_margo_architecture.png` |
| Encoder | `figures/fig_graph2seq_encoder.png` |
| Readout | `figures/fig_triple_readout.png` |
| Decoder | `figures/fig_decoder_action_generation.png` |
| Meta-training | `figures/fig_meta_reptile_training.png` |
| Convergence T1/T2 | `figures/fig_latency_convergence_task*.pdf` |
| Action dist | `figures/fig_action_distribution_iteration100.pdf` |

---

## ۱۳. محدودیت‌ها (صادقانه)

- TensorFlow 1.x legacy
- Synthetic DAGs (not real vehicular traces)
- Encoder code: full-sequence connectivity + pred/succ in features (paper/code nuance)
- Single V2V helper
- Ablation short-run — mention in defense

---

## ۱۴. منابع پیشنهادی

- Finn et al., MAML
- Nichol et al., Reptile
- Schulman et al., PPO, GAE
- Hamilton et al., GraphSAGE
- Xu et al., Graph2Seq
- Wang et al., MRLCO (baseline)

---

## ۱۵. چک‌لیست قبل از تحویل

- [ ] Ablation E4 (readout) انجام شده
- [ ] Ablation E5 (GNN) انجام شده
- [ ] نمودار convergence در اسلاید
- [ ] جدول MARGO vs MRLCO با stability
- [ ] گزارش ۱۰+ صفحه
- [ ] پرزنت rehearse ۱۵–۱۸ دقیقه
- [ ] عنوان متفاوت از پروژه A
- [ ] Algorithm 1 (meta-training) در appendix

---

## ۱۶. جدول توازن با پروژه A

| معیار | پروژه A | پروژه B |
|--------|---------|---------|
| صفحات گزارش | 10–12 | 10–12 |
| زمان پرزنت | 15–18 min | 15–18 min |
| آزمایش اصلی | 5 | 5–6 |
| کار اضافه | ~1 هفته | ~1 هفته |
| نوآوری declared | 4 | 4 |
| پیاده‌سازی hands-on | env + baselines | policy + meta train |
| نمودار اختصاصی | Pareto, Gantt | Convergence, ablation |

---

*نسخه: 1.0 — سند پروژه B (Meta-RL و Graph2Seq)*
