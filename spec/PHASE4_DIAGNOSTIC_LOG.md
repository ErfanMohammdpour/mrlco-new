# Phase 4 diagnostic log (not a paper result)

**Canonical ledger (numbers + exact jobs):** `spec/PHASE4_WORKLOG.md` and `spec/PHASE4_WORKLOG.json`. Raw Kish JSON under `spec/kish_log_archive/`. This file is the narrative.

Status: working notes for a later report. **Do not cite as evaluation.** `paper_result=false` on every row. Frozen primary remains `margo_v0.1_primary` / 3500 / 0.5/0.5 / `parallel=False`.

Host: kish-ai RTX 4090. Image: `margo-phase4-tf115-nv2212`. Repo: `/opt/margo/mrlco-new`. Logs: `/opt/margo/logs/`.

Freeze tags `phase0-freeze-v0.1` … `phase3-freeze-v0.1` were not rewritten.

## Frozen v0.1 (context, not reopened)

20-task DAGs, 25 dists, latin split, 3 actions, energy 0.5/0.5 on **primary**, V2V on, `k_steps=3`, outer first-order mean pseudogradient + Adam `5e-4`, meta-batch 10, train 15 / val `{2,6,10,16,17}` / meta-test `{7,12,14,20,23}`. Publication reward telescoping all_UE fill. Energy scope `total_mobile_joules`. `entropy_coefficient=0.0`. Encoder 128. Sample budget: `max_path_length=20000` → ~1000 trajs/task then `select_support_rows` random 20.

Publication `greedy_plan()` = sequential greedy from empty with all_UE fill. **Not** greedy-from-MEC.

## What we ran

### 0. Learning / parallel probes

- `margo_v0.1_learning_probe` — 5 iter audit. Stack alive.
- `margo_v0.1_parallel_probe` — 5 iter spawn-parallel. Speed OK. Primary sampler stays serial.

### 1. `margo_v0.1_diag_500_parallel` — incomplete, stopped ~iter 180

- Objective: frozen **0.5/0.5** (not latency-only). parallel=True. seed 0.
- Kish: `runs/phase4/margo_v0.1_diag_500_parallel/seed_0/` (local compact copy under same path). Log: `gpu_par500.log`.
- Collapse to MEC from **iter 21** (`max_action_frac≥0.95`). Iter 180 still MEC **0.998**, entropy ~0.01, T around all-MEC (≈570–650). Extra iters after collapse did **not** recover diversity.
- Lesson: more outer iters ≠ more exploration once entropy dies.

### 2. `margo_v0.1_diag_latency_tmec` — finished 50/50

- Reward: `R=-(ΔT/T_allMEC)` unclipped, energy off, entropy 0, parallel, seed 0.
- Kish: `runs/phase4/margo_v0.1_diag_latency_tmec/seed_0/`. Log: `/opt/margo/logs/gpu_lat50.log`.
- Physics OK: no collapse in 50 iters. T **958→796** (−17%). Mix → more MEC. H 1.07→0.72.
- Skill no: T still ~34% worse than publication greedy (~594). **0** plans with 1–2 non-MEC. Best-of-repeats per graph could beat greedy; policy mean did not. Inner alive (`adapt_l2`~0.64). Val only at iter 0.

### 3. `margo_v0.1_diag_pomo_tmec` — finished 50/50, worse than lat50

- Same latency reward + per-graph advantage + elite lowest-T into frozen 20.
- Kish: `runs/phase4/margo_v0.1_diag_pomo_tmec/seed_0/`. Log: `/opt/margo/logs/gpu_pomo50.log`.
- Iter 49: T **921**, greedy 594, T/g **1.55**, mix Local/MEC/V2V **53/33/14**, 1–2 non-MEC **0**. T 881→921 (got worse).
- Not full POMO: computed A on repeats then **dropped losers**; only 20 elites into Adam. Elites structurally wrong: elite T≈607≈greedy but elite `n_nonMEC` median **9** (min 4, never ≤3). Rare sparse samples discarded if not argmin on that graph (iter 0 had n_non=2 with T=521, not selected).
- Lesson: more samples of the same dense π do not invent 1–2-flip plans. 3^20 coverage.

### 4. `margo_v0.1_diag_bc_greedy_tmec` — finished 50/50

- Expert: `greedy_from_mec_plan` (start all-MEC, flip token only if makespan strictly drops, 2 passes). BC 8 epochs on 1500 train graphs, then PPO same latency reward. Inner **random** 20 (not elite).
- Kish: `runs/phase4/margo_v0.1_diag_bc_greedy_tmec/seed_0/`. Log: `/opt/margo/logs/gpu_bc50.log`. `bc_pretrain.json` present.
- Expert: mix L/M/V **0.204/0.745/0.052**. T **448** vs all-MEC **628** (99.2% beats MEC). `n_non` mean 5.10, p50 **4**, `≤3` only **35.5%**. Not the myth “17 MEC + 2 Local”.
- BC not converged: CE `0.653→0.560` still falling. `token_acc=0.779`, teacher-force `greedy_decode_acc=0.762`. **No env greedy-rollout T** in this run.
- Iter 0 (after BC, sampled): mix **0.216/0.738/0.046** ≈ expert. Mean T **668**, greedy-pub **612**. Eval trajs n=10000: `n_non≤2` **4.71%**, `≤3` **12.7%**, T_p10 **445** ≈ expert, T_best 190. Skill is in the **tail**, mean pulled by token errors.
- Iter 9: T **931**, Local **55%**, `n_non≤3 = 0`. Manifold gone.
- Iter 49: T **720**, mix **0.376/0.590/0.034**, greedy-pub **594**, T/g ≈1.21. `n_non≤2` = 2/10000. Health ok all 50. No MEC-collapse. Best mean T was **iter 0**.
- Value head was not BC-trained. Iter 0 `value_loss=2.66` then ~0.05. First PPO steps used a broken critic.
- Lesson: unconstrained PPO after policy-only BC is occupancy shift + mean-gradient, not “needs more iters”. 1000 latency-only from scratch would likely lock all-MEC (see run 1). Continuing this PPO would drift toward T_MEC, not T=448.

### 5. `margo_v0.1_diag_bc_only_eval` — finished (~15 min)

- No PPO. 40 BC epochs. Greedy **env** rollout on 1500 train graphs. `ckpt/bc_core.ckpt` saved (~10 MB).
- Kish: `runs/phase4/margo_v0.1_diag_bc_only_eval/seed_0/`. `greedy_rollout.json`. Log: `/opt/margo/logs/gpu_bconly.log`.
- CE `0.653→0.302` still falling. Teacher token_acc **0.887** (was 0.779 at 8 epochs). Greedy token vs expert **0.811**.
- Argmax schedule T **mean 533 / p50 498 / p10 320**. Expert 448. all-MEC 628. Publication greedy on same train was ~594.
- Beats MEC on **78.1%** of graphs. Beats expert on **2.2%**. Mean **1.19×** expert, **0.85×** all-MEC.
- Mix L/M/V **0.188/0.776/0.036** vs expert **0.204/0.745/0.052** — a bit more MEC, fewer exceptions.
- `n_non` p50 **4** (same as expert), mean 4.48 vs expert 5.10. `≤2` **28.1%**, `≤3` **44.9%** (expert `≤3` was 35.5%).
- Verdict: representation **not broken**. Skill is real (beats MEC and publication greedy). Clone of expert T **incomplete** — missing useful Local/V2V flips, slightly too MEC-conservative. Gap is which tokens to flip, not “network cannot express the policy”. Next: more BC until CE flats, **or** KL/CE-anchored PPO from this ckpt + critic warmup. Not 1000 latency from scratch.

### 6. `margo_v0.1_diag_bc_continue` — finished

- Load `bc_only` `bc_core.ckpt`. Extra BC max 80, early stop patience 10 / min_delta 0.005. Greedy env rollout. No PPO.
- Kish: `runs/phase4/margo_v0.1_diag_bc_continue/seed_0/`. Log: `/opt/margo/logs/gpu_bccont.log`. Cache: `runs/phase4/expert_greedy_mec_train.npz`.
- Load worked: CE epoch1 **0.282** (bc_only ended ~0.302, not 0.65). Early stop epoch **68**, best CE **0.027**, last **0.054**. Teacher token_acc **0.988**. Greedy token vs expert **0.968**.
- Argmax T **mean 464.3 / p50 439.0 / p10 287.5**. Expert 447.9. all-MEC 627.8. Gate ≤480 **pass**. Mean **1.037×** expert, **0.74×** MEC.
- Beats MEC **94.7%**. Beats expert **0.87%** (was 2.2% at 40 epochs — tighter clone, fewer lucky overshoots).
- Mix L/M/V **0.202/0.747/0.051** ≈ expert **0.204/0.745/0.052**. `n_non` mean 5.05 vs expert 5.10, p50 **4**, `≤3` **37.3%** (expert 35.5%).
- Verdict: IL **was** the lever. Remaining ~16s mean gap is residual clone / exposure, not “cannot express policy”. Next if wanted: KL/CE to **frozen π_BC** + critic warmup. Not unconstrained PPO. Not 1000 latency from scratch. Not a paper result.

## Rejected (do not run as “the fix”)

- 1000-iter latency-only from scratch as exploration of 3^20.
- Frozen 3500 primary until this diagnostic chain says otherwise.
- Entropy as a substitute for a manifold.
- POMO/elite rerun with the same dense sampler.
- Force action-share / α=0.8 / MORL / deadline.

## Combinatorial fact (for the report)

`3^20 ≈ 3.5e9` plans/graph. ~1e4 trajs/iter. Uniform π never samples a specific plan. Mass of uniform π is dense (`E[n_non]≈13`), not the expert neighborhood. Expert neighborhood appears only after π is already MEC-dominant; that is also when entropy collapse kills search (run 1).

### 7. `margo_v0.1_diag_kl_bc_ppo` — crashed on load

- Load `bc_continue` into π and a frozen `bc_frozen` copy failed: Graph2Seq `meanaggregator_N` UIDs differ across policy scopes. No PPO iter ran. GPU empty.
- Fix: match weights by canonical slot + UID order (`spec/kl_bc_anchor.py`). Relaunch only after unseen decode gate.

### 8. `margo_v0.1_diag_bc_unseen` — finished. Gate **pass** (`launch_kl=True`)

- Greedy decode `bc_continue` on held-out val `{2,6,10,16,17}` and meta-test `{7,12,14,20,23}`. No PPO. Fat/density latin OOD, CCR still `{0.3,0.4,0.5}`.
- Kish: `runs/phase4/margo_v0.1_diag_bc_unseen/seed_0/unseen_eval.json`. Log: `/opt/margo/logs/gpu_bcunseen.log`.
- Train was T 464 / token 0.968 / mix 0.202/0.747/0.051. Unseen:
  - val n=500: T **575** vs expert **464** vs MEC **634**. **1.24×** expert, **0.91×** MEC. Beat MEC **67%**. Token **0.711**. Mix **0.193/0.770/0.037**. `n_non` p50 **4**.
  - meta-test n=500: T **557** vs expert **444** vs MEC **631**. **1.26×** expert, **0.88×** MEC. Beat MEC **71%**. Token **0.707**. Mix **0.197/0.755/0.048**. `n_non` p50 **4**.
- Mix and `n_non` transferred. Token/T gap grew (OOD clone incomplete). Not a train lookup. Some dists already near MEC (16/17/20/7); fat dists 10/12 still big win vs MEC.
- Verdict: transferable node/DAG rule, incomplete. KL-PPO allowed.

### 7b. `margo_v0.1_diag_kl_bc_ppo` — finished 50/50. **Fail** as refinement

- Load OK (`frozen_vars=30`). Warmup itr 0–4 `vf_only`: mix held ~20/74/5. Sample T ~447–508 (itr3 **447**). Skill still there.
- KL-on itr 5–49 `β=0.1`. Immediate occupancy shift. T **648→~1000** (itr14–17). Mix Local **21%→74%**. `kl_bc_mean` **1.18→16**. Anchor did **not** hold π_BC.
- Late recovery toward MEC not expert: itr49 T **571**, mix **14.2/85.4/0.3**, V2V مرد. `kl_bc` back ~2.6 but basin gone. Health ok all 50 (health ≠ skill).
- Vs BC greedy train T **464** / expert **448**: sampled KL-PPO worse than iter0 **489**. Same failure mode as `bc50`, slower. β=0.1 too weak **or** inner/outer mean-PG still walks off expert tokens.
- Not a paper result. Do not start 3500. Next: not more free PPO. Options: stronger CE-anchor on expert tokens, freeze encoder, or skip RL and few-shot from `bc_continue` ckpt.

### 9. `margo_v0.1_diag_bc_fewshot` — finished. Inner PPO hurts; inner CE no query gain

- Load `bc_continue` `ckpt/bc_core.ckpt`. Val `{2,6,10,16,17}` + meta-test `{7,12,14,20,23}`. Support 20 / query 80 sliced. Sampled query (`HeldOutQueryEvaluator`), not greedy decode. Encoder freeze 20 / adapt 10. `paper_result=false`.
- Kish: `runs/phase4/margo_v0.1_diag_bc_fewshot/seed_0/fewshot_eval.json`. Log: `/opt/margo/logs/gpu_bcfew.log`.
- Sampled k0 ≈ unseen greedy decode (val **580** vs 575; meta-test **559** vs 557). Mix held ~20/75/4–5. Zero-shot BC is the held-out number; few-shot did not add skill.
- k3_ppo: T val **580→657** (+13%), meta-test **559→640** (+15%). Local 0.19/0.20 → **0.29/0.30**. `n_non` p50 4.8/5.2 → **7.8**. Code flag `ppo_destroyed=False` because T ratio just under 1.15; occupancy still left the expert basin. Same failure family as bc50/klppo, one inner loop.
- k3_ce (3 Adam on greedy-from-MEC support tokens, encoder frozen): val **583**, meta-test **565**. ≈ k0, `ce_helps=False`. 20-graph inner IL does not close the OOD T gap vs expert (~464/444 greedy-decode).
- Verdict: remaining gap is **clone/exposure on OOD graphs**, not “need inner PPO” and not “need 3 CE steps on 20 support”. Frozen inner k=3 PPO is the wrong adapt. Do not start 3500. Do not treat BC k0 as a paper result.

### 10. `margo_v0.1_diag_bc_scheduled` — finished. Exposure **not** the OOD gap

- Load `bc_continue` `ckpt/bc_core.ckpt`. Encoder frozen. 40 epochs scheduled sampling, ε **0.8 → 0.3**. Best epoch **38**, train greedy token **1.000**. No PPO. `paper_result=false`.
- Kish: `runs/phase4/margo_v0.1_diag_bc_scheduled/seed_0/scheduled_eval.json`. Log: `/opt/margo/logs/gpu_bcss.log`.
- Train n=1500: T **448.4** vs expert **447.9** vs MEC **628**. Token **0.999**. Mix **0.204/0.745/0.052**. Clone tighter than `bc_continue` (464 / 0.968).
- Held-out greedy vs bcunseen (575 / 0.711 and 557 / 0.707):
  - val n=500: T **574** vs expert **464** vs MEC **634**. Token **0.716**. Mix **0.190/0.778/0.032**. `n_non` p50 **4**.
  - meta-test n=500: T **555** vs expert **444** vs MEC **631**. Token **0.709**. Mix **0.196/0.760/0.045**. `n_non` p50 **4**.
- Δ vs unseen ≈ noise. Target test T **<500** **fail**. Train ~448, test ~555–574 → **OOD fat/density**, not teacher-force / exposure.

### 11. `margo_v0.1_diag_hamming2_expert` — finished. Verdict **mixed**

- One-shot Hamming-2 from greedy_from_mec, 200 stratified train graphs. CPU. No PPO. `paper_result=false`.
- Kish: `runs/phase4/margo_v0.1_diag_hamming2_expert/seed_0/hamming2_eval.json`. Log: `/opt/margo/logs/cpu_h2.log`.
- greedy T **445.2** vs h2 **427.3** vs MEC **618.6**. Mean ΔT **17.9**, p50 **11.3**, max **140**. `frac_h2` **0.73**. `frac_h1` **0.145** (`max_passes=2` not always Hamming-1 min).
- `n_non` 4.80 → 5.15. Pairwise synergy is common, not rare.
- Fat-ish train dists bigger: dist 11 Δ **42.8** (100%), dist 25 Δ **36.1** (100%). Dist 21 Δ **6.3** (38%).
- Vs OOD clone gap (~110 T on val 574 vs expert 464): teacher headroom **real but smaller**. Both A and B live. Do not start 3500.

### 12. `margo_v0.1_diag_motif_expert` — finished. Verdict **mixed_structure**

- Motif audit of `greedy_from_mec`. CPU. No PPO. `paper_result=false`.
- Kish: `runs/phase4/margo_v0.1_diag_motif_expert/seed_0/motif_eval.json`. Log: `/opt/margo/logs/cpu_motif.log`.
- Train n=200 (same stratified seed-0 as Hamming-2; live greedy): n_non **4.80**, n_cc **2.48**, cc_ratio **0.63**, max_cc **3.12**, local_nbr **0.45**, homo **0.75**, CP changes **0.66** / path **4.75**. Single-CC among n_non≥2: **18%**.
- val n=500 / meta-test n=500 (cache): same mixed. n_non 5.07 / 5.22, n_cc 2.53 / 2.70, cc_ratio 0.62 / 0.64, local_nbr 0.49 / 0.45.
- Mean component size ≈ **2**. Not one 5-node region. Not 5 islands. Hamming-2 pairwise fits this scale.
- Fat-ish more clustered: train dist 11 cc_ratio **0.31** (n_non 7.6, n_cc 2.0). val dist 10 **0.33**. meta-test dist 12 **0.37**. Skinny train dist 1 **0.97**.
- OOD fat/density is **not** a different motif law. Holdout even more region-like. Skip DiffPool / 4×3^5. Region head optional only as small-CC / pair bias on Graph2Seq, not a replacement encoder.
- Do not start 3500.

### 13. `margo_v0.1_diag_2opt_expert` — finished. CPU. No PPO

- Iterative 2-opt from `greedy_from_mec`: H1 local min, then repeat best H2+H1 until empty (cap 8). `paper_result=false`.
- Kish: `runs/phase4/margo_v0.1_diag_2opt_expert/seed_0/twopt_eval.json`. Caches: `expert_2opt_mec_{train,validation,metatest}.npz`. Log: `/opt/margo/logs/cpu_twopt.log`.
- Train n=1500: greedy **447.9** → H1 **441.0** → first H2 **428.2** → 2-opt **414.1**. Δ vs greedy **33.8**. Extra vs one-shot H2 **14.2**. `h2m` **1.17**. `n_non` 5.10 → **5.91**. Cap hit **0**. Join-200 vs Hamming-2: greedy 445.2 / first H2 425.5 / tw **411.4** (H2 one-shot was 427.3).
- val n=500: greedy **463.5** → 2-opt **423.6**. Δ **39.9**. Fat dist 10: 656→**581** (Δ76).
- meta-test n=500: greedy **443.5** → 2-opt **411.6**. Δ **32.0**. Dist 12: 588→**532** (Δ57). Cap **0.4%**.
- Teacher headroom real and larger than one-shot 18s. Holdout ceiling now **424 / 412**, still far from BC unseen **574 / 555**. Gap B lives. Do not start 3500.

## Next after 2-opt

Done: section 14. Teacher closed. OOD remains.

### 14. `margo_v0.1_diag_bc_2opt` — finished. GPU. No PPO. Verdict: **teacher closed, OOD remains**

- First launch crashed after epoch 39 (`end_token=2` == V2V). Relaunch: `end_token=3`. Early-stop epoch **44**, best CE **0.0508**.
- Train n=1500: greedy T **435.6** vs 2-opt **414.1** vs MEC **628**. Token teacher **0.990**, greedy vs expert **0.965**. Mix L/M/V **0.222/0.706/0.072**. Clone works (~+22s).
- val n=500: T **581.8** vs 2-opt **423.6** vs MEC **634**. Token **0.646**. vs old BC unseen **574**. Fat dist 10: **794** vs expert **581**.
- meta-test n=500: T **560.8** vs 2-opt **411.6** vs MEC **631**. Token **0.659**. vs old **555**. Fat dist 12: **725** vs expert **532**.
- Better teacher moved train (464→436) not holdout (~574/555 stayed). Gap B vs new ceiling **~158 / ~149s**, larger because ceiling dropped not because π improved.
- Kish: `runs/phase4/margo_v0.1_diag_bc_2opt/seed_0/`. Artifact: `spec/kish_log_archive/bc2opt_eval.json`. Log: `/opt/margo/logs/gpu_bc2opt.log`.
- `paper_result=false`. Not 3500. PPO still negative. Paper method = IL k=0 (greedy or 2-opt teacher for NN does not matter on OOD). CPU 2-opt is the honest deploy/oracle baseline, not the learned method.

## Next after BC-2opt

Pair/motif diagnostic. Not PPO. Not 3500. Backbone unchanged.

### 15. `margo_v0.1_diag_pair_head` — finished. GPU extract + numpy head. No PPO. Verdict: **decoder_interaction**

- Freeze `bc_2opt` encoder. Auxiliary 9-way `P(a_i,a_j)` on direct / sibling / join. LSTM decoder unchanged.
- val n=500: BC **581.8** → top-20×9 **457.8** vs 2-opt **423.6**. Δ **124**. improved **0.98**. ~180 evals/graph.
- meta-test n=500: BC **560.8** → **443.5** vs 2-opt **411.6**. Δ **117**. improved **0.96**.
- Oracle n=100 val (all motif pairs, ≤8 rounds): BC **595.2** → **441.6**. Δ **154**. ~3057 evals/graph. Sample harder than full val; not apples-to-apples with 457.8.
- Head weak: val joint-acc **0.443**, acc on expert≠BC **0.328**, recall@20 **0.373**. Ranking incomplete; 9-way search on top-20 still closes ~78% of BC→2-opt gap.
- Kish: `runs/phase4/margo_v0.1_diag_pair_head/seed_0/`. Artifact: `spec/kish_log_archive/pair_eval.json`. Log: `/opt/margo/logs/gpu_pairhead.log`.
- `paper_result=false`. Not 3500. PPO still unused.

### 16. `margo_v0.1_diag_pair_ksweep` — finished. No PPO. Verdict: diminishing returns, k=20 sweet

- val n=500, same graphs. BC **581.8**, 2-opt **423.6**. k10 **473.8** (90 eval). k20 **457.8** (180). k50 **446.7** (389). Motif oracle **436.6** (2732 eval). improved oracle **0.986**.
- meta-test: k10 **458.8**, k20 **443.5**, k50 **433.5** vs 2-opt **411.6**.
- k=10 takes most of the gain. 10→20 ~16s. 20→50 ~11s. k50 still **10s** above same-split oracle, oracle **13s** above 2-opt.
- Kish: `runs/phase4/margo_v0.1_diag_pair_ksweep/seed_0/`. Artifact: `spec/kish_log_archive/ksweep_eval.json`. Log: `/opt/margo/logs/gpu_pairksweep.log`.
- `paper_result=false`. Not 3500.

## Next after k-sweep

ΔT pair ranker. Not PPO. Not 3500. Backbone unchanged.

### 17. `margo_v0.1_diag_pair_ranker` — finished. GPU extract + numpy ridge. No PPO. Verdict: **one_pass_gain_rank_insufficient**

- Freeze `bc_2opt` encoder. LSTM unchanged. Labels: one-shot `T_BC − min_9 T(pair)` on motif pairs.
- Train gains sparse (BC clones 2-opt): **6274 / 90824** pos (**6.9%**). Val **52%** / test **48%** pos — ranker label shift, not just action OOD.
- Ridge train pearson **0.311**. Val learned pearson **0.020**, recall@20 graph-mean **0.40** vs CE **0.57** vs gain-oracle **0.70**.
- val n=500, BC **581.8**, 2-opt **423.6**. k20: CE **457.8** / learned **475.3** / gain-oracle **457.4**. k10: CE **473.8** / learned **496.8** / gain-oracle **465.7**.
- meta-test k20: CE **443.5** / learned **461.4** / gain-oracle **442.0**.
- Learned ΔT **worse** than CE. Perfect one-shot ΔT rank at k=20 **ties** CE (457.4 vs 457.8). Remaining ~21s to sequential motif oracle **436.6** is multi-round search, not first-pass ranking.
- Kish: `runs/phase4/margo_v0.1_diag_pair_ranker/seed_0/`. Artifact: `spec/kish_log_archive/ranker_eval.json`. Log: `/opt/margo/logs/gpu_pairranker.log`.
- `paper_result=false`. Not 3500.

## Next after ranker

Sequential pair refine. Not PPO. Not 3500. Backbone unchanged.

### 18. `margo_v0.1_diag_pair_seq` — finished. GPU extract + CE sequential search. No PPO. Verdict: **seq_helps_need_more_pairs**

- Same frozen encoder + CE ranking. LSTM unchanged. Unit `margo-pairseq` `Result=success`. Artifact `runs/phase4/margo_v0.1_diag_pair_seq/seed_0/seq_eval.json`. Log: `/opt/margo/logs/gpu_pairseq.log`.
- val n=500, BC **581.8**, 2-opt **423.6**, oracle ref **436.6**: scan k20 **457.8** / multipass **451.0** / bestimp k20 **447.9** (~718 eval) / bestimp k50 **438.9** (~1671 eval).
- meta-test: scan **443.5** / multi **437.0** / best20 **434.5** / best50 **426.8** vs 2-opt **411.6**.
- Sequential on the same top-20 buys ~10s (458→448). Does not close oracle. k50 sequential ≈ oracle → remaining 21s was **more pairs**, not scan vs bestimp.
- Heuristic ceiling only. Not the paper method. `paper_result=false`. Not 3500.

## Next after pair_seq

CAVIA-on-z T-only. Not pair search. Not PPO. Not 3500.

### 19. `margo_v0.2_diag_cavia_frozen` — finished. GPU. Energy off. Verdict: **cavia_no_gain**

- Identity PASS. val greedy `z=0` T **581.8** / test **560.8**. Mix val `0.224/0.713/0.064`, `n_non` p50 **5**. Same BC 2-opt.
- Inner 20 × lr `5e-4`. Support T flat on 9/10 dists. `z_l2≈0.01–0.02`.
- query val **584.5 → 585.1**. test **562.0 → 561.9**. Inner never moved decode.
- Artifact: `runs/phase4/margo_v0.2_diag_cavia_frozen/seed_0/cavia_eval.json`. Log: `/opt/margo/logs/gpu_cavia.log`.
- `paper_result=false`. Not 3500. Do not turn energy on yet.

## Next after cavia_frozen

Stronger inner, same θ freeze. If still flat → pair-sup, not energy.

### 20. `margo_v0.2_diag_cavia_strong` — finished. GPU. Energy off. Verdict: **cavia_no_gain**

- Identity PASS. val T **581.8**. Inner 50 × lr `1e-2`.
- `z_l2` now **0.43–0.67** (frozen was 0.01). Inner moved z. Query still dead: val **584.5 → 584.2**.
- meta-test query mixed: dist12 **715.6→708.9**, dist20 **423.9→428.7** worse. Mix held.
- Artifact: `runs/phase4/margo_v0.2_diag_cavia_strong/seed_0/cavia_eval.json`. Log: `/opt/margo/logs/gpu_cavias.log`.
- `paper_result=false`. Not 3500. Do not turn energy on. CAVIA-on-z cannot steer this 2-opt clone.

## Next after cavia_strong

Pair-sup from **non-cloned** start. Eval greedy only. No search at inference.

### 21. `margo_v0.2_diag_pairsup` — finished. GPU. Verdict: **pairsup_no_gain**

- Labels from greedy_from_mec train: `frac_pos=0.070` (6354/90824). Same trap as 2-opt-clone ranker (6.9%). Code printed `pairsup_label_frac_pos_low`.
- Train greedy T **511.1** vs A expert-clone **464**. Joint loss dropped; occupancy mix held `0.211/0.738/0.051`.
- Unseen: val **579.1** vs A **575**. test **557.0** vs A **557**. Mix val `0.209/0.748/0.043`, `n_non` p50 **4**. Token **0.693**.
- Artifact: `runs/phase4/margo_v0.2_diag_pairsup/seed_0/pairsup_eval.json`. Log: `/opt/margo/logs/gpu_pairsup.log`.
- `paper_result=false`. Not 3500. Energy still off. Pair-sup does not unstick OOD decode.

## Next after pairsup

CAVIA-on-z from `bc_continue`, not `bc_2opt`. T-only. 20 inner steps. Unique dir.

### 22. `margo_v0.2_diag_cavia_bccont` — finished. GPU. Energy off. Verdict: **cavia_no_gain**

- Identity PASS. val full T **575.1** in `[568, 582]`. Load `bc_continue` `n_vars=30`, `z_l2=0`.
- Inner 20 × lr `5e-4`. `z_l2≈0.012–0.021`. Query val **572.7 → 572.7**. All dists flat except dist 12 718.3→717.9.
- Artifact: `runs/phase4/margo_v0.2_diag_cavia_bccont/seed_0/cavia_eval.json`. Log: `/opt/margo/logs/gpu_caviab.log`.
- `paper_result=false`. Not 3500. CAVIA-on-z dead on both `bc_2opt` and `bc_continue`. C1 negative in this setup.

## Next after cavia_bccont

Cheap CPU `frac_pos` of motif 9-way labels from **all-MEC** train plans (T≈628), not greedy_from_mec clone. No network. Gate: `frac_pos >= 0.20` then rewrite; else CAVIA+pair-sup closed.

### 23. `margo_v0.2_diag_pairfrac_mec` — finished. CPU. Verdict: **pairfrac_mec_holdout_like**

- 200 stratified train. Start all-MEC. `frac_pos=0.494` (6034/12210) vs clone trap **0.070**. `n_eval=97680`. `mec_T_mean=618.6`.
- `mean_delta_pos=42.9s`. `frac_graphs_with_pos=0.99`. Per-dist frac 0.27–0.67. Gate `>=0.20` PASS. `proceed_rewrite=True`.
- Artifact: `runs/phase4/margo_v0.2_diag_pairfrac_mec/seed_0/pairfrac_mec.json`. Log: `/opt/margo/logs/gpu_pairfrac.log`.
- `paper_result=false`. Not 3500. No network. No pair search at inference.

## Next after pairfrac

All-MEC labels have joints. Wire `L_joint` from this start + greedy(+K neural apply). Zero `schedule()` search at inference. Control A = `bc_unseen` val 575. Help if ΔT≥15 and mix OK. Unique dir. `paper_result=false`. Ledger: `spec/PHASE4_WORKLOG.md`.

### 24. `margo_v0.2_diag_rewrite_mec` — finished. GPU. Energy off. Verdict: **rewrite_hurts**

- Labels OK (not the clone trap): `frac_pos=0.492` (44649/90824), `mean_delta_pos=42.0`, n_graphs=1500. Gate `>=0.20` PASS. Cache `rewrite_joint_from_allmec_train.npz`.
- Train 40/40 from `bc_continue`. Loss 2.33→0.815 (bc 0.624→0.302, joint 3.42→1.026). Train greedy T **561.4** vs bc_continue **464** vs expert **447.9**. Mix 0.266/0.695/0.039. Token vs expert **0.787** (was 0.968). Clone damaged by `L_joint`.
- Val k0 **610.9** / best k=2 **610.6** vs A **575**. Mix 0.276/0.692/0.032. local0 still <0.28.
- Test k0=k1=k2=k3 **588.7** vs A **557**. Mix 0.274/0.688/0.038.
- Neural apply almost never fires: val `n_apply` 0.002/0.004, test **0.0**. Factorized `logπ_i+logπ_j` cannot beat a greedy plan that already independently argmaxed each token. K is a no-op; hurt is the shifted one-pass policy.
- Artifact: `runs/phase4/margo_v0.2_diag_rewrite_mec/seed_0/rewrite_eval.json`. Log: `spec/kish_log_archive/gpu_rewrite.log`.
- `paper_result=false`. Not 3500. No PPO. No pair search at inference. Do not λ/lr-relaunch.

### 25. `margo_v0.2_diag_oracle_dist` — finished. GPU. Energy off. Verdict: **oracle_no_gain**

- Identity PASS: val T **575.1** in `[568, 582]`. Mix 0.193/0.770/0.037.
- Train 40/40, encoder frozen. Loss ~0.04→0.012. Train greedy T **453.7** vs bc_continue **464** vs expert **447.9**. Mix **0.204/0.745/0.052** = expert. Token vs expert **0.988**.
- Val **584.2** vs A **575** (local 0.199). Test **553.0** vs A **557** (local 0.204). Mix intact.
- In-dist ID ~10s residual clone, not ≥15. OOD ID lookup does not close 575. Do **not** build φ. Bottleneck remains instance joints (`h`+search ~439).
- Artifact: `runs/phase4/margo_v0.2_diag_oracle_dist/seed_0/oracle_eval.json`. Log: `spec/kish_log_archive/gpu_oracle.log`.
- `paper_result=false`. Not 3500. No PPO. No pair search. No CAVIA. No rewrite mix.

### 26. `margo_v0.2_diag_binary_lat` — running. GPU. Energy off. No V2V.

- Isolation: current Graph2Seq+LSTM, latin split, `k_steps=3`, `parallel=True` (env executor only; worker RNG private so `sample_tasks` stream matches serial), `vocab_size=2`, `end_token=2`, greedy `{UE,MEC}` only, `R=-(ΔT/T_allMEC)`, `use_energy=False`.
- 50 outer iters. Unique dir. Compare occupancy + T vs **binary** publication greedy, not vs control A 575 (ternary+V2V).
- Frozen primary stays ternary 0.5/0.5 V2V-on 3500, not started.
- Log: `/opt/margo/logs/gpu_binary.log`. Unit: `margo-binary`.

## v0.3 Phase 0 — decisions frozen

1. ADR-007: inner PPO on θ, Reptile/mean-PG on θ as method, CAVIA-on-z `mean T` **removed**. Replacement EAS-on-φ (Phase 3). Fallback BC + best-of-k.
2. ADR-008: inference = π ± sample k≤64 + one `schedule()` each; pair/Hamming/2-opt are teacher or ceiling, not π; always report `evals_per_graph`.
3. ADR-009: axis 1 = DAG family, fixed resources; axis 2 = resource profile in Phase 4; context `z` had nothing to infer on axis 1.
4. Architecture header `MARGO-METHOD-v0.3`. CAVIA §9 is a removal stub. FiLM kept as Phase 3 subset candidate.
5. C2 evidence: `par500` MEC≥0.95 iter 21 / 0.998 iter 180; `lat50` 958→796 zero 1–2 non-MEC; `pomo` elite n_non p50=9; `bc50` Local 55% iter 9; `klppo` kl_bc 1.18→16 Local 21%→74%; `fewshot` val 580→657.
6. CAVIA three flats: 584.5→585.1; 584.5→584.2; 572.7→572.7.
7. IL in-dist: `bc_continue` 464.3 / 0.968; OOD val 575 vs expert 464.
8. Phase 1 gate: encoder ΔT val ≥30 s or conclude gap is not representational.
9. Phase 2 gate: val `T_best_32≤510`.
10. Phase 3 gate: query greedy ≤520, Local mix 0.15–0.25.
11. Phase X: PPO-500 control; P1–P3 pre-registered; parallel to 1–3.
12. Phase 4–5: resource axis then energy Pareto. Phase 6: 5 seeds, paper.
13. Audit C-01..C-08,C-12,H-01 closed in this tree (file:line in ADR-007 appendix). C-09/C-10/C-11 reports: Phase 6.
14. `paper_result=false`. Freeze tags `phase0-freeze-v0.1`…`phase3-freeze-v0.1` not rewritten. 3500 not started.
15. Dead method modules tagged `STATUS: baseline/ablation only (ADR-007)`. Kish not touched this phase.

### 27. `margo_v0.3_diag_bestofk` — finished. GPU. No training. Verdict: **sampling_recovers_ood_mean_not_2opt**

- Frozen BC meanagg seed_0 ckpt. Sample 0 = greedy. Prefix k-sweep. One `schedule()` per plan. Seeds {0,1,2}. Validation n=500. Meta-test not opened.
- T_best_1 train = Phase 1 greedy **428.9662928026199** (all seeds). Val greedy **588.1409**.
- Gate: val `T_best_32` 508.93 / 509.67 / 508.76 → **509.12±0.48 ≤ 510** PASS. `T_best_64` **501.54±0.37**. Δ vs greedy **86.6s** — bc50 tail transfers.
- Cost: k=32 → 32 evals, **0.088 s/graph**. pair_seq bestimp k20 **447.9 @ ≈718 evals** is heuristic reference, not the method.
- Temp 1.3 at k=32: **502.28±0.56** (dup k=64 = 0.70). Oracle-within-samples k=64 **0.022**. Dist 10: 804.2 → **672.6** / expert 580.8.
- Remaining gap to 2-opt val 424 ≈ 78s. Phase 3 cannot imitate 2-opt from these samples; can imitate best-of-k vs greedy.
- Artifacts: `spec/kish_log_archive/bestofk_s{0,1,2}_eval.json`, `bestofk_summary.json`. `paper_result=false`. Not 3500.

### 28. `margo_v0.3_diag_encoder` — finished. GPU. No PPO. Verdict: **ood_gap_not_representational**

- Same BC-2opt recipe; encoders meanagg / gatv2 / dagformer; readout triple; seeds {0,1,2}; from scratch; validation only.
- meanagg s0 matched archive `bc2opt` within ±10 s / ±0.02 token.
- Val T mean±std: meanagg **588.78±11.83**; gatv2 **600.54±4.29** (ΔT **−11.76**); dagformer **597.33±2.84** (ΔT **−8.55**). Both candidates worse; token also down.
- Gate: ΔT < 10 for both → gap is **not** representational. Backbone stays meanagg. ADR-010.
- Readout ablation meanagg seed 0 still pending.
- Artifacts: `spec/kish_log_archive/encoder_phase1_summary.json`, `encoder_phase1/*_eval.json`, `spec/decisions/ADR-010-encoder.md`.
- Session: `spec/SESSION_SNAPSHOT_2026-09-08.md`. `paper_result=false`. Not 3500. No meta-test.

### 29. meanagg readout ablation — finished. Verdict: **adopt_mean_readout**

- Seed 0 only. Val vs triple **588.14**: mean **577.02** (Δ−11.1), attn 583.08 (−5.06), max 593.94 (+5.8), zero 595.00 (+6.9).
- None inside ±5 of triple. mean clearly better → backbone readout = **mean**. ADR-010 updated.
- Phase 3 ckpt: `meanagg_mean/seed_0`. Phase 2 best-of-k was on triple (optional rematch).
- Artifacts: `spec/kish_log_archive/encoder_readout_summary.json`. `paper_result=false`.

### 30. `margo_v0.3_diag_eas` — main lastlayer+pg_il seed0 FINISHED. GPU. Verdict: **dist_level_phi_no_greedy_gain** (gate FAIL)

- Design lock: `spec/PHASE3_DESIGN_LOCK.md`. Ckpt meanagg+mean sha in provenance. φ=`lastlayer` = `output_projection/kernel` 128×3 = **384 params**; all other 25 vars frozen (audit printed).
- Identity (100 graphs/dist, greedy): 574.6 / 680.1 / 790.1 / 436.0 / 405.5 → mean **577.27** vs Phase-1 577.02 (Δ0.25; tol 0.1→0.5).
- Smoke PASS (dist2 N=5 k=4). Main N=100 k=16 λ_IL=1 lr=1e-3, 5 val dists, **13.5 min**.
- Query greedy per dist T0→T*: d2 578.5→**587.6** (+9.1); d6 685.6→**692.5** (+6.9); d10 785.5→776.9 (−8.6); d16 431.7→420.4 (−11.3); d17 407.1→404.8 (−2.3). Mean **577.7→576.4** (Δ−1.2). Gate ≤520 **FAIL**.
- Support best-so-far (non-increasing, bookkeeping OK): 559→463 / 658→532 / 808→623 / 447→348 / 399→345. Search on support works; mean-of-best ≈ 462 on d2 (Phase-2 k64 was ≈501 on 500 graphs).
- Query best-of-32 after φ*: 500.4 / 577.7 / 668.9 / 360.3 / 364.6 → mean **494.4** (Phase 2 on triple ckpt k32 = 509; zero-shot k32 on mean ckpt not yet measured → cannot attribute to φ).
- Curves: query T flat/noisy at {5,10,20,50}, then jumps at 100 either way → **not monotone** (0/5 dists satisfy T20≤T5≤T0 cleanly). L_IL stays ≈3.6–6.4 nats/plan (does **not** decrease; target moves as best gets rarer). L_PG ≈ ±0.05 (advantage normalized by T_allMEC ≈ 600 → PG ≈100× weaker than IL; pg_il ≈ il in practice).
- φ_l2 grows linearly 0→≈1.2 (Adam lr×steps bound), mix unchanged (Local 0.219, V2V 0.059, n_non p50 5.7 → in band). **Anchor prevents collapse** (unlike PPO-on-θ) but does not move greedy.
- Interpretation: consistent with §25 `oracle_no_gain` — any φ **shared across graphs of a dist** carries dist-level info only; the 577→439 gap is instance-level joints. Support-best signal is per-instance and does not transfer to unseen query graphs via a 384-param shared head.
- Ablation queue launched (seed 0): `full pg_il` (capacity control), `lastlayer il`, `lastlayer pg`, `film pg_il`. Logs `/opt/margo/logs/eas_<subset>_<loss>_s0.log`, queue `/opt/margo/logs/eas_ablation_queue.log`.
- Artifacts: `spec/kish_log_archive/eas_lastlayer_pgil_s0_eval.json`, `eas_lastlayer_pgil_s0.log`, `eas_smoke_eval.json`. `paper_result=false`. No meta-test.

### 31. `margo_v0.3_bok_profiles` — finished. GPU. No training. Verdict: **sampling_closes_most_of_v2_gap**

- Ckpt: BC-profiles seed0 sha `2a385073e71a0689`. obs v2 packed=54. Unique dir `margo_v0.3_bok_profiles/seed_0/`. Unique-dir count 42.
- Frozen val n=500: greedy **493.82** (matches BC) → T32 **447.67** → T64 **444.48** vs 2-opt **423.62**. Sampling −46.2s at k=32; leftover to teacher **+20.9s**.
- vs Phase 2 bok (v1 triple): T32 509→448, T64 502→444.
- Held-out: `p_5` 621.8→546.9 / expert 516.9; `p_9` 418.4→377.9 / expert 363.6.
- Mix frozen greedy L/M/V 0.226/0.703/0.072. Temp 1.3 slightly better than 1.0 at k=32 (444.8 vs 447.7).
- Gate PASS. Method inference numbered. Next Phase 5 energy. `paper_result=false`. No meta-test.
- Artifacts: `spec/kish_log_archive/bok_profiles_s0_eval.json`, `bokprof_full_s0.log`.

### 32. `margo_v0.3_expert_energy` — parked 2026-09-10. CPU. No Pareto claim.

- Physics tests local PASS (J_1.0=T on toy; helper compute; HV=6; 5-graph 1e-6 J). ADR-001 scope.
- Kish smoke PASS: λ=1.0 match Phase 4 n=8. frozen_7_5_10 8-graph T 333→468 / E 314→39 as λ 1→0.
- Full STOPPED 1/65: `p_3_3_5` λ=1.00 T=718.9 E=736.4 match n=1500. Killed mid λ=0.75.
- No BC/EAS energy. No meta-test. Snapshot: `spec/RESULTS_TO_DATE_2026-09-10.md`. `paper_result=false`.

