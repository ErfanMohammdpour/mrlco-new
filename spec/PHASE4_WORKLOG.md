# Phase 4 work ledger

**Canonical store.** All diagnostic numbers, exact jobs, commands, artifacts live here.
Narrative extras: `PHASE4_DIAGNOSTIC_LOG.md`. Raw Kish JSON: `kish_log_archive/`.
`paper_result=false` on every row. Frozen primary remains `margo_v0.1_primary` / 3500 / 0.5/0.5 / `parallel=False`.

Host: kish-ai RTX 4090. Image: `margo-phase4-tf115-nv2212`. Repo: `/opt/margo/mrlco-new`. Logs: `/opt/margo/logs/`.
Do **not** rewrite freeze tags `phase0-freeze-v0.1` … `phase3-freeze-v0.1`.
Do **not** push `origin` → `MA-RGO/MARGO` unless asked. Feature remote is `erfan`.
Laptop does **not** run Docker/TF for science.

Machine copy: `spec/PHASE4_WORKLOG.json`.
Human snapshot (2026-09-10): `spec/RESULTS_TO_DATE_2026-09-10.md`.

---

## Do / do not

Do:

- Unique `runs/phase4/<method_id>/seed_N/`
- `paper_result=false`
- Kish via `spec/kish_gpu.sh` + `systemd-run --unit=margo-… --collect`
- Copy finished JSON to `spec/kish_log_archive/` and update this file

Do not:

- 3500 primary
- energy-on for method path
- CAVIA relaunch / outer FiLM
- GAT/Transformer “just in case”
- `schedule()` search at inference
- overwrite `cavia_*` / `pairsup` / freeze tags

---

## Locked conclusions

```
h + search          → ~439   (ceiling, not method)
h + LSTM onepass    → ~575   (OOD greedy)
θ adaptation / PPO  → occupancy damage
z + FiLM (CAVIA)    → no query gain (bc_2opt and bc_continue)
pairsup from clone  → frac_pos=0.070, no OOD gain
pairfrac all-MEC    → frac_pos=0.494, proceed_rewrite=True
rewrite all-MEC     → rewrite_hurts; val 611 vs A 575; n_apply≈0
oracle dist_id      → oracle_no_gain; ident 575.1; train 454 vs 464; val 584 vs A 575
```

Bottleneck: one-pass LSTM cannot extract joints. Encoder not empty (pair search closes most of gap). Train clone is already good so clone-start labels are ~7% positive. Holdout / all-MEC starts are ~50% positive.

DAMRL = GAT + step-by-step MDP. Copying it drops the plan-level claim. Factorized neural rewrite from all-MEC joints **failed** (`n_apply≈0`, T worse). Ceiling still `h`+search, not this operator.

---

## Split (frozen)

- train 15 dists `{1,3,4,5,8,9,11,13,15,18,19,21,22,24,25}` × 100 graphs
- val `{2,6,10,16,17}`
- meta-test `{7,12,14,20,23}`
- 20-task DAG, actions `{0=Local,1=MEC,2=V2V}`, `end_token=3`

Control A: `bc_unseen` greedy val **575.1** / test **557**. Mix val ~0.21/0.75/0.04. `n_non` p50 **4**.

Help gate for rewrite: ΔT vs A ≥ **15s** and greedy-k0 `local_frac` < **0.28**. Hurt if local ≥ **0.35** or T ≥ A+15.

---

## Run ledger

| # | method_id | host | verdict | key number | artifact |
|---|-----------|------|---------|------------|----------|
| 1 | `margo_v0.1_diag_500_parallel` | GPU | MEC collapse iter 21 | frac≥0.95 | `kish_log_archive/runs/…` `gpu_par500.log` |
| 2 | `margo_v0.1_diag_latency_tmec` | GPU | physics OK, skill no | T 958→796; 0 plans 1–2 non-MEC | `gpu_lat50.log` |
| 3 | `margo_v0.1_diag_pomo_tmec` | GPU | worse than lat50 | elite n_non p50=9 | `gpu_pomo50.log` |
| 4 | `margo_v0.1_diag_bc_greedy_tmec` | GPU | IL basin then PPO kills | iter0 T668 mix≈expert; iter9 Local 55% | `gpu_bc50.log` |
| 5 | `margo_v0.1_diag_bc_only_eval` | GPU | repr not broken | train greedy T 533 vs expert 448 | `gpu_bconly.log` |
| 6 | `margo_v0.1_diag_bc_continue` | GPU | IL is the lever | train greedy T **464**, token 0.968 | `ckpt/bc_core.ckpt` `gpu_bccont.log` |
| 7 | `margo_v0.1_diag_bc_unseen` | GPU | mix transfers, T not | val **575** / test **557**, token ~0.71 | `gpu_bcunseen.log` |
| 8 | `margo_v0.1_diag_kl_bc_ppo` | GPU | occupancy damage | Local 21%→74%, T~1000 | `gpu_klppo50.log` |
| 9 | `margo_v0.1_diag_bc_fewshot` | GPU | k3 PPO hurts | val 580→657 | `gpu_bcfew.log` |
| 10 | `margo_v0.1_diag_bc_scheduled` | GPU | exposure ≠ OOD | val 574 / test 555 | `gpu_bcss.log` |
| 11 | `margo_v0.1_diag_bc_2opt` | GPU | better teacher, same OOD | val **581.8** / test **560.8**; 2-opt holdout 423.6/411.6 | `gpu_bc2opt.log` |
| 12 | `margo_v0.1_diag_pair_head` | GPU | search ceiling | greedy 581.8 → pair k20 **457.8** | `gpu_pairhead.log` |
| 13 | `margo_v0.1_diag_pair_ksweep` | GPU | k50 **438.9** vs oracle 436.6 vs 2-opt 423.6 | `gpu_pairksweep.log` |
| 14 | `margo_v0.1_diag_pair_ranker` | GPU | label-shift trap | train pos 6.9% | `gpu_pairranker.log` |
| 15 | `margo_v0.1_diag_pair_seq` | GPU | search still needed | bestimp k50 438.9 | `gpu_pairseq.log` |
| 16 | `margo_v0.2_diag_pairsup` | GPU | **pairsup_no_gain** | frac_pos **0.070**; val 579 vs A 575 | `pairsup_eval.json` |
| 17 | `margo_v0.2_diag_cavia_frozen` | GPU | **cavia_no_gain** | query 584.5→585.1 | `cavia_eval.json` |
| 18 | `margo_v0.2_diag_cavia_strong` | GPU | **cavia_no_gain** | z_l2 0.43–0.67, query flat | `gpu_cavias.log` |
| 19 | `margo_v0.2_diag_cavia_bccont` | GPU | **cavia_no_gain** | identity 575.1; query **572.7→572.7** | `cavia_bccont_eval.json` |
| 20 | `margo_v0.2_diag_pairfrac_mec` | CPU | **pairfrac_mec_holdout_like** | frac_pos **0.494** (6034/12210); mean_Δ **42.9s**; `proceed_rewrite=True` | `pairfrac_mec.json` `gpu_pairfrac.log` |
| 21 | `margo_v0.2_diag_rewrite_mec` | GPU | **rewrite_hurts** | labels frac_pos **0.492**; train T **561** (was 464); val **610.6** vs A 575; test **588.7** vs 557; `n_apply` val 0.004 / test **0** | `rewrite_eval.json` `gpu_rewrite.log` |
| 22 | `margo_v0.2_diag_oracle_dist` | GPU | **oracle_no_gain** | ident **575.1**; train **453.7** vs 464; val **584.2** vs A 575; test **553.0** vs 557; mix train = expert | `oracle_eval.json` `gpu_oracle.log` |
| 23 | `margo_v0.2_diag_binary_lat` | GPU | running | 50 iter; `vocab_size=2`; energy off; `parallel=True` env-only; greedy `{0,1}` | `gpu_binary.log` |

CPU Hamming-2 / motif / 2-opt: expert structure real; 2-opt teacher train T 414 vs greedy 448. Details in `PHASE4_DIAGNOSTIC_LOG.md`.

---

## Locked — `margo_v0.2_diag_rewrite_mec` (finished 2026-09-08)

Unit `margo-rewrite` **success**, GPU free. `paper_result=false`. Labels were the right start (`frac_pos=0.492`, `mean_delta_pos=42.0`, 44649/90824). Training ran 40/40. Verdict **`rewrite_hurts`**.

| split | k0 T | best_k T | A | local0 | n_apply k=3 |
|-------|------|----------|---|--------|-------------|
| train greedy | **561.4** | — | expert 447.9 / bc_continue 464 | 0.266 | — |
| val | 610.9 | 610.6 (k=2) | **575** | 0.276 | **0.004** |
| meta-test | 588.7 | 588.7 (k=0) | **557** | 0.274 | **0.0** |

Mix val 0.276/0.692/0.032 (local still <0.28; hurt is T, not occupancy collapse). Train token vs expert **0.787** (bc_continue was 0.968). CE still falling at ep40 (`loss=0.815` bc=0.302 joint=1.026) but clone already damaged.

**Why K did nothing:** eval picks joints by factorized `logπ_i+logπ_j` on a greedy plan. Greedy already independently argmaxed each token, so almost no alternative pair beats the current pair (`n_apply≈0`). `L_joint` only shifted the one-pass policy (worse T). Search still needs `schedule()`; this operator is not a neural rewrite.

Do **not** relaunch rewrite with λ/lr tricks. Do **not** put pair search at inference.

## Locked — `margo_v0.2_diag_oracle_dist` (finished 2026-09-08)

Unit `margo-oracle` **success**, GPU free. Identity PASS val T **575.1**. 40/40. Verdict **`oracle_no_gain`**. `paper_result=false`.

| split | T | ref | mix L/M/V |
|-------|---|-----|-----------|
| identity val | **575.1** | A 575 | 0.193/0.770/0.037 |
| train greedy | **453.7** | bc_continue 464 / expert 447.9 | **0.204/0.745/0.052** |
| val | **584.2** | A **575** | 0.199/0.761/0.040 |
| meta-test | **553.0** | A **557** | 0.204/0.746/0.049 |

Train token vs expert **0.988**. In-dist ID buys ~10s residual clone (not ≥15, not `oracle_in_dist_only`). OOD val +9s vs A — untrained latin slots + trained W. Mix intact.

**Do not build φ.** Val did not move ≥15. Dist identity is not the 575→439 gap. `h`+search still the ceiling. Do not relaunch oracle. Do not mix with rewrite/CAVIA.

Artifacts: `runs/phase4/margo_v0.2_diag_oracle_dist/seed_0/oracle_eval.json`, `spec/kish_log_archive/oracle_eval.json`, `spec/kish_log_archive/gpu_oracle.log`. `pfm-vllm.service` still stopped; restart only if asked.

---

## v0.2 closed (Phase 0 freeze, 2026-09-08)

Campaign 1–26 locked. No more CAVIA-on-z, unconstrained PPO-on-θ, pair search at inference, or 3500 primary as the method. Decisions: ADR-007 (adaptation engine), ADR-008 (inference = π ± best-of-k≤64), ADR-009 (axis 1 = DAG family; axis 2 = resource profile in Phase 4). Ledger rows above are not rewritten.

## v0.3 plan (gates copied from `spec/prompts/README.md`)

| Phase | File | Gate |
|---|---|---|
| 0 | `PHASE0_PRUNE_AND_FREEZE.md` | ADRs + tests; local; blocks everything |
| 1 | `PHASE1_ENCODER_ABLATION.md` | MeanAgg vs GATv2 vs DAG-Transformer; Kish GPU; blocks 2, 3 |
| 2 | `PHASE2_BEST_OF_K_INFERENCE.md` | k-sample decode + one `schedule()` per sample; Kish; blocks 3 |
| 3 | `PHASE3_EAS_SUPPORT_ADAPTATION.md` | EAS on small φ; Kish GPU; blocks 4 |
| X | `PHASEX_PPO500_CONTROL.md` | 500-iter PPO-from-scratch control; parallel to 1–3; blocks 6 |
| 4 | `PHASE4_SCENARIO_AXIS.md` | resource-profile meta-task; blocks 5 |
| 5 | `PHASE5_ENERGY_PARETO.md` | energy on, λ sweep, Pareto; Kish; blocks 6 |
| 6 | `PHASE6_EVALUATION_RELEASE.md` | 5 seeds, CIs, artifacts, paper alignment; blocks paper |

Numeric gates (same README): val greedy_from_mec 464 / 2-opt 424 / all-MEC 634; BC-2opt greedy 581.8; bc_continue greedy 575. Phase 2 `T_best_32≤510`. Phase 3 query ≤520. Units seconds/joules. `paper_result=false` until Phase 6. Frozen primary 3500 not started.

## v0.3 Phase 1 — encoder ablation (finished)

- Fixture: `spec/fixtures/meanagg_{packed,outputs}.npy` dumped on Kish **before** `encoder_type` edit; regression test passed after edit (atol 1e-5).
- `encoder_type` ∈ {meanagg, gatv2, dagformer}; `readout_type` default triple. Unique dir `runs/phase4/margo_v0.3_diag_encoder/<type>_<readout>/seed_s/`.
- Smoke seed 0: shapes `[2,20,256]`, finite. n_params meanagg **429185**, gatv2 **626817** (1.46×), dagformer after width cut d=128 FFN=256 **531073** (1.24×).
- meanagg seed 0 **PASS** vs `bc2opt_eval.json`: train T **428.97**, val T **588.14**, token train **0.9746** / val **0.6323**.
- Full 3×3 seeds finished. Val T mean±std: meanagg **588.78±11.83** (588.14 / 577.28 / 600.92); gatv2 **600.54±4.29** (ΔT **−11.76**); dagformer **597.33±2.84** (ΔT **−8.55**). Token also worse for both.
- **Gate:** ΔT < 10 for both (both negative) → **"OOD gap is not representational"**. Backbone stays **meanagg**. ADR-010.
- Readout ablation meanagg seed 0 **finished**: mean **577.02** / attn 583.08 / max 593.94 / zero 595.00 vs triple **588.14**. Adopt **`readout=mean`** (Δ −11.1 s). None in ±5 of triple. ADR-010 updated.
- Phase 3 ckpt: `meanagg_mean/seed_0` (Phase 2 was on triple — optional re-bok).
- Artifacts: `encoder_phase1_summary.json`, `encoder_readout_summary.json`, ADR-010.
- Session: `spec/SESSION_SNAPSHOT_2026-09-08.md`. `paper_result=false`. No meta-test.


## v0.3 Phase 3 — EAS-on-φ (finished, FAIL; ADR-011)

- Design: `spec/PHASE3_DESIGN_LOCK.md`. Ckpt `meanagg_mean/seed_0`. Method id `margo_v0.3_diag_eas`.
- Smoke PASS (lastlayer, dist2, N=5, k=4): support best↓, φ moves, query flat (expected), mix intact. `eas_smoke_eval.json`.
- Main seed0 lastlayer+pg_il **FAIL** gate: query greedy 577.7→576.4 (2 dists worse, 3 better), support best 559→463, bok32 after φ 494.4, mix intact, L_IL flat ≈4, PG ≈100× weaker than IL. Ledger §30.
- Ablations seed0: il 577.9; pg **571.2** (best, still FAIL); film 577.7 phi_l2=0 (dead); full 580.7 (worse). **ADR-011** written: meta-adapt claim removed; method = BC + best-of-k.
## Path B — per-instance EAS (after ADR-011)

- Lock: `spec/PHASE3B_INSTANCE_EAS_LOCK.md`. Method `margo_v0.3_diag_eas_inst`. Not meta.
- Equal budget B: bok vs EAS (N×k=B). Default lastlayer+pg. Smoke B=8 n=8 then B=32 all val.
- Full B=32 n=500: bok 498.8 vs eas 498.6 (Δ −0.2); gate ≥5s **FAIL**. Artifacts `kish_log_archive/eas_inst_b32_s0_eval.json`.
- Path B closed. Proceed Phase 4 (ADR-012).


## v0.3 Phase 4 — resource-profile axis (seed0 finished; seeds 1–2 not run)

- Lock: `spec/PHASE4_RESOURCE_AXIS_LOCK.md`. ADR-012.
- Grid 45; train 13 / val held-out 2 + frozen continuity / meta-test 8 (Phase 6 only).
- Method ids: `expert_profiles`, `bc_profiles`, `eas_profiles`, `ctx_profiles`, `bok_profiles`. Unique-dir count **42**. Phase 5 adds `expert_energy` / `bc_energy` / `eas_energy` → **45**.
- Obs v2 scaffold: FEATURE_DIM 15 / PACKED_DIM 54 via `MARGO_OBS_VERSION=v2`; v1 stats hash untouched.
- Expert smoke seed0 **PASS** mix sanity (32 graphs):

| profile | greedy | 2opt | mix_mec | mix_local |
|---|---|---|---|---|
| p_3_3_5 | 696.0 | 633.6 | **0.627** | 0.338 |
| p_11_7_20 | 244.1 | 235.7 | **0.848** | 0.119 |
| frozen_7_5_10 | 370.3 | 353.1 | 0.795 | 0.175 |

- Full expert 13 train profiles **DONE** (~22:34). mix_mec UL3≈0.51–0.57 / UL7≈0.65–0.74 / UL11≈0.71–0.80; frozen twopt **414.1**.
- **stats v2 DONE**: `encoder_feature_stats_v2.json` dim=15 packed=54 n_graphs=19500; heldout interpolation **OK**; v1 sha untouched `94e59875…`.
- Val expert (held-out + frozen) **DONE**. frozen twopt 423.62; `p_5_5_10` 516.9; `p_9_5_10` 363.6.
- BC profiles seed0 **DONE** 120 ep / 19500 graphs / 152 min. Frozen greedy **493.8** vs Phase1 **577.0** (Δ −83.2). Two-sided continuity FAIL; **one-sided no-regression PASS**. Held-out: `p_5` 621.8/516.9 tok 0.670; `p_9` 418.4/363.6 tok 0.737. Mix matches expert.
- EAS profiles seed0 **FAIL** ΔT≥15. 10 tasks, 28 min. T0 **521.1** → T* **521.5** (Δ **−0.4s**). Support −80..160s; query noise ±8s. Same ADR-011 pattern on resource axis. Mix intact.
- CTX profiles seed0 **FAIL** ΔT≥10. 40 ep / 195 tasks / 55 min. T0 **528.2** = Tctx **528.2** (Δ **0.0** all 10 held-out + 5 frozen). Continuity one-sided PASS (frozen T0 502.3). Context z unused. Scenario few-shot dead. Phase 4 adaptation closed.
- BOK profiles seed0 **PASS** 11.4 min. Ckpt BC-profiles seed0 sha `2a385073e71a0689`. obs v2. No train. No meta-test. Gate: continuity + ckpt-match + sampling≥10s.

Frozen val n=500 vs Phase2 bok (v1 triple, greedy 588 / T32 509 / T64 502) and 2-opt **423.6**:

| k | T_best frozen | vs greedy | vs 2-opt |
|---|---|---|---|
| 1 | **493.82** | 0 | +70.2 |
| 8 | 459.39 | −34.4 | +35.8 |
| 32 | **447.67** | −46.2 | +24.0 |
| 64 | **444.48** | −49.3 | +20.9 |

Held-out: `p_5` greedy 621.8 → T64 **546.9** / expert 516.9; `p_9` 418.4 → **377.9** / 363.6. Mix frozen greedy L/M/V **0.226/0.703/0.072**. Temp k=32: 0.7→453.1, 1.0→447.7, 1.3→444.8. Method inference now numbered. `paper_result=false`.

## v0.3 Phase 5 — energy Pareto (parked)

- Lock: `spec/PHASE5_ENERGY_LOCK.md`. Scope ADR-001: **E = UE+HELPER; MEC compute excluded**.
- Physics tests PASS locally: J_1.0 2-opt acts = T 2-opt on toy; helper compute on V2V; HV=6 on 3-point set; 5-graph hand energy 1e-6 J.
- Unique-dir count **45**. Kish CPU smoke seed0 **PASS**: λ=1.0 acts match Phase 4 (`n=8`). frozen_7_5_10 8-graph: T 333→341→430→468→468, E 314→245→74→39→39 as λ 1.0→0. Trend T↑ E↓.
- Full CPU **STOPPED** 2026-09-10. 1/65 done: `p_3_3_5` λ=1.00 T=718.9 E=736.4 match Phase 4 n=1500. Killed mid `p_3_3_5` λ=0.75. Partial npz kept. No Pareto claim. No BC/EAS energy. No meta-test.


## v0.3 Phase 2 — best-of-k inference (finished, seed {0,1,2})

- Ckpt: Phase 1 `meanagg_triple/seed_0` sha `01d566ac80e5fe58`. No training. Unique dir `margo_v0.3_diag_bestofk/seed_s/`. `paper_result=false`. No meta-test.
- T_best_1 train **428.9662928026199** = Phase 1 greedy (all 3 seeds). Monotone k all seeds. Cross-check twopt `_score` 1e-6 inside driver.
- Gate val `T_best_32 ≤ 510`: **PASS** every seed. Tail claim transfers (Δk=64 vs greedy **86.6s** >> 20).
- pair_seq bestimp k20 = **447.9 @ ≈718 evals** is a **heuristic reference, not the method**.

Val `T_best_k` (n=500), per seed then mean±std:

| k | evals | s0 | s1 | s2 | mean±std | s/graph |
|---|---|---|---|---|---|---|
| 1 | 1 | 588.1409 | 588.1409 | 588.1409 | **588.1409±0** | 0.0027 |
| 8 | 8 | 533.0132 | 532.5030 | 531.1619 | **532.2261±0.96** | 0.022 |
| 32 | 32 | 508.9313 | 509.6677 | 508.7621 | **509.1204±0.48** | 0.088 |
| 64 | 64 | 501.6941 | 501.7953 | 501.1182 | **501.5359±0.37** | 0.175 |

Train `T_best_64` 415.27 / 415.43 / 415.40 → **415.37±0.09** (2-opt train 414.1).

Temp at k=32 val: 0.7 → **519.89±0.65**; 1.0 → 509.12; 1.3 → **502.28±0.56**. Dup k=64 val **0.703** (>0.5) so 1.3 required.

Oracle-within-samples val: k=32 **0.019±0.006**; k=64 **0.022±0.006**. Almost never beats 2-opt.

Per-dist val k=64 mean T (greedy → best / expert): **2** 586.8→503.2 / 413.6; **6** 688.5→580.9 / 493.0; **10** 804.2→**672.6** / 580.8; **16** 429.2→377.0 / 310.6; **17** 432.0→374.0 / 320.1.

Mix val k=64 ~0.22/0.71/0.06, `n_nonmec_p50=5`. ~7.3 min/seed.

Verdict: sampling recovers OOD **mean** (588→509) cheaply. Does **not** reach 2-opt 424. Phase 3 cannot imitate 2-opt from these samples; can imitate best-of-k vs greedy. Artifacts: `spec/kish_log_archive/bestofk_s{0,1,2}_eval.json`, `bestofk_summary.json`.

---

## Decision note — amortized context vs joints (2026-09-08)

CAVIA negative ≠ context-meta dead. CAVIA failed as: free `z` + REINFORCE + FiLM on LSTM init. That mechanism is closed.

Amortized task-context (PEARL-inspired encoder, no VAE, no PEARL actor-critic) is a **different** hypothesis: `z_D = f_φ(support graphs)`, freeze at meta-test, forward only.

It is **not** the same as the locked bottleneck. Locked:

```
h + search on THIS graph → ~439
h + LSTM onepass         → ~575
mix already transfers; T does not
few-shot CE on 20 support of the target dist ≈ k0
```

So the missing bit looks like **instance joints inside one DAG**, not **which distribution**. Query graph `h` already sees its own topology. Support DAGs of the same dist add a dist-level statistic. Mix already transferred, so dist identity is weakly the gap.

**Oracle probe (locked):** ident 575.1, train 454 vs 464, val 584 vs A 575, **`oracle_no_gain`**. Do **not** build φ. Dist identity is not the 575→439 gap.

Do not clone PEARL VAE / `(s,a,r,s')`. Do not RL² (two memories). Do not put pair search at inference. `T_j` in context: specify source (all-MEC vs teacher vs current π) or it leaks.

Order done: rewrite hurts, oracle-z no OOD gain. Stop V1/V2 context pool unless a new hypothesis (not ID lookup, not CAVIA-on-z).

## Q1 positioning (2026-09-08)

Old MARGO draft vs MRLCO (~0.6% T, ~4% E) is not a Q1 method paper. Submit that story: reject.

Current Kish story is different: occupancy collapse under PPO-on-θ; IL builds basin; OOD is joints not exposure; CAVIA-on-z dead; `h`+search ceiling ~439 vs LSTM ~575.

Q1 method bar (TPDS/TMC/TNSM/FGCS): close a large slice of 575→439 **without** `schedule()` search at test, mix intact, 5 seeds, vs MRLCO-style / greedy_from_mec / 2-opt. Diagnostic gate ΔT≥15 (~2.6%) is too small as headline.

Rewrite did **not** land near the search ceiling. Oracle dist_id did **not** move OOD (≥15s). Remaining = negative CAVIA + occupancy finding (hard as sole TPDS claim). Do not force Q1 method from rewrite or φ.

Do not advertise CAVIA as the method. Docs still say `MARGO-METHOD-v0.2-cavia`; that title is dead.

---

## pairfrac numbers (locked)

- n=200 stratified train, seed 0, start all-MEC
- `frac_pos=0.494185` n_pos=6034 n_pairs=12210 n_eval=97680
- `mean_delta_pos=42.867` `frac_graphs_with_pos=0.99` `mec_T_mean=618.552`
- per-dist frac_pos: 1:0.400 3:0.457 4:0.290 5:0.489 8:0.392 9:0.573 11:0.604 13:0.666 15:0.445 18:0.530 19:0.544 21:0.268 22:0.502 24:0.570 25:0.542
- clone-trap reference: pairsup `frac_pos=0.070`
