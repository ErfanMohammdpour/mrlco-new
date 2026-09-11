# Session snapshot — MARGO v0.3 Phases 0–2 + Phase 1 close (2026-09-08)

`paper_result=false` everywhere. Meta-test untouched. Frozen primary 3500 **not** started.
Host: kish-ai RTX 4090. Image `margo-phase4-tf115-nv2212`. Repo `/opt/margo/mrlco-new`.

This file is a durable narrative of what was done and what the numbers say. Canonical numbers also live in `PHASE4_WORKLOG.md` / `.json` / `PHASE4_DIAGNOSTIC_LOG.md` and ADRs.

---

## Phase 0 — prune & freeze (done, local)

- ADR-007: PPO-on-θ / Reptile / CAVIA-on-z `mean T` **removed** from method. Replacement EAS-on-φ (Phase 3). Fallback BC + best-of-k.
- ADR-008: inference = π ± sample k≤64 + one `schedule()`; pair/Hamming/2-opt = teacher/ceiling.
- ADR-009: meta-task axis 1 = DAG family; axis 2 = resource profile (Phase 4).
- Dead-code tags on CAVIA/pair/rewrite modules. Architecture → `MARGO-METHOD-v0.3`.

## Phase 1 — encoder ablation (finished)

**Question:** is OOD gap (val greedy ~575–590 vs 2-opt ~424) caused by weak Graph2Seq MeanAgg representation?

**Setup:** same BC-2opt recipe; swap only `encoder_type` ∈ {meanagg, gatv2, dagformer}; readout=triple; seeds {0,1,2}; train from scratch; val only.

**meanagg seed 0 vs archive `bc2opt`:** PASS (±10 s / ±0.02 token). train T 428.97 / val 588.14.

### Val T (n=500) — gate numbers

| encoder | s0 | s1 | s2 | mean±std | ΔT vs meanagg |
|---|---|---|---|---|---|
| meanagg | 588.14 | 577.28 | 600.92 | **588.78±11.83** | — |
| gatv2 | 595.59 | 602.88 | 603.16 | **600.54±4.29** | **−11.76** |
| dagformer | 599.67 | 598.16 | 594.17 | **597.33±2.84** | **−8.55** |

n_params: meanagg 429185; gatv2 626817 (1.46×); dagformer 531073 (1.24×). Token val also worse for both candidates.

**Verdict (ADR-010):** **OOD gap is not representational.** Backbone stays **meanagg**.

Readout ablation on meanagg seed 0 still pending (prompt step 3).

Artifacts: `spec/kish_log_archive/encoder_phase1_summary.json`, `encoder_phase1/*_eval.json`, `spec/decisions/ADR-010-encoder.md`.

## Phase 2 — best-of-k (finished, before Phase 1 close)

**Question:** how much of OOD gap recovers by sampling from π + pick best by `schedule()`?

Ckpt: Phase 1 meanagg seed_0. No training. Seeds {0,1,2}.

### Val T_best_k

| k | mean±std | evals | s/graph |
|---|---|---|---|
| 1 | **588.14±0** | 1 | 0.0027 |
| 8 | **532.23±0.96** | 8 | 0.022 |
| 32 | **509.12±0.48** | 32 | 0.088 |
| 64 | **501.54±0.37** | 64 | 0.175 |

Gate `T_best_32 ≤ 510`: **PASS**. Δk=64 vs greedy **−86.6 s** → bc50 tail claim transfers.

Oracle-within-samples k=64: **~2.2%** (almost never beats 2-opt). Dup k=64 ~0.70 → temp 1.3 reported (T_best_32 ≈502). Dist 10: 804→673 / expert 581.

pair_seq heuristic reference (not method): 447.9 @ ≈718 evals.

**Verdict:** sampling recovers OOD **mean** cheaply; does **not** reach 2-opt 424. Phase 3 cannot imitate 2-opt from samples; can imitate best-of-k vs greedy.

Artifacts: `spec/kish_log_archive/bestofk_s{0,1,2}_eval.json`, `bestofk_summary.json`.

## Locked map of the gap (validation)

```
greedy BC meanagg     ~589
best-of-32            ~509   (32 schedule evals)
pair_seq search ref   ~448   (~718 evals, not method)
2-opt teacher         ~424
all-MEC               ~634
```

PPO-on-θ / CAVIA-on-z / oracle dist_id / rewrite already negative (Phase 0 ledger).

## What this means for the paper

1. Encoder swap is a **negative result** (still publishable ablation).
2. Best-of-k is **standard NCO inference**, not a method contribution alone.
3. Make-or-break = **Phase 3 EAS-on-φ** on meanagg, anchoring to best sample not 2-opt.
4. If Phase 3 flat → story = IL + sampling baseline + occupancy finding + negatives (hard Q1 method paper).

## Next actions (ordered)

1. Optional: readout ablation meanagg seed 0 (`mean|max|attn|zero`).
2. Phase 3 implement + Kish (EAS-on-φ; gate must beat best-of-32 at same eval budget, not only greedy ≤520).
3. Phase X PPO-500 control overnight (parallel after Phase 1 closed).
4. Phase 4–6 per `spec/prompts/README.md`.

Do not start `margo_v0.1_primary` 3500.
