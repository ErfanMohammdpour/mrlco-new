# ADR-010: Encoder backbone (v0.3)

Status: Accepted  
Decision date: 2026-09-08 (encoder); 2026-09-08 (readout)  
Depends on: Phase 1 encoder ablation (`spec/prompts/PHASE1_ENCODER_ABLATION.md`)

## Decision

1. The OOD gap (val greedy decode T ≈575–590 vs 2-opt ≈424) is **not representational**. Encoder stays **`meanagg`** (two-layer bidirectional MeanAggregator, hidden 128, node out 256).
2. Readout: adopt **`mean`** (not `triple`). Seed-0 val T **577.02** vs triple **588.14** (Δ **−11.12 s**). `gatv2` / `dagformer` not adopted.

## Gate arithmetic — encoder (validation n=500, 3 seeds, readout=triple)

ΔT = mean_val_T(meanagg) − mean_val_T(candidate). Positive ΔT = candidate better.

| encoder | n_params | ×meanagg | val T per seed | mean±std | ΔT vs meanagg | val token mean | Δtok |
|---|---|---|---|---|---|---|---|
| meanagg | 429185 | 1.00× | 588.14 / 577.28 / 600.92 | **588.78±11.83** | — | 0.6371 | — |
| gatv2 | 626817 | 1.46× | 595.59 / 602.88 / 603.16 | **600.54±4.29** | **−11.76** | 0.6251 | −0.012 |
| dagformer | 531073 | 1.24× | 599.67 / 598.16 / 594.17 | **597.33±2.84** | **−8.55** | 0.6232 | −0.014 |

Pre-registered: ΔT < 10 for both → **"OOD gap is not representational"** — **met** (both ΔT negative).

## Gate arithmetic — readout (meanagg, seed 0 only)

Reference triple val T = **588.1409328751972**. Prompt: any variant within ±5 s of triple → triple is not a contribution.

| readout | n_params | train T | val T | Δ vs triple | within ±5? | val tok |
|---|---|---|---|---|---|---|
| triple | 429185 | 428.97 | **588.14** | 0 | — | 0.6323 |
| **mean** | 232064 | 434.42 | **577.02** | **−11.12** | no | 0.6457 |
| attn | 232321 | 434.87 | 583.08 | −5.06 | no (border) | 0.6413 |
| max | 232064 | 446.29 | 593.94 | +5.80 | no | 0.6392 |
| zero | 199168 | 465.10 | 595.00 | +6.86 | no | 0.6342 |

None strictly inside ±5. **mean** is clearly better than triple (−11 s). Triple is **not** privileged; backbone readout = **mean**.

Caveat: single seed. Phase 2 best-of-k used **triple** ckpt (`meanagg_triple/seed_0`). Phase 3 should load **`meanagg_mean/seed_0`**. Optional: re-run best-of-k on mean ckpt for matched numbers (not blocking).

## Interpretation

Encoder capacity / attention topology under fixed BC-2opt does not close OOD. LSTM init from mean-pooled node embeddings beats concat(mean,max,attn). Phase 3 must attack adaptation / residual joints, not encoder type.

## Artifacts

- Encoder: `spec/kish_log_archive/encoder_phase1_summary.json`, `encoder_phase1/*_eval.json`
- Readout: `spec/kish_log_archive/encoder_readout_summary.json`, `encoder_readout/*_eval.json`
- Queue: `enc_phase1_queue.log`, `enc_readout_queue.log`

## Follow-ups

1. Phase 3 on `meanagg` + `readout_type=mean`, ckpt `runs/phase4/margo_v0.3_diag_encoder/meanagg_mean/seed_0/ckpt/bc_core.ckpt`.
2. Do not re-run encoder candidates for Phase 2/3.
3. Optional matched best-of-k on mean ckpt.

`paper_result=false`. Frozen primary 3500 not started. Meta-test not opened in Phase 1.
