# Phase 4 — resource-profile meta-task axis

Date: 2026-09-09. `paper_result=false`. ADR-012.

## Claim

Meta-task = `(DAG distribution, resource profile)`. Scenario rates/CPU are the latent axis where few-shot / context adaptation is testable vs MRLCO/DAMRL. Phases 1–3 stay reproducible on frozen profile + obs v1.

## Locked grid

- Conversion: `B/s = Mi-Mbps × 1024×1024 / 8` (7 → 917504; 5 → 655360).
- Train profiles: 13 (`meta_train_profile_ids` in yaml; includes `frozen_7_5_10`).
- Val held-out: 2 (`p_5_5_10`, `p_9_5_10`). Continuity: +5 frozen-profile val tasks.
- Meta-test: 8 profiles × 5 dists = 40 — **Phase 6 only; do not touch**.

## Method ids (isolated dirs)

| id | job |
|---|---|
| `margo_v0.3_expert_profiles` | CPU 2-opt teachers per train profile |
| `margo_v0.3_bc_profiles` | BC retrain on 195 tasks, obs v2, meanagg+mean |
| `margo_v0.3_eas_profiles` | Phase-3-style EAS on held-out profile val |
| `margo_v0.3_ctx_profiles` | PEARL-style context-only baseline |
| `margo_v0.3_bok_profiles` | best-of-k on BC-profiles ckpt; frozen + held-out val |

## Obs

- v1: FEATURE_DIM=11, PACKED_DIM=50, `encoder_feature_stats.json` (hash frozen; Phases 1–3).
- v2: FEATURE_DIM=15, PACKED_DIM=54, `encoder_feature_stats_v2.json`. Per-node `[log UL, log V2V, log MEC_CPU, log UE_CPU]` + optional `resource_ctx[B,4]`.
- Activate v2 via `MARGO_OBS_VERSION=v2` at process start (before policy import).

## Order

1. Expert smoke: `p_3_3_5`, `p_11_7_20`, `frozen_7_5_10` × 32 graphs — mix sanity UL=3→Local/V2V, UL=11→MEC.
2. Expert full: 13 train profiles × 1500 graphs.
3. Fit stats v2 on meta_train × train profiles only.
4. BC profiles seeds {0,1,2}; **one-sided continuity**: frozen val T ≤ Phase1 meanagg_mean + 10s (no-regression). Two-sided `|Δ|≤10` rejected a real gain (seed0 493.8 vs 577.0).
5. EAS + context on 10 held-out val tasks. Stop only if frozen val **regresses** vs Phase 1.

## Kish

- Target: `expertprof <seed> [smoke]` / `bcprof <seed> [smoke]` / `easprof <seed> [smoke]` / `ctxprof <seed> [smoke]` / `bokprof <seed> [smoke]`
- Image: `margo-phase4-tf115-nv2212` (CPU path OK for expert)
- Repo: `/opt/margo/mrlco-new`
