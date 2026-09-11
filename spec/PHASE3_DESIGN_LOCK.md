# Phase 3 design lock — EAS-on-φ (from Phases 0–2 evidence)

Date: 2026-09-09. `paper_result=false`.

## Why Phase 3 is the only remaining method claim

Mapped validation gap after Phases 1–2:

```
greedy meanagg+mean   ~577   (seed0; triple was ~588)
best-of-32 (triple)   ~509   (32 schedule evals; Phase 2 on triple ckpt)
pair_seq search       ~448   (heuristic ceiling, not method)
2-opt teacher         ~424
```

Encoder ablation (ADR-010): gatv2/dagformer **worse**. Gap is **not representational**.
Best-of-k: recovers mean cheaply; oracle-within-samples **2.2%** → π has almost **no mass** near 2-opt on OOD. Dup 0.70 → peaked wrong mode; temp helps a little.
PPO-on-θ / CAVIA-on-z already dead (ADR-007).

So method paper needs: **adaptation that pulls mean toward the policy's own good samples** without occupancy collapse. That is exactly EAS (Hottung et al.) with IL-to-best + PG with shared baseline on a **small** φ.

## Locked inputs

- Ckpt: `runs/phase4/margo_v0.3_diag_encoder/meanagg_mean/seed_0/ckpt/bc_core.ckpt` (ADR-010).
- Encoder `meanagg`, readout `mean`. Identity greedy val seed0 ≈ **577.02** (±0.1).
- Splits: validation dists `{2,6,10,16,17}`; support 20 / query 80 via `support_query_indices`; query **never** in update loop.
- Phase 2 best-of-32 was on **triple** ckpt. For fair "not just best-of-k" check: also report zero-shot best-of-32 on **mean** ckpt (quick), then after EAS.

## Locked algorithm (from PHASE3 prompt — do not simplify)

φ ∈ {film, emb, lastlayer}; default main = **`lastlayer`** first for smoke (smallest change), then **`film`** as architecture candidate (γ=1+Δγ(z), β=Δβ(z), z∈R^32, identity at z=0).

Per dist d:
1. reset φ to identity; Adam lr=1e-3
2. best[g] ← greedy
3. for it in 1..N (100; log 0,5,10,20,50,100):
   - sample k=16 plans; schedule each; POMO baseline mean_j T; A = -(T-b)/T_allMEC
   - update best if improved
   - L = L_PG + λ_IL * L_IL ; λ_IL ∈ {0.5,1.0,2.0} sweep
4. eval query: greedy + best-of-32; mix; n_non

No entropy, no critic, no clip. schedule() count per iter == 20*k.

## Gate (must tick)

- Query greedy ≤ **520** (from ~577). Occupancy Local 0.15–0.25, V2V 0.02–0.08, n_non p50 ∈ {4,5,6}.
- Also report best-of-32 after φ* vs Phase-2-style zero-shot best-of-32 on **same** mean ckpt.
- Ablation PG-only worse by ≥10s or anchor claim dies.
- Fail → ADR-007 fallback: method = BC + best-of-k; write ADR-011 either way.

## Risk

If φ adaptation flat like CAVIA: paper loses meta claim. Controls (PG-only, IL-only, full-θ) make that result publishable as negative.
