# Per-instance EAS at test time (path B after ADR-011)

Date: 2026-09-09. `paper_result=false`. **Not** meta-adaptation.

## Claim (different from Phase 3)

Phase 3 failed: φ shared across a distribution does not move query greedy.
Path B: adapt φ on the **same graph** being solved (classic EAS / active search).

Compare under **equal schedule budget B**:
- best-of-B: sample B plans from frozen π, take min T
- EAS-inst: N steps × k samples, N·k = B; update φ between steps; result = best T among all B scored plans

Secondary: greedy T after φ* (extra 1 schedule).

## Locked defaults

- Ckpt: meanagg_mean seed0 (ADR-010)
- φ: `lastlayer` (only subset that moved; pg was best in Phase 3)
- Loss: `pg` first (Phase 3: pg beat pg_il/il). Optional `pg_il` later.
- Budget primary **B=32** (N=2, k=16). Also B=64 (N=4, k=16). Smoke B=8 (N=2, k=4), n_graphs=8.
- lr=1e-3, reset φ per graph, Adam state reset per graph
- Validation only. No meta-test.
- Gate (informal): mean T_best(EAS,B=32) **strictly <** mean T_best(bok,B=32) on same graphs/ckpt by ≥5s, else B fails.

## Method id

`margo_v0.3_diag_eas_inst` — isolated from Phase 3 dirs.
