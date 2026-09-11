# ADR-011 — Phase 3 adaptation result (EAS-on-φ)

Date: 2026-09-09. Status: **accepted (negative)**. `paper_result=false`.

## Context

Phases 0–2 locked: PPO-on-θ / CAVIA-on-z dead (ADR-007); OOD gap not representational (ADR-010, meanagg+mean); best-of-k recovers mean (~509) not 2-opt (~424). Phase 3 tested EAS-style support adaptation of a small φ shared across a distribution (20 support / 80 query), aiming query greedy ≤520 from ~577.

## Decision

**Dist-level φ adaptation FAILS the Phase 3 gate.** Meta-adaptation claim for v0.3 is **removed**.

Fallback method (ADR-007 path):

```
zero-shot BC (meanagg + readout=mean) + best-of-k inference
```

No few-shot / support-set weight update in the paper method claim.

## Evidence (validation, seed 0, N=100, k=16, λ_IL=1, lr=1e-3, ckpt meanagg_mean)

| config | query greedy T* | Δ vs T0≈577.7 | gate ≤520 |
|---|---|---|---|
| lastlayer + pg_il (main) | 576.4 | −1.2 | FAIL |
| lastlayer + il | 577.9 | +0.2 | FAIL |
| lastlayer + pg | **571.2** | **−6.5** | FAIL (best) |
| film + pg_il | 577.7 | 0 | FAIL (`phi_l2=0` all dists — dead φ) |
| full θ + pg_il | 580.7 | +3.0 | FAIL (worse than subset) |

- Support best-so-far always non-increasing (bookkeeping OK); search finds good samples (~463 on d2).
- Query greedy flat/noisy; curves not monotone.
- Mix stayed in band (IL/pg_il/full) — anchor prevents PPO-style collapse but does not move mean.
- Ablation gate "PG-only worse by ≥10s" **not** met: PG-only was *best*. Main pg_il ≈ IL. PG term scaled by 1/T_allMEC ≈1/600 so ~100× weaker than IL under λ=1.
- Ablation "full worse than subset" **holds** for pg_il (580.7 > 576.4) — but neither is useful.
- Identity at φ=0: mean 577.27 vs Phase-1 577.02 (tol 0.5).

Consistent with §25 `oracle_no_gain`: any φ **shared across graphs of a dist** is dist-level; the 577→439 gap is instance-level joints. Support-best signal does not transfer to held-out query via shared head.

## Path B result (2026-09-09)

Per-instance EAS vs best-of-k, equal budget B=32, lastlayer+pg, meanagg_mean ckpt, n=500 val:

| | mean T |
|---|---|
| best-of-32 | **498.8** |
| EAS-inst | **498.6** |
| Δ | −0.2 |
| frac EAS better | 0.29 |
| gate (≥5s better) | **FAIL** |

Verdict: gradient search ≈ sampling under same schedule budget. Not a method claim. Artifacts: `eas_inst_b32_s0_eval.json`. Proceed Phase 4 resource axis (ADR-012).

1. Do **not** advertise EAS-on-φ / CAVIA / inner PPO as the v0.3 method.
2. Paper method story = BC basin + sampling inference (Phase 2 numbers), plus negative results (occupancy collapse under PPO; dist-level adapt no greedy gain).
3. Optional follow-ups (not method until gated):
   - **A.** Stop adaptation track; proceed Phase 4 scenario / Phase 5 energy on BC+best-of-k stack.
   - **B.** Per-instance EAS at test time (φ adapted on the query graph itself; budget-matched to best-of-k). Different claim: "gradient search vs sample search", not meta.
   - **C.** Fix film bug (`phi_l2=0`) only if B needs FiLM; not needed for ADR closeout.

## Artifacts

- `spec/kish_log_archive/eas_lastlayer_pgil_s0_eval.json`
- `eas_lastlayer_il_s0_eval.json`, `eas_lastlayer_pg_s0_eval.json`
- `eas_film_pgil_s0_eval.json`, `eas_full_pgil_s0_eval.json`
- Ledger §30; `spec/PHASE3_DESIGN_LOCK.md`

## Gate checklist (Phase 3 prompt)

- [x] Query ≤520 — **FAIL** (best 571)
- [x] Occupancy band — PASS on reported runs
- [x] Monotone curve ≥4/5 dists — **FAIL**
- [x] PG-only worse than main by ≥10s — **FAIL** (PG better)
- [x] Full-θ vs subset — full worse; subset claim weak because both fail gate
- [x] ADR-011 written either way
