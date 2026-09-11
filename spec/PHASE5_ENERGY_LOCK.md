# Phase 5 — energy Pareto (parked 2026-09-10)

Date: 2026-09-10. `paper_result=false`. ADR-001.

## Scope (must appear in paper)

`E = total_mobile_joules = UE + HELPER` (compute + radio). **MEC server compute excluded.**

## λ

`J_λ = λ T_norm + (1−λ) E_norm`, λ ∈ {1.00, 0.75, 0.50, 0.25, 0.00}.
λ=1.0 teacher scores raw T (bit-for-bit with Phase 4 2-opt). Search uses unclipped norms; report clips.

## Method ids

| id | job |
|---|---|
| `margo_v0.3_expert_energy` | CPU 2-opt per (profile, λ) |
| `margo_v0.3_bc_energy` | later: λ-conditioned BC, obs v3 |
| `margo_v0.3_eas_energy` | later: EAS on J_λ |

Unique-dir count **45**. No meta-test. Kish: `expertenergy <seed> [smoke]`.

Parked before full teacher. Smoke PASS. Full: 1/65 (`p_3_3_5` λ=1.00 match Phase 4 n=1500). Resume later; do not claim Pareto. Phase 6 without energy = latency-only paper.
