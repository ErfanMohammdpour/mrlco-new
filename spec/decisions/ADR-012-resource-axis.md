# ADR-012 — Resource-profile meta-task axis

Date: 2026-09-09. Status: **accepted (in progress)**. `paper_result=false`.

## Context

Phases 1–3 used one frozen resource profile. Dist-level φ adaptation failed (ADR-011); per-instance EAS ≈ best-of-k under equal budget. Scenario latents (rates, CPU ratio) are the axis where context meta-RL is testable vs MRLCO/DAMRL.

## Decision

Extend meta-task to `(DAG distribution, resource profile)`.

### Conversion

```
bytes_per_second = declared_mbps * 1024 * 1024 / 8
```

7 Mi-Mbps → **917504** B/s; 5 → **655360** B/s. UE CPU fixed **1048576** B/s; MEC CPU = ratio × UE.

### Grid

- MEC UL=DL ∈ {3,5,7,9,11} Mi-Mbps
- V2V ∈ {3,5,7} Mi-Mbps
- MEC/UE CPU ratio ∈ {5,10,20}
- Total **45** profiles. Frozen = `frozen_7_5_10`.

### Role lists (from `spec/resource_profiles.yaml`)

**meta_train (13):** UL∈{3,7,11} × V2V∈{3,7} × ratio∈{5,20} (12) + frozen.

**validation_heldout (2):** `p_5_5_10`, `p_9_5_10` (interpolation).

**meta_test_heldout (8):** UL∈{5,9} × V2V∈{3,7} × ratio∈{5,20}. Phase 6 only.

### Task counts

- Train: 15 dists × 13 profiles = **195**
- Val: 5 dists × 2 held-out + 5 dists × frozen = **15** tasks (10 held-out + 5 continuity)
- Meta-test: 5 × 8 = **40** (Phase 6)

### Artifacts

- YAML: `spec/resource_profiles.yaml`
- Loader: `spec/resource_profiles.py`
- YAML sha256: computed at write time in tests / provenance

## Consequences

1. Regenerate 2-opt teachers per train profile.
2. Obs v2: FEATURE_DIM 11→15, PACKED_DIM 50→54; stats `encoder_feature_stats_v2.json` (v1 untouched).
3. BC retrain on 195 tasks; **one-sided continuity**: frozen val T ≤ Phase-1 + 10s (improvement allowed). Seed0 two-sided `|Δ|≤10` was a false FAIL (493.8 vs 577.0, token 0.72 vs 0.65).
4. Re-test adaptation on held-out profiles (EAS + context-only).
5. Method ids: `margo_v0.3_expert_profiles`, `margo_v0.3_bc_profiles`, `margo_v0.3_eas_profiles`, `margo_v0.3_ctx_profiles`, `margo_v0.3_bok_profiles`.
