# ADR-009: Meta-task axes

Status: Accepted  
Decision date: 2026-09-08

## Decision

**Current axis (frozen, Phases 1–3):** a meta-task is a DAG family (fat × density cell). Resources are the single frozen profile in `spec/frozen_experiment.yaml`. Split: `latin_grid_holdout_v1`.

**Second axis (Phase 4):** a meta-task is `(DAG distribution, resource profile)`. Profiles vary MEC UL/DL, V2V rate, and MEC/UE CPU ratio (`spec/prompts/PHASE4_SCENARIO_AXIS.md`). Hold out interpolation values, same convention as MRLCO transmission-rate experiment and DAMRL bandwidth sweep.

No third axis (vehicle count, multi-helper) in v0.3.

## Why context-vector adaptation had nothing to infer

CAVIA/PEARL treat the meta-task as a latent scenario. Under **fixed** CPU and channel rates, every Kish distribution shares the same `ResourceConfig`. The only shift is DAG topology, which is **per-instance** (which tokens to flip on this graph). A per-distribution `z ∈ R^32` cannot name those joints.

Evidence:

- Mix and `n_non` already transfer OOD (`bc_unseen` log §8: mix val **0.193/0.770/0.037**, `n_non` p50 **4**). Token identity and T do not (val T **575** vs expert **464**).
- Inner CE on 20 support graphs of the target dist ≈ k0 (`bc_fewshot` k3_ce, log §9, `ce_helps=False`).
- Oracle dist_id lookup: val **584.2** vs control A **575**, verdict `oracle_no_gain` (log §25). Dist identity is not the 575→439 gap.
- Three CAVIA-on-z runs: query T flat within ±1 s (ADR-007).

So a free `z` on LSTM init is the wrong variable for axis 1. Phase 3 therefore adapts φ from **support instances** (EAS), not from a scenario embedding.

Axis 2 (rates, CPU ratio) **does** create a scenario latent. There, a context encoder (Phase 4, gradient-free) is a fair DAMRL comparison. It is not a relaunch of CAVIA-on-z with `mean T` on axis 1.

## Consequence

- Do not claim PEARL/CAVIA context inference on the current split.
- Phase 4 must regenerate teachers per profile; 2-opt and greedy_from_mec depend on `ResourceConfig`.
- Continuity gate: frozen-profile validation T within ±10 s of Phase 1 after adding resource features.
