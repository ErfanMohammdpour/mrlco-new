# PHASE 4 — Add a resource-profile meta-task axis (bandwidth / CPU), so few-shot adaptation has a latent to infer

## Role
You are the systems-and-data engineer for MARGO (`MARGO_BASELINE/mrlco-new`). You extend the meta-task definition from "DAG family only" to "DAG family × resource profile", regenerate teachers, retrain the BC policy, and re-run the Phase 3 adaptation gate on the new axis. This phase is where MARGO becomes directly comparable to MRLCO (transmission-rate experiment, 4–22 Mbps) and DAMRL (bandwidth [2,3]…[10,11] MHz, vehicle count).

## Why this phase exists
All 26 diagnostics used one frozen resource profile (`spec/frozen_experiment.yaml`: UE/HELPER CPU 1048576 B/s, MEC CPU 10485760 B/s, MEC UL/DL 917504 B/s, V2V 655360 B/s). Under fixed resources the only shift is DAG structure, which is per-instance and combinatorial; a per-distribution context vector had nothing to infer (three CAVIA runs, zero effect). Scenario-level latents (rates, CPU ratio) are exactly what context-based meta-RL was built for, and every competitor evaluates on them. Without this axis the "meta" in MARGO is not testable against DAMRL/MRLCO.

## Deliverables
1. **Resource profile grid** in `spec/resource_profiles.yaml`:
   - MEC UL=DL ∈ {3, 5, 7, 9, 11} Mi-Mbps → bytes/s (7 is the frozen value)
   - V2V ∈ {3, 5, 7} Mi-Mbps (5 is frozen)
   - MEC CPU / UE CPU ratio ∈ {5, 10, 20} (10 is frozen)
   Total 45 profiles. Meta-train profiles: all with UL∈{3,7,11}, V2V∈{3,7}, ratio∈{5,20} (12) plus the frozen profile (13). Held-out profiles for validation: UL∈{5,9}, V2V=5, ratio=10 (2). Held-out for meta-test: UL∈{5,9}, V2V∈{3,7}, ratio∈{5,20} (8, none seen in train). Write the exact lists and a sha256 of the yaml into `spec/decisions/ADR-012-resource-axis.md`.
   Design rule: hold out **interpolation** profiles (between seen values), the same convention as MRLCO's unseen rates {5.5, 8.5, 11.5} inside 4–22.
2. **Meta-task = (DAG distribution, resource profile).** Extend `spec/split_loader.py` with `iter_meta_tasks(role)` yielding `(dist_id, profile_id)`; support/query graph indices unchanged per dist. Train tasks: 15 dists × 13 profiles = 195. Validation tasks: 5 dists × 2 profiles = 10 (on held-out profiles) **plus** the 5 dists on the frozen profile (to keep continuity with Phases 1–3). Meta-test: 5 dists × 8 profiles = 40 (Phase 6 only).
3. **Observation extension** (`encoder_obs.py`): append a per-graph resource vector to every node row: `[log UL_bytes/s, log V2V_bytes/s, log MEC_CPU, log UE_CPU]` z-scored with stats from the meta-train profile set only; `FEATURE_DIM` 11 → 15, `PACKED_DIM` 50 → 54. Add `spec/fit_encoder_stats.py` support and regenerate `spec/encoder_feature_stats.json` as a **new file** `encoder_feature_stats_v2.json` (keep the old one for reproducing Phases 1–3). Also emit the same 4-vector as `resource_ctx [B,4]` for models that want it at the readout.
4. **Teachers per profile.** `greedy_from_mec_plan` and `iterate_2opt` depend on resources; regenerate label caches per (dist, profile) for train: `runs/phase4/expert_2opt_mec_train_p<pid>.npz`. CPU cost: 2-opt was ≈ 1500 graphs in minutes on Kish CPU; 13 profiles ≈ 13× that. Run as one CPU unit `expert-profiles`; record per-profile summary (greedy T, 2-opt T, all-MEC T, mix). Sanity: for UL=11 the expert mix must shift toward MEC; for UL=3 toward Local/V2V. If not, the resource injection into `ResourceConfig` is wrong.
5. **BC retrain** with the Phase 1 winner encoder on the 195 train tasks (labels = 2-opt per profile). Same recipe as Phase 1 otherwise. Seeds {0,1,2}. Evaluate greedy decode on: frozen-profile validation (continuity: must be within ±10 s of Phase 1 result for that encoder), held-out-profile validation (10 tasks).
6. **Adaptation on the new axis.** Re-run the Phase 3 main config (subset + `pg_il`) on the 10 held-out-profile validation tasks. Also run a **context-only** variant: FiLM `z` inferred by a permutation-invariant encoder from the support set's `(obs, plan, T)` tuples (PEARL-style deterministic context, no VAE) trained jointly during BC as an auxiliary head — gives a gradient-free adaptation baseline comparable to DAMRL. Compare: zero-shot, EAS-adapted, context-only, and both.

Registration for every job: unique method ids `margo_v0.3_expert_profiles`, `margo_v0.3_bc_profiles`, `margo_v0.3_eas_profiles`, `margo_v0.3_ctx_profiles`, run dirs, flags, driver functions, Kish targets, gate isolation, unique-dir test. Unit tests: (a) profile yaml → `ResourceConfig` round trip; (b) obs resource features change when the profile changes and are constant across nodes of one graph; (c) `iter_meta_tasks` counts 195/15/40 and no profile id leaks between roles; (d) old stats file untouched (hash asserted).

## Gate (validation, held-out profiles)
- Zero-shot BC on held-out profiles ≤ 1.10 × its own train T on the nearest seen profile (interpolation works) — else the resource features are not being used; check feature scaling.
- EAS adaptation on held-out profiles: ΔT ≥ 15 s vs zero-shot, occupancy band as in Phase 3.
- Context-only: any ΔT ≥ 10 s means the scenario latent is real and inferable; record it, because it enables a gradient-free deployment story parallel to DAMRL's.
- Continuity: frozen-profile validation within ±10 s of Phase 1 for the same encoder and seed. A regression here means the extra features hurt; try `resource_ctx` at readout only instead of per-node concat.

## Double-check (mandatory)
1. Every bytes/s value in `resource_profiles.yaml` recomputed by hand from Mi-Mbps in the ADR (7 Mi-Mbps = 7·1048576/8 = 917504 B/s; keep this exact convention).
2. `ResourceConfig` for the frozen profile id equals `frozen_experiment.yaml` field by field (test).
3. Expert sanity trend (bullet 4) checked and pasted into the ledger as a small table: profile → mix.
4. z-score stats v2 computed on meta-train profiles only; assert the held-out profile values are inside the training range (interpolation) — log min/max.
5. Support/query disjointness re-asserted per (dist, profile).
6. Continuity gate checked first; if it fails, stop and fix before running adaptation.
7. No meta-test profiles or distributions touched (grep logs for the 8 held-out profile ids).
8. Re-read ADR-012 and confirm the profile lists in code match it exactly (generate the lists from the yaml in the test rather than hard-coding).

## Report
Table: profile → expert greedy / 2-opt / all-MEC / expert mix. Table: validation held-out profiles → zero-shot / EAS / context / both, T and mix, per seed. Continuity check line. ADR-012 written; ADR-011 updated with the scenario-axis result. `paper_result=false`.
