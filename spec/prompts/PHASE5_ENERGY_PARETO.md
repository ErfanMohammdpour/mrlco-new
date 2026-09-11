# PHASE 5 — Energy on: joint objective, λ sweep, Pareto front

## Role
You are the optimization-and-evaluation engineer for MARGO (`MARGO_BASELINE/mrlco-new`). The latency track has passed its gates (Phases 1–4). You now turn energy on in the teacher, the policy, and the evaluation, and produce the latency–energy Pareto analysis that no competitor (MRLCO, DAMRL, FedMAGS, GTrXL-PR) reports.

## Fixed definitions (from `spec/OBJECTIVE_AND_ENERGY.md`, ADR-001, ADR-003)
- `E = total_mobile_joules = UE + HELPER` energy (MEC server compute excluded; documented scope).
- Normalization per graph by the three pure plans (all-UE, all-MEC, all-HELPER) as already implemented in `scheduler/reward.py` and `energy_api.py`.
- `J_λ = λ · T_norm + (1−λ) · E_norm`, λ ∈ {1.0, 0.75, 0.5, 0.25, 0.0}. λ=1.0 must reproduce the latency track exactly (regression).
- Units: seconds and joules everywhere. Report raw T and raw E alongside normalized J.

## Deliverables
1. **Teacher per λ.** `greedy_from_mec_plan` and `iterate_2opt` accept a score function; pass `J_λ` (they already take `score_fn` in `spec/twopt_expert.py`). Generate label caches `expert_2opt_J<λ>_train_p<pid>.npz` for the 13 train profiles and all five λ. Also compute the pure-plan references and store `T_allUE, T_allMEC, T_allHELPER, E_*` per graph in the cache so evaluation never recomputes them. Sanity: as λ decreases, expert mix must move toward the lower-energy modes and E must decrease monotonically in λ on average; T must increase. Paste the trend table into the ledger.
2. **Policy conditioning on λ.** One model for all λ: append λ to the resource context vector (`resource_ctx [B,5]`) and to every node row (`FEATURE_DIM` 15 → 16, stats v3). Training draws λ uniformly from the five values per batch and uses the matching labels. This is preference-conditioned imitation (Pareto-set learning, cf. PSL / multi-objective NCO). Ablation: five separate models (one per λ) on seed 0 only, to check whether conditioning costs accuracy.
3. **Adaptation with energy.** Phase 3 loss with `A^(j) = -(J_λ^(j) − b_g)`; best-so-far by `J_λ`. Run on held-out-profile validation for λ ∈ {1.0, 0.5, 0.0}.
4. **Pareto evaluation** (validation only in this phase): for each λ, report (T, E) of: all-UE, all-MEC, all-HELPER, publication greedy, greedy_from_mec(J_λ), 2-opt(J_λ), BC zero-shot, BC best-of-32, BC+EAS. Compute the non-dominated set across methods per graph and the hypervolume with reference point (T_allUE, E_allUE). Plot T vs E with one marker per method per λ, error bars over 3 seeds.
5. **Physics regression tests**: (a) `J_1.0` labels equal the Phase 4 latency labels bit-for-bit; (b) all-MEC has the lowest E and all-UE the highest E for the frozen profile on ≥ 95% of train graphs (the audit flagged this assumption as unverified; measure it and report the actual fraction — if it is lower, the normalization bounds must be min/max over the three pure plans, not assumed); (c) energy of a V2V task includes helper compute (ADR-001), assert on the toy graphs.

Registration: method ids `margo_v0.3_expert_energy`, `margo_v0.3_bc_energy`, `margo_v0.3_eas_energy`, run dirs, flags, driver, Kish targets, isolation, unique-dir test.

## Gate (validation, held-out profiles, 3 seeds)
- λ=1.0 regression: T within ±5 s of Phase 4 numbers for the same seed.
- λ=0.5: BC zero-shot `J_0.5` ≤ 1.10 × 2-opt `J_0.5`; BC+EAS ≤ 1.06 ×.
- The learned method is on or within 3% of the heuristic Pareto front at every λ (distance measured in normalized J).
- Conditioning ablation: single conditioned model within 2% J of the per-λ models; otherwise keep per-λ models and say so.

## Double-check (mandatory)
1. Test (a)–(c) pass before any GPU job.
2. Trend table (bullet 1) shows monotone E↓ and T↑ as λ decreases; a non-monotone row means the score function or the normalization is wired wrong.
3. Verify normalization denominators are never zero (graphs where two pure plans tie) — count and handle with ε, log count.
4. Confirm λ is visible to the policy (feature index asserted in a test) and that stats v3 do not overwrite v1/v2 files (hashes).
5. Cross-check five random (graph, plan) energies against a hand computation from `energy_api.py` formulas in a notebook-free script; must match to 1e-6 J.
6. Pareto/hypervolume code unit-tested on a 3-point synthetic set with a known hypervolume.
7. No meta-test access.
8. State explicitly in the ledger which energy scope is used (ADR-001) and that MEC server energy is excluded; the paper must say the same sentence.

## Report
Per λ: table of (T, E, J) for all methods with 3-seed mean±std; Pareto plot paths; hypervolume table; ablation line. Ledger rows per README. `paper_result=false` until Phase 6.
