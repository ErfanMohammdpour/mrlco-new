# PHASE 3 — EAS-style support-set adaptation of a small parameter subset (the v0.3 adaptation engine)

## Role
You are a reinforcement-learning researcher who has implemented Efficient Active Search (Hottung, Kwon & Tierney, ICLR 2022) and POMO-style shared baselines before. You work in `MARGO_BASELINE/mrlco-new` (TF 1.15). You replace the failed adaptation engines (inner PPO on θ; CAVIA-on-z with a pure `mean T` loss) with an adaptation rule that has an explicit anchor to the policy's own best samples. Every design choice below is motivated by a specific prior failure; do not "simplify" any of them away.

## What failed and why (read `spec/PHASE4_DIAGNOSTIC_LOG.md` sections 4, 7b, 9, 19, 20, 22)
- PPO on all θ (`bc50`, `kl_bc_ppo`, `bc_fewshot k3`): mean policy gradient with a weak/cold critic drove occupancy from ≈0.20/0.75/0.05 (Local/MEC/V2V) to either all-MEC or Local-heavy within 5–15 updates; `n_non` p50 4 → 7.8. Soft KL to π_BC at β=0.1 did not hold.
- CAVIA on z with `L_z = mean T` (three runs): gradient through the sampled objective only; z moved (`z_l2` 0.43–0.67) but query T did not (±1 s). No anchor to good samples; no per-instance baseline.
- Conclusion: adaptation must (a) touch few parameters, (b) use a per-instance baseline so the gradient carries plan-level credit, and (c) include an imitation term toward the best plan found so far, so the mean is pulled toward the tail instead of toward the marginal mode.

## Design (EAS-Lay / EAS-Emb transplanted to our macro-plan setting)
Adapted subset φ (choose via flag `adapt_subset`):
- `film`: FiLM (γ, β) on the LSTM initial state `s` (spec §6; γ←1, β←0 init so step 0 ≡ BC). ≈ 2×128×(d_z) params with d_z=32 → also a free `z ∈ R^32` per distribution.
- `emb`: a residual additive vector `[20,256]` on node embeddings h (EAS-Emb analogue), initialized zero, shared across the distribution's graphs by decoder index.
- `lastlayer`: the output Dense (128→3) of the decoder.
Everything else frozen (`tf.stop_gradient` or variable filtering, verify with `tf.trainable_variables()` count).

Per distribution d in validation, support set S_d (20 graphs, from `split_loader`), query Q_d (80 graphs, never used for updates):
```
reset φ to identity, fresh Adam(lr=1e-3)
best_d[g] ← greedy plan and its T for every support graph g
for it in 1..N_adapt (N_adapt=100, log at {0,5,10,20,50,100}):
    for each support graph g:
        sample k=16 plans a^(1..k) ~ π_{θ,φ}(·|g); T^(j) = schedule(g, a^(j))
        b_g = mean_j T^(j)                      # POMO shared baseline, no critic
        A^(j) = -(T^(j) - b_g) / T_allMEC(g)    # normalized advantage
        update best_d[g] if min_j T^(j) < T(best_d[g])
    L_PG  = - mean_{g,j} A^(j) · log π_{θ,φ}(a^(j)|g)
    L_IL  = - mean_g log π_{θ,φ}(best_d[g] | g)      # imitation of best-so-far (EAS term)
    L     = L_PG + λ_IL · L_IL,   λ_IL ∈ {0.5, 1.0, 2.0}  (sweep on validation)
    φ ← Adam step on ∇_φ L
report query Q_d: greedy T and best-of-32 T with φ*; occupancy mix; n_non_p50
```
No entropy bonus. No critic. No clipping (single-step on fresh samples each iteration, so on-policy). If wall-clock per distribution exceeds 10 minutes, reduce k to 8 and report it.

Ablations (all validation, seed 0 unless stated; the main config on seeds {0,1,2}):
1. `L_PG` only (no IL) — expected to drift like PPO; proves the anchor matters.
2. `L_IL` only — proves PG adds something beyond self-imitation.
3. CE-to-2-opt-labels on the same φ subset, same N_adapt — the "few-shot IL with search labels on support" alternative (`spec/bc_fewshot.py::k3_ce` was CE on greedy labels with frozen encoder and 3 steps; this is the corrected version).
4. Full-θ variant of the main loss (same anchor, all parameters) — isolates "subset" from "anchor".
5. Existing negatives are already in the ledger: PPO on θ (`bc_fewshot`), CAVIA-on-z (`cavia_*`); cite, do not rerun.

## Registration
`DIAG_EAS_METHOD_ID = "margo_v0.3_diag_eas"`, run dir `runs/phase4/margo_v0.3_diag_eas/<subset>_<loss>/seed_<s>/`, flag `--diagnostic-eas --adapt-subset X --loss {pg_il,pg,il,ce2opt} --lambda-il L --n-adapt N`, driver `run_diagnostic_eas`, Kish target `eas`, gate isolation, unique-dir test. Module `spec/eas_adapt.py`; reuse `spec/cavia_loop.py::_schedule_batch`, `sample_actions_and_neglogp`, `greedy_actions`. Provenance: ckpt sha256, subset, n_trainable_params_adapted, λ_IL, k, N_adapt, lr.

Unit tests `test_eas_adapt.py`: (a) adapted variable list equals the intended subset exactly (names asserted); (b) at iteration 0 with identity φ, greedy query T equals the Phase 1/2 zero-shot number within 0.1 s (identity check); (c) `L_IL` gradient is zero when `best_d[g]` equals the argmax plan and π is one-hot (sanity on a toy); (d) advantage normalization is per-graph (mean over j is zero to 1e-6).

## Gate (validation, main config, mean of 3 seeds)
- Query greedy T: 575–582 → **≤ 520** at N_adapt ≤ 100. Also report best-of-32 with φ*.
- Occupancy of query plans stays within Local 0.15–0.25, V2V 0.02–0.08; `n_non` p50 in {4,5,6}. If T improves but occupancy leaves this band, report it as "different basin", not success.
- Adaptation curve must be monotone-ish: T at 20 ≤ T at 5 ≤ T at 0 on ≥ 4 of 5 distributions. Non-monotone curves indicate drift → tune λ_IL up, not lr up.
- Ablation 1 must be worse than main by ≥ 10 s or the anchor claim is unsupported.
- Ablation 4 (full θ) worse than main → "subset" is a real design element; equal → drop the subset claim and keep the anchor claim.
Fail → the v0.3 method is zero-shot BC + best-of-k (Phases 1–2), ADR-007 fallback; the paper's meta-adaptation claim is removed. Record either way in `spec/decisions/ADR-011-adaptation-result.md`.

## Double-check (mandatory)
Before launch:
1. Trainable-variable audit printed and asserted in code: exactly the subset, count logged.
2. Identity check (test b) passes on Kish with the actual checkpoint.
3. Baseline math: verify `sum_j A^(j) == 0` per graph in a smoke run; verify `T_allMEC(g)` is taken from refs cache, not recomputed each iteration.
4. Support/query disjointness: assert set intersection of graph indices is empty for every distribution, from `split_loader`, and log the sha256 of both index lists.
5. Smoke: 1 distribution, N_adapt=5, k=4; check `L_PG`, `L_IL` finite, φ norm grows from 0, no NaN.
After runs:
6. For each distribution and seed, paste the {0,5,10,20,50,100} curve for query greedy T and support best-so-far T into the ledger. Support best-so-far must be non-increasing by construction; if it is not, the bookkeeping is wrong.
7. Confirm query graphs were never scheduled inside the update loop: count `schedule()` calls per iteration and assert `== 20·k` (support only).
8. Check that the gain is not just best-of-k in disguise: compare query greedy T after adaptation with the Phase 2 best-of-32 **before** adaptation. Report both.
9. No meta-test access.
10. Re-read this file's gate and tick each item explicitly in the ledger.

## Report
Per subset × loss: table of query T at {0,20,100}, best-of-32 T, mix, n_non, evals per distribution, minutes. Curves for the main config on all five validation distributions. ADR-011 written. `paper_result=false`.
