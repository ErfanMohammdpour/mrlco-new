# PHASE X — Long-budget PPO-from-scratch control (settles "50 iterations was too short")

## Role
You are an RL engineer running one controlled experiment whose result is pre-registered below. You work in `MARGO_BASELINE/mrlco-new`. This job runs in parallel with Phases 1–3 on Kish and must not share a run directory or GPU process with them.

## Hypothesis under test
H_budget: "PPO-based training in this codebase would beat the strong heuristics if given enough outer iterations; the 50-iteration diagnostics stopped before the policy could find sparse plans."

Evidence against, already in the ledger: `par500` collapsed at iteration 21 and stayed collapsed to 180; the old 3500-iteration run (`results/ckpt_ours_final_3500/meta_offloading20_log-inner_step1/progress.csv`) plateaued at sampled T 860–870 from iteration ~500 against publication greedy 829; `binary_lat` reached its best T (502) at iteration 5 and drifted back to 535 while MEC share went 45%→95%.
Evidence for: `lat50` was still improving at iteration 50 (958→796); every 50-iteration run was judged before plateau.

This run removes every known collapse accelerator so that, if H_budget is true, it has its best chance.

## Configuration (all deviations from the frozen v0.1 primary are deliberate and listed)
- `vocab_size=3`, V2V on, `end_token=3`, latency-only reward `R = -(ΔT / T_allMEC)` (energy off; the latency track must pass first).
- Inner loop: `k_steps=1`, **no** `select_support_rows` subsampling — use the full ~1000 trajectories per task per iteration in mini-batches of 100, one epoch. This mirrors the old repo default that produced the 3500 run.
- Outer loop: existing first-order mean pseudo-gradient + Adam 5e-4, meta-batch 10. (Reptile/Adam correctness fixes are a Phase 6 baseline item; do not change the outer update here, otherwise the comparison with the old 3500 run breaks.)
- Entropy bonus coefficient 0.01, linearly decayed to 0.001 over 500 iterations. (`entropy_coefficient=0.0` in all prior runs.)
- Target-KL early stop inside the inner update: stop the epoch if approximate KL(old‖new) > 0.02.
- PPO value clip centred on `v_old` (fix C-04 for this run and log that you did; it changes the critic loss only).
- `parallel=True` (env executor only; sampler RNG private per worker as in `samplers/vectorized_env_executor.py`).
- 500 outer iterations, seed 0. If wall-clock exceeds 20 h, stop at the last completed iteration and report it; do not shorten in advance.
- Logging every iteration: sampled train T, greedy-decode train T on a fixed 200-graph subset (add this: it did not exist in the 3500 run), occupancy mix, entropy, `max_action_frac`, `n_non` p50, approximate KL, clip fraction. Every 25 iterations: greedy-decode validation T on the five validation distributions (n=500) — reading validation is allowed here because nothing is selected on it; the run is a control.

Registration: `DIAG_PPO500_METHOD_ID = "margo_v0.3_diag_ppo500_control"`, run dir, flag `--diagnostic-ppo500`, driver `run_diagnostic_ppo500`, Kish target `ppo500`, gate isolation, unique-dir test. Provenance lists every bullet above.

## Pre-registered predictions (write them in the ledger **before** launch)
P1: `max_action_frac ≥ 0.90` (MEC) by iteration ≤ 150 and it does not recover.
P2: greedy-decode train T plateaus within ±3% of all-MEC (628) or worse; never ≤ 480.
P3: greedy-decode validation T never ≤ 575 (BC zero-shot).
If P2 is falsified (train greedy-decode T ≤ 448 = greedy_from_mec at any logged iteration), H_budget wins; Phase 0's ADR-007 must be reopened and Phases 1–3 results reinterpreted as "IL was a shortcut, not a necessity". If P1–P3 hold, H_budget is closed permanently and the negative-result contribution (C2) gains its strongest data point.

## Double-check (mandatory)
Before launch:
1. Confirm in code (file:line in the ledger) that: subsampling is disabled, entropy coefficient is non-zero and decays, KL early stop exists, value clip is `v_old + clip(v_new − v_old)`.
2. Smoke run 3 iterations; confirm greedy-decode logging on the fixed 200-graph subset works and the subset indices are saved (`fixed_train_subset.json`).
3. Confirm the job is a separate systemd unit / process from Phase 1–3 jobs and writes to its own directory.
During run (check at iterations 25, 100, 250):
4. If `max_action_frac ≥ 0.95` and entropy < 0.05 at iteration 100, do **not** kill the run; the point is to observe whether the remaining 400 iterations recover. Note the iteration in the ledger.
After run:
5. Plot four curves from the CSV: sampled T, greedy-decode train T, validation greedy-decode T, `max_action_frac`. Attach paths.
6. Compare against the old 3500 run's banded means (0–50: 1383; 200–500: 891; 500–1000: 871; 3000–3500: 860 vs greedy 829) — same units, sampled T.
7. State P1/P2/P3 verdicts explicitly with the iteration and value that decided each.
8. No model selection on validation happened (this is a control; the checkpoint is not used later).

## Report
Verdict on H_budget in one sentence, then the three predictions with evidence, then the curves. Ledger rows per README. `paper_result=false`, but this run **is** eligible to appear in the paper as the PPO-from-scratch baseline once Phase 6 reruns it with the corrected outer update and 5 seeds.
