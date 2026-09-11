# PHASE 2 — Best-of-k neural inference (sampling from π, one `schedule()` per sample)

## Role
You are an ML engineer with neural-combinatorial-optimization background working in `MARGO_BASELINE/mrlco-new`. You add a standard NCO inference protocol (POMO / PolyNet style multi-sample decoding) to the frozen BC policy and quantify the mean-vs-tail gap. No training in this phase.

## Why this phase exists
The sampled-plan tail of the BC policy is already expert-level while its mean is not: at `bc50` iteration 0, T_p10 = 445 vs mean 668 (validation-like graphs); `bc_only_eval` argmax p10 = 320 vs mean 533 on train. Drawing k plans from π and keeping the best by `schedule()` is standard neural inference in the NCO literature (POMO 8×N augmentations, PolyNet 64×n samples, AM sampling). It is **not** handcrafted neighbourhood search (ADR-008). This phase measures how much of the OOD gap (val 575–582 vs 2-opt 424) is recoverable by sampling alone, and at what eval cost.

## Inputs
- Checkpoint: the Phase 1 winner (or `runs/phase4/margo_v0.1_diag_bc_2opt/seed_0/ckpt/bc_core.ckpt` if Phase 1 has not decided). Record ckpt path and sha256 in provenance.
- Splits: train (n=1500, for sanity) and validation (n=500). Meta-test untouched.
- Scheduler: `schedule_via_adapter` (`env/mec_offloaing_envs/scheduler/adapter.py`), latency-only objective, frozen resources (`spec/frozen_experiment.yaml`).

## Implementation
New module `spec/best_of_k.py`:
1. `sample_plans(sess, policy, obs, k, temperature)` → `[B,k,20]` using the existing `FixedSequenceLearningSampleEmbedingHelper` path (`model="sample"`); temperature applied to logits before `Categorical`; temperature 1.0 default; also support `top_p=None`.
2. `greedy_plan(sess, policy, obs)` → `[B,20]` (existing greedy path) always included as sample 0 so best-of-k ≥ greedy by construction.
3. `score_plans(graphs, plans, resources)` → `[B,k]` makespan seconds via one `schedule_via_adapter` per plan. Cache refs per graph (all-UE / all-MEC) as `cavia_loop._schedule_batch` does.
4. Metrics per graph: `T_greedy`, `T_best_k`, `T_mean_k`, `T_p10_k`, argmin index, whether best plan equals greedy, `n_non` of best plan, mix of best plan, duplicates ratio among the k samples.
5. Sweep `k ∈ {1, 4, 8, 16, 32, 64}` and `temperature ∈ {0.7, 1.0, 1.3}` (temperature sweep only at k=32). Seed the sampler with `np.random.RandomState(seed)` and TF seed; three seeds.
6. Also compute the **oracle within samples**: fraction of graphs where any of the k samples beats the 2-opt label (cached `expert_2opt_mec_validation.npz`). This bounds what Phase 3's imitation-to-best anchor can exploit.

Registration: `DIAG_BOK_METHOD_ID = "margo_v0.3_diag_bestofk"`, `diag_bestofk_run_dir(seed)`, flag `--diagnostic-bestofk`, driver `run_diagnostic_bestofk`, Kish target `bestofk`, gate isolation, unique-dir test, unit tests `test_best_of_k.py`: (a) sample 0 equals greedy; (b) `T_best_k` is monotone non-increasing in k on a fixed seed; (c) scoring the hand-built plans in `spec/toy_oracles/*.yaml` (chain / UE→MEC / UE→HELPER / MEC→UE / MEC→HELPER cases) reproduces the makespans recorded in those yaml files.

## Gate (validation)
- Report `T_best_32` and `T_best_64`. Target: val `T_best_32 ≤ 510` (from 575–582). Reference: 2-opt 424, all-MEC 634.
- Also report evals-per-graph = k and wall-clock per graph. Compare to `pair_seq` bestimp k20 (447.9 at ≈718 evals) to show the cost curve; make clear in the ledger that pair_seq is a heuristic reference, not the method.
- If `T_best_64` improves < 20 s over greedy, the tail claim from `bc50` does not transfer to greedy-trained BC on OOD; state that and pass the finding to Phase 3 (the imitation anchor then has little to anchor to).

## Double-check (mandatory)
Before launch:
1. Unit tests pass locally (numpy parts) and on Kish (TF parts).
2. Sanity on train n=200: `T_best_1` equals the Phase 1 greedy number for the same ckpt within 0.1 s. If not, the sampling path and the greedy path disagree on obs ordering or padding; fix before continuing.
3. Confirm `end_token=3` is never emitted in samples (assert max token == 2); confirm fixed length 20.
After runs:
4. Monotonicity in k holds per seed; if any k has a higher mean than k−1 on the same seed, the RNG is not nested (samples for k must be a prefix of samples for k+1); fix and rerun.
5. Duplicates ratio at k=64 reported; if > 0.5 the policy is near-deterministic and temperature 1.3 must be included in the report.
6. Per-distribution breakdown on validation (five distributions) — fat distribution 10 must be shown separately (BC 794 vs expert 581 in `bc_2opt`).
7. No meta-test access (`rg` on logs).
8. Cross-check three random graphs by recomputing the best plan's makespan with `spec/twopt_expert.py`'s score function; must match to 1e-6.

## Report
Curves T_best_k vs k (three seeds, mean±std), table at k=1/8/32/64 with evals and seconds per graph, oracle-within-samples fraction, per-distribution table. Ledger rows per README. `paper_result=false`.
