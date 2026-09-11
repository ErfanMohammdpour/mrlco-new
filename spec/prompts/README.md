# MARGO v0.3 roadmap prompts

One self-contained prompt per phase. Each prompt is written to be pasted into a fresh agent session with no prior memory. Every prompt ends with a mandatory double-check section; do not skip it.

| File | Phase | Runs on | Blocks |
|---|---|---|---|
| `PHASE0_PRUNE_AND_FREEZE.md` | Decisions, ledger, dead-code isolation | local | everything |
| `PHASE1_ENCODER_ABLATION.md` | MeanAgg vs GATv2 vs DAG-Transformer, triple-readout ablation, same BC-2opt recipe | Kish GPU | 2, 3 |
| `PHASE2_BEST_OF_K_INFERENCE.md` | k-sample decoding + one `schedule()` per sample | Kish GPU/CPU | 3 |
| `PHASE3_EAS_SUPPORT_ADAPTATION.md` | EAS-style adaptation of a small parameter subset on the support set | Kish GPU | 4 |
| `PHASEX_PPO500_CONTROL.md` | Long-budget PPO-from-scratch control (settles the "50 iterations was too short" hypothesis) | Kish GPU, parallel to 1–3 | 6 |
| `PHASE4_SCENARIO_AXIS.md` | Add a resource-profile meta-task axis (bandwidth / CPU) | local + Kish | 5 |
| `PHASE5_ENERGY_PARETO.md` | Energy on, λ sweep, Pareto front | Kish | 6 |
| `PHASE6_EVALUATION_RELEASE.md` | Baseline suite, 5 seeds, CIs, artifacts, paper alignment | Kish + local | paper |

## Shared conventions (apply to every phase)

- Repository: `MARGO_BASELINE/mrlco-new`. Remote host: `kish-ai`, repo at `/opt/margo/mrlco-new`, logs at `/opt/margo/logs/`, launcher `spec/kish_gpu.sh`, image `margo-phase4-tf115-nv2212`. TensorFlow 1.15, Python 3.6 on the host image; local machine uses `python3` for pure-numpy tests only.
- Every job gets a unique `method_id` and a unique run directory under `runs/phase4/<method_id>/seed_<s>/`. Register it in `spec/phase4_campaign.py` (`*_run_dir` helper + CLI flag), `spec/phase4_train_driver.py` (`run_*` function), `spec/kish_gpu.sh` (target), `spec/phase4_gate.py` (isolation check), and `env/mec_offloaing_envs/scheduler/tests/test_phase4_campaign.py` (unique-dir count).
- Every job writes `provenance.json` with all hyperparameters and `paper_result=false` until Phase 6 declares otherwise.
- Ledger: append one row to `spec/PHASE4_WORKLOG.md`, update `current` in `spec/PHASE4_WORKLOG.json`, and add a narrative section to `spec/PHASE4_DIAGNOSTIC_LOG.md`. Never rewrite freeze tags `phase0-freeze-v0.1` … `phase3-freeze-v0.1`.
- Split is frozen: `latin_grid_holdout_v1`, meta_train `{1,3,4,5,8,9,11,13,15,18,19,21,22,24,25}`, validation `{2,6,10,16,17}`, meta-test `{7,12,14,20,23}`; support 20 / query 80 per distribution via `spec/split_loader.py` (`stratified_sha256_rank_v1`). Query graphs are never used for adaptation or model selection.
- Reference numbers (validation n=500 / meta-test n=500 unless stated): all-MEC 634 / 631; publication greedy ≈594 (train); greedy_from_mec 464 / 444; 2-opt 424 / 412; BC-2opt greedy decode 581.8 / 560.8; bc_continue greedy decode 575 / 557. Train n=1500: expert greedy_from_mec 447.9, 2-opt 414.1, all-MEC 628.
- Units are seconds and joules. Never label anything "ms".
- The meta-test split is touched only in Phase 6. Phases 1–5 select on validation.
- Do not start the frozen `margo_v0.1_primary` 3500-iteration run in any phase.
- Report format for every run: `{method_id, seed, split, n_graphs, T_mean, T_p50, T_p10, mix_local/mec/v2v, n_non_p50, token_acc_vs_expert (if labels exist), evals_per_graph, wall_clock_s}` as JSON in the run dir plus a 10-line summary in the ledger.
