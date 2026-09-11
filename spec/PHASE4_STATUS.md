# Phase 4 — Evaluation campaign

Status: IN PROGRESS

Parent freeze: `phase3-freeze-v0.1` (`0c776924b49da6c66c511c12a8cde70be732e25d`)
Branch: `phase4-eval`

**Do not move or rewrite `phase0-freeze-v0.1`, `phase1-freeze-v0.1`, `phase2-freeze-v0.1`, or `phase3-freeze-v0.1`.**

Phase 4 is evaluation of the frozen v0.1 config. It is not hyperparameter search.
Phase 4 closure does not exist yet. No paper figures until raw run artifacts exist.

## GPU

Default: GPU forbidden.

Train 3500 outer iterations only after **all** of:

1. explicit human approval in chat
2. CLI `--i-allow-gpu`
3. env `MARGO_ALLOW_GPU=1`

Do not start GPU from the agent without that chat approval.

Kish Ada runtime (not a paper result): `margo-phase4-tf115-nv2212` = CUDA 11.8 + `nvidia-tensorflow==1.15.5+nv22.12` + `tf.contrib`. Official `tensorflow-gpu==1.15.5` CUDA 10 cannot GEMM on sm_89.

## Frozen primary campaign

- method: `margo_v0.1_primary`
- seeds: `0 1 2 3 4`
- outer_iterations: 3500
- report `k_steps=0` and `k_steps=3` on every meta-test dist `{7,12,14,20,23}`
- validation `{2,6,10,16,17}` only for checkpoint selection
- objective `0.5/0.5`, V2V on, 20-task DAGs

Do **not** rewrite `outer_iterations: 3500` to 1000 or 200. A 5-iter learning probe (`margo_v0.1_learning_probe`), 5-iter spawn-parallel env probe (`margo_v0.1_parallel_probe`), 500-iter spawn-parallel diagnostic (`margo_v0.1_diag_500_parallel`), 50-iter latency-only `T/T_allMEC` diagnostic (`margo_v0.1_diag_latency_tmec`), 50-iter POMO+elite latency diagnostic (`margo_v0.1_diag_pomo_tmec`), 50-iter BC greedy-from-MEC then latency PPO (`margo_v0.1_diag_bc_greedy_tmec`), BC-only greedy-from-MEC eval (`margo_v0.1_diag_bc_only_eval`), BC-continue to CE plateau (`margo_v0.1_diag_bc_continue`), BC unseen greedy decode on val+meta-test (`margo_v0.1_diag_bc_unseen`), KL to frozen π_BC + critic warmup (`margo_v0.1_diag_kl_bc_ppo`), few-shot from `bc_continue` k0 vs k3 PPO vs encoder-frozen CE (`margo_v0.1_diag_bc_fewshot`), scheduled-sampling BC from `bc_continue` then unseen greedy (`margo_v0.1_diag_bc_scheduled`), CPU Hamming-2 from greedy_from_mec on 200 train graphs (`margo_v0.1_diag_hamming2_expert`), CPU motif audit of greedy_from_mec (`margo_v0.1_diag_motif_expert`), CPU iterative 2-opt teacher from greedy_from_mec (`margo_v0.1_diag_2opt_expert`), BC on 2-opt labels (`margo_v0.1_diag_bc_2opt`), frozen-encoder pair/motif head (`margo_v0.1_diag_pair_head`), pair k-sweep vs same-split oracle (`margo_v0.1_diag_pair_ksweep`), ΔT pair ranker vs CE vs true-gain rank (`margo_v0.1_diag_pair_ranker`), sequential pair refine scan/multipass/bestimp (`margo_v0.1_diag_pair_seq`), CAVIA-on-z energy-off (`margo_v0.2_diag_cavia_frozen`), CAVIA-on-z energy-on 0.5/0.5 (`margo_v0.2_diag_cavia_energy`), CAVIA-on-z strong inner lr=1e-2 k=50 (`margo_v0.2_diag_cavia_strong`), pair-sup B vs A greedy_from_mec + λ L_joint (`margo_v0.2_diag_pairsup`), CAVIA-on-z from bc_continue (`margo_v0.2_diag_cavia_bccont`), CPU motif pair frac_pos from all-MEC (`margo_v0.2_diag_pairfrac_mec`), all-MEC motif rewrite greedy+K neural (`margo_v0.2_diag_rewrite_mec`), oracle dist_id embed every decoder step (`margo_v0.2_diag_oracle_dist`), 200-iter diagnostic (`margo_v0.1_diag_200`), and optional 1000-iter diagnostic (`margo_v0.1_diag_1k`) are `paper_result=false` and write per-stage audit JSON under `runs/phase4/<method>/seed_N/audit/`. They are not the primary campaign. Frozen primary sampler stays `parallel=False`. Frozen primary objective stays `0.5/0.5`. Frozen primary inner select stays random `select_support_rows`.

Diagnostic narrative and numbers for a later report: `spec/PHASE4_DIAGNOSTIC_LOG.md` and `spec/PHASE4_DIAGNOSTIC_RESULTS.json`. Not paper figures.

## Gate

```bash
python3 spec/phase4_gate.py
```

Must print `Phase 4 campaign: PASS` and `Phase 4 closure: NOT CLAIMED`.
