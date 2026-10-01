# MARGO automotive training-integration contract

Frozen dataset: **MARGO-AUTOMOTIVE-MC-v1** (release `abe432163ad7df7263339130b48f20fbce3787b1`).
Checkout under test: `/opt/margo/mrlco-new-6b` (never `/opt/margo/mrlco-new`).
Nothing in `env/mec_offloaing_envs/data/automotive_mc_v1/` or `spec/automotive_mc_v1/`
was modified or regenerated.

## Primary flow (unchanged identity)

```
spec/automotive_gpu_smoke.py
 -> spec/phase4_train_driver.py : run_automotive_gpu_smoke / run_automotive_primary_seed
    -> _train(dataset="automotive_mc_v1")
       -> meta_trainer.build_frozen_primary_stack(dataset="automotive_mc_v1")
          -> spec/automotive_training/automotive_primary.build_automotive_primary_stack
             -> MetaSeq2SeqPolicy + Seq2SeqMetaSampler + MRLCO + Trainer.train()
```

Diagnostics (2-opt, BC, pair search/ranker, EAS, CAVIA, best-of-k) are untouched and
are never on this path.

## What the integration changes

| axis | value |
|---|---|
| dataset | frozen `graphs.jsonl` + `splits.jsonl` + `manifest.jsonl` + `calibration_report.json` + `provenance.json` |
| meta-task | one frozen graph; 20 support trajectories of 20 tokens per meta-task |
| observation | `automotive_mc_obs_v1` = v3 (31) + 9 bounded MC columns → `FEATURE_DIM=40`, `PACKED_DIM=79` |
| normalization | frozen meta-train-only stats; the degenerate v3 `std(log_ue_cpu)=6.99e-11` is neutralised caller-side (identity), no refit |
| scheduler | per-graph CO-PHYSICAL config: `timing_model=physical_rates`, `radio_timing_model=physical_rates`, `energy_model=physical_v1`, `radio_model=physical_v1`, `energy_scope=system` |
| MC runtime | `execution_uncertainty_v1` + M7 mode machine; LO→HI only on HIGH over C_LO in equivalent work; HI sticky; HIGH never dropped/degraded; dependency-safe LOW drops; degradation unsupported/not applied |
| objective | `latency_only` + terminal Lagrangian penalty; raw latency, each violation, each λ and the penalized objective logged separately |
| constraints | `C_GRAPH_HARD_DEADLINE` (hard), `C_HI_TASK_TARDINESS` (firm), `C_MED_TASK_TARDINESS` (firm); energy `not_configured` |
| duals | observed on the trainer's controller from rollout telemetry (the legacy path was inert) and broadcast to the env |
| validation | frozen validation split, 20 support / 40 query, k=0 and k=3 |
| checkpoint rule | `automotive_lexicographic_v1` (hard-deadline rate → MC violations → HIGH tardiness → MEDIUM tardiness → energy if configured → latency) |
| masks | deadline mask **off** (queue-blind, no hard-deadline claim); legality mask unchanged |
| meta-test | guard counts every access; `meta_test_access_count == 0` before the freeze |

## Preserved learning algorithm

autoregressive 20-token plan; actions `{0:UE, 1:MEC, 2:HELPER}`; Graph2Seq encoder;
LSTM decoder + attention; inner PPO lr 5e-4, exactly 3 applies, clip 0.2, value clip
0.2, vf 0.5, grad norm 0.5, gamma 0.99, GAE lambda 0.95, entropy 0; meta-batch 10;
support 20 per meta-task; fresh query after adaptation; first-order mean
pseudogradient outer update with one apply per meta-batch.

## Sampler budget

`total_samples = meta_batch_size x support_trajectories x tokens_per_trajectory`
(explicit trajectory count), never derived from `max_path_length`; `max_path_length`
is the episode cap only. Counters: selected_meta_tasks, support/query trajectories and
tokens, unique graph ids, unique rollout seeds.
