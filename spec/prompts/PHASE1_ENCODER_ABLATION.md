# PHASE 1 — Encoder ablation under a fixed BC-2opt recipe (attacks the OOD gap directly)

## Role
You are a senior deep-learning engineer with graph-representation expertise, working in `MARGO_BASELINE/mrlco-new` (TensorFlow 1.15). Your job is a controlled ablation: change **only** the encoder, keep everything else byte-identical, and measure out-of-distribution decode quality. You are not allowed to touch PPO, adaptation, or the scheduler.

## Why this phase exists
Twenty-six diagnostics kept the backbone unchanged (`Graph2SeqEncoderAdapter`: two-layer bidirectional `MeanAggregator`, `hidden_dim=128`, node output 256, sum+ReLU across directions, triple readout → LSTM initial state). The remaining gap is out-of-distribution: train greedy-decode T 436–448 vs validation 575–582 (expert 424–464), token accuracy vs expert 0.97 train → 0.65–0.71 validation. Exposure bias (scheduled sampling), teacher quality (2-opt), pair supervision, and distribution-ID oracles were all ruled out. The encoder is the only untested variable, and every 2024–2026 competitor (DAMRL, FedMAGS, DVTP, HiGNN-PPO) uses attention-based graph encoders.

## Fixed recipe (do not change)
- Data: meta_train `{1,3,4,5,8,9,11,13,15,18,19,21,22,24,25}`, labels = 2-opt plans from cache `runs/phase4/expert_2opt_mec_train.npz` (see `spec/twopt_expert.py`). If the cache is missing on Kish, regenerate with `spec/twopt_expert.py` and record its sha256 in provenance.
- Observation: `env/mec_offloaing_envs/scheduler/encoder_obs.py`, `[B,20,50]`, 11 z-scored features + 19 successor + 19 predecessor decoder indices + mask. Stats from `spec/encoder_feature_stats.json` (meta_train only).
- Decoder: unchanged 2-layer LSTM 128 + Luong attention over node embeddings, `vocab_size=3`, `start_token=0`, `end_token=3`, greedy pad action 1.
- Training: CE teacher forcing, Adam 5e-4, batch 32, early stop on train CE with patience 10 / min_delta 0.005, max 120 epochs, same as `bc_2opt` (`spec/bc_greedy_mec.py::run_bc_greedy_mec` path). Seeds `{0,1,2}`.
- Evaluation: greedy decode + one `schedule_via_adapter` per graph on train n=1500 and validation n=500 (the five validation distributions, all 100 graphs each). Meta-test is **not** evaluated in this phase.

## Encoders to implement (three variants, one flag)
Add `encoder_type` to `Graph2SeqEncoderAdapter` (`policies/graph2seq_encoder.py`) and thread it through `MetaSeq2SeqPolicy` / `Seq2SeqPolicy` hparams and `provenance.json`.
1. `meanagg` — current implementation. Must reproduce `bc_2opt` numbers within noise (train ≈436, val ≈582) before anything else is trusted.
2. `gatv2` — bidirectional GATv2 (Brody et al. 2022) with 4 heads × 32 dims per direction, 2 layers, ELU, same self‖neigh concat to 256, same sum+ReLU across directions, dropout 0. Attention uses the packed neighbour tables already in obs; masked softmax over ≤19 neighbours with `-1e9` on PAD. No self-loops (keep parity with meanagg).
3. `dagformer` — DAG-aware transformer (Luo et al., "Transformers over Directed Acyclic Graphs", NeurIPS 2023 style): 2 layers, 4 heads, d=256, attention mask = reachability (ancestor ∪ descendant ∪ self) computed from the DAG per graph, plus a learned depth embedding (`depth` feature already in obs, integer-bucketed 0..19) added to the input projection. Pre-LN, FFN 512, dropout 0. Provide the reachability mask as an extra placeholder `[B,20,20]` built in `encoder_obs.py` (new function `reachability_mask(dag)`) so the rest of the pipeline does not change shape.
All three must emit `[B,20,256]` node embeddings and feed the unchanged triple readout.

## Readout ablation (second factor, on the winning encoder only)
`readout_type ∈ {triple, mean, max, attn, zero}` where `zero` initializes the LSTM state with zeros. Run on seed 0 only after the encoder winner is known.

## Registration
- `spec/phase4_campaign.py`: `DIAG_ENC_METHOD_ID = "margo_v0.3_diag_encoder"`, `diag_encoder_run_dir(seed, encoder_type, readout_type)` producing `runs/phase4/margo_v0.3_diag_encoder/<encoder_type>_<readout_type>/seed_<s>/`; CLI `--diagnostic-encoder --encoder-type X --readout-type Y --seed S`.
- `spec/phase4_train_driver.py`: `run_diagnostic_encoder(...)` calling the BC path with the new flags; provenance must include `encoder_type`, `readout_type`, `label_cache_sha256`, `n_params`.
- `spec/kish_gpu.sh`: target `encoder <type> <readout> <seed>`.
- Tests in `env/mec_offloaing_envs/scheduler/tests/`: `test_phase1_encoders.py` (pure numpy where possible; TF tests marked to skip without TF): (a) `reachability_mask` on a 4-node chain and a fork/join equals hand-written matrix; (b) permutation test — relabel nodes of one graph, remap edges, assert encoder outputs are permuted identically (tolerance 1e-5) for all three encoders; (c) topology sensitivity — identical node features, different edges → different embeddings for all three encoders; (d) meanagg output equals pre-change output on a fixed random input (regression against a saved `.npy` fixture you create from the current code **before** editing).
- Unique-dir count in `test_phase4_campaign.py` must be updated.

## Execution order on Kish
1. `encoder meanagg triple 0` → compare to `bc_2opt` archive (`spec/kish_log_archive/bc2opt_eval.json`): train T within ±10 s, val T within ±10 s, token within ±0.02. If not, stop and debug before running anything else.
2. `gatv2 triple {0,1,2}`, `dagformer triple {0,1,2}`, `meanagg triple {1,2}`.
3. Readout ablation on the winner, seed 0.
GPU usage: BC ≈ 15–25 min per run on the RTX 4090; ≤ 12 runs total.

## Gate (decided on validation only)
Let ΔT = val T(meanagg) − val T(candidate), mean over 3 seeds.
- ΔT ≥ 30 s and token accuracy up ≥ 0.05 → encoder becomes the v0.3 backbone; record in `spec/decisions/ADR-010-encoder.md`.
- 10 s ≤ ΔT < 30 s → keep as option; Phase 2/3 run on both encoders.
- ΔT < 10 s for both candidates → conclusion "OOD gap is not representational"; backbone stays meanagg; write this explicitly (it is a result, not a failure).
Readout: any variant within ±5 s of triple means triple is not a contribution; say so.

## Double-check (mandatory)
Before launch:
1. Fixture for meanagg regression saved and test (d) passes on the **unmodified** code first, then on the modified code.
2. `python3 -m pytest env/mec_offloaing_envs/scheduler/tests -q` locally: no new failures.
3. `rg -n "encoder_type" spec/phase4_train_driver.py policies/graph2seq_encoder.py policies/meta_seq2seq_policy.py` shows the flag reaching the constructor and the provenance payload.
4. Dry-run with `--smoke` (2 epochs, 64 graphs) for each encoder on Kish; confirm shapes `[B,20,256]`, no NaN, `n_params` logged.
5. Parameter budget: `n_params` for gatv2 and dagformer must be within 2× of meanagg; otherwise reduce heads/d and note it. Unequal capacity confounds the ablation.
After runs:
6. For each run, verify from the JSON: `n_graphs` train 1500 / val 500; `evals_per_graph == 1`; mix and `n_non_p50` reported; token accuracy computed against the same 2-opt labels for train and against the cached validation 2-opt labels (`expert_2opt_mec_validation.npz`).
7. Re-derive the gate arithmetic by hand from the three seeds and paste the per-seed numbers into the ledger; do not report only means.
8. Confirm no meta-test file was opened: `rg -n "metatest|meta_test" <run_dir>/logs` must be empty.
9. Confirm the meanagg seed 0 reproduction matched the archive before trusting any comparison.

## Report
Table with rows = encoder × readout, columns = train T / val T / token train / token val / mix / n_non_p50 / n_params / minutes, per seed and mean±std. One paragraph stating the gate outcome and the ADR-010 decision. Update ledger files per README conventions. `paper_result=false`.
