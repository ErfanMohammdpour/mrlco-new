# MARGO automotive training integration - status

status: **READY_FOR_LONG_TRAIN** (all 11 gates green; the long run was NOT started)

## Gate 11 preflight
Kish `kish-ai`, checkout `/opt/margo/mrlco-new-6b` fast-forwarded 134a00b -> abe4321 (tree clean);
`/opt/margo/mrlco-new` only inspected; dataset SHA-verified by the loader; Step-2 gate PASS; TF1.15
nv2212 docker image with working GPU passthrough; GPU 4090 shared with a non-MARGO vLLM (not stopped,
~2.5 GiB free); no stale MARGO process. Details: KISH_PREFLIGHT_REPORT.md.

## Gates 12-19 (all green)
* 12 loader/split: strict schema + canonical-hash + provenance cross-checks; split-isolated accessors;
  mechanical MetaTestGuard (even reading the full `.graphs` view counts as meta-test access);
  meta-task = one frozen graph; `meta_test_access_before_freeze = 0`.
* 13 observation: `automotive_mc_obs_v1` = v3 (31) + 9 bounded MC columns -> FEATURE_DIM 40 / PACKED_DIM 79;
  criticality one-hot; `has_c_hi` separates N/A from zero; meta-train-only normalization; the degenerate
  v3 `std(log_ue_cpu)=6.99e-11` is neutralised caller-side (identity), no refit.
* 14 scheduler: per-graph CO-PHYSICAL config (timing/radio timing `physical_rates`, energy/radio
  `physical_v1`, scope `system`); `R = B*eta` rebuilt from each graph's frozen capacity. Parity with the
  frozen M10 certification config verified on makespan, per-task finish/availability, transfers and sink
  return (6 graphs x 3 pure plans, 3 tests).
* 15 MC runtime: frozen `execution_uncertainty_v1`; trigger decided in equivalent work (tier-invariant);
  HI sticky; HIGH never dropped/degraded; dependency-safe LOW drops (mean 2.9 tasks); HI capping on HIGH
  only; degradation unsupported/not-applied; open-loop documented. Smoke: mode-switch rate 0.975,
  MC policy violations 0.
* 16 objective/constraints: `latency_only` + terminal Lagrangian penalty; raw latency, each violation,
  each lambda and the penalized objective logged separately; LIVE duals (the legacy dual path was inert):
  lambdas after one iteration = C_GRAPH_HARD_DEADLINE 0.00394, C_HI_TASK_TARDINESS 0.02426,
  C_MED_TASK_TARDINESS 0.00144; energy_constraint = not_configured (no fake budget); dual fixtures pass.
* 17 sampler: explicit trajectory budget `10 x 20 x 20`; `max_path_length` is only the episode cap;
  counters exact: support 200 trajectories / 4000 tokens, query 200 / 4000, unique_rollout_seeds 200,
  unique_graph_ids 10; support/query realizations use disjoint seeds.
* 18 validation/meta-test: validation split only, k=0 and k=3 executed (firm miss 0.0675, graph hard
  violation 0.10, HIGH tardiness 0.375, MEDIUM 0.025, mode switch 0.975); checkpoint rule
  `automotive_lexicographic_v1` (sha a3bd00d5108237acdb706f561212b5d840bac151f48b5250a61c573d9c9ffb6c);
  meta-test guard PASS.
* 19 audits: deadline-signal probe on meta_train with bad policies - all_UE hard 0.383 / HIGH 0.425 /
  MED 0.158 / firm miss 0.194; random 0.289 / 0.367 / 0.261 / 0.191; all_MEC and greedy 0 (declared
  saturation caveat, requirements NOT tightened). HEFT defect found and fixed as `heft_reference_v2`
  (v2 <= best pure on 240/240, HELPER used on 112/240; frozen M10 kept as historical evidence).
  TF primary tests green in the container (obs 23, mc_runtime 38, constraints 40, heft 10, parity 3);
  full non-TF repo suite 1154 passed / 0 failed / 5 skipped.

## Gate 20 smokes
* CPU plumbing smoke (canonical budgets; MRLCO forbids shrinking them): PASS, wall 126 s.
* Real-GPU canonical primary smoke: **PASS** - seed 0, 1 iteration through
  `spec/automotive_gpu_smoke.py -> run_automotive_gpu_smoke -> _train -> build_frozen_primary_stack ->
  Trainer.train()`, wall 70.6 s, peak GPU 258.8 MB, inner PPO exactly 3 applies, meta pseudogradient
  finite, validation k0/k3 executed, meta_test accesses 0, co-physical scheduler, graph-specific rates,
  MC fixtures trigger LO->HI (below C_LO -> LO; within/above C_HI -> HI; HIGH preserved; capping 1).

## Gate 21 freeze
* final config: `spec/automotive_training/frozen_automotive_primary.yaml`
* training fingerprint: 6b46de6aa39dffcb9ef370e1dfd4713e54d1da2dcd09d676001abb6bd75ee710
* pins: git_sha unknown | dataset_manifest_sha a190e9dbb5599803fd5b9457e3e7648964849a1a7078fd0b82362150757b9e14 | graphs_sha 4341c3b65c6e07b7fea341e2f3c3947723c6ebe01b5eb115dbf51c8c7f38404f | splits_sha cd391da6ac12ef12e0f552f11f3c4bb551df50592a5a9753b08873d37c912907
  | split_policy_sha 7e051478b4b44b8c645d6df21ef0953183bda06bfd735b5440451fee4c606193 | provenance_sha 14eda3e7b75cde2e08fb6487ef2971c59b52eb3b5ee577c7bee881245f9d6590 | calibration_report_sha 3d5db9414b21c7774d0d5e9e8807ffe2f99252d30a0263121c8156b5f3ae7cd7
* mask: `deadline_mask=off`, `queue_blind_mask_used=false`
* energy: primary metric system scope, constraint not_configured
* evidence: `spec/automotive_training/reports/gpu_primary_smoke_report.json`,
  `reports/training_fingerprint.json`, `reports/deadline_signal_probe.json`

## Exact long-run command (DO NOT RUN YET)

```bash
ssh kish-ai
cd /opt/margo/mrlco-new-6b
git fetch origin phase4-eval && git checkout phase4-eval && git merge --ff-only origin/phase4-eval
for SEED in 0 1 2 3 4; do
  docker run --rm --gpus all \
    -e PYTHONPATH=/work -e MARGO_ALLOW_GPU=1 -e CUDA_VISIBLE_DEVICES=0 \
    -e TF_FORCE_GPU_ALLOW_GROWTH=true -e TF_CPP_MIN_LOG_LEVEL=2 \
    -v /opt/margo/mrlco-new-6b:/work -w /work \
    margo-phase4-tf115-nv2212:latest \
    python spec/automotive_gpu_smoke.py --long --seed "$SEED" --i-allow-gpu --gpu 0 \
    > /opt/margo/logs/automotive_mc_v1_primary_seed${SEED}.log 2>&1
done
```
Outputs: `/opt/margo/mrlco-new-6b/runs/automotive_mc_v1/primary/seed_<n>/` with
`ckpt/meta_model_best_val.ckpt` (+ `.metric.json`, lexicographic rule), `logs/progress.csv`,
`automotive_smoke_report.json`, `training_fingerprint.json`, `provenance.json`.
meta-test stays unopened until the model-side freeze.
