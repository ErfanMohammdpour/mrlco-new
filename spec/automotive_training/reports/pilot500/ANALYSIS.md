# MARGO-AUTOMOTIVE-MC-v1 primary pilot - 5 seeds x 500 iterations

Everything in this directory is the raw and consolidated output of the pilot campaign
that ran on Kish (`kish-ai`, `/opt/margo/mrlco-new-6b`) on 2026-10-02. Nothing here is a
paper result: one pilot budget (500 outer iterations), 5 seeds, validation split only,
meta-test untouched.

## 1. What was run

| item | value |
|---|---|
| dataset | MARGO-AUTOMOTIVE-MC-v1 (release `abe4321`), `dataset_manifest_sha=a190e9dbb5599803fd5b9457e3e7648964849a1a7078fd0b82362150757b9e14` |
| graphs sha | `4341c3b65c6e07b7fea341e2f3c3947723c6ebe01b5eb115dbf51c8c7f38404f` |
| splits sha | `cd391da6ac12ef12e0f552f11f3c4bb551df50592a5a9753b08873d37c912907` |
| observation | `automotive_mc_obs_v1` (FEATURE_DIM 40 / PACKED_DIM 79), meta-train-only normalization |
| scheduler | co-physical per-graph rates (`timing_model=physical_rates`, `radio_timing_model=physical_rates`, `energy_model=physical_v1`, `radio_model=physical_v1`, `energy_scope=system`) |
| objective | `latency_only` + live Lagrangian penalty (three truthful channels) |
| constraints | `C_GRAPH_HARD_DEADLINE` (hard), `C_HI_TASK_TARDINESS`, `C_MED_TASK_TARDINESS` (firm); energy `not_configured` |
| checkpoint rule | `automotive_lexicographic_v1` sha `a3bd00d5108237acdb706f561212b5d840bac151f48b5250a61c573d9c9ffb6c` |
| deadline mask | `off` (queue-blind; no hard-deadline claim) |
| meta-test | CLOSED - `meta_test_access_count=0` in all five runs |
| budget | 500 outer iterations/seed, meta-batch 10, 20 support trajectories x 20 tokens, inner PPO 3 applies |
| training fingerprint | `3b68036f526a3c7917c41321c831a8a63c8373256b62ab5579b7757496f01195` (identical across seeds: the fingerprint hashes the config, not the seed) |
| wall | 10.06 h total, 2.01 h/seed (single shared RTX 4090, ~258 MB peak, vLLM co-resident) |

## 2. Protocol of the numbers in this bundle

* `validation_k3_latency_s`: the policy is adapted for 3 inner PPO steps on the frozen
  20-graph support set, then evaluated on the disjoint 40-graph query set; the value is
  the mean makespan over those 40 rollouts under the frozen mixed-criticality runtime.
* `validation_k0_latency_s`: the same 40-graph query rollout with NO adaptation (the raw
  core policy). The CSV logs only the k=0 discounted return, and since that return equals
  minus the mean latency, k0 latency is recovered exactly as
  `-validation/objective_discounted_return_k0`.
* Validations happen every 50 iterations: 0, 50, ..., 450. There is no iteration-500
  validation; the final checkpoint (`meta_model_final.ckpt`, iteration 499) is never
  evaluated.
* The deterministic references in `reference_comparison.json` were evaluated on the SAME
  40 validation-query graphs with the SAME mixed-criticality realizations
  (`base_seed=303`, the evaluator's seed), so the comparison is instance-for-instance.

## 3. Results (5 seeds, final validation = iteration 450)

| seed | wall (h) | k3 latency (s) | k0 latency (s) | best_val itr (k3) | hard-viol rate | HIGH tard rate | MED tard rate | firm miss | k3<k0 (of 10) | first feasible itr | k3 vs itr0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 2.03 | 0.07876 | 0.07159 | 250 (0.08546) | 0.025 | 0.150 | 0.050 | 0.0250 | 4 | 250 | -51.5% |
| 1 | 2.01 | 0.14427 | 0.12770 | 100 (0.11730) | 0.100 | 0.325 | 0.025 | 0.0488 | 4 | None | -11.4% |
| 2 | 2.02 | 0.03105 | 0.03717 | 450 (0.03105) | 0.000 | 0.000 | 0.000 | 0.0000 | 8 | 100 | -77.3% |
| 3 | 2.01 | 0.08641 | 0.10992 | 450 (0.08641) | 0.000 | 0.075 | 0.050 | 0.0112 | 7 | 200 | -26.8% |
| 4 | 1.99 | 0.12644 | 0.12766 | 150 (0.10879) | 0.125 | 0.200 | 0.025 | 0.0462 | 5 | None | -23.0% |
| **mean +- std** | 2.01 | **0.09339 +- 0.03957** | 0.09481 +- 0.03536 | 0.08580 +- 0.03006 | 0.050 | - | - | - | - | - | - |

* median k3 = 0.08641 s, min = 0.03105 s (seed 2), max = 0.14427 s (seed 1).
* MC policy violations (HIGH dropped or degraded): **0** in every one of the 50 validations.
* Mode-switch rate 0.975 and dropped-task mean 2.925 are constant across seeds and
  iterations: they are properties of the frozen instance distribution and the
  dependency-safe LOW drops, not of the learned policy.

### k3 validation curves (iteration 0,50,...,450)

| seed | k3 latency per validation |
|---|---|
| 0 | 0.16229 | 0.15743 | 0.11308 | 0.10590 | 0.09296 | 0.08546 | 0.09000 | 0.08392 | 0.10138 | 0.07876 |
| 1 | 0.16285 | 0.15422 | 0.11730 | 0.13264 | 0.13284 | 0.11181 | 0.13761 | 0.13035 | 0.12264 | 0.14427 |
| 2 | 0.13648 | 0.12456 | 0.08950 | 0.08868 | 0.07905 | 0.06369 | 0.04023 | 0.03884 | 0.03667 | 0.03105 |
| 3 | 0.11798 | 0.13180 | 0.09589 | 0.10875 | 0.09440 | 0.11168 | 0.09959 | 0.09057 | 0.10334 | 0.08641 |
| 4 | 0.16414 | 0.13247 | 0.14718 | 0.10879 | 0.14224 | 0.16599 | 0.13617 | 0.14221 | 0.13188 | 0.12644 |

### k0 validation curves (no adaptation)

| seed | k0 latency per validation |
|---|---|
| 0 | 0.16540 | 0.14708 | 0.15697 | 0.11401 | 0.09058 | 0.08528 | 0.09823 | 0.07115 | 0.08628 | 0.07159 |
| 1 | 0.14560 | 0.15474 | 0.15127 | 0.14692 | 0.13005 | 0.13780 | 0.10519 | 0.11106 | 0.11930 | 0.12770 |
| 2 | 0.13502 | 0.16146 | 0.12691 | 0.09169 | 0.08280 | 0.08020 | 0.05172 | 0.05077 | 0.03605 | 0.03717 |
| 3 | 0.15796 | 0.12325 | 0.11914 | 0.10833 | 0.10299 | 0.11435 | 0.10170 | 0.10364 | 0.09696 | 0.10992 |
| 4 | 0.15360 | 0.13815 | 0.14425 | 0.12986 | 0.10914 | 0.15288 | 0.16249 | 0.16367 | 0.11095 | 0.12766 |

### hard-deadline violation rate per validation

| seed | hard-violation rate per validation |
|---|---|
| 0 | 0.100 | 0.100 | 0.050 | 0.025 | 0.025 | 0.000 | 0.050 | 0.025 | 0.025 | 0.025 |
| 1 | 0.100 | 0.175 | 0.075 | 0.075 | 0.050 | 0.050 | 0.075 | 0.075 | 0.050 | 0.100 |
| 2 | 0.100 | 0.050 | 0.000 | 0.025 | 0.000 | 0.025 | 0.000 | 0.000 | 0.000 | 0.000 |
| 3 | 0.050 | 0.025 | 0.025 | 0.025 | 0.000 | 0.050 | 0.050 | 0.000 | 0.000 | 0.000 |
| 4 | 0.175 | 0.025 | 0.100 | 0.050 | 0.075 | 0.125 | 0.100 | 0.100 | 0.050 | 0.125 |

## 4. Feasibility (the checkpoint rule's first priority)

`automotive_lexicographic_v1` orders: (1) zero graph hard-deadline violation rate,
(2) zero MC policy violations, (3) lowest HIGH tardiness, (4) lowest MEDIUM tardiness,
(5) energy only if configured, (6) lowest latency.

* Seeds with a feasible checkpoint (hard rate 0 somewhere): **3 of 5** - seeds 0 (only itr 250), 2 and 3.
* Seeds whose FINAL validation is feasible: **2 of 5** (2, 3).
* Seeds 1 and 4 never produced a feasible checkpoint, so the rule had to fall back to an
  infeasible one (their `selection_key[0]` is `1`). This is the rule behaving correctly,
  not a bug - but it means two of five runs ended with a model that violates at least one
  graph deadline on the validation split (seed 1: 0.100 = 4/40 graphs; seed 4: 0.125 = 5/40).
* A hard-violation rate of 0.025 is exactly 1 graph out of 40.

## 5. Does the 3-step adaptation help? (k3 vs k0)

* At the final validation, k3 beats k0 in **3 of 5** seeds (2, 3, 4 - seed 4 only marginally: 0.12644 vs 0.12766).
* Over all 10 validations: seed 0 4/10, seed 1 4/10, seed 2 8/10, seed 3 7/10, seed 4 5/10.
* Mean `k3-k0` over the last three validations: seed 0 **+0.01168 s** (adaptation hurts),
  seed 1 **+0.01306 s** (hurts), seed 2 **-0.00581 s** (helps), seed 3 **-0.01007 s** (helps),
  seed 4 **-0.00058 s** (neutral).
* Reading: adaptation is only beneficial where the run learned something (seeds 2, 3). The
  two weakest runs are also the two where adaptation is harmful at the end. This is a real
  open question, not a rounding effect: on seed 1 the adapted policy is 1.3e-2 s (10%)
  slower than the unadapted one at iteration 450.

## 6. Duals

| seed | lambda hard | lambda HIGH | lambda MED |
|---|---|---|---|
| 0 | 0.1239 | 0.9418 | 0.0575 |
| 1 | 0.0959 | 0.8854 | 0.0814 |
| 2 | 0.0325 | 0.3457 | 0.0346 |
| 3 | 0.0562 | 0.4190 | 0.0192 |
| 4 | 0.0925 | 0.8052 | 0.0575 |

* Every seed drives `lambda_HIGH` up (0.35 to 0.94) because HIGH tardiness never reaches
  zero; the two infeasible seeds end with the largest duals (0.885, 0.805), the best seed
  with the smallest (0.346). This is the expected behaviour of a fixed-step projected
  ascent, and no divergence or NaN was observed.
* `lambda_hard` grows monotonically in the infeasible seeds (up to 0.124) and stays small
  where the run became feasible early (seed 2: 0.033).

## 7. Comparison against deterministic references (same 40 graphs, same realizations)

| reference | mean (s) | median (s) | min (s) | hard-viol rate | mean model / reference |
|---|---|---|---|---|---|
| all_HELPER | 0.17488 | 0.17076 | 0.07559 | 0.125 | 0.534 |
| all_MEC | 0.02622 | 0.02395 | 0.01031 | 0.000 | 3.561 |
| all_UE | 0.27698 | 0.24933 | 0.12461 | 0.425 | 0.337 |
| greedy_coordinate_descent_MC | 0.03478 | 0.02996 | 0.01102 | 0.000 | 2.685 |
| heft_reference_v2_nominal | 0.02988 | 0.02690 | 0.01179 | n/a | 3.125 |
| heft_reference_v2_on_MC_instance | 0.02732 | 0.02551 | 0.01026 | 0.000 | 3.418 |

* The mean model is **3.56x slower than all-MEC**,
  **2.69x slower than the MC-aware greedy**,
  and **3.42x slower than HEFT v2** on the same instances.
  It is about 2.97x faster than all-UE and 1.87x faster than all-HELPER.
* Seeds that beat a reference at the final validation: greedy -> [2], all-MEC -> [], HEFT v2 -> [].
* Seed 2 (0.03105 s, zero violations) is the only run inside the reference band
  (0.026-0.035 s); it is better than the greedy baseline and slightly behind all-MEC and
  HEFT v2.

## 8. What this pilot establishes - and what it does not

Established:
1. The whole frozen chain trains end to end on the GPU: build -> sampler -> inner PPO
   (3 applies) -> first-order meta update -> duals -> energy/constraint telemetry ->
   validation k0/k3 -> lexicographic checkpoint -> fingerprint. Five seeds, zero crashes,
   zero NaN, zero MC policy violations, 10.06 h total.
2. The constraint machinery is live and behaves: duals rise, the selection rule refuses
   lower-latency infeasible checkpoints (seed 0 picked itr 250 over a faster itr 450), and
   every validation is feasible for seeds 2 and 3.
3. Meta-test isolation held (`0` accesses).

Not established (and must not be claimed):
1. That the learned policy is competitive. With 500 iterations the mean is
   3.56x behind all-MEC and only 1 of 5 seeds reaches
   the reference band. A 500-iteration pilot is far below the frozen 3500-iteration budget.
2. Anything about meta-test performance (closed by design).
3. Anything about statistical significance: n=5 seeds, and each validation is a single
   40-graph rollout with no per-instance error bars; the validation set is also reused for
   every validation (no separate model-selection split beyond the lexicographic rule).
4. That adaptation helps in general - it is seed-dependent in this pilot.

## 9. Recommended next steps (in order of information gained per GPU hour)

1. **Long run on the best seed only** (seed 2, 3500 iterations, ~14 h): does the reference
   band hold or improve? This is the cheapest decisive experiment.
2. **Adaptation study** (no new training): re-evaluate the five saved
   `meta_model_best_val.ckpt` for k=0,1,2,3,5,10 inner steps on the same support/query
   split to find whether 3 steps is simply too many/too few. Uses `eval/` only, ~30 min.
3. **Diagnose the two failed seeds** (1 and 4): inspect `action_fraction/*` and the
   per-iteration support reward to see whether they collapse to a single action or
   oscillate; check whether the dual on HIGH tardiness is over-penalising early.
4. Only then decide whether to launch the full 5-seed x 3500 campaign.

## 10. Reproduction

```bash
ssh kish-ai
cd /opt/margo/mrlco-new-6b
git checkout phase4-eval && git pull        # training code as of the campaign: 62f49e9
for SEED in 0 1 2 3 4; do
  docker run --rm --gpus all \
    -e PYTHONPATH=/work -e MARGO_ALLOW_GPU=1 -e CUDA_VISIBLE_DEVICES=0 \
    -e TF_FORCE_GPU_ALLOW_GROWTH=true -e TF_CPP_MIN_LOG_LEVEL=2 \
    -v /opt/margo/mrlco-new-6b:/work -w /work margo-phase4-tf115-nv2212:latest \
    python spec/automotive_gpu_smoke.py --long --seed "$SEED" --i-allow-gpu --gpu 0 \
      --iters 500 --run-kind primary_500
done
# references on the same instances
python spec/automotive_training/validation_reference_comparison.py --json /tmp/ref.json
```

## 11. Files

| file | content |
|---|---|
| `summary.json` | per-seed + aggregate metrics, curves, fingerprints, counters, references |
| `validation_table.csv` | all 50 validation rows (5 seeds x 10 validations) as tabular data |
| `reference_comparison.json` | deterministic references on the same 40 graphs/realizations |
| `seed_N/logs/progress.csv` | the full per-iteration training log (500 rows, all logged metrics) |
| `seed_N/automotive_smoke_report.json` | end-of-run report: counters, lambdas, fixtures, peak GPU, fingerprint |
| `seed_N/training_fingerprint.json` | config fingerprint with dataset/graph/split/calibration pins |
| `seed_N/provenance.json` | run provenance (method id, run kind, outer iterations, paths) |
| `seed_N/config.resolved.json` | resolved config at launch |
| `seed_N/ckpt/meta_model_best_val.metric.json` | the lexicographic selection decision (key + rule sha) |
| `seed_N/ckpt/checkpoint_sha256.txt`, `checkpoint_sizes.txt` | hashes/sizes of the (not committed) weights |
| `configs/` | the frozen pilot and primary configs, the iteration-count audit, obs normalization stats |
| `campaign.log.gz`, `campaign_markers.txt`, `campaign_crashed_500.log.gz` | server-side campaign log, seed start/exit markers, and the earlier crashed 50-iteration campaign log |
| `MANIFEST.json` | sha256 + size of every file here, plus the checkpoint tarball location/hash |

Checkpoint weights (130 MB, 30 files) are NOT in git; they stay on the server at
`/opt/margo/automotive_pilot500_checkpoints.tar` (sha256 in `MANIFEST.json`) and in
`runs/automotive_mc_v1/primary_500/seed_*/ckpt/`.
