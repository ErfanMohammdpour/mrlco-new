# Pilot A report (Phase P2, first 25 iterations)

Code SHA: `fe90a11617baf030ebfef3dd3c8d07ad9e29c74b` (tree clean). Kish image
`margo-phase4-tf115-nv2212` (TF 1.15.5), RTX 4090. Run dir
`runs/mask_sanity_v3/off/seed_0`, exit **0**, wall **5829 s** (25 iterations).
Evidence: `pilot_a_evidence.json`, `pilot_a_progress.csv`, `pilot_a_bundle.txt`.

## Contract actually run

`reward_mode=latency_only`, `objective_mode=log_only`, constraints off,
`mask_mode=off`, `obs_version=v3`, seed 0, 25 iterations, `energy_model=physical_v1`,
`radio_model=physical_v1`, timing `legacy_frozen_rates`, production decoder order
`legacy_current`, `entropy_coefficient=0`. Baselines (all-MEC and Greedy) are
computed inside the same validation call with the SAME scheduler/config — no
legacy-scheduler baseline is used for the comparison.

## Watchdog and metrics

* `collapse/flag = 0` for every iteration; MEC share max **0.608** (< 0.95), so no
  action collapse. Local remained **0.311–0.452** and V2V **0.0505–0.354** (never
  eliminated).
* All series finite; `critic/value_abs_max` max 0.862 (limit 1e3);
  `policy/approx_kl` 0.0089–0.0444, `clip_fraction` 0.130–0.471,
  `grad_norm` 0.325–0.812, policy loss -0.041–0.081, value loss 0.022–0.072.
* CSV: 25 rows, 85 aligned fields, no NaN/Inf.

## Trend (training)

| metric | first | last | min | max |
|---|---|---|---|---|
| Average reward | 0.538 | **0.687** | 0.538 | 0.711 |
| Average latency (s) | 948.8 | **784.2** | 742.1 | 948.8 |
| action_fraction/mec | 0.320 | 0.545 | 0.320 | 0.608 |
| action_fraction/local | 0.326 | 0.404 | 0.311 | 0.452 |
| action_fraction/v2v | 0.354 | 0.051 | 0.051 | 0.354 |
| policy/entropy_valid | 1.096 | 0.761 | 0.761 | 1.096 |

## Validation (itr 0 only)

`validation_all_mec_latency=630.28 s`, `validation_greedy_latency=626.60 s`,
policy latency 948.83 s → `gap_to_all_mec=+263.08 s`, `gap_to_greedy=+266.75 s`;
co-location rate 0.395, cross-location edges 18.19, MEC task fraction 0.425,
utilisation mean 0.393.

## Verdict

**INCONCLUSIVE (validation criterion unmeasurable at this length).** No collapse
and a clear training trend, but the frozen `validation_interval=50` exceeds 25
iterations, so there is exactly one validation point (init). The PASS criteria
"validation improves vs init" and "best validation beats all-MEC" therefore
cannot be evaluated, and the contract says extend the same run instead of
restarting.

Action taken: `pilot-a-extend` (same contract/seed/config, 40 iterations) is
launched from the same SHA. Note: no resume path exists in the harness, so the
extension re-runs the 40-iteration schedule from scratch with the same seed
rather than resuming the 25-iteration checkpoint; validation still occurs at
itr 0 only because the interval is frozen at 50. Nothing else was changed.
