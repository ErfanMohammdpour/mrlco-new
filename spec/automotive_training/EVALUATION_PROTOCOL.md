# Frozen evaluation protocol v1 (`automotive_eval_protocol_v1`)

Frozen before any campaign; `spec/automotive_training/eval_protocol.py` is the single
source of truth and `protocol_sha()` is recorded in every report and fingerprint.

## Why

The 5x500 pilot compared numbers that were not comparable:

| defect | pilot behaviour | frozen protocol |
|---|---|---|
| pairing | k0 used `base_seed=101`, the k3 query used `303`, and the k3 path resets the env once more, so even the same seed drew a different MC realization | one pinned `realization_epoch` + base seed per replicate; k0, k3 and **every** baseline see the same graph realization |
| replicates | one realization per graph (a single sample) | `SELECT_REPLICATES = 5` for in-training validation and checkpoint selection; `REPORT_REPLICATES = 20` for reporting |
| seeds | chosen implicitly | committed list `seed_r = 1000 + 17 r`; `realization_seeds(n)` only accepts a prefix, so no seed can be dropped or added after seeing a result |
| decoding | stochastic sampling for the reported value | primary inference is **argmax (greedy) decoding**; the support adaptation stays stochastic (inner PPO needs a distribution); the stochastic policy is a secondary robustness run |
| baselines | computed on a different seed | all-UE / all-MEC / all-HELPER / MC-aware greedy / HEFT v2 are evaluated on the SAME realization as the model |
| metric | single scalar latency | per-graph **candidate-oracle regret**: `MARGO - min(all_MEC, greedy, HEFT)` on the same graph+realization |

## Freeze

```text
protocol_id        automotive_eval_protocol_v1
select_replicates  5      (validation + lexicographic checkpoint selection)
report_replicates  20     (final report, adaptation sweep, salvage re-evaluation)
primary_decoding   deterministic (argmax)
secondary_decoding stochastic
adaptation_k_steps 0, 1, 2, 3, 5
realization_seeds  1000, 1017, 1034, ..., 1323   (committed, prefix-only)
baseline_candidates all_UE, all_MEC, all_HELPER, heft_reference_v2, greedy_coordinate_descent_MC
```

## Regret

For every graph and realization:

```text
regret = T_MARGO - min(T_all_MEC, T_greedy, T_HEFT)
```

Negative regret means MARGO beat every candidate. This is the metric that answers the
scientific question the benchmark actually poses: *did the policy learn when to leave
all-MEC?* - not merely whether it beats one fixed heuristic.

## Cost

A select-panel validation is ~5x the old single-realization validation (≈5-6 min per
validation instead of ~1 min on the current shared GPU), i.e. about +1 h per 500-iteration
seed with 10 validations. Report panels (20 replicates) are only used for
reporting/sweeps, never inside the training loop.

## Guarantees enforced by tests

```text
test_automotive_eval_protocol.py    (no TF): seed manifest, prefix-only, decoding flags,
                                             oracle/regret, metric denominators,
                                             realization independent of reset_count
test_automotive_validation_sync.py  (TF):   per-replicate verified core sync, scratch ==
                                             core, k0/k3 same realization inside a
                                             replicate, evaluate_all uses the select panel,
                                             report contract (k0 latency, penalty feedback,
                                             dual reset, counters, meta_test 0)
```
