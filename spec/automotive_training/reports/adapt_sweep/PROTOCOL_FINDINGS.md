# Argmax collapse and the adaptation sweep (protocol v1 findings)

Evidence: `reports/adapt_sweep/` (sweep JSONs) and `spec/automotive_training/diagnose_decoding.py`
(reproduce with `--decoding deterministic|stochastic`; the numbers below are its printed output).

## 1. Deterministic (argmax) decoding is degenerate for this benchmark state

| core | argmax mix UE/MEC/HELPER | argmax k0 (s) | stochastic mix UE/MEC/HELPER | stochastic k0 (s) |
|---|---|---|---|---|
| random_core | 0.100 / 0.000 / 0.900 | 0.185290 | 0.3650 / 0.2725 / 0.3625 | 0.178530 |
| gatef_proof_final | 0.000 / 1.000 / 0.000 | 0.026489 | 0.0050 / 0.9938 / 0.0013 | 0.029586 |
| pilot_seed0_final | 0.000 / 1.000 / 0.000 | 0.026489 | 0.0013 / 0.9975 / 0.0013 | 0.026480 |
| pilot_seed2_final | 0.000 / 1.000 / 0.000 | 0.026489 | 0.0000 / 0.9975 / 0.0025 | 0.026517 |

* The RANDOM core behaves differently (90% HELPER under argmax, 0.185 s), so the checkpoint load
  and the decoding path work; the collapse belongs to the trained policies plus the runtime.
* Under argmax EVERY trained core emits exactly 100% MEC and the per-replicate makespans are
  bit-identical across checkpoints (0.02645229349574521 / 0.026526405831819956); k=0..5 are
  identical too, so the argmax plan is invariant to support adaptation.
* Consequence: argmax cannot be the PRIMARY metric here - it cannot rank checkpoints, cannot show
  adaptation, and equals the all-MEC baseline by construction. Keep it as a secondary
  'policy collapsed to all-MEC' indicator.

## 2. Paired stochastic adaptation sweep (corrected 101-iteration checkpoint, R=5)

| k | latency (s) | std (s) | regret vs candidate oracle (s) |
|---|---|---|---|
| k0 | 0.027687 | 0.002867 | +0.002373 |
| k1 | 0.027656 | 0.002680 | +0.002342 |
| k2 | 0.025939 | 0.000659 | +0.000625 |
| k3 | 0.026057 | 0.000729 | +0.000743 |
| k5 | 0.028631 | 0.004383 | +0.003317 |

Paired deltas vs k=0: `{"k1": {"better_replicates": 3, "mean_delta_vs_k0_s": -3.1895399767706276e-05, "replicates": 5}, "k2": {"better_replicates": 5, "mean_delta_vs_k0_s": -0.0017488819222784605, "replicates": 5}, "k3": {"better_replicates": 3, "mean_delta_vs_k0_s": -0.0016301729278092093, "replicates": 5}, "k5": {"better_replicates": 2, "mean_delta_vs_k0_s": 0.0009433484742663028, "replicates": 5}}`

* Adaptation is NOT dead under stochastic decoding: k=2 is the best budget (0.025939 s, 1.75 ms better than k=0) while k=5 clearly hurts (0.028631 s).
* Regret against the candidate oracle stays POSITIVE at every k (+0.62 ms best case): on the
  frozen paired panel the learned policy does not yet beat the best candidate (usually all-MEC).
* The deterministic sweep of the same checkpoint returned bit-identical values for k=0..5
  (0.025871 s), recorded in `gatef_proof_final.json`.

## 3. Recommended protocol revision (sign-off needed before the fresh campaign)

1. PRIMARY = seeded stochastic expectation: fixed graph-level seed, R paired realizations and
   S stochastic rollouts per graph per replicate; report mean and std.
2. SECONDARY = argmax, reported as the collapse indicator (it equals all-MEC here).
3. Keep the paired realizations, the committed seed panel, the same-realization baselines and the
   candidate-oracle regret - those parts worked.
4. Re-run the sweep (6 checkpoints x k in {0,1,2,3,5}) under the revised protocol and freeze k
   before the fresh 5x500.
