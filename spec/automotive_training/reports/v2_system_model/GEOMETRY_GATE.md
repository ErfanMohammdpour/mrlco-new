# v2 geometry gate (regimes A-L, 20 frozen validation graphs per regime)

Classifier: **STRONG EMPIRICAL EVIDENCE (v2 model)** for the headroom and winner
fractions; **SURROGATE/MODEL DIFFERENCE** for the absolute gap to v1, because the
v1<->v2 single-DAG parity study measured a systematic bias (see below).

Panel: all-UE, all-MEC, all-HELPER, HEFT v2, v2-aware greedy coordinate descent,
and a bounded stronger search labelled **candidate-search lower bound**
(budget 800 scheduler evaluations, 2 starts + 1 ILS kick, reproducible).

| regime | all-MEC winner (search incl.) | all-MEC winner (no search) | all-UE | all-HELPER | mixed | headroom mean | headroom max | search gain vs all-MEC |
|---|---|---|---|---|---|---|---|---|
| A_mec_idle_stable_v2i | 0.00 | 0.15 | 0.00 | 0.00 | 1.00 | 7.5% | 10.5% | 7.4% |
| B_mec_light | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 11.4% | 20.3% | 11.4% |
| C_mec_moderate | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 14.5% | 48.0% | 14.5% |
| D_mec_heavy | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 31.3% | 51.8% | 38.7% |
| E_helper_idle_stable_v2v | 0.00 | 0.15 | 0.00 | 0.00 | 1.00 | 7.5% | 10.5% | 7.4% |
| F_helper_busy | 0.00 | 0.10 | 0.00 | 0.00 | 1.00 | 6.1% | 11.8% | 6.1% |
| G_helper_short_contact | 0.00 | 0.15 | 0.00 | 0.00 | 1.00 | 7.5% | 10.5% | 7.5% |
| H_poor_v2v | 0.00 | 0.15 | 0.00 | 0.00 | 1.00 | 6.3% | 9.4% | 6.2% |
| I_network_high_variance | 0.00 | 0.55 | 0.00 | 0.00 | 1.00 | 5.7% | 10.3% | 5.6% |
| J_high_unreliable_remote | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.0% | 0.0% | 0.0% |
| K_high_reliable_idle_mec | 0.00 | 0.15 | 0.00 | 0.00 | 1.00 | 7.1% | 10.3% | 7.0% |
| L_low_good_remote | 1.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.0% | 0.0% | 0.0% |

## Verdict

```
gate = PASS   (BLOCK rule: all-MEC winner > 0.80 in EVERY regime)
all_MEC_wins_everywhere_over_80pct = False
max all-MEC winner fraction = 1.00 (regime L) ; min = 0.00
max mean headroom = 31.3% (regime D) ; min = 0.0%
helper-only wins somewhere = False ; local wins somewhere = True
```

## What this does and does not establish

1. **MEC contention is the driver.** Headroom rises monotonically with the number of
   concurrent DAGs sharing the MEC: A 7.5% -> B 11.4% -> C 14.5% -> D **31.3%** (median
   40.8%, max 51.8%, search gain 38.7% in D). This is the mechanism the literature
   review predicted and it is now produced by a real shared queueing scheduler, not by
   the `f_mec/N` surrogate.
2. **All-MEC is not universally dominant.** Without the search candidate, all-MEC wins
   0-15% of graphs in eleven regimes (55% under high network variance) and 100% only in
   regime L (LOW criticality + good remote), which is a legitimate 'MEC is best' case.
3. **A helper-specific advantage is NOT established.** HELPER-only plans never win, and
   the helper regimes (E idle, F busy, G short contact, H poor V2V) differ little from
   the no-contention baseline (7.5 / 6.1 / 7.5 / 6.3%). Either HELPER is genuinely
   redundant under this workload/rate geometry, or the v2 helper model still does not
   create a regime where a helper is decisive. **Do not claim V2V/helper gains.**
4. **Reliability changes placement.** Regime J (HIGH + unreliable remote) rejects every
   remote token and local execution wins 100%; regime K (HIGH + reliable idle MEC)
   behaves like the stable baseline, so HIGH is not hard-coded local.
5. **The absolute headroom carries a model difference.** v1<->v2 parity on identical
   single-DAG inputs: all_UE 0.00%, all_MEC -1.33% (v2 faster), all_HELPER -0.66%,
   alternate MEC/HELPER +2.06% (max +7.9%), alternate MEC/UE +0.26%. Transfer-count
   accounting still differs on 24/48 rows. The gate headroom is therefore reported as a
   MODEL+SEARCH difference, not as pure geometric headroom, and the v1 candidate-panel
   number (1.96%) is not directly comparable.

## Consequence for training

The BLOCK condition is not met, so the program may proceed to the CRN evaluator (stage 7)
and the v2 1x500 run (stage 9) **under the following stated limits**: no helper/V2V claim,
the contention axis is the supported finding, and any learned mixed placement must beat
the cheap panel (greedy/HEFT), not merely the search-based candidate.

## Reproduce

```bash
python3 spec/automotive_training/v2/geometry_gate.py --graphs 20 --search-budget 800 \
  --json spec/automotive_training/reports/v2_system_model/GEOMETRY_GATE.json
python3 spec/automotive_training/v2/parity_study.py --graphs 8
```

Parity detail: `V1_V2_PARITY.json`. Stage history and errata: `V2_STAGE_LOG.md`.
