# Correctness repair: Gate F proof, Gate G salvage, Gate I headroom

All numbers in this file were produced AFTER the Gate B/C/D/E fixes (commits `63f91cb` .. `c6b7673`). Raw pilot data stays untouched under `reports/pilot500/` and is marked superseded by `PILOT500_ERRATA.md`.

## Gate F - 101-iteration GPU proof (live constraint loop, corrected evaluator)

| itr | k0 latency (s) | k3 latency (s) | hard-viol | HIGH task rate | graph HIGH incidence | firm miss |
|---|---|---|---|---|---|---|
| 0 | 0.16552 | 0.17275 | 0.150 | 0.1675 | 0.550 | 0.1059 |
| 50 | 0.03073 | 0.04284 | 0.025 | 0.0150 | 0.050 | 0.0088 |
| 100 | 0.02547 | 0.02639 | 0.000 | 0.0000 | 0.000 | 0.0000 |

Correctness evidence logged on every iteration:

| check | value | meaning |
|---|---|---|
| `correctness/lambda_broadcast_targets` | 11 | 1 original env + 10 executor clones receive lambda (Gate C) |
| `correctness/core_scratch_sync_count` | 6 per validation | core -> scratch sync happens twice per validation, verified (Gate B) |
| `correctness/core_unchanged_after_adaptation` | True | adaptation never mutates the trained core |
| `constraint/batch_size_after_reset_*` | 0 | the dual batch is emptied every outer iteration (Gate D2) |
| `constraint/penalty` non-zero | 74/101 iterations | the Lagrangian penalty reaches the rollout rewards (D1 + C); iteration 0 is unpenalised by construction because lambda starts at 0 |
| `meta_test_access_count` | 0 | meta-test stayed closed |

At iteration 100 the core policy (k0, no adaptation) reaches **0.02547 s** with zero hard violations and zero HIGH tardiness, and k0 beats k3 at every validation: with the corrected evaluator the 3-step adaptation slightly hurts at this budget.

## Gate G - salvage of the old 5x500 cores (corrected evaluator)

Label: `old_training_latency_only_corrected_evaluator` - the pilot reward was latency-only in practice, the evaluation is correct.

| pilot run | k0 (core) s | k3 (3-step) s | hard-viol (k0) | HIGH task rate (k0) | sync verified | core preserved |
|---|---|---|---|---|---|---|
| seed 0 best_val | 0.02698 | 0.03214 | 0.000 | 0.0000 | yes | True |
| seed 0 final | 0.02572 | 0.02630 | 0.000 | 0.0000 | yes | True |
| seed 1 best_val | 0.02606 | 0.02631 | 0.000 | 0.0000 | yes | True |
| seed 1 final | 0.02579 | 0.02631 | 0.000 | 0.0000 | yes | True |
| seed 2 best_val | 0.02624 | 0.02629 | 0.000 | 0.0000 | yes | True |
| seed 2 final | 0.02540 | 0.02628 | 0.000 | 0.0000 | yes | True |
| seed 3 best_val | 0.02538 | 0.02596 | 0.000 | 0.0000 | yes | True |
| seed 3 final | 0.02538 | 0.02691 | 0.000 | 0.0000 | yes | True |
| seed 4 best_val | 0.02539 | 0.02622 | 0.000 | 0.0000 | yes | True |
| seed 4 final | 0.02591 | 0.02622 | 0.000 | 0.0000 | yes | True |

| statistic | model k0 | all-MEC | HEFT v2 | MC greedy | all-UE |
|---|---|---|---|---|---|
| mean latency (s) | **0.02583** | 0.02622 | 0.02732 | 0.03478 | 0.27698 |
| hard-violation rate | 0.000 (10/10) | 0.000 | 0.000 | 0.000 | 0.425 |

* k0 range across the ten salvage runs: **0.02538 - 0.02698 s** (mean 0.02583, std 0.00048); k3 mean 0.02689.
* The pilot's *reported* numbers were 0.07876 s (seed 0) and 0.14427 s (seed 1) - **3.0x to 5.6x worse than the cores actually are**. The defect, not the training, produced the pessimistic table.
* Every salvage run has zero hard violations and zero HIGH task tardiness, and k0 is better than k3 in 9/10 cases (3-step adaptation on 20 support graphs slightly hurts).
* Interpretation: the 500-iteration latency-only cores had effectively converged to a near-all-MEC policy that is at or slightly better than the best pure reference on the same instances. This is a *measurement* result, not yet a paper claim: one MC realization per graph, 40 validation graphs, no meta-test.

## Gate I - does mixed placement still pay off under the MC runtime?

Every method is evaluated on the SAME graph and the SAME frozen MC realization; `best_pure` = min over all-UE / all-MEC / all-HELPER, `best_mixed` = min over the MC-aware coordinate-descent greedy and HEFT v2 (`heft_reference_v2`).

### Validation query (n=40, base_seed=303)

| quantity | value |
|---|---|
| mean_all_MEC_s | 0.02622 |
| mean_best_pure_s | 0.02622 |
| mean_best_mixed_s | 0.02679 |
| share_all_MEC_best_pure | 1.00000 |
| share_mixed_strictly_better | 0.60000 |
| mean_improvement_pct | -2.15473 |
| median_improvement_pct | 0.72132 |
| max_improvement_pct | 6.86451 |
| share_best_plan_uses_helper | 0.42500 |
| mean_helper_fraction_in_best_plan | 0.03125 |
| best mixed plan winner | {'greedy_cd': 15, 'heft_v2': 25} |

### Meta-train (n=40, base_seed=101)

| quantity | value |
|---|---|
| mean_all_MEC_s | 0.04708 |
| mean_best_pure_s | 0.04708 |
| mean_best_mixed_s | 0.04758 |
| share_all_MEC_best_pure | 1.00000 |
| share_mixed_strictly_better | 0.52500 |
| mean_improvement_pct | -0.03527 |
| median_improvement_pct | 0.51312 |
| max_improvement_pct | 22.81147 |
| share_best_plan_uses_helper | 0.37500 |
| mean_helper_fraction_in_best_plan | 0.02000 |
| best mixed plan winner | {'greedy_cd': 8, 'heft_v2': 32} |

Verdict:

1. **all-MEC is the best pure plan on 100% of graphs** in both splits - the MC runtime (dropping, HI capping, per-graph physical rates) makes the trivial policy extremely strong.
2. Mixed placement has **instance-level headroom but no average advantage**: it wins on 52.5% (meta-train) / 60.0% (validation) of graphs with a median gain of +0.51% / +0.72% (max +22.8% / +6.9%), yet the mean over the split is -0.04% / -2.15% because the search loses more than it wins on the graphs where it loses.
3. HELPER appears in the winning plan on 37.5% / 42.5% of graphs but only **2.0% / 3.1% of tokens** - V2V/HELPER is a marginal resource here, not the main lever.
4. The learned policy (k0 0.02583 s mean) is *slightly better* than all-MEC (0.02622 s) on the same instances, i.e. whatever advantage it has comes from MC-aware scheduling decisions, not from using HELPER more.

Scientific consequence: a claim of the form 'mixed V2V-aware placement improves the objective' is **not supported by this runtime**. Either (a) scope the claim to latency + deadline/MC feasibility and report the mixed-placement headroom as a negative result, or (b) build a new benchmark version whose MC runtime restores a real mixed-placement advantage (different dropping/capping/budget regime) - never by post-hoc editing v1.

## Gate J recommendation (budget decision)

* The corrected 101-iteration proof already reaches k0 = 0.02547 s with zero violations, and the salvaged 500-iteration cores reach 0.02583 s mean; both are at/inside the reference band. The 500 -> 1500 -> 3500 escalation is therefore **not** justified by a latency gap.

* Recommended order: (1) decide the Gate I claim scope, (2) freeze the evaluation protocol (k, stochastic vs argmax decoding, number of MC realizations per graph - a single realization is the weakest link in every number here), (3) run a fresh 5x500 with the corrected loop as the first scientifically valid pilot, (4) only then decide about a longer budget.

