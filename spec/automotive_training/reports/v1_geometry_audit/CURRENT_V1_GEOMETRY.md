# CURRENT_V1_GEOMETRY (audited)

Scope: frozen MARGO-AUTOMOTIVE-MC-v1 (graphs sha `4341c3b6…`, splits sha `cd391da6…`).
No v1 file modified. Evidence classes: STRONG EMPIRICAL EVIDENCE / SURROGATE SENSITIVITY.

## 1. What the candidate panel says on frozen v1

40 validation-query (lpp) graphs and a family-stratified 40-graph meta-train sample
(20 map + 20 ppc), same MC realizations, candidate panel = {all-UE, all-MEC, all-HELPER,
MC-aware greedy coordinate descent, HEFT v2}.

| split | all-MEC candidate winner | mixed candidate winner | candidate-panel headroom (mean / median / max) | helper token share in winner |
|---|---|---|---|---|
| validation_query (n=40) | **0.400** | **0.600** | **1.96 % / 0.72 % / 6.86 %** | 1.0 % |
| meta_train stratified (n=40) | **0.350** | **0.650** | **5.52 % / 1.19 % / 26.86 %** | 1.0 % |

Winner composition (validation baseline): all-MEC 16, greedy 12, HEFT 12 graphs.
So mixed candidates win often, but usually by little (median 0.72 % on validation);
the mean is carried by a few large wins (max 6.86 %).

## 2. Geometry facts (medians, per task / per edge)

| quantity | median | source |
|---|---|---|
| UE compute | 6.70 ms | `ue_cpu_bytes_per_second` |
| MEC compute | 0.56 ms | `mec_cpu_bytes_per_second` |
| HELPER compute | 4.04 ms | `helper_cpu_bytes_per_second` |
| MEC_UL transfer | 0.56 ms | realized transfers / derived rate |
| MEC_DL transfer | 0.48 ms | realized transfers / derived rate |
| V2V transfer | 1.04 ms | realized transfers / derived rate |
| MEC->UE->HELPER cut | 1.60 ms | MEC_DL + V2V |
| MEC busy (winner plan) | 24.8 ms (val) / 28.0 ms (meta) | `resource_intervals` |
| MEC wait proxy | 0.53 ms (val) / 2.44 ms (meta) | MEC span - busy |

f_MEC/f_HELPER = **6.5** (median over 160 graphs; 8.59 GHz vs 1.32 GHz), f_MEC/f_UE = 11.
MEC wait of ~0.5 ms on validation is direct evidence that the MEC is **effectively
dedicated** in v1 (one DAG per env instance; sampler clones are independent rollouts).

## 3. Rate provenance (see `rate_provenance.csv`, `rate_provenance.json`)

Bits->bytes conversion is exactly 8.0 on all three links: no unit error (ERRATA E2).
The 7-11 Mbps class figures are the documented historical/degraded points; the scheduler
uses per-graph realized rates whose medians are 18.5 / 22.5 / 10.6 Mbps (ul/dl/v2v).

## 4. Sensitivity axes (surrogate unless stated)

| axis | baseline -> effect | class |
|---|---|---|
| MEC contention `mec_share_2/4/8` | validation all-MEC winner 0.400 -> 0.225 / 0.200 / **0.075**; headroom 1.96 % -> 2.86 / 4.56 / **23.26 %**; helper share 1.0 % -> **42.5 %** (meta-train at N=8: all-MEC 0.000, headroom 42.18 %, helper share 55.6 %) | SURROGATE SENSITIVITY |
| helper tier x0.5/2/4 | headroom 1.96 % -> 2.31 / 2.39 / 2.84 % (meta 5.52 -> 5.67 / 6.34 / 7.17) | STRONG (frozen model knob) |
| V2V rate x0.25 / x4 | all-MEC winner 0.500 / 0.275; headroom 1.61 / 2.31 % | STRONG (knob) |
| direct HELPER<->MEC V2I | headroom 1.96 -> 2.28 % (val), 5.52 -> 5.97 % (meta) | SURROGATE topology variant |
| contention4 + helper x2 (+direct) | all-MEC winner 0.075 (val) / 0.000 (meta); headroom 18.70 / 32.36 %; helper share 57.1 / 65.5 % | SURROGATE |

Read: **contention is the dominant axis**; helper tier strength and the 2-hop topology are
second order; V2V rate is a modest lever. All N>1 rows are surrogates.

## 5. Consequences for the v2 plan

1. On frozen v1 the all-MEC collapse is rational: the candidate panel itself prefers
   all-MEC in 60 % of validation graphs and the remaining headroom is ~2 %.
2. The binding mechanisms to fix in v2 are therefore: shared/contended MEC, dynamic link
   state, helper availability/contact, and criticality-aware remote reliability - not PPO.
3. Any v2 claim must be produced by the real shared scheduler (not `mec_share_N`) and
   reported with candidate-panel + stronger-search labels, not "oracle".
