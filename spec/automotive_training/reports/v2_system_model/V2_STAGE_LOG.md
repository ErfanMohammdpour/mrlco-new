# v2 stage log (per-stage evidence)

| stage | commit | tests run | result | files | meta_test |
|---|---|---|---|---|---|
| 1 v1 geometry/rate audit + errata | `fdfc7a2` | `build_v1_geometry_audit.py` (160 graphs) | OK | `reports/v1_geometry_audit/*`, `geometry_sensitivity.py`, `reports/gateI/geometry_sensitivity.json` | 0 |
| 2 shared multi-DAG MEC scheduler | `596b885` | `test_v2_shared_scheduler.py` x3 fresh procs = 9/9 each; full non-TF suite 1183 passed / 0 failed | OK | `v2/{__init__,shared_scheduler,adapters}.py`, `tests/test_v2_shared_scheduler.py` | 0 |
| 3 dynamic links (estimated vs realized) | `901f342` | `test_v2_link_model.py` + `test_v2_shared_scheduler.py` = 17/17 x3 fresh procs; full non-TF 1191 passed / 0 failed | OK | `v2/{link_model.py,link_regimes.yaml}`, scheduler link-process hook, `tests/test_v2_link_model.py` | 0 |
| 4 helper availability/contact/busy | `9d0afcc` | `test_v2_helper_model.py` (10) + scheduler/link suites = 28/28 x3 fresh procs; full non-TF 1202 passed / 0 failed | OK - plus a real calendar fix | `v2/helper_model.py`, `v2/shared_scheduler.py` (earliest-fit calendars, helper admissibility/failure), `tests/test_v2_helper_model.py` | 0 |
| 5 criticality reliability + fallback hooks | staged | `test_v2_reliability.py` (9) + all v2 suites = 37/37 x3 fresh procs; full non-TF 1211 passed / 0 failed | OK | `v2/{reliability.py,reliability_classes.yaml}`, scheduler gate + standby hook, `tests/test_v2_reliability.py` | 0 |
| 6 geometry gate + stronger search | `7719661`, `8baf4db`, `2f7d072` | `test_v2_search_and_gate.py` + all v2 suites = 41/41 x3 fresh procs; full non-TF 1215 passed / 0 failed | gate harness ready; full 20x12 run in flight | `v2/{geometry_gate.py,stronger_search.py,heft_bridge.py}`, `tests/test_v2_search_and_gate.py` | 0 |
| 7 CRN evaluator (Gumbel) | `3993ead` + fix | `test_v2_crn.py` 11 x3 fresh procs; full non-TF 1226 passed / 0 failed / 18 skipped | protocol + stateless policy path done; TF-gated container check pending | `v2/crn.py`, `policies/meta_seq2seq_policy.py` (opt-in CRN helper), `tests/test_v2_crn.py`, `tests/test_v2_crn_tf.py` | 0 |
| 8 repeated/adversarial/v1-parity battery | pending | - | - | - | 0 |
| 9 v2 1x500 + checkpoint eval | BLOCKED until gate 6 passes | - | - | - | 0 |

## v1 corrections delivered in stage 1 (see V1_ERRATA.md)

1. `P(all-MEC candidate winner)` = **0.400 (validation) / 0.350 (meta-train)**; mixed = 0.600 / 0.650. An earlier sentence had this inverted.
2. "oracle" renamed to **candidate-panel oracle** (5 candidates); headroom = **candidate-panel headroom**, not an upper bound.
3. No bits/bytes unit error: conversion is exactly 8.0 on every link (160 graphs). The 7-11 Mbps figures are the documented historical/degraded points; per-graph realized medians are 18.5 / 22.5 / 10.6 Mbps (ul/dl/v2v), profile-stratified in `V1_RATE_PROVENANCE.csv`.
4. `mec_share_N` rows relabelled **SURROGATE CONTENTION SENSITIVITY** (processor sharing), not a multi-user simulation.
5. Earlier "16x unit error" and the single-graph rate quote are retracted in the errata.

## ERRATUM (stage 6, first run)

The first full gate run wrote `GEOMETRY_GATE.json` with `verdict.gate = PASS`, but its
**MEC-load axis was a no-op**: `_with_background` built the N-1 background DAGs and was
never called, so regimes A-D were identical and the load axis carried no signal. The
verdict from that run is **INVALID**. Fix: the background DAGs are now scheduled in the
same `schedule_shared` batch as the foreground and the reported time is the foreground
DAG's completion (`completion_by_dag["fg"]`). A corrected 20-graph x 12-regime run is in
flight; only its verdict may be reported, and only together with both
`all_MEC_winner_fraction` and `all_MEC_winner_fraction_no_search` (the latter excludes the
search-based candidate, which by construction starts from all-MEC).

## Parity finding (stage 6 follow-up, `V1_V2_PARITY.json`)

The gate's headroom must be read against a measured v1<->v2 model difference. Single-DAG,
same plan, same MC realization, helper present with unlimited contact (as in v1):

| plan | mean v2-v1 | median | max abs | v2 faster |
|---|---|---|---|---|
| all_UE | -0.00 % | +0.00 % | 0.00 % | 0.00 |
| all_MEC | **-1.33 %** | -1.47 % | 1.98 % | 1.00 |
| all_HELPER | -0.66 % | -0.50 % | 1.55 % | 1.00 |
| alternate MEC/HELPER | **+2.06 %** | +2.24 % | 7.90 % | 0.38 |
| alternate MEC/UE | +0.26 % | +0.00 % | 1.01 % | 0.00 |
| front MEC/back UE | +0.12 % | +0.13 % | 0.60 % | 0.38 |

So v2 is within about +/-2 % of v1 on realistic plans, with a small systematic bias
(MEC-heavy ~1.3 % faster in v2, mixed MEC/HELPER ~2 % slower). Transfer-count accounting
still differs on 24/48 rows and must be quantified before any "richer geometry" claim;
the gate headroom is therefore reported as a MODEL+PLAN-SEARCH difference, not as pure
geometric headroom.

## Gate outcome (stage 6, corrected run) - PASS

`GEOMETRY_GATE.json` / `GEOMETRY_GATE.md`, 20 graphs per regime, corrected helper
availability and real background MEC load:

```
A idle      7.5%   B N=2 11.4%   C N=4 14.5%   D N=8 31.3% (search gain 38.7%)
E helper idle 7.5%  F busy 6.1%  G short 7.5%  H poor V2V 6.3%  I variance 5.7%
J HIGH+unreliable: local wins 100%   K HIGH+reliable idle MEC 7.1%   L LOW+good remote: all-MEC 100%
gate = PASS (BLOCK rule = all-MEC >0.80 winner in EVERY regime -> false)
```

Stated limits carried forward: no helper/V2V claim (HELPER-only plans never win; helper
regimes barely differ), contention is the supported mechanism, and the absolute gap to v1
includes a measured model difference (+/-2% single-DAG parity bias, transfer accounting
differs on 24/48 rows).

## Stage-7 notes and open scoping item

Done: the CRN protocol (`automotive_crn_gumbel_v1`) with deterministic Gumbel keys,
Gumbel-max actions, nested R x S aggregation and paired deltas; plus an opt-in stateless
sampling path in `Seq2SeqPolicy` (`enable_crn=False` by default, so the frozen v1 graph is
untouched - the full suite still passes).

Test-isolation defect found and fixed: some suite modules install a lightweight
`tensorflow` stub in `sys.modules`, so a bare `import tensorflow` made the TF-gated tests
run against the stub and fail (`3 failed`) when the suite ran as a whole while passing
(skipping) in isolation. The gate now requires real TF1.15 APIs
(`tf.compat.v1`, `tf.random.stateless_multinomial`, a version string).

Open scoping item for stage 9: stages 3-6 built a **scheduler-level** v2 system model
(numpy) plus the geometry gate. Running the v2 1x500 requires a v2 *training-integration*
layer that does not exist yet: a v2 env exposing the 20-token plan API with the v2
observation (estimated link state, helper contact/occupancy, reliability context), the v2
reward/telemetry from the shared scheduler, and the trainer/sampler bridge. That is a new
build item (comparable in size to the v1 automotive integration) and it is the next major
work item; the geometry gate's PASS permits it but nothing on the GPU may start before it
land with its own tests.
