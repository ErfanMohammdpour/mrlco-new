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

## Stage 9 prerequisite landed: v2 macro-step environment

`v2/env.py` composes the frozen `AutomotiveEnv` (MC realization + v1 79-column observation)
with the v2 shared scheduler for execution, reward and telemetry:
* v1-compatible surface (`input_dim`, `total_task`, `set_task`, `reset`, `step`,
  `sample_tasks`, `set_constraint_lambdas`), single-distribution layout so one slot = one
  graph (the layout the frozen evaluation uses);
* telescoping reward under the v2 scheduler (prefix marginal makespans), verified to sum to
  `-(L_final - L0)/L_scale`;
* per-episode telemetry: makespan, queue wait, outage wait, helper rejections/contact
  failures, reliability rejections, fallback reservation, location mix, deadline miss rate
  with HIGH/MEDIUM counts, scheduler invariants;
* `v2_context()` exposes the new context (estimated link multipliers, confidences, helper
  contact remaining, helper busy, reliability epsilon, criticality shares, MEC workers)
  as a 12-vector per slot.
Explicit limitation: the TF observation is still the v1 schema, so until the encoder bump
the policy cannot see the v2 context; a v2 run therefore measures v2 dynamics, not
v2-informed decision making.
Tests: `test_v2_env.py` 11 tests, 3 fresh processes, green.

## v2 observation extension contract (additive, not yet wired)

`v2/observation.py` freezes the extension contract for the encoder bump:
`automotive_v2_obs_v1`, FEATURE_DIM 40 -> 52, PACKED_DIM 79 -> 91, where the first 40/79
columns are the frozen v1 schema untouched (verified bit-for-bit by `split_v2_row`) and the
12 appended columns are the v2 context (estimated link multipliers, link confidences, helper
contact remaining, helper busy, reliability epsilon, criticality shares, MEC workers).
`write_v2_stats_file()` derives `encoder_feature_stats_automotive_v2.json` (52 entries) from
the frozen 40-entry artifact, with mean 0 / std 1 for the 12 identity-normalised columns.

NOT wired into the frozen packer yet: `encoder_obs.set_obs_version` still rejects
`automotive_v2_obs_v1` (asserted by test) because adding it changes FEATURE_DIM/PACKED_DIM
at import time and therefore must land together with the trainer bridge, in one reviewed
step, so no half-wired state can exist. Until then the policy input remains the v1 schema
and `v2_context()` is telemetry only.

## Observation version wired (automotive_v2_obs_v1) with v1 invariance proven

`encoder_obs` now carries a fifth version: FEATURE_DIM 52, PACKED_DIM 91, layout
`[features(40) | v2_context(12) | fw(19) | bw(19) | mask(1)]`. The context block is written
only for that version; the MC block is written for both automotive versions.

Evidence:
* `V1_OBS_GOLDEN.json` (captured before the edit through the real `AutomotiveEnv._observation`
  path, base_seed 303, validation_query()[:3]): all three sha16 hashes UNCHANGED after the
  edit -> the frozen v1 observations are byte-identical.
* equality invariant: `encode(v2, mc_context={... v2_context=ctx}) == pack_v2_row(encode(v1), ctx)`
  and `v1_form_of_v2_row(v2) == v1` (both exact), asserted by test.
* the v2 stats artifact loads for the new version (52 names, mean/std 52).
* REGRESSION FOUND AND FIXED: adding the v2 context names to the module-level
  `NON_STANDARDIZED_FEATURES` made the frozen v1 stats tool report the v1 file as STALE.
  The exclusion is now version-scoped (`NON_STANDARDIZED_FEATURES_V2` used only by the v2
  version), the tool reports "up to date" again, and the v1 module state is untouched.

## v2 train/val stack wired (trainer bridge)

`v2/stack.py::build_automotive_v2_stack` mirrors `build_automotive_primary_stack` exactly
(same frozen budgets: meta_batch 10, support 20, 3 inner applies, same policy/sampler/
MRLCO/processor chain, same validation split guard) and changes only:
* env family `AutomotiveEnv` -> `V2AutomotiveEnv` (v2 shared scheduler dynamics),
* obs version -> `automotive_v2_obs_v1` (set before the policy import),
* run/method ids, v2 system config and the CRN protocol recorded on the trainer
  (`auto_protocol_id = automotive_crn_gumbel_v1`, `auto_crn = {r_select, s_select}`).
`meta_trainer.build_frozen_primary_stack` routes `dataset="automotive_mc_v2"` to it, with
`MARGO_V2_LINK_REGIME` / `MARGO_V2_MEC_WORKERS` / `MARGO_V2_RELIABILITY` selecting the
system configuration. `V2AutomotiveEnv.__getattr__` delegates the v1 env surface
(configs, graph_objects, orders, dags, graph_indices, encoder_batchs, ...) so the frozen
builder/evaluator keep working; unknown attributes still raise.

Tests: `test_v2_stack.py` 5 tests x3 fresh processes (routing, pre-TF guards, regime list,
obs version, delegation, AttributeError behaviour); full non-TF suite 1249 passed / 0 failed.

RESOLVED: the concern above was real. `AutomotiveHeldOutEvaluator` built `AutomotiveEnv`
directly at three sites (`_env`, the single-dist rollout env, `_paired_env`), so v2
validation would have measured v1 dynamics. The evaluator now takes an optional
`env_factory=None` and routes all three sites through `_make_env(...)`; the default path is
byte-identical (still `AutomotiveEnv`, single_dist handled internally, input_dim 79), and
`build_automotive_v2_stack` injects `v2_env_factory` returning `V2AutomotiveEnv` with the
configured link regime / MEC workers / reliability. Proven without TF: with a factory the
evaluator returns `V2AutomotiveEnv` for both the plain and the single-dist paired layout
(factory call log `[(4, 4, 7, True)]`), and without a factory it returns the frozen
`AutomotiveEnv`. Tests: 8 stack tests (incl. 3 for this hook) x3 fresh processes; full
non-TF suite 1252 passed / 0 failed.

## Stage 8 (partial): adversarial fixtures + branch pushed

`test_v2_adversarial.py` (8 fixtures, 3 fresh processes, 1.5 s): 32-way MEC contention makes
local win; idle MEC beats local on a MEC-heavy graph; a helper with ~zero contact is rejected
(not crashed) and every completion stays finite; 0.1x V2V removes the helper advantage;
HIGH criticality under an unreliable link rejects every remote token (all locations UE);
a degraded-regime outage episode stays finite and logs non-negative outage waits; single-task
and all-local plans keep every scheduler invariant; the contention ordering is monotone in
background load (0 <= 4 <= 16).

Branch `phase5-realistic-system-v2` pushed to the `erfan` remote (tracking set). Full non-TF
suite 1260 passed / 0 failed / 18 skipped. Still open in stage 8: the transfer-count
quantification versus v1 and the ">=3 fresh processes for every v2 suite" record (currently
recorded per suite as it was added).

## Stage 8: v1<->v2 transfer accounting quantified (was an open item)

`V1_V2_TRANSFER_ACCOUNTING.json` (4 validation graphs x 3 plans, same MC realization):

| plan | v1 records | v2 events | v1 bytes | v2 bytes |
|---|---|---|---|---|
| all_MEC | 2 | 1 | 3385 | 3200 |
| half MEC/UE | 10 | 10 | 70380 | 70380 |
| half MEC/HELPER | 22 | 21 | 144145 | 143960 |

Finding: the raw-count difference is NOT a physics difference. v1 emits one transfer record per
graph EDGE (with a hop count inside), v2 emits one booking per HOP; on single-hop plans the
counts match exactly (10/10) and the byte totals match exactly. The only byte discrepancies
(8 of 12 rows) are exactly 185 B each: v1 books a second, very small sink return that v2 does
not book. Impact bound: 185 B / ~20 Mbps ~ 74 us against a ~25 ms makespan (< 0.3 %), so it
cannot carry the geometry-gate headroom; but the cause is NOT yet identified (candidate:
differing MC survivor sets for the tiny-output sink), so it stays OPEN and is not used as an
explanation for any v2 result.

Also in this stage: `V2ScheduleResult` now exposes the booked `radio_events` and
`mechanics["radio_bytes"]` (needed for the accounting above); 49 v2 tests re-run green.
