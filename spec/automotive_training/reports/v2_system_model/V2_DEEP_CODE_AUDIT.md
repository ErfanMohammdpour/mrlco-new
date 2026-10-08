# V2_DEEP_CODE_AUDIT — deep audit, repair, and evidence-based readiness

> **CURRENT STATUS (tested HEAD `1e1aa5962ed345af6ffa749cb370c7234bbf98eb`).** F33 exact
> fresh-process continuation is now **SOLVED and measured**: control vs a two-iteration process
> resumed in a fresh process to total target three produce **identical** third-iteration records
> (`mean_system_joules` 110.8925 both, `mean_violation` 0.50871 both, both lambdas equal to 8
> decimals, 200 penalised episodes both, `EXACT MATCH: True`). Sections below that describe a
> "34 % divergence", an "unexplained zero-penalty" guard, or the earlier 0.06-4.8 % residual are
> **SUPERSEDED** and retained only as history; see section 0 for the resolution chain.
>
> Superseded status statements, by date: the 34 % resume gap (2026-10-07, revision `a9e8f10`),
> the unconditional penalty guard (2026-10-07, `4771b1e`), the "0.06-4.8 % unexplained" residual
> (2026-10-08, `dcae810`), and the "no feasible witness" claim withdrawn the same day.
> **Still open** (not superseded): F06/F11 unit consistency, the F13 weighted-objective
> resolution, F08's remaining fallback accounting, F17-F29 contract executions, and F34 measured
> kernel placement. These are listed in the matrix below.

## 0. F33 resolution chain (each step measured, in order)

| # | defect | evidence | status |
|---|---|---|---|
| 1 | world/episode identity not restored; the counters live on the sampler's **deep-copied clones**, built before any restore | `world_id` control `ep11_slot19` vs resume `ep3_slot19` | FIXED |
| 2 | multipliers broadcast only at iteration END, so the first post-resume iteration collected with a stale zero dual | resumed itr-2 penalised 0 episodes while violation and lambda were positive | FIXED |
| 3 | `sampler_counters` written into the bundle but **never applied** | all counters zero at the resumed boundary | FIXED |
| 4 | stdlib `random` stream never carried, although the stack seeds it | B1 trace digest differed | FIXED |
| 5 | per-clone `task_id`/`graph_indices` not restored | B1 trace: control 117/84/... vs resume -1 | FIXED |
| 6 | **ROOT CAUSE** — training called `get_actions(obs)` with no `crn_seed`, so it took the **stateful** `Categorical(...).sample(seed=...)` branch; the TF op-level random stream's advanced draw history is not in any `Saver` | source `seq2seq_meta_sampler.py:95` + `meta_seq2seq_policy.py:158`; residual survived a bit-exact variable restore | FIXED |
| 7 | my own restore wrote the draw cursor to the **wrapper**, not the inner sampler (`__getattr__` forwards reads, not writes) | `_draw_cursor: 4` in the bundle but the resumed draw sequence restarted | FIXED |

Evidence standard used throughout: two uninterrupted controls on the fixed CPU configuration are
**bit-identical**, so no tolerance was enlarged to absorb a difference; and all 1043 shared TF
variable values were compared (not counted), which refuted the optimizer-slot hypothesis by value.

Sampling-contract version: `crn_token_keyed_v2` (`FixedSequenceLearningCRNHelper`). It also fixes
F20: the previous helper reused ONE seed pair at every token, replaying the same draw across the
horizon. The draw law is unchanged (a categorical draw from the same masked softmax), masks /
temperature / top-p preserved, `enable_crn` defaults to False so V1 keeps historical sampling.

 — deep audit, repair, and evidence-based readiness

**Audited and tested revision: `a9e8f1049bcd16a7c2ae22f3eae5e719755ac7ea`** (this document).

## 0. Revision and execution provenance (resolved)

| role | FULL SHA | date | note |
|---|---|---|---|
| base | `0f9cb096b76a84d3c7b260d539b4e2ad15dceac6` | 2026-10-03 23:14 | "identity-keyed environmental realizations…" |
| **historical 500-run source** | `49e5cb832110ab6528d7ad767c5ba9d0d66d5ebb` | 2026-10-05 02:24 | `feat(v2): emit per-constraint measurements…` |
| repair (prior session) | `e07ba6a66e16c244d217ecaafcf6032104f2a8e8` | 2026-10-06 14:57 | model energy discarded every rollout |
| **audited HEAD** | `a9e8f1049bcd16a7c2ae22f3eae5e719755ac7ea` | 2026-10-07 12:07 | F01/F02/F07 repairs |

All four are ancestors of HEAD (`git merge-base --is-ancestor` = YES for each). Working tree clean.

**Repository / remotes.** The mirror's only remote is a local git bundle
(`_scratch/repo/v2.bundle`), created from the server checkout. The server git repo is
`/root/v2-verify` with remotes `origin https://github.com/MA-RGO/MARGO.git` and
`erfan https://github.com/ErfanMohammdpour/mrlco-new.git`; its HEAD is `0f9cb096`. The
independently retrieved SHA `4a4e40f160e153b0dbf7baf134254ab12b77cacd` is **not** object-present in
either the mirror or `/root/v2-verify`; it could not be reconciled here. `e07ba6a` / `a9e8f10` are
present only in this mirror (they were created here). **No test result, timing or verdict in this
document is transferred from `4a4e40f`.**

**Provenance of the historical 500-run source (`49e5cb8`) — F01/F02 were LIVE in it:**
```
git show 49e5cb8:spec/automotive_training/v2/env.py        -> line 392: dataset_graph_id = int(indices[flat])
git show 49e5cb8:spec/automotive_training/automotive_env.py -> line 266: for slot in self.graph_indices:
                                                               line 268: seed = self.rollout_seed(graph.graph_id, int(slot))
```
Both sites are byte-identical at the base `0f9cb096` as well (lines 365 / 266). The 500-run's
checkpoints are therefore **products of the defective environment**; they are preserved and may be
used for diagnostic re-evaluation only, never as evidence of training the repaired system.

**Container used for every executed TF/GPU check** (recorded from the image):
`margo-phase4-tf115-nv2212:latest` @ `sha256:82bc781bc2888a171379e1b45da6846c559ef712e33a76f74948b1cd7172dc46`,
Python 3.8.10, TensorFlow 1.15.5, `tf.contrib` importable, CUDA 11.8, NumPy per that image.
Host `aizarepor` (ssh `kish-ai`), RTX 4090, co-resident vLLM left untouched.

---

## 1. Corrections to the previous report (read before any configuration change)

### C1 — F10 was WRONG and is withdrawn
The previous report claimed the energy budget was infeasible because "measured all-MEC energy
equals its anchor, so g(all-MEC)=+0.5 ⇒ fraction ≥ 1.0 is required". **Both halves are wrong.**

1. The claim tested the wrong quantity. `_score` (eval_loop) computes `schedule_energy(result)`
   with **no `dag_filter`** (unfiltered, includes background), while the constraint measures
   `schedule_energy(..., dag_filter=[foreground_id])`. Comparing those two produced a spurious
   "3.06× anchor mismatch". Measured correctly, the anchor matches the realized plan:
   ```
   E_mec (anchor, background-free reference)             = 20.059 J
   all-MEC measured (background=2, FOREGROUND-filtered)  = 20.089 J   ratio 1.001x
   E_ue (anchor) = 0.192 J ; all-UE measured (FG-filter) = 0.192 J    ratio 1.000x
   ```
2. Even if all-MEC violates, that alone proves nothing about the feasible set. A TRAIN-ONLY study
   (12 `meta_train` graphs, background=2, moderate link, canonical v2 scorer, `fraction=0.5`,
   B = 10.0295 J) shows the feasible set is **large**:

| candidate | fg energy J | latency s | energy OK | joint OK |
|---|---|---|---|---|
| `all_UE` | 0.192 | 0.1919 | 12/12 | 12/12 |
| `all_HELPER` | 2.124 | 0.1339 | 12/12 | 12/12 |
| `mix_mec25` | 2.145 | 0.2103 | 12/12 | 12/12 |
| `rand_01` | 7.297 | 0.2752 | 12/12 | 12/12 |
| `greedy_cd` | 9.195 | 0.1804 | 4/12 | 4/12 |
| `mix_mec50` | 10.140 | 0.1215 | 6/12 | 6/12 |
| `rand_00` | 9.436 | 0.3446 | 11/12 | 11/12 |
| `mix_mec75` | 14.493 | 0.0783 | 0/12 | 0/12 |
| `heft` | 18.255 | 0.2096 | 0/12 | 0/12 |
| `all_MEC` | 20.089 | 0.0437 | 0/12 | 0/12 |

**Feasible witness exists** (all-UE at 0.192 J; also all-HELPER, 25 %-MEC mixes and random mixes).
**Proven infeasibility bound: none.** Excluding all-MEC is a legitimate design choice for a
constrained formulation — it trades latency for energy, which is the intended behaviour.
`fraction ≥ 1.0` is **not** required. No calibration is required on infeasibility grounds, and the
fraction was **not** changed.

**Binding-axis caveat:** on these 12 train graphs every candidate has `miss_rate = 0` and
`hard = 0`, so the deadline axis is **not** binding on this fixture. The joint column therefore
equals the energy column here. A harder fixture (tighter deadlines or higher load) is needed to
make the deadline axis informative; that is a limitation of this study, stated rather than hidden.

### C2 — F11 ratio direction was inverted
Correct relation, for a common workload unit with `c` cycles/bit:
```
R_physical        = f / (8*c)
E_duration/E_work = R_physical / R_scheduled      <-- this, not its inverse
```
Verified numerically against the shipped coefficients:
```
UE  : R_phys/R_sched = 416666.7/1048576 = 0.39744  == E_dur/E_work 0.39738
HLP : 625000.0/1048576                  = 0.59605  == 0.59600
MEC : 4166666.7/10485760                = 0.39744  == 0.39738
```
The ledger **charges** the duration form (`compute_joules_from_duration`) and only tracks the work
form (`cpu_work_form_j`). Timing rates come from per-graph resource frequencies; energy prices use
the frozen tier frequencies. The contract conflict stands; the previous report's explanation of the
ratio (R_scheduled/R_physical) was wrong.

### C3 — F02 proves MC-realization reuse, not identical trajectories
Per-stream independent unit, from the code:

| stream | keying | pre-fix independence |
|---|---|---|
| MC realization | `rollout_seed(graph_id, int(slot))` — the `graph_indices` **value** | **1 per meta-task** (defect) |
| helper state | `stable_seed(base_seed, "helper", world_key, flat)` | already 1 per slot |
| background workload | `world_id = "w_<key>_slot<flat>"` | already 1 per slot |
| TF policy sampling | graph2seq/decoder RNG | not audited on this axis (NOT_VERIFIED) |

F02 therefore proves the **MC axis** collapsed to 1 effective sample per meta-task. It does **not**
prove identical full trajectories, and it does not license a claim about the other streams.

### C4 — Evidence counts, exactly
The previous report said "8 new tests, all fail on the pre-fix code". **Correct: 8 new tests,
4 fail pre-fix.** Raw before/after (parent `e07ba6a` file versions vs HEAD):
```
pre-fix : 4 failed, 4 passed
post-fix: 8 passed
```
Exact failing test IDs:
```
test_v2_graph_identity.py::GraphIdentityTests::test_non_single_dist_world_equals_task_graph
test_v2_graph_identity.py::GraphIdentityTests::test_nonzero_task_ids_across_the_corpus
test_v2_graph_identity.py::MCIndependenceTests::test_repeated_slots_draw_independent_realizations
test_v2_helper_return_queue.py::HelperReturnQueueTests::test_return_behind_another_v2v_booking_falls_back
```

### C5 — Learning verdict must not mix environments
Historical metrics (`k=0` 0.0806 s, `all_MEC` 2.69×, the "−66 % validation latency") describe the
**old defective environment** (F01/F02 live). Repaired-training performance is
**NOT_YET_MEASURED**. Old checkpoints are retained for diagnostic re-evaluation only.

### C6 — The anchor knob was renamed, not reinterpreted
`total_energy_frac_of_all_ue` multiplies `refs.E_ue` **literally** (V1 semantics preserved). The
all-MEC anchor is a NEW, explicitly named field `total_energy_frac_of_all_mec`, selected by the new
mapping name `total_energy_mec`, and it changes the constraint fingerprint
(`c06c0598…` vs `fa3668af…`). The production default and `run_tf.py` now pass `total_energy_mec`.

---

## Highest-priority findings (reproduced first)

### F01 — selected/observed/scheduled graph can disagree — **REPRODUCED → FIXED**
- **Severity:** CRITICAL (data-contract). Affects the actual 500-run.
- **Path:** `spec/automotive_training/v2/env.py::_build_slot_world`; base `automotive_env.py::_graph_index/_draw_slots`.
- **Proof:** with `single_dist=False` (the training default), `sample_tasks` sets
  `graph_indices=zeros(slots_per_task)` and `_graph_index(slot)` returns `task_id`. The v2 world
  builder read `graph_indices[flat]` directly and therefore built **graph0** for every meta-task,
  while the base env drew the MC realization for the task's own graph.
  Reproduction (task_id=2, graph_indices=[0,0]):
  ```
  world_graph = mcv1_lpp_s0_whigh_dloose_rhigh_end  (D_G=0.34728)
  mc_graph    = mcv1_lpp_s0_whigh_dtight_rhigh_end  (D_G=0.18469)   MATCH=False
  ```
- **Fix:** `_build_slot_world` now resolves the dataset graph through the base `_graph_index(flat)`
  (the single authoritative resolver that understands BOTH layouts).
- **Regression:** `test_v2_graph_identity.py` (4 tests; 2 fail on pre-fix code).
- **Affected claims / old results:** the 500-run scheduled graph0's world for every meta-task
  while drawing MC from the task graph — its latency/energy/reward numbers are NOT valid evidence
  for the repaired environment. **Old checkpoints require re-training; do not reinterpret them.**

### F02 — repeated support rows reuse MC seeds — **REPRODUCED → FIXED**
- **Severity:** CRITICAL (independence). Affects the actual 500-run.
- **Path:** `automotive_env.py::_draw_slots`.
- **Proof:** with `graph_indices=[0]*20`, `rollout_seed(graph_id, int(slot))` used the VALUE 0 for
  all 20 slots → **1 unique seed / 1 unique realization** across 20 "independent" trajectories.
- **Fix:** seed by the FLAT slot position (`enumerate`) while resolving the graph separately.
- **Regression:** `test_v2_graph_identity.py::MCIndependenceTests` (2 tests; 1 fails pre-fix).
- **Affected claims:** the 500-run's "20 support trajectories per task" were 20 copies of one
  realization → effective independent count was 1 per meta-task. Variance-based claims collapse.

### F07 — helper return ignores radio queue; validator accepts it — **REPRODUCED → FIXED**
- **Severity:** CRITICAL (a wrong success verdict: helper result delivered after disconnection).
- **Path:** `shared_scheduler.py` helper return feasibility check (~:632) + `validate_schedule`.
- **Proof:** background V2V booking occupied [0,0.4]; the foreground helper's 10 B return was
  planned with the queue-free `_service_detail` and the validator passed with
  `location=HELPER, contact_failure=False` even though the return would finish after contact end.
- **Fix:** the return is now planned against the ACTUAL shared calendar via a non-mutating
  `_planned_delivery` (uses `Calendar.plan_service`, the same fixed point `book()` uses). The
  reproduction now falls back to UE with `contact_failure=True`.
- **Regression:** `test_v2_helper_return_queue.py` (2 tests; 1 fails pre-fix).

### F09 — world energy constrained against a background-free reference — **ALREADY_FIXED_WITH_PROOF**
- **Path:** `v2/env.py::step` + `constraints.py::budgets_and_scales`.
- **Fix:** `step` evaluates the constraint on the foreground-only ledger
  (`schedule_energy(..., dag_filter=[world.foreground_id])`); the total-energy anchor is the
  background-free pure all-MEC plan. Both sides are now the same DAG set and the same boundary.
- **Measured validation (TRAIN-ONLY, 12 graphs, background=2):** the anchor reproduces the realized
  plan to within 0.1 %:
  ```
  E_mec (anchor)  = 20.059 J   vs   all-MEC measured (FG-filtered) = 20.089 J   ratio 1.001x
  E_ue  (anchor)  =  0.192 J   vs   all-UE  measured (FG-filtered) =  0.192 J   ratio 1.000x
  ```
- **Residual (stated, not hidden):** the reference world is deliberately background-free, so it does
  not carry the competitor load; the measured plan does. The 0.1 % agreement shows this does not
  distort the FOREGROUND ledger on this fixture (UE CPU is per-owner and the foreground ledger is
  filtered), but a fixture with heavy shared-MEC contention could reopen the gap. Not re-verified
  under a high-contention setting.

### F10 — energy budget feasibility — **CORRECTED: NOT infeasible** (see C1)
- The previous "proven infeasible / fraction ≥ 1.0 required" claim is **withdrawn**.
- Measured TRAIN-ONLY: feasible witness exists (all-UE 0.192 J vs B = 10.03 J; also all-HELPER,
  25 %-MEC, random mixes). all-MEC (20.089 J) and HEFT (18.255 J) are excluded, which is a
  legitimate trade-off, not a defect.
- Remaining genuine defect: the knob/name mismatch, now fixed by C6.
- Limitation: the deadline axis is non-binding on the studied fixture.

### F11 — physical work/time/energy identities disagree — **REPRODUCED (units)**
- **Path:** `energy_model.py::compute_joules` vs `compute_joules_from_duration`; scheduler rates
  derived from per-graph resource frequencies vs frozen tier frequencies.
- **Proof (analytical, exact):** with `cycles_per_bit=300`, tier f = 1e9 / 1.5e9 / 1e10,
  scheduler rates 1048576 / 1048576 / 10485760:
  ```
  E_dur/E_work = 0.3974 / 0.5960 / 0.3974   (matches the finding verbatim)
  ```
  The ledger CHARGES the duration form (`compute_joules_from_duration`) and only TRACKS the work
  form (`cpu_work_form_j`). The disagreement is `R_scheduled / R_physical` because the scheduler
  rates come from per-graph frequencies that differ from the frozen energy-tier frequencies.
- **Status:** CONTRACT_CONFLICT. `compute_bytes` is an equivalent-work unit (from MC
  `effective_equiv`), but the tier frequency used for timing vs energy pricing is inconsistent.
  Documented; the correct mapping must be versioned before energy comparisons are trusted.

### F12 — average energy multiplies episode joules by 20 — **REPRODUCED → FIXED (V2-scoped)**
- **Path:** `v2/env.py::step` fills `energy_batch` with `np.full(len(rewards), episode_energy)`;
  `meta_trainer.py:581` summed that vector.
- **Measured:** episode `system_joules` 0.118 J → `sum(energy_batch)` 2.368 J = 20 × 0.118.
- **Fix (V2-scoped, V1 byte-for-byte preserved):** the env declares
  `energy_channel_semantics = "episode_replicated"`; `energy_telemetry.aggregate_episode_energy`
  takes ONE value per trajectory for that marker and **asserts** the tokens within a path are
  identical (a wrong marker fails loudly rather than reporting a mis-scaled mean); the trainer
  branches on the declared semantics and logs `energy/v2_episode_system_joules_mean` with an
  explicit `system_foreground` scope, so the V2 number is never described only by the legacy
  MOBILE label. V1 leaves the marker absent and keeps the historical per-token sum exactly.
- **Regression:** `test_v2_energy_aggregation.py` (6 tests; the marker test fails pre-fix).

### F13 — dense shaping and gamma mismatch the objective — **REPRODUCED → FIXED**
- **Path:** `v2/env.py::_telescoping`; `v2/constraints_v2.py::apply_penalty`; `v2/stack.py:177`.
- **Fix:** `_telescoping` now emits the contract form `r_t = J_{t-1} - gamma*J_t`
  (`J = L/L_scale`, `gamma = shaping_discount`, validated in (0,1]). The processor applies the
  same gamma on top; that combination IS the contract (`spec/objective_contract.py:22`), not a
  double discount. The terminal constraint penalty is divided by `gamma^(N-1)` so its contribution
  to the discounted return is exactly `-penalty` (otherwise the declared budget would be
  mis-scaled by 0.826 at N=20).
- **Contract identities now hold:**
  ```
  discounted   sum_t gamma^(t-1) r_t = J_0 - gamma^N * J_N      (path-independent)
  undiscounted sum_t r_t             = legacy_sum_from_potentials(potentials, gamma)
  ```
- **Regression:** `test_v2_discount_contract.py` (6 tests; the env-level test fails pre-fix).
- Three pre-existing tests encoded the old semantics and were corrected with the reasoning inline
  (`test_v2_env` now asserts the two contract identities; `test_v2_constraints` asserts the
  DISCOUNTED charge is exactly `-penalty`).

### F06 — service integration skips rate-change boundaries — **REPRODUCED → FIXED**
- **Path:** `shared_scheduler.py::_service_detail`; `link_model.py::_idx`.
- **Fix:** every integration step now terminates at the NEXT TRACE BOUNDARY (or outage transition
  or payload completion); `_idx` snaps exact grid points (`+1e-9`, because `0.3/0.1 =
  2.9999999999999996` selected the PREVIOUS interval); and the finite horizon is an explicit
  error (`_declared_horizon`) instead of silently extending the final rate sample forever.
- **Before/after on the exact finding fixture** (start 0.05 s, dt 0.1 s, 20 B, base 100 B/s, x10
  from 0.1 s; analytic 0.115 s):
  ```
  PRE-FIX : booked [0.0500, 0.1600]  completion 0.1600 s
  POST-FIX: booked [0.0500, 0.1150]  completion 0.1150 s
  ```
- **Regression:** `test_v2_service_boundary.py` (6 tests; 2 fail pre-fix).
- **Third finding from the same fix:** an adversarial test expected a 1.0 s outage to cost 1.0 s;
  the true overlap from a 0.1 s start against an outage on `t < 1.0` is **0.9 s**, and the old 1.0
  was a floating-point artefact (repeated `t += dt` accumulated to `0.9999999999999999`, so the
  last sample was still classified as an outage). The test now asserts 0.9 with the reason inline.

### F21 — NumPy categorical log_likelihood broken — **REPRODUCED → FIXED (was dead code)**
- **Path:** `policies/distributions/categorical_pd.py::log_likelihood`.
- **Three defects:** normalised with the Python builtin `sum` (wrong axis for batched `[B,C]`
  logits); took `log(one_hot * p)` so `log(0) = -inf` collapsed every entry; built `final_shape`
  from `list.append` (returns None) making the reshape a no-op.
- **Fix:** stable log-softmax (`logits - logsumexp`) gathered at the selected action, with
  action-axis and index-range validation.
- **Before/after, in the TF1 container:**
  ```
  PRE-FIX : Ran 8 tests -> FAILED (failures=5, errors=2)     exact `[-inf, -inf]` reproduced
  POST-FIX: Ran 8 tests -> OK
  ```
- **Regression:** `test_categorical_log_likelihood.py` (8 tests, TF-gated; includes agreement with
  the TF symbolic path on random logits/actions).
- **Reachability unchanged:** ZERO call sites; PPO uses the correct TF `log_likelihood_sym`. This
  defect did **not** cause any TF learning failure and is not offered as an explanation for one.

### F08 — validator did not check the helper contact window — **REPRODUCED → FIXED (validator half)**
- **Path:** `shared_scheduler.py::validate_schedule`.
- **Fix:** the validator now takes `helper_states` and independently enforces, for every transfer
  whose source is HELPER, `start_s >= contact_start_s` and `end_s <= contact_end_s`; it reports
  `helper_contact_window` and `helper_transfers_checked` so a run can show the invariant was
  exercised. A late return raises `V2ScheduleError`.
- **Regression:** `test_v2_helper_return_queue.py::ValidatorContactInvariantTests` (2 tests).
- **Still NOT_VERIFIED in F08:** abandoned CPU/radio reservations, wasted-vs-restart billing on
  fan-out, direct/staged route variants, and the effective-producer location for successors.

### F30 — completion counters depend on validation scheduling — **ALREADY_FIXED_WITH_PROOF**
- `stack.py::_iteration_completed` (the loop's last statement) sets `completed_iterations`,
  `next_iteration`; `_audit` no longer counts completion on the validation phase.
- `--iterations` is the TOTAL target; resume restores `next_iteration`.

### F31 — success JSON committed before final checks — **ALREADY_FIXED_WITH_PROOF**
- `run_tf.py` runs `_assert_real_run` before `REAL TF RUN OK`, returns 2 on failure / 3 on
  incomplete; the `iterations_executed` KeyError is gone.

---

## Not yet reproduced / audited this session

- **F03 (support/query semantics), F04 (graph-specific MC order), F05 (hidden MC in references),**
  **F08 (fallback calendars), F14 (dual ascent sign), F15 (deadline/energy authority),**
  **F16 (parameter routing), F17–F24 (TF gradient/CRN/copying/schema/advantage),**
  **F25–F29 (evaluator/selection/oracle), F32–F34 (runner/telemetry/continuation/device).**
  These are **NOT_VERIFIED** against this HEAD in this session. Several were partially addressed
  in earlier sessions (see `V2_ACCEPTANCE_MATRIX.md`, `V2_ROOT_CAUSE_ANALYSIS.md`); they are not
  re-asserted here.

## 2. Stage status and verdict

### Stages completed this session
- **Stage A (implementation repairs): ALL of the previously open Stage-A items are now repaired
  with executed regressions:** F01, F02, F07, **F06, F08 (validator), F12, F13, F21**, plus F09
  validated by measurement and C6 anchor versioning. Pre-fix failure counts:
  ```
  F01/F02/F07 : 4 of 8 new tests fail pre-fix   -> 8 passed
  F06/F13     : 3 of 12 new tests fail pre-fix  -> 12 passed
  F12         : 1 of 6 new tests fails pre-fix  -> 6 passed
  F21         : 7 of 8 new tests fail pre-fix   -> 8 passed (TF container)
  ```
- **Stage B (NOT_VERIFIED findings):** only the findings reproduced in section 1 were executed.
  F03–F05, F14–F29 and F32–F34 remain **NOT_VERIFIED at this revision**.
- **Stage C (short gate + fresh diagnostic):** in progress — see section 3.

### Stage C — executed (real TF1/contrib, GPU)

Two gates were run with the proven compatible image. Both used `--device gpu`, seed 0,
`background=2`, `link-regime moderate`, `background-policy all_mec`, `--r-select 2 --s-select 2`.

**Gate 1 — `--iterations 3`, budget fraction 0.5. VERDICT: FAILED (correctly).**
```
V2RunError: after 3 completed iterations no episode was ever penalised:
             a constraint that never binds is not evidence that the loop works
```
This is the run-level guard refusing to certify a run whose constraint never bound, and it is the
right outcome: at `fraction=0.5` the policy's foreground energy stayed under `0.5 * E_mec`, so no
constraint ever bound and there was no evidence the constrained loop works. Gate 1 also exposed
two live defects, both now fixed:
- **F32** — `WARNING v2 progress telemetry failed: progress="NameError: name 'json' is not defined"`
  (`v2_progress.json` missing, `v2_phase_timings.jsonl` 0 bytes) → `json` import restored.
- **F31** — `run.json` said `ok: true, error: null` while the guard had raised → verdict now final
  before the artifact is written.

**Gate 2 — `--iterations 3`, budget fraction 0.05 (binding). VERDICT: PASSED.**
```
REAL TF RUN OK -> /out/run.json
ok=True  error=None  incomplete=None
completed=3  validations=1  outer_updates_completed=3  dual_updates_completed=3
lambda_broadcast_targets=11
penalties = 400/600 episodes with a nonzero penalty, penalty_sum=15.7566, steps_with_penalty=2
lambdas   = {total_energy: 0.019716, ue_energy: 0.107848}
mean_violation = 0.9252   mean_system_joules = 101.256 J
```
The constraint BINDS, both duals move above zero, one validation ran, and the machine-readable
verdict matches the exit code. Progress artifacts now exist:
`v2_progress.json` (1229 B, per-iteration counters, lambdas, broadcast targets, violation, energy)
and `v2_phase_timings.jsonl` (1326 B, one record per iteration).

**Measured per-iteration cost on the repaired code** (from `v2_phase_timings.jsonl`,
iteration 2, the worst of the three):
```
iteration_total_s       23.358
  process_samples_ppo   10.882
  process_samples_eval  10.784
  inner_ppo              1.604
  outer_update           0.0780
  constraints            0.0094
  sample_tasks           0.0003
```
=> **~23.4 s per outer iteration**. Pre-F01 code measured ~26 s/iteration over the 500-run, so the
repaired world-build path is not materially slower. Estimate for 500 iterations: **~11 700 s
(3 h 15 m)** plus validation overhead. This is a measured estimate on a repaired 3-iteration run,
not a projection from the defective run.

### Stage C — fresh-process resume: **NOT equivalent (F33 REPRODUCED LIVE)**
Control: uninterrupted `--iterations 3`. Resume: `--iterations 2` (clean exit), then a NEW process
with `--iterations 3 --resume-from /out/ckpt/v2_run.ckpt`.

What DOES restore correctly:
```
resumed from /out/ckpt/v2_run.ckpt at next_iteration=2 target=3
last_completed_iteration 2 == 2      next_iteration 3 == 3
outer_updates_completed  3 == 3      dual_updates_completed 3 == 3
lambda_broadcast_targets 11 == 11
```
What does NOT — comparing each run's iteration-2 snapshot:

| field | control itr 2 | resume itr 2 | match |
|---|---|---|---|
| `mean_system_joules` | 101.256 | 66.961 | **NO (−34 %)** |
| `mean_requester_joules` | 0.3588 | 0.2672 | **NO** |
| `mean_violation` | 0.9252 | 0.9390 | **NO** |
| `lambdas.total_energy` | 0.019716 | 0.017826 | **NO** |
| `lambdas.ue_energy` | 0.107848 | 0.110330 | **NO** |
| episodes with a penalty (process) | 400/600 | 0/200 | **NO** |

**Verdict: the checkpoint, counters, dual state and selection state restore, but the resumed
iteration is a DIFFERENT iteration from the control's.** The fresh process does not reproduce the
sampling/RNG/world sequence, so a resumed run diverges from an uninterrupted one in the quantity
that matters (the training trajectory), even though the bookkeeping looks identical. This is
exactly the F33 defect ("model-weight load is not reproducible training resume"), now demonstrated
end-to-end rather than argued. It also honestly self-reports: the resumed run's `run.json` carries
`ok: false` with the guard's reason.

Two sub-findings from the same run:
1. **The "constraint bound" guard is process-local.** A resumed process that runs one iteration can
   fail the guard even though the run as a whole bound the constraint (the control applied 400
   penalties; the resumed process saw 0). The guard must be evaluated over the RUN, not the
   process, once resume is trustworthy.
2. **`mean_violation = 0.939` coexists with `episodes_with_nonzero_penalty = 0`** in the same
   snapshot, and `mean_violation = 0.925` coexists with 400/600 penalised episodes in the control.
   A positive signed violation with a positive lambda and zero applied penalties is
   self-contradictory and needs its own investigation; it is reported here as an open inconsistency
   rather than explained away.

### F33 — ROOT CAUSE of the resume divergence, localised

The decisive evidence is the WORLD IDENTITY, not the counters:

```
control  last_rollout.world_id = ep11_slot19   makespan 0.512884   system_joules 196.451
resume   last_rollout.world_id = ep3_slot19    makespan 0.595720   system_joules 200.273
```

Both runs are at iteration 2 with identical configuration and `next_iteration=3`, but they are
building DIFFERENT worlds: the resumed process is at episode 3 where the control is at episode 11.
The world identity in the V2 env is derived from the episode counter, the reset counter and the
world key, and those also feed the MC realization, the background workloads and the helper seeds.

`_resume_bundle` (run_tf.py:238-263) persists: schema, iteration counters, seed, background DAGs,
link regime, background policy, mec workers, R/S, obs version, protocol id, dataset/scheduler
fingerprints, the constraint manager state, `sampler_counters`, and `best_val_objective`.

It does **NOT** persist: the environment's episode counter, the reset counter, `_world_key`, the
realization epoch, and any NumPy/TF RNG state.

That is precisely why the bookkeeping restores (counters, duals, broadcast targets, selection
state) while the training trajectory does not (34 % difference in `mean_system_joules`, different
lambdas, no penalty applied in the resumed iteration). It also explains the earlier `--iterations`
accounting work being necessary but not sufficient: the loop resumes at the right iteration number
while replaying a different world sequence.

**Fix direction (not applied):** extend the bundle with an environment-identity section
(`episodes`, `reset_count`, `_world_key`, realization epoch) plus explicitly deterministic,
identity-derived streams for the link/helper/background/MC draws, and re-run this same
control-vs-resume comparison as its acceptance test. A raw TF/Numpy RNG state dump is acceptable
only if the streams are actually stateful; identity-derived streams are preferable because they
survive a process boundary. Record the measured CPU/GPU tolerance and the determinism limits.

**Related design wart found in the same run:** the "a constraint must bind" guard is PROCESS-local
(`penalty.episodes` counts only this process), so a legitimately resumed short run cannot satisfy
it even when the run as a whole bound the constraint. Once resume is trustworthy the guard must be
evaluated over the RUN.

### F33 — continuation repair (this session, 2026-10-08)

Tested HEAD: `4771b1e` (full SHA recorded in the manifest). Three distinct defects were found by
running the control-vs-resume comparison; none by reading code.

**(a) The world sequence was not restored — this was the actual divergence.**
The bundle carried the iteration ledger, duals, sampler counters and fingerprints, but not the
environment identity; the symptom was `world_id` control `ep11_slot19` vs resume `ep3_slot19`.
The decisive detail is that the counters that name the world live on the **sampler's deep-copied
clones** (`MetaIterativeEnvExecutor.__init__`, `vectorized_env_executor.py:24`), built once at stack
construction and therefore BEFORE any restore, so restoring only `trainer.env` could never work.
The bundle is now `v2_resume_state_v2` with an `environment` section carrying the primary and
**per-clone** `episodes`/`reset_count`, the world key, the realization epoch, base seed and regime,
restored positionally and exactly; a clone-count mismatch is refused. Result: **world identities
now match exactly**.

**(b) The first post-resume iteration collected with a stale zero dual.**
The multipliers were broadcast at the END of each iteration, which arms only the next one. Correct
for a fresh run (λ starts at 0) but wrong after a resume: the manager held the restored positive
dual while the clones still carried their deep-copied value, so the resumed iteration applied **no**
penalty although violation and λ were both positive. A pre-collection broadcast now runs at the top
of every iteration; it is idempotent.

Measured, control vs the resumed third iteration (CPU, seed 0, fraction 0.05, `--r-select 2
--s-select 2`):

| field | control itr 2 | resume itr 2 (before) | resume itr 2 (after) |
|---|---|---|---|
| `world_id` | `ep11_slot19` | `ep3_slot19` | **`ep11_slot19`** |
| penalised episodes (delta) | 200 | **0** | **200** |
| `mean_system_joules` | 99.6693 | 101.256 | 100.3308 (+0.66 %) |
| `mean_violation` | 0.9184 | 0.9252 | 0.9366 (+2.0 %) |
| `lambdas.ue_energy` | 0.107262 | 0.110330 | 0.107331 (+0.06 %) |
| `lambdas.total_energy` | 0.018751 | 0.017826 | 0.019648 (+4.8 %) |

The control's own per-iteration records are **bit-identical** before and after the broadcast change
(70.1617 / 67.5686 / 99.6693 J), so fresh-run and V1 semantics are untouched.

**(c) The residual is NOT bit-exact, and the cause is NOT optimizer slots.**
`tf.train.list_variables` on the generation reports **1042 variables, 682 of them Adam-like**
(`beta1_power`, `beta2_power`, `inner_adam_task_*` slots), so slot/power coverage is ruled out.
The policy's sampler is `tf.random.stateless_multinomial(logits, 1, seed=self._seed_pair)`
(`meta_seq2seq_policy.py:211`), i.e. a pure function of logits and the seed pair, so the TF global
RNG is not consumed either. Remaining unexplained: 0.06-4.8 % in the third iteration. Candidate
sources still to test: (i) process-to-process float determinism of the TF ops, (ii) whether any
observation channel depends on process-local state, (iii) the F20 seed-pair construction (the same
`_seed_pair` is reused at every token). **This is reported as an open item, not as equivalence.**

**Acceptance coverage actually executed:** uninterrupted 3-iteration control and a separate
2-iteration process resumed in a fresh process to total target 3, on CPU with `--intra-op 1
--inter-op 1 --device cpu`, comparing the corresponding third-iteration records. Cumulative
counters were compared as counters and the post-resume **increments** were compared separately via
`v2_iterations.jsonl`; the control's 600 episodes were NOT treated as the resume's 200.
**NOT yet executed:** a split that crosses a scheduled validation boundary (validation is frozen at
interval 50, so a 3-iteration gate validates only at iteration 0), and a GPU-tolerance comparison.

### F33 — first-divergence trace (this session, tested HEAD `9b1ceb9`)

**Evidence standard first.** Two uninterrupted controls on the fixed CPU configuration
(`--device cpu --intra-op 1 --inter-op 1`, seed 0, fraction 0.05) are **BIT-IDENTICAL**: every
per-iteration record equal, identical lambdas to 15 significant digits, identical penalty sums.
The control-vs-resume difference is therefore a **real state defect, not floating-point noise**,
and no tolerance was enlarged to absorb it.

**B1 (start of the third iteration), control vs resumed process:**

| | before this session | after |
|---|---|---|
| differing leaves | 152 | **8** |
| training-relevant differences | unknown | **0** |

- **All 1043 SHARED TF variables are identical.** The 62 control-only variables are ALL
  `ppo_update_validation_policy_deterministic/*` — the validation sub-graph, built lazily on the
  control's iteration-0 validation and never built by a process that resumes at iteration 2.
  This **refutes the optimizer-state hypothesis rigorously**, by value comparison, not by counting
  checkpoint keys.
- The 8 residual differences are `python.random` (not consumed anywhere in the training path —
  the only stdlib uses are `random.seed` calls and private `random.Random(...)` instances),
  process-local counters (`auto_*`, `current_iteration`) and one bookkeeping flag, all of which are
  different in a fresh process **by construction**.

**Three genuine restore gaps found by the trace and closed:**
1. `sampler_counters` were WRITTEN into the bundle but **never applied** — a resumed process
   reported zero support/query calls and zero unique graphs/rollout seeds for work already done.
2. the stdlib `random` stream was not carried at all, although the stack seeds it
   (`meta_trainer.py`: `random.seed(SEED)`).
3. each sampler clone's `task_id`/`graph_indices` (its current task assignment) was not restored.

**Localisation result.** B1 is *training-equivalent*, so the remaining iteration-2 difference
originates **downstream of the restored boundary**:

| itr 2 | control | resume |
|---|---|---|
| `mean_system_joules` | 99.6693 | 100.3308 |
| `mean_violation` | 0.9184 | 0.9366 |
| `lambdas.total_energy` | 0.018751 | 0.019648 |
| penalised episodes (delta) | 200 | 200 |

The penalty axis now MATCHES. The residual 0.7 % energy / 4.8 % lambda_total difference is
**localised to B2+ (task selection, worlds, or inference)** and is the next trace target. It is
reported as an open item with the search space halved by evidence; it is NOT attributed to floating
point.

**Tooling added:** `spec/automotive_training/v2/state_trace.py` with a `--state-trace` CLI flag.
`digest` is sha256 over a canonical byte form, so equal digests mean equal values. The variable
fetch is ONE batched `sess.run` (a per-variable fetch cost ~1000 round trips per boundary and made
a traced run take 13 minutes without finishing one iteration).

### F14/F33 — the production penalty gate was wrong and is corrected

The gate required `episodes_with_nonzero_penalty >= 1` for any run of >= 2 iterations. That is not a
property of a valid constraint: `penalty = sum_i lambda_i * max(0, g_i)`, so with λ = 0 the hinge
penalty is exactly zero however large the violation, and a feasible production batch may
legitimately pass with zero penalties. The requirement is **removed**; the guard still requires a
live dual (`updates >= 1`, signed rows reached the trainer, broadcast targets >= 1). The penalty
PATH is now pinned by controlled fixtures whose λ can be set explicitly:
`test_v2_constraint_penalty_path.py` (14 tests) covers λ=0 with a violating row → exactly 0;
λ>0 × violation; feasible → 0; mixed batch → only violating rows; multi-channel independence;
terminal charge applied once and contributing exactly `-penalty` to the discounted return; inactive
costs; rejected negative/unknown/wrong-length λ.

The `mean_violation` vs `nonzero_penalty` "contradiction" recorded earlier is now **explained, not
dismissed**: it is the documented hinge-primal / signed-dual combination. A batch can contain
positive per-episode violations while its MEAN signed cost is <= 0, in which case λ stays at 0 and
no penalty is charged. The production gate below is exactly that case.

### Stage C — final production gate at the APPROVED fraction 0.5 (this session)

```
command: python -m spec.automotive_training.v2.run_tf --device gpu --gpu-fraction 0.03
         --intra-op 8 --inter-op 2 --iterations 3 --seed 0 --background 2
         --link-regime moderate --background-policy all_mec
         --energy-budget-fraction 0.5 --ue-budget-fraction 1.0
         --r-select 2 --s-select 2 --ckpt-dir /out/ckpt --json /out/run.json
verdict: REAL TF RUN OK      ok=True   error=None   resume_mode=fresh
         session device=GPU gpu_fraction=0.03 intra_op=8 inter_op=2
         completed=3  started=3  next_iteration=3  outer_updates_completed=3
         validations_completed=1   checkpoint.status=completed_iterations=3  promotable=true
per-iteration (v2_iterations.jsonl):
   itr 0  dv=1  sysJ=70.3084  viol=0.0228  dPen=0  lam={total_energy 0.0, ue_energy 0.0}
   itr 1  dv=2  sysJ=65.0410  viol=0.0356  dPen=0  lam={total_energy 0.0, ue_energy 0.0}
   itr 2  dv=3  sysJ=100.5105 viol=0.0057  dPen=0  lam={total_energy 0.0, ue_energy 0.0}
artifacts: run.json, v2_progress.json, v2_phase_timings.jsonl, v2_iterations.jsonl
```

This is a **feasible production run under the approved 0.5 budget**: the mean signed cost stays at
or below budget, so λ remains 0 and no penalty is charged — exactly the legitimate zero-penalty
outcome the old guard rejected. The constrained machinery is nonetheless proven live (3 dual
updates, 600 signed rows, 11 broadcast targets) and its penalty path is proven by the fraction-0.05
binding fixture and the controlled tests above. **The fraction-0.05 numbers are NOT mixed into this
gate's results.**

**F34 evidence from this gate:** `--device gpu` was requested and the process reports
`session device=GPU`; the pre-allocated hard cap is `gpu_fraction=0.03` with `allow_growth=false`,
and vLLM was left untouched throughout. Device placement is asserted by the session banner, not by
kernel-level profiling, so this is a **recorded configuration**, not a measured kernel breakdown.

### Stage B — still NOT_VERIFIED at this revision
F03, F04, F05, F14, F15, F16, F17, F18, F19, F20, F22, F23, F24, F25, F26, F27, F28, F29, F34.
No claim is made about any of them. The earlier partial reports are not substitutes; several
findings were only partially addressed in prior sessions and are not re-asserted here.

### Verdicts

**IMPLEMENTATION_READY: NO.** Concrete blockers, in order:
1. **F33 is now REPRODUCED LIVE**: a fresh-process resume restores weights/counters/duals but does
   NOT reproduce the control's training trajectory (iteration-2 `mean_system_joules` differs by
   34 %, both duals differ, the penalty history differs). Until continuation is equivalent, a
   >1-session run cannot be certified.
2. **Stage B is incomplete**: F03–F05, F14–F29 (except the pieces listed) and F34 are NOT_VERIFIED
   at this revision. In particular the TF gradient/CRN/behaviour-likelihood contract (F18/F19/F20),
   schema/import order (F23), the paired evaluator on one immutable world (F25/F26) and
   energy-aware checkpoint selection (F28) have not been executed here.
3. **F11 physical-unit conflict is documented but not resolved.** The ledger charges the
   duration form while tracking the work form, and timing rates come from per-graph resource
   frequencies while energy prices use frozen tier frequencies (`E_dur/E_work =
   R_physical/R_scheduled`). Absolute joule values are not yet dimensionally certified.
4. **F08 is only half done**: the validator half is fixed and tested; abandoned CPU/radio
   reservations, fan-out billing, direct/staged route variants and effective-producer locations
   are not verified.
5. **The "positive violation with zero applied penalties" inconsistency** found in the resume gate
   is unexplained.
6. Every historical artifact (the 500-run, its checkpoints, and all performance numbers derived
   from them) was produced under F01+F02 and is retained only as a defective-environment record.

**LEARNING_COMPETITIVE: NOT_YET_MEASURED.** No performance claim about the repaired code is
supported. The historical numbers (k=0 0.0806 s, the 2.69x all-MEC latency gap, the 2.35x energy
gap, the k-sweep saturation at ~6 %) were all measured on the defective environment or with the
pre-fix reward/evaluator, and are explicitly NOT results of the repaired system. A repaired-code
learning diagnostic (Stage C item 4) has not been run; the three-iteration gates above establish
that the loop WORKS, not that it LEARNS WELL.

### What is actually established
- The constrained chain runs ≥3 real outer iterations on the repaired code, with a binding
  constraint, moving duals, one validation, per-iteration progress, periodic checkpoints, and a
  machine-readable verdict consistent with the exit code (Gate 2).
- Seven implementation defects are repaired with regressions that fail on the pre-fix code:
  F01, F02, F06, F07, F08 (validator), F12, F13, F21, plus F31 and F32 found live by the gate.
- The configured energy budget is NOT infeasible: a feasible witness exists on train data
  (all-UE 0.192 J and other candidates vs B = 10.03 J), and excluding all-MEC is a legitimate
  trade-off (C1).
- F01/F02 were LIVE at the historical 500-run source `49e5cb8`, so that run's evidence is void.

### Eligible long-run configuration (not launched)
```
source   : a9e8f10 (after F06/F13 decisions and a passed Stage C)
image    : margo-phase4-tf115-nv2212:latest @ sha256:82bc781bc288...
command  : python -m spec.automotive_training.v2.run_tf --device gpu --gpu-fraction 0.03
           --intra-op 8 --inter-op 2 --iterations <TOTAL> --seed 0 --background 2
           --link-regime moderate --background-policy all_mec
           --energy-budget-fraction 0.5 --ue-budget-fraction 1.0 --r-select 2 --s-select 2
           --ckpt-dir /out/ckpt --json /out/run.json
```
Measured runtime on the repaired code: **≈23.4 s per outer iteration** (Gate 2, iteration 2:
23.358 s total, of which `process_samples_ppo` 10.88 s and `process_samples_eval` 10.78 s).
Estimate for the total target: **500 iterations ≈ 11 700 s ≈ 3 h 15 m** of GPU time, plus
validation. This is measured on the repaired revision, not projected from the defective run.
It does NOT include the time to re-certify Stage B or the F33 continuation repair.

### Remaining modelling assumptions / unmodelled components
`system_joules = UE CPU + helper CPU + MEC CPU + UE/helper/MEC TX`. Not modelled (from
`run.json.unmodeled_components`): receiver/static radio power (`include_rx_energy=False`), CPU
idle/leakage between tasks, MEC/RSU static power (`server_static_power_w` null), backhaul/core
transport, battery dynamics/charging/reserve, warm-standby reservation energy. Absolute joule values
are a lower bound and must not be presented as measured hardware consumption.
