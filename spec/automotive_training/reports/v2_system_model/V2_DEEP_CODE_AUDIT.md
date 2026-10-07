# V2_DEEP_CODE_AUDIT — deep audit, repair, and evidence-based readiness

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

### F12 — average energy multiplies episode joules by 20 — **REPRODUCED (legacy label only)**
- **Path:** `v2/env.py::step` fills `energy_batch` with `np.full(len(rewards), episode_energy)`;
  `meta_trainer.py:581` sums that vector.
- **Proof:** slot with episode `system_joules=0.118` → `sum(energy_batch)=2.368 = 20×0.118`.
- **Status:** the scoped telemetry columns (`energy/system_joules` etc.) are CORRECT; only the
  legacy "Average energy" scalar is 20× the episode value. It is already labelled
  `legacy_average_energy_kvs` (MOBILE). Fixing the shared `meta_trainer.py` sum would touch the
  frozen V1 path; deferred with this label. Do NOT report "Average energy" as episode joules.

### F13 — dense shaping and gamma mismatch the objective — **CONTRACT_CONFLICT (reproduced)**
- **Path:** `v2/env.py::_telescoping` emits `r_t = J_{t-1} − J_t` (γ=1); PPO applies
  `discount=0.99` (`stack.py:173`).
- **Proof:** `[5,3]`→7.97 vs `[1,7]`→7.93 (both undiscounted = 8) — ranking depends on the
  intermediate prefix, contradicting `spec/objective_contract.py:22-28` which declares
  `r_t = J_{t-1} − γ·J_t` (telescopes to `J_0 − γ^N·J_N`).
- **Status:** CONTRACT_CONFLICT — **NOT fixed** (would change the reward and require re-evaluation;
  out of scope without explicit approval). The discount-consistent form already exists in the
  offloading env (`offloading_env.py:247`, `reward.py:135`). The automotive v2 stack never wires
  `shaping_discount`.

### F06 — service integration skips rate-change boundaries — **REPRODUCED (not yet fixed)**
- **Path:** `shared_scheduler.py::_service_detail` (step advances full `dt`, samples rate only at
  step start); `link_model.py::_idx` (`floor(t/dt)`).
- **Proof:** analytic `0.115 s` vs scheduler `0.160 s` (exact match to the finding).
- **Status:** REPRODUCED. Fix (integrate to the next trace boundary / outage / contact transition)
  is understood but not applied in this session; it changes transfer timing and must be followed
  by re-evaluation.

### F21 — NumPy categorical log_likelihood broken — **REPRODUCED (dead/latent)**
- **Path:** `policies/distributions/categorical_pd.py::log_likelihood` (builtin `sum` wrong axis;
  `log(one_hot*p)` includes `log(0)`).
- **Proof:** zero logits, actions [0,1] → `[-inf,-inf]` (expected `[-log3,-log3]`).
- **Status:** REPRODUCED but **zero call sites** on the training path (TF `log_likelihood_sym` is
  correct and is what PPO uses). Did NOT cause TF training. Low priority; fix or delete the dead API.

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
- **Stage A (implementation repairs):** F01 (graph identity), F02 (MC independence), F07 (helper
  return queue) **fixed with regressions**; F09 validated by measurement; C6 anchor naming
  versioned. **F06, F12, F13, F21 remain open** (see below).
- **Stage B (NOT_VERIFIED findings):** only the findings listed in section 1 were executed. The
  rest (F03–F05, F08, F14–F29, F32–F34) are **NOT_VERIFIED at `a9e8f10`**.
- **Stage C (short gate + fresh diagnostic):** **NOT RUN.** No real-TF 3-iteration gate, no
  fresh-process resume comparison, no repaired-code training diagnostic was executed in this
  session.

### Why F06/F12/F13/F21 are left open (explicit, not "out of scope")
- **F06** (rate-boundary integration) changes transfer timing → requires re-evaluation of every
  timing/energy number. Reproduced with the exact claimed numbers; fix direction known.
- **F12** (legacy ×20 energy) lives in the shared `meta_trainer.py` sum that V1 also uses; the
  scoped telemetry columns are correct, so the fix must be a V2-specific adapter, not a shared edit.
- **F13** (shaping/γ) is a genuine contract conflict; correcting it changes the objective and
  therefore all learning results, which requires the authorized re-evaluation campaign.
- **F21** is dead code (zero call sites); TF `log_likelihood_sym` is correct and is what PPO uses.

### Verdicts
**IMPLEMENTATION_READY: NO.** Concrete blockers:
1. Stage C not executed: no real-TF ≥3-iteration gate on the repaired revision, no fresh-process
   resume equivalence, no repaired-code learning diagnostic.
2. F06 open (transfer timing can be wrong across rate/outage boundaries).
3. F13 open (the reward shaping does not implement the declared discount-compatible contract).
4. F03–F05, F08, F14–F29, F32–F34 unverified at this revision.
5. The historical 500-run and all its checkpoints were produced under F01+F02; they are retained
   as defective-environment artifacts and cannot certify the repaired system.

**LEARNING_COMPETITIVE: NOT_YET_MEASURED.** Every performance number in the repository
(k=0 0.0806 s, the 2.69× all-MEC gap, the −66 % validation latency, the k-sweep saturation) was
measured either on the defective environment or with the pre-fix evaluator. No performance claim
about the repaired code is currently supported.

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
Measured runtime on the repaired code (per-phase instrumentation): the pre-F01 code completed
500 iterations in 13 055 s (≈26 s/iteration) **with the YAML memoisation**. On the repaired code the
world-build path now resolves a different graph per meta-task, so the per-iteration cost must be
re-measured; **no estimate is offered** because none has been taken on `a9e8f10`.

### Remaining modelling assumptions / unmodelled components
`system_joules = UE CPU + helper CPU + MEC CPU + UE/helper/MEC TX`. Not modelled (from
`run.json.unmodeled_components`): receiver/static radio power (`include_rx_energy=False`), CPU
idle/leakage between tasks, MEC/RSU static power (`server_static_power_w` null), backhaul/core
transport, battery dynamics/charging/reserve, warm-standby reservation energy. Absolute joule values
are a lower bound and must not be presented as measured hardware consumption.
