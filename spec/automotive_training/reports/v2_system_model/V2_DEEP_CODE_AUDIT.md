# V2_DEEP_CODE_AUDIT — deep audit, repair, and evidence-based readiness

**Audited revision.** The independently readable GitHub SHA `4a4e40f160e153b0dbf7baf134254ab12b77cacd`
(phase5-realistic-system-v2) is **NOT present** in the local mirror (`git cat-file` fails) nor on
the server git repo (`/root/v2-verify` HEAD = `0f9cb096b76a84d3c7b260d539b4e2ad15dceac6`). The
actual executed/trained code is the local mirror at:

- base: `0f9cb096b76a84d3c7b260d539b4e2ad15dceac6` ("identity-keyed environmental realizations…")
- this session's repairs: `48a4367` → `e07ba6a` (the 500-run used `49e5cb8`)

Every finding below is therefore marked against **HEAD `e07ba6a`** (this audit's tested SHA).
Findings that were only reproduced at the audited remote SHA and could not be re-verified here
are marked NOT_VERIFIED.

**Status vocabulary:** REPRODUCED / ALREADY_FIXED_WITH_PROOF / CONTRACT_CONFLICT / HYPOTHESIS /
NOT_VERIFIED.

**Regression baseline:** `1393 passed, 19 skipped` before this audit; `1401 passed, 19 skipped`
after (8 new tests, all fail on the pre-fix code).

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
- **Proof of fix:** `step` evaluates the constraint on the foreground-only ledger
  (`dag_filter=[world.foreground_id]`); the total-energy anchor is `refs.E_mec`. Comment corrected
  this session to state the measurement is foreground-only (it had said "whole-world").
- **Residual:** the reference worlds are still built background-free by design (`pure_location_worlds`),
  which is now consistent with the foreground-only measurement. Attribution axes remain: requester /
  mobile / system boundary × foreground / background / world ownership — documented in
  `V2_ENERGY_SCOPE_FEASIBILITY.md`.

### F10 — default budget analytically infeasible even without background — **REPRODUCED (config)**
- **Path:** `constraints.py` C_TOTAL_ENERGY anchor + `--energy-budget-fraction 0.5`.
- **Proof:** under the fixed scope, `all-MEC` foreground energy **equals** the anchor `E_mec` by
  construction, so budget `0.5 × E_mec` gives `g = (1.0 − 0.5)/1.0 = +0.5` — **all-MEC is
  infeasible by construction** and it is the near-optimal-latency plan (only 5 % behind the panel
  oracle). A fraction ≥ 1.0 is required for all-MEC feasibility.
- **Status:** this is a **configuration/calibration** defect, not a code bug; the fraction was
  disclosed as "an engineering setting, not calibrated". The feasibility study
  (`V2_ENERGY_SCOPE_FEASIBILITY.md`) states the requested 0.5 spec is infeasible for the
  near-optimal plan and a calibrated fraction ≥ 1.0 (or a different constraint scope) is required.

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

## Verdict

- **IMPLEMENTATION_READY: NO.** F01/F02 invalidate the 500-run's evidence (repaired, but the run
  itself used the defective graph/realization identity). F06/F07 are transfer-semantics defects;
  F07 is now fixed, F06 is not. F13 is a contract conflict. F10 is a feasibility defect.
- **LEARNING_COMPETITIVE: NO / NOT_YET_MEASURED.** Under the repaired environment the trained
  policy is dominated by all-MEC on both latency (2.69×) and energy (2.35×) at the pre-fix
  revision; but that measurement itself predates F01/F02, so it is **NOT_YET_MEASURED** on the
  repaired code.
