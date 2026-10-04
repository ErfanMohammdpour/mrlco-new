# V2_ACCEPTANCE_MATRIX — MARGO v2 repair round

**Branch** `phase5-realistic-system-v2`. **Base HEAD at the start of this round** `0f9cb096`
("identity-keyed environmental realizations + verified checkpoint/resume equivalence").
Everything below is stated against that base plus this round's commits; a claim is only
`PASS` when the command in the "command" column was actually executed and its raw output is
quoted in "raw evidence". `NOT RUN` is never reported as `PASS`.

Verification environments (all real, none simulated):

| id | environment | purpose |
|---|---|---|
| `ENV-CPU-LOCAL` | macOS, CPython 3.14, pytest 9.1.1 | non-TensorFlow suites (pure Python scheduler/energy/constraint/v2 modules) |
| `ENV-CPU-IMG` | `margo-phase4-tf115-gpu:latest` = `sha256:af7550b0d2cc…`, Python 3.7.5, TF 1.15.5 | container-only Python 3.7 checks |
| `ENV-GPU-IMG` | `margo-phase4-tf115-nv2212:latest` = `sha256:82bc781bc288…`, Python 3.8.10, TF 1.15.5, CUDA 11.8, `tf.contrib` present | real TensorFlow execution on the RTX 4090 |
| `ENV-GPU-HOST` | `kish-ai` (aizarepor), 32 cores, 1× RTX 4090 24564 MiB, 21508 MiB held by a co-resident `VLLM::EngineCore` | GPU runs |

`python3 -c "import tensorflow.contrib"` fails locally, so every TF-dependent criterion is
verified only in `ENV-GPU-IMG`.

---

## A. The two contradictions that were open at the start of this round

### A1. The CPU gate2 report said `ok=True`, `started=2`, `completed=1`, `error=None` — with a traceback

| field | value | how it was established |
|---|---|---|
| crash site | `/work/spec/automotive_training/v2/run_tf.py:365` in `main` | full traceback in `gate2_20261004T103331/logs/stdout.log:284` |
| exception | `KeyError: 'iterations_executed'` | same |
| when | **after** `trainer.train()` returned and **after** the checkpoint was written | `run.json` exists and reports a real `v2_run.ckpt`; the traceback is in the summary `print`, not in training |

**Verdict: the run did not fail. The REPORT writer failed.** `_collect_evidence` never
produced a key called `iterations_executed` (it produces `iterations_started`,
`iterations_completed`, `last_completed_iteration`), so the terminal summary raised `KeyError`
after training had completed both outer iterations and saved a checkpoint.

The three reported fields are explained exactly:

* `error=None` — the `failed = es: exc` capture only wraps `trainer.train()`. The `KeyError`
  happened later, outside that `try`, so `evidence["error"]` was never set. The traceback went
  to stderr and the process died; it was never recorded in the JSON.
* `ok=True` — `report["ok"] = (failed is None)`, i.e. it asserted only that `trainer.train()`
  did not raise. It did not assert that the target was reached, that the report was written,
  or that the assertions in `_assert_real_run` had run. **The `ok` flag was written before the
  assertions, and the process died before reaching them.**
* `completed=1` — a real counter defect, see A2.

**Repair.** `iterations_executed` now exists as an explicit alias, the terminal summary reads
only keys `_collect_evidence` is contracted to produce, and `_assert_real_run` runs before the
success exit. Regression coverage: `ReportContractTests` in
`env/mec_offloaing_envs/scheduler/tests/test_v2_iteration_accounting.py`
(`test_summary_printer_has_no_missing_keys`, `test_assert_real_run_rejects_counter_disagreement`).

**Raw evidence (gate2, preserved before any repair):**

```
gate2_20261004T103331/run.json:
  evidence.iterations_requested = 2
  evidence.iterations_started   = 2
  evidence.iterations_completed = 1
  evidence.error                = None
  report.ok                     = True
logs/stdout.log:284  Traceback (most recent call last):
logs/stdout.log:293  KeyError: 'iterations_executed'
```

### A2. The completion counter under-reported (and was not the only instance)

`stack.AutomotiveV2Trainer._audit` set `completed_iterations = itr + 1` **when the `validation`
audit fired**. Two independent defects followed:

1. **Counting a phase ENTRY as completion.** Validation is not the end of the iteration; a crash
   after it would still have counted the iteration as complete.
2. **Iterations without a scheduled validation were never counted.** With
   `validation_interval = 50`, only `itr % 50 == 0` runs validation, so iterations 1–49 were
   invisible in the counter.

**Both were reproduced in the field, not merely in a unit test.** The independent GPU run
`gpu3_20261004T185340` (2-iteration target) reported:

```
iterations_started   = 2
iterations_completed = 1        <-- a real 2nd iteration was started and never counted
last_completed_iteration = 0
report.ok = True                <-- with a KeyError traceback in the same process
checkpoint.status = 'completed_iterations=1'
progress = {exists: False, path: /out/v2_progress.json}
```

So the same contradiction as gate2 is present in the GPU run: `ok=True` alongside a traceback,
`started=2` against `completed=1`, and a checkpoint promoted as `completed_iterations=1`.

**Repairs.**

* Completion is recorded by `Trainer._iteration_completed(itr)`, a new **no-op hook** called as
  the loop's LAST statement (`meta_trainer.py`), and overridden in `AutomotiveV2Trainer` to set
  `last_completed_iteration`, `completed_iterations` and `next_iteration`. Reaching any phase
  marks nothing.
* Separate persistent counters: `started_iterations`, `completed_iterations`,
  `last_completed_iteration`, `next_iteration`, `outer_updates_completed`,
  `dual_updates_completed`, `validations_completed`. The outer-update and dual counters are
  incremented only at their own phase.
* The in-process progress ledger is written atomically at every completed iteration, and
  `--iterations` is now the **TOTAL target**: `_restore_resume_state` sets `start_itr =
  next_iteration`, so a resumed run finishes at exactly the target instead of re-running
  iterations 0..N.

**Regression coverage — 15 tests, and the same suite fails on the old code:**

```
ENV-CPU-LOCAL  python3 -m pytest env/mec_offloaing_envs/scheduler/tests/test_v2_iteration_accounting.py -q
  before (git stash of this round's changes):  7 failed, 8 passed
  after:                                      15 passed
```

The decisive case is `test_two_complete_iterations_report_two_completed`, which drives the real
audit phase names for iteration 0 (validated) and iteration 1 (not validated) and asserts
`completed_iterations == 2`.

---

## B. The energy constraint was measuring a different world than its reference

This is the largest scientific finding of the round. It subsumes the reported
"normalized violation around 139". Two independent audits and a direct calculation agree.

### B1. What was measured vs what the budget was anchored on

| quantity | axis | source |
|---|---|---|
| `metrics.total_energy_j` | **whole-world** SYSTEM scope: foreground requester + every declared background DAG of other owners | `schedule_energy(result, background_dag_ids=world.background_ids)` with **no** `dag_filter`, `spec/automotive_training/v2/env.py` |
| `refs.E_ue` (the anchor) | **foreground-only, background-free** SYSTEM scope | `pure_location_worlds` forces `ref_config = replace(base, background_dags=0)` (`world.py`) and schedules without a link process (`env.py`) |

The reference is built from background-free pure-location plans; the measurement included the
background. On the production configuration the background was ~99 % of the measurement and
plan-independent.

**Direct calculation from the preserved GPU evidence** (`gpu3_20261004T185340/run.json`),
solving `g = (M − 0.5·E_ue)/E_ue` for `E_ue`:

```
M   (system_joules)              = 38.32089625767425 J
g   (total_energy_signed)        = 136.60424529994333
implied E_ue (background-free)   = 0.279502 J
implied budget B = 0.5 * E_ue    = 0.139751 J
M / B                            = 274.2
requester_joules (foreground)    = 0.279454 J
mobile_joules                    = 1.096257 J
background_joules                = 35.808284 J
lambda_total_energy              = 13.696921144223943
penalty per episode = lambda * g = 1871.06
penalty_sum over 400 episodes    = 187396.93
```

So the "≈139" was **274 × the budget**, and the "938" penalty in the earlier CPU run is the
same artifact one iteration later: with `dual_lr = 0.05`, `λ₁ = 0.05 × 132.4 = 6.6…7.08`, and
`penalty = λ_prev × violation = 7.08 × 132.4 ≈ 938`. The second audit independently
reconstructed the same arithmetic from the published λ values.

### B2. Why this is a defect and not a hard budget

* **Scope mismatch**, per the project's own rule: a whole-world system measurement was compared
  against a foreground-only background-free reference.
* **Structurally infeasible**: the background energy was approximately plan-independent
  (35.81 J of the 38.32 J episode), so the minimum achievable violation exceeded +1.4 no matter
  what the policy did. λ therefore pins at `max_lambda` rather than regulating anything.
* **Two further, independently fatal anchor problems** were found in the pre-fix anchor
  `E_ue`: it is the global minimum of the system-boundary measurement (all-UE is the cheapest
  plan; measured over a 403-plan sweep, zero plans fall below it), so any fraction `< 1.0`
  anchored on it is unsatisfiable **even with `background_dags = 0`**; and the knob's name
  (`total_energy_frac_of_all_ue`) described neither the corrected measurement nor the anchor.

### B3. Repair

1. **Measurement attribution.** `V2AutomotiveEnv.step` now evaluates the constraints on the
   **foreground-only** ledger (`dag_filter=[world.foreground_id]`), so the measured value, the
   budget and the normalisation scale all describe the same DAG set and the same energy
   boundary. The whole-world ledger is still built and still feeds the world-scope telemetry
   fields, so nothing is hidden: `system_joules`, `background_joules`,
   `world_requester_joules` and `foreground_system_joules` remain separately reported.
2. **Coherent anchor.** `budgets_and_scales` anchors the total-energy budget and scale on
   `refs.E_mec`, the pure all-MEC plan's SYSTEM energy measured on the same world, instead of
   `refs.E_ue`. A signed cost of +1.0 now means "this episode spent twice the reference plan's
   whole system energy".
3. **The observation channel mirrors the canonical source.** `env._context_for_graph`'s
   `budget_ratio` reads the same `E_mec` anchor, so the policy observes the budget it is
   actually scored on.

**Declared change.** `--energy-budget-fraction 0.5` keeps its numeric value but changes
meaning: it is now 0.5 × (the pure all-MEC plan's system energy), not 0.5 × (the pure all-UE
plan's system energy). This is **disclosed**, not a silent loosening: the previous formulation
was unsatisfiable by construction, so no fraction of the old anchor could have produced a
meaningful constraint. The fraction remains an engineering setting; it is not claimed to be
scientifically calibrated.

**Regression coverage.**

```
ENV-CPU-LOCAL
  test_v2_constraints.py::TestTotalEnergyScopeConsistency   2 passed
  test_v2_constraints.py (whole file)                      12 passed
  full non-TF scheduler suite                    1389 passed, 0 failed, 19 skipped
```

`test_anchor_is_all_mec_system_energy_not_all_ue` asserts the scale equals `refs.E_mec`, the
budget equals `0.5 × refs.E_mec`, that the scale is **not** `refs.E_ue`, and that the signed
cost is O(1) rather than O(100).

Two existing tests encoded the old anchor (`.5 × E_ue` could not bind once `E_mec` is the
anchor, because the all-MEC plan is exactly `E_mec`). They were updated to a genuinely binding
fraction with the reason recorded in the test body; they were not weakened or deleted:

```
test_v2_constraints.py    TestRewardPenaltyIsReal, TestDualsMoveBothWays   -> 12 passed
test_v2_observation_v2.py TestNewChannels (budget 2.0 -> 0.5)              -> 14 passed
```

---

## C. Acceptance matrix

| # | criterion | code path | command / artifact | raw evidence | result | limitation |
|---|---|---|---|---|---|---|
| 1 | gate2 traceback resolved; `ok` cannot precede the assertions | `v2/run_tf.py::main`, `_assert_real_run` | traceback + report inspected; `test_v2_iteration_accounting.py::ReportContractTests` | `KeyError: 'iterations_executed'` reproduced at `run_tf.py:365`; `ok=True` written before assertions | **PASS** | the original gate2 process cannot be re-run (its 19-minute CPU cost is not budgeted); the defect is reproduced statically and by test |
| 2 | completion counters separate, persistent and order-correct | `v2/stack.py::_audit/_iteration_completed`, `meta_trainer.py` hook | `test_v2_iteration_accounting.py` (15 tests; 7 fail on the old code) | `4 passed` for the counter group; `15 passed` overall | **PASS** | — |
| 3 | `--iterations` is a TOTAL target; resume never repeats completed work | `v2/run_tf.py::_restore_resume_state` | `ResumeBundleTests` | `start_itr` restored to `next_iteration`; a checkpoint past the target is rejected | **PASS** | resume equivalence is asserted on the Python-side state, not yet by a TF fresh-process comparison (see #9) |
| 4 | failure produces a nonzero exit; partial checkpoints are labelled | `v2/run_tf.py::main` | `test_assert_real_run_*` | exit 2 on a trainer exception, exit 3 when the total target is not reached; `status='partial_failed'` / `partial_incomplete=n_of_N`; `promotable=False` | **PASS** | — |
| 5 | energy measurement scope == reference scope | `v2/env.py::step`, `scheduler/constraints.py::budgets_and_scales` | `TestTotalEnergyScopeConsistency` | scale `= E_mec`, budget `= 0.5·E_mec`, scale `≠ E_ue`, `|g| < 10` | **PASS** | the foreground/background split is by `dag_id`; there is no owner dimension in the ledger |
| 6 | the reported ≈139 violation is explained and no longer appears | calculation from preserved evidence | `gpu3`/`gate2` `run.json` | `g = 136.604`, `M/B = 274.2`, implied `E_ue = 0.2795 J`, `penalty_sum = 187396.93` | **PASS (explained)** | it is an attribution/scope artifact, not a 139-fold physical violation |
| 7 | penalty is applied once, on the terminal token, and the reward identity holds | `v2/constraints_v2.py::apply_penalty` | `test_the_penalty_is_charged_exactly_once_on_the_terminal_token` | non-terminal tokens bit-identical; terminal `== base − penalty` | **PASS** | — |
| 8 | duals use the SIGNED cost (can fall), update once per iteration, training-only | `scheduler/constraints.py::dual_step/observe` | `test_lambda_rises_above_budget_and_falls_below_it` | `λ` rises over budget and falls below it; validation envs are built without constraints | **PASS** | λ is logged AFTER the dual step, so a table pairing row N's λ with row N's penalty is misaligned by one iteration; the λ used by a rollout is the one broadcast at the end of iteration N−1 |
| 9 | λ reaches every rollout clone | `v2/stack.py::broadcast_constraint_lambdas` | independent probe + published `constraint/lambda_broadcast_targets` | `11` targets = 1 original env + 10 deepcopy clones (`meta_batch_size=10 × envs_per_task=1`); a clone receives the trainer's λ and its next rollout is penalised | **PASS** | no in-repo test drives the v2 executor broadcast end to end; this is a named gap |
| 10 | the guard rejects a run in which the Lagrangian loop never closed | `v2/run_tf.py::_assert_real_run` | new assertions | raises when `updates < 1`, `signed_rows_observed < 1`, `lambda_broadcast.targets < 1`, or ≥2 iterations with no penalty | **PASS** | — |
| 11 | non-TensorFlow suites have no unexplained failures | whole scheduler suite | `python3 -m pytest env/mec_offloaing_envs/scheduler/tests -q -p no:nameko --ignore=…_tf.py` | `1389 passed, 19 skipped` | **PASS** | all 19 skips are TensorFlow-gated and carry an explicit reason |
| 12 | frozen v1 untouched | `git diff 92212d1d -- scheduler/engine.py spec/automotive_training/automotive_env.py` | — | empty | **PASS** | `meta_trainer.py` gains one additive no-op hook; declared, v1-inert |
| 13 | real TF two-iteration run: support, inner PPO, query, outer update, validation, checkpoint | `v2/run_tf.py` in `ENV-GPU-IMG` | `gate_final_20261005T0015` | see the server verification report for the executed numbers | **see §D** | — |
| 14 | measured per-phase timings and an honest 500-iteration estimate | `v2/stack.py::_audit` phase clock + `v2_phase_timings.jsonl` | `v2_progress.json`, `v2_phase_timings.jsonl` | written per completed iteration | **see §D** | the estimate is from measured phases, not from a microbenchmark |
| 15 | fresh-process TF resume reproduces an uninterrupted run | `v2/run_tf.py --resume-from` | not yet executed end to end | — | **NOT RUN** | the Python-side bundle is implemented and tested; the TF fresh-process comparison remains outstanding |
| 16 | diagnostic/long run status recorded honestly | — | this document | — | **see §E** | — |

---

## D. GPU acceptance run

See `V2_SERVER_VERIFICATION.md` for the executed evidence of
`gate_final_20261005T0015` (job identity, image digest, source hashes, per-phase timings,
counters, λ progression, penalty arithmetic).

---

## E. Long-run status

Recorded in `V2_RUN_MANIFEST.json` and the final report. Nothing is claimed as completed until
its own counters, validation, checkpoint and exit status are evidenced.

---

## F. Remaining limitations (stated, not hidden)

1. **Fresh-process TF resume is NOT RUN end to end.** The Python-side resume bundle
   (iteration ledger, duals, sampler counters, config/split fingerprints, `next_iteration`
   semantics, incompatible-checkpoint rejection) is implemented and unit-tested; the
   TensorFlow `Saver` already carries the model variables and the Adam slots that live in the
   default graph's collections. A fresh-process CLI comparison (uninterrupted vs
   stop-at-boundary vs resume-to-same-target, comparing sampled actions, rewards, costs, λ and
   optimizer state) has not been executed.
2. **The energy reference is foreground-only by construction.** The constraint now measures the
   foreground ledger, so measurement and reference agree, but this means the constraint does
   **not** price the background owners' energy. Whether the system should be constrained on
   whole-world energy with a background-inclusive reference is a separate, disclosed design
   question; the alternative (rebuild the references with `background_dags=N` and the same link
   process, then re-derive the fractions) was **not** taken.
3. **`--energy-budget-fraction = 0.5` of `E_mec` is an engineering setting**, not a calibrated
   scientific constant. No candidate-feasibility sweep over the corrected formulation has been
   executed on training/validation fixtures; meta-test data was not consulted at any point.
4. **λ is logged after the dual step.** The loop is correct; the log column is off by one
   iteration relative to the penalty it produced.
5. **`steps_with_penalty` counted outer iterations, not steps.** The definition is now
   documented and the value is also exposed as `iterations_with_nonzero_penalty`; the legacy
   key is retained for continuity.
6. **`episodes` counts query-rollout episodes only** (the observer sees `new_paths`), so
   support-rollout episodes are not counted.
7. **Constraint observability.** `M_i`, `B_i`, `S_i`, `g_i`, the positive violation and the
   rollout λ all exist in the per-episode record
   (`energy_telemetry["constraints"]["<name>_raw|budget|scale|signed|violation|lambda"]`), but
   the aggregate logger drops the nested block and `run_tf._collect_evidence` does not surface
   them. A run therefore cannot be audited from its own logged output alone.
8. **Unmodeled energy components** (explicit, not counted as zero): receiver/static radio
   power, CPU idle/leakage between tasks, MEC/RSU static power, backhaul,
   battery/charging/reserve dynamics, warm-standby reservation. A hop priced to zero appends a
   warning that is currently dropped on the production telemetry path.
9. **Reliability remains a surrogate**, not a calibrated chance constraint.
10. `background_policy="all_ue"` crashes (`IndexError` in `shared_scheduler`); observed while
    auditing, not on the launched path (`all_mec`).
