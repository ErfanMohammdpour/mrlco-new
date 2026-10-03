# V2_REMAINING_WORK — corrections to my own claims, the exact failure inventory, and the path

This file records the review of HEAD `ddfcfa7` and what it changed. Several of my earlier
statements were **stronger than the evidence supported**; the corrections are stated first.

## Corrections to my previous claims

| my claim | correction |
|---|---|
| "138 tests, real TF verification, not a numpy substitute" | Only `test_v2_crn_tf.py` executes the TF policy. The other 13 v2 modules are numpy/CPU tests that HAPPEN to run inside the TF container; running a numpy test in a TF image does not make it a TF-network integration test. `test_v2_ppo_contract.py` checks a numpy reference of the ratio plus **source-text guards**; `test_v2_stack.py` checks builder guards and delegation, not a built trainer. |
| "checkpoint/resume verified" | The bit-identical resume belongs to the **numpy reference learner** (`smoke.py`). Weight/Adam/learner state of the real TF PPO/MRLCO is NOT proven. Correct statement: *the reference-chain resume is exact; real PPO/MRLCO resume is not yet demonstrated.* |
| "the constraint channel is wired" | At `ddfcfa7` it was **not**: `stack.py` built the env without `constraints_enabled`, then attached the legacy controller and set `constraint_spec=None`, and the telemetry did not emit the `violation/<NAME>` keys the frozen observer reads, so `constraint_violations_from_telemetry` returned `{}`. Independently reproduced. |
| "no new failures, so the suite is acceptable" | A baseline comparison is evidence about REGRESSIONS, not about acceptability. The variable-reuse error fires exactly on the back-to-back k=0/k=3 validations, i.e. on the trainer's main path. |

## The 25 remaining failures — exact inventory

My previous grouping (18+2+1+2) did not sum to 25. The corrected grouping is **22 + 2 + 1 = 25**,
and **all 25 are present in the `c6ad98a` baseline**:

| group | count | tests | exception | cause | effect on the main path |
|---|---|---|---|---|---|
| pre-existing global `encoder_obs` schema state | **22** | `TestEncoderPacking` (7), `TestEncoderTFsmoke::setUpClass`, `TestMeanaggFixtureRegression`, `TestMetaTrainStats` (4), `TestTopologyAndPermutation` (2), `TestEncoderPermutationAndTopology` (2), `TestTaskGraphEncodePath`, `TestBestOfKTF::test_sample0_equals_greedy`, `TestSeq2SeqPolicyCRN` (3) | `ValueError: encoder packed dim 50/109 != 79`, `EncoderGraphError: obs automotive_mc_obs_v1 requires resource_vec [4]`, `encoder stats feature_names/order mismatch` | `encoder_obs.set_obs_version` mutates MODULE GLOBALS; per-test save/restore captures an already-leaked value, so a leak propagates. Pre-existing design flaw in the suite. | Indirect but real: it is why the TF-side v2 test could not see the v2 schema, and any run that mixes v1 and v2 encoder tests in one process is order-dependent. |
| pre-existing preflight CLI exit code | **2** | `TestPreflightCli::test_preflight_only_passes_and_records_provenance`, `::test_preflight_off_mode_has_no_placeholder_requirement` | `AssertionError: 2 != 0` | the CLI exits 2 on this host (env/config dependent) | None on training; it is a pre-flight reporting entry point. |
| pre-existing TF variable reuse | **1** | `TestValidationMeasuresTheTrainedCore::setUpClass` | `ValueError: Variable inner_update_parameters_task_0/.../inner_adam_task_0/ already exists` | the same `variable_scope` is entered twice in one process | **HIGH: this is the trainer's own path.** Validation runs k=0 and k=3 back to back, so this must be fixed before any multi-iteration TF run. |

Attempted and REVERTED: adding `tearDownModule` schema resets to the encoder test modules. It
changed cross-module ordering semantics (14 previously-passing local tests began failing), so I
reverted it rather than land a half-understood test modification. The 22-group therefore stays
open with its cause documented above.

## What was actually fixed in this round

`371d69e` (local suite 1382 passed / 0 failed / 19 skipped):

1. **slot/context mapping** — `_packed_observation` passed dataset graph ids where flat slot
   positions were expected; `[0, 0]` collapsed both slots onto slot 0's context. Now `[0, 1]`,
   with tests for the call sequence, for two same-graph slots with different contact, and for
   slot-vs-dataset-graph identity.
2. **the trainer constraint channels** — NEW `v2/constraint_channels.py` computes
   `violation/<NAME>`, `lambda/<NAME>`, `n_violating/<NAME>` by handing the FROZEN v1
   `evaluate_constraints` the v2 foreground availability mapping. No formula reimplemented.
   MC-pruned tasks are excluded through the world's own DAG view; tasks with no declared
   subdeadline are reported as `unjudged_tasks_no_subdeadline` rather than given a fake one.
   The frozen observer now reads non-empty channels.
3. **`last_constraint_costs`** is a `ConstraintCostBatch(list)` exposing `.active`/`.as_dict()`.
4. **manager controller API + stack wiring** — `observe_signed` (rejects non-finite values),
   `reset_batch`, `batch_size`, `status`; `stack.py` enables the v2 manager, sets
   `env.constraint_controller = env.constraint_manager` so ONE lambda vector drives both the
   reward penalty and the dual ascent, and adds `AutomotiveV2Trainer` routing the signed costs.
   Verified: telemetry signed 81.1/81.4 (over budget) → lambda 0 → 4.062; 30 under-budget rows →
   lambda falls to 4.012.

## Not done (still open)

1. **`V2HeldOutEvaluator` real R/S execution.** It still only carries constructor/metadata; the
   inherited `evaluate_all`/`baseline_panel` reaches v1 through `query_env._schedule(...)` via
   `__getattr__`. `eval_loop.py` being correct does NOT fix the evaluator the trainer uses.
   Also, some legacy metrics are still read with a `0.0` default, which preserves the
   false-feasibility risk.
2. **Per-node epsilon is a graph summary** (`min/mean/max` over the tasks). It is not a per-node
   feature column; making it one changes the packed layout and needs a per-node context block.
3. **The TF runner.** No `main`/`trainer.train()` entry exists for `v2/stack.py`; the manifest's
   `python -m spec.automotive_training.v2.stack` command is therefore wrong and will be fixed
   when the runner is written. The frozen trainer iteration, the TF checkpoint save/restore of
   the real learner, and the 1x500 diagnostic run are all NOT RUN.
4. **The TF variable-reuse fix (k=0/k=3 validation)** — a prerequisite for any multi-iteration
   run, listed in the inventory above.

## Order for the next round (unchanged from the review)

1. fix the connections/bugs above (evaluator R/S + v2 baselines, per-node epsilon, TF variable
   reuse, the runner);
2. targeted tests in the same container (shared-graph slots with distinct contexts — done;
   non-zero violation reaching the trainer — done at the manager level, still needed through a
   real iteration; dual up/down — done at the manager level; double validation in one process;
   proof that no baseline reaches the v1 scheduler);
3. a real runner with `build_automotive_v2_stack` + a TF session + `trainer.train()`;
4. several complete iterations incl. validation, weight movement, real energy/penalty and
   checkpoint, then restore in a fresh process and compare;
5. regenerate the gate/acceptance and measure full-iteration throughput; only then start 1x500.

GPU is not required for any of this: the TF 1.15 container runs on CPU.
