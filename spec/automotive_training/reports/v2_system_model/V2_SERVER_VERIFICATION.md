# V2_SERVER_VERIFICATION — execution on the project's own TF 1.15 server environment

Round 2 executed the suite on `kish-ai` (aizarepor, x86_64, RTX 4090, 88 GB RAM) inside the
project's own image `margo-phase4-tf115-gpu:latest` (TensorFlow **1.15.5 with `tf.contrib`**,
Python 3.7.5, numpy 1.18.5). This is the environment the frozen PPO/MRLCO chain requires.

Isolation and safety, as required:

* the existing `/opt/margo` checkout (26 GB, the old working copy) was **not touched**; a
  separate copy of the verified commit was placed at `/root/v2-verify`;
* the baseline comparison ran from a separate `git worktree` at `/root/v2-baseline`;
* the running **vLLM service was never disturbed**: its 21.5 GB of the 24.6 GB GPU was left
  alone and every container was started with `CUDA_VISIBLE_DEVICES=""` (CPU-only), so no GPU
  memory was taken and no service was stopped;
* `qdrant` and the other containers were left running.

## Baseline vs current branch (same container, same command, same day)

Command (both runs):

```
docker run --rm -e CUDA_VISIBLE_DEVICES= -e TF_CPP_MIN_LOG_LEVEL=2 \
  -v <checkout>:/work -w /work margo-phase4-tf115-gpu:latest \
  bash -lc 'python -m unittest discover -s env/mec_offloaing_envs/scheduler/tests \
            -t env/mec_offloaing_envs/scheduler/tests -p "test_*.py" -v'
```

| run | commit | tests | failures | errors | problems |
|---|---|---|---|---|---|
| baseline (entry point) | `c6ad98a` | 1265 | 5 | 38 | **43** |
| current branch, before this round's compat fixes | `36cf741` | 1368 | 3 | 44 | 47 |
| **current branch, after the compat fixes** | `9428d2e` | **1394** | **2** | **23** | **25** |

Set comparison of the individual failing tests:

* **NEW problems introduced by this branch: 0.**
* baseline problems RESOLVED: 18 — including
  `TestHelperInScheduler::test_idle_helper_with_long_contact_beats_mec_when_mec_is_overloaded`
  (the originally reported failure), the 6 `TestREV2Invariants` PEP-584 errors, the 3
  `end_lineno` errors, the `statistics.fmean` errors, and `_FailedTest::test_effective_rank`
  (walrus operator).
* REMAINING: 25, **every one of them present in the baseline list** — i.e. pre-existing, not
  regressions. They are: 15 encoder tests whose failure is caused by pre-existing global
  `encoder_obs` schema state in the test suite, 3 `TestSeq2SeqPolicyCRN` tests with the same
  cause, 2 `TestPreflightCli` exit-code failures, `TestValidationMeasuresTheTrainedCore`
  (TF variable-reuse when validation runs twice in one process), and 2 `setUpClass` errors.

## The findings that mattered

1. **The frozen TF chain cannot run on this Mac.** `import tensorflow` fails for Python 3.14 and
   the chain needs `tf.contrib` (removed in TF 2). The server image has it.
2. **The TF suite had never been executed on this branch.** Running it for the first time
   revealed 43 problems at the entry commit — all pre-existing, most of them Python-3.7
   incompatibilities that a 3.12/3.14 interpreter silently tolerates.
3. **`tf.contrib` verification now exists for v2**: in the container the 14 v2 modules report
   `Ran 138 tests ... OK`, including `test_v2_crn_tf` (the CRN/Gumbel path against the real
   `Seq2SeqPolicy`), `test_v2_stack`, `test_v2_observation*`, `test_v2_energy`,
   `test_v2_constraints`, `test_v2_booking`, `test_v2_contact_and_fallback`,
   `test_v2_future_blindness`, `test_v2_ppo_contract` and
   `test_v2_realization_and_resume`. That is genuine TF-side verification of the v2 changes,
   not a numpy substitute.

## What is still NOT RUN even on the server

* the **full v2 training smoke** (`build_automotive_v2_stack` + `trainer.train()` with the
  frozen `meta_batch_size=10` / `support=20` and the MRLCO outer update). The builder and its
  guards are exercised by `test_v2_stack.py`, and the environment side is exercised by the
  tests above, but a complete iteration through the frozen trainer has not been executed.
* the **diagnostic 1 seed × 500 outer iterations** run. It must not start before the above.
* GPU execution: deliberately not used (the GPU is occupied by another service).

Exact commands are in `V2_RUN_MANIFEST.json`; raw logs are preserved on the server at
`/root/v2-verify/_verify/{baseline_tf115,full_after_fixes,targeted}.log`.
