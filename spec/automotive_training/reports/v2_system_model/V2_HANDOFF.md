# V2_HANDOFF — state at session boundary

**STATUS: RUNNING 500** (launched, verified progressing; nowhere near 500 completed)

## Source revision

| item | value |
|---|---|
| branch | `phase5-realistic-system-v2` |
| base HEAD at round start | `0f9cb096b76a84d3c7b260d539b4e2ad15dceac6` |
| repair commits | `48a4367`, then `ef88073` |
| deployed tree | `/root/v2-final` (rsynced from the local mirror of the above) |
| server git repo | `/root/v2-verify` (the only real repo; `/root/v2-repair` is NOT a repo) |
| local mirror | `<workspace>/_scratch/repo/margo` |

## Active job

```
path      : /root/v2-runs/final500_seed0_20261005T0200
container : v2final500
image     : margo-phase4-tf115-nv2212:latest @ sha256:82bc781bc288...
source    : /root/v2-final @ ef88073
command   : bash run.sh   (see run.sh / SOURCE_REVISION.txt in the run dir)
config    : seed 0 | --iterations 500 (TOTAL) | background 2 | link moderate
            background-policy all_mec | energy-budget-fraction 0.5 | ue 1.0
            r-select 2 | s-select 2 | meta_batch 10 | support/query 20 | k 3
            validation_interval 50
artifacts : logs/stdout.log, run.json, v2_progress.json,
            v2_phase_timings.jsonl, ckpt/v2_run.ckpt, ckpt/v2_resume_state.json
```

Verified at launch: the container resolves `run_tf` from `/work/spec/automotive_training/v2/run_tf.py`
(i.e. the repaired tree), the run reaches `---- Iteration 0 ----`, completes the support rollout,
the inner PPO and the query rollout, and logs `Average energy per iteration 0`.

## What was completed this round

1. **Resolved both open contradictions** against preserved artifacts (`V2_SERVER_VERIFICATION.md`).
   * gate2 and GPU3 both died in the REPORTING phase on `KeyError: 'iterations_executed'` after
     training had finished, while `run.json` said `ok=true`, `error=null`. Fixed, with tests.
   * `completed_iterations` counted a validation phase ENTRY and never counted iterations without
     a scheduled validation. GPU3 reported `started=2 / completed=1`. Fixed with an explicit
     end-of-iteration hook and separate persistent counters.
2. **Found and fixed the headline scientific defect** (`V2_ACCEPTANCE_MATRIX.md` §B): the
   total-energy constraint measured the WHOLE WORLD while its budget/scale were anchored on a
   foreground-only, background-free reference. Measured `M/B = 274.2`, normalised violation
   ≈137, penalty ≈1.9e3/episode. The "≈139" was that scope artifact, **not** a measured
   139-fold violation and **not** evidence of global infeasibility. Measurement now uses the
   foreground ledger; the anchor is `refs.E_mec`. Disclosed, not loosened.
3. **Added a run guard** so a run whose Lagrangian loop never closed cannot report success.
4. **Resume bundle** (`v2_resume_state.json`) with `--iterations` as a TOTAL target; Python-side
   tested.
5. `1389 passed, 19 skipped, 0 failed` on the non-TF scheduler suite; the new accounting suite is
   a genuine regression test (7 failures on the previous code).

## Acceptance result on the repaired tree

GPU run `gate_final_20261005T0015`, source `48a4367`:

```
ok                     = True
iterations_started     = 2
iterations_completed   = 2      <-- the previous contradiction is resolved
last_completed_iteration = 1
next_iteration         = 2
validations_completed  = 1      <-- real R=2/S=2 validation executed
error                  = None   (and no traceback: the KeyError is gone)
checkpoint.status      = completed_iterations=2
checkpoint.promotable  = True
checkpoint.resume_state_path = /out/ckpt/v2_resume_state.json
lambda_broadcast.targets = 11
last_rollout system/requester/background = 36.311 / 0.290 / 35.808 J
last_rollout signed = {total_energy: -0.4715, ue_energy: +0.0384}
```

`total_energy` signed **-0.47** and `ue_energy` signed **+0.038**: after the scope fix the energy
constraint is genuinely near its budget and no longer structurally infeasible. In that run
λ stayed 0 because the observed batch mean was ≤ 0 — correct Lagrangian behaviour for a
satisfied batch, but it means this particular 2-iteration run does **not** by itself demonstrate
a nonzero penalty appearing in a later iteration (see the gaps below).

Two further defects were found **in that run's own output** and fixed in `ef88073`:
`v2_phase_timings.jsonl` came out 0 bytes and `v2_progress.json` was never written (the rich
record raised, and the exception escaped before the append); and `outer_updates_completed` read 0
because the `outer_update` audit call was gated on `audit_writer`.

## Measured timing

* `gate_final` iteration 0 completed ≈9 min 40 s after process start (container start 20:02:05,
  `meta_model_0.ckpt` at 20:11:46); iteration 1 reached its banner at 20:17 and the whole
  2-iteration run took `wall_seconds = 3070` ≈ 51 min, dominated by the initial iteration's
  validation panel.
* The v2 energy ledger is NOT the bottleneck: measured `reset` 8.6 s and `env.step` 0.65 s for
  10 slots, and `schedule_energy` with the new `dag_filter` costs the same as without it
  (ratio 1.00, 19.6 ms per call).
* **The 500-run is slower than the acceptance run under vLLM contention** (iteration 0 still in
  flight ≈27 min in). At the observed rate a 500-iteration run is on the order of **days**, not
  hours. `v2_phase_timings.jsonl` from the 500-run is the authoritative per-phase breakdown once
  its first iterations land.

## Exact blockers / open gaps

1. **Fresh-process TensorFlow resume equivalence is NOT RUN.** The Python-side bundle is
   implemented and unit-tested; the CLI comparison (uninterrupted vs stop-at-boundary vs
   resume-to-same-total-target) has not been executed.
2. **NVML/NVML-free device placement evidence is not captured.** GPU runs demonstrably load the
   CUDA 11.8 image and `tf.contrib`, but per-op device placement for the production policy and
   optimizer was not enumerated.
3. **Candidate-feasibility sweep** over the corrected energy formulation on training/validation
   fixtures is NOT RUN; `--energy-budget-fraction 0.5` of `E_mec` remains a declared engineering
   setting.
4. The corrected 2-iteration acceptance run had `episodes_with_nonzero_penalty = 0`, so
   "a later rollout is penalised with the updated λ" is demonstrated by manager-level tests, by
   the out-of-repo clone-broadcast probe, and by the historical runs' arithmetic — not yet by a
   separate clean GPU iteration pair on the fixed code.
5. The constraint measures the **foreground** ledger by design, so background owners' energy is
   deliberately not priced by the constraint.

## Next commands

```bash
# 1. monitor and confirm the first completed iteration writes its telemetry
ssh kish-ai 'cat /root/v2-runs/final500_seed0_20261005T0200/v2_progress.json'
ssh kish-ai 'cat /root/v2-runs/final500_seed0_20261005T0200/v2_phase_timings.jsonl'

# 2. if telemetry warnings appear, they are printed as:
#    "WARNING v2 progress telemetry failed: progress=... phase_timings=..."

# 3. continue/monitor
ssh kish-ai 'tail -f /root/v2-runs/final500_seed0_20261005T0200/logs/stdout.log'

# 4. resume from a completed boundary (only if a compatible checkpoint exists)
#    see "resume_command" in V2_RUN_MANIFEST.json

# 5. next evidence to collect, in priority order
#    a) first iteration boundary of the 500-run: counters + phase timings
#    b) fresh-process resume comparison
#    c) candidate-feasibility sweep on the corrected formulation
#    d) an iteration pair that demonstrates a nonzero penalty with the updated lambda
```

## Do not repeat

* Do not re-derive the "≈139 violation" as a physical result; it is a fixed scope artifact.
* Do not treat `--iterations` as an additional count.
* Do not launch a duplicate diagnostic job; `gate2`/`gpu3` evidence is preserved and read.
* Do not stop or reconfigure vLLM, qdrant, or unrelated jobs.
