# V2_SERVER_VERIFICATION — kish-ai (aizarepor) state and executed GPU evidence

Everything in this document was read from or executed on the host (`ssh kish-ai`), not inferred
from prior reports. **Treat the identifiers in the previous round's brief as historical.** The
"latest reported GPU run" it named was still alive when this round began and then **failed with
the same reporting defect as the CPU gate** — that is recorded below as evidence, not as a
success.

---

## 1. Host and services

| item | value |
|---|---|
| host | `aizarepor` (ssh alias `kish-ai`, 185.213.165.199:3031) |
| uptime at inspection | up 6 days, load average ≈1.2 |
| CPU | 32 logical cores |
| GPU | 1× NVIDIA GeForce RTX 4090, 24564 MiB total |
| co-resident service | `VLLM::EngineCore` pid 777801 holding ≈21508 MiB — **left untouched** |
| other services | `qdrant/qdrant:latest`, up 6 days — **left untouched** |

The protected vLLM and qdrant services were never stopped or reconfigured. GPU runs used
`--gpu-fraction 0.03` (a hard allocator ceiling of ≈737 MiB) while the card showed a measured
1124 MiB footprint for the training process, i.e. the process stayed far below vLLM's
allocation. Actual headroom on the card is ≈3.0 GiB; this is reported as an operational
constraint, and no claim is made that a constant VRAM figure proves unchanged vLLM latency.

## 2. Images

| image | image id | python | TensorFlow | notes |
|---|---|---|---|---|
| `margo-phase4-tf115-nv2212:latest` | `sha256:82bc781bc2888a171379e1b45da6846c559ef712e33a76f74948b1cd7172dc46` | 3.8.10 | 1.15.5, `import tensorflow.contrib` **succeeds**, CUDA 11.8 | used for every GPU run in this document |
| `margo-phase4-tf115-gpu:latest` | `sha256:af7550b0d2ccfcf4a6fc135693ab6e17a41cd05e5b39dc488a4f8ba2b4cc4a40` | 3.7.5 | 1.15.5 | used for the CPU gate run |

`ENV-GPU-IMG` verification command and result:

```
docker run --rm --entrypoint bash margo-phase4-tf115-nv2212:latest -lc \
  "python -V; python -c 'import tensorflow as tf; print(tf.__version__); import tensorflow.contrib'"
=> Python 3.8.10
=> tf 1.15.5
=> contrib OK
```

## 3. Repository topology (an important correction)

The brief listed `/root/v2-repair` and `/root/v2-verify` as isolated checkouts. The actual layout
is:

| path | what it is |
|---|---|
| `/root/v2-verify` | the only real git repository (remote `origin` = `MA-RGO/MARGO`, `erfan` = `ErfanMohammdpour/mrlco-new`), branch `phase5-realistic-system-v2` |
| `/root/v2-baseline` | a **git worktree** of that repository, detached at `c6ad98a` |
| `/root/v2-repair` | **not** a git repository — a plain working directory, mounted read-write into the running containers |
| `/root/v2-final` | this round's clean deployment tree (rsynced from the local mirror at commit `48a4367`) |

`/root/v2-repair` carried uncommitted modifications to exactly six files that the git repository
did not have (`constraint_channels.py`, `constraints_v2.py`, `energy.py`, `env.py`, `run_tf.py`,
`stack.py`). They were synced into the local mirror, audited, repaired, committed as `48a4367`,
and deployed to `/root/v2-final`.

## 4. The "latest reported GPU run" resolved

`/root/v2-runs/gpu3_20261004T185340` — container `v2gpu3`, started `2026-10-04T18:53:41Z`,
finished `2026-10-04T23:14Z` (51 minutes wall).

```
command: python -m spec.automotive_training.v2.run_tf --device gpu --gpu-fraction 0.03
         --intra-op 8 --inter-op 2 --iterations 2 --seed 0 --background 2
         --link-regime moderate --energy-budget-fraction 0.5 --ue-budget-fraction 1.0
         --r-select 2 --s-select 2 --ckpt-dir /out/ckpt --json /out/run.json
mounts:  /root/v2-repair -> /work (rw)   /root/v2-runs/gpu3_20261004T185340 -> /out
```

**It did not complete successfully.** The preserved `run.json`:

```
iterations_requested     = 2
iterations_started       = 2
iterations_completed     = 1
last_completed_iteration = 0
wall_seconds             = 3057.05
ok                       = True          <-- while the process died with a traceback
error                    = None          <-- the KeyError was outside the guarded try
checkpoint.status        = completed_iterations=1
checkpoint.was_resume    = False
progress                 = {exists: False, path: /out/v2_progress.json}
log tail                 = KeyError: 'iterations_executed'   (run_tf.py:365)
```

So `gpu3` reproduced **both** open contradictions of the previous round in one run:

* the report writer's `KeyError` after training had finished (`ok=True` + `error=None` + traceback);
* `started=2` against `completed=1`, because iteration 1 has `1 % 50 != 0` and therefore never
  ran validation — the only event that used to increment the completion counter.

It nevertheless produced valid per-iteration evidence, which is what made the energy analysis in
the acceptance matrix possible:

```
trained_core_norms        = {n_tensors: 30, total_norm: 173.96, all_finite: true}
constraints.lambdas       = {total_energy: 13.6969, ue_energy: 0.0}
lambda_broadcast.targets  = 11
last_rollout.makespan_s   = 0.316952
last_rollout.system_joules     = 38.320896
last_rollout.requester_joules  = 0.279454
last_rollout.mobile_joules     = 1.096257
last_rollout.background_joules = 35.808284
last_rollout.constraint_signed.total_energy   = 136.604245
last_rollout.constraint_violations.total_energy = 136.604245
penalty.episodes                    = 400
penalty.episodes_with_nonzero_penalty = 200
penalty.penalty_sum                 = 187396.93
```

`iterations_completed = 1` is wrong (two iterations ran), `wall_seconds = 3057` gives an early
upper bound of ≈51 min per validated iteration on the old code, and the energy figures are the
basis of the scope-mismatch finding.

## 5. The CPU gate resolved

`/root/v2-runs/gate2_20261004T103331` — image `margo-phase4-tf115-gpu:latest`, `--device cpu`,
`--iterations 2`, `CUDA_VISIBLE_DEVICES` empty.

The full traceback, in order, is:

```
logs/stdout.log:284  Traceback (most recent call last):
logs/stdout.log:285    File "/usr/lib/python3.7/runpy.py", line 193, in _run_module_as_main
logs/stdout.log:287    File "/usr/lib/python3.7/runpy.py", line 85, in _run_code
logs/stdout.log:289    File "/work/spec/automotive_training/v2/run_tf.py", line 370, in <module>
logs/stdout.log:290      sys.exit(main())
logs/stdout.log:292    File "/work/spec/automotive_training/v2/run_tf.py", line 355, in main
logs/stdout.log:292      "v2_constraints": getattr(trainer, "auto_v2_constraints", None),
logs/stdout.log:293  KeyError: 'iterations_executed'
```

**Resolution of the contradiction.** The run did **not** fail in training: the printed
`run.json` for gate2 shows the loop reached `outer_update_count = 2` and wrote a real
checkpoint. The process failed in the *reporting* phase, after the checkpoint, because the
terminal summary printed `evidence["iterations_executed"]`, a key `_collect_evidence` has never
produced. Consequences:

* `error=None` — correct for what it measures (whether `trainer.train()` raised); the
  `KeyError` happened outside that `try`, so it was never recorded.
* `ok=True` — the report's success flag was written *before* `_assert_real_run`, which would have
  caught this. **The JSON reports success prematurely. This is a defect, now fixed.**
* `completed=1` — **the counter is also wrong** (two iterations ran). Also fixed.

## 6. Executed evidence on the repaired tree

### 6.1 Non-TensorFlow suites (local mirror at `48a4367`, `ENV-CPU-LOCAL`)

```
python3 -m pytest env/mec_offloaing_envs/scheduler/tests -q -p no:nameko \
  --ignore=env/mec_offloaing_envs/scheduler/tests/test_phase2_encoder_tf.py \
  --ignore=env/mec_offloaing_envs/scheduler/tests/test_phase3_learning_tf.py
=> 1389 passed, 19 skipped, 0 failed  (194 warnings, 4 subtests passed)
```

All 19 skips are TensorFlow-gated and carry an explicit reason.

The new accounting suite is a genuine regression test: against the previous code it reports
`7 failed, 8 passed`; against the repaired code, `15 passed`.

```
python3 -m pytest env/mec_offloaing_envs/scheduler/tests/test_v2_iteration_accounting.py -q
  old code: 7 failed, 8 passed
  new code: 15 passed
```

### 6.2 GPU acceptance run on the repaired tree

See the log/artifact paths recorded in `V2_RUN_MANIFEST.json` under
`acceptance_run_gpu_final`. The following table is filled from the run's own `run.json`,
`v2_progress.json` and `v2_phase_timings.jsonl`.

| item | value |
|---|---|
| run dir | `/root/v2-runs/gate_final_20261005T0015` |
| source | `/root/v2-final` @ `48a4367` |
| image | `sha256:82bc781bc288…` |
| job | detached `docker run --name v2gatefinal`, `run.sh` preserved in the run dir |

## 7. Reproduce / monitor

```bash
# state
ssh kish-ai 'docker ps --format "{{.Names}} {{.Status}}"; nvidia-smi --query-gpu=memory.used,memory.total --format=csv'

# acceptance run
ssh kish-ai 'tail -f /root/v2-runs/gate_final_20261005T0015/logs/stdout.log'
ssh kish-ai 'cat  /root/v2-runs/gate_final_20261005T0015/v2_progress.json'
ssh kish-ai 'cat  /root/v2-runs/gate_final_20261005T0015/v2_phase_timings.jsonl'

# final 500 run
ssh kish-ai 'tail -f /root/v2-runs/final500_seed0_20261005T0200/logs/stdout.log'
ssh kish-ai 'cat  /root/v2-runs/final500_seed0_20261005T0200/v2_progress.json'
```

## 8. Operational notes and constraints reported honestly

* The GPU is shared with a persistent vLLM instance holding ≈21.5 GiB of 24564 MiB, leaving
  ≈3.0 GiB. `--gpu-fraction 0.03` is a hard allocator ceiling, not a cap on total process
  memory; the measured process footprint was 1124 MiB.
* `per_process_gpu_memory_fraction` does not bound non-allocator overhead, so actual usage was
  measured rather than assumed.
* A CPU-only timing run (`tim1`) was started and then discarded: it was still inside the
  support-rollout phase of iteration 0 after ≈10 minutes, which is not representative of the GPU
  path and would not have produced a usable estimate. It was replaced by per-phase
  instrumentation in the code itself (`v2_phase_timings.jsonl`).
* No protected service was stopped, reconfigured or restarted. No duplicate diagnostic job was
  left running.
