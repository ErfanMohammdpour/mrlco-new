# V2_PRODUCTION_CONNECTION_MATRIX / TF_INTEGRATION / ACCEPTANCE (IN PROGRESS)

Status of this file: **IN PROGRESS — not a READY claim.** It records the production call graph,
which links are connected and executed, which are still broken, and the exact commands/logs.
Large run artifacts live on kish-ai under `/root/v2-runs/` (not in Git); source and tests live in
Git as normal.

## Provenance (recorded before every run)

| item | value |
|---|---|
| starting HEAD | `774ad29f9bbc67767d89f1389ea6d646f881bd04` |
| repair commit | `17cf527` (counters, single dual authority, energy attribution, reward/telemetry consistency, counter names) |
| image | `margo-phase4-tf115-gpu:latest` = TF 1.15.5 + `tf.contrib`, Python 3.7.5, NumPy 1.18.5 |
| isolated checkouts | `/root/v2-verify` (superseded run), `/root/v2-repair` (current, stack.py sha256[:16] `a2934880f3789cd0`, verified identical inside the container) |
| GPU policy | every container runs `CUDA_VISIBLE_DEVICES=""`; vLLM (21.5/24.6 GB) and qdrant were never touched |
| meta-test | closed, `meta_test_access_count = 0` |

## Connection matrix

| # | link (producer → consumer) | state | proof |
|---|---|---|---|
| 1 | resolved config → world builder (`v2.world.build_world`) | CONNECTED | one builder used by env, gate, parity, eval_loop |
| 2 | world → scheduler (`schedule_shared`) | CONNECTED | `test_v2_world`, `test_v2_booking` |
| 3 | scheduler → energy ledger (`schedule_energy`) | CONNECTED + FIXED | `background_dag_ids` now passed on the production path (`env.step`); `dag_filter` added for foreground-requester attribution |
| 4 | energy ledger → constraint costs (`v2_metrics`, `costs_from_metrics`) | CONNECTED | `test_v2_energy`, `test_v2_constraints` |
| 5 | costs → telemetry `violation/<NAME>` | CONNECTED + FIXED | `v2/constraint_channels.py` uses the FROZEN `evaluate_constraints`; observer returns non-empty |
| 6 | telemetry → trainer observer → dual ascent | CONNECTED + EXECUTED | real run: 200 signed rows, λ 0 → 6.9504 |
| 7 | dual → broadcast to env + clones | CONNECTED + FIXED | measured 11 targets (1 env + 10 clones); the legacy overwrite that created two authorities is removed |
| 8 | reward ← penalty (terminal token, once) | CONNECTED | telemetry no longer divides the penalty twice |
| 9 | iteration counters → runner status | CONNECTED + FIXED | started/completed counted from the real loop phases; atomic `v2_progress.json` |
| 10 | trainer → validation → baseline panel | PARTIALLY CONNECTED | panel override scores on the v2 world with the v1 `_schedule` trap armed; **canonical-world identity with the policy world is NOT yet proven** |
| 11 | trainer evaluator → real TF R/S | **NOT CONNECTED** | the evaluator still has no executed R/S loop over the TF policy |
| 12 | runner → `trainer.train()` | CONNECTED + EXECUTED | `v2/run_tf.py` (new); iteration 0 reaches PPO and energy logging |
| 13 | checkpoint → fresh-process resume | **NOT CONNECTED** | `Saver.restore` restores TF variables only; `start_itr`, dual, optimizer, RNG and world identity are not restored |
| 14 | geometry gate winner evidence | CONNECTED + FIXED | winning plan taken from `sr.plan` with objective replay assertion |

## Executed evidence (real TF, CPU-only)

Iteration 0 of the real trainer (container `v2confirm`, source `/root/v2-verify`):

```
average task losses 0.31763643 | average value losses 2.1352415
"Average energy per iteration 0: 1259.5906"
mean system 62.98 J | requester 0.275 J | mean violation 139.02
last rollout signed total_energy +128.80, ue_energy -0.506
lambda 0 -> 6.950428978747282 (1 update, 200 signed rows) | broadcast targets 11
30 core tensors, all finite, norm 173.9137 | checkpoint saved
```

Two crashes were found and fixed by this run (the mounted source was proven current by
identical sha256 inside the container): `env.last_constraint_costs` lacked `.active`
(fixed → `ConstraintCostBatch`), and the lambda broadcast carried the legacy deadline channel
names (fixed → single dual authority).

## Acceptance gate (current, RUNNING)

```
run dir : /root/v2-runs/gate_20261004T090347
command : docker run --rm --name v2gate -e CUDA_VISIBLE_DEVICES= -v /root/v2-repair:/work \
          -v <rundir>:/out -w /work margo-phase4-tf115-gpu:latest \
          python -m spec.automotive_training.v2.run_tf --iterations 2 --seed 0 \
          --background 2 --link-regime moderate --energy-budget-fraction 0.5 \
          --r-select 2 --s-select 2 --ckpt-dir /out/ckpt --json /out/run.json
logs    : <rundir>/logs/stdout.log     progress: <rundir>/ckpt/v2_progress.json
```

Observed at the time of writing: `Iteration 0` reached, `average task losses 0.31763643`
logged, container up 7 minutes. **No completed iteration yet, so the 500-iteration run has NOT
been launched.**

## Remaining blockers before 500 (implementation, not modelling)

1. **Evaluator R/S (#11)** — the trainer's evaluator needs an executed R=2/S=2 loop over the
   real TF policy, with policy and baselines on proven-identical canonical worlds.
2. **Fresh-process resume (#13)** — a versioned checkpoint bundle (iteration, duals, inner/outer
   optimizer slots, RNG/realization, sampler/CRN counters, config/schema/split hashes) restored
   before the first reset; `--iterations` must mean a TOTAL target.
3. **Budget feasibility (2.4)** — measured mean violation 139 at fraction 0.5 needs a scope
   decision and a candidate-feasibility study on training/validation fixtures. The physical cause
   is that the world-level measured energy (including background and MEC) is compared against a
   foreground all-UE reference; that mismatch is an accounting-scope question, not proof of
   infeasibility. No budget was loosened.

## Timing

Measured ~830 s of CPU TF time to reach the constraint/validation stage of ONE iteration
(10 meta tasks × 20 support × 20 tokens, 3 inner steps). A 500-iteration estimate must be
recomputed from completed-iteration and validation measurements, not from this figure alone.
