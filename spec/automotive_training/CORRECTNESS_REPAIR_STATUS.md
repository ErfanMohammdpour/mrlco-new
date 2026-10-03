# Correctness repair status (pre-long-run gates A-F)

`status = BLOCKED_FOR_LONG_RUN`: Gates A-I are green; the remaining blockers are
protocol/claim decisions (Gate J/K), not defects - a fresh campaign must not start before
the evaluation protocol is frozen and the adaptation/claim questions are answered.

| gate | item | state |
|---|---|---|
| A | preserve the 5x500 pilot + errata (no raw file touched) | DONE (`reports/pilot500/PILOT500_ERRATA.md`, status `ENGINEERING_PASS` / `SCIENTIFIC_PERFORMANCE_SUPERSEDED`) |
| B | validation measures the trained core: verified core->scratch sync on every evaluation, adaptation never mutates the core, no accumulation across validations | DONE (code + blocking tests) |
| C | multipliers reach every rollout env (iterative + parallel executors), hard failure if nobody accepts them | DONE (code + blocking tests) |
| D1 | Lagrangian penalty applied exactly once | DONE (code + blocking tests) |
| D2 | dual batch reset after every outer iteration | DONE (code + blocking tests) |
| E | provenance: real git sha + dirty flag, run-kind method id, exact sampler counters with a contract check, correctly named/denominated metrics, k0 latency logged | DONE (code + blocking tests) |
| F | 101-iteration GPU proof: validations at 0/50/100, penalty non-zero in 74/101 iterations (iteration 0 starts at lambda=0), lambda broadcast to 11 envs, 2 verified core syncs per validation, core preserved, dual batch empty every iteration, no NaN, meta-test 0 | **PASS** (`reports/gateF_report.json`, `reports/CORRECTNESS_REPAIR_ANALYSIS.md`) |
| G | salvage-evaluate the five old core checkpoints with the corrected evaluator | **DONE**: 10/10 checkpoints at k0 mean 0.02583 s (range 0.02538-0.02698), zero hard violations, zero HIGH tardiness - the pilot table was 3.0-5.6x pessimistic (`reports/gateG/`) |
| I | MC headroom audit: does mixed placement still beat the best pure plan under the MC runtime? | **DONE**: all-MEC is the best pure plan on 100% of graphs; mixed wins on 52.5-60% of instances with a median +0.5-0.7% but a negative mean; HELPER is 2-3% of tokens (`reports/gateI/mc_headroom_audit.json`) |
| J/K/L | budget decision, frozen evaluation protocol, meta-test opening | PENDING: the corrected pilot already matches the references, so a longer budget is not justified by a latency gap; decide the Gate I claim scope and freeze the rollout protocol (multiple MC realizations per graph) first |

Blocking tests added (must stay green):

```
env/mec_offloaing_envs/scheduler/tests/test_automotive_correctness_gates.py   (9 tests, no TF)
env/mec_offloaing_envs/scheduler/tests/test_automotive_validation_sync.py    (4 tests, TF-gated)
```

Gate F command (Kish, GPU):

```bash
docker run --rm --gpus all -e PYTHONPATH=/work -e MARGO_ALLOW_GPU=1 \
  -e CUDA_VISIBLE_DEVICES=0 -e TF_FORCE_GPU_ALLOW_GROWTH=true -e TF_CPP_MIN_LOG_LEVEL=2 \
  -v /opt/margo/mrlco-new-6b:/work -w /work margo-phase4-tf115-nv2212:latest \
  python spec/automotive_gpu_smoke.py --long --seed 0 --i-allow-gpu --gpu 0 \
    --iters 101 --run-kind gatef_proof
```
