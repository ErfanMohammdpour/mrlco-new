# Correctness repair status (pre-long-run gates A-F)

`status = BLOCKED_FOR_LONG_RUN` until Gate F passes on the GPU.

| gate | item | state |
|---|---|---|
| A | preserve the 5x500 pilot + errata (no raw file touched) | DONE (`reports/pilot500/PILOT500_ERRATA.md`, status `ENGINEERING_PASS` / `SCIENTIFIC_PERFORMANCE_SUPERSEDED`) |
| B | validation measures the trained core: verified core->scratch sync on every evaluation, adaptation never mutates the core, no accumulation across validations | DONE (code + blocking tests) |
| C | multipliers reach every rollout env (iterative + parallel executors), hard failure if nobody accepts them | DONE (code + blocking tests) |
| D1 | Lagrangian penalty applied exactly once | DONE (code + blocking tests) |
| D2 | dual batch reset after every outer iteration | DONE (code + blocking tests) |
| E | provenance: real git sha + dirty flag, run-kind method id, exact sampler counters with a contract check, correctly named/denominated metrics, k0 latency logged | DONE (code + blocking tests) |
| F | 101-iteration GPU proof: validations at 0/50/100, penalty strictly non-zero in the rollouts, lambda visible in the clones, dual batch empty after each iteration, no NaN, no TF variable collision, meta-test closed | PENDING (run below) |
| G | salvage-evaluate the five old core checkpoints with the corrected evaluator | PENDING |
| I | MC headroom audit: does mixed placement still beat the best pure plan under the MC runtime? | PENDING |
| J/K/L | budget decision, frozen evaluation protocol, meta-test opening | PENDING (after F/G/I) |

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
