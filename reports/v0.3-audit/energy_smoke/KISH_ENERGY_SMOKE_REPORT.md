# Kish energy/reward/constraint trainer-integration verification

Verdict line (see the end): **energy/reward/constraint trainer integration is
ready for diagnostics**. Not "ready for final training".

## Final SHA and environment

* final SHA: `c93ddf87cae672080ee01499b42c4fb75c39d896` on `erfan/phase4-eval`
* clone: `/opt/margo/mrlco-new-6b`, tree clean before and after every gate
* image: `margo-phase4-tf115-nv2212:latest`
  `sha256:82bc781bc2888a171379e1b45da6846c559ef712e33a76f74948b1cd7172dc46`
* TensorFlow: **1.15.5** (verified inside the image)
* host: kish-ai, NVIDIA GeForce RTX 4090 24 GB, 32 cores, 409 GB free on `/`
* note: a foreign `VLLM::EngineCore` held ~21 GB of VRAM during the early runs
  (0 % utilisation). It was not a blocker: every gate ran to completion, and by
  the final run the GPU was free (1 MiB used).

## Gates (sequential, one clean SHA)

| target | command | exit | wall |
|---|---|---|---|
| energy-tests | `MARGO_ROOT=/opt/margo/mrlco-new-6b bash spec/kish_gpu.sh energy-tests` | 0 | 117 s |
| mask-smoke | `... mask-smoke` | 0 | 23 s |
| mask-runtime | `... mask-runtime` | 0 | 5 s |
| mask-metrics | `... mask-metrics` | 0 | 11 s |
| energy-train-smoke | `... energy-train-smoke` | 0 | 1439 s |

`energy-tests`: `Ran 734 tests in 112.455s` — **OK** (no errors, no failures).
The 13 local TF collection errors are gone in the real image.

Raw logs: `reports/v0.3-audit/energy_smoke/*.log`. Structured evidence:
`reports/v0.3-audit/energy_smoke/kish_energy_smoke_evidence.json`.

## One-iteration smoke (`energy-train-smoke`)

Command: `python -m spec.mask_sanity --mode static --itr 1 --seed 0
--i-allow-gpu --reward-mode latency_only --objective-mode log_only`, in
`mask_mode=static`, `obs_version=v3`, `obs_dim=70`.

* `git_sha = c93ddf8…`, `git_dirty = false`
* `failures = []`, `live_preflight.failures = []` (30000 tasks, 0 deadlines)
* scheduler config: `timing=legacy_frozen_rates radio_timing=legacy_frozen_rates
  energy=physical_v1 radio=physical_v1 scope=system sha=c364f65545f9`
* objective mode: `log_only`, `energy_budget_j=1e12`

CSV: **64 header fields, 1 data row, 64 fields** (aligned, no late-field
desync). Values:

| column | value |
|---|---|
| `energy/requester_joules` | 2009.119920411363 |
| `energy/mobile_joules` | 12516.689593097555 |
| `energy/system_joules` | 156263.55172447627 |
| `energy/primary_joules` | 156263.55172447627 |
| `energy/primary_scope` | system |
| `Average energy,` | 12516.68959309757 |
| `energy/average_energy_scope` | mobile |

Checks: `requester <= mobile <= system` PASS; `primary_scope == system` PASS;
`primary_joules == system_joules` PASS; `Average energy == mobile_joules` and
labelled `mobile` PASS; every value finite, no NaN/Inf.

Objective (from the same row, `validation_per_graph_plans` produced by the
held-out evaluator — **not** `objective/unavailable`):

| key | value |
|---|---|
| `objective/J` | 0.6435688628505934 |
| `objective/latency_norm_mean` | 0.6435688628505934 |
| `objective/soft_tardiness_norm_mean` | 0.0 |
| `objective/c_E_max` | 0.0 |
| `objective/c_E_raw_max` | -0.9999995978973358 |
| `objective/c_H` / `c_F` | 0.0 / 0.0 |
| `objective/energy_violation_rate` | 0.0 |
| `objective/feasible` | 1.0 |
| `objective/n_graphs` | 400.0 |

Plot reward is latency-only by contract; `Average energy` is still logged for the
legacy mobile metric.

## Constraint channel: covered by CPU evidence, not by this smoke

`spec/mask_sanity` sets `MARGO_CONSTRAINTS=off` on purpose, so the one-iteration
CSV carries no `constraint/*` columns and the `not_configured` / penalty=0 case
is not exercised here. It is covered by:

* `test_energy_consumer_migration.py` — `C_TOTAL_ENERGY` raw ==
  `energy_scalar(scope=system)`, `C_UE_ENERGY` raw ==
  `energy_scalar(scope=requester)`, a mobile reference is rejected by the system
  objective/constraint;
* `test_constraints.py` — `not_configured` status, no violation/penalty,
  Lagrangian off when no budget;
* `reports/v0.3-audit/pure_plan_evidence.json` — system raw == system joules with
  and without a total-energy budget.

A constraints-enabled one-iteration smoke is the natural next diagnostic; it was
not run here (out of scope for this verification).

## No-replay evidence

`test_validation_plans.py` pins the schedule-call count with and without the
validation channel (equal), and the smoke itself reports
`objective/n_graphs = 400.0`, i.e. the objective consumed the rollout results the
reward already produced rather than a second scheduler run.

## Fixes made during this verification

| commit | fix |
|---|---|
| `9f7dedb` | smoke stack now receives the resolved primary scheduler config (scope=system); `energy-tests` uses the stdlib runner (image has no pytest) |
| `c790939` | CSV/telemetry tests no longer shadow an installed TensorFlow with a stub under `unittest discover` |
| `df92985` | EAS TF policy smoke uses `PACKED_DIM` instead of a hardcoded 13 |
| `c93ddf8` | `CSVOutputFormat` buffers rows and rewrites header+rows, so a row can never desync (the 64-header/128-row corruption) |

All gates were re-run from the final clean SHA `c93ddf8` after the last fix.

## Verdict

All five gates exited 0 and the one-iteration smoke reported `failures = []`.
The scoped-energy path (reference cache, latency-only reward, system objective,
scoped telemetry, CSV alignment) is verified end to end on Kish.

**energy/reward/constraint trainer integration is ready for diagnostics.**

Still do NOT start a pilot or final training. Next step after review:
queue/parallelism + upstream-parity diagnostic.
