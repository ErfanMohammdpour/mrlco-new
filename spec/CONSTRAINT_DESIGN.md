# Constrained V2V / energy budgets (`MARGO-CONSTRAINTS-v0.1`)

Status: **proposal, not frozen. `paper_result=false`.** No ADR yet; do not report
constrained numbers as paper results until ADR-013 + Phase 6.

## Why this layer exists (measured, `spec/energy_reward_audit.py`)

| finding | number |
|---|---|
| all-MEC is the argmin of BOTH pure-plan latency and mobile energy | 20/20 graphs, 4 distributions |
| mixed plans below the pure-location latency reference (`L_ref_min`) | 100% |
| `j_report` (clipped 0.5/0.5 composite) saturates on the clip floor: a faster plan scores worse | 40/40 graphs |
| ...fixed by `REFERENCE_MODE_PANEL` (candidate panel); the scalar's own preference for all-MEC is real and needs the constraint/energy-scope fix | 7 tests |
| `E_MEC / E_UE` (MEC compute is out of scope, ADR-001) | 0.015–0.046 |
| helper-energy / V2V-share budgets violated by the all-MEC shortcut | never |

Consequence: an energy-only or V2V-only budget is **degenerate** — "send
everything to MEC" satisfies it. A **MEC capacity** budget is required to make
the constrained problem non-degenerate; the energy/V2V budgets are kept as
safety budgets (helper battery, half-duplex airtime, V2V fairness) that start
biting as soon as the energy scope changes.

Second consequence: `j_report` must not be used for reporting until the
reference range is rebuilt from a candidate panel (`pure ∪ greedy_from_mec ∪
2-opt`) instead of pure-location plans only.

## Formulation

```
minimize  J(pi) = E[ 0.5*L_norm + 0.5*E_norm ]      # unchanged objective
s.t.      E[ c_i(pi) ] <= b_i
reward   -= sum_i lambda_i * max(0, (c_i - b_i)/scale_i)
lambda_i <- max(0, lambda_i + eta * mean_batch((c_i - b_i)/scale_i))   # outer loop
```

Budgets are episode-local: `*_frac_of_all_ue` scales by the all-UE plan energy,
`*_frac_of_makespan` by the current plan makespan, `deadline_frac_of_all_mec` by
the all-MEC makespan. Absolute joule/second budgets are also supported and take
precedence over the fractional form.

## Active constraints

| name | metric | default scale | purpose |
|---|---|---|---|
| `ue_energy` | UE-side energy | `E_ue` | device battery |
| `helper_energy` | helper compute + helper radio | `E_ue` | the **V2V** budget that matters |
| `total_energy` | UE + helper | `E_ue` | optional global budget |
| `v2v_airtime` | seconds of V2V channel occupancy | makespan | half-duplex fairness |
| `v2v_task_fraction` | share of tasks on HELPER | 1 | helper availability/fairness |
| `mec_task_fraction` | share of tasks on MEC | 1 | **edge capacity — the binding one** |
| `deadline` | makespan | `L_mec` | optional latency constraint |

Measured design note: a 15% V2V *task* share already occupies 27–54% of the
makespan as V2V airtime (two V2V hops per helper task over a serialized 5 Mbps
half-duplex channel). Keep `v2v_airtime_frac_of_makespan` above the task-share
cap or V2V becomes unusable.

## Code map

| file | role |
|---|---|
| `scheduler/constraints.py` | `ConstraintSpec`, `ConstraintMetrics`, `ConstraintCosts`, `ConstraintController` |
| `scheduler/reward.py` | `telescoping_token_rewards(..., constraints=, duals=)`; penalty applied on the last token (`attribution="terminal"`) or telescoped as signed-cost deltas |
| `scheduler/energy_api.py` | unchanged; supplies `ReferenceRanges` |
| `offloading_env.py` | reads `energy_config["constraints"]`, builds/attaches the controller, observes per-trajectory costs |
| `meta_trainer.py` | `build_frozen_primary_stack(constraints=..., constraint_dual_lr=...)`, one `dual_step()` per outer iteration, `constraint/*` log keys |
| `spec/phase4_train_driver.py` | `MARGO_CONSTRAINTS=<yaml>` for every driver mode; budgets recorded in `config.resolved.json` |
| `spec/constraints.yaml` | the budget profile |
| `spec/constraints_config.py` | loader (`constraints_from_env`) |
| `spec/energy_reward_audit.py` | the audit that produced the table above |
| `spec/constraint_dual_smoke.py` | dual-ascent smoke on real graphs (no TF) |
| `scheduler/tests/test_constraints.py` | 25 tests incl. off-by-default bit-exactness |

## Run

```bash
# unconstrained (default, byte-identical to before)
python meta_trainer.py

# constrained
MARGO_CONSTRAINTS=spec/constraints.yaml python meta_trainer.py

# checks that run on a laptop
python spec/energy_reward_audit.py --graphs 20 --dist 1
python spec/constraint_dual_smoke.py --graphs 6 --dist 1 --iters 16 --dual-lr 0.3
python -m pytest env/mec_offloaing_envs/scheduler/tests -q
```

## Log keys

`constraint/updates`, `constraint/buffer`, `constraint/lambda_<name>`,
`constraint/mean_signed_<name>` (>0 means violated), `constraint/total_violation`.
Audit records add `constraints.lambdas` and the raw cost dict per iteration.

## Open decisions (owner: Erfan)

1. Fix the reporting metric range (candidate panel) before any composite number.
2. Decide the MEC energy scope: include `mec_compute_joules_optional` (system
   energy) or keep mobile-only + a MEC capacity cap (current default).
3. `rho_helper` vs `rho_ue` asymmetry (0.7 vs 1.0 at identical throughput).
4. DVFS: rename to constant-power, or make `f_l` a real optimization axis.
5. Calibrate radio power values against a citable reference (PC5 / LTE).
6. Objective for constrained runs: 0.5/0.5 makes the unconstrained optimum
   all-MEC, so the MEC capacity cap is what carries the trade-off. A
   latency-first constrained run needs an explicit non-publication flag.
