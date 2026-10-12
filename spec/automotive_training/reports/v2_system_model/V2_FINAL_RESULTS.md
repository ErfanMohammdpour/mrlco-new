# MARGO v2 — FINAL CAMPAIGN RESULTS (seed 0, 1500 outer iterations)

Completed run: `/opt/margo/campaigns/primary_1500_seed0_20261011T065431Z`
Source: `phase5-realistic-system-v2 @ ce562cf4` (the repaired scheduler), read-only mount.
`run.json`: `ok=true`, `iterations_completed=1500/1500`, `outer_updates_completed=1500`,
`validations_completed=30`, `resume_mode=fresh`, `state_finite=true`, zero tracebacks.
Duals at completion: `total_energy=0.348`, `ue_energy=0.0`.

Evaluation protocol: production `evaluate_all(k_steps=0/3)`, validation query split (40 graphs),
R=2/S=2, paired worlds, same scheduler. Final checkpoint `gen_00001499`.

## 1. Action mix (what the model chose)

| location | count (40 graphs × 20 tasks) | fraction |
|---|---|---|
| MEC | 800 | **100 %** |
| UE | 0 | 0 % |
| HELPER | 0 | 0 % |

The trained policy's deterministic plan is **all-MEC on every graph** (identical to the `all_MEC`
baseline). It uses no UE and no HELPER.

## 2. Latency (mean makespan, seconds)

| method | latency_s |
|---|---|
| greedy (coordinate descent + MC) | **0.0870** |
| HEFT v2 | 0.0906 |
| all-MEC | 0.0917 |
| **model k3 (adapted)** | **0.0917** |
| model k0 (zero-shot) | 0.0917 |
| oracle (min per graph) | 0.0868 |

`k3_regret_s = +0.0049`, `k0_regret_s = +0.0049`. `k0 == k3` exactly: the 3 inner adaptation
steps change nothing — the meta-policy already zero-shots to all-MEC.

## 3. Energy (system scope, joules)

| method | system_J |
|---|---|
| greedy | **86.08** |
| HEFT v2 | 90.41 |
| **model k3** | **91.56** |
| all-MEC | 91.61 |

Model energy ≈ all-MEC (its plan). `requester_joules`/`mobile_joules` ≈ 0.002 J because every
task offloads, so the vehicle itself does almost no local compute. Greedy is ~6 % cheaper.

## 4. Deadline / criticality compliance

| metric | value |
|---|---|
| graph hard-deadline violation rate | 0.000 |
| MC policy violation count (HIGH dropped in HI) | 0.0 |
| HIGH-tardiness incidence rate (graphs with ≥1 HIGH tardy) | 0.025 |
| MEDIUM-tardiness incidence rate | 0.000 |
| HIGH task tardiness rate (task-level) | 0.003 |
| mode-switch rate (graphs that switched LO→HI) | 0.988 |

The model is feasible on every constraint (hard deadline, MC preservation, MEDIUM tardiness) and
keeps HIGH-task tardiness near zero — but it achieves this by placing everything on MEC, which
trivially satisfies the all-MEC-anchored energy budget and never uses the helper, so the
mixed-criticality/helper machinery is never actually exercised by the policy.

## 5. Verdicts (kept separate)

* **Implementation acceptance: PASS.** F05/F04/F08/F11/F13/F25–F29/F34 closed with executed
  regressions; local 1577 passed / 42 skipped / 0 failed; TF1.15 container 106 tests across 10
  modules all `EXIT=0`; Gate A (CPU continuation, exact to 1.6e-16), Gate B (recovery/legacy
  refusal/paired R×S) and Gate C (3/3 production GPU iterations, observed 100 % GPU kernel
  placement) all passed on the repaired revision.
* **Campaign operational acceptance: PASS.** 1500/1500 completed outer iterations, matching
  outer-update counts, 30 validations, finite state, transactional checkpoint generations, zero
  tracebacks, detached monitor throughout, vLLM/qdrant untouched, meta-test closed.
* **Learning competitiveness: NOT ESTABLISHED (negative result).** The model collapsed to
  all-MEC, matches the all-MEC baseline (latency 0.0917 s, energy 91.56 J) and loses to the
  MC-aware greedy baseline on both latency (−5.4 %) and energy (−6.0 %); adaptation (k3) adds
  nothing over zero-shot (k0). One seed only. No selective offloading, no V2V/HELPER use was
  learned.

## 6. Why the collapse (root cause, already known from the v1 audit)

1. The total-energy budget anchor is the foreground **all-MEC** plan, so all-MEC has zero energy
   violation by construction — the cheapest feasible point is to copy all-MEC.
2. The geometry favours MEC (fast, cheap internal communication), and with the budget anchored
   on all-MEC there is no counter-force pushing the policy toward UE or HELPER.

A policy that only learns to copy the strongest trivial baseline is a genuine negative result for
the *selective offloading* claim; it is reported as such, not hidden.
