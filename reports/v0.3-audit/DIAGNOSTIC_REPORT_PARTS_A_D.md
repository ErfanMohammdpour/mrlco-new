# Parts A–D diagnostic report (constraints, queue, upstream, ranking)

No training, pilot, deadline regeneration or runtime-shield run was executed.
All numbers are from the committed machine-readable evidence.

## Final SHA / environment

* verified code SHA: `7f5b63a7cb55c11d331001d2272e32b6b505f95e` (tree clean)
* clone `/opt/margo/mrlco-new-6b`, image `margo-phase4-tf115-nv2212:latest`
  (`sha256:82bc781b…`), TensorFlow 1.15.5, RTX 4090.
* `energy-tests`: **`Ran 803 tests … OK`**.
* gates from the clean SHA: `energy-tests` 0 (128 s), `energy-constraint-smoke`
  0 (1450 s), `energy-train-smoke` 0 (1440 s), all `failures = []`.
* evidence: `reports/v0.3-audit/{constraint_smoke,queue_audit,upstream_parity,rank_audit}/`.

## Part A — constraints-enabled trainer integration

`spec/mask_sanity --constraints-scenario {total_absent,total_big,total_small}` +
the `energy-constraint-smoke` target (latency_only reward, log_only objective,
obs v3, mask static, constraints measured, Lagrangian OFF with `dual_lr=0.0`).

Replay-free fixture (`spec/constraint_smoke_evidence.py`, one ScheduleResult, one
SYSTEM reference) passes all checks:
*A1* budget absent -> `not_configured`, controller off, no active constraint,
penalty 0, raw == system energy, reward identical to constraints-off;
*A2* big budget -> `active`, raw == system, under budget, violation 0;
*A3* small budget -> `active`, raw == system, signed/violation > 0, penalty 0;
`C_UE_ENERGY.raw == requester`; a mobile reference is rejected by both the
objective and the constraint; a zero-learning-rate controller keeps lambda at 0.

Wording/scope notes: the `1 J` and `1e12 J` budgets are **integration fixtures
only**, not scientific budgets. In `not_configured` (A1) the constraint `raw`
field is deliberately `null`; the check is therefore named
`A1_underlying_metric_equals_system` and compares the underlying system metric
(`measure_metrics.total_energy_j`) to the system telemetry scalar.

Kish trainer CSV (`total_small`) carries the full constraint column set
(`constraint_status/total_energy = active`, `constraint/total_energy_raw`,
`_budget`, `_signed`, `_violation`, `constraint/penalty_applied = 0`,
`constraint/lambda_total_energy = 0`, `constraint/updates = 0`) and the CSV
validator (which checks the exact untruncated columns) reported `failures = []`.
`energy-train-smoke` (constraints off) has no constraint columns, as designed.

Two real bugs fixed on the way: `ConstraintSpec.as_dict()` was not re-parseable
(`to_config_dict()` added), and with `parallel=True` the reward runs in worker env
copies so the trainer never observed the constraint costs — they now travel on
the same telemetry channel and are aggregated episode-weighted.

## Part B — queue / parallelism + counterfactual prefix audit

`spec/queue_audit.py`, 5 graphs × 5 plans. Resource totals (busy / uncontended
critical-path contribution / max queue / p95 wait, seconds):

| resource | busy | critical | max queue | p95 wait |
|---|---|---|---|---|
| UE_CPU | 24.03 | 18.17 | 2.670 | 2.434 |
| HELPER_CPU | 18.64 | 14.93 | 2.670 | 2.434 |
| MEC_CPU | 5.05 | 4.13 | 0.267 | 0.243 |
| MEC_UL | 3.31 | 0.00 | 0.458 | 0.431 |
| MEC_DL | 3.10 | 0.00 | 0.000 | 0.000 |
| V2V_CHANNEL | 2.55 | 0.00 | 0.000 | 0.000 |

Counterfactual (prefix fixed, BASE-PLAN suffix, documented contract):

| graph | all_MEC | two_opt_mixed | static prefers MEC but not actual | Spearman static↔actual | mean abs error |
|---|---|---|---|---|---|
| sparse | 0.635 | 0.635 | 0/8 | 1.000 | 0.000 |
| medium | 0.703 | 0.703 | 0/9 | 0.984 | 0.099 |
| dense | 0.635 | 0.635 | 0/8 | 0.984 | 0.078 |
| narrow_deep | 0.936 | 0.936 | 0/12 | 1.000 | 0.000 |
| wide_shallow | 0.776 | 1.070 | 1/10 | 0.535 | 0.157 |

No causal claim is made about the suffix: it is a counterfactual diagnostic.

## Part C — upstream parity (`linkpark/metarl-offloading`)

Binary Local/MEC, one graph, one topological order, one plan, identical rates,
latency semantics only. Upstream scheduler extracted by AST from commit
`a55094f`; a byte-identical copy is vendored at
`spec/fixtures/upstream_metarl/` (sha256 `41c9b90c…`) so the diagnostic runs
offline.

| scheduler | makespan |
|---|---|
| upstream_original | 36.771429 s |
| legacy_control | 36.771429 s |
| canonical | 21.100000 s |

`legacy_control ≡ upstream_original` bit-exactly (max per-step delta 0.0).
Canonical is 15.671 s faster; the difference decomposes into (all `applies=true`):
root-only external input (8.39 MB vs 32.51 MB charged upstream), MEC residency /
MEC→MEC zero transfer (5 edges), explicit dependency-edge transfer (6.29 MB),
sink-only return (1.05 MB vs 13.63 MB), independent UL/DL calendars
(12.57 s UL / 4.57 s DL vs one uplink counter and no downlink), contention and
4-vs-3 calendars. `decoder_order` is reported `applies=false`: a single
topological order cannot isolate it.

## Part D — decoder-order / ranking audit

10 graphs (5 frozen daggen + 5 synthetic), 5 orders, order and assignment kept
separate (one fixed per-task mapping replayed under every order; greedy searched
per order independently). Production order untouched.

| order | Kendall vs current | fixed latency (s) | greedy (s) | static err abs (s) |
|---|---|---|---|---|
| legacy_current | 1.0000 | 444.119 | 246.156 | 65.327 |
| stable_topo | 0.4952 | 461.218 | 298.463 | 82.426 |
| heft_upward | 0.9875 | 444.131 | 245.348 | 65.338 |
| canonical_aware | 0.6532 | 474.743 | 397.398 | 95.951 |
| deadline_criticality | 0.6972 | **443.897** | **245.325** | **65.105** |

The current legacy rank is the **best tested heuristic on this diagnostic set** —
indistinguishable from HEFT-upward under a fixed assignment (Kendall 0.988); this
is a statement about the tested set, not a global optimality claim. stable_topo
is +3.8 % and a naive scheduler-aware rank +6.9 %.

## Answers to the six questions

1. **Real bottleneck resource.** A mobile-side calendar, not MEC: `UE_CPU` has
   the largest uncontended critical contribution (18.17 s) and p95 wait (2.43 s),
   with `HELPER_CPU` second (14.93 s); `MEC_CPU` contributes 4.13 s and the
   network calendars never appear on the uncontended critical path. On a frozen
   daggen sample the bound resource was `V2V_CHANNEL` (utilisation 0.914 vs
   `MEC_CPU` 0.045), so the bound calendar is graph-dependent but always
   mobile-side.
2. **Why all-MEC looks worse than mixed.** It does not in this diagnostic:
   all-MEC equals the best mixed plan on 4/5 graphs and beats the two-opt mixed
   plan on `wide_shallow` (0.776 vs 1.070). Where a mixed plan loses, the cause
   is the myopic assignment/local search and the queue-blind static bound
   (1/10 positions on `wide_shallow`), not MEC per se.
3. **How often the static bound mis-ranks because of queue.** 1 of 10 positions
   on `wide_shallow` (static prefers MEC, actual does not); 0/8–0/12 on the other
   four graphs. Static↔actual Spearman is 0.535 on `wide_shallow` and
   0.984–1.000 elsewhere.
4. **Where the upstream/canonical difference comes from.** The pre-canonical
   project path is bit-identical to upstream under binary+latency; canonical's
   −15.67 s comes from root-only external input, MEC→MEC zero transfer,
   explicit dependency-edge transfer, sink-only return, separate UL/DL calendars
   (4 vs 3 calendars), and the resulting contention/parallelism. `decoder_order`
   was not isolated by this contract.
5. **Is the legacy ranking meaningfully inconsistent with the canonical
   scheduler?** No. Under a fixed assignment the legacy order is the best tested
   heuristic on the diagnostic set — statistically tied with
   deadline/criticality-aware and HEFT (444.119 vs 443.897 vs 444.131 s,
   <0.06 %) and better than stable-topological (+3.8 %) and a naive
   scheduler-aware order (+6.9 %). No production order change is warranted.
6. **Scheduler/ranking fix or exploration?** The evidence does not support a
   scheduler/ranking fix as the next lever: the legacy order is the best tested
   heuristic on this diagnostic set (indistinguishable from HEFT) and the
   queue-blind static bound only mis-ranks on wide/shallow graphs. The
   open gap is policy-side: a queue-aware assignment/exploration that does not
   trust the static bound, to be tested after review. Keep the production order
   and the scheduler as they are for now.

## Deliverable commits

`b326ab8` Part A code, `e41b498` constraint telemetry channel, `acbbeee` Part B,
`cef6772` Part C, `8b51a57` Part D, `7f5b63a` offline upstream fixture, plus this
evidence commit. `spec/upstream_parity.py` and `spec/decoder_order_audit.py` were
briefly captured in `acbbeee` by a concurrent `git add -A`; their final revisions
land in `cef6772` / `8b51a57` / `bb77453`.
