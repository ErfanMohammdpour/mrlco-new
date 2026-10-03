# V2_BUG_AUDIT — audit baseline `d0af6f1` (HEAD), v1 freeze `92212d1d`

Statuses: FIXED (fix + regression test committed) | PARTIAL | NOT FIXED | NOT ATTEMPTED.
Everything below was checked against the working tree at `d0af6f1`; `git diff 92212d1d -- env/mec_offloaing_envs/scheduler/engine.py spec/automotive_training/automotive_env.py` is EMPTY, i.e. frozen v1 execution is untouched.

| # | Finding | Status | Evidence / location |
|---|---|---|---|
| 3.1 | training used one DAG per slot (`compute_spec([graph])`, `schedule_shared([dag])`) while the geometry gate used background DAGs => training and gate solve different problems | **NOT FIXED in training** | `v2/env.py::_slot_specs` still builds ONE dag per slot; the gate (`v2/geometry_gate.py`) adds background. No shared world builder exists. |
| 3.2 | one builder/scheduler resolver for training+validation+baselines+gate | **NOT ATTEMPTED** | three separate paths today (env, gate, parity script) |
| 3.3 | clone/snapshot prefix evaluation must not mutate the world | PARTIAL | prefix schedules are independent `schedule_shared` calls (no shared mutable world), but there is no world object at all (see 3.1) |
| 3.4 | mandatory two-owner integration test (background changes foreground queues/waits/observable load) | PARTIAL | `test_v2_shared_scheduler.py::test_two_concurrent_dags_queue_on_shared_mec` and `test_v2_env.py` prove shared-calendar queueing and per-slot isolation, but there is no multi-owner world with observably changed load |
| 4.1 | transfer rate evaluated at ready time; mid-transfer rate/outage change ignored | **NOT FIXED** | `v2/shared_scheduler.py::reserve_transfer` samples the realized rate once at the (outage-adjusted) start and holds it constant; no service-integral (B/∫R dt) |
| 4.2 | `queue_wait_ul_s` constant zero; TX stats from nominal predecessor sizes | **NOT FIXED** | `TaskTiming.queue_wait_ul` is populated only for the root ingress hop; no per-task UL queue accounting |
| 4.3 | contact margin `need/window` makes a LONGER window reduce predicted success | **NOT FIXED (and extended by me)** | `v2/env.py::_frozen_telemetry`/`_schedule_slot` evidence uses `min(1, need/window)`; this is the audited monotonicity defect |
| 4.4 | recovery transfers HELPER→UE after contact ended | FIXED | `shared_scheduler` books the return transfer at `local_ready = contact_end`; regression `test_v2_helper_model.py::test_realized_contact_shorter_than_predicted_fails_and_restarts` |
| 4.5 | `locals().get` for failure flags / restart penalty leaks state between tasks | **PARTIAL** | per-task init added (task_outage_wait/events, tx_dl, fallback_*), but `contact_failure`/`restart_penalty` are still read via `locals().get(...)` in the timing record |
| 4.6 | helper completion check ignores whether the RESULT can return before contact ends | **NOT FIXED** | only CPU completion is compared with `contact_end_s` |
| 4.7 | predicted contact = 1.15 x realized is not an uncertainty model | **NOT FIXED** | `v2/helper_model.py::predicted_contact(bias=1.15)`; labelled explicit assumption but not a decision-time predictor |
| 4.8 | free recovery of remaining work is invalid without a checkpoint | **PARTIAL** | restart is used (no free recovery); checkpoint transfer cost / wasted work are NOT recorded |
| 5.1 | reliability evidence used mean `_outage` over the whole FUTURE horizon | **NOT FIXED — future-blindness violation retained** | `v2/env.py::_schedule_slot` builds `outage_fraction` as `np.mean(link_process._outage[...])` over the full horizon, i.e. hidden future truth enters admission |
| 5.2 | confidence compares estimate against hidden realized truth | **NOT FIXED** | `link_model.confidence()` = `1 - |est-real|/max(...)`, i.e. it reads realized truth; must come from past residuals |
| 5.3 | HIGH/MED epsilon grounded in 3GPP communication reliability mapped onto a whole computational task | PARTIAL | `v2/reliability_classes.yaml` cites R.5.3-006/R.5.3-001 with evidence sha and states the communication scope, but no calibration report exists and no end-to-end event mapping is proven |
| 5.4 | deadline semantics (hard/firm miss/failure/rejection) and no denominator hiding | **NOT FIXED** | v2 telemetry has miss counts but no admission/rejection ledger; rewritten tokens are not counted separately |
| 5.5 | no empirical 1e-5 claim from small samples | RESPECTED | no empirical reliability claim is made anywhere in the v2 reports |
| 6.1 | derive schema/version from feature names, no hardcoded 91/52 | PARTIAL | dims are derived (`PACKED_DIM = FEATURE_DIM + 2*MAX_NEIGH + 1`) but `v2/observation.py` hardcodes `V1_FEATURE_DIM=40`, `V1_PACKED_DIM=79` as contract constants |
| 6.2 | one shared feature/mask contract for encoder/policy/sampler/PPO/validation/restore | PARTIAL | the packer and `input_dim` agree (91) and a TF container test proves the encoder accepts it; no single contract module is shared by PPO/validation/restore |
| 6.3 | observation content (workload, occupancy/queue, rate uncertainty + age, predicted contact, energy/budget, per-node epsilon, lambda conditioning) | **NOT ATTEMPTED** | the 12-column context covers a subset; no age-of-estimate, no per-node epsilon, no budget/lambda channels |
| 6.4 | `_slot_of_graph` must use explicit identity fields (world/slot/dag/owner/dataset graph id) | **NOT ATTEMPTED** | `v2/env.py::_slot_of_graph` matches by object identity in `graph_objects` and returns None otherwise |
| 6.5 | normalization fitted on training data only | RESPECTED | the v2 stats artifact is DERIVED from the frozen v1 stats (no refit on validation/meta-test) |
| 6.6 | perturb-a-feature end-to-end test incl. gradients | **NOT ATTEMPTED** | only shape/schema tests exist |
| 7.1 | v2 energy ledger (replace zero joules) | **NOT ATTEMPTED — explicitly labelled** | `v2/env.py::_frozen_telemetry` emits exact zeros with `energy_constraint="not_configured"` |
| 7.2 | primary-literature review table (2024-2026 sources) | **NOT ATTEMPTED** | `ENERGY_LITERATURE_REVIEW.md` does not exist |
| 8.1 | episode penalty must reach rewards + lambdas (audited defect: `env.step` used zero penalty) | **NOT FIXED** | `v2/env.py::step` computes `penalty = 0.0` and never applies a lambda term; no dual path |
| 8.2 | telemetry top-level violations + task denominators; fail fast on missing metrics | PARTIAL | frozen energy schema now validates/fails fast (regression: telemetry aggregation error), but violations/denominators are absent |
| 8.3 | signed-cost dual updates; broadcast before the iteration | **NOT ATTEMPTED** | no v2 dual path |
| 9.1 | PPO ratio on stored actions/teacher forcing, masks from observables, GAE/padding audit | **NOT ATTEMPTED** | no v2-specific PPO audit performed |
| 9.2 | MRLCO outer update demonstrated via parameter displacement; per-task reset | PARTIAL (v1 evidence only) | v1 session evidence (Gate F) exists for the frozen algorithm; not re-verified for v2 |
| 9.3 | checkpoint contents incl. duals/schema/protocol/RNG; feasibility-aware selection | **NOT ATTEMPTED** | not inspected for v2 |
| 10.1 | CRN + R/S axes actually executed in the evaluator; no best-of-S | PARTIAL | protocol + stateless TF sampling proven in the container (R/S not yet wired into an executed evaluation loop) |
| 10.2 | V2HeldOutEvaluator must not fall back to v1 | FIXED (env path) | `env_factory` hook routes all three construction sites to `V2AutomotiveEnv`; regression `test_v2_stack.py::TestEvaluatorEnvFactory` |
| 10.3 | baselines/HEFT/greedy must use the same v2 scheduler | PARTIAL | the gate and parity study score candidates on the v2 scheduler, but the frozen HEFT plan source is still the v1 reference |
| 10.4 | exogenous randomness keyed by stable identities, not RNG call order | PARTIAL | CRN seed pairs are identity-keyed (`v2/crn.py`); the v2 link/helper processes are still seeded by episode counters |
| 11 | geometry gate + parity regenerated on the final HEAD | **NOT DONE — existing artifacts are STALE** | `GEOMETRY_GATE.json` predates the MC-aware sink fix and the v2 env changes |
| 12 | full end-to-end smoke (rollout -> inner PPO -> query/eval -> outer -> checkpoint save/restore) | PARTIAL | the CPU smoke reached rollout+PPO+outer and failed at telemetry (now fixed); checkpoint save/restore NOT demonstrated |
| 12b | throughput benchmark over several complete iterations | PARTIAL | one measured rollout: build 31.2 s, env.reset 2.14 s, 20-slot step 1.473 s, support rollout 91.7 s (CPU container, no GPU); inner/outer update times NOT measured |
| 13 | required report artifacts | PARTIAL | present: `V2_STAGE_LOG.md`, `GEOMETRY_GATE.{json,md}`, `V1_V2_PARITY.json`, `V1_V2_TRANSFER_ACCOUNTING.json`, `V1_OBS_GOLDEN.json`, `CURRENT_STATE.md`, `V1_ERRATA.md`, `V1_RATE_PROVENANCE.*`; missing: `V2_FLOW_CONTRACT.md`, `ENERGY_LITERATURE_REVIEW.md`, `V2_ENERGY_SPEC.md`, `OBS_SCHEMA_V2.json`, `V2_END_TO_END_SMOKE.md`, `V2_THROUGHPUT.json`, `GEOMETRY_GATE_V2.*`, `FINAL_DIAGNOSIS.md` |

## Findings reproduced in this session and fixed with regression tests

1. FIFO-by-arrival shared calendars serialised every helper DAG through one radio channel -> earliest-fit reservation (`test_v2_shared_scheduler.py::test_later_long_reservation_does_not_block_earlier_short_transfer`).
2. `remaining` shadowed by a float in the contact-failure branch -> crash (`test_v2_helper_model.py`).
3. task criticality case (`medium` vs `MEDIUM`) -> reliability gate crash.
4. data-arrival invariant used base rates -> false violation under time-varying links; now checked against the booked ready time.
5. helper contact failure restarted locally without returning the data -> explicit HELPER->UE return transfer.
6. geometry-gate MEC-load axis was a no-op (background DAGs built, never scheduled) -> background scheduled in the same batch; the pre-fix PASS verdict was declared INVALID.
7. MC-aware sink semantics: a task whose successors were dropped by the MC realization must return its output -> v1/v2 transfer accounting now matches exactly (2/2, 10/10, 22/22 records and bytes).
8. v2 training-layout observation mapping (`graph_indices` None/scalar) -> identity-safe per-row context mapping + explicit `set_task` guard.
9. import-time global obs-version mutation in a TF test poisoned 14 v1 encoder tests -> gated on `HAS_TF`.
10. `input_dim` disagreed with the emitted observation width (79 vs 91) -> pinned to what the env emits.

## Retracted claims (kept for the record)

* "counts differ because v1 records per EDGE and v2 per HOP" - WRONG; the difference was the missing MC-dropped-successor return.
* "~8400 schedules per iteration => weeks" throughput BLOCK - WRONG; measured ~0.4-3.5 ms per schedule and 92 s per support rollout on CPU.

## OPEN FAILING TEST (must be resolved before any gate claim)

After the event-based transfer change (`515789a`) the full non-TF suite is **1268 passed /
1 failed / 19 skipped**, and the earlier "suite green" statement in that commit's message is
**incorrect for this test**:

```
env/mec_offloaing_envs/scheduler/tests/test_v2_helper_model.py::
  TestHelperInScheduler::test_idle_helper_with_long_contact_beats_mec_when_mec_is_overloaded
  AssertionError: 8.2869 not less than 7.300049999999998
  (12 concurrent 6 MB DAGs; helper CPU 4 MB/s idle with unlimited contact vs one shared
   10 MB/s MEC server)
```

Status: **OPEN / unresolved**. The fixture asserts a helper win under MEC overload; with the
service-integral transfer path the helper batch now takes 8.2869 s against the MEC's 7.30 s.
Either (a) the shared V2V serialisation genuinely dominates at this size, in which case the
fixture's premise (helper CPU advantage wins) is wrong and must be re-derived against the
scheduler rather than relaxed, or (b) the booking fixed point introduced a real regression.
Not diagnosed yet; no threshold was changed. This must be settled before any geometry/energy
gate is regenerated, because helper usefulness is one of the gate criteria.
