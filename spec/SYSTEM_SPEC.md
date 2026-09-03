# MARGO System Specification

Status: Draft for Phase 0  
Specification ID: `MARGO-SPEC-v0.1`  
Scope: `MARGO_BASELINE/mrlco-new` only

## 1. Purpose

This document defines the intended scientific behavior of MARGO before code repair.
It is normative. When released code disagrees with this specification, code is wrong unless this specification is revised in a versioned decision record.

## 2. Scope

MARGO models DAG task offloading with three execution locations:

- `UE` (local execution)
- `MEC` (edge server execution)
- `HELPER` (single V2V helper)

The system optimizes a joint latency-energy objective for fixed-size DAG applications with 20 tasks.

## 3. Non-goals

This version of the specification does not model:

- multiple helpers
- stochastic fading during one episode
- MEC backhaul internals
- packet loss / retransmission
- preemptive scheduling
- partial task execution

## 4. Canonical execution model

MARGO v0.1 MUST be modeled as a `macro-action autoregressive planner`.

- Policy emits a length-`N` action sequence.
- `N` equals task count in decoder order.
- Environment schedules the full plan after sequence generation.
- Environment returns shaped rewards derived from the full schedule.
- Policy does NOT observe updated resource availability after each decoder token.

The paper MUST NOT describe this implementation as a true stepwise MDP with observed resource state after each task unless the environment interface is redesigned in a future spec version.

## 5. Entities

- `task`: DAG node with processing workload and output data attributes
- `edge`: directed dependency from predecessor task to successor task
- `location`: one of `UE`, `MEC`, `HELPER`
- `support set`: graphs used for adaptation
- `query set`: disjoint graphs used only for reporting
- `distribution`: graph family identified by `distribution_id`
- `makespan`: completion time of whole application in seconds

## 6. DAG schema

Each task MUST have:

- unique integer `task_id`
- `compute_workload_bytes`
- `task_output_bytes`
- `external_input_bytes`

Each edge MUST have:

- `src_task_id`
- `dst_task_id`
- `edge_output_bytes`

The dataset validator MUST reject:

- cyclic graphs
- duplicate task IDs
- missing required task/edge fields
- negative sizes or rates

## 7. Execution locations and data residency

Output data location MUST equal execution location:

- Local execution -> output stored at `UE`
- MEC execution -> output stored at `MEC`
- V2V execution -> output stored at `HELPER`

For every edge `i -> j`, task `j` MUST NOT start before predecessor output is available at `j`'s execution location.

## 8. Communication semantics

Communication cost MUST be computed from `edge_output_bytes`, not from task-global proxy sizes, unless a future spec version explicitly changes this rule.

All predecessor-successor location pairs MUST use explicit routes defined in `SCHEDULING_SEMANTICS.md`.

Task-input semantics for `MARGO-SPEC-v0.1`:

- root task input = `external_input_bytes` initially resident at `UE`
- non-root task input = predecessor edge outputs plus optional `external_input_bytes`
- `compute_workload_bytes` affects compute time only
- predecessor-successor transfer basis = `edge_output_bytes`

This specification MUST NOT double-count both a task-global network upload and the predecessor edge transfers for the same logical dependency payload.

## 9. Resource model summary

MARGO v0.1 uses single-capacity non-preemptive resources:

- `UE_CPU`
- `MEC_UL`
- `MEC_CPU`
- `MEC_DL`
- `HELPER_CPU`
- `V2V_CHANNEL`

All reservations are interval-based. A task MAY overlap compute on one resource with communication on another only when explicitly allowed by resource rules.

## 10. Units

Internal simulator units MUST be:

- time: `seconds`
- energy: `joules`
- power: `watts`
- data: `bytes`
- communication rate: `bytes_per_second`

Presentation MAY convert seconds to milliseconds, but raw logged values MUST remain in seconds.

## 11. Objective summary

Primary scientific metrics MUST be logged separately:

- `makespan_seconds`
- `total_mobile_joules`
- normalized latency score
- normalized energy score
- composite objective

For `MARGO-SPEC-v0.1` the reporting objective is:

`J_report = 0.5 * L_norm + 0.5 * E_norm`

Where:

- `L_norm` and `E_norm` use per-episode reference-range normalization
- reference range is computed from three pure-location plans: all-`UE`, all-`MEC`, all-`HELPER`
- out-of-range behavior = `clip_and_log`

Training reward for position `t` is:

`r_t = -(0.5 * delta_latency_t / L_scale + 0.5 * delta_energy_t / E_scale)`

Where:

- `delta_latency_t` is incremental makespan increase caused by token `t`
- `delta_energy_t` is incremental mobile-energy increase caused by token `t`
- `L_scale = max(L_ref_max - L_ref_min, epsilon)`
- `E_scale = max(E_ref_max - E_ref_min, epsilon)`
- `epsilon` MUST be a positive documented constant

Reward shaping and final reporting objective MUST both be logged and traceable.

## 12. Learning protocol summary

- PPO inner adaptation semantics are defined in `LEARNING_PROTOCOL.md`.
- Reptile-style outer update MUST use one order-invariant mean displacement from a common starting core parameter set.
- Optimizer-state lifecycle MUST be explicit and testable.
- `MARGO-SPEC-v0.1` freezes `ppo_inner_optimizer = adam`
- `entropy_coefficient = 0.0`
- `entropy claim in manuscript = forbidden unless spec version changes`

## 13. Terminal result semantics

Application completion requires sink outputs to be available at `UE`.

- sink tasks MAY finish at `MEC` or `HELPER`
- application makespan MUST include required transfer of each sink output to `UE`
- terminal return bytes for sink task = `task_output_bytes`

## 14. Dataset split summary

Split rules are defined in `DATA_SPLIT.md`.
Every graph file MUST appear exactly once in:

- `meta_train`
- `validation_support`
- `validation_query`
- `meta_test_support`
- `meta_test_query`

or be marked as `excluded` with a reason.

## 15. Invariants

The implementation MUST satisfy all of the following:

1. For every edge `i -> j`, required predecessor data arrival time at `j`'s location is `<= start_j`.
2. No single-capacity resource has overlapping reserved intervals.
3. Same-location dependencies incur zero communication delay unless explicitly overridden by a future spec.
4. Two graphs with identical node features but different edges MUST produce different encoder adjacency structures.
5. Reptile outer update MUST be invariant to meta-task ordering up to numerical tolerance.
6. Sink outputs are not counted complete until required terminal results are available at `UE`.

## 16. Traceability requirement

Every paper claim MUST map to:

- one specification clause
- one config field or schema field
- one implementation module
- one acceptance test
- one reported artifact

## 17. Phase 0 blocker checklist

Phase 1 MUST NOT start while any of the following remain unresolved in frozen config:

- split policy and held-out IDs
- support/query counts
- split seed
- dataset manifest
- toy oracle suite
- exact normalization constants source
- exact `epsilon` safeguard for reward scaling

## 18. Phase 0 exit rule

Phase 0 completes only when:

- semantic choices are versioned
- split is frozen
- dataset manifest is generated and validated
- toy oracle examples exist
- all critical terms in paper are aligned with this spec
