# Phase 0 Status Report

Status: `IN PROGRESS`  
Specification ID: `MARGO-SPEC-v0.1`  
Date: `2026-09-03`

## 1. Executive status

Phase 0 started correctly, but it is NOT ready to hand off to Phase 1.

Current state:

- core semantic scaffold: `GOOD`
- inventory generation readiness: `YES`
- final manifest generation readiness: `YES (ADR-004 Accepted; --mode final)`
- behavior-changing simulator work readiness: `NO`

Reason:
Data/Split frozen. Learning *form* + validation selection grid frozen (`ADR-006`). Toy oracle suite exists. Phase 0 still open until selected LR/eps/`k_steps` evidence is recorded after validation-only search. Phase 1 simulator still blocked until that search is *not* confused with simulator repair: simulator repair may start against oracles, but hyperparameter *values* stay unpicked.

## 2. Closed decisions

The following are already frozen at semantic level:

- environment model = `macro_action_autoregressive`
- data residency = output stays at execution location
- 3x3 communication routing matrix exists
- resources are explicit single-capacity non-preemptive intervals
- `MEC_DL` is a real limited resource
- `V2V_CHANNEL` is shared half-duplex
- base units = seconds / joules / bytes / bytes_per_second
- primary energy scope = `total_mobile_joules`
- outer update shape = one mean displacement, order-invariant
- terminal result policy = sink outputs MUST become available at `UE`
- workload model = byte-based compute in v0.1
- task input model = root external input + dependency edge outputs

## 3. Still-open blockers

These are still active Phase 0 blockers:

| topic | status | note |
| --- | --- | --- |
| distribution split IDs | frozen | `ADR-004` Accepted; `spec/split_policy.json` |
| support/query counts | frozen | 20/80 from policy |
| split seed | frozen | `spec/split_policy.json` |
| distribution assignment rule | frozen | `latin_grid_holdout_v1` + `stratified_sha256_rank_v1` |
| dataset manifest | frozen | `spec/dataset_manifest.jsonl` + sidecar |
| manifest validator | implemented | full scan + policy-backed assignment re-check + mutation tests |
| toy oracle suite | in progress | 15 graphs + checker; hand proofs for 01/04/09 |
| inner learning rate | grid frozen | selected on validation only (`ADR-006`) |
| outer step size | grid frozen | Reptile interpolation eps, not legacy Adam |
| `k_steps` | grid frozen | means optimizer apply count |
| `ppo_batch_size_trajectories` | frozen | 20 |
| `meta_batch_size_distributions` | frozen | 5 |
| canonical graph hash | implemented | canonical_graph_sha256 computed from canonical graph object |

## 4. Corrected execution order

The correct Phase 0 order is:

1. finish semantic blockers
2. generate raw dataset inventory
3. generate draft distribution manifest
4. accept split policy and held-out IDs
5. change `ADR-004` from `Proposed` to `Accepted`
6. complete `frozen_experiment.yaml`
7. generate final `dataset_manifest.jsonl`
8. implement and run manifest validator
9. generate `dataset_manifest.sha256`
10. build toy oracle suite
11. sign off and freeze spec

This means final manifest creation MUST NOT happen before split acceptance.

## 5. Artifacts created so far

Created:

- `SYSTEM_SPEC.md`
- `OBJECTIVE_AND_ENERGY.md`
- `SCHEDULING_SEMANTICS.md`
- `LEARNING_PROTOCOL.md`
- `DATA_SPLIT.md`
- `frozen_experiment.yaml`
- `schemas/dataset-manifest.schema.json`
- `decisions/ADR-001-energy-scope.md`
- `decisions/ADR-002-data-residency.md`
- `decisions/ADR-003-resource-capacity.md`
- `decisions/ADR-004-evaluation-split.md`
- `decisions/ADR-005-edge-canonicalization.md`
- `dataset_inventory.jsonl`
- `distribution_manifest.json`
- `dataset_manifest.jsonl`
- `dataset_manifest.jsonl.sha256`
- `split_summary.json`
- `split_policy.json`
- `manifest_validator.py`

## 6. Inventory findings from actual dataset

Observed directly from repository dataset tree:

- distributions found: `25`
- graphs per distribution: `100`
- total graphs: `2500`
- node count in all inventory rows: `20`

Inventory artifacts remain inventory-only (no `role`). Final roles live in `dataset_manifest.jsonl`.

## 7. Spec hardening done in this revision

Compared with initial draft, this revision additionally freezes:

- task-input semantics
- sink-return semantics
- workload unit choice
- reporting weight choice `0.5 / 0.5`
- reward shaping formula family
- out-of-range normalization rule `clip_and_log`
- entropy claim disallowed for v0.1
- inventory-first split workflow
- stricter manifest schema with `additionalProperties: false`

## 8. What is still intentionally not frozen

These require explicit sign-off before Phase 1:

- exact held-out distribution IDs (frozen)
- deterministic split seed (frozen in `split_policy.json`)
- exact distribution-parameter coverage algorithm (frozen)
- final numeric optimizer budgets
- canonical graph hashing procedure (defined; scheduler/encoder consumption pending)
- executable validator behavior (implemented; `--mode final` required for release)
- toy example expected numeric outputs

## 9. Gate decision

Current gate:

- `ADR-001`: accepted
- `ADR-002`: accepted
- `ADR-003`: accepted
- `ADR-004`: accepted
- `ADR-005`: proposed
- `ADR-006`: accepted
- semantic scaffold: strong
- final Phase 0 gate: FAIL for now

Therefore:

- final manifest generation: done
- simulator repair: blocked (Phase 0 toy/learning numerics remain)
- encoder repair: blocked
- PPO/Reptile repair: blocked

## 10. Recommended next step

Immediate next step:

- treat toy oracles as Phase 1 acceptance tests; record selected LR/eps/`k_steps` only from frozen validation grid (`ADR-006`)

Concrete next artifacts:

- `distribution_manifest.json`
- accepted split summary
- final `dataset_manifest.jsonl`
- `manifest_validator.py`
- toy oracle files
