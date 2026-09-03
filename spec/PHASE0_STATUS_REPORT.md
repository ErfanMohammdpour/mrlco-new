# Phase 0 Status Report

Status: `IN PROGRESS`  
Specification ID: `MARGO-SPEC-v0.1`  
Date: `2026-09-03`

## 1. Executive status

Phase 0 is **not** fully closed. Data/Split gate is PASS. Behavior-changing Phase 1 claims still blocked until gate script stays green and selected LR/eps/`k_steps` evidence is recorded.

Current gate table:

```text
Phase 0 preflight checks:        PASS
Phase 0 closure gate:            BLOCKED_PENDING_SELECTION_EVIDENCE
Data/Split gate:                 PASS
ADR-004:                         ACCEPTED
ADR-005:                         ACCEPTED (semantics now; sim/encoder adoption Phase 1/2)
Learning protocol form:          PASS
Reward attribution:              DEFINED (post-hoc telescoping + all_UE fill)
Numeric selection evidence:      MISSING
Toy oracle existence:            PASS
Toy oracle strength:             HARDENED (required expected fields)
Phase 0 overall:                 IN PROGRESS
Ready for Phase 1 behavior fix:  NO
```

Run: `python3 spec/phase0_gate.py`

- preflight PASS ≠ Phase 0 closed
- closure checks `hyperparameter_selection_evidence.json` + numeric `learning.*` in `frozen_experiment.yaml`

## 2. Closed decisions

- environment model = `macro_action_autoregressive`
- reward attribution = post-hoc telescoping with completion policy `all_UE`
- data residency = output stays at execution location
- 3x3 communication routing matrix
- single-capacity non-preemptive resources; `MEC_DL` real; `V2V_CHANNEL` half-duplex
- units = seconds / joules / bytes / bytes_per_second
- primary energy = `total_mobile_joules`
- Reptile outer = one mean displacement, order-invariant
- terminal sink results MUST return to `UE`
- workload = byte-based compute
- edge canonicalization semantics = `ADR-005` Accepted
- split = `ADR-004` Accepted + `split_policy.json` normative

## 3. Still-open blockers

| topic | status | note |
| --- | --- | --- |
| selected `inner_learning_rate` / `outer_step_size` / `k_steps` | pending evidence | grid + protocol frozen in ADR-006 |
| Phase 1 simulator vs oracles | not started | oracles are acceptance tests |
| encoder adjacency on canonical edges | Phase 2 criterion | ADR-005 adoption |

## 4. Artifacts

- specs + ADRs 001–006
- `dataset_manifest.jsonl` + `dataset_manifest.jsonl.sha256`
- `split_policy.json` / `split_summary.json`
- `manifest_validator.py` + mutation harness (`--mode final` on base)
- `toy_oracles/*.yaml` + hardened `oracle_checker.py`
- `phase0_gate.py`

## 5. Recommended next step

1. Keep `phase0_gate.py` green on clean tree.
2. Run validation-only grid search; write winning triple + seed metrics artifact.
3. Tag `phase0-freeze-v0.1` only after evidence artifact exists.
4. Then Phase 1 simulator repair against oracles.
