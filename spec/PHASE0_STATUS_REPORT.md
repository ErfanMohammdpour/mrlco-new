# Phase 0 Status Report

Status: `CLOSED` (tag `phase0-freeze-v0.1`)
Specification ID: `MARGO-SPEC-v0.1`
Date: `2026-09-03`

## 1. Executive status

```text
Phase 0 preflight checks:        PASS
Phase 0 closure gate:            PASS
Data/Split gate:                 PASS
ADR-004 / ADR-005 / ADR-006:     ACCEPTED
Outer update:                    mrlco_first_order_mean_pseudogradient
Hyperparameters:                 fixed_literature_derived_defaults (MR-LCO Table 2)
Optimization claim:              false
Phase 0 overall:                 CLOSED
Ready for Phase 1 behavior fix:  YES after tag phase0-freeze-v0.1
```

Run: `python3 spec/phase0_gate.py`

## 2. Closed decisions

- Data/Split frozen (`ADR-004`)
- Edge canonicalization semantics (`ADR-005`); sim/encoder adoption Phase 1/2
- Learning defaults from MR-LCO Table 2 without optimality claim (`ADR-006`)
- Outer update = mean first-order pseudogradient + outer Adam (`β=5e-4`), not Reptile
- Reward = post-hoc telescoping + `all_UE` fill
- Toy oracles hardened

## 3. Not copied from MR-LCO

Architecture (`2×256` LSTM etc.) remains MARGO-specific; freeze in Phase 2.

## 4. Next steps

1. DONE: tag `phase0-freeze-v0.1`
2. DONE: Phase 1 simulator / energy / reward (`db3b36a` + closeout)
3. Phase 2: encoder (DAG adjacency, pred/succ, canonical features)
4. Phase 3: trainer split, PPO clip, outer mean-pseudogradient, evaluation protocol
5. Multi-seed evaluation of the fixed config (evaluation, not tuning)

Phase 1 closure does not imply encoder/PPO/evaluation readiness.
