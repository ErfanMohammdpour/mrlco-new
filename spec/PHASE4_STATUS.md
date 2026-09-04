# Phase 4 — Evaluation campaign

Status: IN PROGRESS

Parent freeze: `phase3-freeze-v0.1` (`0c776924b49da6c66c511c12a8cde70be732e25d`)
Branch: `phase4-eval`

**Do not move or rewrite `phase0-freeze-v0.1`, `phase1-freeze-v0.1`, `phase2-freeze-v0.1`, or `phase3-freeze-v0.1`.**

Phase 4 is evaluation of the frozen v0.1 config. It is not hyperparameter search.
Phase 4 closure does not exist yet. No paper figures until raw run artifacts exist.

## GPU

Default: GPU forbidden.

Train 3500 outer iterations only after **all** of:

1. explicit human approval in chat
2. CLI `--i-allow-gpu`
3. env `MARGO_ALLOW_GPU=1`

Do not start GPU from the agent without that chat approval.

## Frozen primary campaign

- method: `margo_v0.1_primary`
- seeds: `0 1 2 3 4`
- outer_iterations: 3500
- report `k_steps=0` and `k_steps=3` on every meta-test dist `{7,12,14,20,23}`
- validation `{2,6,10,16,17}` only for checkpoint selection
- objective `0.5/0.5`, V2V on, 20-task DAGs

## Gate

```bash
python3 spec/phase4_gate.py
```

Must print `Phase 4 campaign: PASS` and `Phase 4 closure: NOT CLAIMED`.
