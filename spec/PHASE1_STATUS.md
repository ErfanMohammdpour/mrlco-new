# Phase 1 — Production Scheduler

Status: CLOSED

## Commits

- Commit 1 (`937e4c5`): canonical engine — DONE
- Commit 2A (`b49b60f`): adapter + env wrapper — DONE
- Commit 2B (`5c3029e`): greedy + callers via `schedule()` — DONE
- Energy API (`7058b88`): pure-location refs + attribution — DONE
- Reward (`f2453b7`): post-hoc telescoping — DONE
- Closure: weight freeze + gate — this change

## Frozen contracts

- Objective weights: exactly `0.5 / 0.5` (publication fail-fast)
- Reward: post-hoc telescoping, fill `all_UE`, unclipped deltas
- `J_report`: evaluation-only (`compute_j_report=True`); training path skips it
- P0 metrics reused from pure-location refs (no extra schedule)

## Gate

```bash
python3 spec/phase1_gate.py
```

Must print `Phase 1 closure: PASS`. Tag only after that (separate step).
