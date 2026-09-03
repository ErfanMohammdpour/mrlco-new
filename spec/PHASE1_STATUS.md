# Phase 1 — Production Scheduler

Status: IN PROGRESS

## Commits

- Commit 1 (`937e4c5`): canonical engine — DONE
- Commit 2A (`b49b60f`): adapter + env wrapper — DONE
- Commit 2B: greedy + remaining callers via `schedule()` — this change
- Later: Energy API freeze / reward telescoping (NOT this commit)

## Greedy baseline

- Metric: `makespan_seconds` (latency baseline)
- Unevaluated suffix fill: `all_UE`
- Tie-break: UE → MEC → HELPER
- Every candidate scored only by `schedule()`

## Still open for Phase 1 gate

- Reward still legacy (`get_reward_batch_step_by_step`)
- Energy bounds in reward still old all-local / all-MEC heuristic
- No Phase 1 complete tag
