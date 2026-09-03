# Phase 1 — Production Scheduler

Status: IN PROGRESS

## Commits

- Commit 1 (`937e4c5`): canonical engine — DONE / pushed
- Commit 2A: adapter + `get_scheduling_cost_step_by_step` wrapper — this change
- Commit 2B: remaining callers + greedy via engine — pending

## Rules

- Production engine MUST NOT import `spec/toy_oracles`
- Oracle checker MAY use `--engine production`
- No reward/PPO changes in Phase 1 Commit 2
