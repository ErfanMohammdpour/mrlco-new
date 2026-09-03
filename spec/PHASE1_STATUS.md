# Phase 1 — Production Scheduler

Status: IN PROGRESS  
Commit 1: DONE locally (canonical engine + calendars + routes)

## Package

```text
env/mec_offloaing_envs/scheduler/
  model.py / calendar.py / routes.py / resources.py / engine.py
```

Production engine MUST NOT import `spec/toy_oracles`.
Oracle checker MAY consume engine via `--engine production`.

## Next

- Commit 2A: adapter + `get_scheduling_cost_step_by_step` wrapper
- Commit 2B: migrate remaining callers + greedy via engine
- No reward/PPO changes yet
