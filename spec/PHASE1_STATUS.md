# Phase 1 — Production Scheduler

Status: IN PROGRESS

## Commits

- Commit 1 (`937e4c5`): canonical engine — DONE
- Commit 2A (`b49b60f`): adapter + env wrapper — DONE
- Commit 2B (`5c3029e`): greedy + callers via `schedule()` — DONE
- Energy API (`7058b88`): pure-location refs + attribution — DONE
- Reward: post-hoc telescoping + normalize hardenings — this change

## Reward (this commit)

- Method: post-hoc telescoping, completion fill `all_UE`
- `r_t = -(0.5 * ΔL_t / L_scale + 0.5 * ΔE_t / E_scale)` (unclipped deltas)
- `sum(r_t) == -(0.5*(L_N-L_0)/L_scale + 0.5*(E_N-E_0)/E_scale)`
- `J_report` clipped scientific metric stays separate
- `normalize` fail-fast on NaN/Inf and inverted ranges
- `attribute_energy_components_by_task()` → `dict[int, EnergyBreakdown]`

## Still open for Phase 1 gate

- Phase 1 closure gate script / tag (not this commit)
- Clean-clone deps (`pydotplus` / `jsonschema`) — Phase 5
