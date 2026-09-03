# Phase 1 — Production Scheduler

Status: IN PROGRESS

## Commits

- Commit 1 (`937e4c5`): canonical engine — DONE
- Commit 2A (`b49b60f`): adapter + env wrapper — DONE
- Commit 2B (`5c3029e`): greedy + remaining callers via `schedule()` — DONE
- Energy API: pure-location refs + component/attribution API — this change
- Later: reward post-hoc telescoping (NOT this commit)

## Energy API (this commit)

- Primary scope: `total_mobile_joules` (UE + HELPER)
- `mec_compute_joules_optional` field present; v0.1 value = 0 (no frozen MEC power)
- Normalization: all_UE / all_MEC / all_HELPER schedules → `ReferenceRanges`
- Out-of-range: `clip_and_log`
- `J_report = 0.5 * L_norm + 0.5 * E_norm`
- Per-task attribution sums to `total_mobile_joules`
- Evaluator V2V uplink/downlink split from `TransferRecord` direction

## Still open for Phase 1 gate

- Reward still legacy (`get_reward_batch_step_by_step`) — need post-hoc telescoping + `all_UE`
- No Phase 1 complete tag
