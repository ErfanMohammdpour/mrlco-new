# Phase 1 — Production Scheduler

Status: CLOSED

Technical closure commit: `db3b36a`  
Closeout (docs / gate hygiene / attribution completeness): this commit

**Phase 1 closure does not imply encoder/PPO/evaluation readiness.**

Encoder, split-wired trainer, PPO value clip, outer mean-pseudogradient, evaluation support/query, and GPU training remain Phase 2/3. Do not treat a Phase 1 tag as permission to run paper experiments.

## Commits

- Commit 1 (`937e4c5`): canonical engine — DONE
- Commit 2A (`b49b60f`): adapter + env wrapper — DONE
- Commit 2B (`5c3029e`): greedy + callers via `schedule()` — DONE
- Energy API (`7058b88`): pure-location refs + attribution — DONE
- Reward (`f2453b7`): post-hoc telescoping — DONE
- Closure (`db3b36a`): weight freeze + `phase1_gate.py` — DONE
- Closeout: precedence test, status enforcement, zero-energy keys — this change

## Frozen contracts

- Objective weights: exactly `0.5 / 0.5` (publication fail-fast)
- Reward: post-hoc telescoping, fill `all_UE`, unclipped deltas
- `J_report`: evaluation-only (`compute_j_report=True`); training path skips it
- P0 metrics reused from pure-location refs (no extra schedule)
- Per-task energy map includes every scheduled task (zeros allowed)

## Gate

```bash
python3 spec/phase1_gate.py
```

Must print `Phase 1 closure: PASS`. Tag `phase1-freeze-v0.1` only after that.
