# Phase 2 — Graph Encoder

Status: IN PROGRESS

Branch: `phase2-graph-encoder` from tag `phase1-freeze-v0.1` (`b611f7047b1c28e3aa6f0026683e87a3ae249b34`).

**Do not move or rewrite `phase1-freeze-v0.1`.**

Phase 2 does **not** imply PPO correctness, outer meta-update, trainer split compliance, GPU training, or paper figures.

Until Phase 2 **and** Phase 3 close: no scientific GPU training and no new paper figures. Unit tests and CPU smoke tests are allowed.

## Scope

In:

- canonical DAG adjacency (not clique)
- full task-id → HEFT/decoder-index remap
- no 6-neighbor truncation; `MAX_NEIGH = MAX_TASKS - 1` (19). Degree overflow fail-fast.
- `edge_output_bytes` in node payload features
- drop legacy scheduler time features from the observation
- real neighbor padding + node mask
- z-score from `spec/encoder_feature_stats.json` fit on `role=meta_train` only; file pins SHA-256 of `dataset_manifest.jsonl` and `split_policy.json` plus feature name order
- topology-sensitivity and permutation-consistency tests
- degree>6, duplicate-edge, non-contiguous task-id tests

Out:

- graph encoder as a paper-ready claim until this status is CLOSED
- PPO / value clip / outer mean-pseudogradient
- trainer split wiring
- evaluation support/query
- GPU training / new figures

## Notes

- Decoder **order** still comes from HEFT `prioritize_tasks` (legacy times). That is ranking, not encoder features.
- Packed observation dim = `FEATURE_DIM + 2*MAX_NEIGH + 1` (`PACKED_DIM`). Neighbor capacity comes from spec `task_count`, not meta-train degree stats.
- `MeanAggregator` uses masked mean over `neigh_len`; adapter dropout is passed through.
- Phase 1 gate discovers only `test_phase1*.py` so encoder tests do not enter the frozen Phase 1 suite.

## Gate

```bash
python3 spec/phase2_gate.py
```

Must print `Phase 2 encoder: PASS` for the encoder contract. This is **not** Phase 2 closure.
