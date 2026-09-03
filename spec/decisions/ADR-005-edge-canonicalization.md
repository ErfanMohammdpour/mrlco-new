# ADR-005: Edge Canonicalization for Dataset Semantics

Status: Proposed  
Decision date: 2026-09-03

## Proposed decision

Duplicate edge records in the raw `.gv` graph files MUST be canonicalized before any topology-derived statistics are computed (degrees, indegree/outdegree maxima, canonical graph hash, `is_dag`, and any encoder input remapping).

Canonicalization rules:

1. Parse every raw directed edge record as a triple: `(src_task_id, dst_task_id, edge_output_bytes)` where `edge_output_bytes` comes from the raw `size` attribute on the edge.
2. Collapse exact duplicates by canonical key: `(src_task_id, dst_task_id, edge_output_bytes)`.
3. For any fixed `(src_task_id, dst_task_id)`:
   - if multiple distinct `edge_output_bytes` values exist across duplicates, validation MUST FAIL (conflicting duplicates are forbidden).
4. All topology semantics MUST use the canonical edge set from steps (1-3):
   - `edge_record_count` counts raw edge records (including exact duplicates)
   - `unique_edge_count` counts canonical edges after collapsing exact duplicates
   - degrees (`max_indegree_unique`, `max_outdegree_unique`) are computed on canonical unique edges
   - `is_dag` is computed on canonical unique edges via independent cycle detection
5. `canonical_graph_sha256` MUST be computed from the canonical graph object:
   - nodes ordered by `task_id`
   - edges ordered by `(src_task_id, dst_task_id, edge_output_bytes)`
   - stable JSON serialization (sorted keys; stable separators)

## Why

Raw `.gv` files contain exact duplicate edge records in a subset of graphs. If degrees and derived topology stats are computed on raw edge records, then `distribution_manifest.draft.json` and downstream split/reward/encoder semantics can silently drift from the intended graph structure.

## Phase 0 gate impact

Phase 0 MUST NOT treat `distribution_manifest.draft.json` / final manifest semantics as final until:

- canonicalization policy is implemented in the manifest generator/validator
- scheduling/encoder inputs consume the canonical edge set (not raw records)

This ADR remains `Proposed` until validator + canonical consumption are fully integrated into the end-to-end Phase 0 pipeline.

