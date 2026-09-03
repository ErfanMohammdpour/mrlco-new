# ADR-004: Evaluation Split Policy

Status: Accepted  
Decision date: 2026-09-03

## Decision

Freeze all 25 tracked distributions into explicit roles.
No silent omission.

Held-out evaluation MUST use:

- disjoint support graphs
- disjoint query graphs
- distributions excluded from meta-train

Normative input: `spec/split_policy.json`.
Derived output: `spec/split_summary.json`.

For v0.1, distribution IDs are fixed by `latin_grid_holdout_v1`:

- `validation`: `{2, 6, 10, 16, 17}`
- `meta-test`: `{7, 12, 14, 20, 23}`
- `meta-train`: remaining 15 distributions

Within each held-out distribution, support/query assignment MUST be stratified by `generator_ccr ∈ {0.3,0.4,0.5}` using `stratified_sha256_rank_v1` (exactly 20 support graphs and 80 query graphs per held-out distribution).

### Frozen rule (v0.1)

```yaml
distribution_rule:
  name: latin_grid_holdout_v1
  fat_index_map: {0.4: 0, 0.5: 1, 0.6: 2, 0.7: 3, 0.8: 4}
  density_index_map: {0.4: 0, 0.5: 1, 0.6: 2, 0.7: 3, 0.8: 4}
  validation_condition: fat_index == density_index
  meta_test_condition: fat_index == (density_index + 2) mod 5

graph_rule:
  name: stratified_sha256_rank_v1
  support_count: 20
  query_count: 80
  stratum: generator_ccr
  strata_allowed: [0.3, 0.4, 0.5]
  quota_method: proportional_largest_remainder
  quota_tie_break: ascending_ccr
  assignment_sort: ascending assignment_hash
  assignment_hash:
    formula: SHA256(split_version + "\0" + split_seed + "\0" + distribution_id + "\0" + relative_path + "\0" + raw_sha256)

split_seed:
  value: 7ccf0bc4773713d76be07a79f5c82857009fec7d624f84fdce5e75ee93ea2a5d
  derivation: SHA256("MARGO-SPLIT-v1|53fe08d0d6861e43078b2785263afee4b1bc972f")

split_sets:
  validation: [2, 6, 10, 16, 17]
  meta_test: [7, 12, 14, 20, 23]
  meta_train: [1, 3, 4, 5, 8, 9, 11, 13, 15, 18, 19, 21, 22, 24, 25]
```

### Hashing contract (`raw_sha256`)

Field name stays `raw_sha256` for `MARGO-DATA-v1` schema compatibility.

For `MARGO-DATA-v1`, `raw_sha256` is defined as SHA-256 over source-file bytes after deterministic EOL normalization:

1. CRLF (`0D 0A`) → LF (`0A`)
2. remaining CR (`0D`) → LF (`0A`)
3. no other byte transformation is permitted

`assignment_hash` MUST consume this same `raw_sha256`.
`.gitattributes` is a checkout aid and does not replace this definition.

Edge canonicalization policy: `ADR-005`.

## Why

Current paper/trainer/evaluator/exact-solver use conflicting splits, so scientific claims are not auditable.

## Acceptance evidence

```yaml
acceptance_evidence:
  split_policy: spec/split_policy.json
  dataset_manifest: spec/dataset_manifest.jsonl
  draft_manifest_sha256: e336074f2b149533a45fffaf068c4315aaf9e8de8d6843f1bf278ddc20b5cc26
  final_manifest_sha256: e336074f2b149533a45fffaf068c4315aaf9e8de8d6843f1bf278ddc20b5cc26
  validator_result: OK manifest validation passed
  mutation_result: ALL mutation tests OK
  dataset_graph_count: 2500
  distribution_count: 25
```

## Phase 0 gate

Data/Split decision is frozen.
Learning hyperparameters are fixed literature-derived defaults (`ADR-006`); no validation grid.
Phase 0 closure is determined by `spec/phase0_gate.py`.
