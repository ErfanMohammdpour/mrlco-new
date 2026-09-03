# Data Split Contract

Status: Frozen for Data/Split (`ADR-004` Accepted)

## 1. Role set

Every graph file MUST be assigned exactly one of:

- `meta_train`
- `validation_support`
- `validation_query`
- `meta_test_support`
- `meta_test_query`
- `excluded`

## 2. Separation rules

- no graph hash may appear in more than one role
- distribution roles MUST be disjoint at distribution level:
  - `D_train ∩ D_validation = ∅`
  - `D_train ∩ D_test = ∅`
  - `D_validation ∩ D_test = ∅`
- `meta_test_*` distributions MUST be disjoint from `meta_train`
- support and query hashes within the same held-out distribution MUST be disjoint

## 3. Current repository inconsistency

Current code, paper, exact-solver, and released artifacts disagree about which distributions belong to train and test.
This specification freezes the rule before any rerun.

## 4. Split version requirements

The split generator MUST log:

- `split_version`
- `split_seed` = `7ccf0bc4773713d76be07a79f5c82857009fec7d624f84fdce5e75ee93ea2a5d`
- distribution selection rule: `latin_grid_holdout_v1`
- support selection rule: `stratified_sha256_rank_v1`
- held-out distribution IDs
- support count per held-out distribution (MUST be 20)
- query count per held-out distribution (MUST be 80)
- exclusion reasons (if `role = excluded`)

## 5. Inventory-first workflow

Phase 0 MUST produce two artifacts in sequence:

1. `dataset_inventory.jsonl`
   - no final `role`
   - may use `inventory_role = unassigned`
2. `dataset_manifest.jsonl` (final frozen split roles)

Manifest generation MUST follow accepted `ADR-004` / `spec/split_policy.json`.

## 6. Distribution policy for v0.1

Frozen choice (`ADR-004` Accepted):

- tracked corpus size = 25 distributions, 2500 graphs total
- no silent omission is allowed
- preferred split family = `15 train / 5 validation / 5 meta-test`
- all 25 distributions MUST be used explicitly (no distribution excluded)

### latin_grid_holdout_v1 (deterministic distribution IDs)

Let:

- `fat ∈ {0.4,0.5,0.6,0.7,0.8}`
- `density ∈ {0.4,0.5,0.6,0.7,0.8}`

Define indices:

- `fat_index`: `0.4→0, 0.5→1, 0.6→2, 0.7→3, 0.8→4`
- `density_index`: `0.4→0, 0.5→1, 0.6→2, 0.7→3, 0.8→4`

Then:

- `validation`: `fat_index = density_index`
- `meta-test`: `fat_index = (density_index + 2) mod 5`
- `meta-train`: remaining combinations

For the current inventory:

- validation distribution IDs: `{2,6,10,16,17}`
- meta-test distribution IDs: `{7,12,14,20,23}`
- meta-train distribution IDs: `{1,3,4,5,8,9,11,13,15,18,19,21,22,24,25}`

## 7. Held-out graph partition

For each held-out distribution in validation or meta-test:

- support count = `20`
- query count = `80`

### stratified_sha256_rank_v1 (support/query stratify by CCR)

Within a held-out distribution:

1. Strata by `generator_ccr ∈ {0.3,0.4,0.5}`.
2. Let `n_c` be number of graphs in stratum `c`.
3. Proportional support quota: `q_c = support_count * n_c / (support_count + query_count)`.
4. Integer support allocation:
   - assign `floor(q_c)` first
   - distribute remaining slots by largest remainder
   - quota remainder tie-break: `ascending_ccr` (smaller CCR first)
5. Deterministic pick inside each stratum:
   - `assignment_hash = SHA256(split_version + "\0" + split_seed + "\0" + distribution_id + "\0" + relative_path + "\0" + raw_sha256)`
   - sort ascending by `assignment_hash`
   - first `support_count_c` → support, rest → query

Normative parameters live in `spec/split_policy.json`.

## 8. Manifest row requirements

Each row in dataset manifest MUST include:

- `graph_id`
- `relative_path`
- `raw_sha256`

  For `MARGO-DATA-v1`, `raw_sha256` is SHA-256 over source-file bytes after deterministic EOL normalization:

  1. CRLF (`0D 0A`) → LF (`0A`)
  2. remaining CR (`0D`) → LF (`0A`)
  3. no other byte transformation is permitted

  Field name stays `raw_sha256` for frozen schema compatibility. Assignment `assignment_hash` MUST use this same value.

  `.gitattributes` (`*.gv text eol=lf`) is a checkout aid and does not replace this definition.
- `canonical_graph_sha256`
- `distribution_id`
- `role`
- `node_count`
- `edge_record_count`
- `unique_edge_count`
- `duplicate_edge_record_count`
- `is_dag`
- `max_indegree_unique`
- `max_outdegree_unique`
- `spec_version`
- `manifest_version`
- `split_version`

Optional but preferred:

- generator seed
- generator version
- exclusion reason

## 9. Validation fail conditions

Validator MUST fail if:

- a file is missing from manifest
- a manifest row points to a missing file
- one graph appears in multiple roles
- support/query overlap exists
- a meta-test distribution appears in meta-train
- a graph is cyclic
- hash mismatch occurs
- an included graph does not have `node_count = 20`
- absolute paths, `..`, or dataset-root escape are detected
- `role = excluded` without `exclusion_reason`

## 10. Phase 0 deliverable

Phase 0 Data/Split gate is closed when `dataset_manifest.jsonl` and sidecar `dataset_manifest.jsonl.sha256` exist and validator passes `--mode final` on the frozen dataset tree.
