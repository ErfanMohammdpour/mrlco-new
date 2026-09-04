# Phase 2 — Graph Encoder

Status: CLOSED

Technical closure commit: this commit  
Tested encoder SHA: `06b97e9d556fe9daa3b0d35c452c7ac0114e58d4`  
Same-SHA TF evidence commit: `4625f697af046e90ea4f54658357f947b80eff80`

**Do not move or rewrite `phase1-freeze-v0.1`.**  
Tag `phase2-freeze-v0.1` MUST point at this closure commit. Do not rewrite it.

**Phase 2 closure does not imply PPO / outer-update / trainer-split / evaluation readiness.**

Until Phase 3 closes: no scientific GPU training and no new paper figures. Unit tests and CPU smoke tests are allowed.

## Provenance

```text
phase1-freeze-v0.1  b611f7047b1c28e3aa6f0026683e87a3ae249b34
  └── c0cbecc  feat(encoder): implement canonical DAG observations
        └── 367c10e  fix(encoder): harden phase 2 runtime and reproducibility
              └── 06b97e9  docs(encoder): record Python 3.7 + TF 1.15 CPU smoke evidence
                    └── 4625f69  docs(encoder): record same-SHA TF 1.15 evidence for 06b97e9
                          └── this commit  Phase 2 CLOSED
```

Execution evidence is for `06b97e9` (clean tree, `git_status=0 dirty_paths`). `4625f69` changed only `spec/phase2_tf_smoke_evidence.txt`.

## Frozen contracts

- Canonical DAG adjacency, not clique
- Packed obs: `FEATURE_DIM + 2*MAX_NEIGH + 1` = 50; `MAX_NEIGH = MAX_TASKS - 1` = 19
- Neighbor indices are decoder-order positions, not raw task IDs
- Neighborhood: predecessor AND successor; combine by sum
- 2-layer masked mean; self∥neigh concat; encoder dropout 0.0
- Degree overflow fail-fast; no silent 6-neighbor truncation
- Features: workload, output, external, incoming/outgoing `edge_output_bytes`, degree, decoder index, depth, root/sink
- No legacy scheduler times (`T_loc` / `T_up` / …) in the observation
- Z-score from `spec/encoder_feature_stats.json`, `role=meta_train` only (1500 graphs / 30000 nodes); canonical-LF hash pins
- Neighbor sampler is fixed-capacity padded adjacency with masked aggregation, not stochastic uniform sampling
- Production `obs_dim=env.input_dim`

## Evidence (CPU)

- Python 3.7.17
- TensorFlow 1.15.5 with `tf.contrib`
- linux/amd64 Docker
- 26/26 tests including 5/5 TF smoke
- `Phase 2 encoder: PASS`

## Gate

```bash
python3 spec/phase2_gate.py
```

Must print `Phase 2 closure: PASS`.
