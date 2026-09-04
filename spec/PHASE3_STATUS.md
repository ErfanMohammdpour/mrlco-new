# Phase 3 — Learning loop

Status: CLOSED

Technical closure commit: this commit
Tested learning SHA: `f17d264b49d0886b3f0e9f438d203af8ab0ef9af`
Same-SHA TF evidence: `spec/phase3_tf_smoke_evidence.txt`

Branch: `phase3-learning` from tag `phase2-freeze-v0.1` (`a8ced73b32b1b565fdd1d0330a724282127f376b`).

**Do not move or rewrite `phase1-freeze-v0.1` or `phase2-freeze-v0.1`.**
Tag `phase3-freeze-v0.1` MUST point at this closure commit. Do not rewrite it.

**Phase 3 closure does not imply paper results, multi-seed evaluation as a claim, or a completed GPU campaign.**

Phase 4 scientific GPU training is now permitted. No new paper figures until Phase 4 evaluation artifacts exist.

## Provenance

```text
phase2-freeze-v0.1  a8ced73b32b1b565fdd1d0330a724282127f376b
  └── fdfc98b  feat(learn): wire frozen PPO, outer mean-PG, and split
        └── f246006  fix(learn): align eval, PPO, and val with protocol
              └── f17d264  test(learn): add inner/outer TF smoke and split scenarios
                    └── this commit  Phase 3 CLOSED
```

Execution evidence is for `f17d264` (source tree). Docker rewrote tracked `*.pyc` only.

## Frozen contracts

- PPO value clip `v_old + clip(v_new - v_old, -eps, eps)`
- `k_steps = 3` means three Adam `apply_gradients` calls
- fresh inner Adam state per meta-task / task slot
- outer update = `mrlco_first_order_mean_pseudogradient`
- mini-batch shuffle before each inner epoch
- `ppo_batch_size_trajectories = 20`, `support_graphs_per_meta_task = 20`
- trainer loads exactly the 15 `meta_train` distributions
- evaluator adapts on support 20, reports query 80
- zero-shot `k_steps=0` and `k_steps=3` both reported
- sync task slots from core at the start of every outer iteration
- validation every 50 outer iters on a scratch copy of core
- LEARNING_PROTOCOL §9 log fields

## Evidence (CPU)

- Python 3.7.17
- TensorFlow 1.15.5 with `tf.contrib`
- linux/amd64 Docker `margo-phase2-tf115`
- 41/41 tests including 8/8 TF smoke
- `Phase 3 learning: PASS`

## Gate

```bash
python3 spec/phase3_gate.py
```

Must print `Phase 3 closure: PASS`.
