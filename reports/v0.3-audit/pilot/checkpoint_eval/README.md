# checkpoint_eval status

## Current verdict (real persisted evidence)

`checkpoint_eval_comparison.json` is now produced from three **real** per-label runs
(`true_init.json`, `final_itr24.json`, `final_itr39.json`, committed next to this
file). They were run on the Kish CPU container (no GPU) at evaluation code
`c644fc5eb603b6f34011f22daad689646a594e08` on checkpoints
`fe90a11…` whose sha256 were verified before the run
(itr24 `ef24d7b1…`, itr39 `2332d5f3…`).

| label | k0 (s) | k3 (s) | all-MEC | Greedy | weights_changed | k3 < k0 |
|---|---|---|---|---|---|---|
| true_init | 1131.3418 | 960.6429 | 630.2798 | 626.6017 | n/a | yes |
| itr0 (in-training CSV) | 982.8035 | 893.3548 | 630.2798 | 626.6017 | n/a | yes |
| final_itr24 | 771.7004 | 780.3524 | 630.2798 | 626.6017 | true | **no** |
| final_itr39 | 832.4153 | 839.2161 | 630.2798 | 626.6017 | true | **no** |

**Verdict: `BLOCKED_CHECKPOINT_EVALUATION`** — the fresh determinism probe
(`deterministic_k0_fresh`) is `false` for all three labels: two k=0 rollouts from
the same weights in the same process disagree. Until the held-out evaluation is
seeded/paired, per-checkpoint differences of a few seconds are not measurable and
no `READY_FOR_LONG_LATENCY_DIAGNOSTIC` label can be granted.

Correction of an earlier claim: the first attempt's logs (`*.log`, SHA `d3ff0c8`)
already contained `deterministic_k0: False`; the statement "cross-process
determinism holds exactly" in an earlier revision of this file was wrong. The
recovered k0 values (e.g. 1136.7784 vs today's 1131.3418) also move between runs,
consistent with unseeded sampling.

Two findings that do survive the non-determinism, because they are much larger than
the observed run-to-run spread: both trained checkpoints beat true-init at k3, and
for both, k3 is *worse* than k0 (the extra inner PPO steps degrade the held-out
plan).

## Reproducing the persisted comparison (one clean re-run)

Each label runs in its own process for full graph isolation; the merge step is
pure and re-runnable (`spec/checkpoint_eval_compare.py`), so the verdict is derived
from committed JSON rather than from prose:

```bash
# 1) per-label evaluation (GPU, one process per label)
spec/kish_gpu.sh checkpoint-eval --one true_init \
  --json "$ROOT/reports/v0.3-audit/pilot/checkpoint_eval/true_init.json"
spec/kish_gpu.sh checkpoint-eval --one final_itr24="$ROOT/runs/pilot_checkpoints/meta_model_final_itr24.ckpt" \
  --json "$ROOT/reports/v0.3-audit/pilot/checkpoint_eval/final_itr24.json"
spec/kish_gpu.sh checkpoint-eval --one final_itr39="$ROOT/runs/pilot_checkpoints/meta_model_final_itr39.ckpt" \
  --json "$ROOT/reports/v0.3-audit/pilot/checkpoint_eval/final_itr39.json"

# 2) pure merge + verdict (no GPU maths; reads only the JSONs and the Pilot A CSV)
spec/kish_gpu.sh checkpoint-eval-compare \
  --eval true_init="$ROOT/reports/v0.3-audit/pilot/checkpoint_eval/true_init.json" \
  --eval final_itr24="$ROOT/reports/v0.3-audit/pilot/checkpoint_eval/final_itr24.json" \
  --eval final_itr39="$ROOT/reports/v0.3-audit/pilot/checkpoint_eval/final_itr39.json" \
  --pilot-csv "$ROOT/reports/v0.3-audit/pilot/pilot_a_progress.csv" \
  --json "$ROOT/reports/v0.3-audit/pilot/checkpoint_eval/checkpoint_eval_comparison.json"
```


## Criterion (objective_contract_v1)

The verdict compares candidates on the **contract objective** (`query_discounted_return`,
higher is better), which is what PPO optimises; latency in seconds is kept as a
labelled companion and no longer selects a candidate. Documents written before the
contract (including the recovered logs below, which only carry latencies) are
compared through a clearly labelled `mean_latency_fallback(-seconds)` and the row
records `objective_source`, so recovered evidence stays readable without pretending
to be the contract metric.

The merge writes `schema=checkpoint_eval_comparison_v1` with the true-init row, the
in-training itr-0 row from the Pilot A CSV, the checkpoint rows, all correctness
checks, and a verdict from the fixed vocabulary
`BLOCKED_CHECKPOINT_EVALUATION` / `READY_FOR_LONG_LATENCY_DIAGNOSTIC` /
`LONG_RUN_ALLOWED_BUT_NO_VALIDATION_IMPROVEMENT_YET` plus a `verdict_detail`.

The merge publishes `schema=checkpoint_eval_comparison_v1` with the true-init row, the
in-training itr-0 row from the Pilot A CSV, the checkpoint rows, all correctness
checks, the criterion, and a verdict from the fixed vocabulary
`BLOCKED_CHECKPOINT_EVALUATION` / `READY_FOR_LONG_LATENCY_DIAGNOSTIC` /
`LONG_RUN_ALLOWED_BUT_NO_VALIDATION_IMPROVEMENT_YET` plus a `verdict_detail`.

The four `*.log` files in this directory are the recovered stdout of the first
attempt at SHA `d3ff0c8`; their JSON was lost because `--json` pointed at a host
path the container cannot see (fixed in e3fd3cd). They are kept as history only.

Caveat on the itr-0 row: it comes from the in-training validation CSV (k3 adaptation
on the same validation split it reports), so it is not the same protocol as the
evaluator's `true_init` row; the verdict only ever compares evaluator rows.

