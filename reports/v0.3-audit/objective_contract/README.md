# objective_contract_v1 — integration evidence

`objective_contract_smoke.json` is the output of one real outer iteration of the
primary contract on the Kish **CPU** container (no GPU; the checkpoint below is
TensorFlow 1.15, so the run is deterministic in nothing but its artifacts):

* run: `runs/objective_contract_smoke/seed_0`, code `d74c4477…`
  (`spec/objective_contract_smoke.py --itr 1`), obs v3 / mask off / constraints off
* `passed: true` — all eleven checks, including:
  * `selection_scalar` == `validation/objective_discounted_return_k3` **exactly**
    (0.555946990805303),
  * `checkpoint_selection_metric == "validation/objective_discounted_return"` and
    `checkpoint_selection_metric_name == "discounted_return"`,
  * the sidecar `ckpt/meta_model_best_val.metric.json` names the contract and
    carries the same value,
  * `legacy_scalar_is_a_different_number`: the pre-contract undiscounted sum is
    0.6023961999171015 vs the objective 0.555946990805303, so the two functionals
    really do disagree on the same rollout (that divergence is why the criterion
    was changed),
  * the human-readable companion is present (`query_mean_latency = 897.53 s`).

Reproduce (CPU is enough):

```bash
spec/kish_gpu.sh energy-tests                      # 898 tests, OK
docker run --rm -e PYTHONPATH=/work -e MARGO_ALLOW_GPU=1 \
  -v "$ROOT":/work -w /work "$IMAGE" \
  python -m spec.objective_contract_smoke --run-dir runs/objective_contract_smoke/seed_0 --itr 1 --i-allow-gpu
```
