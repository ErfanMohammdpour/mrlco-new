# HEFT reference audit — the frozen M10 "HEFT" alias and the corrected v2 baseline

## The bug (frozen M10 evidence, not repaired)

`spec/automotive_mc_v1/certify_automotive_m10.py:166` (used at lines 183–184):

```python
tiers = ("f_ue_hz", "f_helper_hz", "f_mec_hz")
...
f = tiers[action]
dur = W_i * 8.0 * XI / float(r[f])
```

The canonical action encoding is `0 = UE, 1 = MEC, 2 = HELPER`
(`scheduler/model.Location.from_action`), so this table prices action 1 (MEC) with the
**HELPER** frequency and action 2 (HELPER) with the **MEC** frequency. Every frozen
profile has `f_mec > f_helper > f_ue`, so the arg-min always returned action 2 and the
"HEFT-style" plan degenerated to `all_HELPER` bit-for-bit — hence the frozen audit's
`heft_reference == all_HELPER`. M10 is historical evidence; it is left as is.

Corrected mapping (action → frequency): `0 → f_ue_hz`, `1 → f_mec_hz`, `2 → f_helper_hz`
(`spec/automotive_training/automotive_dag.py::tier_frequencies`, already correct).

## What `heft_reference_v2.py` fixes

1. **Placement loop / residual-plan construction.** The pre-fix residual forced every
   not-yet-placed task to action 0 (UE), burying the signal under the slowest tier; on
   the six sampled graphs it lost to the best pure plan on four. v2 now anchors on the
   best pure plan and walks the tasks in **descending upward rank**, probing the three
   actions against the real canonical scheduler and accepting a placement only when it
   strictly lowers the makespan. Result: never worse than any pure plan, never a pure
   alias, transfer- and contention-aware.
2. **Rank recursion.** Successor adjacency is taken from `edges` (the same source
   `canonical_dag` uses), not the record's redundant `successors` field; `xi` is read
   from the frozen dataset workload model instead of a hardcoded literal; non-finite
   ranks raise `AutomotiveDagError` instead of propagating.
3. **Historical alias.** `historical_m10_alias` now runs the frozen table's per-task
   arg-min verbatim and exposes `historical_fastest_action` (the key the audit asserts).

## Per-graph evidence (frozen graphs, `json.loads` per line)

Times in seconds; counts are `UE / MEC / HELPER` of 20 tasks.

| # | graph_id | v2 makespan | all_HELPER | best pure (all_MEC) | counts | alias action |
|---|----------|-------------|------------|---------------------|--------|--------------|
| 0 | mcv1_coop_s0_whigh_dloose_rhigh_end | 0.036862933 | 0.340617682 | 0.048214948 | 3/17/0 | 2 |
| 1 | mcv1_coop_s0_whigh_dloose_rlow_end | 0.145220969 | 0.466482287 | 0.154436280 | 2/18/0 | 2 |
| 6 | mcv1_coop_s0_wnominal_dloose_rhigh_end | 0.033388201 | 0.246835115 | 0.035241034 | 2/17/1 | 2 |
| 7 | mcv1_coop_s0_wnominal_dloose_rlow_end | 0.056312804 | 0.431508968 | 0.069272991 | 0/19/1 | 2 |
| 8 | mcv1_coop_s0_wnominal_dnominal_rhigh_end | 0.033687987 | 0.300591689 | 0.038169782 | 3/17/0 | 2 |
| 11 | mcv1_coop_s0_wnominal_dtight_rlow_end | 0.030225811 | 0.260994748 | 0.034624896 | 3/17/0 | 2 |

* v2 is mixed (not all_UE/MEC/HELPER) on 6/6; v2 ≤ all_HELPER on 6/6; v2 ≤ best pure on
  6/6 (strictly better on 6/6).
* `historical_m10_alias` gives `historical_fastest_action == 2`,
  `historical_plan_is_all_helper == True` and `historical_makespan == all_HELPER` to
  1e-12 on 6/6 — the documented regression proof of the M10 bug.
* Audit sweep over all **240** frozen graphs (same code, `audit_graph` per line):
  240/240 strictly beat the best pure plan (always `all_MEC`), 0 pure-plan aliases,
  0 `all_HELPER`, upward ranks finite and `rank[i] > rank[j]` for every edge `i→j`,
  HELPER used on 112/240.

## Counter-example (why the fix is load-bearing)

The pre-fix residual construction (`not-yet-placed → UE`) produced `0.058991816 s` on
graph 0 versus `0.048214948 s` for `all_MEC`, and lost to the best pure plan on 4/6
sampled graphs (lines 0, 1, 7, 8); the corrected placement beats best pure on all six.

## Provenance

`spec/automotive_mc_v1/**` and `env/mec_offloaing_envs/data/**` are **untouched**: the frozen dataset and the frozen M10 certification result stand as history. Evidence: `python3 -m pytest env/mec_offloaing_envs/scheduler/tests/test_heft_reference_v2.py -q` (10 passed) reads the frozen JSONL directly and never imports the M10 module.

## Recommendation

Use `spec.automotive_training.heft_reference_v2` (with
`spec.automotive_training.automotive_dag`) as the HEFT reference for all post-dataset
model evaluation, and quote the frozen M10 `heft_reference` only as the documented
`all_HELPER` defect.
