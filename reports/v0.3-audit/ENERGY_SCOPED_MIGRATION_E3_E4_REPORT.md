# E3/E4 closing report — scoped-energy migration

Scope: `MARGO_BASELINE/mrlco-new`, branch `erfan/phase4-eval`. No training,
pilot, deadline or shield run was launched, and no large artifact was committed.

## Commits (all pushed)

| commit | phase |
|---|---|
| `394a038` | E1.2 — scope-aware, cache-keyed reference ranges |
| `d885809` | E2.1 — consumer scope migration, no semantic change |
| `8139d32` | E2.2 — `energy_telemetry_v1` through rollout logging |
| `827d80c` | E3.1 — primary reward is latency-only; publication is opt-in |
| `57dfc5b` | E3.2 — objective and total-energy constraint aligned on system |
| this commit | E4.1 — legacy `Average energy` label, closing report |

## Test suite (non-TF)

`python3 -m pytest env/mec_offloaing_envs/scheduler/tests -q`

| point | result |
|---|---|
| before window | 630 passed, 5 skipped, 13 errors |
| after E1.2 | 648 passed |
| after E2.1 | 660 passed |
| after E2.2 | 675 passed |
| after E3.1 | 685 passed |
| after E3.2 | 687 passed |
| after E4.1 | 689 passed, 5 skipped, 13 errors |

The 13 errors are TF collection errors on a machine without TensorFlow; they are
expected in the non-TF suite.

## Consumer scope inventory (regenerated from code)

| consumer | boundary | note |
|---|---|---|
| primary reward | none | latency-only: `J_t = L_t / L_scale` |
| `legacy_publication_reward` | mobile | explicit `reward_mode="publication"` only, byte-exact |
| `reward` logged per-task energy | mobile | `attribute_scoped_energy_by_task(scope=mobile)` |
| `constraints.C_UE_ENERGY` | requester | metric requester, all-UE anchor scope-invariant |
| `constraints.C_HELPER_ENERGY` | helper component | metric `mobile - requester` |
| `constraints.C_TOTAL_ENERGY` | system | metric, reference and budget system |
| `objective.c_E` / budget / reference | system | `require_reference_scope(refs, system)` |
| `j_report` / `j_lambda` | mobile (reporting) | NOT mass-converted |
| greedy log | mobile | `energy_scalar(scope=mobile)` |
| `meta_evaluator` policy/greedy energy | mobile | unchanged reporting consumer |
| `Average energy` | mobile | legacy key kept, now labelled `energy/average_energy_scope=mobile` |
| `energy_telemetry_v1` | all three | requester/mobile/system + primary scope/value |

Direct boundary reads (`total_*_joules`) outside the accessor are allowlisted to
three files — `scheduler/model.py` (definitions + serialization),
`spec/toy_oracles/oracle_checker.py`, `spec/energy_pareto_audit.py` — and enforced
by `test_energy_consumer_inventory.py` (new read or stale entry => fail).

## Reference cache

`reference_ranges_cache_key` = graph content (canonicalized legacy or
`CanonicalDAG`) + decoder order + reference mode + energy scope + panel params +
energy/radio model + resolved scheduler fingerprint. The env caches per scope;
`get_or_build_reference_ranges` re-validates scope and fingerprint on every hit
and raises `EnergyReferenceMismatch` on any mismatch — no silent fallback, no
recompute. `compute_reference_ranges` remains the mobile-only legacy wrapper.

## Telemetry dataflow

env (`get_reward_batch_step_by_step`, same `ScheduleResult` as the reward) ->
sampler parse (2- or 3-tuple `info`) -> process (`samples_data['energy_telemetry']`)
-> trainer (`collect_energy_telemetry` -> `aggregate_energy_telemetry`, episode
weighted) -> CSV (`energy/{requester,mobile,system,primary}_joules`,
`energy/primary_scope`). `build_energy_telemetry` performs zero schedules (tested
with the adapter mocked), telemetry on/off changes neither reward nor schedule,
and missing/malformed/non-finite records or inconsistent scopes/fingerprints
raise.

## Remaining real items

* E4.2+: wire a producer for `validation_per_graph_plans` so the objective
  `log_only`/`lexicographic` path is actually fed; today the evaluator never emits
  it, so those modes report `objective/unavailable`.
* A publication run under `reward_mode="publication"` plus enabled constraints is
  now rejected by the system-scope guard; if that combination is ever wanted it
  needs an explicitly labelled legacy-constraint path.
* TF-dependent tests remain unrun locally (no TensorFlow); they must be exercised
  in the TF image before any training.
