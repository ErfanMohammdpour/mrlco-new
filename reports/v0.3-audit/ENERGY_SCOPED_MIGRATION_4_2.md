# 4.2 scoped-energy migration — corrections, contracts and plan

Status: **design/lock only; 4.2a/b/c not implemented.** Written so the migration
starts from the corrected concepts instead of the inventory I reported earlier.

Review of `01888d0` (evaluator parity) and `8f133f6` (canonical accessor) is green.
Three conceptual corrections were requested and are accepted here.

## 1. Corrections to my inventory

**1.1 `refs.E_ue` is a PLAN, not a boundary.** It is the energy of the all-UE plan
under `ReferenceRanges.energy_scope`, not the requester scalar:

    E_ue     = energy_scalar(all_UE_result,     scope=reference_scope)
    E_mec    = energy_scalar(all_MEC_result,    scope=reference_scope)
    E_helper = energy_scalar(all_HELPER_result, scope=reference_scope)

Names stay for compatibility, but the schema and docstrings must say
`plan_all_ue_energy_j` / `plan_all_mec_energy_j` / `plan_all_helper_energy_j`.

**1.2 The objective has a scope mismatch TODAY and 4.2 must not hide it.**
`objective.py:233` reads `total_system_joules` while its budget/reference comes from
`refs.E_ue` built with the mobile boundary. That is recorded here, asserted in the
inventory test, and left unfixed until 4.3 -- where the numerator and the
reference/budget are aligned on `system` on purpose. Silently "fixing" it in 4.2
would make the refactor and the scientific change indistinguishable.

**1.3 Migration must NOT use `configured_energy_scalar`.** The primary config now
carries `energy_scope=system`, so a consumer that reads the scope from the config
would jump from mobile to system immediately. Every migrated consumer states its
CURRENT scope explicitly (`energy_scalar(..., scope=SCOPE_MOBILE)` etc.); only 4.3
switches the two intended consumers to `system`.

## 2. Attribution ownership contract (4.2a)

Attribution is an accounting contract, not a causal claim. One owner per energy
component; no duplicates, no unowned components.

| component | owner |
|---|---|
| compute energy | the executed task |
| external input transfer (`src=None`, `dst=task`) | the destination task |
| dependency transfer (`src`, `dst` both tasks) | whatever the CURRENT code owns -- extract it exactly first and keep it; 4.2 may not change legacy attribution |
| terminal sink return (`dst=None`) | the sink/source task |
| MEC compute | the task executed on MEC |
| MEC/RSU TX | the owner of the transfer above |

API:

    attribute_energy_by_task(result, resources, *, scope) -> dict[int, float]

Rules: scope mandatory; every task present as a key even at zero; output finite and
non-negative; an unknown task in a transfer raises; a component without an owner
raises; and for every scope

    sum(per_task.values()) == energy_scalar(result, scope=scope)

The old mobile-only entry point stays as an explicit wrapper
(`attribute_mobile_energy_by_task`) and must remain numerically identical.

Attribution tests: all-UE, all-MEC, all-HELPER, mixed, UE->MEC, MEC->HELPER,
HELPER->MEC, sink return, external root upload, zero-byte transfer; for each scope
the sum identity; under physical_v1 both MEC compute attribution > 0 and MEC TX
attribution > 0 in a plan with a MEC downlink.

## 3. Scope-aware ReferenceRanges (4.2a)

`ReferenceRanges` records `energy_scope`, `scheduler_config_sha256`, `reference_mode`
and is built from the pure plans through `energy_scalar(..., scope=energy_scope)`.
Reference caches are keyed by graph identity/hash, reference mode, energy scope, the
resolved scheduler fingerprint, the energy model and the radio accounting model;
`id(task_graph)` alone is not a primary cache key. `require_reference_scope(refs,
expected)` raises `EnergyScopeMismatch` on mismatch, with the objective mismatch of
§1.2 either deferred to 4.3 as a documented remaining consumer or reproduced through
an explicitly labelled `legacy_scope_mismatch=True` replay path.

## 4. Consumer migration map (4.2b) — every row must keep its number

| consumer | current expression | effective scope now | accessor call after 4.2 | value changes? |
|---|---|---|---|---|
| `reward.py:200,216-219` | `total_mobile_joules` | mobile | `scope=SCOPE_MOBILE` | **NO** |
| `constraints.C_TOTAL_ENERGY` | `total_mobile_joules` | mobile | `scope=SCOPE_MOBILE` | **NO** (4.3: system) |
| `constraints.C_UE_ENERGY` | `total_ue_joules` | requester | `scope=SCOPE_REQUESTER` | **NO** |
| `constraints` budgets | `refs.E_ue` (mobile-built) | mobile ref | `scope` explicit, same refs | **NO** |
| `objective.c_E` | `total_system_joules` | system | `scope=SCOPE_SYSTEM` | **NO** |
| `objective` budget | `refs.E_ue` | mobile ref | unchanged, mismatch recorded | **NO** |
| `energy_api` reference ranges | `total_mobile_joules` | mobile | `scope=SCOPE_MOBILE` | **NO** |
| `energy_api.j_lambda` / `j_report` | `total_mobile_joules` | mobile | parameter renamed, scope validated from refs | **NO** |
| `energy_api.attribute_energy_by_task` | mobile | mobile | wrapper, scope explicit | **NO** |
| `offloading_env` greedy log | `total_mobile_joules` | mobile | `scope=SCOPE_MOBILE` | **NO** |
| `meta_evaluator` metric | `total_mobile_joules` | mobile | `scope=SCOPE_MOBILE` | **NO** |
| `meta_trainer` `Average energy` | per-task mobile | mobile | unchanged, documented legacy | **NO** |

No consumer outside `energy_scope.py` (or raw breakdown serialization) may read a
boundary property directly afterwards.

## 5. Telemetry contract (4.2c)

Three episode-level scalars are taken from the SAME `ScheduleResult` of the rollout --
no second `sess.run`, no scheduler replay:

    {"schema": "energy_telemetry_v1",
     "requester_joules": ..., "mobile_joules": ..., "system_joules": ...,
     "primary_scope": "system", "primary_joules": ...}

The legacy `info` tuple shape stays intact for old callers; the production sampler
extracts the new dict explicitly; parallel subprocess/pickle paths are tested. CSV
aggregation is episode-weighted (not mean-of-means), logs
`energy/{requester,mobile,system}_joules` plus `energy/primary_scope` and
`energy/primary_joules`, and keeps `Average energy` as the legacy MOBILE metric with
an explicit label. CSV tests: header/row field counts equal, the three scalars
finite, `primary_joules == system_joules` under the primary config,
`Average energy == mobile_joules` in 4.2, comma-containing keys still quoted.

## 6. Semantic-no-op acceptance test

For a fixed set of plans, before and after the migration must be equal: reward energy
contribution, per-task mobile attribution, mobile `ReferenceRanges`, `j_report`,
constraint metrics, greedy energy log, evaluator metric and `Average energy`. The
three new columns are additive telemetry only. Timing invariance must still hold
(`timing legacy + accounting physical` gives the same schedule/bounds/mask/obs as
`timing legacy + accounting legacy`).

## 7. Commit plan and forbidden list

1. `feat(energy): add scoped per-task attribution and references` (4.2a)
2. `refactor(energy): migrate energy consumers without semantic change` (4.2b)
3. `feat(energy): carry scoped telemetry through rollout logging` (4.2c)
4. tests/docs/evidence with their commits

Forbidden in 4.2: latency-only reward, `C_TOTAL_ENERGY` -> system, enabling the
Lagrangian, changing the energy budget, silently fixing the objective mismatch,
training, deadline/shield changes, and any second scheduler replay for telemetry.
