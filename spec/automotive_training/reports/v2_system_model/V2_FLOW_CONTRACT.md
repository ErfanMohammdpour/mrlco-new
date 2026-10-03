# V2_FLOW_CONTRACT — world / meta-task / observation / action / result / reset

## Pipeline (actual, as executed by the code)

```
resolved frozen config (spec/frozen_experiment.yaml)
        |
        v
v2.world.build_world(graph, slot_id, mc, V2WorldConfig, helper_seed, link, compute)
   |                     |
   |                     +--> foreground V2DAGSpec (owner 0) + N background V2DAGSpecs
   |                          (owners >= 1, workload-derived, distinct identities/seeds)
   |                     +--> per-owner UE calendars, helper states (seeded per world),
   |                          shared V2LinkSpec / V2ComputeSpec, arrivals, provenance
   v
V2World.with_foreground_actions(graph, actions)      <-- ONLY the plan changes per candidate
        |
        v
schedule_shared(...)  -> V2ScheduleResult            (one shared calendar set per world)
   |                       timings, radio_ledger, queue_stats, invariants,
   |                       completion_by_dag, makespan_s, world_makespan_s, episode_latency_s
   v
v2.energy.schedule_energy(result)   -> V2EnergyLedger (requester / mobile / system,
   |                                    foreground / background, unmodeled components)
   v
v2.constraints_v2.v2_metrics + costs_from_metrics (v1 ConstraintCosts)
        |
        v
telescoping token rewards  ->  -penalty on the TERMINAL token only
        |
        v
trainer: observe(signed costs)  ->  dual_step() once per iteration (training data only)
        |
        v
PPO (inner)  ->  MRLCO outer update  ->  held-out adaptation/validation  ->  checkpoint
```

## Sampling unit and identities

| concept | definition |
|---|---|
| **batch slot** | one INDEPENDENT world. Unrelated slots are never merged. |
| **world** | the foreground DAG plus `background_dags` DAGs of OTHER owners, scheduled in ONE `schedule_shared` call against the same MEC CPU, MEC radio and V2V channel. |
| **DAG** | one graph instance with a stable `dag_id` (`<world_id>_fg`, `<world_id>_bg<i>`) and `(owner)`; repeated instances of the same dataset graph still have distinct identities. |
| **owner** | an index with its OWN local CPU calendar (`UE_CPU_<i>`); the MEC and the radios are genuinely shared. |
| **episode** | the FOREGROUND DAG. `episode_latency_s = completion_by_dag[foreground]`; the batch `world_makespan_s` is reported separately. |
| **meta-task** | one graph from `meta_train` sampled by the frozen meta-sampler; environmental parameters (load via `background_dags`, link regime, helper contact/busy, MEC workers, reliability class) vary the same task. |
| **rollout batch** | `meta_batch_size` worlds, one per meta task. |
| **support / query** | support = adaptation data; query = a DIFFERENT realization of the same meta-task distribution. Reset semantics: an independent episode resets its own world; a continuous world is never "cleared per DAG". |

## Action contract

Preserved from v1: autoregressive generation of a complete placement plan over UE / MEC /
HELPER, one token per task in decoder order. `schedule_shared` may REWRITE a token (helper
inadmissible, reliability rejected, contact failure → local restart); the sampled action and
the executed location are therefore recorded separately (`task_id → action` in the plan,
`TaskTiming.location` / `attempted_location` after execution).

## Reset semantics

`reset()` rebuilds one world per slot (background, helpers, arrivals) and clears the per-episode
caches. Candidate evaluation never calls `reset()`: `with_foreground_actions` /
`with_foreground_plan_map` produce a snapshot with a different plan and everything else
untouched, so prefix replay, baselines and search see the SAME world and the same exogenous
stream.

## Declared model choices (explicit, not implicit)

* background policy is FROZEN (`all_mec` by default) and declared as a workload generator —
  **no jointly learned multi-user policy is claimed**;
* calendar discipline: non-preemptive earliest-fit, exactly ONE booking per transfer;
* channel discipline: an interrupted transmission RETAINS the channel; a pause is never counted
  as active TX;
* helper contact: the planner sees a prediction, execution is bounded by the REALIZED window; if
  the result cannot return before the contact ends, the remote work is WASTED and the task
  restarts locally IN FULL (no checkpointing is implemented);
* reliability: `p_success = link_confidence × (1 − outage_fraction) × contact_slack`, a
  surrogate, not a calibrated chance constraint;
* energy primary scope: `system` (frozen). `requester` and `mobile` are always reported.
