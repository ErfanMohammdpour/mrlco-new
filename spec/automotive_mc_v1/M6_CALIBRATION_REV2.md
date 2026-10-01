# M6-CALIBRATION-REV2 — decoupling activation period from application deadline

Status: **specification frozen, implementation NOT yet applied.** M6 stays BLOCKED
until this revision is implemented, M5/M6 are rerun from scratch, and the hashes and
reports are regenerated. No number may be enlarged by hand; every change below is a
semantic or calibration-rule change.

## Diagnosis carried over from M6 v1

```
family                               D_G (s)   CP_ref (s)  v1 status
perception_planning_control           0.0750     0.1543    infeasible_reference_lower_bound
cooperative_perception                0.0875     0.1689    infeasible_reference_lower_bound
localization_prediction_planning      0.0750     0.1402    infeasible_reference_lower_bound
mapping_background                    1.0625     0.0480    ok, slack_min 1.014 (over-slack)
```

Three separate defects were conflated. Each is fixed on its own terms; none is
fixed by raising a period or a deadline.

## 1. Analytical lower bound must be a real lower bound (fixes the transfer term)

`CP_ref` is an analytical LOWER bound, so it may not charge every edge a mandatory
20 Mbps transfer when producer and consumer are allowed to co-locate:

```
q_e^LB = min over permitted placements of T_tx,e
q_e^LB = 0                        if co-location of the edge's endpoints is permitted
q_e^LB = 8 * B_e / R_max_allowed  otherwise (cheapest physically permitted link)
```

The fixed 20 Mbps reference transfer may still be reported as a *reference
construction scenario*, clearly labelled, but it must never be used as the lower
bound that decides feasibility.

## 2. Obi timing granularity (fixes the planning-motif double count)

86.83 ms is a **planning-motif** anchor, not one node's independent execution time.
Assigning it to `planning` while also charging trajectory, safety-check and control
nodes separately double counts a composite measurement.

```
sum over tasks i in the planning motif of t_ref_i  ==  86.83 ms   (aggregate preserved)
```

Partitioning across motif tasks uses a pre-frozen rule. If no defensible mapping
exists, the split is `source_calibrated_synthetic`, **not** `measured`. The same
audit applies to the Validator (0.05 ms) and Trajectory (4.42 us) anchors: each is
charged to its own node only, not to every planning-like role.

## 3. Decouple the two axes (the central semantic fix)

```
P_f  = activation / update period          (workload and activation semantics)
D_f  = application-chain relative deadline (the SLA axis)
```

v1 collapsed them into `D_G = rho_f * P_f`, which silently made an update period
serve as an end-to-end latency budget. They are independent quantities and stay
independent.

## 4. Replace the only SLA construction rule

`D_G = rho_f * P_f` is retired as the sole rule. The revised construction is:

```
D_G ~ D_f          family-level frozen latency-budget distribution
```

`D_f` is `source_calibrated_synthetic` unless a direct application-E2E requirement
exists, and is declared as:

```yaml
source_type: source_calibrated_synthetic
measurement_scope: application_e2e
semantic_role: requirement
rule_id: E2E-SLA-V2
```

## 5. Hard prohibitions that survive the revision

```
D_G = kappa * CP_graph          FORBIDDEN (would rebuild the deadline from the instance)
D_G chosen from witness/search  FORBIDDEN
D_G adjusted after feasibility  FORBIDDEN
P_f raised to rescue feasibility FORBIDDEN (period and deadline are not repair tools for each other)
communication-scope latency as D_G FORBIDDEN
measured latency named as an SLA   FORBIDDEN
```

## 6. Period evidence (independent of feasibility)

If a source supports a different activation period (for example the ~120 ms chain
periodicity reported by DATE 2021), `P_f` is corrected from that evidence alone.
This correction is recorded separately and is never justified by the resulting
feasibility numbers.

## 7. Re-run, version and compare

1. bump the M5/M6 artifact versions (`workload_model_v2`, `sla_registry_v2`,
   `deadline_model_v2`) so the old hashes remain reproducible;
2. rerun the M6 validator over all four templates from scratch;
3. regenerate `deadline_report.json`, the SLA config hash and the deadline-model
   hash, and record the old-vs-new comparison;
4. audit **both tails**: no main family structurally infeasible, and no family with
   absurd slack such as `mapping_background`'s v1 `D_G = 1.0625 s` against
   `CP_ref = 0.048 s` — that family's deadline distribution needs the same scrutiny
   as the failing ones.

## Acceptance criteria for M6 after REV2

- the three primary families are no longer structurally infeasible because of a
  period/deadline definition error;
- individual graphs may still report `infeasible_reference_lower_bound` as explicit
  stress cases, and that is acceptable;
- no family carries pathological slack from an unjustified deadline distribution;
- `E_i <= d_i <= L_i`, `S_i >= 0`, deterministic rerun and witness/scheduler
  independence all still hold;
- the report contains the old-vs-new hash comparison and states plainly which
  numbers changed and why.
