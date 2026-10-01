# DATASET CARD — MARGO-AUTOMOTIVE-MC-v1

## Identity

| field | value |
|---|---|
| dataset name | MARGO-AUTOMOTIVE-MC-v1 |
| version | v1 (annotations frozen at materialization; any semantic change needs a version bump) |
| status | Step 2 (source/evidence gate) PASS; generation not yet materialized |
| nature | **semantics-, source- and trace-grounded SYNTHETIC** automotive mixed-criticality DAG benchmark |
| not | a collection of measured production automotive DAGs; a real-world measurement dataset; a reproduction of any single published vehicle stack |

## Intended use

Meta-RL and scheduling research on latency-optimised offloading with
mixed-criticality guarantees: per-task placement (UE / helper / MEC) on 20-task
automotive application DAGs, with energy and deadline requirements carried as
constraints and reported metrics, and with held-out application families and
resource regimes that require generalisation rather than memorisation.

## What is measured, derived, calibrated or simulated

| class | meaning | examples in v1 |
|---|---|---|
| `standard-derived` | value read from an official standard clause/table | V2X payload, message rate, latency, reliability, data rate, range from 3GPP TS 22.186 V16.2.0 (Tables 5.2-1/5.3-1/5.4-1), ledgered with requirement ids |
| `measured` | value read from a measurement in a cited source | timing rows with execution context; note the platform caveats below |
| `derived` | computed from verified inputs by a stated rule | raw camera `B = W*H*bits_per_pixel/8`; LiDAR/radar by points/detections |
| `source-calibrated-synthetic` | generated under a frozen rule whose anchors are cited | resource profiles, E2E `rho_f`, task classes without a direct CPU measurement |
| `synthetic_calibrated_rule` | internal rule with a `generation_rule_id` | the 17 rule rows instantiated in Step 2 (payload models, capacities, E2E) |

## Reference execution tier (locked interpretation)

| field | value |
|---|---|
| platform | Intel Core i9-12900HX **CPU** |
| `f_ref` | **2.30e9 Hz**, 8 logical CPUs pinned |
| source | Obi et al., IEEE OJIES 2026 — Table 4 (hardware) and Table 11 (timings) |
| meaning | ONE canonical CPU-compatible tier with exact frequency and a small set of exact CPU-measured timings. It is **not** a requirement that one source measured all 20 roles. |
| other classes | built on this same tier by frozen `source-calibrated-synthetic` rules and recorded as calibrated, **never** as measured |
| `xi` | **300 cycles/bit** — Hu et al. Table I, a peer-reviewed **simulation parameter**: not a measured automotive constant, not a universal physical property |
| formula | `W_i = t_ref_i * f_ref / (8 * xi)`; `T_i^UE = 8*xi*W_i/f_UE`, and likewise for HELPER and MEC |
| invariance | `W_i` is a property of the task; the chosen action must never change it |

## Communication payload (separate from compute)

`T_tx,e = 8 * B_e / R_link`. Payload categories are distinct rules: camera, LiDAR,
radar, feature tensor, object list, trajectory, control, V2X message. `B_e` is
never derived from `W_i`, never resized to make a deadline feasible, and a
3GPP *service-required* data rate is never used as scheduler link capacity
(`R_required != R_offered != R_achievable`; only achievable capacity enters the
scheduler).

## SLA construction

`D_G = rho_f * P_f` only — the `eta * B_G` branch is **not** active in v1.
`rho_f` and `P_f` are frozen before any scheduling or search, and no deadline may
be rewritten after observing feasibility. Scopes stay distinct
(`communication`, `processing_stage`, `application_e2e`); only a row with
`semantic_role: requirement` and `measurement_scope: application_e2e` may define
`D_G`. Subdeadlines are derived afterwards from the frozen `D_G` with
`E_i <= d_i <= L_i` and `S_i = L_i - E_i >= 0`; criticality may redistribute slack
by a frozen rule but never defines the external SLA.

## Mixed-criticality semantics

Four independent axes: safety criticality, deadline urgency, tardiness weight,
execution-demand uncertainty. Criticality comes from application semantics and is
never derived from DAG depth, topological or decoder position, workload magnitude
or deadline tightness. HIGH tasks carry
`empirical_execution_budget_lo <= empirical_execution_budget_hi`; quantiles are
never called WCET unless the source provides WCET. Operating modes are LO and HI
with explicit triggers; HIGH tasks are never silently dropped in HI mode, MEDIUM
and LOW behaviour is declared per class and mode, and every mode switch is
recorded with its reason.

## Splits and certification

Splits are built from latent generating factors (application family, motif
lineage, parent RNG seed, topology/workload/resource/SLA regimes, criticality
mixture); near-duplicate siblings never cross roles, support and query stay
disjoint, and meta-test is not opened for model selection or calibration.
Certification runs only after a graph is frozen and materialized, may report
`certified_feasible`, `witness_not_found` or `stress_or_infeasible`, and must
never mutate workloads, payloads, deadlines, criticality, budgets or the resource
profile. `witness_not_found` is not mathematical infeasibility.

## Known limitations

- Reference timings come from a research **WSL2** setup; the source itself presents
  them as an empirical characterisation, not production-vehicle timing and not a
  general guarantee.
- The ROS 2 stage distributions were measured on a **1:10 scale vehicle** with a
  Jetson Orin Nano, and the IET evidence on a Jetson Orin Nano with TensorRT/INT8:
  both are GPU-class, verified, and **ineligible** as `t_ref` under the current
  contract.
- DATE 2021 gives clear platform attribution (i5-3210M, ROS2/Linux, ~120 ms chain
  periodicity) but only boxplot values; the plot is not digitised.
- Obi et al. covers planning/control and derived Autoware nodes, not heavy
  perception, so classes such as object detection rely on calibrated rules on the
  reference tier rather than direct CPU measurement.
- The historical engineering baselines (1.0/1.5/10.0 GHz, 7/7/5 Mbps) carry no
  authority of their own; they sit inside calibrated ranges as one point.

## Reproduction

Regeneration is deterministic from generator code + `generation_config.yaml` +
`SOURCE_REGISTRY.yaml` + seeds. `provenance.json` pins the git SHA, the source
registry, the required-parameter manifest, the evidence SHAs, the generation
config, the generator, the semantics/SLA/criticality/resource/split artifacts and
the dataset manifest. The two manifest hashes are named unambiguously and never
collapsed into a bare `manifest_sha`:

```
required_parameter_manifest_sha   hash of REQUIRED_PARAMETER_MANIFEST.yaml
dataset_manifest_sha              hash of the materialized dataset manifest
```

