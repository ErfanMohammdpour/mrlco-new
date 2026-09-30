# PARAMETER_SOURCE_MATRIX — MARGO-AUTOMOTIVE-MC-v1

Every dataset field maps to an evidence source or to a pre-declared generation
rule. A field whose row is not `allowed_for_generation: true` keeps **Step 2
BLOCKED** and the generator does not start.

Verification vocabulary: `official_verified`, `primary_verified`,
`peer_reviewed_verified`, `secondary_lead_unverified`, `derived`,
`synthetic_calibrated`. A quoted-but-unchecked number is a **lead**, not synthetic.

| Dataset field | Source / evidence | semantic_role | measurement_scope | verification_status | allowed | generator use |
|---|---|---|---|---|---|---|
| semantic roles, motif structure | `AUTOWARE-OFFICIAL-DOCS`, `ECLIPSE-APP4MC-AMALTHEA` | structural | — | official_verified | yes | defines task/edge semantics and the Perception→Planning→Control direction |
| execution-needs vs data-movement split (invariant-task rule) | `ECLIPSE-APP4MC-AMALTHEA` | structural | — | official_verified | yes | justifies `W_i` per task and `B_e` per edge |
| C_LO / C_HI structure | `VESTAL-2007-RTSS` | structural | — | peer_reviewed_verified | yes | model for multi-assurance execution budgets (no numbers taken) |
| V2X message payload | `3GPP-TS22186-R16` (T 5.2-1, 5.3-1) | payload | communication | official_verified | yes | message payload semantics, **only for the matching use case** |
| V2X message rate | `3GPP-TS22186-R16` (T 5.2-1, 5.3-1) | offered_load | communication | official_verified | yes | cooperative-message generation period |
| V2X latency | `3GPP-TS22186-R16` (T 5.2-1 … 5.5-1) | requirement | communication | official_verified | yes | hop/message budget — **never `D_G`** |
| V2X reliability | `3GPP-TS22186-R16` | requirement | communication | official_verified | yes | stress/sensitivity reporting only |
| V2X data rate | `3GPP-TS22186-R16` | requirement | communication | official_verified | yes | sanity upper bound only; **not a link capacity** |
| V2X range | `3GPP-TS22186-R16` (R.5.3-006) | requirement | communication | official_verified | yes | context; not modelled in v1 scheduling |
| per-stage execution distributions (perception, planning, control) | `MDPI-SENSORS-2026-ROS2-AD` (1:10 scale vehicle) | measured_latency / execution_time | processing_stage | secondary_lead_unverified | **no** | pending full-text extraction with table/figure numbers |
| perception soft deadline & mean latency | `IET-ITS-2025-JETSON-ORIN-NANO` | requirement / measured_latency | processing_stage | secondary_lead_unverified | **no** | pending full-text extraction |
| fresh-perception-input constraint (40–50 ms) | secondary quotation only | requirement | processing_stage | secondary_lead_unverified | **no** | must not be treated as an SLA or as synthetic |
| task-set period/ACET/WCET structure | `WATERS-2015-REAL-WORLD-BENCHMARKS` | execution_time / offered_load | processing_stage | secondary_lead_unverified | **no** | pending extraction with table numbers |
| sub-deadline allocation method | `ECRTS-2024-DAG-TIME-CONSTRAINTS` | structural (method) | — | primary_verified (identifiers) | review | methodological authority for Step 6/13 |
| `t_ref_i` | derived from verified per-stage distributions | derived | processing_stage | derived (rule pending) | **no** | needs ≥2 verified platform families |
| `C_LO`, `C_HI` | derived from verified execution distributions | derived | processing_stage | derived (rule pending) | **no** | naming: `empirical_execution_budget_*`, never WCET |
| `W_i` (compute workload) | `W_i = t_ref_i · f_ref / (8ξ)`, one reference tier | derived | processing_stage | derived (rule pending) | **no** | task property; never per-action |
| `B_e` (payload bytes) | application/message semantics from TS 22.186 + sensor-message sources | payload | communication | partially verified | **partial** | edge property; never resized per radio link |
| `D_G` (graph deadline) | direct `application_e2e` anchor **or** rule `E2E-SLA-V1` | requirement | application_e2e | unassigned | **no** | non-circular: frozen before any scheduling |
| radio `R` per profile | measurement/model, or `source-calibrated-synthetic` with rule | capacity | communication | unassigned | **no** | 7/5 Mbps carry no authority; required data rate ≠ capacity |
| criticality classes | application semantics (Autoware/APP4MC role taxonomy) | structural | — | rule pending | **no** | must not be depth-derived |
| tardiness weights | policy, not a measurement | structural | — | rule pending | **no** | independent axis |

## Blocking summary before Step 3

1. `application_e2e` anchor: run the `E2E-SLA-V1` stopping rule. Either a direct
   requirement is found (then `official_verified`/`primary_verified`), or the
   family-level `D_G = ρ_G P_G` / `D_G = η B_G` rule is instantiated with its
   calibration parameters frozen in advance.
2. Execution distributions: at least two platform families extracted from full
   text with table/figure numbers, so no overfitting to one scale car or Jetson.
3. Payload semantics: sensor-frame and feature-map sizes need a source or an
   explicit `source-calibrated-synthetic` rule; a 2000 B V2X message is not a
   camera frame.
4. Radio profiles: separate required rate, offered load and achievable capacity;
   define nominal/degraded/strong distributions that are verified or explicitly
   synthetic-calibrated.
5. Every row above must reach `allowed_for_generation: true` before the generator
   runs. Until then: **Step 2 BLOCKED**.

---

# FINAL GATE STATE — Step 2 closure attempt (`source_registry_v3`)

The gate is now a conjunction, because an authoritative document does not prove
that the number written in this repository was transcribed correctly:

```
allowed_for_generation = source_acceptable AND transcription_verified
                         (generated values: generation_rule_verified AND rule_id != ∅)
```

**Current state: 26 numeric rows, 0 allowed, 26 blocked.** Every TS 22.186 row
carries `source_verification: official_verified` together with
`transcription_verified: false`, so the standard is registered as authoritative
while no transcribed value is admitted yet. Example (cooperative collision
avoidance, R.5.3-001):

```yaml
value: 10            unit: ms
table: Table 5.3-1   requirement_id: R.5.3-001
measurement_scope: communication     semantic_role: requirement
source_verification: official_verified
transcription_verified: false
allowed_for_generation: false
```

## The five gates that must close

| # | gate | requirement for `allowed_for_generation: true` | next action |
|---|---|---|---|
| 1 | 3GPP transcription | every value checked against page + table + requirement id in the official text | fetch ETSI TS 122 186 V16.2.0 PDF, record page/table/row per value, set `transcription_verified: true` |
| 2 | execution timing | ≥2 independent platforms, numbers read from table/figure/full text (not a summary) | extract the ROS 2 scaled-vehicle stage table and the Jetson Orin Nano results table |
| 3 | payload semantics | a payload rule per category actually used by the generator (`raw_image`, `feature_tensor`, `object_list`, `trajectory_control`) | define the schemas: `B = w·h·bpp` for raw; a conditional distribution over resolution/codec/quality for compressed; separate models for feature/object messages |
| 4 | radio resource profiles | simulated achievable capacity separated from required data rate, each profile with provenance or rule | define `capacity_model: source_calibrated_distribution` with `source_ids` per profile |
| 5 | application E2E | a real direct requirement, or `E2E-SLA-V1` instantiated per family | freeze `period_source`, `period_s`, `rho_distribution`, `rho_source` and `rule_id` per application family, plus `p`/`eta` for the `B_G` branch |

## Instantiation requirements for gate 5 (formula alone is not enough)

```yaml
family: cooperative_perception
period_source: <source id or synthetic_calibrated>
period_s: <frozen value or distribution>
rho_distribution: <frozen distribution>
rho_source: synthetic_calibrated
rule_id: E2E-SLA-V1
```

`rho_f` (and `p`, `eta` in the `D_G = eta * B_G` branch) are frozen **before** any
schedule search and may never be adjusted after observing feasibility. Payload is
never a single distribution over all edges: the 2000 B V2X message is valid only
for its own use case.

## Verdict

Step 2 is **BLOCKED**, and honestly so: the registry is now well-formed with zero
unresolved *schema* fields, but zero rows are admissible to the generator because
no transcription has been verified against full text yet. Step 3 does not start
until gates 1-5 close and this matrix has no blocked field that the generator
needs.

---

# GATE DEFINITION v2 — made non-blind to absent parameters

Counting only rows that already exist made `blocked_required` a **lower bound**: a
parameter the generator needs but that was never registered was invisible. A
manifest independent of the registry now declares what the generator requires:

```
spec/automotive_mc_v1/REQUIRED_PARAMETER_MANIFEST.yaml
```

Step 2 passes only when ALL FOUR hold:

```
missing_required_parameters == 0
blocked_required_parameters == 0
violations                  == 0
registry_pin_stale          == false
```

Current report (validator, `spec/automotive_mc_v1/validate_source_registry.py`):

```
required_parameter_entries   = 26   (of which rule-only = 3, satisfied by structural sources)
missing_required_parameters  = 15   <- parameters with no registry row at all
blocked_required_parameters  = 8    <- bound rows that are not yet allowed
registry_pin_stale           = false
violations                   = 0
gate                         = BLOCKED
```

`registry_pin_stale` is checked by comparing `sha256(SOURCE_REGISTRY.yaml)` with
the hash recorded in `registry_pin.json`, so editing the registry without
regenerating the pin now fails instead of silently passing.

The 15 missing parameters are the invisible ones: per-category payload models
(`raw_camera`, `lidar`, `radar`, `feature_tensor`, `object_list`, `trajectory`,
`control`), the achievable-capacity radio profiles (`mec_ul`, `mec_dl`, `v2v`,
`helper_compute`), and one `application_e2e` anchor per v1 family
(`perception_planning_control`, `cooperative_perception`,
`localization_prediction_planning`, `mapping_background`).

## Payload sizing semantics (corrected, unambiguous)

```
bits_per_pixel:    B = W * H * bits_per_pixel / 8
bits_per_channel:  B = W * H * C * bits_per_channel / 8
```

Never multiply the channel count by a quantity that is already per pixel. Camera,
LiDAR and radar keep **separate** rules — there is no single `raw_sensor` bucket —
and compressed payloads never follow the raw formula: they need a measurement or a
calibrated distribution with provenance.

## Per-family E2E anchors (one is not enough)

A single shared `D_G` distribution across families is forbidden. Each family listed
in the manifest must have either a direct application-level requirement or its own
`E2E-SLA-V1` instance with frozen `period_source`, `period_s`, `rho_distribution`,
`rho_source` and `rule_id`, hashed before the first scheduling call. A family that
is deliberately not SLA-constrained in v1 must say so explicitly in the manifest
instead of inheriting another family's deadline.

---

# ONTOLOGY, NAMING AND BINDING POLICY (locked)

## Resource ontology

A profile must be able to say "poor V2V but strong helper CPU" independently, so
the two capacity kinds are separate categories:

| kind | members |
|---|---|
| `communication_capacity` | `MEC_UL`, `MEC_DL`, `V2V` |
| `compute_capacity` | `UE`, `HELPER`, `MEC` |

`helper_compute_profile` is a compute resource, not a radio profile. The three
compute entries carry `satisfied_by: co_physical_config(f_hz=...)` and
`pending_source: true`, so the tier frequencies are recognised explicitly rather
than inherited implicitly; they become admissible only once sourced or backed by a
declared rule. A physical profile is the tuple
`R_p = (f_UE, f_H, f_MEC, R_UL, R_DL, R_V2V)`.

## Two manifest hashes, two names

```
required_parameter_manifest_sha   hash of REQUIRED_PARAMETER_MANIFEST.yaml
dataset_manifest_sha              hash of the MATERIALIZED dataset manifest (not created)
```

The earlier single `manifest_sha` label was ambiguous and is retired.

## Binding policy

A required parameter backed by more than one registry row must declare
`binding_policy`; otherwise a value would satisfy every family by accident, e.g. a
2000 B cooperative message silently becoming the payload of every use case.

```
single                exactly one row (default)
use_case_conditioned  rows selected per use case via binding_map, which must cover
                      exactly the declared bindings
all_must_be_allowed   every bound row must be admissible
```

Current multi-row bindings, all `use_case_conditioned` with explicit maps:
`v2x_message_payload_model`, `v2x_message_latency_budget`, `v2x_message_rate`.

## Gate state after the cleanup

```
required_parameter_entries   = 28   (rule-only = 3)
missing_required_parameters  = 17   (payload 7, communication_capacity 3,
                                     compute_capacity 3, application_e2e 4)
blocked_required_parameters  = 8
violations                   = 0
registry_pin_stale           = false
gate                         = BLOCKED
```
