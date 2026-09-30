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
