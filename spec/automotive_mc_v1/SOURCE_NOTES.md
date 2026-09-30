# SOURCE_NOTES — MARGO-AUTOMOTIVE-MC-v1

## What is verified right now

Three primary sources are available and registered:

1. **3GPP TS 22.186 V16.2.0 / ETSI TS 122 186 V16.2.0** — "Service requirements for
   enhanced V2X scenarios" (SA1, Release 16). Tables 5.2-1 (platooning),
   5.3-1 (advanced driving), 5.4-1 (extended sensors) and 5.5-1 (remote driving)
   provide payload, message rate, latency, reliability, data rate and range values.
   These clauses/tables were recorded from the official text and are registered in
   `SOURCE_REGISTRY.yaml`. An independent PDF fetch-and-spot-check is still pending
   and is listed under `pending_primary_recon`.
2. **Eclipse APP4MC / AMALTHEA** — official structural model (Task, Runnable,
   Label/Channel, ActivityGraph, execution needs, generic instructions). Used for
   generator structure and for the separation of execution needs from data
   movement. Its tutorial numbers are NOT automotive measurements and are recorded
   with `label_for_any_number: forbidden`.
3. **Vestal 2007 (RTSS)** — mixed-criticality model with several compute-time
   estimates at different assurance levels. The theoretical basis for `C_LO`/`C_HI`.
   No numerical parameter is taken from it.

## The one decision that matters most

Every TS 22.186 latency in the registry is `scope: communication`. The 10 ms of
cooperative collision avoidance, the 3 ms of emergency trajectory alignment, the
5 ms of remote driving and the 3/10/50 ms sensor-sharing rows are
**message-exchange / service requirements**. None of them may be assigned to the
graph deadline:

```
R.5.3-001: 10 ms   !=  D_G for a DAG containing perception + planning + control
```

Doing that would silently turn a communication requirement into an application
end-to-end requirement and would make the dataset indefensible. The contract
(`DATASET_CONTRACT.md`) already forbids it; the registry now enforces it at the
field level by carrying `scope` on every row.

The same caution applies to data rates: a *required* data rate (10, 25, 30, up to
700 Mbps for video sharing) is not a link capacity. It may be used as a sanity
upper bound for radio profiles, never as the simulated rate without a PHY/RAN
model.

## What is still missing (calibration is blocked until these are filled)

| gap | what is needed | why it blocks |
|---|---|---|
| `application_e2e_deadlines` | source-backed application-level E2E deadlines or measured chain times | without it `D_G` cannot be fixed before scheduling, and the whole non-circular deadline construction has no anchor |
| `per_stage_execution_distributions` | measured per-stage distributions for perception, localization, prediction, planning, control, from at least two platform families | `t_ref_i` and the empirical `C_LO`/`C_HI` come from here; without it workloads would be invented bytes. PARTIAL: scaled-vehicle (1:10) stages + Jetson Orin Nano perception |
| `payload_message_semantics` | sensor-frame / feature-map / trajectory / control message sizes | payloads must be properties of edges, calibrated from application semantics |
| `radio_resource_profiles` | defensible radio throughput model | `resource_profiles.yaml` must not be frozen on the current frozen 7/5 Mbps assumptions |

## Step 2 completion gates (all four must close before Step 3)

| gate | requirement | state |
|---|---|---|
| application-level E2E | a source must state a requirement/deadline for the WHOLE chain, not a measured latency. If no defensible public number exists, `D_G` becomes `source-calibrated-synthetic` with a stated rule rather than mislabelling a measured latency as an SLA | **OPEN** |
| execution distributions | at least TWO platform/source families so the dataset does not overfit one Raspberry/Jetson/scale car | **PARTIAL** — scaled vehicle (1:10) stage distributions and a Jetson Orin Nano perception measurement are registered; WATERS extraction still pending |
| payload semantics | raw frame, feature-level data, object list / CPM, trajectory and control messages must be separated; a 2000 B V2X message is not a camera frame | **PARTIAL** — TS 22.186 message payloads registered; sensor-frame and feature-map sizes still need sources or an explicit synthetic rule |
| radio profiles | a defensible throughput model, not the frozen 7/5 Mbps and not a TS 22.186 required data rate used as capacity | **OPEN** |

Two further rules recorded with the gates:

- a `requirement` and a `measured_latency` with the same `measurement_scope` are
  different quantities; the registry keeps them apart with `semantic_role`;
- the 150 ms soft deadline and ~133 ms mean perception latency on Jetson Orin Nano
  and the 40-50 ms fresh-perception-input constraint are **stage-level** values.
  They constrain perception, not the end-to-end DAG.

## Identifiers confirmed in recon (read before use)

- [ROS 2-Based Architecture for Autonomous Driving Systems: Design and
  Implementation](https://www.mdpi.com/1424-8220/26/2/463) — Sensors (MDPI) 2026,
  26(2):463, Bonci, Brunella et al. Stage-level mean/P95/max plus pipeline
  end-to-end for lane/object/obstacle. **Platform is a 1:10 scale vehicle**, so it
  is registered with `platform_qualifier: scaled_vehicle_1_10` and must not be
  presented as production-vehicle timing.
- [Deadline-Adherent Edge AI for Intelligent Vehicles (quantized YOLOv8n on Jetson
  Orin Nano)](https://ietresearch.onlinelibrary.wiley.com/doi/pdf/10.1049/itr2.70135)
  — IET Intelligent Transport Systems, doi 10.1049/itr2.70135. Supplies the second,
  independent platform for perception.
- [Real World Automotive Benchmarks For
  Free](http://waters.ecrts.org/forum/download/RealWorldAutomotiveBenchmarksForFree-ECRTS-WATERS2015.pdf)
  — WATERS (ECRTS) 2015, Bosch benchmark: realistic but IP-free automotive task
  sets with period/ACET/WCET properties; the structural and distributional
  grounding for the generator. Values still to be extracted with table numbers.
- [Deadline Miss Early Detection Method for DAG Tasks Considering Variable
  Execution Time](https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.ECRTS.2024.8)
  — ECRTS 2024, doi 10.4230/LIPIcs.ECRTS.2024.8: autonomous-driving-style DAG from
  sensor input to control command with time constraints allocated to nodes from an
  end-to-end deadline. The closest methodological match for the sub-deadline
  scheme; to be read in full before Step 6.

## Next recon steps

1. Fetch the ETSI TS 122 186 V16.2.0 PDF and verify every registered value against
   its table (record page/section per value).
2. Find primary sources for application-level autonomous-driving E2E deadlines.
3. Find primary sources for per-stage measured execution distributions.
4. Find primary sources (or declare `source-calibrated-synthetic` with a stated
   rule) for payload/message sizes.
5. Establish a defensible radio resource profile.

Only after all five are done does generator calibration begin.
