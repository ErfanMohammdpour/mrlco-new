# BATCH2 source record — execution evidence + xi (reported readings)

Status: recorded readings awaiting mechanical transfer into `SOURCE_REGISTRY.yaml`.
Nothing here is evidence until the registry row carries it with a ledger whose
`evidence_sha256` resolves against a tracked file; the registry itself is not
modified by this file.

## 1. ROS 2 autonomous-driving architecture — Sensors (MDPI) 2026, 26(2):463

- Platform: **Jetson Orin Nano 8GB on a 1:10 scale vehicle**; framework ROS 2.
- Table 6 — per-stage latency, mean / P95 / max:
  - object detection: **28.853 / 33.243 / 49.458 ms**
  - decision making: **6.675 / 11.082 / 15.869 ms**
  - (additional rows reported for preprocessing, lane detection, obstacle detection
    and state machine; values to be transcribed from the table, not paraphrased)
- Table 7 — pipeline end-to-end latency, mean / P95 / max for three pipelines
  (lane / object / obstacle).
- Verdicts: `conversion_to_cycles: forbidden`, `eligible_for_t_ref: false`
  (**per-stage CPU/GPU allocation is not stated**, so t*f -> cycles is not
  defensible), but fully usable for distribution calibration and `C_LO`/`C_HI`
  shapes.
- Binds: `perception_execution_distribution` (Table 6 rows),
  `pipeline_reference_latency` (Table 7).

## 2. Deadline-adherent edge AI — IET Intelligent Transport Systems, doi 10.1049/itr2.70135

- Platform: **Jetson Orin Nano with TensorRT / INT8**; 423 frames.
- Soft deadline: **150 ms**. Mean end-to-end: **133.03 ms**. Max: **180.17 ms**
  (recorded as a maximum, NOT called WCET unless the source does).
- Verdicts: `conversion_to_cycles: forbidden`, `eligible_for_t_ref: false`.
- Binds: `perception_execution_distribution_platform2` (the 133.03 ms mean) and
  `perception_stage_soft_deadline` (the 150 ms requirement, with 180.17 ms as the
  observed maximum).

## 3. xi = 300 cycles/bit — attribution correction

The historical repository comment credits "Zhao2018 Table I". The source that
actually states it is **Hu et al., accepted manuscript of the paper later published
in IEEE Transactions on Wireless Communications (2018)**, Table I:
**"Required CPU cycles per bit = 300 cycles/bit"**.

- The registry must name **Hu et al.**, and the incorrect Zhao attribution in the
  repository comment must be recorded as corrected rather than repeated.
- Label: **simulation parameter**, not a measured automotive constant. It is a
  modelling choice used for MEC offloading simulation, so it closes provenance but
  must not be presented as measured hardware behaviour.
- Closes `cycles_per_bit_conversion` (the last blocked required parameter), so
  `blocked_required` should reach 0 in this batch.

## 4. WATERS 2015 (kept for distribution only)

Table IV gives min/avg/max ACET in microseconds and Table V the WCET scaling
factors; the paper states the values assume a multicore architecture and are
scaleable. Use for realistic task-time distributions and task-set structure ONLY:
not for semantic stage naming and not for a direct conversion into CPU cycles of a
specific machine.

## 5. Expected gate state after this batch

```
blocked_required                    = 0
execution_evidence_verified          = 4/4
execution_rows_eligible_for_t_ref    = 0/4    # acceptable, not a failure
missing_required                     = 18
violations                           = 0
stale                                = false
```

Reference-tier lead for the next step (not closed here): a DATE 2021 work measures
the Autoware.Auto perception stack on an **Intel Core i5-3210M** and traces the
classifier/object-detection/planning chain; the numbers live inside Figure 2 and
must be read from the figure, not guessed. That is a CPU-compatible candidate for
`reference_execution_tier`, unlike the Jetson rows.
