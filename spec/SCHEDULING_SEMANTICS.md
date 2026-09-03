# Scheduling Semantics

Status: Draft for Phase 0  
Related ADRs:

- `ADR-002-data-residency.md`
- `ADR-003-resource-capacity.md`

## 1. Core rule

For every edge `i -> j`:

`start(j) >= data_available_at_execution_location(i, j)`

`data_available_at_execution_location(i, j)` equals predecessor finish time plus any required communication route delay from predecessor output location to successor execution location.

## 2. Data residency rule

`output_location(task) = execution_location(task)`

Locations:

- `UE`
- `MEC`
- `HELPER`

## 3. Communication matrix

| predecessor output | successor execution | route | network delay basis |
| --- | --- | --- | --- |
| UE | UE | none | `0` |
| UE | MEC | `UE -> MEC` via `MEC_UL` | `edge_output_bytes / mec_uplink_rate` |
| UE | HELPER | `UE -> HELPER` via `V2V_CHANNEL` | `edge_output_bytes / v2v_rate` |
| MEC | UE | `MEC -> UE` via `MEC_DL` | `edge_output_bytes / mec_downlink_rate` |
| MEC | MEC | none | `0` |
| MEC | HELPER | `MEC -> UE -> HELPER` via `MEC_DL` then `V2V_CHANNEL` | sum of both hops |
| HELPER | UE | `HELPER -> UE` via `V2V_CHANNEL` | `edge_output_bytes / v2v_rate` |
| HELPER | MEC | `HELPER -> UE -> MEC` via `V2V_CHANNEL` then `MEC_UL` | sum of both hops |
| HELPER | HELPER | none in v0.1 single-helper model | `0` |

## 4. Resource reservation rules

Single-capacity resources:

- `UE_CPU`
- `MEC_UL`
- `MEC_CPU`
- `MEC_DL`
- `HELPER_CPU`
- `V2V_CHANNEL`

Rules:

- all compute tasks are non-preemptive
- `MEC_UL`, `MEC_DL`, and `V2V_CHANNEL` reserve closed-open intervals `[start, end)`
- `V2V_CHANNEL` is half-duplex: no overlapping transmit or receive intervals
- helper compute MAY overlap with communication because `HELPER_CPU` and `V2V_CHANNEL` are distinct resources
- MEC compute MAY overlap with MEC downlink of other tasks only if both resources are distinct and capacities permit; in v0.1 both are distinct single-capacity resources

## 5. Execution semantics by action

### Local

- root external input is already at `UE`
- non-root task consumes predecessor edge outputs that have arrived at `UE`
- wait for `UE_CPU` availability
- wait for all predecessor outputs to become available at `UE`
- reserve `UE_CPU`
- finish location = `UE`

### MEC

- root external input for task resides at `UE`
- predecessor outputs required by task MUST become available at `MEC`
- reserve predecessor-transfer routes first
- reserve `MEC_UL` only for root or intrinsic `external_input_bytes` originating at `UE`
- reserve `MEC_CPU`
- do NOT automatically reserve `MEC_DL` after every MEC task
- finish location = `MEC`

For v0.1, task output of MEC execution stays at `MEC`. Downlink to `UE` occurs only when a successor route or terminal sink-return rule requires it.

### HELPER

- root external input for task resides at `UE`
- predecessor outputs required by task MUST become available at `HELPER`
- reserve predecessor-transfer routes first
- reserve `V2V_CHANNEL` for root or intrinsic `external_input_bytes` that originate at `UE`
- reserve `HELPER_CPU`
- task output stays at `HELPER`
- no automatic return to `UE` unless a successor route requires it

## 6. Important modeling consequences

- same-location successor consumes zero network delay
- V2V -> MEC dependency is NOT satisfied until helper output reaches `MEC` through declared two-hop route
- MEC -> HELPER dependency is NOT satisfied until MEC output reaches helper through declared two-hop route
- task-global `compute_workload_bytes` MUST NOT replace `edge_output_bytes` for predecessor-successor communication
- sink completion for application-level latency occurs only after sink result reaches `UE`

## 7. Scheduling invariants

1. Precedence:
   for each edge `i -> j`, successor starts no earlier than required input availability at successor location.
2. Exclusivity:
   no overlap on any single-capacity resource.
3. Route completeness:
   every cross-location dependency uses a declared route.
4. Zero-transfer correctness:
   same-location dependency incurs zero network transfer.

## 8. Oracle examples required

Toy examples in `spec/toy_graphs/` MUST include:

- `chain_local_mec_helper`
- `fork_helper_and_mec`
- `join_mixed_locations`

Expected intervals and finish times MUST be recorded in `toy_expected_results.json`.
