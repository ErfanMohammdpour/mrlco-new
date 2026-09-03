# Objective And Energy Contract

Status: Draft for Phase 0  
Related ADR: `ADR-001-energy-scope.md`

## 1. Energy scope decision

Primary optimization scope for `MARGO-SPEC-v0.1`:

- objective energy = `total_mobile_joules`
- where `total_mobile_joules = total_ue_joules + helper_compute_joules + helper_v2v_txrx_joules`

Rationale:

- UE-only energy is too narrow for cooperative offloading claims.
- Full MEC datacenter energy is not credibly modeled by current repository.
- `UE + HELPER` keeps battery-powered devices in scope without pretending to model datacenter power.

MEC compute energy MUST be logged separately as optional accounting, but it is NOT part of primary objective in v0.1.

## 2. Required logged energy components

Per task and per episode, implementation MUST expose:

- `ue_local_cpu_joules`
- `ue_mec_uplink_joules`
- `ue_mec_downlink_joules`
- `ue_v2v_tx_joules`
- `ue_v2v_rx_joules`
- `helper_compute_joules`
- `helper_v2v_tx_joules`
- `helper_v2v_rx_joules`
- `mec_compute_joules_optional`

Derived aggregates:

- `total_ue_joules`
- `total_helper_joules`
- `total_mobile_joules`
- `total_system_joules_optional`

## 3. Payer rules

- Local execution energy payer: `UE`
- MEC uplink/downlink radio energy payer: `UE`
- V2V transmission/reception energy payer: endpoint that transmits or receives
- Helper compute energy payer: `HELPER`
- MEC compute energy payer: `MEC` / infrastructure

## 4. Reward and reporting separation

Scientific metrics MUST remain separate from reward shaping.

Reported scientific metrics:

- `makespan_seconds`
- `total_mobile_joules`
- `total_ue_joules`
- `total_helper_joules`

Composite reporting metric for `MARGO-SPEC-v0.1`:

`J_report = 0.5 * L_norm + 0.5 * E_norm`

This weight choice is frozen for v0.1 to match current published narrative and avoid post-hoc tuning.

## 5. Normalization contract

Normalization ranges MUST come only from frozen training metadata or from analytically declared reference plans.

Validation/test data MUST NOT define normalization ranges.

For v0.1 reporting, use episode-local reference ranges derived from:

- all-`UE` plan
- all-`MEC` plan
- all-`HELPER` plan

Define:

- `L_ref_min = min(L_UE, L_MEC, L_HELPER)`
- `L_ref_max = max(L_UE, L_MEC, L_HELPER)`
- `E_ref_min = min(E_UE, E_MEC, E_HELPER)`
- `E_ref_max = max(E_UE, E_MEC, E_HELPER)`

Then:

- `L_norm_raw = (L - L_ref_min) / max(L_ref_max - L_ref_min, 1e-12)`
- `E_norm_raw = (E - E_ref_min) / max(E_ref_max - E_ref_min, 1e-12)`
- `L_norm = clip(L_norm_raw, 0.0, 1.0)`
- `E_norm = clip(E_norm_raw, 0.0, 1.0)`

Bounds MUST be documented with:

- source
- unit
- validity assumptions
- failure behavior when out of range

Out-of-range behavior for v0.1: `clip_and_log`.

## 6. Training reward contract

Environment model is macro-action: decoder emits full plan `a_1..a_N`, then one schedule call returns episode metrics.

Token-level training reward for v0.1 MUST use **post-hoc telescoping with completion policy `all_UE`**:

1. Let `fill(a_{t+1}..a_N) = UE` for every undecided task.
2. Define provisional plan `P_t = (a_1..a_t) + fill(a_{t+1}..a_N)` for `t = 0..N`.
   - `P_0` is the all-`UE` plan.
3. Schedule each `P_t` once under frozen scheduling semantics → `(L_t, E_t)` where `L` is makespan and `E` is `total_mobile_joules`.
4. Reference scales from pure-location plans on the same graph:
   - `L_scale = max(L_ref_max - L_ref_min, 1e-12)`
   - `E_scale = max(E_ref_max - E_ref_min, 1e-12)`
5. Incremental terms:
   - `delta_latency_t = L_t - L_{t-1}`
   - `delta_energy_t = E_t - E_{t-1}`
6. Token reward:
   - `r_t = -(0.5 * delta_latency_t / L_scale + 0.5 * delta_energy_t / E_scale)` for `t = 1..N`

Properties:

- telescoping: `sum_t delta_latency_t = L_N - L_0` and likewise for energy
- `delta_*` MAY be negative (a token can improve the provisional plan vs all-UE fill)
- no online env step after each decoder token
- `J_report` remains episode-level scientific metric on the final plan `P_N`

Forbidden without a new spec version: terminal-only reward claiming to implement this `r_t` formula; prefix schedule without the declared `all_UE` fill.

## 7. Current repository defects this contract addresses

- inconsistent energy scope
- MEC compute omitted while helper compute included
- ambiguous min-energy bound
- reward normalization mixed with scientific metric interpretation

## 8. Phase 0 acceptance examples

At least three hand-worked cases MUST be recorded:

1. all-local chain
2. local -> MEC dependency
3. V2V -> MEC dependency

For each case, expected joule totals MUST be manually derivable from declared powers and durations.
