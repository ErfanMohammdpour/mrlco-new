# V2_ENERGY_SPEC - physical energy for the v2 system model

Generated from the LIVE frozen configuration by `python3 -m spec.automotive_training.v2.benchmark_v2`.

## Model

```
C_executed = executed_bytes * 8 * cycles_per_bit
P_cpu(tier) = kappa_tier * f_tier^3
E_cpu = P_cpu(tier) * cpu_service_seconds   (= integral P dt)
E_tx  = P_tx(transmitter) * ACTIVE transfer service
```

Primary scope: **system** (frozen). RX energy: OFF.

## Parameters

| parameter | value | unit | source |
|---|---|---|---|
| kappa_helper | 5e-27 | J/(cycle*Hz^2) | Liu et al., Digital Communications and Networks 9(6):1399-1410, 2023, Table 1 (F_j in [1,2] GHz, gamma_j = 5e-27); that paper's objective contains the service-vehicle compute energy (Eq. 15). |
| f_helper | 1500000000.0 | Hz | frozen tier specification |
| P_helper | 16.875 | W | derived: kappa * f^3 |
| kappa_mec | 1e-27 | J/(cycle*Hz^2) | Liu et al., DCN 9(6), 2023, Table 1 (F_m = 10 GHz, gamma_m = 1e-27); frequency double-sourced with Zhao et al. arXiv:1807.02311 Table I. |
| f_mec | 10000000000.0 | Hz | frozen tier specification |
| P_mec | 1000.0000000000001 | W | derived: kappa * f^3 |
| kappa_ue | 1e-27 | J/(cycle*Hz^2) | chosen inside the verified band (kappa 1e-28..1e-26, f 0.2..2.5 GHz); kappa=1e-27 matches the Gu2025/Liang2024 vehicle tier. NOT imported from Liu2023, which is fully offloaded and has no UE-local tier. |
| f_ue | 1000000000.0 | Hz | frozen tier specification |
| P_ue | 1.0 | W | derived: kappa * f^3 |
| ue_tx_w | 1.0 | W | frozen radio configuration |
| mec_tx_w | 3.162 | W | frozen radio configuration |
| helper_tx_w | 1.0 | W | frozen radio configuration |
| ue_rx_w | None | W | frozen radio configuration |
| helper_rx_w | None | W | frozen radio configuration |
| cycles_per_bit | 300.0 | cycles/bit | frozen task data model |

## Boundaries

* `E_requester` = UE CPU + UE radio
* `E_mobile` = requester + helper CPU + helper radio
* `E_system` = mobile + MEC compute + MEC TX  (primary)

## Explicitly OUT of scope (declared, never reported as a measured zero)

* receiver/static radio power (include_rx_energy=False in the frozen config)
* CPU idle/leakage power between tasks
* MEC/RSU static (non-compute) power: server_static_power_w is null
* backhaul / core-network transport energy
* battery state dynamics, charging efficiency and reserve accounting
* warm-standby reservation energy (the reservation costs no joules in this model)

## Known modelling note

The frozen rate table is not the physical rate implied by `f` and `cycles_per_bit`. The ledger therefore uses the duration form of the SAME allocation the scheduler executed, and reports the work form and the ratio as provenance rather than mixing two inconsistent allocations.

