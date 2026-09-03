# Toy oracle suite

Spec-faithful hand cases for Phase 0. Production simulator MUST match these after Phase 1.

Constants: `spec/frozen_experiment.yaml` `resource_rates` and `power`.

Exact convenient sizes:

- compute `1048576` bytes = 1.0 s on UE/HELPER, 0.1 s on MEC
- edge / output `458752` bytes = 0.5 s MEC UL/DL, 0.7 s V2V

## Hand checks (independent of checker)

### 01 all-local chain

UE CPU serial 1+1+1 s. Same-location edges 0. Sink already at UE. `makespan=3`, `E=3` J local.

### 04 UE → MEC

T1 UE `[0,1)`. Edge UL `[1.0,1.5)`. T2 MEC `[1.5,1.6)`. Sink DL `[1.6,2.1)`.
`makespan=2.1`
`E = 1.0 + 0.5*0.1 + 0.5*0.05 = 1.075` J

### 09 HELPER → MEC

Root V2V 1048576 B = 1.6 s. Helper CPU 1.0 s. Edge two-hop V2V 0.7 + UL 0.5. MEC 0.1. Sink DL 0.5.
`makespan=4.4`
`E=0.982` J (mobile)

## Checker

```text
python3 spec/toy_oracles/oracle_checker.py
```
