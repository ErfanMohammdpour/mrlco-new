# v2 stage log (per-stage evidence)

| stage | commit | tests run | result | files | meta_test |
|---|---|---|---|---|---|
| 1 v1 geometry/rate audit + errata | `fdfc7a2` | `build_v1_geometry_audit.py` (160 graphs) | OK | `reports/v1_geometry_audit/*`, `geometry_sensitivity.py`, `reports/gateI/geometry_sensitivity.json` | 0 |
| 2 shared multi-DAG MEC scheduler | `596b885` | `test_v2_shared_scheduler.py` x3 fresh procs = 9/9 each; full non-TF suite 1183 passed / 0 failed | OK | `v2/{__init__,shared_scheduler,adapters}.py`, `tests/test_v2_shared_scheduler.py` | 0 |
| 3 dynamic links (estimated vs realized) | `PENDING-COMMIT` | `test_v2_link_model.py` + `test_v2_shared_scheduler.py` = 17/17 x3 fresh procs; full non-TF 1191 passed / 0 failed | OK | `v2/{link_model.py,link_regimes.yaml}`, scheduler link-process hook, `tests/test_v2_link_model.py` | 0 |
| 4 helper availability/contact | pending | - | - | - | 0 |
| 5 criticality reliability + fallback hooks | pending | - | - | - | 0 |
| 6 geometry gate + stronger search | pending | - | - | - | 0 |
| 7 CRN evaluator (Gumbel) | pending | - | - | - | 0 |
| 8 repeated/adversarial/v1-parity battery | pending | - | - | - | 0 |
| 9 v2 1x500 + checkpoint eval | BLOCKED until gate 6 passes | - | - | - | 0 |

## v1 corrections delivered in stage 1 (see V1_ERRATA.md)

1. `P(all-MEC candidate winner)` = **0.400 (validation) / 0.350 (meta-train)**; mixed = 0.600 / 0.650. An earlier sentence had this inverted.
2. "oracle" renamed to **candidate-panel oracle** (5 candidates); headroom = **candidate-panel headroom**, not an upper bound.
3. No bits/bytes unit error: conversion is exactly 8.0 on every link (160 graphs). The 7-11 Mbps figures are the documented historical/degraded points; per-graph realized medians are 18.5 / 22.5 / 10.6 Mbps (ul/dl/v2v), profile-stratified in `V1_RATE_PROVENANCE.csv`.
4. `mec_share_N` rows relabelled **SURROGATE CONTENTION SENSITIVITY** (processor sharing), not a multi-user simulation.
5. Earlier "16x unit error" and the single-graph rate quote are retracted in the errata.
