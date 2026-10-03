# CURRENT_STATE (v2 program start)

## Git

| field | value |
|---|---|
| local branch | `phase4-eval` |
| local HEAD | `92212d1de5684b238eefc6c30f90fc6cca0fa14e` |
| local status | clean except two NEW untracked audit files (see below) |
| Kish checkout | `/opt/margo/mrlco-new-6b`, branch `phase4-eval`, HEAD `92212d1d`, dirty=0 |
| remote | `erfan/phase4-eval` at `92212d1d` |

Untracked at audit time (created by the previous round, never committed):
`spec/automotive_training/geometry_sensitivity.py`,
`spec/automotive_training/reports/gateI/geometry_sensitivity.json`.

## Runtime

| field | value |
|---|---|
| MARGO processes on Kish | 0 |
| MARGO containers | 0 |
| GPU | 21508 MiB used / 2574 MiB free (vLLM owner only) |
| vLLM | alive (1 process), untouched |
| `meta_test_access_count` | **0** |

## v1 inputs (verified unchanged, sha256 recomputed locally)

| artifact | pinned = actual |
|---|---|
| `graphs.jsonl` | `4341c3b65c6e07b7fea341e2f3c3947723c6ebe01b5eb115dbf51c8c7f38404f` |
| `splits.jsonl` | `cd391da6ac12ef12e0f552f11f3c4bb551df50592a5a9753b08873d37c912907` |
| `dataset_manifest.jsonl` | `a190e9dbb5599803fd5b9457e3e7648964849a1a7078fd0b82362150757b9e14` |
| `calibration_report.json` | `3d5db9414b21c7774d0d5e9e8807ffe2f99252d30a0263121c8156b5f3ae7cd7` |

## Staged execution plan (stage = one reviewable commit)

| stage | content | gate |
|---|---|---|
| 1 | v1 geometry/rate-provenance corrections + errata + this state file | no v1 change; `meta_test=0` |
| 2 | v2 shared multi-DAG MEC scheduler (queueing, calendars, invariants) | scheduler invariants + N=1 parity |
| 3 | v2 stochastic/mobility-aware links (estimated vs realized rates) | determinism + regime separation |
| 4 | v2 helper availability/contact/busy model | helper conditional usefulness |
| 5 | v2 criticality-aware reliability (+ minimal local fallback hooks) | HIGH not hard-local; epsilon provenance |
| 6 | geometry gate + stronger-search reference | PASS/FAIL per section 14 |
| 7 | CRN evaluator (Gumbel common random numbers) | bit-identical CRN tests |
| 8 | repeated/adversarial test battery + v1 parity report | 3 fresh-process reps |
| 9 | v2 1x500 training + checkpoint evaluation | CRN protocol frozen |

Any gate failure stops the program with a BLOCK report (no tuning around it).
