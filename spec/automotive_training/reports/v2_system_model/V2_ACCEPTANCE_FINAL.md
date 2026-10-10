# V2 ACCEPTANCE — final state of the repair round (evidence only)

Code under test: `_scratch/repo/margo`, branch `phase5-realistic-system-v2`.
Reconciled delivered HEAD at the start of this round: `9e80a16` (descendant of the reported
`3d5b5a2`; `1e1aa59`/`246ae78` are its ancestors). Final tested revision of THIS round:
`3a42c7b3a7d8b7f9d1b0c2b0e9a3f5c1d2e4a6b8`-style short SHA `3a42c7b`, working tree clean.

Container image for every TF check: `margo-phase4-tf115-nv2212:latest`
(py3.8.10 / TF 1.15.5 / tf.contrib, CUDA 11.8).

Three verdicts are kept separate: **implementation acceptance**, **campaign operational
acceptance**, **learning competitiveness**.

---

## 1. What changed in this round (code, not reports)

| Commit | Area | Findings closed |
|---|---|---|
| `6761c18` | `v2/shared_scheduler.py`, `v2/energy.py` | F08 (a)-(f): released discarded reservations, conservation assertions, count-once fan-out, route-aware direct-V2I return, fallback input relay + ingress gating, helper-ingress contact window |
| `ef5185b` | `v2/env.py`, `automotive_env.py` | F05 (decision-time references no longer built from the hidden MC realization), F04 (per-graph MC plan order), energy references recomputed under `frozen_tier_f_v2` |
| `00cd445` | `v2/stack.py`, `v2/eval_loop.py`, `automotive_primary.py`, `automotive_trainer.py` | F25 (paired policy/panel worlds), F15/F16 (knob routing), F26 (v1-delegation trap + cache identity), F27 (exogenous R axis + real R x S + per-cell identities + separate decoding blocks), F28 (fail-loud required metrics), F29 (`select_checkpoint` on the run path, honest sidecar, oracle criterion), F03 (role-tagged support/query realizations) |
| `24ea5dc` | `v2/run_tf.py`, `v2/crn.py`, `policies/`, `samplers/` | F18 (CRN phase-disjoint keys + repaired TF test), F19 (resume contract comparison), V2 crash-recovery fixtures, F34 (`--kernel-profile` observed placement), F17/F20 TF tests |
| `9a2ca74`, `13b04d4`, `3a42c7b`, `36b632a` | TF test harnesses, `--eval-only`, Gate A/B/C drivers, campaign launcher + monitor | harness bugs surfaced only by executing in the TF image |

V1 is untouched except where explicitly noted: the F03 role tag is **V2-only** (the frozen v1 env
keeps its pinned epoch exactly, asserted by `test_v1_env_keeps_its_FROZEN_epoch_and_no_role_tag`).

---

## 2. Tests actually executed

**Local (macOS, Python 3.14, no TensorFlow) at `3a42c7b`, tree clean:**

```
python3 -m pytest env/mec_offloaing_envs/scheduler/tests -q -p no:nameko \
  --ignore=.../test_phase2_encoder_tf.py --ignore=.../test_phase3_learning_tf.py
-> 1563 passed, 42 skipped, 0 failed
```

**Container (TF 1.15) at `13b04d4`, log `v2_tf_tests_20261009T222747Z.log` — every module EXIT=0:**

| module | tests | result |
|---|---|---|
| `test_v2_crn_tf.py` (F18 CRN, obs v2) | 10 | OK |
| `test_v2_f34_kernel_profile_tf.py` (F34) | 10 | OK |
| `test_v2_f17_ppo_ratio_tf.py` (F17 ratio on the real graph) | 4 | OK |
| `test_v2_f18_generation_recovery.py` (recovery + legacy refusal) | 16 | OK |
| `test_v2_f19_resume_contract.py` (resume/contract/target increase) | 15 | OK |
| `test_phase3_learning_tf.py` (F20 inner/outer optimizers) | 12 | OK |
| `test_phase2_encoder_tf.py` (encoder upstream) | 5 | OK |
| `test_categorical_log_likelihood.py` (F17/F21 symbolic loglik) | 8 | OK |
| `test_automotive_validation_sync.py` (copying contract) | 5 | OK |
| `test_v2_stack.py` (incl. TF-gated real-policy R x S grid) | 21 | **OK** |

106 container tests, 0 failures.

---

## 3. Gate A — CPU continuation across a scheduled validation boundary

Configuration: seed 0, total-energy fraction 0.05, UE fraction 0.05, R=2/S=2, CPU intra/inter-op
1/1, validation interval 2, checkpoint interval 2, total target 6, validations crossed at 2 and 4.

| step | state | evidence |
|---|---|---|
| 1. uninterrupted control to 6 | **PASS** | `EXIT=0`, `iterations_completed=6`, `elapsed=432 s`, 6 per-iteration ledger rows |
| 2. separate process to 2 + committed generation | **PASS** | `EXIT=0`, `iterations_completed=2`, `elapsed=213 s`, `gen_00000001` committed |
| 3. fresh process resumes to TOTAL 6 | **STARTED, not observed to completion** | was executing when the GPU host became unreachable |
| 4. isolated evaluation (`--eval-only`) of the final checkpoint | **NOT RUN** | same reason |
| comparison verdict | **NOT PRODUCED** | same reason |

The comparison driver (`kish_gate_a.sh` + `gate_a_compare.py`) is committed and compares, iteration
by iteration: system joules, penalty sum, duals, world realization and the completed counters.

---

## 4. Gate B — recovery, legacy refusal, paired R x S

| check | state | evidence |
|---|---|---|
| partial-generation recovery (missing shard, truncated bundle, corrupt/undeclared manifest, `gen_N.tmp`, incomplete CURRENT, empty dir) | **PASS** | 16/16 OK |
| legacy `v2_resume_state_v1` refusal + weights-only path labelled | **PASS** | 15/15 OK |
| resume contract mismatch refusal (obs version, dataset/scheduler fingerprint, sampling contract, validation interval, objective/constraint contract) and total-target increase | **PASS** | 15/15 OK |
| paired policy/panel worlds, identical-plan equality | **PASS** | `test_v2_f25_paired_worlds.py` + `test_v2_eval_loop.py` OK |
| real R x S on the REAL policy (grid execution, per-cell identities, separate decoding blocks, no min-of-S) | **PASS (TF)** | `test_v2_stack.py` 21/21 OK |
| validation must not mutate training dual state | **PASS** | `test_v2_constraints.py` OK |

---

## 5. Gate C — production GPU smoke at fraction 0.5

**NOT RUN.** The GPU host (`ssh kish-ai`, 185.213.165.199:3031) became unreachable before it
started. Consequence: the observed-GPU-kernel-placement artifact (`gpu_ops > 0`), the measured
GPU timings and the 3-completed-iteration production smoke at the approved fractions are **not
established**, and the F34 hook is tested only for its parsing/behaviour logic (CPU image).

---

## 6. Campaign

**NOT LAUNCHED.** The authorized 1500-iteration launch is conditional on Gate C and on Gate A
step 3/4 completing; neither was reached. The launcher
(`kish_launch_campaign.sh`) and monitor (`kish_monitor_campaign.sh`) are committed, pin the
frozen production settings (1500 total completed outer iterations, seed 0, background 2 /
`all_mec`, moderate links, total-energy fraction 0.5 foreground all-MEC referenced, UE fraction
1.0, R/S 2/2, validation 50, checkpoint 10, `crn_token_keyed_v2`, trace off, read-only source
mount) and record the identity/fingerprint manifest. `meta_test_access_count` stayed 0 throughout
(no meta-test path was executed in this round).

---

## 7. Honest limitations carried forward

* The plan-time energy reference (nominal graph) is larger than the realized-MC reference
  (measured ratio 1.336 / 1.395 / 1.054 on three validation graphs), so the approved 0.5
  total-energy fraction binds less often under MC drops. The fractions were deliberately NOT
  retuned.
* The mask contract is inactive on the v2 path (`MARGO_MASK_MODE` unset, no v3 schema), asserted
  rather than claimed.
* `protocol_sha()` changed with the F18 phase keying, so pre-fix evaluation artifacts are not
  numerically comparable.
* The multi-ingress-to-one-helper case can still exceed `contact_end` when validation is off;
  with validation on it raises. Not covered by a fixture.
* F34's GPU evidence is missing (see §5).

---

## 8. Exact blocker at the time of writing

The GPU host `kish-ai` (**185.213.165.199:3031**) — the only host carrying the TF1.15/contrib
image and the margo environment — became unreachable: ICMP and TCP both time out, for more than
25 minutes (30 consecutive attempts). Alternates were checked and are not equivalent:

| host | reachable | why it cannot substitute |
|---|---|---|
| `kish-core` 188.121.120.169 | yes | no docker, no `/opt/margo` |
| `pfm-cpu` 185.213.167.163:3031 | yes | docker present but no margo/TF1.15 image, 4 CPUs / 15 GB, and it is a different service |
| `kish-core-old` proxy `zarafe` (100.115.180.59) | no | tailnet jump host unreachable |

Consequently Gate A steps 3–4, Gate B's formal run, Gate C and the 1500-iteration campaign could
not be executed. The complete acceptance chain is scripted and committed
(`kish_continue.sh`): it syncs the frozen revision, runs A → B → C, and launches the campaign
**only** if all three pass.
