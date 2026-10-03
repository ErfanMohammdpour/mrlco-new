# V2_FINAL_ACCEPTANCE — MARGO v2 repair program, honest status

Branch `phase5-realistic-system-v2`, this round's commits on top of `c6ad98a`:

| commit | subject |
|---|---|
| `65e1a5f` | fix(v2): resolve the failing helper test — single-reservation booking, contact/fallback semantics |
| `ff3997e` | feat(v2): real event-based energy, budget/dual constraints, one canonical world builder |
| `d55d5bc` | feat(v2): constraint integration tests — real reward penalty, signed duals both ways, fail-fast |
| *(final)* | feat(v2): canonical-builder gate/parity, future-blindness tests, benchmark, artifacts |

Verification environment for every number below (there is no GPU and **no TensorFlow** in it):

```
python3 -m pytest env/mec_offloaing_envs/scheduler/tests -q -p no:nameko \
  --ignore=.../test_phase2_encoder_tf.py --ignore=.../test_phase3_learning_tf.py
=> 1362 passed, 0 failed, 19 skipped  (all 19 skips are "requires TensorFlow"; see §Skips)
```

## Acceptance table

| # | criterion | verdict | evidence |
|---|---|---|---|
| 1 | previously failing helper test diagnosed and resolved with regression coverage | **PASS** | root cause measured: 46 reservations for 24 transfers, 6.789 s booked vs 2.401 s served. Fixed by `Calendar.plan_service` + `reserve_at` (one booking per transfer, non-mutating fixed point, explicit convergence/failure). `test_v2_booking.py` (15 tests) + `test_v2_contact_and_fallback.py` (12). Helper plan 8.2869 s → **3.9001 s** vs MEC 7.300 s |
| 2 | transfer scheduling satisfies booked-start/service/capacity invariants across adversarial fixtures | **PASS** | `test_v2_booking.py`: no transfer booked twice; occupancy == active service + outage on every ledger record (also a hard `validate_schedule` invariant); no overlapping reservations; capacity never exceeded; queue-delayed start inside an outage; two rate/outage boundaries integrated (0.45 s service + 0.2 s outage over 8 MB); zero-service period; long payload; horizon exhaustion raises; multi-hop/reverse routes; same-location transfers price zero radio energy |
| 3 | shared-world behaviour and foreground accounting exercised in the actual trainer path | **PASS** | `v2/world.py::build_world` is the single builder used by `V2AutomotiveEnv`, the geometry gate and the parity study; `_slot_specs` (the divergent second path) is removed and now raises. Background DAGs of other owners share the MEC CPU/radio in the same call, are workload-derived with distinct identities/seeds, and the episode latency is the foreground completion while `world_makespan_s` is reported separately |
| 4 | future-blindness covers observation, admission, masks, confidence and decision-time policy behaviour | **PASS** | `test_v2_future_blindness.py`: two link processes with identical observable history at t_now=0 and different hidden futures produce bit-identical observations, identical `v2_context`, identical confidence, identical admission evidence and identical un-penalised decision-time rewards, while `confidence_vs_truth` (diagnostic) differs and is provably absent from the observation; the past-window statistic is unchanged when every post-`t_now` step is flipped |
| 5 | real energy, nonzero in nontrivial rollouts, agrees with hand calculations; scopes sum; failure/fallback accounted | **PASS** | `test_v2_energy.py` (17 tests). Hand-checked from the frozen constants: all-MEC 600 J CPU + 0.3162 J MEC TX + 5e-5 J UE UL = 600.31625 J; all-UE 6 J; all-HELPER 1.5·16.875 + 0.2001 = 25.5126 J; contact failure = 1.5·16.875 + 6·1.0 + 1e-4 = 31.3126 J (remote attempt AND local restart both paid). Boundaries nested requester ≤ mobile ≤ system; foreground+background == total component-by-component. Environment telemetry now reports real joules (`energy_constraint` no longer `not_configured`) |
| 6 | budgets influence actual penalties/costs; duals update both ways; broadcast to every environment | **PASS** | `test_v2_constraints.py` (10 tests). An over-budget rollout is measured as violated, the trainer-side dual step makes λ>0, and the *same* rollout then has its terminal reward reduced by exactly the penalty (non-terminal tokens bit-identical); a within-budget rollout is not penalised. λ rises over-budget and **falls** below budget (signed ascent). λ broadcast to a second environment; state round-trips and rejects a spec mismatch. Missing/non-finite metric or zero denominator raises; `select_checkpoint` never labels an infeasible checkpoint feasible |
| 7 | new observations reach training, adapted policies, validation and restored checkpoints under one schema | **PASS (numpy) / NOT RUN (TF)** | ONE leaf contract (`v2/context_fields.py`) is read by BOTH the environment and the frozen encoder — the duplicated 12-name list is gone and nothing hardcodes 52/91; the encoder refuses to activate the v2 version if the contract cannot be resolved. The context grew 12 → 30 decision-time entries (packed row 91 → **109**): estimate age per link, **per-node** ε (min/mean/max), decision-time queues and system-scope energies of the three pure-location reference plans, the active budget ratio, the broadcast duals, the round-trip contact slack and the competitor load; `helper_busy_s` replaces the mislabelled busy "fraction" (a real fraction accessor is provided). `test_v2_observation_v2.py` (14): width arithmetic from one list, the encoder reading the same list, a world-feature perturbation reaching the declared tensor column **and only that column**, v1-prefix preservation, repeated-graph and nonzero-index slot identity, the new channels, and decision-time-only reference waits/energies. **NOT RUN:** the TF encoder consumption and any gradient check through the context columns |
| 8 | PPO and the MRLCO outer update execute correctly with independent support/query and audited reset behaviour | **PARTIAL: PASS (numpy contract + static guards) / NOT RUN (TF)** | `test_v2_ppo_contract.py` (8) EXECUTES a numpy reference of `likelihood_ratio_sym` and checks the audited properties: the ratio is `p_new(a)/p_old(a)` at the **stored** action, is exactly 1 for unchanged logits, discriminates against a re-sampled action, and is per token rather than collapsed. Source guards (REVIEWED, not executed) assert: the ratio consumes `self.actions` bound to `decoder_targets` and the stored old logits; `UpdatePPOTargetPerTask` calls `reset_inner_optimizer(task_id)` before adapting; `meta_trainer` calls `sync_task_policies_from_core()` at the TOP of every iteration; the outer step is `mean_pseudogradient(...)` with no `tf.gradients` anywhere, so it is a declared FIRST-ORDER update and is not called MAML. **NOT RUN:** any execution of the TF graph, support/query independence in a run, inner/outer update magnitudes, checkpoint contents and resume. See `V2_PPO_META_AUDIT.md` |
| 9 | R/S evaluation executes on v2; baselines use the same world/scheduler/objective; CRN reproducible | **PASS with a declared policy substitution** | NEW `v2/eval_loop.py` executes the frozen `automotive_crn_gumbel_v1` protocol with BOTH axes: R environmental realizations (r-identity-keyed worlds) × S policy samples (Gumbel noise keyed by protocol/r/s/graph). Executed at R=5, S=5 → **1000 indexed cells** over 8 graphs and 5 candidates, written to `V2_CRN_EVALUATION.json` with per-cell `r`/`s`, nested means/stderr and paired deltas. Baselines (`all_ue`/`all_mec`/`all_helper`, latency-aware `greedy_cd`) and the stochastic candidate all score on the SAME world object per replicate (asserted via `structure_fingerprint_sha256`: a candidate change never regenerates the world). Aggregation is NESTED, never best-of-S. **Declared substitution:** the TF PPO policy is replaced by a numpy reference policy over the same action space — the protocol/world/scheduler/objective are the real v2 ones, the *trained* policy remains NOT RUN |
| 10 | required non-TF and TF suites have no unexplained failures; every skip has a reason | **PASS (non-TF) / NOT RUN (TF)** | non-TF: 1362 passed / 0 failed / 19 skipped, every skip carries an explicit "requires TensorFlow" reason. TF modules are excluded from the non-TF command; a missing required TF test is reported NOT RUN, never PASS |
| 11 | geometry and degenerate v1/v2 parity artifacts regenerated on the final code, honest verdicts | **PASS** | `GEOMETRY_GATE.json` regenerated (20 graphs × 12 regimes, search budget 600) on the canonical builder; verdict **PASS** (all-MEC does not win >80 % everywhere: it wins 100 % only in regime L). Helper evidence now measured from EXECUTED actions plus a no-helper ablation: helpers appear in winning plans in 9/12 regimes (up to 19/20 graphs) with a positive ablation gain (up to 61.5 %). `V1_V2_PARITY.json` regenerated (12 graphs × 6 plans); in the degenerate configuration the max absolute difference is 3.3 % and is reported as a model difference, not as headroom |
| 12 | full smoke, checkpoint/resume and complete-iteration throughput evidence | **FAIL (partial)** | CPU phases benchmarked over complete env iterations (`V2_THROUGHPUT.json`: world build 0.35 s, reset 1.78 s (10 slots × 3 reference worlds), validated schedule 0.123 ms, unvalidated 0.111 ms, `env.step` **58 ms for 10 slots**, energy ledger 3.7 ms/slot, 10-slot CPU iteration ≈ **2.36 s**). A swallowed exception had made every context call rebuild the reference worlds (2.17 s → 0.058 s per step after the fix). The TF smoke chain (support rollout → inner PPO → query → outer update → checkpoint save/restore → resume) was **NOT RUN** |
| 13 | frozen v1 unchanged; no protected service or meta-test boundary disturbed | **PASS with a declared caveat** | `git diff 92212d1d -- scheduler/engine.py spec/automotive_training/automotive_env.py` is **EMPTY**. `automotive_primary.py` carries one additive, default-off v2 commit (`65de171`, `env_factory=None` keeps the exact v1 path) — declared, not hidden. No protected service, vLLM, qdrant or the `/opt/margo/mrlco-new` checkout was touched; meta-test guard reports 0 accesses |
| 14 | diagnostic training status recorded separately, no fabricated results | **NOT STARTED** | No long training was run. It is correctly not started: criterion 7/8/9 gates are unmet and there is no TF/GPU execution capacity in this environment |

## Skips (all 19, with reasons)

Every skip is TensorFlow-related; none hides a failure:
`test_automotive_validation_sync.py` ×5 (requires TF), `test_automotive_long_run_entry.py` ×5
(TF), `test_v2_crn_tf.py` ×4 (TF), `test_phase1_encoders.py` ×3 (tf.contrib), `test_eas_adapt.py`
×1, `test_best_of_k.py` ×1.

## What the reported state got right and what it got wrong

* `3281de2` background world / future-blind confidence and outage evidence — **present and now verified**.
* `515789a` event-based transfers, booked-start rate, pause-on-outage — the *service integral* was
  present, but the **booking was double-reserving** (46 reservations for 24 transfers), which is
  what the failing test actually measured. The recorded diagnosis ("booked channel time ≈ 2.8×
  the served time") was correct; the cause was the two-step booking leaving a provisional
  reservation behind, not shared-radio serialisation.
* `7c75ae6` audit correction — present.
* "1268 passed / 1 failed / 19 skipped" reproduced exactly before the fix.
* "Energy is still a zero-joule placeholder; `env.step` applies no λ penalty; observations,
  evaluator/CRN, final geometry gate and checkpoint/resume verification remain incomplete" —
  **all still true at the start of this round**; energy, the λ penalty, the canonical builder,
  the gate and parity are now repaired, the rest is listed as FAIL/NOT RUN above.

## Remaining limitations (stated, not hidden)

1. The observation contract is complete and single-sourced, but its **TF encoder consumption
   and any gradient check are NOT RUN**; the perturbation evidence is numpy-level. Estimate age
   is 0 at plan time under every shipped regime because the estimator refreshes each decision
   epoch — the channel exists so a stale-estimate regime can populate it.
2. PPO/MRLCO **execution** for v2, checkpoint/resume and the end-to-end smoke are **NOT RUN**;
   the CRN R/S loop DOES execute but with a declared numpy reference policy instead of the
   trained TF policy (`eval_loop.py`), and the PPO ratio/reset contract is verified in numpy and
   by source guards (`V2_PPO_META_AUDIT.md`). TensorFlow is not installed and the frozen chain
   needs TF 1.15 `tf.contrib`, which is unavailable for this interpreter. Exact commands are in
   `V2_RUN_MANIFEST.json`.
3. `kappa_helper`, `kappa_mec` and the TX powers trace to a 2023 source that could **not** be
   re-verified (publisher 403); they are flagged as unverified transplants in
   `V2_ENERGY_SPEC.md`. No accessible 2024–2026 primary source publishes a helper/V2V relay
   energy equation: that tier is a transparent assumption.
4. Reliability is a **surrogate** (`link_confidence × (1 − outage_fraction) × contact_slack`),
   not a calibrated chance constraint; no empirical 1e-5 assurance is claimed.
5. Warm standby reserves local CPU time but is not priced in joules; checkpointing is
   explicitly unsupported (`checkpoint_transfer_bytes == 0`).
6. The geometry gate's `stronger_search` result remains a candidate-search lower bound, never an
   oracle, and the all-HELPER *pure* baseline still never wins — the helper benefit appears in
   **mixed** plans, which is the claim actually supported.
