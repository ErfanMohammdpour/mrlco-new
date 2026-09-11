# ADR-007: Adaptation engine (v0.3)

Status: Accepted  
Decision date: 2026-09-08  
Supersedes: `MARGO-METHOD-v0.2-cavia` §9 (CAVIA-on-z as the method)

## Decision

Inner PPO on θ, Reptile / first-order mean-PG on θ as the **method**, and CAVIA-on-z with a pure `mean T` loss are **removed from the method**. They remain as baselines / negative ablations only.

Replacement: EAS-style support adaptation of a small parameter subset φ with a best-sample imitation anchor (Phase 3, `spec/prompts/PHASE3_EAS_SUPPORT_ADAPTATION.md`).

Fallback if Phase 3 fails its gate (query greedy T 575–582 → ≤520, occupancy Local 0.15–0.25): zero-shot BC + best-of-k (Phases 1–2). Then the paper has no meta-adaptation claim.

## Evidence (numbers copied from `spec/PHASE4_DIAGNOSTIC_LOG.md`)

### PPO on θ collapses occupancy

| run id | log section | number |
|---|---|---|
| `margo_v0.1_diag_500_parallel` | §1 | collapse `max_action_frac≥0.95` from **iter 21**; iter 180 still MEC **0.998**, entropy ~0.01 |
| `margo_v0.1_diag_latency_tmec` | §2 | T **958→796**; **0** plans with 1–2 non-MEC |
| `margo_v0.1_diag_pomo_tmec` | §3 | elite `n_nonMEC` median **9** (min 4, never ≤3); T **881→921** |
| `margo_v0.1_diag_bc_greedy_tmec` | §4 | iter 0 mix ≈ expert; **iter 9** T **931**, Local **55%**, `n_non≤3 = 0` |
| `margo_v0.1_diag_kl_bc_ppo` | §7b | β=0.1; Local **21%→74%**; `kl_bc_mean` **1.18→16**; T **648→~1000** (itr14–17) |
| `margo_v0.1_diag_bc_fewshot` | §9 | k3 PPO val T **580→657**; Local 0.19/0.20 → **0.29/0.30**; `n_non` p50 4.8/5.2 → **7.8** |
| `margo_v0.2_diag_binary_lat` | §26 | isolation: `vocab_size=2`, `use_energy=False`, no V2V. Log at freeze still listed the job as running; V2V/energy are not the occupancy cause by design of this run |

Seven configurations. Same failure family: mean policy gradient on a combinatorial plan walks occupancy off the sparse expert basin.

### CAVIA-on-z, `L_z = mean T`: no query gain

| run id | log section | number |
|---|---|---|
| `margo_v0.2_diag_cavia_frozen` | §19 | query val **584.5 → 585.1**; `z_l2≈0.01–0.02` |
| `margo_v0.2_diag_cavia_strong` | §20 | `z_l2` **0.43–0.67**; query val **584.5 → 584.2** |
| `margo_v0.2_diag_cavia_bccont` | §22 | identity val **575.1**; query val **572.7 → 572.7** |

Query T unchanged within ±1 s on the two `bc_2opt` runs; the `bc_continue` run is exactly flat.

### IL works in-distribution; OOD remains

| run id | log section | number |
|---|---|---|
| `margo_v0.1_diag_bc_continue` | §6 | train greedy T **464.3**, token vs expert **0.968** |
| `margo_v0.1_diag_bc_2opt` | §14 | train greedy T **435.6**, token **0.965**; val **581.8** vs 2-opt **423.6** |
| `margo_v0.1_diag_bc_unseen` | §8 | val T **575** / token **0.711**; meta-test **557** / **0.707** |
| `margo_v0.1_diag_bc_scheduled` | §10 | val **574** / token **0.716** — exposure is not the OOD gap |

Encoder was never varied (Phase 1).

### Long budget on the old loop

Old repo CSV `results/ckpt_ours_final_3500/meta_offloading20_log-inner_step1/progress.csv` (not a Kish diagnostic row): sampled T plateaus ~860–870 from iteration ~500 vs publication greedy 829. Extra iters after collapse do not recover (`par500` §1). Phase X re-tests H_budget under a cleaned PPO-from-scratch control.

## Why

Mean-field PG on a factorized Seq2Seq learns the marginal mode (MEC-heavy). Expert skill is joint (`n_non` p50 **4**, Hamming-2 in 73% of graphs, log §11). Soft KL β=0.1 does not hold the basin. A free `z` with REINFORCE on mean T has no scenario latent under fixed resources (ADR-009) and no best-sample anchor.

EAS (Hottung et al., ICLR 2022) updates a small subset with a POMO shared baseline **plus** imitation of the best sample so far. That is the only adaptation rule not yet falsified here.

## Consequence

- Do not advertise CAVIA as the method. Title `MARGO-METHOD-v0.2-cavia` is dead.
- Do not start `margo_v0.1_primary` 3500.
- Modules listed in Phase 0 deliverable 7: `STATUS: baseline/ablation only (ADR-007)`.
- Phase 3 gate failure → fallback Phases 1–2, no meta-claim.

## Appendix — audit Critical C-01 … C-12

Source: `MARGO_Deep_Technical_Audit_and_arXiv_Readiness.docx` (public SHA `53fe08d0…`). Status is **this** tree (`mrlco-new`), not the audited public commit.

| ID | Status |
|---|---|
| C-01 clique encoder | closed in mrlco-new (`policies/graph2seq_encoder.py:UniformNeighborSampler`; `encoder_obs.py:NEIGHBORHOOD="predecessor_and_successor"`; `tests/test_phase2_encoder.py`) |
| C-02 unused DAG edge bytes | closed in mrlco-new (`scheduler/engine.py`; `scheduler/routes.py:ROUTE_TABLE`; `tests/test_phase1_commit1.py`) |
| C-03 V2V→MEC precedence | closed in mrlco-new (`scheduler/engine.py`; `tests/test_phase1_properties.py:test_precedence`) |
| C-04 PPO value clip on `v_new` | closed in mrlco-new (`meta_algos/MRLCO.py:147-152` `vpredclipped = old_v + clip(vpred-old_v)`; `spec/learning_ops.py:clipped_value_prediction`; `tests/test_phase3_learning.py:TestValueClip`) |
| C-05 sequential per-task outer Adam | closed in mrlco-new (`meta_algos/MRLCO.py:UpdateMetaPolicy` reads common `theta0`, `mean_pseudogradient`, one outer Adam; `tests/test_phase3_learning.py:test_order_invariant`). Not interpolation Reptile; that remains a Phase 6 named baseline if wanted, not a bug in this stack. |
| C-06 no core→task sync before iter 0 | closed in mrlco-new (`meta_trainer.py:138` `sync_task_policies_from_core` before `obtain_samples`; also `meta_trainer.py:575` after init) |
| C-07 15/4 split vs code | closed in mrlco-new (`spec/split_loader.py` `latin_grid_holdout_v1` 15/5/5; `tests/test_phase3_learning.py:TestSplitWiring`) |
| C-08 fine-tune and report same 100 graphs | closed in mrlco-new (`spec/split_loader.py:support_query_tasks` 20/80; `spec/eval_protocol.py:require_sliced_task`) |
| C-09 missing final ckpt / raw logs | open — handled in Phase 6 |
| C-10 manuscript / README / figures disagree | open — handled in Phase 6 |
| C-11 seconds labelled ms | closed in spec (`units = seconds`, `spec/prompts/README.md`); leftover publication labels open — Phase 6 |
| C-12 stepwise MDP not exposed | closed as documented reformulation (`spec/LEARNING_PROTOCOL.md` §1: whole-plan `env.step`, `done=True`; decoder does not observe the calendar) |

H-01 inner Adam leak across reassigned tasks: closed in mrlco-new (`meta_algos/MRLCO.py:reset_inner_optimizer` at `UpdatePPOTargetPerTask:295`). Entropy bonus claim: closed as forbidden (`frozen_experiment.yaml` `entropy_coefficient: 0.0`, `entropy_claim_allowed: false`). Multi-seed / final checkpoint: open — Phase 6.
