# PHASE 0 — Prune, decide, freeze the v0.3 method surface

## Role
You are the lead ML engineer and scientific reviewer for MARGO (`MARGO_BASELINE/mrlco-new`). You are not writing a paper yet. You are closing the v0.2 diagnostic campaign and freezing the decision record that every later phase depends on. Precision over speed. Nothing in this phase touches Kish.

## Context you must load first
1. `spec/PHASE4_DIAGNOSTIC_LOG.md` sections 1–26 and `spec/PHASE4_WORKLOG.md`. Twenty-six diagnostics exist. Key verdicts:
   - Every PPO-on-θ run collapsed occupancy: `par500` MEC≥0.95 at iter 21 and still 0.998 at iter 180; `bc50` left the expert basin by iter 9; `kl_bc_ppo` β=0.1 did not hold (`kl_bc` 1.18→16); `bc_fewshot` k3 PPO raised val T 580→657; `binary_lat` (no V2V, no energy) still went MEC 45%→95% and its best iteration was 5, not 49.
   - The old 3500-iteration run (`results/ckpt_ours_final_3500/.../progress.csv`) plateaued at sampled T ≈ 860–870 from iteration ~500 onward against publication greedy 829. Long budget did not help.
   - CAVIA-on-z was tried three times (`cavia_frozen`, `cavia_strong`, `cavia_bccont`): query T unchanged within ±1 s.
   - IL works in-distribution: `bc_continue` train 464 / token 0.968; `bc_2opt` train 436 / token 0.965. OOD gap: val 575–582 vs expert 424–464. Scheduled sampling, better teacher, pair-sup, oracle-dist-ID all failed to close it. Encoder has never been varied.
2. `spec/FINAL_ARCHITECTURE.md` and `spec/FINAL_CONTRIBUTIONS.md` (v0.2-cavia). These are now superseded in the parts that name CAVIA-on-z as the adaptation engine.
3. `MARGO_Deep_Technical_Audit_and_arXiv_Readiness.docx` (root). 51 findings. Confirm which are already closed in `mrlco-new` (scheduler precedence, directed adjacency, split manifests, `end_token` outside action set) and which remain (multi-seed, final checkpoint, units in reports, Reptile correctness in the baseline path, entropy claim).

## Deliverables (all files, no runs)
1. `spec/decisions/ADR-007-adaptation-engine.md`. Record the decision: inner PPO on θ, Reptile on θ, and CAVIA-on-z with a pure `mean T` loss are **removed from the method** and retained only as baselines/negative ablations. Cite the exact run ids and numbers above. State the replacement (EAS-style support adaptation of a small parameter subset with a best-sample imitation anchor, Phase 3) and the fallback (zero-shot BC + best-of-k, Phases 1–2) if Phase 3 fails its gate.
2. `spec/decisions/ADR-008-inference-protocol.md`. Define what counts as "the learned method" at inference: sampling k plans from π and calling `schedule()` once per sample is part of the neural inference protocol (standard in neural combinatorial optimization: POMO, PolyNet, EAS). Handcrafted neighbourhood search (`pair_head`, `pair_seq`, Hamming-2, 2-opt) is **not** part of π; it is a teacher (training labels only) or a reported reference ceiling. Fix the maximum k that may be reported as the method (k ≤ 64) and require that evals-per-graph is always reported alongside T.
3. `spec/decisions/ADR-009-meta-task-axes.md`. Record that the current meta-task axis is DAG family only (fat × density, fixed resources) and that a second axis (resource profile) will be added in Phase 4. Explain why context-vector adaptation had nothing to infer under fixed resources.
4. `spec/FINAL_ARCHITECTURE.md`: bump header to `MARGO-METHOD-v0.3`. Replace §9 (CAVIA) with a stub pointing to ADR-007 and Phase 3. Keep §1–§8, §10 unchanged. Mark §6 (FiLM on LSTM initial state) as "retained as the candidate parameter subset for Phase 3".
5. `spec/FINAL_CONTRIBUTIONS.md`: rewrite the claim list to the v0.3 set: (C1) problem + protocol (ternary plan-level, half-duplex V2V, energy, structural holdout with disjoint support/query); (C2) negative result with mechanism (meta-PPO on θ collapses occupancy against strong heuristics; seven configurations); (C3) search-distilled Graph2Seq policy + best-of-k + support adaptation (conditional on Phase 1–3 gates); (C4) strong baseline suite (greedy_from_mec, 2-opt). Every claim must list the run ids that support it and the gate that still has to pass.
6. `spec/PHASE4_WORKLOG.md`: add a "v0.2 closed" line under section 26 and a "v0.3 plan" block listing Phases 1–6 with their gates (copy from `spec/prompts/README.md`).
7. Isolate dead method code without deleting it: add a module docstring line `STATUS: baseline/ablation only (ADR-007)` at the top of `spec/cavia_loop.py`, `spec/cavia_objective.py`, `spec/kl_bc_anchor.py`, `spec/pair_head.py`, `spec/pair_ranker.py`, `spec/pair_seq.py`, `spec/pair_sup.py`, `spec/rewrite_mec.py`, `spec/oracle_dist.py`. Do not change behaviour; existing tests must still pass.

## Constraints
- No hyperparameter changes, no training, no Kish access.
- Do not delete files, runs, or ledger rows. Do not edit anything under `spec/kish_log_archive/`.
- Do not rewrite freeze tags.
- Keep Persian in existing Persian docs where you only edit sections; new ADRs are in English.

## Double-check (mandatory, in this order)
1. Run `python3 -m pytest env/mec_offloaing_envs/scheduler/tests -q -x` locally. All tests that passed before your edits must still pass (TF-dependent tests skip locally; that is expected).
2. Grep for contradictions: `rg -n "CAVIA" spec/FINAL_ARCHITECTURE.md spec/FINAL_CONTRIBUTIONS.md` must return only lines that refer to CAVIA as removed/ablation. `rg -n "paper_result=true"` must return nothing.
3. Re-read ADR-007 and verify every number quoted matches `spec/PHASE4_DIAGNOSTIC_LOG.md` exactly (run id, iteration, value). Fix any mismatch; do not round.
4. Verify the audit cross-walk: for each of the 12 Critical findings (C-01 … C-12), write one line in ADR-007's appendix: `closed in mrlco-new (file:symbol)` or `open — handled in Phase N`. C-04 (value clip), C-05 (Reptile), C-06 (pre-sync), H-01 (Adam leak) must be marked "open — Phase 6 baseline fix" unless you verified the fix in code with a file and line reference.
5. `git status` must show only the files listed under Deliverables plus the docstring-tagged modules. Anything else is a mistake; revert it.
6. Write a 15-line summary at the end of `spec/PHASE4_DIAGNOSTIC_LOG.md` under a new heading `## v0.3 Phase 0 — decisions frozen`, listing the three ADRs and the exact gate numbers copied from `spec/prompts/README.md`.

## Stop condition
Stop when all seven deliverables exist, the six double-check items pass, and nothing was launched. Report the list of changed files and the pytest summary line.
