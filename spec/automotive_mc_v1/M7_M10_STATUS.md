# MARGO-AUTOMOTIVE-MC-v1 — M7…M10 completion record

Status: **READY_FOR_INTEGRATION** (dataset completion only; no training/PPO/sampler/
policy/optimizer code was touched, and no training run was started).

Branch: `phase4-eval`. Base commit for this work: `134a00b` (M6-CALIBRATION-REV2).
M5-v2 (`workload_model_v2.yaml`) and M6-v2 (`sla_registry_v2.yaml`,
`deadline_model_v2.py`, `deadline_report_v2.json`) are treated as immutable: M7–M10
only read them, and the M7 validator re-derives the M6-v2 deadline hashes to prove
they did not move.

## M7 — mixed criticality

Artifacts: `criticality_policy.yaml`, `criticality_model.py`,
`validate_automotive_m7.py`, `criticality_report.json`.

* Classes `LOW / MEDIUM / HIGH`, resolved from (semantic_role, application_family,
  motif_role, explicit policy) only. The resolver is a total function, and the
  adversarial tests relabel task ids, permute topology, scale the workload and change
  the deadline range without moving a single class.
* `mapping_background` is declared `non_safety_background` and is the only family
  allowed to lower a role; safety families may raise only, and motif floors are
  enforced inside safety scope.
* `C_LO`/`C_HI` are **execution-demand uncertainty** budgets, never a safety level and
  never WCET (`wcet_claim: false` everywhere). `C_HI >= C_LO > 0` on every HIGH task;
  LOW/MEDIUM report `budget_hi_status: not_applicable`.
* Modes: initial `LO`; the only `LO -> HI` trigger is a HIGH task whose observed
  execution exceeds its `C_LO`; HI is sticky until graph completion in v1; HIGH is
  never drop- or degrade-allowed in either mode; every switch logs
  `triggering_task_id, semantic_role, criticality, observed_execution, C_LO, reason,
  previous_mode, new_mode, logical_schedule_point`.
* Criticality is provably not deadline tightness: depth NMI 0.46, urgency NMI 0.40,
  and both probes carry mixed-class levels, so neither is a deterministic function of
  the class.
* Immutability: `criticality_report.json` pins the sha256 of every M4/M5-v2/M6-v2
  artifact before and after and re-derives all four M6-v2 deadline outputs; the M7
  model is scanned for mutation tokens (it has none).

## M8 — deterministic generator + materialization

Artifacts: `generation_config.yaml`, `automotive_dataset.py`,
`automotive_generator.py`, `validate_automotive_m8.py`,
`m8_materialization_report.json`, and the materialized dataset under
`env/mec_offloaing_envs/data/automotive_mc_v1/`.

* Every draw comes from a sha256-counter PRNG seeded by a canonical string; there is
  no `random`, no clock, no dict/FS ordering. Materializing twice into two clean
  directories is byte-identical (graphs, manifest, semantic annotations and
  lineage fields all compared).
* 240 graphs = 4 families x 5 sibling parent seeds x 12 regime cells
  (2 workload x 3 SLA x 2 resource-level); exactly 20 tasks and 4800 tasks total.
* Each graph carries dataset/generator version, graph id, parent seed, family,
  template id and lineage, semantic and topology signature, topology regime (frozen
  per template — the generator never rewires a semantic payload edge), workload
  regime, resource profile + the six drawn rates, SLA id/regime, `P_f`, `D_f` range,
  `D_G`, `criticality`/`C_LO`/`C_HI`, drop/degrade policy for both modes, per-task
  `W_i`, `E_i`, `L_i`, `d_i`, slack, per-edge `B_e`, provenance refs, and the raw and
  canonical hashes.
* `W_i = round(t_ref * f_ref / (8 * xi))` is recomputed by the validator and is
  invariant under a changed resource profile; `B_e` is per edge, comes from the
  declared per-class construction band, is never derived from `W_i`, and the eight
  realised payload bands are pairwise disjoint.
* The planning-motif partition seed (`86.83 ms`) stays invariant; only the
  range-based role classes are drawn, inside their frozen `t_ref_s_range`.
* `D_G` is the pre-declared draw from the frozen family `D_f` range (SLA-regime band),
  fixed before any schedule/search call; the validator re-derives `E/L/d/slack` from
  the M6-v2 construction and refuses a stale on-disk dataset.

## M9 — splits and leakage

Artifacts: `split_policy.json`, `automotive_splits.py`, `validate_automotive_m9.py`,
`m9_split_report.json`, and `splits.jsonl` / `split_summary.json` in the dataset dir.

* The split unit is the **template lineage**. In v1 topology is template-frozen, so
  two graphs of one template are topological near-duplicates; a leakage-free split can
  therefore only put a lineage on one side. That is the "held-out application
  families" generalisation axis: `meta_train = {perception_planning_control,
  mapping_background}` (120), `validation = {localization_prediction_planning}` (60),
  `meta_test = {cooperative_perception}` (60).
* Held-out roles are stratified by the regime cells: 20 support + 40 query, disjoint,
  selected by a sha256 rank over (split rule, seed, graph id, canonical hash).
* Leakage is checked on `graph_id`, `canonical_sha256`, `parent_seed`,
  `template_lineage`, `semantic_signature`, `topology_signature` and a
  near-duplicate signature: **0 violations**, and the checker is proven to fire on
  injected sibling-crossing, support/query-overlap, missing-role and opened-meta-test
  faults.
* Calibration is declared meta-train only; `meta_test` stays unopened
  (`opened_for: []`) for calibration, tuning and model selection.

## M10 — independent certification + final audit

Artifacts: `certify_automotive_m10.py`, `validate_automotive_m10.py`,
`m10_validation_report.json`, and `calibration_report.json`,
`feasibility_report.json`, `manifest.jsonl`, `provenance.json`,
`audit_report.json` in the dataset dir.

* Certification is read-only: the sha256 of `graphs.jsonl`,
  `dataset_manifest.jsonl` and `splits.jsonl` and every frozen field
  (`D_G`, workload, payload, criticality, budgets, resource profile, SLA) are
  identical before and after.
* Evaluated methods: `all_UE`, `all_MEC`, `all_HELPER`, `greedy`, `heft_reference`,
  and a bounded mixed-placement search (240 schedule evaluations per graph).
* Statuses are exactly `certified_feasible | witness_not_found | stress_or_infeasible`.
  `witness_not_found` is never equated with mathematical infeasibility: the
  infeasibility proof is a true lower bound (per-task fastest-tier compute critical
  path with zero communication). The M6-v2 reference-tier critical path is reported
  separately as a stress flag because it is NOT a lower bound when a faster tier
  exists.
* `rho_G = T_reference / D_G` is reported as a diagnostic and never redefines `D_G`.
* Results: 240/240 `certified_feasible`, 0 `witness_not_found`, 0
  `stress_or_infeasible`; best makespan 0.0113–0.1308 s (median 0.0335 s) against
  `D_G` 0.150–1.994 s; `rho_G` min 0.0086 / median 0.0995 / max 0.7357; mixed
  placement strictly beats every pure-location plan on 240/240 graphs.

### Final audits (all PASS)

1. timing scale — median best makespan 0.0335 s, max 0.1308 s; no return to the
   historical 528–1245 s regime.
2. deadlines — 4800/4800 tasks carry a positive subdeadline; the graph-level
   requirement is `makespan <= D_G`.
3. criticality — HIGH 1860, MEDIUM 1200, LOW 1740; HIGH exists in every safety family.
4. MC budgets — `C_HI >= C_LO > 0` on all 1860 HIGH tasks; no WCET claim.
5. payload semantics — all 8 classes instantiated with disjoint realised bands
   (control 153–250, trajectory 1350–1500, v2x 2000, object 2600–3200,
   radar 3602–4400, lidar 7202–8797, feature 10804–13200, camera 18016–21992 bytes).
6. placement difficulty — no pure location wins any graph (pure-best share 0.0).
7. feasibility — certified_feasible 1.000, witness_not_found 0.000,
   stress/infeasible 0.000. Requirements were **not** rewritten to change these
   numbers.
8. split leakage — 0 violations.
9. deterministic regeneration — PASS.
10. provenance — every numeric field group resolves to a verified source, a derived
    rule or a frozen source-calibrated-synthetic rule.
11. mixed placement — useful on 100% of graphs.
12. family balance — all four families populated; `mapping_background` is expected to
    be loose (declared non-safety slack), the three safety families are comparable.

### Scientific caveats (recorded, not hidden)

* **Feasibility is saturated by construction.** The frozen M6-v2 `D_f` ranges sit
  above the achievable makespan of the frozen resource model, so every graph admits a
  witness and the `witness_not_found`/`stress` statuses are never exercised by this
  population. Difficulty is expressed through the optimality gap `rho_G`
  (0.009–0.736) and the placement structure. Tightening a deadline after seeing
  certification was explicitly forbidden and was not done.
* `mapping_background` is a declared non-safety family: all-LOW criticality and large
  declared slack. It is included in meta-train only.
* The per-graph `D_G` draw is a pre-declared binning of the frozen family `D_f` range
  (`E2E-SLA-V2-BIN-V1`); it is not the M6-v2 midpoint. The generator reproduces the
  M6-v2 construction exactly at the midpoint, which the tests assert.
* Seven of the eight payload classes still have no numeric registry value (only rule
  rows); the realised bands are labelled `source-calibrated-synthetic` and anchored to
  the frozen `nominal_construction_bytes`. Only `v2x_message` is `standard-derived`.
* Energy is a **diagnostic** here: the timing axis is physical but energy accounting is
  the frozen legacy-normalized model, so no energy claim is made by this dataset.

## Reproduction

```
python3 spec/automotive_mc_v1/automotive_generator.py     # materialize M8
python3 spec/automotive_mc_v1/automotive_splits.py        # M9 splits + leakage
python3 spec/automotive_mc_v1/validate_automotive_m7.py
python3 spec/automotive_mc_v1/validate_automotive_m8.py
python3 spec/automotive_mc_v1/validate_automotive_m9.py
python3 spec/automotive_mc_v1/validate_automotive_m10.py
python3 -m pytest env/mec_offloaing_envs/scheduler/tests -q
```

## Next step for training integration (not started here)

Consume `env/mec_offloaing_envs/data/automotive_mc_v1/graphs.jsonl` +
`splits.jsonl`; use `calibration_report.json` (meta-train only) for normalization
constants (`workload_scale_bytes`, `latency_scale_s`, `deadline_slack_scale_s`); read
`manifest.jsonl` for certification status and `provenance.json` for SHA pins. The
support/query roles are already disjoint, and meta-test must stay unopened until the
model-side freeze.
