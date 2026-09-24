# Deadline dataset + shield experiment — design (needs approval before code)

Status: **design only, no code yet.** Everything below is grounded in the code that
already exists (`scheduler/deadlines.py`, `static_bounds.py`, `encoder_obs.py`,
`mask_metrics.py`, `spec/mask_sanity.py`) and in the measured results of the
500-iteration run (`MASKED_PPO_INTERFACE_6b.md` §17).

Goal: turn the pipeline that is now proven into a measurement of the shield. The
500-iteration pair showed the physics of the situation: the frozen dataset has
**0 deadlines in 30000 task slots**, so `static` and `off` were the same run up to
nondeterminism (the shield mask was all-True). Nothing about the shield's effect
can be learned until deadlines exist.

---

## 1. Why a sidecar, and where it is stamped

`to_canonical_dag` already reads four optional per-task fields from the dataset
objects (`adapter.py:30-51`):

    deadline_s, deadline_type, criticality_class, tardiness_weight

They are absent from the `.gv` files, and both consumers — the v3 observation
block and the scheduling engine — go through `to_canonical_dag(task_graph)` on the
same `task_list` objects. So stamping those four attributes once, right after the
graph is parsed, is a single choke point that covers everything:

    .gv files (unchanged)  ->  OffloadingTaskGraph.task_list[i].deadline_s = ...
                           ->  to_canonical_dag -> { v3 obs features, engine metrics }

Proposed artifact: one JSON sidecar per regime, hashed into the run provenance,
rather than rewriting the `.gv` dataset:

    spec/deadline_regimes/<regime>.json
    {
      "regime": "tight_hard",
      "kappa": 1.02, "alpha": 0.5, "deadline_type": "hard",
      "generator": "deadline_regime_v1",
      "resources_hash": "...", "cycles_per_bit": 300.0,
      "graphs": {
        "3/random.20.7.gv": {"task_deadlines_s": {"0": 1.83, ...},
                             "criticality_class": {...}, "tardiness_weight": {...}}
      }
    }

Why not rewrite the dataset: the split manifest pins file paths and hashes, the
regression runs must stay byte-identical, and a sidecar lets us sweep κ without
regenerating 30000 graphs.

## 2. Deadline construction (the actual science)

Per graph, using the frozen `ResourceConfig` and the admissible lower bounds that
the observation already uses:

1. `bounds = static_action_bounds(dag, order, resources, cycles_per_bit)`
2. graph budget `D_G = kappa * bounds.max_ready_lb`, where `max_ready_lb` is the
   optimistic graph completion (sink includes the return hop). `kappa >= 1` keeps
   the relaxation feasible; `kappa < 1` is the deliberately infeasible bucket.
3. per-task optimistic finish `EFT_i = min_a bounds.finish[i][a]`, and for sinks
   the basis is `bounds.ready_lb[i]` (return hop included), because the engine
   measures a miss on `all_consumers_ready`, never on compute finish.
4. per-task deadline `d_i = EFT_i + alpha * (D_G - EFT_i)`, `alpha in [0, 1]`:
   `alpha = 1` puts every task at the graph budget (only the global constraint
   binds), `alpha -> 0` makes every task as early as its own relaxation allows
   (maximum per-task tightness). `d_i >= EFT_i` by construction, so the relaxation
   never violates its own deadline.
5. `deadline_type`: `hard` for the shield regimes, `soft`/`firm` for the objective
   and constraint channels (locked semantics: soft -> tardiness in J, firm ->
   constraint channel, hard -> shield).
6. `criticality_class` and `tardiness_weight`: independent fields (locked). Class
   from structure (e.g. depth/sink), weight set separately so the paper can talk
   about mixed criticality without implying a penalty.

Optional refinement, if the interpolation proves too crude: a backward pass
`LFT_i = min(D_G, min_j (LFT_j - lb_transfer(i->j) - lb_compute(j)))` and
`d_i = EFT_i + alpha * (LFT_i - EFT_i)`. Only worth it if the gate metrics below
show the interpolation cannot reach the target mask rate.

## 3. Regimes to generate

| regime | kappa | alpha | type | purpose |
|---|---|---|---|---|
| `none` | – | – | none | regression: reproduces the 500-run byte-exactly |
| `loose_hard` | 1.50 | 1.00 | hard | shield rarely bites; sanity that nothing breaks |
| `medium_hard` | 1.10 | 0.50 | hard | main candidate: non-trivial mask, globally feasible |
| `tight_hard` | 1.02 | 0.25 | hard | aggressive mask; check `all_invalid_rate` stays small |
| `infeasible_labelled` | 0.80 | 0.25 | hard | guard/dead-end tests only, never for training |
| `soft_mix` | 1.10 | 0.50 | soft | tardiness channel for J, mask stays all-True |
| `firm_mix` | 1.10 | 0.50 | firm | constraint channel (`mec_task_fraction` etc.) |

## 4. Acceptance gates per regime (computed, not assumed)

A regime is usable for the shield experiment only if all of these hold, measured
over the 15 meta-train prefixes (30000 task slots):

1. `has_deadline == 1` for every task; the v3 deadline block is no longer constant.
2. `mask/active_rate` in **[0.05, 0.40]** — the shield must bite, but not dominate.
3. `mask/all_invalid_rate` **< 0.01** — dead-end rows must stay rare.
4. `mask/forced_rate` reported (expected non-zero: exactly one action feasible is
   the interesting case), no target, just recorded.
5. The reference plan (greedy-from-MEC) is feasible on **>= 95%** of graphs at
   `kappa >= 1`; for the labelled infeasible bucket this is expected to be ~0.
6. Per-action breakdown of the mask (UE / MEC / HELPER closed) reported, because
   the whole question is whether MEC — the action the deadline-free policy always
   picks — becomes infeasible.
7. `policy/argmax_masked_rate > 0` on the trained policy: the raw policy must
   actually *want* a forbidden action, otherwise the shield is untested.

Gate tool: extend the existing `spec/mask_metric_smoke.py` pattern — a dataset
sweep that runs the env under each regime and prints the table above; no GPU
needed beyond the env's own scheduling.

## 5. Experiment design for the shield (uses the measured noise floor)

From §17.4: two nominally identical runs differ by mean 1.19%, p95 4.57%, max
13.4% in `AverageReturn`, and the divergence grows with training. Therefore:

* **Pilot (do this first):** 1 seed, `off` vs `static`, **200 iterations**,
  sequential, on `medium_hard` (or whichever regime passes §4 with
  `active_rate` 0.10-0.30). Cost: 2 x ~13 h = ~27 h wall. Deliverable: does the
  shield change `invalid_action_rate`, tardiness, return, latency and energy by
  more than the noise floor?
* **Full run (only if the pilot effect exceeds the noise floor):** 3 seeds x
  {off, static} x 500 iterations, **sequential** (parallel execution adds the
  contention component to the noise). Cost ~8 days; can be batched overnight.
* Every run keeps the same frozen trainer config, obs v3, `--seed` recorded, and
  the regime sidecar hash in `config.resolved.json`.
* Stop rules: the existing watchdog, plus `mask/active_rate > 0` expected (a
  regime that silently produces an all-True mask is a failure, not a pass).

Metrics to compare, per condition: `policy/invalid_action_rate` (should be 0 in
static by construction, > 0 in off — that is the causal mechanism),
`mask/*` rates, deadline miss rate and tardiness, `AverageReturn`,
`validation_query_composite_objective`, `Average latency`, energy per boundary
(`total_requester_joules`, `total_mobile_joules`, `total_system_joules`), and
constraint costs when enabled. All against the noise-floor table.

## 6. Energy and objective channels (switch-on order)

The 500 runs used the **legacy** byte-exact accounting (`energy_model: legacy`),
which is why the code default stayed legacy. For the deadline runs:

1. `energy_model: physical_v1` + `radio_model` from `spec/frozen_experiment.yaml`
   (system-scope primary, requester/mobile as sensitivity), verified once on kish
   against the audited per-MiB numbers (UE/helper/MEC = 2.52 / 28.3 / 252 J).
2. `objective_mode="log_only"`: log `J`, `c_E` (raw + capped), `c_H`, `c_F` per
   iteration **without** changing the optimizer's target. This is the review step
   where the channels are checked against intuition before they can influence
   anything.
3. `MARGO_CONSTRAINTS=spec/constraints.yaml` + lexicographic checkpoint selection
   (feasibility gate first, then min J) — only after (2) is reviewed.
4. Energy/firm terms stay OUT of `J` until that review passes (locked decision).

Note: with `log_only` the reward is unchanged, so a `legacy` vs `physical_v1` pair
on the same regime isolates the energy accounting effect on the logged channels
only. That pair is cheap and worth including in the pilot.

## 7. Explicitly out of scope

No encoder change, no decoder width change (structural effective rank still
~29/128; trained-v3 activations are the next evidence, not an assumption), no new
reward terms, no simultaneous obs+reward+PPO changes. One variable at a time.

## 8. Decisions needed from you

1. Sidecar JSON + stamp-on-load (§1), or rewrite the `.gv` dataset with deadline
   attributes? Sidecar is cheaper and keeps the regression byte-exact.
2. The `alpha` interpretation in §2.4 (graph-budget interpolation) — accept, or
   require the LFT backward pass from the start?
3. Regime table in §3 — which κ values, and is `medium_hard` the pilot regime?
4. Pilot size: 1 seed x 200 iterations x 2 modes, sequential (~27 h) — accept?
5. Turn on `physical_v1` in the same pilot, or keep legacy for the pilot and
   isolate energy afterwards?

---

## 9. Implementation report (no training, CPU only)

Commits: `de70131` (schema/generator/stamper), `78efd56` (witness + anchoring),
`f831e69` (mask breakdown + gate rules + sweeper), plus the evidence commit.
Local suite `544 passed, 5 skipped`; 64 new tests; `py_compile` clean.

### 9.1 What is standard now

* `schema deadline_regime_v1` with a content hash per graph, the dataset manifest
  hash, a scheduling-relevant resource hash, model provenance, seed, policies, and
  per task `deadline_s` / `deadline_type` / `criticality_class` / `tardiness_weight`.
* a stamper that fails loudly on coverage mismatch, hash mismatch, non-finite or
  non-positive deadlines, an unknown type, a duplicate id, and on stamping a
  different regime over an already stamped graph; idempotent for the same one;
  regime `none` writes nothing and is asserted byte-exact against the legacy path.
* a witness pipeline that uses the REAL scheduler (`schedule_via_adapter`, six
  single-capacity calendars) and stores, per graph, the witness actions, makespan,
  and per-task finish / all_consumers_ready / deadline / slack / missed.
* `build_regime_with_witness`: find a fastest plan, anchor the deadlines to it
  (`d_i = EFT_i + alpha * (kappa * W_i - EFT_i)`), stamp, then certify with a
  miss-first search seeded by that plan. Graphs without a witness are returned with
  a reason instead of being kept.
* a sweep + gate report per split, with the action-closure breakdown and two
  fully-worked example graphs.

### 9.2 The gate result: no regime passes, and the reason is structural

Anchored sweep (alpha = 1.0, limit 8 per distribution, all three splits):

| regime | kappa | split | witness rate | active | forced | all-invalid | UE closed | MEC closed | HELPER closed | gate |
|---|---|---|---|---|---|---|---|---|---|---|
| loose_hard | 1.05 | meta_train | 1.000 | 0.042 | 0.030 | 0.000 | 0.030 | **0.000** | 0.042 | fail |
| loose_hard | 1.05 | validation | 1.000 | 0.055 | 0.045 | 0.000 | 0.045 | **0.000** | 0.055 | fail |
| loose_hard | 1.05 | meta_test | 1.000 | 0.016 | 0.015 | 0.000 | 0.015 | **0.000** | 0.016 | fail |
| loose_hard | 1.25 | all three | 1.000 | 0.028 | 0.022 | 0.000 | 0.022 | **0.000** | 0.028 | fail |
| (alpha = 0.5 probe) | 1.10 | meta_train | **0.000** | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | no graphs |

`witness_rate = 1.000` with `kappa = 1.05` and `alpha = 1.0` is by construction:
the deadlines come from a real plan and that plan is replayed as the witness. The
gate still fails, on two independent counts:

1. **`active_rate` is 0.016-0.055, below the 0.05 floor** (and far below the
   0.10-0.30 preferred band), so the shield would almost never fire.
2. **`mec_closed_rate = 0.0000` in every configuration and every split**, and
   `graphs_with_mec_closure = 0`. The action the policy actually collapses onto can
   never be closed.

The mechanism is measurable, not speculative. Over 600 tasks of three
distributions: MEC has the strictly smallest `ready_lb` in **600/600** cases, and
`UE/MEC` ranges 1.12-10.0 (mean 4.86), `HELPER/MEC` 1.31-10.0. So any deadline that
closes MEC also closes UE and HELPER, the row becomes all-invalid, and the
dead-end guard drops the mask: a deadline-based static shield cannot express
"MEC forbidden, something else allowed". On top of that the relaxation is loose
relative to reality -- in the worked example MEC's bound is 7.58 s while the
achievable plan reaches that task at 119.2 s -- so a deadline anchored to any
achievable schedule (125.1 s) leaves every action open.

Lowering `alpha` to create pressure destroys the instance instead: at
`alpha = 0.5`, `kappa = 1.10` **zero** graphs certify a witness, because the
deadlines fall below what any plan can achieve. The two knobs pull in opposite
directions and there is no window where the shield both exists and bites.

### 9.3 Worked example (regime `loose_hard`, kappa 1.05, alpha 1.0)

(graph `1/random.20.0.gv`, witness method `anchored_seed`, makespan 427.58 s)

| task | ready_lb MEC | ready_lb UE | deadline | witness action | actual ready | slack |
|---|---|---|---|---|---|---|
| 3 | 7.58 | 75.83 | 125.11 | 0 (UE) | 119.16 | 5.96 |
| 4 | 6.06 | 60.64 | 115.52 | 1 (MEC) | 110.02 | 5.50 |
| 2 | 5.43 | 54.26 | 272.48 | 0 (UE) | 259.50 | 12.98 |

The witness is mixed-action in ~100% of graphs, which is itself informative: the
H1/H2 search does move tasks off MEC when contention or the root upload makes it
worthwhile, while the final policy in the 500-iteration run collapsed to 98.7-99.6%
MEC.

### 9.4 Consequences for the shield experiment

The planned pilot (`off` vs `static` on a deadline regime) cannot be run as
designed: with no regime passing the gate, `off` and `static` would again be the
same run. Three honest options, in the order I would try them:

1. **Runtime / prefix shield.** The only mechanism that can close MEC is a mask
   computed from the actual prefix state (`suffix.dag_lower_bound_masks` with real
   transfers and contention). That needs the environment to hand the scheduler
   back per decoded token, which is the token-by-token stepping gap already
   documented in `MASKED_PPO_INTERFACE_6b.md` §13.4, and the mask must travel with
   the batch as `MARGO_MASK_MODE=runtime` expects.
2. **Reframe the claim.** The shield is a safety net, and on this workload it is
   provably almost never needed: MEC dominates every task's lower bound, so a
   deadline that forbids MEC forbids the instance. That is a legitimate result to
   report -- and it is the opposite of the assumption behind the pilot.
3. **Change the physics, not the shield.** If the intended story needs MEC to be
   deadline-infeasible while other actions survive, the model must make remote
   execution pay an unavoidable cost that local execution does not (uplink of the
   task input on a contention-limited link, RSU admission/queueing, or MEC
   capacity limits). That is a modelling change with its own audit, not a dataset
   knob.

Nothing here changes the frozen training path: `off` mode is untouched, the `.gv`
files are unmodified, and the sidecars are separate artifacts.

### 9.5 Energy plumbing: still open

The recon for commit 4 is complete and confirms the audit suspicion with exact
drop points: `adapter.py:82-101` builds `ResourceConfig` without `energy_model` /
`radio_model`, so every train/val env runs legacy (MEC compute energy = 0,
`total_system_joules == total_mobile_joules`); `energy_scope` is parsed but has
zero production consumers; and reward/reference-ranges/step-log/constraints read
`total_mobile_joules` while the log-only objective reads `total_system_joules`
and the constraint `ue` channel reads `total_requester_joules`. Commit 4 is the
threading + canonical-accessor fix, with legacy kept byte-exact.
