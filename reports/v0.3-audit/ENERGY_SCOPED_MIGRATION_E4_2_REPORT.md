# E4.2 report — validation objective producer + pure-plan evidence

Branch `erfan/phase4-eval`. HEAD after this report: see `git log`. No training,
pilot, deadline or shield run was launched. No large artifact was committed.

## Commits

| commit | content |
|---|---|
| `536bd8a` | E4.2 — `validation_per_graph_plans` producer + objective wiring |
| `2234e42` | E4.1 — pure-plan evidence runner + committed JSON evidence |
| this commit | smoke CLI flags + CSV contract checks + report + Kish runbook |

Local suite after everything: `711 passed, 5 skipped, 13 errors` (errors = TF
collection on a machine without TensorFlow). The suite is NOT fully green until
the TF image runs it: that is blocker 2 and it is still open.

## Blocker 1 — validation objective had no input (now fixed in code)

Before: `_validation_plan_objective` read `metrics["validation_per_graph_plans"]`,
nothing produced it, and every `log_only`/`lexicographic` validation logged
`objective/unavailable`.

After (`536bd8a`):

* `OffloadingEnvironment.get_reward_batch_step_by_step` records, per
  current-task graph, the SAME `ScheduleResult` and SYSTEM `ReferenceRanges` the
  reward step already built, plus `graph_fingerprint`, `order`, `plan`,
  `scheduler_config_sha256` and `energy_scope` (no second schedule; a counting
  test pins the schedule-call count with and without the channel).
* `_validation_plan_record` is fail-loud: a result without the resolved
  fingerprint, or a mobile/metadata-free reference, raises
  (`EnergyReferenceMismatch` / `ValueError`).
* `HeldOutQueryEvaluator.evaluate_one/evaluate_all` hoist the payload onto the
  validation metrics; the trainer's objective path consumes it.
* The heavy `(ScheduleResult, refs)` payload is stripped to
  `n_validation_plans` before audit/logging, so audit JSON stays serializable.
* Opt-in: `energy_config["validation_plans"]`, set by the trainer only when
  `objective_mode != "off"`.

Tests: `test_validation_plans.py` proves the payload is objective-ready
(`objective_from_plans` returns a finite aggregate, not `None`), identities are
JSON-safe, the disabled channel emits nothing, a mobile reference raises, and no
extra schedule happens.

Not yet proven end-to-end: the real `HeldOutQueryEvaluator` run is TF-dependent
and has not been executed. Blocker 1 is closed at the contract/unit level, not at
the trainer-smoke level.

## E4.1 — pure-plan evidence (CPU, real numbers)

`spec/pure_plan_evidence.py` runs four plans (all-UE, all-MEC, all-HELPER, mixed)
over two deterministic graphs and records latency, requester/mobile/system
energy, MEC compute, MEC TX, primary scalar, constraint raw/status and the
scheduler fingerprint. Gates: boundary identity, per-scope component identity,
timing invariance (legacy vs physical accounting, fixed timing axis),
fingerprint parity, finiteness. All gates PASS; `mec_tx_joules > 0` under the
primary physical radio model.

Evidence: `reports/v0.3-audit/pure_plan_evidence.json`.

Selected rows (joules, primary config):

| graph | plan | latency_s | requester | mobile | system | MEC compute | MEC TX |
|---|---|---|---|---|---|---|---|
| chain | all_UE | 6.0 | 15.10 | 15.10 | 15.10 | 0 | 0 |
| chain | all_MEC | 4.03 | 2.29 | 2.29 | 1515.85 | 1509.95 | 3.61 |
| chain | all_HELPER | 10.8 | 3.20 | 174.67 | 174.67 | 0 | 0 |
| chain | mixed | 9.69 | 7.78 | 66.00 | 572.93 | 503.32 | 3.61 |
| fork | mixed | 3.6 | 3.89 | 33.00 | 286.46 | 251.66 | 1.81 |

Both graphs show `system > mobile` exactly when a MEC task is present, and the
constraint status vocabulary is demonstrated: a configured total-energy budget
reports `active`, an absent one reports `not_configured` with no penalty.

## Blocker 2 — TensorFlow path unverified, and Kish is unreachable from here

Attempted from this session:

    ssh -o BatchMode=yes -o ConnectTimeout=10 kish-ai 'hostname'
    -> kex_exchange_identification: Connection closed by remote host

    ssh -o BatchMode=yes -o ConnectTimeout=10 kish-core 'hostname'
    -> Connection closed by 188.121.120.169 port 22

`nc -vz` shows the TCP ports open, but the SSH key exchange is closed before
authentication (no key is even offered), so this is not a credential problem:
raw SSH egress is not available to this environment. HTTPS works, `git push`
works, SSH does not. **Kish evidence cannot be produced from this session.**

### Kish runbook (run on kish-ai, clean SHA `2234e42`)

    cd /opt/margo/mrlco-new
    git fetch erfan && git checkout <SHA> && git status --porcelain   # must be empty

    # 1. full TF-compatible suite
    bash spec/kish_gpu.sh energy-tests

    # 2. mask smoke (legacy parity + static shield), then runtime
    bash spec/kish_gpu.sh mask-smoke
    bash spec/kish_gpu.sh mask-runtime

    # 3. metric smoke
    bash spec/kish_gpu.sh mask-metrics

    # 4. one-iteration trainer smoke on the NEW contract
    #    latency_only reward + log_only objective (exercises the E4.2 producer
    #    and the energy/objective CSV checks in validate_progress_csv)
    bash spec/kish_gpu.sh energy-train-smoke

Do NOT start a pilot or a long train. `spec/mask_sanity.py` now accepts
`--reward-mode {publication,latency_only,latency_over_all_mec}` and
`--objective-mode {off,log_only,lexicographic}`; defaults keep the historical
publication/off behaviour.

### One-iteration validation checklist

* primary reward is latency-only (no energy term);
* CSV has `energy/requester_joules`, `energy/mobile_joules`,
  `energy/system_joules`, `energy/primary_joules`, `energy/primary_scope`;
* `energy/primary_scope == system` and `energy/primary_joules == system_joules`;
* `objective/*` keys are present — no `objective/unavailable`;
* `constraint/status` shows `total_energy=not_configured` when no total budget,
  with no penalty;
* scheduler fingerprint parity across train / validation / evaluator / workers;
* CSV header and every row have equal field counts (quoting + late fields);
* no NaN/Inf anywhere;
* process exit code 0.

## Verdict

Contract, migration, telemetry, reward, constraints, the validation producer and
CPU pure-plan evidence are done. The TF path and the real one-iteration smoke are
not executed, so the phrase "ready to train" must NOT be used yet. Next step:
run the runbook above on Kish and attach the resulting logs/CSV.
