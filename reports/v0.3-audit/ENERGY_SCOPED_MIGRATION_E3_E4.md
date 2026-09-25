# E3/E4 scoped-energy migration — reward, objective and logging contract

Status: **design lock, implementation follows in E3.1 → E3.2 → E4.1.** Written
under the review that closed 4.2 (`394a038`/`d885809`/`8139d32`). The 4.2
refactor is a semantic no-op; E3 is where the intended numbers move, and every
move is deliberate, explicit and tested. Nothing here may be folded back into a
"refactor" commit.

## 0. What 4.2 left behind

* `energy_scalar(result, scope=...)` is the only boundary accessor; direct
  `total_*_joules` reads are allowlisted to the implementation and two
  oracle/serialization paths.
* `ReferenceRanges` is `energy_reference_ranges_v2` with `energy_scope` +
  `scheduler_config_sha256`; scoped construction is `compute_scoped_reference_ranges`,
  the old wrapper is mobile-only.
* The reference cache is keyed by graph content, mode, scope, panel and the
  resolved scheduler fingerprint; a mismatched hit is fatal.
* Consumer scopes after 4.2: reward MOBILE, `C_TOTAL_ENERGY` MOBILE,
  `C_UE_ENERGY` REQUESTER, objective numerator SYSTEM with a MOBILE-built
  reference/budget (the recorded mismatch), greedy/evaluator/reporting MOBILE.
* `energy_telemetry_v1` carries requester/mobile/system + primary scope/value +
  fingerprint and is built from the rollout's own `ScheduleResult`.

## 1. E3.1 — latency-only primary reward

The primary training reward carries NO energy: not requester, not mobile, not
system. It is the pure latency telescoping potential.

    J_t = L_t / L_scale          (primary)
    r_t = J_{t-1} - discount * J_t

* New `REWARD_MODE_LATENCY_ONLY = "latency_only"` is the primary mode; the
  training env defaults to it.
* `legacy_publication_reward` (`REWARD_MODE_PUBLICATION = "publication"`) is
  reachable ONLY by explicit opt-in (`reward_mode="publication"` in
  `energy_config`). Under it the reward energy term stays the historical MOBILE
  boundary and every reward/token number is byte-exact against the pre-E3 run.
* `latency_over_all_mec` stays the diagnostic it always was.
* Reward does not read a reference energy at all in the primary mode; the
  `L_scale` denominator comes from the mobile reference ranges, which are
  scope-invariant for latency.
* Logging is decoupled from the reward term: when `use_energy` is on, the env
  still emits the per-task MOBILE energy and `energy_telemetry_v1`, so disabling
  the reward energy does not remove `Average energy` or the scoped telemetry.
* Acceptance:
  * primary reward == a telescoping run with `energy_weight=0` and latency
    weight 1.0, token-for-token;
  * `publication` opt-in before == after, byte-exact (rewards, makespans,
    energies, per-task energy);
  * switching reward mode changes neither the schedule, the mask/observation
    stream, nor the telemetry fields.

## 2. E3.2 — system energy channel (objective) and system constraint

The objective and `C_TOTAL_ENERGY` become SYSTEM consumers, and the reference
they use is SYSTEM-built. This is the scientific change 4.2 refused to hide.

* Objective numerator, reference and budget are all SYSTEM:
  `energy_scalar(result, scope=SCOPE_SYSTEM)` and a reference built with
  `energy_scope=SCOPE_SYSTEM`. `require_reference_scope(refs,
  expected_scope=SCOPE_SYSTEM)` guards the objective path.
* `C_TOTAL_ENERGY` is SYSTEM: metric, reference and budget all system. The
  fractional budget is a fraction of the SYSTEM all-UE plan energy (for an
  all-UE plan system == mobile, so the anchor keeps its meaning) and the metric
  is the SYSTEM boundary.
* `C_UE_ENERGY` stays REQUESTER. `C_HELPER_ENERGY` stays the helper component.
* A system consumer must never receive a mobile reference: the env builds and
  caches references per requested scope, and the system path asks for and
  validates the system object. Same cache-key rule as 4.2 (scope + fingerprint +
  mode + panel + graph content).
* `C_TOTAL_ENERGY` without a configured budget reports
  `status="not_configured"`: no metric, no violation, no penalty and the
  Lagrangian stays off for it. The same status vocabulary applies to every
  constraint so a log can distinguish "satisfied" from "not measured".
* Acceptance:
  * the objective's system numerator and system reference are consistent (no
    mismatch assertion may remain in the migrated path);
  * `C_TOTAL_ENERGY` raw `== energy_scalar(result, scope=SCOPE_SYSTEM)`;
    `C_UE_ENERGY` raw `== energy_scalar(result, scope=SCOPE_REQUESTER)`;
  * a system-requested reference is never a mobile object (and vice versa);
  * every other consumer keeps its 4.2 scope and number.

## 3. E4.1 — telemetry, logging and closing inventory

* `energy_telemetry_v1` continues to report all three boundaries and the primary
  scope/value; with the primary config `primary_scope == "system"` and
  `primary_joules == system_joules`.
* `Average energy` remains the legacy MOBILE metric and must carry an explicit
  legacy label in the log/report; reporting consumers are NOT mass-converted to
  system.
* CSV keeps the five production columns and the legacy `Average energy,` column;
  episode-weighted aggregation and the header/quoting/late-field guarantees of
  E2.2 stand.
* The boundary-read inventory test stays green; the per-consumer scope table in
  the final report is regenerated from the code, not from memory.
* Acceptance: reward and schedule are unchanged by telemetry on/off; no second
  scheduler replay anywhere; the final suite is green.

## 4. Forbidden throughout E3/E4

* Any energy term inside the primary reward (latency-only means zero).
* Reading a scope from config for a consumer that has not been explicitly
  migrated (`configured_energy_scalar`).
* A system consumer silently falling back to a mobile reference.
* Reintroducing a direct `total_*_joules` read outside the accessor,
  serialization or the energy implementation.
* Training, pilot, deadline or shield runs; committing large artifacts.
