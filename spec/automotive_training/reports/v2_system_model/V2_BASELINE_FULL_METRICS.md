# V2_BASELINE_FULL_METRICS — every metric for every method, model included

**Single source:** the v2 `V2HeldOutEvaluator` on 40 query graphs x 5 paired realizations
(`seeds = [1000, 1017, 1034, 1051, 1068]`). Every row is averaged over those realizations.
`rollouts = 200` for the model (40 graphs x 5); 200 plans per baseline candidate.

---

## 1. MAIN TABLE — model and baselines on the SAME worlds

| candidate | latency (s) | **system J** | requester J | mobile J | **miss** | tardy HIGH | V2V airtime (s) | cpu_wait (s) | fUE | fMEC | fV2V |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `candidate_oracle_s` | **0.028452** | 27.933 | 0.0146 | 0.0449 | **0.0000** | 0 | 0.00238 | 0.00083 | 0.103 | 0.877 | 0.020 |
| **`all_MEC`** | 0.029931 | **29.483** | 0.0000 | 0.0000 | **0.0000** | **0** | 0.00000 | 0.00114 | 0.000 | 1.000 | 0.000 |
| `greedy_coordinate_descent_MC` | 0.031239 | **27.190** | 0.0224 | 0.0749 | 0.0003 | low | 0.00440 | 0.00067 | 0.145 | 0.823 | 0.033 |
| `heft_reference_v2` | 0.032665 | 27.890 | 0.0131 | 0.0848 | 0.0013 | low | 0.00270 | 0.00037 | 0.084 | 0.865 | 0.051 |
| **MODEL — k=0** | **0.080600** | **69.278** | 0.0330 | 0.5933 | **0.0000** | **3.4/400** | n/a | n/a | 0.446 | 0.352 | 0.201 |
| **MODEL — k=3** | 0.082751 | **68.982** | 0.0388 | 0.5776 | **0.0000** | 3.4/400 | n/a | n/a | ~ | ~ | ~ |
| `all_HELPER` | 0.202054 | 3.399 | 0.0000 | 3.3986 | **0.1418** | high | 0.01325 | 0.00776 | 0.000 | 0.000 | 1.000 |
| `all_UE` | 0.319593 | 0.320 | 0.3196 | 0.3196 | **0.3350** | high | 0.00000 | 0.01244 | 1.000 | 0.000 | 0.000 |

Model action mix is the training-log mean (`action_fraction/*`); the evaluator does not expose a
per-candidate mix for the model path yet, so the k=3 mix is not separately measured.

## 2. THE DECISIVE RESULT — the model loses on BOTH axes at once

```
                        latency (s)     system J
all_MEC                   0.029931        29.483
MODEL k=0                 0.080600        69.278
MODEL k=3                 0.082751        68.982

latency  : 2.69x slower
energy   : 2.35x MORE
```

**The model is dominated: slower AND more energy-hungry than the trivial all-MEC baseline.**
There is no trade-off to excuse it. It is not "buying" latency with energy, or energy with
latency — it is worse on both.

**Why that is possible:** the model spreads compute as `local 0.446 / MEC 0.352 / v2v 0.201`,
whereas all-MEC puts everything on the fastest processor. Moving a task to a slow local CPU
costs latency AND still costs energy — so a hedged, off-MEC mix can be Pareto-dominated by
all-MEC. That is exactly what the measurement shows.

## 3. STILL THE TRADE-OFF CURVE AMONG THE OTHER CANDIDATES

```
                 latency(s)   system_J   miss
all_UE             0.319593      0.320    0.3350
all_HELPER         0.202054      3.399    0.1418
greedy_cd          0.031239     27.190    0.0003
HEFT               0.032665     27.890    0.0013
all_MEC            0.029931     29.483    0.0000
oracle             0.028452     27.933    0.0000
```
- `all_MEC` is simultaneously **fastest** and **most expensive** among the non-trivial methods:
  29.483 J against 27.190 J for greedy coordinate descent = **7.8% energy saved for 4.4%
  latency**.
- `all_UE` and `all_HELPER` look cheap (0.320 J and 3.399 J) but lose **33.5%** and **14.2%**
  of deadlines. Their cheapness is a hidden reliability cost.

## 4. `requester_joules` is a reporting trap

```
all_MEC:  requester 0.000 J   system 29.483 J
all_UE:   requester 0.320 J   system  0.320 J
```
`all_MEC` shows **zero requester energy** because all compute sits on MEC, while it spends
29.483 J there. Reporting `requester_joules` as "the energy" makes all-MEC look free. Always
report `system_joules`, and label `requester` as the user's battery boundary only.

---

## 5. Two defects found and fixed on this path

### B1 — `deadline_miss_rate` was hard-coded to 0.0
`spec/automotive_training/v2/eval_loop.py:131` returned a literal `0.0`, so **every baseline
reported a perfect deadline record that was never measured.** It lied in the FAVOURABLE
direction, which is why it survived. Now computed per criticality from the same
`ScheduleResult`: all_UE 0.3350, all_HELPER 0.1418, HEFT 0.0013, greedy_cd 0.0003, all_MEC 0.0.

### B2 — the model's energy was measured on every rollout and then discarded
`spec/automotive_training/automotive_primary.py::per_graph_makespans` already had the
per-episode `energy_telemetry` record in hand — with `system_joules`, `requester_joules` and
`mobile_joules` — and copied out only latency and deadline fields. `rollout_metric_block`
therefore never reported energy, and the evaluator aggregate carried no energy key at all.
That is the entire reason the baselines had joules and the model did not. Both functions now
carry the three boundaries plus `primary_scope`, `constraint_penalty` and `n_energy_episodes`.

### B3 — a label was being averaged as a number
`evaluate` averaged every key of the metric block with `float()`, so the new `primary_scope`
("system") raised `ValueError: could not convert string to float: 'system'`. Labels are now
listed in `_NON_NUMERIC_METRIC_KEYS` and passed through.

---

## 6. Modelling boundary (state in the paper)

`system_joules = UE CPU + helper CPU + MEC CPU + UE/helper/MEC TX`

**Not modelled** (from `run.json.unmodeled_components`):
```
receiver / static radio power (include_rx_energy=False)
CPU idle / leakage power between tasks
MEC / RSU static (non-compute) power (server_static_power_w is null)
backhaul / core-network transport energy
battery state dynamics, charging efficiency, reserve accounting
warm-standby reservation energy
```
`system_joules` is a **lower bound**; do not present the absolute values as real consumption.

---

## 7. What to put in the paper

**Columns:** `latency (mean) | system energy | miss rate | tardy HIGH | fMEC`

**Three sentences that the data forces:**
1. "all-MEC is simultaneously the fastest and the most energy-expensive non-trivial method:
   29.48 J against 27.19 J for greedy coordinate descent — 7.8% energy saved for 4.4% latency."
2. "all-UE has the lowest system energy (0.32 J) but forfeits 33.5% of deadlines; its cheapness
   is a hidden reliability cost."
3. "The learned policy is dominated by all-MEC on both axes: 2.69x the latency at 2.35x the
   energy. The failure is not a trade-off choice, it is a positioning failure."

## 8. Claim scope
- One training seed, two checkpoints, `decoding=deterministic`, 5 realizations.
- `candidate_oracle_s` is a **latency-oracle** (best-of-panel on makespan) even though it shows
  `miss = 0.0` here.
- Model tardiness comes from the evaluator aggregate; baseline tardiness from the panel's
  `n_tardy_high` / `n_tardy_medium` (present in the raw JSON).
- B1 affected **baselines only**; the model's own reported tardiness came from
  `rollout_metric_block` and was always real.
- The model's `system_joules` here is the `role="validation"` foreground ledger, which does not
  subtract the background term the way the training telemetry's `energy/system_joules` column
  does — do not compare 69.278 J against the training-log mean 82.74 J. Only compare it against
  the panel, which is built the same way.
