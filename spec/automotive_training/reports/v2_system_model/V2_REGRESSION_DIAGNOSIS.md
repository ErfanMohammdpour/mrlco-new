# V2_REGRESSION_DIAGNOSIS — failing helper-overload test, reproduced with a trace

Command (reproduces the failure through the test module's OWN fixtures):

```python
import test_v2_helper_model as T
dags = [T._dag(helper_id=i, compute=6e6) for i in range(12)]
for i, d in enumerate(dags): d.dag_id = "d%d" % i
helpers = {i: HelperState(i, 4e6, contact_end_s=100.0, predicted_contact_end_s=100.0) for i in range(12)}
compute = T.V2ComputeSpec(10e6, 1, tuple([1e6]*12), tuple([4e6]*12))
h = schedule_shared(dags, {d.dag_id: {0: 2} for d in dags}, link=T.LINK, compute=compute, helper_states=helpers)
m = schedule_shared(dags, {d.dag_id: {0: 1} for d in dags}, link=T.LINK, compute=compute, helper_states=helpers)
```

Observed (identical to the test's assertion values):

```
helper = 8.2869 s   mec = 7.3000 s
helper: V2V busy 6.789 s (utilisation 0.754 x makespan 8.287) | 24 recorded radio events
        per-DAG finishes [1.700, 2.300, 4.098, 4.697] s
mec:    MEC_CPU busy 7.200 s | 24 recorded radio events
helper events (first four): V2V 1000 B = 0.0001 s, V2V 2 000 000 B = 0.200 s, repeated
```

## Concrete unexplained discrepancy (the actual finding)

The 24 recorded helper transfers sum to `12 x 0.0001 s + 12 x 0.200 s = 2.4 s`, but the
shared V2V calendar reports **6.79 s busy** over a horizon of 8.287 s. Booked channel time is
therefore ~2.8x the time implied by the recorded events. Either the booking reserves intervals
the events do not describe (over-reservation), or events are under-recorded. Both would break
capacity accounting, queue waits, and any TX-energy ledger built on these events, so this must
be resolved before energy/constraints/gate work continues.

Direction of the arithmetic that explains the helper loss (to be confirmed after the booking
audit): the MEC plan serialises 12 x 6 MB at 10 MB/s = 7.2 s of CPU and about 2.4 s of radio
(the same 12 sink returns), while the helper plan parallelises the CPU (12 x 1.5 s, one helper
each) but serialises the SAME 12 x 0.2 s returns plus 12 ingress hops on one V2V channel, so
the helper can only win if its CPU saving exceeds the shared-radio cost. With 2 MB outputs and
10 MB/s V2V the radio term is 2.4 s, which should still leave the helper ahead (~3.9 s); the
observed 8.29 s is NOT explained by serialisation alone, which is why the booking discrepancy
above is the leading suspect.

## Status

* The test premise is NOT rejected: with the measured booking discrepancy removed, a
  long-contact idle helper SHOULD beat the overloaded MEC in this fixture.
* No threshold was relaxed; the failing test remains failing and is documented here plus in
  `V2_BUG_AUDIT.md`.
* Next concrete step: audit `Calendar.reserve` / `_earliest_fit` / `busy_intervals` against the
  recorded `radio_events` (per-server interval dump) to find the over-reservation, then re-run
  this exact reproduction as the regression.

---

# RESOLUTION (this round) — the diagnosis above was CORRECT in its measurement

The trace in this file measured the right thing: 24 helper transfers with 2.4012 s of recorded
service against 6.789 s of booked V2V channel time. The cause is now isolated exactly.

## Root cause (instrumented, `Calendar.reserve` call log)

`reserve_transfer` (a) passed `end - start` — which INCLUDES the queue wait — as the channel
duration, and (b) when the calendar shifted the start it called `Calendar.reserve` a **second
time without releasing the first interval**. For the failing fixture the shared V2V calendar
therefore held **46 reservations for 24 transfers**:

```
HELPER makespan 8.286900   V2V reserve calls: 46   sum of requested durations 6.789100
V2V events: 24   sum of (end - start) = 2.401200
e.g. ready=1.500300 dur=0.399800 -> [1.700100,2.099900]   (wait + service booked as service)
     ready=1.700100 dur=0.200000 -> [2.099900,2.299900]   (the SAME transfer booked again)
```

The 0.3998 s interval is `start0 + payload/rate - ready` (the wait plus the service), and the
0.2 s interval that follows is the re-reservation of the same transfer. Neither is shared-radio
serialisation.

## Fix

* `Calendar.plan_service(ready, duration_fn)`: non-mutating fixed point for
  `start = earliest_fit(ready, duration(start))` with an explicit tolerance (1e-9), an iteration
  bound (64), a feasibility re-check against the chosen server, and explicit failure
  (`V2ScheduleError` on a non-finite duration; `fixed_point_fallbacks` reported in
  `mechanics` when the best provably feasible candidate is returned instead of a converged one).
  No provisional reservation is ever left behind.
* `Calendar.reserve_at(idx, start, end)`: commits the proved interval exactly once.
* `validate_schedule` now enforces `occupancy == active service + outage` on every ledger
  record — the exact identity the old code violated — so this class of bug cannot return
  silently.

## Verified outcome (same fixture, same realization)

```
HELPER makespan 3.900100  (< MEC 7.300049)
V2V reserve calls: 24      sum of durations 2.401200   (= the recorded service, exactly)
test_v2_helper_model.py::TestHelperInScheduler::test_idle_helper_with_long_contact_beats_mec_when_mec_is_overloaded  PASSED
full non-TF suite: 1329 passed / 0 failed / 19 skipped
```

The fixture's premise (a long-contact idle helper beats an overloaded MEC) was therefore
CORRECT and was NOT relaxed; the threshold and the assertion are unchanged.

## Regression coverage added

`test_v2_booking.py` (15 tests) and `test_v2_contact_and_fallback.py` (12 tests): single
booking per transfer, occupancy identity, no overlap, capacity, queue-delayed start inside an
outage, multiple rate/outage boundaries with byte conservation, zero-service periods, long
payloads, horizon exhaustion, multi-hop/reverse routes, same-location zero radio energy,
convergence/failure of the fixed point, contact-margin monotonicity in both directions,
result-return-within-contact, no post-disconnection transfer, full-restart fallback with wasted
work, and task-local state isolation.
