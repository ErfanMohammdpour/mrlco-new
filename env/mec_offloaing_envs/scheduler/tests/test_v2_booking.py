#!/usr/bin/env python3
"""v2 booking invariants (stage C): one reservation per transfer, service integral,
capacity, adversarial outage/rate patterns, multi-hop and horizon exhaustion.

Regression target: the audited over-reservation in which `reserve_transfer`
  (a) booked `end - start` (queue wait + service) as the channel duration, and
  (b) re-reserved the SAME transfer a second time when the calendar shifted the start,
so the shared V2V calendar reported 6.79 s busy for 2.40 s of recorded service.
The scheduler now resolves start/duration with a non-mutating fixed point
(`Calendar.plan_service`) and commits exactly one interval (`Calendar.reserve_at`).
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spec.automotive_training.v2.helper_model import HelperState  # noqa: E402
from spec.automotive_training.v2.shared_scheduler import (  # noqa: E402
    Calendar, V2ComputeSpec, V2DAGSpec, V2LinkSpec, V2ScheduleError, V2TaskSpec,
    schedule_shared,
)

LINK = V2LinkSpec(20e6, 20e6, 10e6)


class _Regime:
    def __init__(self, dt):
        self.dt_s = dt
        self.name = "stub"
        self.evidence_class = "test_stub"
        self.estimation_sigma = 0.0


class _StubProcess:
    """Deterministic realized-multiplier stub: `fn(link, t) -> multiplier`."""

    def __init__(self, fn, dt=0.1):
        self._fn = fn
        self.regime = _Regime(dt)

    def realized(self, link, t):
        return float(self._fn(link, t))


def _compute(n_owner=1, n_helper=1, mec_workers=1):
    return V2ComputeSpec(10e6, mec_workers, tuple([1e6] * n_owner), tuple([4e6] * n_helper))


def _task(tid=0, compute=1e6, out=2e6, preds=(), root=False, sink=False, external=0.0,
          crit="MEDIUM"):
    return V2TaskSpec(tid, compute, out, tuple(preds), root, sink, crit, None, 0,
                      external_input_bytes=external)


class TestCalendarBooking(unittest.TestCase):
    def test_plan_service_does_not_mutate_the_calendar(self):
        cal = Calendar("c")
        cal.reserve(0.0, 1.0)                      # [0, 1)
        busy_before = list(cal.busy_intervals)
        idx, start, duration, end = cal.plan_service(0.0, lambda s: 0.5)
        self.assertEqual(cal.busy_intervals, busy_before,
                         "plan_service must not leave a provisional reservation behind")
        self.assertAlmostEqual(start, 1.0, places=12)
        self.assertAlmostEqual(duration, 0.5, places=12)
        self.assertAlmostEqual(end, 1.5, places=12)
        cal.reserve_at(idx, start, end)
        self.assertEqual(len(cal._intervals[idx]), 2)

    def test_plan_service_is_a_true_fixed_point_when_the_duration_depends_on_start(self):
        cal = Calendar("c")
        cal.reserve(0.0, 1.0)
        # duration shrinks once the start moves past 0.5 -> the naive "two-step" booking
        # would reserve 2.0 s of channel time; the fixed point reserves 0.1 s
        idx, start, duration, end = cal.plan_service(
            0.0, lambda s: 2.0 if s < 0.5 else 0.1)
        self.assertAlmostEqual(start, 1.0, places=12)
        self.assertAlmostEqual(duration, 0.1, places=12)
        self.assertAlmostEqual(end, 1.1, places=12)

    def test_plan_service_raises_instead_of_booking_an_impossible_interval(self):
        cal = Calendar("c")
        with self.assertRaises(V2ScheduleError):
            cal.plan_service(0.0, lambda s: math.inf)

    def test_reserve_at_rejects_an_invalid_interval(self):
        cal = Calendar("c")
        with self.assertRaises(V2ScheduleError):
            cal.reserve_at(0, 1.0, 0.5)


class TestSharedChannelBooking(unittest.TestCase):
    def _twelve_helper_dags(self):
        dags = []
        for i in range(12):
            d = V2DAGSpec("d%d" % i, 0, [_task(root=True, sink=True, external=1000, out=2e6)],
                          helper_id=i)
            dags.append(d)
        helpers = {i: HelperState(i, 4e6, contact_end_s=100.0, predicted_contact_end_s=100.0)
                   for i in range(12)}
        return dags, helpers

    def test_a_transfer_is_booked_exactly_once(self):
        dags, helpers = self._twelve_helper_dags()
        res = schedule_shared(dags, {d.dag_id: {0: 2} for d in dags}, link=LINK,
                              compute=_compute(n_helper=12), helper_states=helpers)
        v2v = [r for r in res.radio_ledger if r["hop"] == "V2V"]
        self.assertEqual(len(v2v), 24, "24 transfers -> 24 ledger records")
        self.assertEqual(res.mechanics["radio_events"], 24,
                         "24 transfers -> 24 radio events (no invisible extra reservations)")
        # constant rate: channel occupancy == active service, so busy time == bytes/rate
        expected = 12 * (1000 / 10e6) + 12 * (2e6 / 10e6)
        self.assertAlmostEqual(sum(r["end_s"] - r["start_s"] for r in v2v), expected, places=9)
        self.assertAlmostEqual(sum(r["service_s"] for r in v2v), expected, places=9)
        self.assertAlmostEqual(res.utilizations["V2V"], expected / res.makespan_s, places=9)
        self.assertEqual(res.mechanics["booking_fixed_point_fallbacks"], 0)

    def test_shared_channel_reservations_never_overlap(self):
        dags, helpers = self._twelve_helper_dags()
        res = schedule_shared(dags, {d.dag_id: {0: 2} for d in dags}, link=LINK,
                              compute=_compute(n_helper=12), helper_states=helpers)
        for hop in ("V2V", "MEC_DL", "MEC_UL"):
            iv = sorted((r["start_s"], r["end_s"]) for r in res.radio_ledger
                        if r["hop"] == hop)
            for (s1, e1), (s2, e2) in zip(iv, iv[1:]):
                self.assertGreaterEqual(s2 + 1e-12, e1, "overlapping %s reservations" % hop)

    def test_ledger_occupancy_equals_service_plus_outage(self):
        dags, helpers = self._twelve_helper_dags()
        res = schedule_shared(dags, {d.dag_id: {0: 2} for d in dags}, link=LINK,
                              compute=_compute(n_helper=12), helper_states=helpers)
        for r in res.radio_ledger:
            self.assertAlmostEqual(r["service_s"] + r["outage_s"],
                                   r["end_s"] - r["start_s"], places=9)
            self.assertGreaterEqual(r["queue_wait_s"], 0.0)

    def test_capacity_is_never_exceeded(self):
        dags, helpers = self._twelve_helper_dags()
        res = schedule_shared(dags, {d.dag_id: {0: 2} for d in dags}, link=LINK,
                              compute=_compute(n_helper=12), helper_states=helpers)
        busy = {}
        for r in res.radio_ledger:
            busy[r["hop"]] = busy.get(r["hop"], 0.0) + (r["end_s"] - r["start_s"])
        for hop, total in busy.items():
            self.assertLessEqual(total, res.makespan_s + 1e-9, "%s over capacity" % hop)


class TestAdversarialLinks(unittest.TestCase):
    def test_queued_transfer_that_starts_inside_an_outage_pauses(self):
        # DL rate is 20e6; DL is OUT during [0.3, 0.5). d0 holds DL in [0.1, 0.3), so d1's
        # return is queued to 0.3 and then runs straight into the outage.
        process = _StubProcess(lambda link, t: 0.0 if (link == "mec_dl" and 0.3 <= t < 0.5)
                               else 1.0)
        d0 = V2DAGSpec("d0", 0, [_task(root=True, sink=True, compute=1e6, out=4e6)])
        d1 = V2DAGSpec("d1", 0, [_task(root=True, sink=True, compute=1e6, out=4e6)])
        res = schedule_shared([d0, d1], {"d0": {0: 1}, "d1": {0: 1}}, link=LINK,
                              compute=_compute(), link_process=process)
        dl = sorted([r for r in res.radio_ledger if r["hop"] == "MEC_DL"],
                    key=lambda r: r["start_s"])
        self.assertEqual(len(dl), 2)
        queued = dl[-1]
        self.assertAlmostEqual(queued["queue_wait_s"], 0.1, places=9)
        self.assertAlmostEqual(queued["start_s"], 0.3, places=9)
        self.assertAlmostEqual(queued["service_s"], 0.2, places=9)
        self.assertAlmostEqual(queued["outage_s"], 0.2, places=9)
        self.assertAlmostEqual(queued["end_s"], 0.7, places=9)
        # the pause is NEVER counted as active TX
        self.assertAlmostEqual(queued["service_s"] + queued["outage_s"],
                               queued["end_s"] - queued["start_s"], places=9)
        self.assertGreater(res.queue_stats["outage_wait_total_s"], 0.0)

    def test_a_long_zero_service_period_costs_time_but_no_service(self):
        process = _StubProcess(lambda link, t: 0.0 if t < 1.0 else 1.0)
        dag = V2DAGSpec("d", 0, [_task(root=True, sink=True, compute=1e6, out=2e6)])
        res = schedule_shared([dag], {"d": {0: 1}}, link=LINK, compute=_compute(),
                              link_process=process)
        rec = [r for r in res.radio_ledger if r["hop"] == "MEC_DL"][-1]
        self.assertAlmostEqual(rec["outage_s"], 1.0, places=9)
        self.assertAlmostEqual(rec["service_s"], 2e6 / 20e6, places=9)
        self.assertGreaterEqual(res.makespan_s, 1.0 + 2e6 / 20e6 - 1e-9)

    def test_multiple_rate_and_outage_boundaries_are_integrated(self):
        # [0,0.1) full, [0.1,0.2) OUT, [0.2,0.3) HALF, [0.3,0.4) full, [0.4,0.5) OUT, then full.
        # 8 MB at 20 MB/s needs 0.4 s of full-rate service, spread over two outages:
        #   service = 0.1 + 0.1 + 0.1 + 0.1 + 0.05 = 0.45 s, outage = 0.2 s, span = 0.65 s
        def fn(link, t):
            if link != "mec_dl":
                return 1.0
            if 0.1 <= t < 0.2 or 0.4 <= t < 0.5:
                return 0.0
            if 0.2 <= t < 0.3:
                return 0.5
            return 1.0

        process = _StubProcess(fn)
        dag = V2DAGSpec("d", 0, [_task(root=True, sink=True, compute=0.0, out=8e6)])
        res = schedule_shared([dag], {"d": {0: 1}}, link=LINK, compute=_compute(),
                              link_process=process)
        rec = [r for r in res.radio_ledger if r["hop"] == "MEC_DL"][-1]
        self.assertAlmostEqual(rec["service_s"], 0.45, places=9)
        self.assertAlmostEqual(rec["outage_s"], 0.2, places=9)
        self.assertAlmostEqual(rec["end_s"] - rec["start_s"], 0.65, places=9)
        self.assertAlmostEqual(rec["end_s"] - rec["start_s"],
                               rec["service_s"] + rec["outage_s"], places=9)
        # byte conservation: rate x service time integrated over the segments == payload
        served = (0.1 * 20e6) + (0.1 * 10e6) + (0.1 * 20e6) + (0.1 * 20e6) + (0.05 * 20e6)
        self.assertAlmostEqual(served, rec["bytes"], places=3)

    def test_horizon_exhaustion_raises_instead_of_hanging(self):
        process = _StubProcess(lambda link, t: 0.0)
        dag = V2DAGSpec("d", 0, [_task(root=True, sink=True, compute=0.0, out=2e6)])
        with self.assertRaises(V2ScheduleError):
            schedule_shared([dag], {"d": {0: 1}}, link=LINK, compute=_compute(),
                            link_process=process, max_service_steps=64)

    def test_long_payload_completes_and_conserves_bytes(self):
        dag = V2DAGSpec("d", 0, [_task(root=True, sink=True, compute=1e6, out=1e9)])
        res = schedule_shared([dag], {"d": {0: 1}}, link=LINK, compute=_compute())
        rec = [r for r in res.radio_ledger if r["hop"] == "MEC_DL"][-1]
        self.assertAlmostEqual(rec["bytes"], 1e9, places=6)
        self.assertAlmostEqual(rec["service_s"], 1e9 / 20e6, places=6)


class TestRoutesAndDirections(unittest.TestCase):
    def test_multi_hop_reverse_direction_records_both_hops(self):
        # task0 on the helper, task1 on the MEC: HELPER -> MEC relays through the UE
        tasks = [_task(0, compute=1e6, out=1e6, root=True, external=1000),
                 _task(1, compute=1e6, out=1e6, preds=(0,), sink=True)]
        dag = V2DAGSpec("d", 0, tasks, helper_id=0)
        helper = HelperState(0, 4e6, contact_end_s=100.0, predicted_contact_end_s=100.0)
        res = schedule_shared([dag], {"d": {0: 2, 1: 1}}, link=LINK, compute=_compute(),
                              helper_states={0: helper})
        hops = [(r["task_id"], r["hop"]) for r in res.radio_ledger]
        self.assertIn((0, "V2V"), hops)            # UE -> HELPER ingress
        self.assertIn((1, "V2V"), hops)            # HELPER -> UE
        self.assertIn((1, "MEC_UL"), hops)         # UE -> MEC
        for r in res.radio_ledger:
            if r["hop"] == "V2V":
                self.assertIn(r["direction"], ("UE->HELPER", "HELPER->UE"))
            if r["hop"] == "MEC_UL":
                self.assertEqual(r["direction"], "UE->MEC")

    def test_same_location_transfer_has_no_radio_event_and_no_energy(self):
        tasks = [_task(0, compute=1e6, out=2e6, root=True, sink=False, external=0.0),
                 _task(1, compute=1e6, out=2e6, preds=(0,), sink=True)]
        dag = V2DAGSpec("d", 0, tasks)
        res = schedule_shared([dag], {"d": {0: 0, 1: 0}}, link=LINK, compute=_compute())
        self.assertEqual(res.radio_ledger, [],
                         "two UE-local tasks must produce no radio transfer at all")
        self.assertEqual(res.mechanics["radio_bytes"], 0.0)
        self.assertEqual(res.timings[("d", 1)].tx_service_s, 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
