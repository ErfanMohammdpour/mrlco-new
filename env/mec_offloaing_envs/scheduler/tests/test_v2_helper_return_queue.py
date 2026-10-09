#!/usr/bin/env python3
"""F07: a helper result is useful only if its return completes inside the contact window.

The audited scheduler estimated the return with the queue-free `_service_detail`, so another
DAG's V2V booking was invisible to the feasibility check: the helper completed, its return was
booked AFTER contact_end, and the schedule validated with contact_failure=False. This test pins
the repair: the return must be planned against the actual shared calendars.
"""
import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spec.automotive_training.v2.shared_scheduler import (  # noqa: E402
    Calendar, V2ComputeSpec, V2DAGSpec, V2LinkSpec, V2ScheduleError, V2TaskSpec,
    schedule_shared, validate_schedule,
)
from spec.automotive_training.v2.helper_model import HelperState  # noqa: E402


class HelperReturnQueueTests(unittest.TestCase):
    def test_return_behind_another_v2v_booking_falls_back(self):
        LINK = V2LinkSpec(20e6, 20e6, 100.0)
        COMPUTE = V2ComputeSpec(10e6, 1, (1e6, 1e6), (1e6, 1e6))
        bg = V2DAGSpec("bg", 0, [V2TaskSpec(0, 1.0, 1.0, (), True, True, "LOW", None, 0,
                                            external_input_bytes=40.0)], helper_id=1)
        fg = V2DAGSpec("fg", 1, [V2TaskSpec(0, 100.0, 10.0, (), True, True, "HIGH", None, 0)],
                       helper_id=0)
        helper = {0: HelperState(0, 1e6, contact_end_s=0.5, predicted_contact_end_s=1.0),
                  1: HelperState(1, 1e6, contact_end_s=5.0, predicted_contact_end_s=1.0)}
        res = schedule_shared([bg, fg], {"bg": [2], "fg": [2]}, link=LINK, compute=COMPUTE,
                              helper_states=helper, validate=True)
        tm = res.timings[("fg", 0)]
        # the return cannot fit before contact_end (background holds V2V until 0.4, then the
        # 40 B return would finish after 0.5), so the task must not be reported as a successful
        # HELPER execution.
        self.assertNotEqual(tm.location, "HELPER",
                            "helper result booked after contact_end was reported successful")
        self.assertTrue(tm.contact_failure, "the late return was not flagged as contact failure")

    def test_uncontended_return_still_succeeds(self):
        LINK = V2LinkSpec(20e6, 20e6, 100.0)
        COMPUTE = V2ComputeSpec(10e6, 1, (1e6,), (1e6,))
        fg = V2DAGSpec("fg", 0, [V2TaskSpec(0, 100.0, 10.0, (), True, True, "HIGH", None, 0)],
                       helper_id=0)
        helper = {0: HelperState(0, 1e6, contact_end_s=0.5, predicted_contact_end_s=1.0)}
        res = schedule_shared([fg], {"fg": [2]}, link=LINK, compute=COMPUTE,
                              helper_states=helper, validate=True)
        tm = res.timings[("fg", 0)]
        self.assertEqual(tm.location, "HELPER")
        self.assertFalse(tm.contact_failure)
        self.assertLessEqual(tm.finish_s, 0.5 + 1e-9)


if __name__ == "__main__":
    unittest.main(verbosity=2)


class ValidatorContactInvariantTests(unittest.TestCase):
    """F08: the validator must REJECT a helper transfer that leaves after contact ends.

    The pre-fix validator checked precedence, calendars, ledger, completion and utilization but
    never the contact window, so a late return passed validation.
    """

    def _schedule(self, contact_end, validate=True):
        LINK = V2LinkSpec(20e6, 20e6, 100.0)
        COMPUTE = V2ComputeSpec(10e6, 1, (1e6,), (1e6,))
        fg = V2DAGSpec("fg", 0, [V2TaskSpec(0, 100.0, 10.0, (), True, True, "HIGH", None, 0)],
                       helper_id=0)
        helper = {0: HelperState(0, 1e6, contact_end_s=contact_end,
                                 predicted_contact_end_s=5.0)}
        return schedule_shared([fg], {"fg": [2]}, link=LINK, compute=COMPUTE,
                               helper_states=helper, validate=validate)

    def test_validator_reports_the_contact_check(self):
        res = self._schedule(contact_end=5.0)
        self.assertIn("helper_contact_window", res.invariants)
        self.assertGreaterEqual(res.invariants["helper_transfers_checked"], 1)

    def test_late_helper_transfer_is_rejected_by_the_validator(self):
        """A HELPER transfer ending after contact_end must raise, not silently pass."""
        # contact ends at 0.05 s; the 10 B return at 100 B/s needs 0.1 s, so it cannot fit.
        # The scheduler must therefore NOT report a successful helper execution; if it did, the
        # validator has to reject it. Either way a late HELPER transfer must never validate.
        try:
            res = self._schedule(contact_end=0.05)
        except V2ScheduleError:
            return                      # rejected during planning -> also acceptable
        for rec in res.radio_ledger:
            if str(rec.get("src")) == "HELPER":
                self.assertLessEqual(float(rec["end_s"]), 0.05 + 1e-9)


class ContactStartWindowTests(unittest.TestCase):
    """F08/f: the contact window has a START too, and it binds ingress as well.

    The validator checked only `src == HELPER` records, so `UE -> HELPER` ingress booked before
    the window opened (or after it closed) was invisible, and no test ever set
    `contact_start_s > 0`.
    """

    LINK = V2LinkSpec(20e6, 20e6, 10e6)
    COMPUTE = V2ComputeSpec(10e6, 1, (1e6,), (4e6,))

    def _dag(self):
        return V2DAGSpec("d", 0, [V2TaskSpec(0, 1e6, 1e6, (), True, True, "MEDIUM", None, 0,
                                             external_input_bytes=1000.0)], helper_id=0)

    def _helpers(self):
        return {0: HelperState(0, 4e6, contact_start_s=0.5, contact_end_s=5.0,
                               predicted_contact_end_s=5.75)}

    def test_ingress_waits_for_the_contact_window_to_open(self):
        res = schedule_shared([self._dag()], {"d": {0: 2}}, link=self.LINK,
                              compute=self.COMPUTE, helper_states=self._helpers())
        ingress = [r for r in res.radio_ledger if r["dst"] == "HELPER"]
        self.assertEqual(len(ingress), 1)
        self.assertGreaterEqual(ingress[0]["start_s"], 0.5 - 1e-9,
                                "ingress was transmitted before the helper was in contact")

    def test_validator_rejects_ingress_outside_the_contact_window(self):
        res = schedule_shared([self._dag()], {"d": {0: 2}}, link=self.LINK,
                              compute=self.COMPUTE, helper_states=self._helpers())
        # inject a window-violating ingress record (well formed for the ledger check)
        res.radio_ledger.append({
            "dag_id": "d", "task_id": 0, "hop": "V2V", "direction": "UE->HELPER",
            "src": "UE", "dst": "HELPER", "bytes": 1000.0, "start_s": 0.1, "end_s": 0.2,
            "service_s": 0.1, "outage_s": 0.0, "queue_wait_s": 0.0, "outage_events": 0,
            "retained_channel": True})
        with self.assertRaises(V2ScheduleError):
            validate_schedule([self._dag()], {"d": {0: 2}}, self.LINK, res,
                              Calendar("MEC_CPU"), {0: Calendar("UE_CPU_0")},
                              {0: Calendar("HELPER_CPU_0")}, Calendar("MEC_UL"),
                              Calendar("MEC_DL"), Calendar("V2V"),
                              helper_states=self._helpers())


class DirectV2IReturnFeasibilityTests(unittest.TestCase):
    """F08/d: the fallback pre-check must use the route the data ACTUALLY takes.

    With `direct_helper_v2i=True` the route from the helper to a MEC successor is a single slow
    `MEC_ULH` hop, not `HELPER -> UE`. The pre-fix pre-check planned `HELPER -> UE` unconditionally,
    so it passed and the successor then booked a helper departure after `contact_end`:
    `validate=True` raised (`helper 0 transfer for d/1 completes at 10.100000000, after contact
    ends at 1.000000000`) while `validate=False` silently accepted it.
    """

    LINK = V2LinkSpec(1.0, 20e6, 100.0, direct_helper_v2i=True)
    COMPUTE = V2ComputeSpec(10e6, 1, (1e6,), (1e6,))

    def _world(self):
        t0 = V2TaskSpec(0, 100.0, 10.0, (), True, False, "HIGH", None, 0)
        t1 = V2TaskSpec(1, 10.0, 10.0, (0,), False, True, "HIGH", None, 0)
        dag = V2DAGSpec("d", 0, [t0, t1], helper_id=0)
        hel = {0: HelperState(0, 1e6, contact_end_s=1.0, predicted_contact_end_s=1.15)}
        return dag, hel

    def _no_post_contact_departure(self, res):
        for rec in res.radio_ledger:
            if rec["src"] == "HELPER":
                self.assertLessEqual(rec["end_s"], 1.0 + 1e-9,
                                     "a HELPER departure was scheduled after contact_end")

    def test_direct_v2i_successor_forces_the_fallback_under_validation(self):
        dag, hel = self._world()
        res = schedule_shared([dag], {"d": {0: 2, 1: 1}}, link=self.LINK, compute=self.COMPUTE,
                              helper_states=hel, validate=True)
        self.assertTrue(res.timings[("d", 0)].contact_failure,
                        "the direct V2I return cannot leave the helper before contact ends")
        self.assertEqual(res.timings[("d", 0)].location, "UE")
        self._no_post_contact_departure(res)

    def test_validate_false_no_longer_emits_a_post_contact_departure(self):
        dag, hel = self._world()
        res = schedule_shared([dag], {"d": {0: 2, 1: 1}}, link=self.LINK, compute=self.COMPUTE,
                              helper_states=hel, validate=False)
        self.assertTrue(res.timings[("d", 0)].contact_failure)
        self._no_post_contact_departure(res)
