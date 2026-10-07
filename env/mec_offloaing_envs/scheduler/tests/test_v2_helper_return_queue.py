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
    V2ComputeSpec, V2DAGSpec, V2LinkSpec, V2TaskSpec, schedule_shared,
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
        from spec.automotive_training.v2.shared_scheduler import V2ScheduleError
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
