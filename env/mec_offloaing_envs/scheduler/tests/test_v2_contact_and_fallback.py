#!/usr/bin/env python3
"""v2 contact-window, fallback and task-local-state semantics (stage C3).

Audited defects covered here:
  * 4.3 contact margin `min(1, need/window)` was NON-monotone (a longer contact reduced the
        predicted success probability);
  * 4.5 `locals().get("restart_penalty"/"contact_failure")` let one task's failure state leak
        into the NEXT task's timing record;
  * 4.6 checking helper CPU completion alone ignored whether the RESULT could be returned
        before the contact ended;
  * 4.4/4.8 recovery booked a HELPER -> UE transfer at `contact_end`, i.e. AFTER the link was
        gone, and treated the remainder as freely recoverable.
Declared model: an interrupted transmission retains the channel; when contact ends before the
result is back, the remote work is WASTED and the task restarts locally IN FULL because v2
implements no checkpointing.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spec.automotive_training.v2.helper_model import HelperState  # noqa: E402
from spec.automotive_training.v2.reliability import (  # noqa: E402
    contact_slack, make_gate, remote_success_probability,
)
from spec.automotive_training.v2.shared_scheduler import (  # noqa: E402
    V2ComputeSpec, V2DAGSpec, V2LinkSpec, V2TaskSpec, schedule_shared,
)

LINK = V2LinkSpec(20e6, 20e6, 10e6)
COMPUTE = V2ComputeSpec(10e6, 1, (1e6,), (4e6,))


def _one_task(compute=6e6, out=2e6, external=1000, crit="MEDIUM"):
    return V2DAGSpec("d", 0, [V2TaskSpec(0, compute, out, (), True, True, crit, None, 0,
                                        external_input_bytes=external)], helper_id=0)


def _helper(contact_end, predicted=None):
    return {0: HelperState(0, 4e6, contact_end_s=contact_end,
                           predicted_contact_end_s=(1.15 * contact_end if predicted is None
                                                    else predicted))}


class TestContactSlackMonotonicity(unittest.TestCase):
    def test_slack_is_non_decreasing_in_the_contact_window(self):
        values = [contact_slack(predicted_contact_end_s=w, now_s=0.0, payload_in_bytes=1000.0,
                                compute_bytes=6e6, v2v_bytes_per_s=10e6,
                                helper_bytes_per_s=4e6, output_bytes=2e6)
                  for w in (0.5, 1.0, 1.5, 1.7, 2.0, 5.0, 50.0)]
        self.assertEqual(values, sorted(values), "longer contact must not lower the slack")
        self.assertAlmostEqual(values[0], 0.5 / 1.7002, places=9)
        self.assertEqual(values[-1], 1.0)

    def test_infinite_and_absent_contact_windows_are_full_slack(self):
        for predicted in (None, float("inf")):
            self.assertEqual(contact_slack(predicted_contact_end_s=predicted, now_s=0.0,
                                           payload_in_bytes=1000.0, compute_bytes=6e6,
                                           v2v_bytes_per_s=10e6, helper_bytes_per_s=4e6),
                             1.0)

    def test_success_probability_is_non_decreasing_in_the_contact_factor(self):
        p = [remote_success_probability(link_confidence=1.0, outage_fraction=0.0,
                                        contact_margin=m, location="HELPER")
             for m in (0.0, 0.25, 0.5, 0.9, 1.0)]
        self.assertEqual(p, sorted(p))


class TestResultMustReturnBeforeContactEnds(unittest.TestCase):
    def test_cpu_fits_but_the_result_does_not(self):
        # helper CPU needs 1.5 s and the 2 MB result needs 0.2 s of V2V, so a 1.6 s contact
        # lets the CPU finish but NOT the return. The old code accepted this placement.
        res = schedule_shared([_one_task()], {"d": {0: 2}}, link=LINK, compute=COMPUTE,
                              helper_states=_helper(1.6))
        self.assertEqual(res.queue_stats["helper_contact_failures"], 1,
                         "CPU completion alone must not be enough")
        tm = res.timings[("d", 0)]
        self.assertEqual(tm.attempted_location, "HELPER")
        self.assertEqual(tm.location, "UE")
        self.assertTrue(tm.contact_failure)

    def test_a_contact_long_enough_for_the_return_succeeds(self):
        res = schedule_shared([_one_task()], {"d": {0: 2}}, link=LINK, compute=COMPUTE,
                              helper_states=_helper(2.0))
        self.assertEqual(res.queue_stats["helper_contact_failures"], 0)
        self.assertEqual(res.timings[("d", 0)].location, "HELPER")

    def test_no_helper_to_ue_transfer_is_booked_after_disconnection(self):
        res = schedule_shared([_one_task()], {"d": {0: 2}}, link=LINK, compute=COMPUTE,
                              helper_states=_helper(1.6))
        for rec in res.radio_ledger:
            if rec["src"] == "HELPER":
                self.fail("a HELPER -> %s transfer exists after the link was gone: %r"
                          % (rec["dst"], rec))

    def test_wasted_work_and_full_restart_are_recorded(self):
        res = schedule_shared([_one_task(compute=6e6)], {"d": {0: 2}}, link=LINK,
                              compute=COMPUTE, helper_states=_helper(1.6))
        qs = res.queue_stats
        self.assertGreater(qs["wasted_work_bytes_total"], 0.0)
        self.assertAlmostEqual(qs["restart_work_bytes_total"], 6e6, places=6)
        self.assertGreater(qs["helper_restart_penalty_s"], 0.0)
        tm = res.timings[("d", 0)]
        self.assertAlmostEqual(tm.cpu_restart_bytes, 6e6, places=6)
        self.assertAlmostEqual(tm.restart_penalty_s, 6e6 / 1e6, places=9)
        self.assertAlmostEqual(tm.cpu_wasted_bytes, tm.cpu_executed_bytes, places=6)

    def test_checkpoint_transfer_is_labelled_unsupported_not_free(self):
        res = schedule_shared([_one_task()], {"d": {0: 2}}, link=LINK, compute=COMPUTE,
                              helper_states=_helper(1.6))
        self.assertEqual(res.queue_stats["checkpoint_transfer_bytes_total"], 0.0)
        self.assertEqual(res.timings[("d", 0)].checkpoint_transfer_bytes, 0.0)


class TestTaskLocalState(unittest.TestCase):
    def _two_task_dag(self):
        # task 0 on the helper fails contact; task 1 runs on the UE (no contact at all)
        t0 = V2TaskSpec(0, 6e6, 2e6, (), True, False, "HIGH", None, 0, external_input_bytes=1000)
        t1 = V2TaskSpec(1, 1e6, 2e6, (0,), False, True, "HIGH", None, 0)
        return V2DAGSpec("d", 0, [t0, t1], helper_id=0)

    def test_a_failure_does_not_leak_into_the_next_task(self):
        res = schedule_shared([self._two_task_dag()], {"d": {0: 2, 1: 0}}, link=LINK,
                              compute=COMPUTE, helper_states=_helper(1.6))
        first, second = res.timings[("d", 0)], res.timings[("d", 1)]
        self.assertTrue(first.contact_failure)
        self.assertFalse(second.contact_failure,
                         "task 0's contact failure leaked into task 1")
        self.assertEqual(second.restart_penalty_s, 0.0)
        self.assertEqual(second.fallback_used_s, 0.0)
        self.assertEqual(second.cpu_wasted_bytes, 0.0)
        self.assertEqual(second.attempted_location, "UE")

    def test_every_task_starts_with_clean_fallback_state(self):
        # same DAG but the FAILING task runs second: nothing may be inherited in either order
        t0 = V2TaskSpec(0, 1e6, 2e6, (), True, False, "HIGH", None, 0, external_input_bytes=1000)
        t1 = V2TaskSpec(1, 6e6, 2e6, (0,), False, True, "HIGH", None, 0)
        dag = V2DAGSpec("d", 0, [t0, t1], helper_id=0)
        # 2.0 s passes helper admissibility for task 1 (need 1.9 s, predicted 2.3 s) but the
        # helper CPU only finishes at 2.7 s, so the task must fail contact
        res = schedule_shared([dag], {"d": {0: 0, 1: 2}}, link=LINK, compute=COMPUTE,
                              helper_states=_helper(2.0))
        self.assertFalse(res.timings[("d", 0)].contact_failure)
        self.assertEqual(res.timings[("d", 0)].restart_penalty_s, 0.0)
        self.assertTrue(res.timings[("d", 1)].contact_failure)


class TestReliabilityGateMonotonicityEndToEnd(unittest.TestCase):
    def test_more_predicted_contact_never_creates_more_rejections(self):
        gate = make_gate()
        evidence = {"link_confidence": 1.0, "outage_fraction": 0.0}
        rejections = []
        for realized in (1.45, 1.46, 1.47, 1.48, 1.50, 2.0, 5.0):
            res = schedule_shared([_one_task(crit="HIGH")], {"d": {0: 2}}, link=LINK,
                                  compute=COMPUTE, helper_states=_helper(realized),
                                  reliability_gate=gate, reliability_evidence=evidence)
            rejections.append(res.queue_stats["reliability_rejections"])
        self.assertEqual(rejections, sorted(rejections, reverse=True),
                         "a longer contact window must not add reliability rejections")
        self.assertEqual(rejections[0], 1, "a marginal window must still be rejected")
        self.assertEqual(rejections[-1], 0)

    def test_contact_factor_is_informative_inside_the_admissible_region(self):
        # this is the regression for "the factor is identically 1 and carries nothing":
        # the placement passes helper admissibility but the round-trip slack is < 1
        res = schedule_shared([_one_task(crit="HIGH")], {"d": {0: 2}}, link=LINK,
                              compute=COMPUTE, helper_states=_helper(1.45),
                              reliability_gate=make_gate(),
                              reliability_evidence={"link_confidence": 1.0,
                                                    "outage_fraction": 0.0})
        tm = res.timings[("d", 0)]
        self.assertFalse(tm.helper_rejected_inadmissible,
                         "the fixture must pass helper ADMISSIBILITY")
        self.assertTrue(tm.reliability_rejected,
                        "the round-trip contact slack must be able to reject")


if __name__ == "__main__":
    unittest.main(verbosity=2)
