#!/usr/bin/env python3
"""F08/g: a HELPER placement is only committed when its result can LEAVE the helper.

The 1500-iteration campaign aborted at 321 with

    V2ScheduleError: helper 0 transfer for ep1285_slot17_fg/14 (HELPER -> UE) completes at
    0.386488626, after contact ends at 0.334854754: no link to the helper exists after
    disconnection

i.e. the PLANNER committed a HELPER -> UE relay that could not finish before the helper's
`contact_end`, and the validator (run with `validate=True`) then aborted the run. Two audited
dead-ends were live:

(4) the ingress feasibility probe planned each predecessor route INDEPENDENTLY, so with several
    simultaneous ingresses to one helper the second booked route could still exceed
    `contact_end`;
(5) a fallback whose predecessor was stranded on a helper that disconnected before the relay
    could complete was booked anyway and rejected by the validator.

What this file pins (pre-fix behaviour of each fixture is recorded in the class docstrings; the
one that reproduces the abort verbatim is `TestReportedShapeOvershoot`):
  * the exact reported shape -- a 0.334854754 s window and a relay that would end at 0.3865 s --
    is planned on the UE, and `validate_schedule` accepts the resulting schedule
    (pre-fix: `V2ScheduleError` under `validate=True`, a post-contact relay under
    `validate=False`);
  * two successive ingresses plus an egress in one short window: only the ingress that
    actually fits the serialised V2V channel is placed on the helper, and no helper transfer
    leaves the window;
  * a successor whose HELPER-resident predecessor can no longer reach the requester's UE is
    downgraded before its ingress is committed, which cascades the predecessor into its own
    local restart -- no stranded data, no fabricated transfer, DAG still valid (pre-fix:
    `V2ScheduleError: helper 0 transfer for d/1 (HELPER -> UE) completes at 0.500100000, after
    contact ends at 0.500000000`);
  * `validate=False` (the prefix/telescoping path PPO rewards are computed on) produces the
    SAME admissible plan as `validate=True`, and its ledger re-validates clean;
  * the planner-side counters (`helper_egress_infeasible`, `helper_ingress_infeasible`) fire,
    and stay at zero for a fixture whose placement is admissible;
  * the F08 work/energy conservation identities still hold for the downgraded fixtures.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spec.automotive_training.v2.helper_model import HelperState  # noqa: E402
from spec.automotive_training.v2.shared_scheduler import (  # noqa: E402
    HELPER, UE, Calendar, V2ComputeSpec, V2DAGSpec, V2LinkSpec, V2ScheduleError,
    V2TaskSpec, schedule_shared, validate_schedule,
)

try:      # the slack the planner gates and `validate_schedule` share (pre-fix builds lack it)
    from spec.automotive_training.v2.shared_scheduler import CONTACT_SLACK_S  # noqa: E402
except ImportError:  # pragma: no cover - the value the pre-fix validator already hard-codes
    CONTACT_SLACK_S = 1e-9

LINK = V2LinkSpec(20e6, 20e6, 10e6)
CONTACT_END = 0.334854754            # the reported window length


def _task(tid=0, compute=1e6, out=1e6, preds=(), root=False, sink=False, external=0.0,
          crit="MEDIUM"):
    return V2TaskSpec(tid, compute, out, tuple(preds), root, sink, crit, None, 0,
                      external_input_bytes=external)


def _dag(dag_id, tasks, helper_id=0, arrival=0.0):
    return V2DAGSpec(dag_id, 0, tasks, arrival_s=arrival, helper_id=helper_id,
                     helper_contact_window=(0.0, CONTACT_END))


def _helper(contact_end, predicted=None, hid=0, cpu=4e6):
    return {hid: HelperState(hid, cpu, contact_end_s=contact_end,
                             predicted_contact_end_s=(1.6 * contact_end if predicted is None
                                                      else predicted))}


def _assert_helper_records_inside_window(test, res, contact_end, contact_start=0.0):
    """Every ledger record touching a helper must sit inside its contact window.

    This is the invariant the validator enforces; the test uses the SAME slack the planner and
    the validator share (`CONTACT_SLACK_S`), so the assertions cannot disagree with either.
    """
    for rec in res.radio_ledger:
        start, end = float(rec["start_s"]), float(rec["end_s"])
        if rec["src"] != HELPER and rec["dst"] != HELPER:
            continue
        if rec["src"] == HELPER:
            test.assertLessEqual(
                end, float(contact_end) + CONTACT_SLACK_S,
                "a HELPER -> %s departure was booked after contact_end: %r"
                % (rec["dst"], rec))
        if rec["dst"] == HELPER:
            test.assertLessEqual(
                start, float(contact_end) + CONTACT_SLACK_S,
                "a transfer INTO the helper was booked after contact_end: %r" % (rec,))
        test.assertGreaterEqual(
            start, float(contact_start) - CONTACT_SLACK_S,
            "a helper transfer was booked before contact_start: %r" % (rec,))
        test.assertGreaterEqual(end, start - CONTACT_SLACK_S,
                                "inverted interval: %r" % (rec,))

def _assert_ready_times_are_resident(test, dags, res):
    """Each task's `ready_s` must be backed by its predecessors' FINAL locations.

    When a predecessor ended on the HELPER and the consumer runs on the UE, a booked
    `HELPER -> UE` relay must witness the data movement; and in every case the consumer may
    not claim data before the producer finished.
    """
    if not isinstance(dags, list):
        dags = [dags]
    for dag in dags:
        by_id = {t.task_id: t for t in dag.tasks}
        for t in dag.tasks:
            tm = res.timings[(dag.dag_id, t.task_id)]
            for p in t.predecessors:
                ptm = res.timings[(dag.dag_id, p)]
                test.assertGreaterEqual(float(tm.ready_s) + 1e-9, float(ptm.finish_s),
                                        "consumer ready before its producer finished")
                if ptm.location == tm.location or float(by_id[p].output_bytes) <= 0.0:
                    continue
                if ptm.location == HELPER and tm.location == UE:
                    relays = [r for r in res.radio_ledger
                              if str(r["dag_id"]) == dag.dag_id
                              and str(r["src"]) == HELPER and str(r["dst"]) == UE]
                    test.assertTrue(
                        relays, "consumer %s/%s restarts at the UE without the HELPER -> UE "
                                "relay of %s/%s" % (dag.dag_id, t.task_id, dag.dag_id, p))


def _validate(dag, res, helper_states, compute):
    return validate_schedule([dag], {dag.dag_id: {}}, LINK, res, Calendar("MEC_CPU"),
                             {0: Calendar("UE_CPU_0")},
                             {h: Calendar("HELPER_CPU_%d" % h) for h in helper_states},
                             Calendar("MEC_UL"), Calendar("MEC_DL"), Calendar("V2V"),
                             helper_states=helper_states, compute=compute)


class TestReportedShapeOvershoot(unittest.TestCase):
    """The exact reported abort: a ~0.3349 s window and a ~0.05 s late relay.

    Task 0 runs on the helper (0.2865 s of compute after a 1000-byte ingress) and task 1
    follows it there and fails its own sink return. The pre-fix planner then booked
    `HELPER -> UE` for task 0's 1 MB result at 0.3865 s -- 0.0516 s after `contact_end` -- and
    `validate=True` raised the reported error, aborting the campaign:

        helper 0 transfer for ep1285_slot17_fg/1 (HELPER -> UE) completes at 0.386500000,
        after contact ends at 0.334854754: no link to the helper exists after disconnection

    With the fix, task 0's egress requirement (its successor may have to restart from the
    requester, so the result must be able to reach the UE) fails first: the placement is
    attempted, found inadmissible, and both tasks restart locally.
    """

    def _fixture(self):
        t0 = _task(0, compute=1.1456e6, out=1e6, root=True, sink=False, external=1000.0)
        t1 = _task(1, compute=4e5, out=1e6, preds=(0,), sink=True)
        dag = _dag("ep1285_slot17_fg", [t0, t1])
        helpers = _helper(CONTACT_END, predicted=1.6)
        compute = V2ComputeSpec(10e6, 1, (1e6,), (4e6,))
        return dag, helpers, compute

    def test_the_reported_shape_no_longer_aborts_the_campaign(self):
        dag, helpers, compute = self._fixture()
        res = schedule_shared([dag], {dag.dag_id: {0: 2, 1: 2}}, link=LINK, compute=compute,
                              helper_states=helpers, validate=True)
        _assert_helper_records_inside_window(self, res, CONTACT_END)
        _assert_ready_times_are_resident(self, dag, res)
        _validate(dag, res, helpers, compute)
        self.assertTrue(all(tm.location == "UE" for tm in res.timings.values()))
        self.assertGreaterEqual(res.queue_stats["helper_egress_infeasible"], 1,
                                "the egress downgrade counter must fire")

    def test_the_relay_that_used_to_overshoot_is_never_booked(self):
        dag, helpers, compute = self._fixture()
        res = schedule_shared([dag], {dag.dag_id: {0: 2, 1: 2}}, link=LINK, compute=compute,
                              helper_states=helpers, validate=True)
        for rec in res.radio_ledger:
            if str(rec["src"]) == HELPER:
                self.assertLessEqual(float(rec["end_s"]), CONTACT_END + CONTACT_SLACK_S,
                                     repr(rec))

    def test_the_same_task_stays_on_the_helper_when_the_window_allows_it(self):
        """A window wide enough for the relay must keep the helper placement (0 downgrades)."""
        dag, _helpers, compute = self._fixture()
        wide = _helper(0.5, predicted=1.6)
        res = schedule_shared([dag], {dag.dag_id: {0: 2, 1: 2}}, link=LINK, compute=compute,
                              helper_states=wide, validate=True)
        self.assertEqual(res.timings[("ep1285_slot17_fg", 0)].location, "HELPER")
        self.assertEqual(res.queue_stats["helper_egress_infeasible"], 0)
        self.assertEqual(res.queue_stats["helper_ingress_infeasible"], 0)
        self.assertEqual(res.queue_stats["helper_contact_failures"], 0)
        _assert_helper_records_inside_window(self, res, 0.5)
        _validate(dag, res, wide, compute)


class TestSerialisedIngressToTheSameHelper(unittest.TestCase):
    """Two root DAGs share the V2V channel and one helper; only one ingress fits the window.

    Audited gap (4): the probe must account for the other ingresses already booked to the same
    helper in the same pass, so the second/third ingress cannot silently exceed `contact_end`.
    Here both DAGs have a 0.3 s ingress into a window of 0.7 s: the first alone fits, the second
    can only start once the first has finished and then pushes its own CPU/return past the
    window, so exactly one of the two may be placed on the helper.

    Both assertions hold on the pre-fix code as well (the real calendars already serialised the
    two routings there, so the fixture does not reproduce a pre-fix abort); it is the regression
    guard for the invariant and for the `helper_ingress_infeasible` counter.
    """

    def _fixture(self):
        f1 = _dag("f1", [_task(0, compute=5e5, out=1e6, root=True, sink=True, external=3e6)])
        f2 = _dag("f2", [_task(0, compute=5e5, out=2e6, root=True, sink=True, external=3e6)])
        helpers = _helper(0.7, predicted=2.0)
        compute = V2ComputeSpec(10e6, 1, (1e6,), (4e6,))
        return [f1, f2], helpers, compute

    def test_no_post_contact_helper_transfer_is_ever_booked(self):
        dags, helpers, compute = self._fixture()
        res = schedule_shared(dags, {d.dag_id: {0: 2} for d in dags}, link=LINK,
                              compute=compute, helper_states=helpers, validate=True)
        _assert_helper_records_inside_window(self, res, 0.7)
        for dag in dags:
            _validate(dag, res, helpers, compute)
        self.assertEqual(res.queue_stats["helper_egress_infeasible"] +
                         res.queue_stats["helper_ingress_infeasible"] +
                         res.queue_stats["helper_contact_failures"], 1,
                         "exactly the second (serialised) ingress must be rejected")

    def test_the_second_ingress_is_decided_against_the_serialised_channel(self):
        """The second DAG's plan must not assume the V2V channel was free for it."""
        dags, helpers, compute = self._fixture()
        res = schedule_shared(dags, {d.dag_id: {0: 2} for d in dags}, link=LINK,
                              compute=compute, helper_states=helpers, validate=True)
        locations = [tm.location for tm in res.timings.values()]
        self.assertIn("UE", locations,
                      "the second ingress cannot fit and must be planned locally")
        self.assertEqual(locations.count("HELPER"), 1,
                         "exactly one of the two ingresses fits the window")
        v2v = sorted((float(r["start_s"]), float(r["end_s"])) for r in res.radio_ledger
                     if r["hop"] == "V2V")
        for (s1, e1), (s2, e2) in zip(v2v, v2v[1:]):
            self.assertGreaterEqual(s2 + 1e-12, e1, "V2V reservations overlap")


class TestStrandedHelperPredecessorIsNeverRelayedLate(unittest.TestCase):
    """Audited gap (5): a fallback relayed a stranded predecessor after disconnection.

    Task 0 runs on the helper, task 1 follows it on the helper and fails its own return. The
    pre-fix planner then booked `HELPER -> UE` for task 0's output at a time the helper was
    already out of contact, and `validate=True` raised exactly the reported error (while
    `validate=False` emitted the invalid schedule). The planner now refuses the helper
    placement BEFORE it books anything: the predecessor's own egress requirement includes the
    requester's UE (no successor can restart from a copy that never arrived), so task 0 is
    attempted remotely, fails contact, and restarts locally.
    """

    def _fixture(self):
        t0 = _task(0, compute=1.6e6, out=1e6, root=True, sink=False, external=1000.0)
        t1 = _task(1, compute=4e5, out=1e6, preds=(0,), sink=True)
        dag = _dag("d", [t0, t1])
        helpers = _helper(0.5, predicted=1.6)
        compute = V2ComputeSpec(10e6, 1, (1e6,), (4e6,))
        return dag, helpers, compute

    def test_the_dag_is_planned_locally_instead_of_relaying_after_disconnection(self):
        dag, helpers, compute = self._fixture()
        res = schedule_shared([dag], {dag.dag_id: {0: 2, 1: 2}}, link=LINK, compute=compute,
                              helper_states=helpers, validate=True)
        _assert_helper_records_inside_window(self, res, 0.5)
        _assert_ready_times_are_resident(self, dag, res)
        _validate(dag, res, helpers, compute)
        self.assertTrue(all(tm.location == "UE" for tm in res.timings.values()),
                        "the plan must degrade to the UE rather than strand helper data")
        self.assertGreaterEqual(res.queue_stats["helper_egress_infeasible"], 1)

    def test_no_ledger_record_leaves_the_helper_after_disconnection(self):
        dag, helpers, compute = self._fixture()
        res = schedule_shared([dag], {dag.dag_id: {0: 2, 1: 2}}, link=LINK, compute=compute,
                              helper_states=helpers, validate=True)
        for rec in res.radio_ledger:
            if str(rec["src"]) == HELPER:
                self.assertLessEqual(float(rec["end_s"]), 0.5 + CONTACT_SLACK_S, repr(rec))

    def test_a_window_wide_enough_for_the_relay_still_uses_the_helper(self):
        """Same DAG, wider contact: the helper is usable and the relay is booked in-window."""
        dag, _helpers, compute = self._fixture()
        wide = _helper(0.7, predicted=1.6)
        res = schedule_shared([dag], {dag.dag_id: {0: 2, 1: 2}}, link=LINK, compute=compute,
                              helper_states=wide, validate=True)
        self.assertTrue(all(tm.location == "HELPER" for tm in res.timings.values()))
        _assert_helper_records_inside_window(self, res, 0.7)
        _validate(dag, res, wide, compute)


class TestValidateFalseIsAdmissibleByConstruction(unittest.TestCase):
    """`validate=False` (the prefix/telescoping path) must not emit an invalid schedule.

    PPO computes rewards on `_schedule(..., validate=False)` output; if that path could still
    commit a post-contact helper departure the reward would be computed on -- and the validator
    would reject -- a different schedule. The planner now makes the same decision with or
    without validation, so the two must agree exactly and the unchecked output must pass
    `validate_schedule` when re-checked.
    """

    def _fixtures(self):
        out = []
        dag, helpers, compute = TestReportedShapeOvershoot()._fixture()
        out.append(("reported_shape", dag, helpers, compute))
        dag2, helpers2, compute2 = TestStrandedHelperPredecessorIsNeverRelayedLate()._fixture()
        out.append(("stranded_predecessor", dag2, helpers2, compute2))
        dags, helpers3, compute3 = TestSerialisedIngressToTheSameHelper()._fixture()
        out.append(("serialised_ingress", dags, helpers3, compute3))
        return out

    def test_validate_true_and_false_agree_and_the_unchecked_plan_revalidates(self):
        for name, dags, helpers, compute in self._fixtures():
            if not isinstance(dags, list):
                dags = [dags]
            plans = {d.dag_id: {t.task_id: 2 for t in d.tasks} for d in dags}
            checked = schedule_shared(dags, plans, link=LINK, compute=compute,
                                      helper_states=helpers, validate=True)
            unchecked = schedule_shared(dags, plans, link=LINK, compute=compute,
                                        helper_states=helpers, validate=False)
            self.assertEqual(unchecked.invariants.get("validated"), False)
            for dag in dags:
                for spec in dag.tasks:
                    key = (dag.dag_id, spec.task_id)
                    self.assertEqual(checked.timings[key].location,
                                     unchecked.timings[key].location,
                                     "%s: validate changed the plan for %r" % (name, key))
                    self.assertAlmostEqual(checked.timings[key].finish_s,
                                           unchecked.timings[key].finish_s, places=12)
            self.assertEqual(checked.radio_ledger, unchecked.radio_ledger,
                             "%s: validate changed the booked transfers" % name)
            self.assertEqual(checked.queue_stats["helper_egress_infeasible"],
                             unchecked.queue_stats["helper_egress_infeasible"])
            self.assertEqual(checked.queue_stats["helper_ingress_infeasible"],
                             unchecked.queue_stats["helper_ingress_infeasible"])
            # re-check the unchecked output: it must pass the same hard invariants
            for dag in dags:
                _validate(dag, unchecked, helpers, compute)

    def test_an_invalid_schedule_is_still_rejected_by_the_validator(self):
        """The validator stays a real check: an injected late helper transfer must raise."""
        dag, helpers, compute = TestReportedShapeOvershoot()._fixture()
        res = schedule_shared([dag], {dag.dag_id: {0: 0}}, link=LINK, compute=compute,
                              helper_states=helpers, validate=True)
        res.radio_ledger.append({
            "dag_id": dag.dag_id, "task_id": 0, "hop": "V2V", "direction": "HELPER->UE",
            "src": "HELPER", "dst": "UE", "bytes": 1e6, "start_s": CONTACT_END - 0.01,
            "end_s": CONTACT_END + 0.05, "service_s": 0.06, "outage_s": 0.0,
            "queue_wait_s": 0.0, "outage_events": 0, "retained_channel": True})
        with self.assertRaises(V2ScheduleError):
            _validate(dag, res, helpers, compute)


class TestWorkAndEnergyConservation(unittest.TestCase):
    """The existing F08 conservation identities must survive the new planner gates."""

    def test_wasted_and_restart_work_are_conserved(self):
        dag, helpers, compute = TestStrandedHelperPredecessorIsNeverRelayedLate()._fixture()
        res = schedule_shared([dag], {dag.dag_id: {0: 2, 1: 2}}, link=LINK, compute=compute,
                              helper_states=helpers, validate=True)
        wasted = 0.0
        restarted = 0.0
        for tm in res.timings.values():
            self.assertAlmostEqual(tm.cpu_executed_bytes,
                                   tm.cpu_restart_bytes if tm.contact_failure
                                   else dag.tasks[tm.task_id].compute_bytes, places=6)
            wasted += tm.cpu_wasted_bytes
            restarted += tm.cpu_restart_bytes
        self.assertAlmostEqual(res.queue_stats["wasted_work_bytes_total"], wasted, places=6)
        self.assertAlmostEqual(res.queue_stats["restart_work_bytes_total"], restarted, places=6)

    def test_a_contact_failure_still_bills_its_attempt(self):
        dag, helpers, compute = TestStrandedHelperPredecessorIsNeverRelayedLate()._fixture()
        res = schedule_shared([dag], {dag.dag_id: {0: 2, 1: 2}}, link=LINK, compute=compute,
                              helper_states=helpers, validate=True)
        failures = [tm for tm in res.timings.values() if tm.contact_failure]
        if not failures:      # the placement was downgraded before any attempt was booked
            self.assertEqual(res.queue_stats["helper_egress_infeasible"] +
                             res.queue_stats["helper_ingress_infeasible"], 2)
            return
        for tm in failures:
            rate = compute.helper_cpu_bytes_per_s[0]
            self.assertAlmostEqual(tm.cpu_attempt_seconds * rate, tm.cpu_wasted_bytes,
                                   places=6)
            self.assertAlmostEqual(tm.cpu_restart_bytes,
                                   float(dag.tasks[tm.task_id].compute_bytes), places=6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
