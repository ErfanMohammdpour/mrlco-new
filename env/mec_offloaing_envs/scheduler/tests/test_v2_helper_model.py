#!/usr/bin/env python3
"""v2 stage-4 tests: helper as a real vehicle resource (CPU, occupancy, contact)."""

from __future__ import annotations

import math
import statistics
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spec.automotive_training.v2.helper_model import (  # noqa: E402
    HelperModelError, HelperState, admissible, make_helpers, predicted_contact,
    required_helper_time_s, sample_contact,
)
from spec.automotive_training.v2.shared_scheduler import (  # noqa: E402
    V2ComputeSpec, V2DAGSpec, V2LinkSpec, V2TaskSpec, schedule_shared,
)


def _dag(helper_id=0, compute=5e6, external=1000):
    tasks = [V2TaskSpec(0, compute, 2e6, (), True, True, "HIGH", 1.0, 0,
                        external_input_bytes=external)]
    return V2DAGSpec("d", 0, tasks, helper_id=helper_id)


LINK = V2LinkSpec(20e6, 20e6, 10e6)
COMPUTE = V2ComputeSpec(10e6, 1, (1e6,), (4e6,))


class TestHelperPrimitives(unittest.TestCase):
    def test_contact_sampling_is_seeded_and_positive(self):
        a = [sample_contact(__import__("numpy").random.RandomState(3 + i), mean_s=2.0)
             for i in range(5)]
        b = [sample_contact(__import__("numpy").random.RandomState(3 + i), mean_s=2.0)
             for i in range(5)]
        self.assertEqual(a, b)
        self.assertTrue(all(v >= 0.2 for v in a))
        self.assertGreater(max(a), min(a))

    def test_prediction_is_optimistic_but_bounded(self):
        self.assertGreater(predicted_contact(1.0, bias=1.15), 1.0)
        self.assertEqual(predicted_contact(math.inf), math.inf)

    def test_required_time_and_admissibility(self):
        need = required_helper_time_s(payload_in_bytes=1000, compute_bytes=5e6,
                                      v2v_bytes_per_s=10e6, helper_bytes_per_s=4e6)
        self.assertAlmostEqual(need, 2 * 1000 / 10e6 + 5e6 / 4e6, places=12)
        long_contact = HelperState(0, 4e6, contact_end_s=10.0, predicted_contact_end_s=10.0)
        short_contact = HelperState(0, 4e6, contact_end_s=0.05, predicted_contact_end_s=0.05)
        self.assertTrue(admissible(long_contact, now_s=0.0, payload_in_bytes=1000,
                                   compute_bytes=5e6, v2v_bytes_per_s=10e6))
        self.assertFalse(admissible(short_contact, now_s=0.0, payload_in_bytes=1000,
                                    compute_bytes=5e6, v2v_bytes_per_s=10e6))

    def test_make_helpers_is_seeded_and_labelled(self):
        specs = {0: {"cpu_bytes_per_s": 4e6, "busy_until_s": 0.0}}
        a = make_helpers(specs, seed=11)
        b = make_helpers(specs, seed=11)
        self.assertEqual(a[0].contact_end_s, b[0].contact_end_s)
        self.assertGreater(a[0].predicted_contact_end_s, a[0].contact_end_s)

    def test_invalid_helper_rate_rejected(self):
        with self.assertRaises(HelperModelError):
            HelperState(0, 0.0)


class TestHelperInScheduler(unittest.TestCase):
    def test_idle_helper_with_long_contact_beats_mec_when_mec_is_overloaded(self):
        # 12 concurrent single-task DAGs, one shared MEC server vs 12 idle helpers
        dags, plans = [], {}
        for i in range(12):
            dags.append(_dag(helper_id=i, compute=6e6))
            plans["d"] = {0: 2}
        plans = {"d" if False else d.dag_id: {0: 2} for d in dags}
        # give every DAG a distinct id
        for i, d in enumerate(dags):
            d.dag_id = "d%d" % i
        plans = {d.dag_id: {0: 2} for d in dags}
        helpers = {i: HelperState(i, 4e6, contact_end_s=100.0, predicted_contact_end_s=100.0)
                   for i in range(12)}
        compute = V2ComputeSpec(10e6, 1, tuple([1e6] * 12), tuple([4e6] * 12))
        helper_res = schedule_shared(dags, plans, link=LINK, compute=compute,
                                     helper_states=helpers)
        mec_res = schedule_shared(dags, {d.dag_id: {0: 1} for d in dags}, link=LINK,
                                  compute=compute, helper_states=helpers)
        self.assertLess(helper_res.makespan_s, mec_res.makespan_s)

    def test_busy_helper_pays_a_queue_penalty(self):
        idle = HelperState(0, 4e6, busy_until_s=0.0, contact_end_s=10.0,
                           predicted_contact_end_s=10.0)
        busy = HelperState(0, 4e6, busy_until_s=2.0, contact_end_s=10.0,
                           predicted_contact_end_s=10.0)
        a = schedule_shared([_dag()], {"d": {0: 2}}, link=LINK, compute=COMPUTE,
                            helper_states={0: idle})
        b = schedule_shared([_dag()], {"d": {0: 2}}, link=LINK, compute=COMPUTE,
                            helper_states={0: busy})
        self.assertGreaterEqual(b.makespan_s, a.makespan_s + 1.5)

    def test_short_predicted_contact_rejects_the_placement(self):
        short = HelperState(0, 4e6, contact_end_s=0.02, predicted_contact_end_s=0.02)
        res = schedule_shared([_dag()], {"d": {0: 2}}, link=LINK, compute=COMPUTE,
                              helper_states={0: short})
        self.assertEqual(res.queue_stats["helper_rejections_inadmissible"], 1)
        self.assertEqual(res.timings[("d", 0)].location, "UE")

    def test_realized_contact_shorter_than_predicted_fails_and_restarts(self):
        # planner is told 5 s, reality is 0.2 s -> task fails, remainder restarts locally
        optimistic = HelperState(0, 4e6, contact_end_s=0.2, predicted_contact_end_s=5.0)
        res = schedule_shared([_dag(compute=10e6)], {"d": {0: 2}}, link=LINK,
                              compute=COMPUTE, helper_states={0: optimistic})
        self.assertEqual(res.queue_stats["helper_contact_failures"], 1)
        self.assertGreater(res.queue_stats["helper_restart_penalty_s"], 0.0)
        self.assertEqual(res.timings[("d", 0)].location, "UE")

    def test_helper_absence_still_degrades_to_local(self):
        res = schedule_shared([_dag(helper_id=None)], {"d": {0: 2}}, link=LINK,
                              compute=COMPUTE, helper_states={})
        self.assertEqual(res.timings[("d", 0)].location, "UE")


if __name__ == "__main__":
    unittest.main(verbosity=2)
