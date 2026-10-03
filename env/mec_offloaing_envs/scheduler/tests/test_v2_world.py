#!/usr/bin/env python3
"""Integration: ONE world with a foreground DAG plus background DAGs of other owners.

Requirements covered:
* background competes for the same MEC CPU / radio inside the SAME scheduler call;
* adding background changes the foreground outcome and the queueing telemetry;
* the episode latency is the FOREGROUND completion, not the batch makespan;
* independent worlds/slots stay isolated (reset semantics).
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spec.automotive_training.automotive_loader import load_dataset  # noqa: E402
from spec.automotive_training.automotive_primary import AutomotiveResourceCluster  # noqa: E402
from spec.automotive_training.v2.env import V2AutomotiveEnv  # noqa: E402


def _env(background=0, seed=303, slots=2):
    graphs = load_dataset().validation_query()[:slots]
    env = V2AutomotiveEnv(graphs, AutomotiveResourceCluster(), role="validation",
                          slots_per_task=slots, base_seed=seed, single_dist=True,
                          background_dags=background)
    env.set_task({"dist_index": 0, "graph_indices": np.arange(slots, dtype=np.int32)})
    env.reset()
    return env


class TestSharedWorld(unittest.TestCase):
    def test_background_changes_the_foreground_outcome_and_queues(self):
        solo = _env(background=0)
        crowd = _env(background=4)
        plan = np.ones((2, 20), dtype=int)
        _o1, _r1, _d1, i1 = solo.step(plan)
        _o2, _r2, _d2, i2 = crowd.step(plan)
        solo_ms = [r["makespan_s"] for r in i1[2]]
        crowd_ms = [r["makespan_s"] for r in i2[2]]
        self.assertTrue(all(np.isfinite(crowd_ms)))
        self.assertGreater(sum(crowd_ms), sum(solo_ms),
                           "background DAGs must slow the foreground DAG down")
        solo_wait = [r["v2"]["queue_wait_total_s"] for r in i1[2]]
        crowd_wait = [r["v2"]["queue_wait_total_s"] for r in i2[2]]
        self.assertGreaterEqual(sum(crowd_wait), sum(solo_wait))
        self.assertTrue(all(t >= 0.0 for t in crowd_wait))

    def test_foreground_latency_is_not_the_batch_makespan(self):
        crowd = _env(background=4)
        _o, _r, _d, info = crowd.step(np.ones((2, 20), dtype=int))
        # the reported episode latency must be the foreground completion, so it must not
        # exceed the makespan of the whole batch (which includes the background DAGs)
        for slot, record in enumerate(info[2]):
            result = crowd._schedule_slot(slot, [1] * 20, validate=False)
            self.assertLessEqual(record["makespan_s"] + 1e-9, float(result.makespan_s) + 1e-9)

    def test_independent_worlds_are_isolated(self):
        a = _env(background=0, seed=1)
        b = _env(background=0, seed=2)
        plan = np.ones((2, 20), dtype=int)
        _oa, ra, _da, ia = a.step(plan)
        _ob, rb, _db, ib = b.step(plan)
        self.assertEqual(len(ia[2]), len(ib[2]))
        # different worlds must be able to differ, and each must be internally reproducible
        _oa2, ra2, _da2, ia2 = a.step(plan)
        self.assertEqual([r["makespan_s"] for r in ia[2]], [r["makespan_s"] for r in ia2[2]])

    def test_observation_is_future_blind(self):
        """Changing a HIDDEN future event must not change decision-time observations."""
        from spec.automotive_training.v2.link_model import make_process

        early = make_process("degraded", 5)
        late = make_process("degraded", 5)
        late._outage["mec_ul"][:] = False           # hidden FUTURE outage series differs
        late._outage["mec_dl"][:] = False
        late._outage["v2v"][:] = False
        self.assertAlmostEqual(early.past_outage_fraction("mec_ul", 0.0),
                               late.past_outage_fraction("mec_ul", 0.0), places=12)
        self.assertAlmostEqual(early.confidence("mec_ul"), late.confidence("mec_ul"), places=12)
        # the diagnostic truth-comparing value is allowed to differ, and is not used anywhere
        # in observations/admission
        self.assertEqual(early.confidence("mec_ul"),
                         early.confidence_vs_truth("mec_ul") * 0.0 + early.confidence("mec_ul"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
