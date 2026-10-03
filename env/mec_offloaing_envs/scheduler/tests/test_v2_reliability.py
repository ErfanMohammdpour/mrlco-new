#!/usr/bin/env python3
"""v2 stage-5 tests: criticality-aware remote reliability + warm-standby fallback hook."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spec.automotive_training.v2.reliability import (  # noqa: E402
    ReliabilityError, load_classes, make_gate, remote_admissible,
    remote_success_probability, standby_required,
)
from spec.automotive_training.v2.shared_scheduler import (  # noqa: E402
    V2ComputeSpec, V2DAGSpec, V2LinkSpec, V2TaskSpec, schedule_shared,
)

LINK = V2LinkSpec(20e6, 20e6, 10e6)
COMPUTE = V2ComputeSpec(10e6, 1, (1e6,), (4e6,))


def _dag(criticality="HIGH", deadline=1.0, compute=5e6):
    return V2DAGSpec("d", 0, [V2TaskSpec(0, compute, 2e6, (), True, True, criticality,
                                         deadline, 0, external_input_bytes=1000)])


class TestClassesAndProvenance(unittest.TestCase):
    def test_ordering_and_grounding(self):
        classes = load_classes()
        self.assertLess(classes["HIGH"].epsilon, classes["MEDIUM"].epsilon)
        self.assertLess(classes["MEDIUM"].epsilon, classes["LOW"].epsilon)
        for name in ("HIGH", "MEDIUM"):
            self.assertEqual(classes[name].evidence_class, "dataset_standard_derived")
            self.assertIsNotNone(classes[name].requirement_id)
            self.assertEqual(len(classes[name].evidence_sha256), 64)
        self.assertEqual(classes["LOW"].evidence_class, "explicit_assumption_synthetic")
        self.assertTrue(classes["HIGH"].warm_standby)
        self.assertFalse(classes["LOW"].warm_standby)

    def test_probability_model_bounds(self):
        self.assertAlmostEqual(remote_success_probability(link_confidence=1.0,
                                                          outage_fraction=0.0), 1.0)
        self.assertAlmostEqual(remote_success_probability(link_confidence=0.5,
                                                          outage_fraction=0.5), 0.25)
        with self.assertRaises(ReliabilityError):
            remote_success_probability(link_confidence=1.5, outage_fraction=0.0)

    def test_class_admissibility_ordering(self):
        p = 0.995
        self.assertTrue(remote_admissible("LOW", p))
        self.assertFalse(remote_admissible("MEDIUM", p))
        self.assertFalse(remote_admissible("HIGH", p))
        self.assertTrue(remote_admissible("HIGH", 0.999999))

    def test_criticality_is_independent_of_deadline(self):
        gate = make_gate()
        evidence = {"link_confidence": 0.99995, "outage_fraction": 0.0}
        # deadline never enters the reliability gate
        self.assertEqual(gate("MEC", "HIGH", evidence), gate("MEC", "HIGH", evidence))
        self.assertNotEqual(gate("MEC", "HIGH", evidence), gate("MEC", "LOW", evidence))


class TestGateInScheduler(unittest.TestCase):
    def test_reliable_idle_mec_admits_high_criticality_remotely(self):
        res = schedule_shared([_dag("HIGH")], {"d": {0: 1}}, link=LINK, compute=COMPUTE,
                              reliability_gate=make_gate(),
                              reliability_evidence={"link_confidence": 1.0,
                                                    "outage_fraction": 0.0},
                              standby_for=standby_required)
        self.assertEqual(res.timings[("d", 0)].location, "MEC",
                         "HIGH must NOT be hard-coded local")
        self.assertEqual(res.queue_stats["reliability_rejections"], 0)
        self.assertGreater(res.queue_stats["fallback_reserved_s"], 0.0,
                           "HIGH remote tokens require a warm standby reservation")

    def test_unreliable_network_rejects_high_and_is_counted(self):
        res = schedule_shared([_dag("HIGH")], {"d": {0: 1}}, link=LINK, compute=COMPUTE,
                              reliability_gate=make_gate(),
                              reliability_evidence={"link_confidence": 0.80,
                                                    "outage_fraction": 0.10},
                              standby_for=standby_required)
        tm = res.timings[("d", 0)]
        self.assertEqual(tm.location, "UE")
        self.assertTrue(tm.reliability_rejected)
        self.assertEqual(res.queue_stats["reliability_rejections"], 1)
        self.assertAlmostEqual(res.queue_stats["fallback_reserved_s"], 0.0)

    def test_same_network_low_criticality_may_go_remote(self):
        evidence = {"link_confidence": 0.995, "outage_fraction": 0.0}
        low = schedule_shared([_dag("LOW")], {"d": {0: 1}}, link=LINK, compute=COMPUTE,
                              reliability_gate=make_gate(), reliability_evidence=evidence)
        high = schedule_shared([_dag("HIGH")], {"d": {0: 1}}, link=LINK, compute=COMPUTE,
                               reliability_gate=make_gate(), reliability_evidence=evidence)
        self.assertEqual(low.timings[("d", 0)].location, "MEC")
        self.assertEqual(high.timings[("d", 0)].location, "UE")
        self.assertEqual(low.queue_stats["reliability_rejections"], 0)
        self.assertEqual(high.queue_stats["reliability_rejections"], 1)

    def test_gate_is_off_by_default(self):
        res = schedule_shared([_dag("HIGH")], {"d": {0: 1}}, link=LINK, compute=COMPUTE)
        self.assertEqual(res.timings[("d", 0)].location, "MEC")
        self.assertEqual(res.queue_stats["reliability_rejections"], 0)

    def test_deadline_does_not_change_the_reliability_decision(self):
        evidence = {"link_confidence": 0.995, "outage_fraction": 0.0}
        tight = schedule_shared([_dag("HIGH", deadline=0.01)], {"d": {0: 1}}, link=LINK,
                                compute=COMPUTE, reliability_gate=make_gate(),
                                reliability_evidence=evidence)
        loose = schedule_shared([_dag("HIGH", deadline=10.0)], {"d": {0: 1}}, link=LINK,
                                compute=COMPUTE, reliability_gate=make_gate(),
                                reliability_evidence=evidence)
        self.assertEqual(tight.timings[("d", 0)].location,
                         loose.timings[("d", 0)].location)


if __name__ == "__main__":
    unittest.main(verbosity=2)
