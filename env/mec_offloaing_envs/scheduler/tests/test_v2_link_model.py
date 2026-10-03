#!/usr/bin/env python3
"""v2 stage-3 tests: estimated vs realized links, regimes, reproducibility, outage waiting."""

from __future__ import annotations

import statistics
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spec.automotive_training.v2.adapters import (  # noqa: E402
    compute_spec, dag_spec_from_graph, link_spec, plan_map_from_actions,
)
from spec.automotive_training.v2.link_model import (  # noqa: E402
    LINK_DL, LINK_UL, LINK_V2V, LinkModelError, load_regimes, make_process,
)
from spec.automotive_training.v2.shared_scheduler import (  # noqa: E402
    V2ComputeSpec, V2DAGSpec, V2LinkSpec, V2TaskSpec, schedule_shared,
)
from spec.automotive_training.automotive_loader import load_dataset  # noqa: E402


class TestRegimeProvenance(unittest.TestCase):
    def test_every_regime_is_labelled_synthetic(self):
        for name, spec in load_regimes().items():
            self.assertEqual(spec["evidence_class"], "explicit_assumption_synthetic", name)
            self.assertTrue(str(spec.get("description")))

    def test_regime_separation(self):
        stable = make_process("stable", 0).summary({"mec_ul": 20e6, "mec_dl": 20e6, "v2v": 10e6})
        moderate = make_process("moderate", 0).summary({"mec_ul": 20e6, "mec_dl": 20e6, "v2v": 10e6})
        degraded = make_process("degraded", 0).summary({"mec_ul": 20e6, "mec_dl": 20e6, "v2v": 10e6})
        self.assertEqual(stable["links"][LINK_UL]["realized_std_multiplier"], 0.0)
        # confidence is now estimate-model based (no truth access): it must fall as the
        # declared estimation noise grows, and it must NOT be a truth comparison
        key = "mean_confidence_estimate_model"
        self.assertEqual(stable["links"][LINK_UL][key], 1.0)
        self.assertLess(moderate["links"][LINK_UL][key], 1.0)
        self.assertLess(degraded["links"][LINK_V2V][key],
                        moderate["links"][LINK_V2V][key])
        self.assertLess(degraded["links"][LINK_UL]["realized_mean_multiplier"],
                        moderate["links"][LINK_UL]["realized_mean_multiplier"])
        self.assertGreater(degraded["links"][LINK_DL]["outage_fraction"],
                           moderate["links"][LINK_DL]["outage_fraction"])


class TestReproducibility(unittest.TestCase):
    def test_same_seed_bit_identical(self):
        a, b = make_process("degraded", 7), make_process("degraded", 7)
        for link, t in ((LINK_UL, 0.0), (LINK_DL, 1.23), (LINK_V2V, 4.5)):
            self.assertEqual(a.realized(link, t), b.realized(link, t))
            self.assertEqual(a.estimate_at(link, t), b.estimate_at(link, t))
            self.assertEqual(a.outage_at(link, t), b.outage_at(link, t))

    def test_different_seed_differs(self):
        a, b = make_process("degraded", 7), make_process("degraded", 8)
        series_a = [a.realized(LINK_UL, i * 0.05) for i in range(200)]
        series_b = [b.realized(LINK_UL, i * 0.05) for i in range(200)]
        self.assertNotEqual(series_a, series_b)

    def test_estimate_is_not_the_realized_series(self):
        p = make_process("moderate", 3)
        diffs = [abs(p.estimate_at(LINK_UL, i * 0.05) - p.realized(LINK_UL, i * 0.05))
                 for i in range(200)]
        self.assertGreater(max(diffs), 1e-6)


class _ForcedOutage:
    """Minimal process stub: the UL link is out for the first 0.2 s."""

    class _Regime:
        dt_s = 0.05
        name = "forced"
        evidence_class = "test_stub"

    regime = _Regime()

    def realized(self, link, t):
        return 0.0 if (link == LINK_UL and t < 0.2 - 1e-12) else 1.0

    def realized_rate(self, base, link, t):
        rate = base * self.realized(link, t)
        if rate <= 0.0:
            raise LinkModelError("zero rate")
        return rate


class TestSchedulerWithLinks(unittest.TestCase):
    LINK = V2LinkSpec(20e6, 20e6, 10e6)
    COMPUTE = V2ComputeSpec(10e6, 1, (1e6,), (2e6,))
    TASKS = [V2TaskSpec(0, 5e6, 5e6, (), True, False, "HIGH", 1.0, 0, external_input_bytes=1000),
             V2TaskSpec(1, 5e6, 5e6, (0,), False, True, "HIGH", 1.0, 0)]

    def _dag(self):
        return V2DAGSpec("d", 0, self.TASKS)

    def test_stable_regime_matches_no_process(self):
        plan = {"d": {0: 1, 1: 1}}
        plain = schedule_shared([self._dag()], plan, link=self.LINK, compute=self.COMPUTE)
        stable = schedule_shared([self._dag()], plan, link=self.LINK, compute=self.COMPUTE,
                                 link_process=make_process("stable", 0))
        self.assertEqual(plain.makespan_s, stable.makespan_s)
        self.assertEqual(stable.queue_stats["outage_events_total"], 0)

    def test_degraded_regime_is_slower_and_logs_outages(self):
        plan = {"d": {0: 1, 1: 1}}
        stable = schedule_shared([self._dag()], plan, link=self.LINK, compute=self.COMPUTE,
                                 link_process=make_process("stable", 0))
        results = []
        for seed in range(8):
            results.append(schedule_shared([self._dag()], plan, link=self.LINK,
                                           compute=self.COMPUTE,
                                           link_process=make_process("degraded", seed)))
        # a single realization can be luckier than the stable link; the CLAIM is about the
        # regime mean, not about every seed
        self.assertGreaterEqual(statistics.fmean([r.makespan_s for r in results]),
                                stable.makespan_s * 0.999)
        self.assertTrue(any(r.queue_stats["outage_events_total"] > 0 for r in results),
                        "the degraded regime must eventually stall a transfer")
        for r in results:
            for tm in r.timings.values():
                self.assertGreaterEqual(tm.outage_wait_s, 0.0)

    def test_open_loop_waits_for_outage_recovery(self):
        plan = {"d": {0: 1, 1: 1}}
        res = schedule_shared([self._dag()], plan, link=self.LINK, compute=self.COMPUTE,
                              link_process=_ForcedOutage())
        self.assertGreater(res.makespan_s, 0.0)
        self.assertGreaterEqual(res.queue_stats["outage_events_total"], 1,
                                "the open-loop plan must stall during an outage")
        self.assertGreater(res.queue_stats["outage_wait_total_s"], 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
