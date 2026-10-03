#!/usr/bin/env python3
"""Event-based transfer semantics: rate at the actual start, accumulated service, outages."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spec.automotive_training.v2.link_model import LINK_DL, LINK_UL, make_process  # noqa: E402
from spec.automotive_training.v2.shared_scheduler import (  # noqa: E402
    V2ComputeSpec, V2DAGSpec, V2LinkSpec, V2TaskSpec, schedule_shared,
)

LINK = V2LinkSpec(20e6, 20e6, 10e6)
COMPUTE = V2ComputeSpec(10e6, 1, (1e6,), (2e6,))


def _dag(output_bytes=4e6):
    return V2DAGSpec("d", 0, [V2TaskSpec(0, 1e6, output_bytes, (), True, True, "HIGH", 1.0, 0)])


class _MidOutage:
    """Constant rate except a 0.3 s outage starting 0.05 s after the transfer begins."""

    class _Regime:
        dt_s = 0.05
        name = "mid_outage"
        evidence_class = "test_stub"
        estimation_sigma = 0.0

    regime = _Regime()

    def realized(self, link, t):
        return 0.0 if 0.05 <= t < 0.35 else 1.0


class TestTransferEvents(unittest.TestCase):
    def test_constant_rate_is_exactly_bytes_over_rate(self):
        res = schedule_shared([_dag()], {"d": {0: 1}}, link=LINK, compute=COMPUTE)
        down = [e for e in res.radio_events if e[2].startswith("MEC_DL")]
        self.assertTrue(down)
        self.assertAlmostEqual(down[-1][1] - down[-1][0], 4e6 / 20e6, places=9)

    def test_outage_mid_transfer_pauses_and_is_logged(self):
        res = schedule_shared([_dag()], {"d": {0: 1}}, link=LINK, compute=COMPUTE,
                              link_process=_MidOutage())
        self.assertGreater(res.makespan_s, 0.05 + 0.2,
                           "the outage must extend the transfer, not shrink it")
        self.assertGreater(res.queue_stats["outage_wait_total_s"], 0.0)
        for tm in res.timings.values():
            self.assertTrue(np.isfinite(tm.finish_s))

    def test_variable_rate_is_integrated_not_sampled_once(self):
        res = schedule_shared([_dag()], {"d": {0: 1}}, link=LINK, compute=COMPUTE,
                              link_process=make_process("degraded", 3))
        events = [e for e in res.radio_events if e[2].startswith("MEC_DL")]
        first = events[-1]
        duration = first[1] - first[0]
        process = make_process("degraded", 3)
        rate_at_start = 20e6 * process.realized(LINK_DL, first[0])
        self.assertGreater(abs(duration - 4e6 / rate_at_start), 1e-9,
                           "a single rate sample at the start would give a different duration")

    def test_no_event_after_a_disconnected_transfer(self):
        process = make_process("degraded", 11)
        res = schedule_shared([_dag(2e6)], {"d": {0: 1}}, link=LINK, compute=COMPUTE,
                              link_process=process)
        for tm in res.timings.values():
            self.assertTrue(np.isfinite(tm.finish_s))
        self.assertTrue(res.invariants["precedence_and_data_arrival"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
