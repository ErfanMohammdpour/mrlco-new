#!/usr/bin/env python3
"""F06: transfer service must integrate to the actual rate-trace boundaries.

The rate trace is piecewise constant on [k*dt, (k+1)*dt). The audited `_service_detail`
advanced a full `dt` from an arbitrary start, applying one interval's rate across the next
interval's boundary. Fixture: start 0.05 s, dt = 0.1 s, payload 20 B, base 100 B/s, x10 from
0.1 s. Analytic: 5 B @ 100 B/s over [0.05,0.1) + 15 B @ 1000 B/s = 0.015 s -> completion
0.115 s. The audited scheduler returned 0.160 s.
"""
import sys
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spec.automotive_training.v2.shared_scheduler import (  # noqa: E402
    V2ComputeSpec, V2DAGSpec, V2LinkSpec, V2TaskSpec, schedule_shared,
)
from spec.automotive_training.v2.link_model import LinkProcess, LinkRegime  # noqa: E402


def _process(multipliers, dt=0.1):
    reg = LinkRegime(name="fixture", sigma=0.0, persistence=0.0, outage_prob_per_step=0.0,
                     outage_steps=0, estimation_sigma=0.0, estimation_bias=1.0, dt_s=dt)
    lp = LinkProcess.__new__(LinkProcess)
    lp.regime = reg
    lp.links = ("mec_ul", "mec_dl", "v2v")
    arr = np.asarray(multipliers, dtype=float)
    lp._realized = {k: arr.copy() for k in lp.links}
    lp._estimated = {k: arr.copy() for k in lp.links}
    lp._outage = {k: np.zeros(arr.size, dtype=bool) for k in lp.links}
    return lp


class ServiceBoundaryTests(unittest.TestCase):
    def _completion(self, start, payload, ratios, base=100.0, dt=0.1):
        link = V2LinkSpec(base, base, base)
        comp = V2ComputeSpec(1e9, 1, (1e9,), (1e9,))
        dag = V2DAGSpec("d", 0, [V2TaskSpec(0, 1.0, 1.0, (), True, True, "HIGH", None, 0,
                                            external_input_bytes=payload)],
                        arrival_s=start)
        res = schedule_shared([dag], {"d": [1]}, link=link, compute=comp,
                             link_process=_process(ratios, dt), validate=True)
        ev = [e for e in res.radio_events if abs(e[3] - payload) < 1e-9]
        self.assertTrue(ev, "no booking for the external-input upload")
        return ev[0][0], ev[0][1]

    def test_off_grid_start_splits_at_the_boundary(self):
        s, e = self._completion(0.05, 20.0, [1.0, 10.0, 10.0, 10.0, 10.0])
        self.assertAlmostEqual(s, 0.05, places=9)
        self.assertAlmostEqual(e, 0.115, places=9,
                               msg="service straddled the rate boundary (got %r)" % e)

    def test_on_grid_start_uses_the_new_rate(self):
        s, e = self._completion(0.10, 20.0, [1.0, 10.0, 10.0, 10.0, 10.0])
        self.assertAlmostEqual(s, 0.10, places=9)
        self.assertAlmostEqual(e, 0.12, places=9)   # 20 B @ 1000 B/s

    def test_constant_rate_is_the_simple_quotient(self):
        s, e = self._completion(0.05, 20.0, [1.0] * 8)
        self.assertAlmostEqual(e - s, 0.2, places=9)

    def test_final_partial_interval_is_not_extended_forever(self):
        # horizon = 5 steps = 0.5 s. Start 0.2 s, 20 B @ 100 B/s -> needs 0.2 s -> ends 0.4 s,
        # which still fits, so the transfer completes normally inside the last partial interval.
        s, e = self._completion(0.20, 20.0, [1.0] * 5)
        self.assertAlmostEqual(e - s, 0.2, places=9)
        self.assertAlmostEqual(e, 0.40, places=9)

    def test_horizon_exhaustion_is_an_error_not_a_silent_clamp(self):
        from spec.automotive_training.v2.shared_scheduler import V2ScheduleError
        with self.assertRaises(V2ScheduleError):
            # slow rate (1 B/s effective) cannot deliver 20 B inside the 0.5 s horizon
            self._completion(0.0, 20.0, [0.01, 0.01, 0.01, 0.01, 0.01])

    def test_zero_payload_books_nothing(self):
        """A zero-byte transfer must not hold the channel or add an interval."""
        link = V2LinkSpec(100.0, 100.0, 100.0)
        comp = V2ComputeSpec(1e9, 1, (1e9,), (1e9,))
        dag = V2DAGSpec("d", 0, [V2TaskSpec(0, 1.0, 1.0, (), True, True, "HIGH", None, 0,
                                            external_input_bytes=0.0)], arrival_s=0.05)
        res = schedule_shared([dag], {"d": [1]}, link=link, compute=comp,
                              link_process=_process([1.0, 10.0, 10.0]), validate=True)
        # no zero-byte interval may hold the channel; the only booking is the 1-byte sink return
        self.assertEqual([e for e in res.radio_events if e[3] <= 0.0], [],
                         "a zero-byte transfer must not be booked on the channel")
        self.assertEqual([e[3] for e in res.radio_events], [1.0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
