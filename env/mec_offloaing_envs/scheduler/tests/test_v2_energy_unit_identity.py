#!/usr/bin/env python3
"""F11: one frequency for scheduling AND energy pricing, so E = P*T == kappa*C*f^2.

The v1 contract scheduled with the PER-GRAPH dataset frequency (`f_dataset/(8*xi)`) while the
event ledger priced CPU energy at the FROZEN tier frequency. The two identities therefore
referred to different `f`, and the historical diagnostic was
`E_duration/E_work = R_physical/R_scheduled != 1`.

V2 resolves it with one contract (`frozen_tier_f_v2`): the service rate is the physical rate at
the energy-pricing frequency, `R = f_frozen/(8*xi)`. These tests pin the identity analytically,
including interrupted (wasted / restarted) work.
"""
import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spec.automotive_training.automotive_resources import dataset_reference_parameters  # noqa
from spec.automotive_training.v2.adapters import config_for, V2_RESOURCE_CONTRACT  # noqa
from spec.automotive_training.v2.energy import frozen_energy_spec  # noqa


class EnergyUnitIdentityTests(unittest.TestCase):
    def setUp(self):
        from spec.automotive_training.automotive_loader import load_dataset
        self.graph = list(load_dataset().meta_train())[0]
        self.xi = float(dataset_reference_parameters()["xi_cycles_per_bit"])
        self.spec = frozen_energy_spec()
        self.cfg = config_for(self.graph)

    def _rates(self):
        return {"ue": float(self.cfg.ue_cpu_bytes_per_second),
                "helper": float(self.cfg.helper_cpu_bytes_per_second),
                "mec": float(self.cfg.mec_cpu_bytes_per_second)}

    def test_contract_id_is_versioned(self):
        self.assertEqual(V2_RESOURCE_CONTRACT, "frozen_tier_f_v2")

    def test_rate_is_the_physical_rate_at_the_energy_frequency(self):
        for tier, rate in self._rates().items():
            want = float(self.spec.tiers[tier].f_hz) / (8.0 * self.xi)
            self.assertAlmostEqual(rate, want, places=6,
                                   msg="%s rate is not f_frozen/(8*xi)" % tier)

    def test_work_and_duration_energy_are_identical(self):
        """E = kappa*C*f^2 must equal E = kappa*f^3*T for the SAME work and frequency."""
        for tier, rate in self._rates().items():
            ts = self.spec.tiers[tier]
            for payload_bytes in (1.0, 1000.0, 1e6):
                T = payload_bytes / rate
                C = payload_bytes * 8.0 * self.xi
                e_work = ts.kappa * C * ts.f_hz ** 2
                e_duration = ts.kappa * ts.f_hz ** 3 * T
                self.assertAlmostEqual(
                    e_work / e_duration, 1.0, places=12,
                    msg="%s: E_duration/E_work != 1 at %g bytes (contract broken)"
                        % (tier, payload_bytes))

    def test_interrupted_work_is_priced_once_and_consistently(self):
        """A partial attempt plus a restart must satisfy the identity for each segment."""
        ts = self.spec.tiers["ue"]
        rate = self._rates()["ue"]
        attempted, wasted, restart = 4.0e5, 1.0e5, 4.0e5
        total = 0.0
        for payload in (attempted - wasted, wasted, restart):
            T = payload / rate
            e_dur = ts.kappa * ts.f_hz ** 3 * T
            e_work = ts.kappa * (payload * 8.0 * self.xi) * ts.f_hz ** 2
            self.assertAlmostEqual(e_dur / e_work, 1.0, places=12)
            total += e_dur
        # the sum is the sum of the priced segments, i.e. nothing is free and nothing doubles
        want = ts.kappa * ((attempted + restart) * 8.0 * self.xi) * ts.f_hz ** 2
        self.assertAlmostEqual(total / want, 1.0, places=12)

    def test_historical_ratio_is_no_longer_observed(self):
        """The old diagnostic ratio must be exactly 1 under the resolved contract."""
        hist = {"ue": 0.39744, "helper": 0.59605, "mec": 0.39744}
        for tier, rate in self._rates().items():
            ts = self.spec.tiers[tier]
            r_physical = ts.f_hz / (8.0 * self.xi)
            ratio = r_physical / rate
            self.assertAlmostEqual(ratio, 1.0, places=12)
            self.assertNotAlmostEqual(ratio, hist[tier], places=3,
                                      msg="the historical mismatch is still present for %s" % tier)


if __name__ == "__main__":
    unittest.main(verbosity=2)
