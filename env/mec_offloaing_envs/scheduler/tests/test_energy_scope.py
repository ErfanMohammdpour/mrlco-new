#!/usr/bin/env python3
"""Canonical energy accessor: three boundaries, explicit scope, no default."""

from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
for _name in ("gym", "gym.core"):
    sys.modules.setdefault(_name, types.ModuleType(_name))
sys.modules["gym.core"].Env = type("Env", (), {})

from env.mec_offloaing_envs.scheduler.energy_scope import (  # noqa: E402
    SCOPE_MOBILE,
    SCOPE_REQUESTER,
    SCOPE_SYSTEM,
    configured_energy_scalar,
    energy_scalar,
)
from env.mec_offloaing_envs.scheduler.model import EnergyBreakdown  # noqa: E402


def breakdown(**overrides):
    fields = dict(
        ue_local_cpu_joules=1.0,
        ue_mec_uplink_joules=2.0,
        ue_mec_downlink_joules=3.0,
        ue_v2v_tx_joules=4.0,
        ue_v2v_rx_joules=5.0,
        helper_compute_joules=6.0,
        helper_v2v_tx_joules=7.0,
        helper_v2v_rx_joules=8.0,
        mec_compute_joules_optional=9.0,
        mec_tx_joules_optional=10.0,
    )
    fields.update(overrides)
    return EnergyBreakdown(**fields)


class TestScopes(unittest.TestCase):
    def test_three_boundaries(self):
        b = breakdown()
        requester = energy_scalar(b, scope=SCOPE_REQUESTER)
        mobile = energy_scalar(b, scope=SCOPE_MOBILE)
        system = energy_scalar(b, scope=SCOPE_SYSTEM)
        self.assertAlmostEqual(requester, 1 + 2 + 3 + 4 + 5)
        self.assertAlmostEqual(mobile, requester + 6 + 7 + 8)
        self.assertAlmostEqual(system, mobile + 9 + 10)
        self.assertLessEqual(requester, mobile)
        self.assertLessEqual(mobile, system)

    def test_result_wrapper_is_accepted(self):
        class Result:
            def __init__(self, energy):
                self.energy = energy

        b = breakdown()
        self.assertAlmostEqual(
            energy_scalar(Result(b), scope=SCOPE_SYSTEM),
            energy_scalar(b, scope=SCOPE_SYSTEM),
        )

    def test_scope_is_mandatory(self):
        for bad in (None, "", "solar", "SYSTEM "):
            with self.assertRaises(ValueError, msg=repr(bad)):
                energy_scalar(breakdown(), scope=bad)  # type: ignore[arg-type]

    def test_bad_input_is_rejected(self):
        with self.assertRaises(TypeError):
            energy_scalar(42, scope=SCOPE_SYSTEM)
        with self.assertRaises(TypeError):
            energy_scalar(object(), scope=SCOPE_MOBILE)

    def test_non_finite_is_rejected(self):
        with self.assertRaises(ValueError):
            energy_scalar(breakdown(mec_compute_joules_optional=float("nan")), scope=SCOPE_SYSTEM)
        with self.assertRaises(ValueError):
            energy_scalar(breakdown(ue_local_cpu_joules=float("inf")), scope=SCOPE_REQUESTER)

    def test_negative_boundary_is_rejected(self):
        # the BOUNDARY must be non-negative; a single negative component can still
        # leave a positive sum, so the check is on the scalar
        with self.assertRaises(ValueError):
            energy_scalar(breakdown(ue_local_cpu_joules=-100.0), scope=SCOPE_REQUESTER)

    def test_zero_mec_components_leave_system_equal_to_mobile(self):
        b = breakdown(mec_compute_joules_optional=0.0, mec_tx_joules_optional=0.0)
        self.assertAlmostEqual(
            energy_scalar(b, scope=SCOPE_SYSTEM), energy_scalar(b, scope=SCOPE_MOBILE)
        )


class TestConfiguredScalar(unittest.TestCase):
    def test_scope_comes_from_the_config(self):
        class Res:
            energy_scope = SCOPE_SYSTEM
            energy_model = None

        self.assertAlmostEqual(
            configured_energy_scalar(breakdown(), Res()),
            energy_scalar(breakdown(), scope=SCOPE_SYSTEM),
        )

    def test_missing_scope_is_loud(self):
        class Res:
            energy_scope = ""
            energy_model = None

        with self.assertRaises(ValueError):
            configured_energy_scalar(breakdown(), Res())

    def test_falls_back_to_the_model_declaration_only_when_explicit(self):
        class Model:
            energy_scope = SCOPE_MOBILE

        class Res:
            energy_scope = ""
            energy_model = Model()

        self.assertAlmostEqual(
            configured_energy_scalar(breakdown(), Res()),
            energy_scalar(breakdown(), scope=SCOPE_MOBILE),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
