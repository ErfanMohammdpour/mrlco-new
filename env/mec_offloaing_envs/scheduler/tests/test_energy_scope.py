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


class TestReferenceScopeContract(unittest.TestCase):
    """E1.2 scope guard: plan names are not boundaries."""

    class Refs:
        def __init__(self, scope="", sha=""):
            self.energy_scope = scope
            self.scheduler_config_sha256 = sha

    def test_matching_scope_passes(self):
        from env.mec_offloaing_envs.scheduler.energy_scope import require_reference_scope

        self.assertEqual(
            require_reference_scope(self.Refs(SCOPE_SYSTEM), expected_scope=SCOPE_SYSTEM),
            SCOPE_SYSTEM,
        )

    def test_scope_mismatch_raises(self):
        from env.mec_offloaing_envs.scheduler.energy_scope import (
            EnergyReferenceMismatch,
            require_reference_scope,
        )

        with self.assertRaises(EnergyReferenceMismatch):
            require_reference_scope(self.Refs(SCOPE_MOBILE), expected_scope=SCOPE_SYSTEM)

    def test_metadata_free_reference_is_mobile_only(self):
        from env.mec_offloaing_envs.scheduler.energy_scope import (
            EnergyReferenceMismatch,
            require_reference_scope,
        )

        refs = self.Refs()
        self.assertEqual(
            require_reference_scope(refs, expected_scope=SCOPE_MOBILE), SCOPE_MOBILE
        )
        with self.assertRaises(EnergyReferenceMismatch):
            require_reference_scope(refs, expected_scope=SCOPE_SYSTEM)

    def test_strict_fingerprint_requires_metadata(self):
        from env.mec_offloaing_envs.scheduler.energy_scope import (
            EnergyReferenceMismatch,
            require_reference_scope,
        )

        with self.assertRaises(EnergyReferenceMismatch):
            require_reference_scope(
                self.Refs(SCOPE_MOBILE), expected_scope=SCOPE_MOBILE,
                expected_scheduler_config_sha256="a" * 64,
            )
        with self.assertRaises(EnergyReferenceMismatch):
            require_reference_scope(
                self.Refs(SCOPE_MOBILE, "b" * 64), expected_scope=SCOPE_MOBILE,
                expected_scheduler_config_sha256="a" * 64,
            )
        self.assertEqual(
            require_reference_scope(
                self.Refs(SCOPE_MOBILE, "a" * 64), expected_scope=SCOPE_MOBILE,
                expected_scheduler_config_sha256="a" * 64,
            ),
            SCOPE_MOBILE,
        )

    def test_invalid_expected_scope_raises(self):
        from env.mec_offloaing_envs.scheduler.energy_scope import (
            EnergyReferenceMismatch,
            require_reference_scope,
        )

        with self.assertRaises(EnergyReferenceMismatch):
            require_reference_scope(self.Refs(SCOPE_SYSTEM), expected_scope="solar")

    def test_schema_constant(self):
        from env.mec_offloaing_envs.scheduler.energy_scope import (
            REFERENCE_SCHEMA_VERSION,
        )

        self.assertEqual(REFERENCE_SCHEMA_VERSION, "energy_reference_ranges_v2")
