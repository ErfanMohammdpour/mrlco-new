#!/usr/bin/env python3
"""Timing/energy physics consistency: measurement, labelling and the gate.

The primary config schedules time with the frozen rate table while accounting
energy with `physical_v1`, so `E_workload = kappa*C*f^2` and `P(f)*T_scheduled`
disagree by `R_scheduled / R_physical`. These tests pin the measurement, the
labels and the fail-loud gate; the probe (spec/energy_consistency_probe.py) turns
them into evidence on real graphs.
"""

from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _stub_optional(name: str) -> None:
    if name in sys.modules:
        return
    try:
        __import__(name)
    except Exception:
        sys.modules[name] = types.ModuleType(name)


for _name in ("gym", "gym.core", "graphviz", "pydotplus", "pydotplus.graphviz"):
    _stub_optional(_name)
if not hasattr(sys.modules.get("gym.core", types.ModuleType("gym.core")), "Env"):
    sys.modules.setdefault("gym", types.ModuleType("gym"))
    sys.modules.setdefault("gym.core", types.ModuleType("gym.core"))
    sys.modules["gym.core"].Env = type("Env", (), {})

from env.mec_offloaing_envs.scheduler.energy_model import TierSpec  # noqa: E402
from env.mec_offloaing_envs.scheduler.energy_telemetry import (  # noqa: E402
    duration_consistency,
    duration_consistency_kvs,
)
from env.mec_offloaing_envs.scheduler.primary_config import (  # noqa: E402
    EnergyPhysicsMismatch,
    energy_timing_consistency,
    require_energy_timing_consistency,
    resolved_physical_scheduler_config,
    resolved_primary_scheduler_config,
)

BUSY = {"UE_CPU": 4.0, "HELPER_CPU": 2.0, "MEC_CPU": 6.0}


def _fake_result(workload_per_tier):
    intervals = [
        SimpleNamespace(resource=name, start=0.0, end=seconds)
        for name, seconds in BUSY.items()
    ]
    energy = SimpleNamespace(
        ue_local_cpu_joules=workload_per_tier["ue"],
        helper_compute_joules=workload_per_tier["helper"],
        mec_compute_joules_optional=workload_per_tier["mec"],
    )
    return SimpleNamespace(
        resource_intervals=intervals, energy=energy, makespan_seconds=12.0,
        scheduler_config_sha256="",
    )


class TestTierArithmetic(unittest.TestCase):
    def test_duration_consistent_energy_is_power_times_seconds(self):
        tier = TierSpec(f_hz=1e9, kappa=1e-27, source="test")
        self.assertAlmostEqual(tier.compute_joules_from_duration(3.0), 3.0 * tier.implied_dynamic_power_w)
        self.assertEqual(tier.compute_joules_from_duration(0.0), 0.0)

    def test_physical_duration_reproduces_the_workload_energy_exactly(self):
        tier = TierSpec(f_hz=1e9, kappa=1e-27, source="test")
        workload, cpb = 125_000.0, 300.0
        physical_seconds = tier.compute_seconds(workload, cpb)
        self.assertAlmostEqual(
            tier.compute_joules(workload, cpb),
            tier.compute_joules_from_duration(physical_seconds),
            places=18,
        )

    def test_negative_duration_is_rejected(self):
        tier = TierSpec(f_hz=1e9, kappa=1e-27, source="test")
        with self.assertRaises(Exception):
            tier.compute_joules_from_duration(-1.0)


class TestConfigLabels(unittest.TestCase):
    def test_primary_is_labelled_mixed_and_physical_is_not(self):
        primary = resolved_primary_scheduler_config()
        physical = resolved_physical_scheduler_config()
        facts_primary = energy_timing_consistency(primary)
        facts_physical = energy_timing_consistency(physical)
        self.assertTrue(facts_primary["mixed_physics"])
        self.assertEqual(facts_primary["label"], "mixed_physics_legacy_timing_physical_energy")
        self.assertFalse(facts_physical["mixed_physics"])
        self.assertEqual(facts_physical["label"], "co_physical")
        self.assertTrue(facts_physical["timing_physical"])

    def test_gate_rejects_the_mixed_config_and_accepts_the_physical_one(self):
        with self.assertRaises(EnergyPhysicsMismatch):
            require_energy_timing_consistency(resolved_primary_scheduler_config())
        self.assertEqual(
            require_energy_timing_consistency(resolved_physical_scheduler_config())["label"],
            "co_physical",
        )


class TestDurationConsistency(unittest.TestCase):
    def test_mixed_config_reports_the_scheduled_over_physical_ratio(self):
        config = resolved_primary_scheduler_config()
        cal = {"ue": "UE_CPU", "helper": "HELPER_CPU", "mec": "MEC_CPU"}
        # build workload joules that follow the frozen-vs-physical rate ratio, so
        # the assertion is about the identity and not about invented numbers
        from env.mec_offloaing_envs.scheduler.energy_telemetry import _rate_ratio

        workload = {}
        for tier, calendar in cal.items():
            duration_j = config.energy_model.tier(tier).implied_dynamic_power_w * BUSY[calendar]
            workload[tier] = duration_j * _rate_ratio(config, tier)
        record = duration_consistency(_fake_result(workload), config)
        self.assertTrue(record["model_is_physical"])
        self.assertFalse(record["duration_consistent"])
        for tier, row in record["tiers"].items():
            expected = record["ratio_expected_scheduled_over_physical"][tier]
            self.assertAlmostEqual(row["ratio_workload_over_duration"], expected, places=9)
            self.assertAlmostEqual(
                row["duration_consistent_joules"],
                config.energy_model.tier(tier).implied_dynamic_power_w * row["busy_seconds"],
                places=12,
            )
        # the frozen table is slower than the physical rate, so the ratio exceeds 1
        self.assertGreater(record["tiers"]["ue"]["ratio_workload_over_duration"], 1.0)

    def test_co_physical_config_is_exactly_duration_consistent(self):
        config = resolved_physical_scheduler_config()
        tiers = {
            tier: config.energy_model.tier(tier).implied_dynamic_power_w * seconds
            for tier, seconds in (("ue", BUSY["UE_CPU"]), ("helper", BUSY["HELPER_CPU"]),
                                  ("mec", BUSY["MEC_CPU"]))
        }
        record = duration_consistency(_fake_result(tiers), config)
        self.assertTrue(record["duration_consistent"])
        self.assertLessEqual(record["max_abs_ratio_minus_one"], 1e-9)
        for row in record["tiers"].values():
            self.assertAlmostEqual(row["ratio_workload_over_duration"], 1.0, places=9)

    def test_csv_kvs_expose_the_flag_and_ratios(self):
        config = resolved_primary_scheduler_config()
        record = duration_consistency(_fake_result({"ue": 1.0, "helper": 1.0, "mec": 1.0}), config)
        kvs = duration_consistency_kvs(record)
        self.assertEqual(kvs["energy/consistency/model_is_physical"], 1.0)
        self.assertEqual(kvs["energy/consistency/duration_consistent"], 0.0)
        for tier in ("ue", "helper", "mec"):
            self.assertIn("energy/consistency/%s_ratio" % tier, kvs)
            self.assertGreater(kvs["energy/consistency/%s_ratio" % tier], 0.0)

    def test_legacy_accounting_reports_no_duration_number(self):
        from env.mec_offloaing_envs.scheduler.resources import ResourceConfig

        legacy = ResourceConfig.from_frozen_yaml(model="legacy")
        record = duration_consistency(_fake_result({"ue": 1.0, "helper": 0.0, "mec": 0.0}), legacy)
        self.assertFalse(record["model_is_physical"])
        self.assertFalse(record["duration_consistent"])
        self.assertEqual(record["tiers"]["ue"]["duration_consistent_joules"], 0.0)


class TestProbeVerdict(unittest.TestCase):
    def test_verdict_requires_off_unity_primary_and_unity_co_physical(self):
        from spec.energy_consistency_probe import verdict

        def block(ratio):
            return {"tiers": {"ue": {"mean_measured_ratio": ratio,
                                     "max_abs_identity_error": 0.0}}}

        self.assertEqual(verdict(block(2.52), block(1.0))["verdict"], "MIXED_PHYSICS_MEASURED")
        self.assertEqual(verdict(block(1.0), block(1.0))["verdict"], "INCONSISTENT_EVIDENCE")
        self.assertEqual(verdict(block(2.52), block(1.4))["verdict"], "INCONSISTENT_EVIDENCE")
        bad = block(2.52)
        bad["tiers"]["ue"]["max_abs_identity_error"] = 1e-3
        self.assertEqual(verdict(bad, block(1.0))["verdict"], "INCONSISTENT_EVIDENCE")

    def test_summarise_aggregates_synthetic_samples(self):
        from spec.energy_consistency_probe import summarise

        rows = [
            {"tiers": {"ue": {"busy_seconds": 2.0, "workload_joules": 4.0,
                              "duration_consistent_joules": 2.0,
                              "ratio_workload_over_duration": 2.0}},
             "expected_ratio": {"ue": 2.0}},
            {"tiers": {"ue": {"busy_seconds": 4.0, "workload_joules": 8.0,
                              "duration_consistent_joules": 4.0,
                              "ratio_workload_over_duration": 2.0}},
             "expected_ratio": {"ue": 2.0}},
        ]
        out = summarise(rows)
        self.assertEqual(out["ue"]["n_graphs"], 2)
        self.assertAlmostEqual(out["ue"]["mean_measured_ratio"], 2.0)
        self.assertAlmostEqual(out["ue"]["mean_expected_ratio"], 2.0)
        self.assertAlmostEqual(out["ue"]["max_abs_identity_error"], 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
