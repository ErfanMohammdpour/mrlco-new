#!/usr/bin/env python3
"""Production plumbing for the resolved scheduler config.

The resolved config (legacy timing + physical energy + system scope) must travel
from the yaml to train, validation and worker environments without being rebuilt,
dropped or reinterpreted, and its fingerprint must be stable and parameter
sensitive, because the workers are separate processes.
"""

from __future__ import annotations

import copy
import pickle
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
for _name in ("gym", "gym.core"):
    sys.modules.setdefault(_name, types.ModuleType(_name))
sys.modules["gym.core"].Env = type("Env", (), {})

from env.mec_offloaing_envs.scheduler import encoder_obs as eo  # noqa: E402
from env.mec_offloaing_envs.scheduler.adapter import (  # noqa: E402
    SchedulerConfigError,
    resource_config_from_cluster,
    schedule_via_adapter,
)
from env.mec_offloaing_envs.scheduler.energy_model import MODEL_LEGACY  # noqa: E402
from env.mec_offloaing_envs.scheduler.model import CanonicalDAG, CanonicalTask, Location  # noqa: E402
from env.mec_offloaing_envs.scheduler.resources import (  # noqa: E402
    TIMING_LEGACY,
    TIMING_PHYSICAL,
    ResourceConfig,
    canonical_provenance,
    resolved_config_sha256,
)
from env.mec_offloaing_envs.scheduler.static_bounds import static_action_bounds  # noqa: E402
from env.mec_offloaing_envs.scheduler.primary_config import (  # noqa: E402
    PRIMARY_SCHEDULER_AXES,
    resolved_primary_scheduler_config,
)

YAML = ROOT / "spec" / "frozen_experiment.yaml"
XI = 300.0


class FakeCluster:
    """Mirrors the Resources fields the adapter reads, in the frozen units."""

    def __init__(self, scheduler_config=None, **overrides):
        self.scheduler_config = scheduler_config
        self.mec_process_capable = overrides.get("mec", 10.0 * 1024 * 1024)
        self.mobile_process_capable = overrides.get("ue", 1.0 * 1024 * 1024)
        self.v2v_process_capable = overrides.get("helper", 1.0 * 1024 * 1024)
        self.bandwidth_up = overrides.get("up", 7.0)
        self.bandwidth_dl = overrides.get("dl", 7.0)
        self.v2v_bandwidth = overrides.get("v2v", 5.0)
        self.energy_config = overrides.get("energy_config")


def dag(n=3):
    tasks = [
        CanonicalTask(
            task_id=i,
            compute_workload_bytes=2_000_000 if i == 0 else 200_000,
            task_output_bytes=20_000,
            external_input_bytes=2_000_000 if i == 0 else 0,
        )
        for i in range(n)
    ]
    return CanonicalDAG.from_records(tasks, [(i, i + 1, 20_000) for i in range(n - 1)])


class TestObjectPreservation(unittest.TestCase):
    def test_scheduler_config_is_returned_unrebuilt(self):
        cfg = resolved_primary_scheduler_config()
        cluster = FakeCluster(scheduler_config=cfg)
        again = resource_config_from_cluster(cluster, strict=True)
        self.assertIs(again, cfg)
        self.assertEqual(resolved_config_sha256(again), resolved_config_sha256(cfg))
        self.assertEqual(again.energy_model.model, "physical_v1")
        self.assertEqual(again.radio_model.model, "physical_v1")
        self.assertEqual(again.timing_model, TIMING_LEGACY)
        self.assertEqual(again.energy_scope, "system")
        self.assertTrue(again.source_config_sha256)

    def test_wrong_type_is_rejected(self):
        cluster = FakeCluster(scheduler_config={"not": "a config"})
        with self.assertRaises(SchedulerConfigError):
            resource_config_from_cluster(cluster, strict=True)


class TestLegacyFallbackAndStrict(unittest.TestCase):
    def test_fallback_keeps_the_old_behaviour(self):
        cluster = FakeCluster()
        cfg = resource_config_from_cluster(cluster)
        self.assertEqual(cfg.timing_model, TIMING_LEGACY)
        self.assertEqual(cfg.radio_timing_model, TIMING_LEGACY)
        self.assertIsNone(cfg.energy_model)
        self.assertFalse(cfg.physical)
        self.assertAlmostEqual(
            cfg.mec_uplink_bytes_per_second, 7.0 * 1024 * 1024 / 8.0, places=9
        )

    def test_strict_primary_refuses_the_fallback(self):
        with self.assertRaises(SchedulerConfigError):
            resource_config_from_cluster(FakeCluster(), strict=True)


class TestRateConsistency(unittest.TestCase):
    def test_matching_cluster_passes(self):
        cfg = resolved_primary_scheduler_config()
        resource_config_from_cluster(FakeCluster(scheduler_config=cfg), strict=True)

    def test_mismatch_fails_loudly(self):
        cfg = resolved_primary_scheduler_config()
        cluster = FakeCluster(scheduler_config=cfg, mec=1.0)
        with self.assertRaises(SchedulerConfigError) as ctx:
            resource_config_from_cluster(cluster, strict=True)
        self.assertIn("Resources.mec_process_capable disagrees", str(ctx.exception))

    def test_bandwidth_mismatch_fails_loudly(self):
        cfg = resolved_primary_scheduler_config()
        with self.assertRaises(SchedulerConfigError) as ctx:
            resource_config_from_cluster(
                FakeCluster(scheduler_config=cfg, up=3.0), strict=True
            )
        self.assertIn("Resources.bandwidth_up disagrees", str(ctx.exception))


class TestTrainValidationParity(unittest.TestCase):
    def test_both_environments_resolve_the_same_config(self):
        # the trainer builds ONE cluster and hands it to train and validation
        cfg = resolved_primary_scheduler_config()
        cluster = FakeCluster(scheduler_config=cfg)
        train = resource_config_from_cluster(cluster, strict=True)
        validation = resource_config_from_cluster(cluster, strict=True)
        self.assertEqual(train, validation)
        self.assertEqual(
            resolved_config_sha256(train), resolved_config_sha256(validation)
        )
        self.assertEqual(
            (train.timing_model, train.energy_model.model, train.energy_scope),
            (validation.timing_model, validation.energy_model.model, validation.energy_scope),
        )

    def test_axes_are_the_agreed_primary_axes(self):
        cfg = resolved_primary_scheduler_config()
        self.assertEqual(cfg.timing_model, PRIMARY_SCHEDULER_AXES["timing_model"])
        self.assertEqual(cfg.radio_timing_model, PRIMARY_SCHEDULER_AXES["radio_timing_model"])
        self.assertEqual(cfg.energy_model.model, PRIMARY_SCHEDULER_AXES["energy_model"])
        self.assertEqual(cfg.radio_model.model, PRIMARY_SCHEDULER_AXES["radio_model"])
        self.assertEqual(cfg.energy_scope, PRIMARY_SCHEDULER_AXES["energy_scope"])


class TestWorkerSerialization(unittest.TestCase):
    def test_pickle_round_trip_preserves_everything(self):
        cfg = resolved_primary_scheduler_config()
        for protocol in (2, pickle.HIGHEST_PROTOCOL):
            restored = pickle.loads(pickle.dumps(cfg, protocol=protocol))
            self.assertEqual(resolved_config_sha256(restored), resolved_config_sha256(cfg))
            for loc in (Location.UE, Location.MEC, Location.HELPER):
                self.assertAlmostEqual(
                    restored.cpu_rate_bytes_per_second(loc, XI),
                    cfg.cpu_rate_bytes_per_second(loc, XI),
                    places=12,
                )
            for hop in ("MEC_UL", "MEC_DL", "V2V"):
                self.assertAlmostEqual(restored.hop_rate(hop), cfg.hop_rate(hop), places=12)

    def test_deepcopy_preserves_the_fingerprint(self):
        cfg = resolved_primary_scheduler_config()
        self.assertEqual(
            resolved_config_sha256(copy.deepcopy(cfg)), resolved_config_sha256(cfg)
        )


class TestFingerprint(unittest.TestCase):
    def test_deterministic(self):
        self.assertEqual(
            resolved_config_sha256(resolved_primary_scheduler_config()),
            resolved_config_sha256(resolved_primary_scheduler_config()),
        )

    def test_axis_change_changes_the_hash(self):
        base = resolved_primary_scheduler_config()
        physical_timing = resolved_primary_scheduler_config(
            overrides={"timing_model": TIMING_PHYSICAL}
        )
        self.assertNotEqual(
            resolved_config_sha256(base), resolved_config_sha256(physical_timing)
        )
        legacy_radio = resolved_primary_scheduler_config(
            overrides={"radio_model": MODEL_LEGACY}
        )
        self.assertNotEqual(
            resolved_config_sha256(base), resolved_config_sha256(legacy_radio)
        )

    def test_legacy_energy_with_system_scope_is_a_scope_conflict(self):
        # the legacy accounting model declares the mobile boundary, so asking for
        # system scope on top of it is a conflict, not a silent override
        with self.assertRaises(ValueError) as ctx:
            resolved_primary_scheduler_config(overrides={"energy_model": MODEL_LEGACY})
        self.assertIn("energy_scope conflict", str(ctx.exception))

    def test_same_content_different_path_same_fingerprint(self):
        base = resolved_primary_scheduler_config()
        with tempfile.TemporaryDirectory() as tmp:
            other = Path(tmp) / "frozen_experiment.yaml"
            shutil.copy(YAML, other)
            moved = resolved_primary_scheduler_config(path=other)
        self.assertEqual(resolved_config_sha256(moved), resolved_config_sha256(base))
        self.assertEqual(moved.source_config_sha256, base.source_config_sha256)

    def test_provenance_records_the_required_fields(self):
        blob = canonical_provenance(resolved_primary_scheduler_config())
        for key in (
            "schema", "timing_model", "radio_timing_model", "energy_model",
            "radio_model", "energy_scope", "cycles_per_bit", "legacy_frozen_rates",
            "legacy_power", "physical_tiers", "radio_parameters",
            "source_config_sha256", "overrides",
        ):
            self.assertIn(key, blob)
        self.assertEqual(blob["energy_scope"], "system")
        self.assertIn("mec", blob["physical_tiers"])


class TestConflicts(unittest.TestCase):
    def test_scope_conflict_raises(self):
        cfg = resolved_primary_scheduler_config()
        with self.assertRaises(ValueError):
            ResourceConfig(**{**cfg.__dict__, "energy_scope": "mobile"})

    def test_unknown_scope_raises(self):
        cfg = resolved_primary_scheduler_config()
        with self.assertRaises(ValueError):
            ResourceConfig(**{**cfg.__dict__, "energy_scope": "planet"})

    def test_missing_config_in_strict_primary_raises(self):
        # the trainer guard fires before any stack build (asserted textually so
        # this test needs no TensorFlow), and the adapter refuses the fallback
        src = (ROOT / "meta_trainer.py").read_text()
        self.assertIn("strict_scheduler_config=True requires", src)
        with self.assertRaises(SchedulerConfigError):
            resource_config_from_cluster(FakeCluster(), strict=True)


class TestInvarianceAfterPlumbing(unittest.TestCase):
    def test_primary_config_keeps_legacy_timing(self):
        primary = resolved_primary_scheduler_config()          # legacy timing
        legacy = ResourceConfig.from_frozen_yaml(YAML, model=MODEL_LEGACY)
        for loc in (Location.UE, Location.MEC, Location.HELPER):
            self.assertAlmostEqual(
                primary.cpu_rate_bytes_per_second(loc, XI),
                legacy.cpu_rate_bytes_per_second(loc, XI),
                places=12,
            )
        for hop in ("MEC_UL", "MEC_DL", "V2V"):
            self.assertAlmostEqual(primary.hop_rate(hop), legacy.hop_rate(hop), places=12)
        graph = dag()
        order = [0, 1, 2]
        ba = static_action_bounds(graph, order, primary, cycles_per_bit=XI)
        bb = static_action_bounds(graph, order, legacy, cycles_per_bit=XI)
        self.assertEqual(ba.ready_lb, bb.ready_lb)
        self.assertTrue(primary.physical and not legacy.physical)


if __name__ == "__main__":
    unittest.main(verbosity=2)
