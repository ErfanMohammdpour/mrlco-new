#!/usr/bin/env python3
"""E2.2: `energy_telemetry_v1` — same-result scalars, episode weighting, CSV."""

from __future__ import annotations

import pickle
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
for _name in ("gym", "gym.core"):
    sys.modules.setdefault(_name, types.ModuleType(_name))
sys.modules["gym.core"].Env = type("Env", (), {})
# utils.logger imports tensorflow at module level for its TensorBoard writer; the
# CSV path needs none of it.
if "tensorflow" not in sys.modules:
    sys.modules["tensorflow"] = types.ModuleType("tensorflow")
if "joblib" not in sys.modules:
    sys.modules["joblib"] = types.ModuleType("joblib")

import numpy as np  # noqa: E402

from env.mec_offloaing_envs.offloading_env import OffloadingEnvironment  # noqa: E402
from env.mec_offloaing_envs.scheduler.adapter import schedule_via_adapter  # noqa: E402
from env.mec_offloaing_envs.scheduler.energy_api import (  # noqa: E402
    compute_reference_ranges,
)
from env.mec_offloaing_envs.scheduler.energy_scope import (  # noqa: E402
    SCOPE_MOBILE,
    SCOPE_REQUESTER,
    SCOPE_SYSTEM,
    energy_scalar,
)
from env.mec_offloaing_envs.scheduler.energy_telemetry import (  # noqa: E402
    AVERAGE_ENERGY_LEGACY_KEY,
    AVERAGE_ENERGY_LEGACY_SCOPE,
    CSV_COLUMNS,
    TELEMETRY_FIELDS,
    TELEMETRY_SCHEMA_VERSION,
    EnergyTelemetryError,
    aggregate_energy_telemetry,
    build_energy_telemetry,
    collect_energy_telemetry,
    legacy_average_energy_kvs,
    telemetry_csv_kvs,
    validate_energy_telemetry,
)
from env.mec_offloaing_envs.scheduler.primary_config import (  # noqa: E402
    resolved_primary_scheduler_config,
)
from env.mec_offloaing_envs.scheduler.reward import (  # noqa: E402
    telescoping_token_rewards,
)
from env.mec_offloaing_envs.scheduler.resources import (  # noqa: E402
    resolved_config_sha256,
)
from utils.logger import CSVOutputFormat  # noqa: E402


class _FakeTask:
    def __init__(self, proc: int, tx: int):
        self.processing_data_size = proc
        self.transmission_data_size = tx


class _FakeTG:
    def __init__(self):
        self.task_number = 3
        self.task_list = [
            _FakeTask(1_048_576, 458_752),
            _FakeTask(1_048_576, 458_752),
            _FakeTask(1_048_576, 458_752),
        ]
        self.prioritize_sequence = [0, 1, 2]
        self.pre_task_sets = [{}, {0}, {1}]
        self.edge_set = [
            [0, 0, 0, 458_752, 1, 1, 0],
            [1, 1, 0, 458_752, 2, 2, 0],
        ]


def _setup():
    tg = _FakeTG()
    res = resolved_primary_scheduler_config()
    plan = [(0, 0), (1, 2), (2, 1)]
    result, _, _ = schedule_via_adapter(tg, plan, res)
    refs = compute_reference_ranges(tg, res)
    return tg, res, plan, result, refs


class TestBuild(unittest.TestCase):
    def test_fields_values_and_scope(self):
        tg, res, _plan, result, _refs = _setup()
        telemetry = build_energy_telemetry(result, res)
        self.assertEqual(set(telemetry), set(TELEMETRY_FIELDS))
        self.assertEqual(telemetry["schema_version"], TELEMETRY_SCHEMA_VERSION)
        self.assertEqual(telemetry["primary_scope"], SCOPE_SYSTEM)
        self.assertEqual(telemetry["scheduler_config_sha256"], resolved_config_sha256(res))
        self.assertAlmostEqual(telemetry["requester_joules"], energy_scalar(result, scope=SCOPE_REQUESTER), places=12)
        self.assertAlmostEqual(telemetry["mobile_joules"], energy_scalar(result, scope=SCOPE_MOBILE), places=12)
        self.assertAlmostEqual(telemetry["system_joules"], energy_scalar(result, scope=SCOPE_SYSTEM), places=12)
        self.assertEqual(telemetry["primary_joules"], telemetry["system_joules"])
        self.assertLessEqual(telemetry["requester_joules"], telemetry["mobile_joules"])
        self.assertLessEqual(telemetry["mobile_joules"], telemetry["system_joules"])

    def test_no_second_schedule_or_replay(self):
        _tg, res, _plan, result, _refs = _setup()
        with mock.patch(
            "env.mec_offloaing_envs.scheduler.adapter.schedule_via_adapter",
            side_effect=AssertionError("telemetry must not schedule"),
        ):
            telemetry = build_energy_telemetry(result, res)
        self.assertEqual(telemetry["primary_scope"], SCOPE_SYSTEM)

    def test_missing_result_fingerprint_raises(self):
        import dataclasses

        _tg, res, _plan, result, _refs = _setup()
        stripped = dataclasses.replace(result, scheduler_config_sha256="")
        with self.assertRaises(EnergyTelemetryError):
            build_energy_telemetry(stripped, res)

    def test_fingerprint_mismatch_raises(self):
        _tg, _res, _plan, result, _refs = _setup()
        other = resolved_primary_scheduler_config(overrides={"radio_model": "legacy"})
        self.assertNotEqual(resolved_config_sha256(other), result.scheduler_config_sha256)
        with self.assertRaises(EnergyTelemetryError):
            build_energy_telemetry(result, other)

    def test_record_is_picklable_for_parallel_workers(self):
        _tg, res, _plan, result, _refs = _setup()
        telemetry = build_energy_telemetry(result, res)
        self.assertEqual(pickle.loads(pickle.dumps(telemetry)), telemetry)


class TestValidation(unittest.TestCase):
    def _record(self, **overrides):
        record = {
            "requester_joules": 1.0,
            "mobile_joules": 2.0,
            "system_joules": 3.0,
            "primary_scope": SCOPE_SYSTEM,
            "primary_joules": 3.0,
            "scheduler_config_sha256": "a" * 64,
            "schema_version": TELEMETRY_SCHEMA_VERSION,
        }
        record.update(overrides)
        return record

    def test_valid_record_normalises(self):
        checked = validate_energy_telemetry(self._record())
        self.assertEqual(checked["system_joules"], 3.0)

    def test_missing_malformed_non_finite_and_negative_raise(self):
        for bad in (
            {k: v for k, v in self._record().items() if k != "system_joules"},
            self._record(schema_version="energy_telemetry_v0"),
            self._record(primary_scope="solar"),
            self._record(scheduler_config_sha256="not-a-sha"),
            self._record(system_joules=float("nan")),
            self._record(mobile_joules=float("inf")),
            self._record(requester_joules=-1.0),
            self._record(system_joules="joules"),
            "not-a-mapping",
        ):
            with self.assertRaises(EnergyTelemetryError, msg=repr(bad)):
                validate_energy_telemetry(bad)


class TestAggregation(unittest.TestCase):
    def _record(self, requester, mobile, system, sha="a" * 64, scope=SCOPE_SYSTEM):
        return {
            "requester_joules": requester,
            "mobile_joules": mobile,
            "system_joules": system,
            "primary_scope": scope,
            "primary_joules": system,
            "scheduler_config_sha256": sha,
            "schema_version": TELEMETRY_SCHEMA_VERSION,
        }

    def test_episode_weighted_not_mean_of_means(self):
        # two episodes of one kind and one of another: the single episode must
        # not get half the weight of the pair
        rows = [
            self._record(1.0, 2.0, 3.0),
            self._record(1.0, 2.0, 3.0),
            self._record(4.0, 5.0, 6.0),
        ]
        agg = aggregate_energy_telemetry(rows)
        self.assertAlmostEqual(agg["mobile_joules"], (2.0 + 2.0 + 5.0) / 3.0, places=12)
        self.assertNotAlmostEqual(agg["mobile_joules"], (2.0 + 5.0) / 2.0, places=6)
        self.assertEqual(agg["n_episodes"], 3.0)
        self.assertEqual(agg["primary_scope"], SCOPE_SYSTEM)

    def test_empty_raises(self):
        with self.assertRaises(EnergyTelemetryError):
            aggregate_energy_telemetry([])

    def test_inconsistent_scope_or_fingerprint_raises(self):
        with self.assertRaises(EnergyTelemetryError):
            aggregate_energy_telemetry([
                self._record(1.0, 2.0, 3.0),
                self._record(1.0, 2.0, 3.0, scope=SCOPE_MOBILE),
            ])
        with self.assertRaises(EnergyTelemetryError):
            aggregate_energy_telemetry([
                self._record(1.0, 2.0, 3.0),
                self._record(1.0, 2.0, 3.0, sha="b" * 64),
            ])

    def test_csv_columns_are_exactly_the_contract(self):
        agg = aggregate_energy_telemetry([self._record(1.0, 2.0, 3.0)])
        kvs = telemetry_csv_kvs(agg)
        self.assertEqual(
            set(kvs),
            {
                "energy/requester_joules",
                "energy/mobile_joules",
                "energy/system_joules",
                "energy/primary_joules",
                "energy/primary_scope",
            },
        )
        self.assertEqual(kvs["energy/primary_scope"], SCOPE_SYSTEM)
        self.assertEqual(CSV_COLUMNS["primary_scope"], "energy/primary_scope")

    def test_collect_from_samples_data(self):
        a = self._record(1.0, 2.0, 3.0)
        b = self._record(4.0, 5.0, 6.0)
        batch = [{"energy_telemetry": [a]}, {"energy_telemetry": [b]}, {"energy": []}]
        self.assertEqual(collect_energy_telemetry(batch), [a, b])


class _FakeCluster:
    use_energy = True
    reward_mode = "publication"
    energy_config = {"energy_telemetry": True}

    def reset(self):
        pass


class _FakeEnvSelf:
    """Minimal `self` for the unbound env reward method (no data files needed)."""

    def __init__(self, res, refs, telemetry_enabled):
        self.resource_cluster = _FakeCluster()
        self.scheduler_resources = res
        self.constraint_controller = None
        self.constraint_spec = None
        self.shaping_discount = 1.0
        self.energy_telemetry_enabled = telemetry_enabled
        self.last_energy_telemetry = None
        self.validation_plans_enabled = False
        self.last_validation_plans = None
        self._refs = refs

    def get_reference_ranges(self, task_graph, scope=None):
        return self._refs


class TestEnvRolloutPath(unittest.TestCase):
    def test_telemetry_comes_from_the_reward_result_without_extra_schedule(self):
        import env.mec_offloaing_envs.scheduler.reward as reward_module

        tg, res, plan, result, refs = _setup()
        original = reward_module.schedule_via_adapter
        calls = {"n": 0}

        def counting(*args, **kwargs):
            calls["n"] += 1
            return original(*args, **kwargs)

        with mock.patch.object(reward_module, "schedule_via_adapter", counting):
            fake = _FakeEnvSelf(res, refs, telemetry_enabled=True)
            target, finish, energy = OffloadingEnvironment.get_reward_batch_step_by_step(
                fake, [plan], [tg], None, None
            )
        with_telemetry_calls = calls["n"]

        calls["n"] = 0
        with mock.patch.object(reward_module, "schedule_via_adapter", counting):
            plain = _FakeEnvSelf(res, refs, telemetry_enabled=False)
            OffloadingEnvironment.get_reward_batch_step_by_step(plain, [plan], [tg], None, None)
        without_telemetry_calls = calls["n"]

        # enable/disable must not change the number of schedules
        self.assertEqual(with_telemetry_calls, without_telemetry_calls)
        self.assertIsNone(plain.last_energy_telemetry)

        tel = fake.last_energy_telemetry
        self.assertEqual(len(tel), 1)
        self.assertAlmostEqual(tel[0]["mobile_joules"], energy_scalar(result, scope=SCOPE_MOBILE), places=12)
        self.assertAlmostEqual(tel[0]["system_joules"], energy_scalar(result, scope=SCOPE_SYSTEM), places=12)
        # the legacy 3-tuple is unchanged in shape
        self.assertEqual(len(energy), 1)

    def test_telemetry_does_not_change_the_reward(self):
        tg, res, plan, _result, refs = _setup()
        plain = _FakeEnvSelf(res, refs, telemetry_enabled=False)
        reward_plain, _, _ = OffloadingEnvironment.get_reward_batch_step_by_step(
            plain, [plan], [tg], None, None
        )
        fake = _FakeEnvSelf(res, refs, telemetry_enabled=True)
        reward_telemetry, _, _ = OffloadingEnvironment.get_reward_batch_step_by_step(
            fake, [plan], [tg], None, None
        )
        np.testing.assert_array_equal(reward_plain, reward_telemetry)


class TestLegacyAverageEnergyLabel(unittest.TestCase):
    def test_average_energy_stays_mobile_and_is_labelled(self):
        kvs = legacy_average_energy_kvs(3.5)
        self.assertEqual(kvs[AVERAGE_ENERGY_LEGACY_KEY], 3.5)
        self.assertEqual(kvs["energy/average_energy_scope"], SCOPE_MOBILE)
        self.assertEqual(AVERAGE_ENERGY_LEGACY_SCOPE, SCOPE_MOBILE)

    def test_reporting_reference_is_not_mass_converted_to_system(self):
        # j_report still consumes the legacy mobile reference unchanged
        from env.mec_offloaing_envs.scheduler.energy_api import (
            ReferenceRanges,
            j_report,
        )

        refs = ReferenceRanges(L_ue=1.0, L_mec=2.0, L_helper=3.0,
                               E_ue=10.0, E_mec=20.0, E_helper=30.0)
        self.assertGreaterEqual(j_report(1.5, 15.0, refs), 0.0)


class TestCsvAlignment(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="energy_telemetry_csv_"))
        self.path = self.tmp / "progress.csv"

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmp, ignore_errors=True)

    def _record(self):
        return {
            "requester_joules": 1.0, "mobile_joules": 2.0, "system_joules": 3.0,
            "primary_scope": SCOPE_SYSTEM, "primary_joules": 3.0,
            "scheduler_config_sha256": "a" * 64,
            "schema_version": TELEMETRY_SCHEMA_VERSION,
        }

    def test_late_energy_columns_keep_header_and_rows_aligned(self):
        import csv

        first = {"Average energy,": 2.0, "Itr": 1}
        second = dict(first)
        # telemetry columns arrive LATE (after the first row)
        second.update(telemetry_csv_kvs(aggregate_energy_telemetry([self._record()])))
        writer = CSVOutputFormat(str(self.path))
        writer.writekvs(first)
        writer.writekvs(second)
        writer.close()

        with self.path.open() as handle:
            raw = list(csv.reader(handle))
        header, rows = raw[0], raw[1:]
        self.assertIn("Average energy,", header)
        self.assertIn("energy/primary_scope", header)
        for row in rows:
            self.assertEqual(len(row), len(header), row)
        by_row = [dict(zip(header, row)) for row in rows]
        self.assertEqual(by_row[0]["energy/primary_scope"], "")   # not logged yet
        self.assertEqual(by_row[1]["energy/primary_scope"], SCOPE_SYSTEM)
        self.assertEqual(by_row[1]["energy/system_joules"], "3.0")
        self.assertEqual(by_row[1]["Average energy,"], "2.0")


if __name__ == "__main__":
    unittest.main(verbosity=2)
