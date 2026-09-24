#!/usr/bin/env python3
"""E1.2: scoped reference ranges — v2 schema, validation, and the cache contract."""

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


from env.mec_offloaing_envs.scheduler.energy_api import (  # noqa: E402
    REFERENCE_MODE_PANEL,
    REFERENCE_MODE_PURE,
    ReferenceRanges,
    compute_reference_ranges,
    compute_scoped_reference_ranges,
    j_report,
)
from env.mec_offloaing_envs.scheduler.energy_cache import (  # noqa: E402
    get_or_build_reference_ranges,
    reference_ranges_cache_key,
    scheduling_graph_fingerprint,
)
from env.mec_offloaing_envs.scheduler.energy_scope import (  # noqa: E402
    REFERENCE_SCHEMA_VERSION,
    SCOPE_MOBILE,
    SCOPE_REQUESTER,
    SCOPE_SYSTEM,
    EnergyReferenceMismatch,
    require_reference_scope,
)
from env.mec_offloaing_envs.scheduler.primary_config import (  # noqa: E402
    resolved_primary_scheduler_config,
)
from env.mec_offloaing_envs.scheduler.resources import ResourceConfig  # noqa: E402


class _FakeTask:
    def __init__(self, proc: int, tx: int):
        self.processing_data_size = proc
        self.transmission_data_size = tx


class _FakeTG:
    """Legacy OffloadingTaskGraph chain 0->1->2."""

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


def _primary():
    return resolved_primary_scheduler_config()


def _legacy():
    return ResourceConfig.from_frozen_yaml()


class TestScopedConstruction(unittest.TestCase):
    def test_records_boundary_fingerprint_and_schema(self):
        refs = compute_scoped_reference_ranges(
            _FakeTG(), _primary(), energy_scope=SCOPE_SYSTEM
        )
        self.assertEqual(refs.energy_scope, SCOPE_SYSTEM)
        self.assertEqual(refs.schema_version, REFERENCE_SCHEMA_VERSION)
        self.assertEqual(len(refs.scheduler_config_sha256), 64)
        self.assertTrue(refs.is_primary)
        require_reference_scope(
            refs,
            expected_scope=SCOPE_SYSTEM,
            expected_scheduler_config_sha256=refs.scheduler_config_sha256,
        )

    def test_plan_aliases_are_the_historical_fields(self):
        refs = compute_scoped_reference_ranges(
            _FakeTG(), _primary(), energy_scope=SCOPE_SYSTEM
        )
        self.assertEqual(refs.plan_all_ue_energy_j, refs.E_ue)
        self.assertEqual(refs.plan_all_mec_energy_j, refs.E_mec)
        self.assertEqual(refs.plan_all_helper_energy_j, refs.E_helper)

    def test_scoped_energy_matches_the_accessor_each_plan(self):
        from env.mec_offloaing_envs.scheduler.adapter import schedule_via_adapter
        from env.mec_offloaing_envs.scheduler.energy_api import pure_location_plan
        from env.mec_offloaing_envs.scheduler.energy_scope import energy_scalar

        tg = _FakeTG()
        res = _primary()
        scope = SCOPE_SYSTEM
        refs = compute_scoped_reference_ranges(tg, res, energy_scope=scope)
        for action, value in ((0, refs.E_ue), (1, refs.E_mec), (2, refs.E_helper)):
            result, _, _ = schedule_via_adapter(
                tg, pure_location_plan(tg.prioritize_sequence, action), res
            )
            self.assertAlmostEqual(value, energy_scalar(result, scope=scope), places=12)

    def test_system_scope_captures_mec_energy_the_mobile_boundary_misses(self):
        # primary config is physical_v1 + system: an all-MEC plan spends MEC compute
        tg = _FakeTG()
        res = _primary()
        mobile = compute_scoped_reference_ranges(tg, res, energy_scope=SCOPE_MOBILE)
        system = compute_scoped_reference_ranges(tg, res, energy_scope=SCOPE_SYSTEM)
        self.assertGreater(mobile.E_mec, 0.0)
        self.assertGreater(system.E_mec, mobile.E_mec)
        # requester plan energy is identical at both boundaries (UE-only work)
        self.assertAlmostEqual(mobile.E_ue, system.E_ue, places=12)

    def test_legacy_wrapper_is_mobile_and_numerically_identical(self):
        tg = _FakeTG()
        res = _legacy()
        legacy = compute_reference_ranges(tg, res)
        scoped = compute_scoped_reference_ranges(tg, res, energy_scope=SCOPE_MOBILE)
        self.assertEqual(legacy.energy_scope, SCOPE_MOBILE)
        for name in ("L_ue", "L_mec", "L_helper", "E_ue", "E_mec", "E_helper"):
            self.assertAlmostEqual(getattr(legacy, name), getattr(scoped, name), places=12)

    def test_scope_is_mandatory(self):
        with self.assertRaises(TypeError):
            compute_scoped_reference_ranges(_FakeTG(), _primary())  # type: ignore[call-arg]
        with self.assertRaises(ValueError):
            compute_scoped_reference_ranges(_FakeTG(), _primary(), energy_scope="")
        with self.assertRaises(ValueError):
            compute_scoped_reference_ranges(_FakeTG(), _primary(), energy_scope="solar")


class TestValidation(unittest.TestCase):
    def _kwargs(self, **overrides):
        kwargs = dict(
            L_ue=1.0, L_mec=2.0, L_helper=3.0,
            E_ue=10.0, E_mec=20.0, E_helper=30.0,
        )
        kwargs.update(overrides)
        return kwargs

    def test_negative_and_non_finite_are_rejected(self):
        with self.assertRaises(ValueError):
            ReferenceRanges(**self._kwargs(E_ue=-1.0))
        with self.assertRaises(ValueError):
            ReferenceRanges(**self._kwargs(L_mec=float("nan")))
        with self.assertRaises(ValueError):
            ReferenceRanges(**self._kwargs(E_helper=float("inf")))

    def test_bad_scope_and_schema_are_rejected(self):
        with self.assertRaises(ValueError):
            ReferenceRanges(**self._kwargs(energy_scope="solar"))
        with self.assertRaises(ValueError):
            ReferenceRanges(
                **self._kwargs(
                    energy_scope=SCOPE_SYSTEM,
                    scheduler_config_sha256="a" * 64,
                    schema_version="energy_reference_ranges_v1",
                )
            )

    def test_half_metadata_is_rejected(self):
        with self.assertRaises(ValueError):
            ReferenceRanges(**self._kwargs(energy_scope=SCOPE_SYSTEM))
        with self.assertRaises(ValueError):
            ReferenceRanges(**self._kwargs(scheduler_config_sha256="a" * 64))
        with self.assertRaises(ValueError):
            ReferenceRanges(
                **self._kwargs(energy_scope=SCOPE_SYSTEM, scheduler_config_sha256="not-a-sha")
            )

    def test_panel_mode_requires_ordered_fields(self):
        with self.assertRaises(ValueError):
            ReferenceRanges(**self._kwargs(reference_mode=REFERENCE_MODE_PANEL))
        with self.assertRaises(ValueError):
            ReferenceRanges(
                **self._kwargs(
                    reference_mode=REFERENCE_MODE_PANEL,
                    L_panel_min=5.0, L_panel_max=1.0,
                    E_panel_min=1.0, E_panel_max=2.0,
                )
            )

    def test_scoped_construction_satisfies_min_le_max(self):
        refs = compute_scoped_reference_ranges(_FakeTG(), _primary(), energy_scope=SCOPE_SYSTEM)
        self.assertLessEqual(refs.L_ref_min, refs.L_ref_max)
        self.assertLessEqual(refs.E_ref_min, refs.E_ref_max)
        self.assertLessEqual(refs.L_panel_min, refs.L_panel_max)
        self.assertLessEqual(refs.E_panel_min, refs.E_panel_max)

    def test_metadata_free_object_is_not_primary(self):
        refs = ReferenceRanges(**self._kwargs())
        self.assertFalse(refs.is_primary)
        self.assertEqual(require_reference_scope(refs, expected_scope=SCOPE_MOBILE), SCOPE_MOBILE)
        with self.assertRaises(EnergyReferenceMismatch):
            require_reference_scope(refs, expected_scope=SCOPE_SYSTEM)
        with self.assertRaises(EnergyReferenceMismatch):
            require_reference_scope(
                refs, expected_scope=SCOPE_MOBILE,
                expected_scheduler_config_sha256="a" * 64,
            )

    def test_requester_scope_cannot_use_a_mobile_reference(self):
        refs = compute_reference_ranges(_FakeTG(), _legacy())
        with self.assertRaises(EnergyReferenceMismatch):
            require_reference_scope(refs, expected_scope=SCOPE_REQUESTER)

    def test_j_report_rejects_a_scope_mismatch(self):
        refs = compute_reference_ranges(_FakeTG(), _legacy())
        with self.assertRaises(EnergyReferenceMismatch):
            j_report(refs.L_ue, refs.E_ue, refs, energy_scope=SCOPE_SYSTEM)


class TestCacheHelper(unittest.TestCase):
    def setUp(self):
        self.tg = _FakeTG()
        self.res = _primary()
        self.order = [0, 1, 2]
        self.cache: dict = {}

    def _get(self, **overrides):
        kwargs = dict(
            graph=self.tg,
            order=self.order,
            reference_mode=REFERENCE_MODE_PURE,
            energy_scope=SCOPE_SYSTEM,
            resources=self.res,
            panel_max_passes=2,
        )
        kwargs.update(overrides)
        return get_or_build_reference_ranges(self.cache, **kwargs)

    def test_hit_returns_the_same_object_and_misses_once(self):
        first = self._get()
        second = self._get()
        self.assertIs(first, second)
        self.assertEqual(len(self.cache), 1)

    def test_every_input_change_builds_a_new_entry(self):
        self._get()
        self._get(energy_scope=SCOPE_MOBILE)
        self._get(reference_mode=REFERENCE_MODE_PANEL)
        self._get(panel_extra={"x": 1})
        other = resolved_primary_scheduler_config(overrides={"radio_model": "legacy"})
        self._get(resources=other)
        self.assertEqual(len(self.cache), 5)

    def test_poisoned_cache_entry_fails_loud(self):
        key = reference_ranges_cache_key(
            graph=self.tg,
            order=self.order,
            reference_mode=REFERENCE_MODE_PURE,
            energy_scope=SCOPE_SYSTEM,
            resources=self.res,
            panel_max_passes=2,
        )
        # a metadata-free legacy object planted under the system key
        self.cache[key] = ReferenceRanges(
            L_ue=1.0, L_mec=2.0, L_helper=3.0, E_ue=4.0, E_mec=5.0, E_helper=6.0
        )
        with self.assertRaises(EnergyReferenceMismatch):
            self._get()

    def test_legacy_graph_fingerprint_matches_its_canonical_view(self):
        from env.mec_offloaing_envs.scheduler.adapter import to_canonical_dag

        canonical = to_canonical_dag(self.tg)
        self.assertEqual(
            scheduling_graph_fingerprint(self.tg, self.order),
            scheduling_graph_fingerprint(canonical, self.order),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
