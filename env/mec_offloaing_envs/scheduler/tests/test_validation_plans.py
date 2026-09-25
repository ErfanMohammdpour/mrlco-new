#!/usr/bin/env python3
"""E4.2: validation_per_graph_plans comes from the rollout results, no replay."""

from __future__ import annotations

import json
import sys
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

import numpy as np  # noqa: E402

from env.mec_offloaing_envs.offloading_env import OffloadingEnvironment  # noqa: E402
from env.mec_offloaing_envs.scheduler.energy_api import (  # noqa: E402
    compute_reference_ranges,
    compute_scoped_reference_ranges,
)
from env.mec_offloaing_envs.scheduler.energy_scope import (  # noqa: E402
    SCOPE_MOBILE,
    SCOPE_SYSTEM,
    EnergyReferenceMismatch,
)
from env.mec_offloaing_envs.scheduler.objective import (  # noqa: E402
    LATENCY_REF_ALL_UE,
    ObjectiveSpec,
)
from env.mec_offloaing_envs.scheduler.primary_config import (  # noqa: E402
    resolved_primary_scheduler_config,
)
from env.mec_offloaing_envs.scheduler.reward import (  # noqa: E402
    REWARD_MODE_LATENCY_ONLY,
    REWARD_MODE_PUBLICATION,
)
from env.mec_offloaing_envs.scheduler.resources import (  # noqa: E402
    resolved_config_sha256,
)


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


class _FakeCluster:
    use_energy = True
    energy_config = {"energy_telemetry": False}

    def __init__(self, reward_mode):
        self.reward_mode = reward_mode

    def reset(self):
        pass


class _FakeEnvSelf:
    """Minimal `self` for the unbound env methods (no data files)."""

    _validation_plan_record = OffloadingEnvironment._validation_plan_record
    validation_plan_payload = OffloadingEnvironment.validation_plan_payload
    validation_plan_identities = OffloadingEnvironment.validation_plan_identities

    def __init__(self, res, reward_mode, validation_plans=True):
        self.resource_cluster = _FakeCluster(reward_mode)
        self.scheduler_resources = res
        self.constraint_controller = None
        self.constraint_spec = None
        self.shaping_discount = 1.0
        self.energy_telemetry_enabled = False
        self.last_energy_telemetry = None
        self.validation_plans_enabled = validation_plans
        self.last_validation_plans = None
        self._mobile = None
        self._system = None

    def get_reference_ranges(self, task_graph, scope=None):
        if scope == SCOPE_MOBILE:
            return self._mobile
        return self._system


def _make(reward_mode, validation_plans=True):
    tg = _FakeTG()
    res = resolved_primary_scheduler_config()
    fake = _FakeEnvSelf(res, reward_mode, validation_plans)
    fake._mobile = compute_reference_ranges(tg, res)
    fake._system = compute_scoped_reference_ranges(tg, res, energy_scope=SCOPE_SYSTEM)
    return tg, res, fake


class TestValidationPlanPayload(unittest.TestCase):
    def test_payload_uses_the_rollout_results_and_is_objective_ready(self):
        from spec.objective_selection import objective_from_plans

        tg, res, fake = _make(REWARD_MODE_LATENCY_ONLY)
        plan = [(0, 0), (1, 2), (2, 1)]
        OffloadingEnvironment.get_reward_batch_step_by_step(
            fake, [plan], [tg], None, None
        )
        payload = fake.validation_plan_payload()
        self.assertEqual(len(payload), 1)
        result, refs = payload[0]
        self.assertEqual(refs.energy_scope, SCOPE_SYSTEM)
        self.assertEqual(result.scheduler_config_sha256, resolved_config_sha256(res))
        spec = ObjectiveSpec(latency_ref=LATENCY_REF_ALL_UE, energy_budget_j=1e12)
        agg = objective_from_plans(payload, spec)   # not None, no "unavailable"
        self.assertEqual(agg.n_graphs, 1)
        self.assertTrue(np.isfinite(agg.J))

    def test_identities_are_serializable_and_carry_graph_identity(self):
        tg, res, fake = _make(REWARD_MODE_LATENCY_ONLY)
        plan = [(0, 0), (1, 2), (2, 1)]
        OffloadingEnvironment.get_reward_batch_step_by_step(
            fake, [plan], [tg], None, None
        )
        identities = fake.validation_plan_identities()
        self.assertEqual(len(identities), 1)
        record = identities[0]
        self.assertEqual(record["order"], [0, 1, 2])
        self.assertEqual(record["plan"], [[0, 0], [1, 2], [2, 1]])
        self.assertEqual(record["energy_scope"], SCOPE_SYSTEM)
        self.assertEqual(len(record["scheduler_config_sha256"]), 64)
        json.dumps(identities)  # audit-safe: no ScheduleResult objects

    def test_disabled_channel_produces_nothing(self):
        tg, res, fake = _make(REWARD_MODE_LATENCY_ONLY, validation_plans=False)
        plan = [(0, 0), (1, 2), (2, 1)]
        OffloadingEnvironment.get_reward_batch_step_by_step(
            fake, [plan], [tg], None, None
        )
        self.assertIsNone(fake.last_validation_plans)
        self.assertIsNone(fake.validation_plan_payload())

    def test_mobile_reference_is_rejected_fail_loud(self):
        # legacy publication builds MOBILE refs; the objective channel requires
        # SYSTEM, so enabling both must raise rather than emit a wrong plan.
        tg, res, fake = _make(REWARD_MODE_PUBLICATION)
        plan = [(0, 0), (1, 2), (2, 1)]
        with self.assertRaises(EnergyReferenceMismatch):
            OffloadingEnvironment.get_reward_batch_step_by_step(
                fake, [plan], [tg], None, None
            )

    def test_no_extra_schedule_when_the_channel_is_enabled(self):
        import env.mec_offloaing_envs.scheduler.reward as reward_module

        tg, res, fake = _make(REWARD_MODE_LATENCY_ONLY)
        plan = [(0, 0), (1, 2), (2, 1)]
        original = reward_module.schedule_via_adapter
        calls = {"n": 0}

        def counting(*args, **kwargs):
            calls["n"] += 1
            return original(*args, **kwargs)

        with mock.patch.object(reward_module, "schedule_via_adapter", counting):
            OffloadingEnvironment.get_reward_batch_step_by_step(
                fake, [plan], [tg], None, None
            )
        with_plans = calls["n"]

        _tg, _res, off = _make(REWARD_MODE_LATENCY_ONLY, validation_plans=False)
        calls["n"] = 0
        with mock.patch.object(reward_module, "schedule_via_adapter", counting):
            OffloadingEnvironment.get_reward_batch_step_by_step(
                off, [plan], [tg], None, None
            )
        without_plans = calls["n"]
        self.assertEqual(with_plans, without_plans)
        self.assertGreater(with_plans, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
