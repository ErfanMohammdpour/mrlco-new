#!/usr/bin/env python3
"""E3.1: the primary reward is latency-only; publication stays an explicit opt-in."""

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

import numpy as np  # noqa: E402

from env.mec_offloaing_envs.offloading_env import OffloadingEnvironment  # noqa: E402
from env.mec_offloaing_envs.scheduler.adapter import schedule_via_adapter  # noqa: E402
from env.mec_offloaing_envs.scheduler.energy_api import (  # noqa: E402
    ReferenceRanges,
    compute_reference_ranges,
)
from env.mec_offloaing_envs.scheduler.energy_scope import (  # noqa: E402
    SCOPE_MOBILE,
    energy_scalar,
)
from env.mec_offloaing_envs.scheduler.primary_config import (  # noqa: E402
    resolved_primary_scheduler_config,
)
from env.mec_offloaing_envs.scheduler.resources import ResourceConfig  # noqa: E402
from env.mec_offloaing_envs.scheduler.reward import (  # noqa: E402
    REWARD_MODE_LATENCY_ONLY,
    REWARD_MODE_PUBLICATION,
    expected_episode_return,
    telescoping_token_rewards,
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


def _setup():
    tg = _FakeTG()
    res = resolved_primary_scheduler_config()
    plan = [(0, 0), (1, 2), (2, 1)]
    refs = compute_reference_ranges(tg, res)
    return tg, res, plan, refs


def _scaled_energy_refs(refs, factor):
    return ReferenceRanges(
        L_ue=refs.L_ue, L_mec=refs.L_mec, L_helper=refs.L_helper,
        E_ue=refs.E_ue * factor, E_mec=refs.E_mec * factor,
        E_helper=refs.E_helper * factor,
    )


class TestLatencyOnlyPrimary(unittest.TestCase):
    def test_primary_reward_is_energy_free(self):
        tg, res, plan, refs = _setup()
        out = telescoping_token_rewards(
            tg, plan, res, reward_mode=REWARD_MODE_LATENCY_ONLY, refs=refs,
            include_energy=True,  # ignored: latency-only never reads energy
        )
        huge = _scaled_energy_refs(refs, 1000.0)
        out_huge = telescoping_token_rewards(
            tg, plan, res, reward_mode=REWARD_MODE_LATENCY_ONLY, refs=huge
        )
        np.testing.assert_allclose(out.rewards, out_huge.rewards, rtol=0, atol=1e-12)
        # and the schedule traces are identical, so no hidden energy coupling
        np.testing.assert_allclose(out.makespans, out_huge.makespans, rtol=0, atol=1e-12)

    def test_primary_is_independent_of_the_energy_model(self):
        tg, res, plan, refs = _setup()
        legacy = ResourceConfig.from_frozen_yaml()
        physical = telescoping_token_rewards(
            tg, plan, res, reward_mode=REWARD_MODE_LATENCY_ONLY, refs=refs
        )
        legacy_out = telescoping_token_rewards(
            tg, plan, legacy, reward_mode=REWARD_MODE_LATENCY_ONLY,
            refs=compute_reference_ranges(tg, legacy),
        )
        np.testing.assert_allclose(physical.rewards, legacy_out.rewards, rtol=0, atol=1e-12)

    def test_matches_the_closed_form_latency_potential(self):
        tg, res, plan, refs = _setup()
        out = telescoping_token_rewards(tg, plan, res, reward_mode=REWARD_MODE_LATENCY_ONLY, refs=refs)
        expected = expected_episode_return(
            out.makespans, out.energies, refs,
            include_energy=False, latency_weight=1.0, energy_weight=0.0,
            reward_mode=REWARD_MODE_LATENCY_ONLY,
        )
        self.assertAlmostEqual(float(sum(out.rewards)), expected, places=9)

    def test_unknown_mode_is_loud(self):
        tg, res, plan, refs = _setup()
        with self.assertRaises(ValueError):
            telescoping_token_rewards(tg, plan, res, reward_mode="solar", refs=refs)


class TestLegacyPublication(unittest.TestCase):
    def test_opt_in_keeps_the_mobile_energy_term(self):
        tg, res, plan, refs = _setup()
        explicit = telescoping_token_rewards(
            tg, plan, res, reward_mode=REWARD_MODE_PUBLICATION, refs=refs
        )
        legacy_default = telescoping_token_rewards(tg, plan, res, refs=refs)
        np.testing.assert_allclose(explicit.rewards, legacy_default.rewards, rtol=0, atol=0)
        np.testing.assert_allclose(explicit.makespans, legacy_default.makespans, rtol=0, atol=0)
        np.testing.assert_allclose(explicit.energies, legacy_default.energies, rtol=0, atol=0)

    def test_publication_mobile_energy_is_preserved(self):
        tg, res, plan, refs = _setup()
        out = telescoping_token_rewards(
            tg, plan, res, reward_mode=REWARD_MODE_PUBLICATION, refs=refs
        )
        self.assertAlmostEqual(
            out.energies[-1],
            energy_scalar(out.final_result, scope=SCOPE_MOBILE),
            places=12,
        )

    def test_publication_and_latency_only_differ_when_energy_differs(self):
        tg, res, plan, refs = _setup()
        pub = telescoping_token_rewards(tg, plan, res, reward_mode=REWARD_MODE_PUBLICATION, refs=refs)
        lat = telescoping_token_rewards(tg, plan, res, reward_mode=REWARD_MODE_LATENCY_ONLY, refs=refs)
        self.assertFalse(np.allclose(pub.rewards, lat.rewards, rtol=1e-9, atol=1e-12))


class _FakeCluster:
    use_energy = True
    energy_config = {"energy_telemetry": True}

    def __init__(self, reward_mode):
        self.reward_mode = reward_mode

    def reset(self):
        pass


class _FakeEnvSelf:
    def __init__(self, res, refs, reward_mode):
        self.resource_cluster = _FakeCluster(reward_mode)
        self.scheduler_resources = res
        self.constraint_controller = None
        self.constraint_spec = None
        self.shaping_discount = 1.0
        self.energy_telemetry_enabled = True
        self.last_energy_telemetry = None
        self._refs = refs

    def get_reference_ranges(self, task_graph):
        return self._refs


class TestEnvPrimaryPath(unittest.TestCase):
    def test_env_defaults_to_latency_only_and_still_logs_energy(self):
        tg, res, plan, refs = _setup()
        fake = _FakeEnvSelf(res, refs, REWARD_MODE_LATENCY_ONLY)
        rewards, finish, energy = OffloadingEnvironment.get_reward_batch_step_by_step(
            fake, [plan], [tg], None, None
        )
        direct = telescoping_token_rewards(
            tg, plan, res, reward_mode=REWARD_MODE_LATENCY_ONLY, refs=refs
        )
        np.testing.assert_allclose(np.asarray(rewards[0], dtype=float), direct.rewards, rtol=0, atol=1e-12)
        # logging is decoupled from the reward term: per-task mobile energy and
        # telemetry are still produced under latency-only
        self.assertEqual(len(energy), 1)
        self.assertEqual(len(energy[0]), len(plan))
        self.assertIsNotNone(fake.last_energy_telemetry)
        self.assertEqual(fake.last_energy_telemetry[0]["primary_scope"], "system")

    def test_env_publication_opt_in_matches_the_legacy_reward(self):
        tg, res, plan, refs = _setup()
        fake = _FakeEnvSelf(res, refs, REWARD_MODE_PUBLICATION)
        rewards, _finish, _energy = OffloadingEnvironment.get_reward_batch_step_by_step(
            fake, [plan], [tg], None, None
        )
        direct = telescoping_token_rewards(tg, plan, res, refs=refs)
        np.testing.assert_allclose(np.asarray(rewards[0], dtype=float), direct.rewards, rtol=0, atol=0)

    def test_env_rejects_an_unknown_reward_mode(self):
        tg, res, plan, refs = _setup()
        fake = _FakeEnvSelf(res, refs, "solar")
        with self.assertRaises(ValueError):
            OffloadingEnvironment.get_reward_batch_step_by_step(
                fake, [plan], [tg], None, None
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
