#!/usr/bin/env python3
"""Cache identity for energy references: stable, content-based, parameter-sensitive."""

from __future__ import annotations

import copy
import pickle
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

from env.mec_offloaing_envs.scheduler.energy_cache import (  # noqa: E402
    primary_scope_of,
    reference_ranges_cache_key,
    scheduling_graph_fingerprint,
)
from env.mec_offloaing_envs.scheduler.energy_scope import (  # noqa: E402
    SCOPE_MOBILE,
    SCOPE_REQUESTER,
    SCOPE_SYSTEM,
)
from env.mec_offloaing_envs.scheduler.model import (  # noqa: E402
    CanonicalDAG,
    CanonicalTask,
)
from env.mec_offloaing_envs.scheduler.primary_config import (  # noqa: E402
    resolved_primary_scheduler_config,
)


def dag(workloads=(1000, 2000), edges=((0, 1, 100),), cpb=(None, None), deadline=(None, None)):
    tasks = [
        CanonicalTask(
            task_id=i,
            compute_workload_bytes=workloads[i],
            task_output_bytes=50,
            external_input_bytes=10 if i == 0 else 0,
            cycles_per_bit=cpb[i],
            deadline_s=deadline[i],
            deadline_type="hard" if deadline[i] is not None else "none",
        )
        for i in range(len(workloads))
    ]
    return CanonicalDAG.from_records(tasks, list(edges))


def key(graph, order=(0, 1), **overrides):
    kwargs = dict(
        reference_mode="candidate_panel",
        energy_scope=SCOPE_MOBILE,
        resources=resolved_primary_scheduler_config(),
        panel_max_passes=2,
        panel_extra={"greedy": True},
    )
    kwargs.update(overrides)
    return reference_ranges_cache_key(graph=graph, order=order, **kwargs)


class TestGraphFingerprint(unittest.TestCase):
    def test_equal_content_equal_hash(self):
        self.assertEqual(
            scheduling_graph_fingerprint(dag(), (0, 1)),
            scheduling_graph_fingerprint(dag(), (0, 1)),
        )

    def test_content_changes_the_hash(self):
        base = scheduling_graph_fingerprint(dag(), (0, 1))
        self.assertNotEqual(base, scheduling_graph_fingerprint(dag(workloads=(1000, 2001)), (0, 1)))
        self.assertNotEqual(base, scheduling_graph_fingerprint(dag(edges=((0, 1, 101),)), (0, 1)))
        self.assertNotEqual(base, scheduling_graph_fingerprint(dag(cpb=(300.0, None)), (0, 1)))

    def test_order_changes_the_hash(self):
        self.assertNotEqual(
            scheduling_graph_fingerprint(dag(), (0, 1)),
            scheduling_graph_fingerprint(dag(), (1, 0)),
        )

    def test_deadlines_are_not_part_of_the_key(self):
        # references are built from pure plans; deadlines do not change their
        # makespan or energy, so they must not fragment the cache
        self.assertEqual(
            scheduling_graph_fingerprint(dag(), (0, 1)),
            scheduling_graph_fingerprint(dag(deadline=(5.0, 6.0)), (0, 1)),
        )

    def test_bad_inputs_are_loud(self):
        with self.assertRaises(ValueError):
            scheduling_graph_fingerprint(object(), (0, 1))
        with self.assertRaises(ValueError):
            scheduling_graph_fingerprint(dag(), (0, 0))


class TestCacheKey(unittest.TestCase):
    def test_deterministic(self):
        self.assertEqual(key(dag()), key(dag()))

    def test_every_parameter_changes_the_key(self):
        base = key(dag())
        for overrides in (
            {"energy_scope": SCOPE_SYSTEM},
            {"reference_mode": "pure"},
            {"panel_max_passes": 3},
            {"panel_extra": {"greedy": False}},
        ):
            self.assertNotEqual(base, key(dag(), **overrides), overrides)
        self.assertNotEqual(base, key(dag(workloads=(1000, 2001))))
        self.assertNotEqual(base, key(dag(), order=(1, 0)))

    def test_scheduler_fingerprint_changes_the_key(self):
        legacy = resolved_primary_scheduler_config(overrides={"timing_model": "legacy_frozen_rates",
                                                              "radio_timing_model": "legacy_frozen_rates",
                                                              "energy_model": "physical_v1",
                                                              "radio_model": "physical_v1",
                                                              "energy_scope": "system"})
        other = resolved_primary_scheduler_config(overrides={"timing_model": "physical_rates"})
        self.assertNotEqual(
            key(dag(), resources=legacy), key(dag(), resources=other)
        )

    def test_pickle_and_deepcopy_of_the_config_keep_the_key(self):
        cfg = resolved_primary_scheduler_config()
        self.assertEqual(
            key(dag(), resources=cfg),
            key(dag(), resources=pickle.loads(pickle.dumps(cfg))),
        )
        self.assertEqual(key(dag(), resources=cfg), key(dag(), resources=copy.deepcopy(cfg)))

    def test_invalid_scope_and_mode_are_loud(self):
        with self.assertRaises(ValueError):
            key(dag(), energy_scope="solar")
        with self.assertRaises(ValueError):
            key(dag(), energy_scope="")
        with self.assertRaises(ValueError):
            key(dag(), reference_mode="")


class TestPrimaryScope(unittest.TestCase):
    def test_declared_scope_is_returned(self):
        self.assertEqual(primary_scope_of(resolved_primary_scheduler_config()), SCOPE_SYSTEM)

    def test_missing_scope_is_loud(self):
        class Res:
            energy_scope = ""

        with self.assertRaises(ValueError):
            primary_scope_of(Res())


if __name__ == "__main__":
    unittest.main(verbosity=2)
