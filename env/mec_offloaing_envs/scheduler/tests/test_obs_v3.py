#!/usr/bin/env python3
"""⑤a tests: observation v3 (deadline/feasibility awareness) + v1/v2 parity.

The decisive test is `test_dependency_on_deadline`: two states that differ ONLY in
the deadline must produce different observations. With v1/v2 they cannot (no
deadline feature exists) — that is exactly the structural blindness the review
flagged. The embedding-level half of that test needs TF and lives in
`spec/embedding_deadline_probe.py` (run on the GPU host).
"""

from __future__ import annotations

import os
import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

for _name in ("gym", "gym.core"):
    if _name not in sys.modules:
        try:
            __import__(_name)
        except Exception:
            sys.modules[_name] = types.ModuleType(_name)
if not hasattr(sys.modules.get("gym.core", types.ModuleType("gym.core")), "Env"):
    sys.modules.setdefault("gym", types.ModuleType("gym"))
    sys.modules.setdefault("gym.core", types.ModuleType("gym.core"))
    sys.modules["gym.core"].Env = type("Env", (), {})

import numpy as np  # noqa: E402

from env.mec_offloaing_envs.scheduler import encoder_obs as eo  # noqa: E402
from env.mec_offloaing_envs.scheduler.model import (  # noqa: E402
    CanonicalDAG,
    CanonicalTask,
)
from env.mec_offloaing_envs.scheduler.resources import ResourceConfig  # noqa: E402
from env.mec_offloaing_envs.scheduler.static_bounds import (  # noqa: E402
    static_action_bounds,
)

WORKLOAD = 4_166_700
BIG_OUTPUT = 4_166_700
SMALL = 2048


def _resources(model="physical_v1"):
    return ResourceConfig.from_frozen_yaml(model=model)


def _dag(deadline=None, dtype="hard", weight=1.0, cclass="medium", n=3):
    tasks = []
    for i in range(n):
        tasks.append(
            CanonicalTask(
                task_id=i,
                compute_workload_bytes=WORKLOAD if i == 0 else SMALL,
                task_output_bytes=BIG_OUTPUT if i == 0 else SMALL,
                deadline_s=deadline if i == 0 else None,
                deadline_type=dtype if (i == 0 and deadline is not None) else "none",
                tardiness_weight=weight,
                criticality_class=cclass,
            )
        )
    edges = [(i, i + 1, BIG_OUTPUT if i == 0 else SMALL) for i in range(n - 1)]
    return CanonicalDAG.from_records(tasks, edges)


class _VersionGuard(unittest.TestCase):
    def setUp(self):
        self._saved = eo.OBS_VERSION

    def tearDown(self):
        eo.set_obs_version(self._saved)


class TestObsV3Schema(_VersionGuard):
    def test_dims_and_names(self):
        eo.set_obs_version("v3")
        self.assertEqual(eo.FEATURE_DIM, 31)
        self.assertEqual(eo.PACKED_DIM, 31 + 2 * 19 + 1)
        self.assertEqual(
            eo.FEATURE_NAMES_V3,
            eo.FEATURE_NAMES_V2 + eo.DEADLINE_FEATURE_NAMES,
        )
        # bounded features are not standardized
        for name in eo.DEADLINE_FEATURE_NAMES:
            self.assertNotIn(name, eo.STANDARDIZE_FEATURES)

    def test_v1_and_v2_are_untouched(self):
        eo.set_obs_version("v1")
        self.assertEqual((eo.FEATURE_DIM, eo.PACKED_DIM), (11, 50))
        eo.set_obs_version("v2")
        self.assertEqual((eo.FEATURE_DIM, eo.PACKED_DIM), (15, 54))

    def test_unknown_version_rejected(self):
        with self.assertRaises(eo.EncoderGraphError):
            eo.set_obs_version("v9")


class TestObsV3Build(_VersionGuard):
    def setUp(self):
        super().setUp()
        eo.set_obs_version("v3")
        self.res = _resources()
        self.dag = _dag(deadline=5.0)
        self.order = [0, 1, 2]

    def _obs(self, dag=None):
        dag = dag or self.dag
        return eo.encode_canonical_dag(
            dag, self.order, resources=self.res, cycles_per_bit=300.0
        )

    def test_shape_and_finiteness(self):
        obs = self._obs()
        self.assertEqual(obs.shape, (3, eo.PACKED_DIM))
        self.assertTrue(np.all(np.isfinite(obs)))

    def test_flags_are_binary_and_one_hot(self):
        obs = self._obs()
        idx = {n: i for i, n in enumerate(eo.FEATURE_NAMES)}
        row = obs[0]
        self.assertEqual(row[idx["has_deadline"]], 1.0)
        self.assertEqual(row[idx["deadline_is_hard"]], 1.0)
        self.assertEqual(row[idx["deadline_is_soft"]], 0.0)
        self.assertEqual(
            row[idx["criticality_low"]]
            + row[idx["criticality_medium"]]
            + row[idx["criticality_high"]],
            1.0,
        )
        for action in ("ue", "mec", "helper"):
            self.assertIn(row[idx["feasible_%s" % action]], (0.0, 1.0))

    def test_no_deadline_means_zero_deadline_block(self):
        obs = self._obs(_dag(deadline=None))
        idx = {n: i for i, n in enumerate(eo.FEATURE_NAMES)}
        row = obs[0]
        self.assertEqual(row[idx["has_deadline"]], 0.0)
        self.assertEqual(row[idx["slack_ratio_min_lb"]], 0.0)
        self.assertEqual(
            row[idx["deadline_is_hard"]] + row[idx["deadline_is_soft"]]
            + row[idx["deadline_is_firm"]],
            0.0,
        )
        # feasibility stays open without a deadline
        for action in ("ue", "mec", "helper"):
            self.assertEqual(row[idx["feasible_%s" % action]], 1.0)

    def test_requires_resources(self):
        with self.assertRaises(eo.EncoderGraphError):
            eo.encode_canonical_dag(self.dag, self.order)   # no resources


class TestDeadlineDependency(_VersionGuard):
    """The decisive ⑤a test: observation must depend on the deadline."""

    def setUp(self):
        super().setUp()
        self.res = _resources()
        self.order = [0, 1, 2]
        self.loose = _dag(deadline=1e6)
        self.tight = _dag(deadline=1e-3)

    def test_v1_and_v2_cannot_see_the_deadline(self):
        """Documents the blocker: without v3 the deadline is invisible."""
        for version in ("v1", "v2"):
            eo.set_obs_version(version)
            kwargs = {} if version == "v1" else {"resource_vec": [1.0, 1.0, 1.0, 1.0]}
            a = eo.encode_canonical_dag(self.loose, self.order, **kwargs)
            b = eo.encode_canonical_dag(self.tight, self.order, **kwargs)
            np.testing.assert_allclose(a, b, rtol=0, atol=0)

    def test_v3_observations_differ(self):
        eo.set_obs_version("v3")
        a = eo.encode_canonical_dag(self.loose, self.order, resources=self.res)
        b = eo.encode_canonical_dag(self.tight, self.order, resources=self.res)
        self.assertFalse(np.allclose(a, b))
        idx = {n: i for i, n in enumerate(eo.FEATURE_NAMES)}
        self.assertGreater(a[0][idx["slack_ratio_min_lb"]], b[0][idx["slack_ratio_min_lb"]])
        # the tight deadline closes at least one action
        feasible_tight = [b[0][idx["feasible_%s" % a_]] for a_ in ("ue", "mec", "helper")]
        self.assertIn(0.0, feasible_tight)

    def test_slack_ratio_monotone_in_deadline(self):
        eo.set_obs_version("v3")
        idx = {n: i for i, n in enumerate(eo.FEATURE_NAMES)}
        values = []
        for deadline in (1.0, 5.0, 50.0, 1e6):
            obs = eo.encode_canonical_dag(
                _dag(deadline=deadline), self.order, resources=self.res
            )
            values.append(obs[0][idx["slack_ratio_min_lb"]])
        self.assertEqual(values, sorted(values))
        self.assertGreaterEqual(min(values), -1.0)
        self.assertLessEqual(max(values), 1.0)


class TestStaticBounds(_VersionGuard):
    def setUp(self):
        super().setUp()
        self.res = _resources()
        self.order = [0, 1, 2]

    def test_bounds_shape_and_ordering(self):
        dag = _dag()
        b = static_action_bounds(dag, self.order, self.res, cycles_per_bit=300.0)
        self.assertEqual(b.n, 3)
        for i in range(b.n):
            for a in range(3):
                self.assertGreaterEqual(b.finish_lb[i][a], 0.0)
                self.assertGreaterEqual(b.ready_lb[i][a], b.finish_lb[i][a] - 1e-9)
            self.assertAlmostEqual(b.min_ready_lb[i], min(b.ready_lb[i]), places=12)

    def test_faster_tier_has_smaller_bound(self):
        dag = _dag()
        b = static_action_bounds(dag, self.order, self.res, cycles_per_bit=300.0)
        # task 0 is heavy: MEC must be far ahead of UE
        self.assertLess(b.finish_lb[0][1], b.finish_lb[0][0])

    def test_sink_pays_return_hop_only_for_remote_tiers(self):
        dag = _dag()
        b = static_action_bounds(dag, self.order, self.res, cycles_per_bit=300.0)
        sink_pos = b.n - 1
        self.assertTrue(b.is_sink[sink_pos])
        self.assertAlmostEqual(b.ready_lb[sink_pos][0], b.finish_lb[sink_pos][0], places=9)
        self.assertGreater(b.ready_lb[sink_pos][1], b.finish_lb[sink_pos][1])

    def test_bounds_are_admissible_against_real_schedules(self):
        """Static LB must not exceed a real schedule's availability."""
        import itertools

        from env.mec_offloaing_envs.scheduler import schedule

        dag = _dag()
        b = static_action_bounds(dag, self.order, self.res, cycles_per_bit=300.0)
        for actions in itertools.product((0, 1, 2), repeat=3):
            out = schedule(dag, self.order, list(actions), self.res)
            for pos, tid in enumerate(self.order):
                achieved = out.tasks[tid].availability_seconds
                self.assertLessEqual(
                    b.min_ready_lb[pos], achieved + 1e-9 * max(1.0, abs(achieved)),
                    msg="static LB above achieved for %s" % (actions,),
                )

    def test_feasibility_matches_the_runtime_mask_semantics(self):
        """obs feasibility flag == static LB <= deadline (same relaxation)."""
        dag = _dag(deadline=5.0)
        b = static_action_bounds(dag, self.order, self.res, cycles_per_bit=300.0)
        feas = b.feasible_by_deadline([5.0, None, None])
        for a in range(3):
            self.assertEqual(feas[0][a], b.ready_lb[0][a] <= 5.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
