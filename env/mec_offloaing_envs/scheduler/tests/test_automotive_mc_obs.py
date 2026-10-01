#!/usr/bin/env python3
"""`automotive_mc_obs_v1` tests: append-only MC extension of obs v3.

numpy/stdlib only on purpose: the local interpreter has no TensorFlow, so this
file must never import it. The decisive regressions here are
  * the v3 deadline write used to be `rows[:, start:]`, an open-ended slice that
    on a 40-wide row either broadcast-errors or clobbers the appended MC columns;
    `test_deadline_block_does_not_overwrite_the_mc_tail` pins the bounded write.
  * a task with C_HI N/A must stay distinguishable from a task with C_HI == 0.
"""

from __future__ import annotations

import copy
import json
import os
import subprocess
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
from env.mec_offloaing_envs.scheduler.adapter import to_canonical_dag  # noqa: E402
from env.mec_offloaing_envs.scheduler.model import (  # noqa: E402
    CanonicalDAG,
    CanonicalTask,
)
from env.mec_offloaing_envs.scheduler.resources import ResourceConfig  # noqa: E402

VERSION = "automotive_mc_obs_v1"
STATS_FILE = ROOT / "spec/encoder_feature_stats_automotive_mc_v1.json"
STATS_TOOL = ROOT / "spec/make_automotive_mc_obs_v1_stats.py"

WORKLOAD = 4_166_700
SMALL = 2048
BIG_OUTPUT = 4_166_700
CYC = 300.0

# MC columns that are functions of the C_LO/C_HI/drop/degrade/mode context only,
# i.e. the "criticality-derived" half of the block.
MC_CRITICALITY_COLUMNS = (
    "mc_c_lo_scaled",
    "mc_c_hi_scaled",
    "mc_has_c_hi",
    "mc_c_hi_over_lo",
    "mc_drop_allowed",
    "mc_degrade_allowed",
    "mc_mode_is_hi",
)


def _resources(model="physical_v1"):
    return ResourceConfig.from_frozen_yaml(model=model)


def _tasks(n, deadline=None, dtype="hard"):
    out = []
    for i in range(n):
        out.append(
            CanonicalTask(
                task_id=i,
                compute_workload_bytes=WORKLOAD if i == 0 else SMALL,
                task_output_bytes=BIG_OUTPUT if i == 0 else SMALL,
                deadline_s=deadline if i == 0 else None,
                deadline_type=dtype if (i == 0 and deadline is not None) else "none",
            )
        )
    return out


def _chain(n=20, deadline=None):
    return CanonicalDAG.from_records(
        _tasks(n, deadline=deadline),
        [(i, i + 1, BIG_OUTPUT if i == 0 else SMALL) for i in range(n - 1)],
    )


def _branching(n=20, deadline=None):
    """One root fanning out to n-2 children, all feeding one sink."""
    tasks = _tasks(n, deadline=deadline)
    edges = [(0, i, SMALL) for i in range(1, n - 1)]
    edges += [(i, n - 1, SMALL) for i in range(1, n - 1)]
    return CanonicalDAG.from_records(tasks, edges)


def _mc_context(n=3, *, mode_is_hi=True, d_g=1.0, max_c_lo=0.4, has_hi=None):
    if has_hi is None:
        has_hi = [i % 2 == 0 for i in range(n)]
    per_task = {}
    for i in range(n):
        entry = {
            "c_lo_s": 0.1 * (i + 1),
            "c_hi_s": 0.15 * (i + 1) if has_hi[i] else None,
            "has_c_hi": bool(has_hi[i]),
            "c_hi_over_lo": 1.5,
            "drop_allowed": False,
            "degrade_allowed": i == 0,
            "slack_ratio": 0.25 + 0.1 * i,
        }
        per_task[i] = entry
    return {
        "per_task": per_task,
        "mode_is_hi": mode_is_hi,
        "D_G_s": d_g,
        "max_c_lo_s": max_c_lo,
    }


def _index():
    return {name: i for i, name in enumerate(eo.FEATURE_NAMES)}


class _VersionGuard(unittest.TestCase):
    def setUp(self):
        self._saved = eo.OBS_VERSION

    def tearDown(self):
        eo.set_obs_version(self._saved)


class TestAutomotiveMcSchema(_VersionGuard):
    def test_dims_and_names(self):
        eo.set_obs_version(VERSION)
        self.assertEqual(eo.FEATURE_DIM, 40)
        self.assertEqual(eo.PACKED_DIM, 79)
        self.assertEqual(eo.PACKED_DIM, 40 + 2 * eo.MAX_NEIGH + 1)
        self.assertEqual(
            eo.FEATURE_NAMES_AUTOMOTIVE_MC_V1,
            eo.FEATURE_NAMES_V3 + eo.MC_FEATURE_NAMES,
        )
        self.assertEqual(eo.FEATURE_NAMES, eo.FEATURE_NAMES_AUTOMOTIVE_MC_V1)
        self.assertEqual(len(eo.MC_FEATURE_NAMES), 9)

    def test_mc_features_are_non_standardized_identity(self):
        eo.set_obs_version(VERSION)
        for name in eo.MC_FEATURE_NAMES:
            self.assertNotIn(name, eo.STANDARDIZE_FEATURES)
            self.assertIn(name, eo.NON_STANDARDIZED_FEATURES)

    def test_existing_versions_are_unchanged(self):
        eo.set_obs_version("v1")
        self.assertEqual((eo.FEATURE_DIM, eo.PACKED_DIM), (11, 50))
        eo.set_obs_version("v2")
        self.assertEqual((eo.FEATURE_DIM, eo.PACKED_DIM), (15, 54))
        eo.set_obs_version("v3")
        self.assertEqual((eo.FEATURE_DIM, eo.PACKED_DIM), (31, 70))

    def test_unknown_version_still_rejected(self):
        with self.assertRaises(eo.EncoderGraphError):
            eo.set_obs_version("v9")

    def test_feasibility_helpers_accept_the_new_version(self):
        eo.set_obs_version(VERSION)
        idx = eo.feasibility_channel_indices()
        self.assertEqual(
            [eo.FEATURE_NAMES[i] for i in idx],
            ["feasible_ue", "feasible_mec", "feasible_helper"],
        )
        self.assertEqual(eo.FEATURE_NAMES[eo.hard_deadline_channel_index()], "deadline_is_hard")
        # v1/v2 keep raising
        for version in ("v1", "v2"):
            eo.set_obs_version(version)
            with self.assertRaises(eo.EncoderGraphError):
                eo.feasibility_channel_indices()

    def test_column_layout_is_disjoint(self):
        eo.set_obs_version(VERSION)
        names = list(eo.FEATURE_NAMES)
        deadline_start = names.index(eo.DEADLINE_FEATURE_NAMES[0])
        mc_start = names.index(eo.MC_FEATURE_NAMES[0])
        self.assertEqual(deadline_start + len(eo.DEADLINE_FEATURE_NAMES), mc_start)
        self.assertEqual(mc_start + len(eo.MC_FEATURE_NAMES), eo.FEATURE_DIM)


class TestAutomotiveMcBuild(_VersionGuard):
    def setUp(self):
        super().setUp()
        eo.set_obs_version(VERSION)
        self.res = _resources()

    def _encode(self, dag, mc_context, order=None):
        order = order if order is not None else sorted(dag.tasks)
        return eo.encode_canonical_dag(
            dag,
            order,
            resources=self.res,
            cycles_per_bit=CYC,
            mc_context=mc_context,
        )

    def test_chain_packed_shape(self):
        obs = self._encode(_chain(20, deadline=5.0), _mc_context(20))
        self.assertEqual(obs.shape, (20, eo.PACKED_DIM))
        self.assertTrue(np.all(np.isfinite(obs)))

    def test_branching_packed_shape(self):
        obs = self._encode(_branching(20, deadline=5.0), _mc_context(20))
        self.assertEqual(obs.shape, (20, eo.PACKED_DIM))
        self.assertTrue(np.all(np.isfinite(obs)))

    def test_mc_values_are_finite_and_bounded(self):
        obs = self._encode(_chain(20, deadline=5.0), _mc_context(20))
        for name in eo.MC_FEATURE_NAMES:
            col = obs[:, _index()[name]]
            hi = 4.0 if name == "mc_c_hi_over_lo" else 1.0
            self.assertTrue(np.all(np.isfinite(col)), name)
            self.assertGreaterEqual(float(col.min()), 0.0, name)
            self.assertLessEqual(float(col.max()), hi, name)

    def test_no_mc_context_means_zero_mc_columns(self):
        obs = self._encode(_chain(3, deadline=5.0), None)
        idx = _index()
        for name in eo.MC_FEATURE_NAMES:
            np.testing.assert_array_equal(obs[:, idx[name]], np.zeros(3))
        self.assertTrue(np.all(np.isfinite(obs)))

    def test_determinism_byte_identical(self):
        dag = _chain(20, deadline=5.0)
        ctx = _mc_context(20)
        a = self._encode(dag, ctx)
        b = self._encode(dag, ctx)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(a.tobytes(), b.tobytes())

    def test_deadline_block_does_not_overwrite_the_mc_tail(self):
        """Regression for `rows[:, name_index[DEADLINE[0]]:] = ...`.

        The old open-ended slice covered the 9 appended MC columns (and on a
        40-wide row it broadcast-errors outright). The MC context below is
        chosen so every MC write is non-zero, so a clobber is visible.
        """
        ctx = _mc_context(3, mode_is_hi=True, d_g=1.0, max_c_lo=0.4)
        obs = self._encode(_chain(3, deadline=5.0), ctx)
        idx = _index()
        row = obs[0]
        self.assertAlmostEqual(row[idx["mc_c_lo_scaled"]], 0.25)   # 0.1 / 0.4
        self.assertAlmostEqual(row[idx["mc_c_hi_scaled"]], 0.375)  # 0.15 / 0.4
        self.assertAlmostEqual(row[idx["mc_has_c_hi"]], 1.0)
        self.assertAlmostEqual(row[idx["mc_c_hi_over_lo"]], 1.5)
        self.assertAlmostEqual(row[idx["mc_degrade_allowed"]], 1.0)
        self.assertAlmostEqual(row[idx["mc_mode_is_hi"]], 1.0)
        self.assertAlmostEqual(row[idx["mc_slack_ratio"]], 0.25)
        self.assertAlmostEqual(row[idx["mc_deadline_over_D_G"]], 1.0)  # 5.0/1.0 clipped
        # the fix is structural: the deadline write must stop at the MC block
        mc_start = idx["mc_c_lo_scaled"]
        self.assertEqual(mc_start, idx[eo.DEADLINE_FEATURE_NAMES[0]] + 16)
        self.assertTrue(np.any(obs[:, mc_start:] != 0.0))

    def test_mc_context_is_never_nan(self):
        ctx = _mc_context(3)
        ctx["per_task"][1] = {
            "c_lo_s": float("nan"),
            "c_hi_s": None,
            "has_c_hi": False,
            "c_hi_over_lo": float("inf"),
            "drop_allowed": None,
            "degrade_allowed": 0,
            "slack_ratio": float("nan"),
        }
        ctx["D_G_s"] = float("nan")
        ctx["max_c_lo_s"] = 0.0
        obs = self._encode(_chain(3, deadline=5.0), ctx)
        self.assertTrue(np.all(np.isfinite(obs)))


class TestC_Hi_NotApplicable(_VersionGuard):
    def setUp(self):
        super().setUp()
        eo.set_obs_version(VERSION)
        self.res = _resources()
        self.dag = _chain(3, deadline=5.0)

    def _col(self, context, name):
        obs = eo.encode_canonical_dag(
            self.dag, [0, 1, 2], resources=self.res, cycles_per_bit=CYC,
            mc_context=context,
        )
        return obs[0, _index()[name]]

    def test_na_is_distinguishable_from_zero_c_hi(self):
        base = _mc_context(3, has_hi=[False, False, False])
        zero_hi = copy.deepcopy(base)
        zero_hi["per_task"][0] = {
            "c_lo_s": 0.1,
            "c_hi_s": 0.0,
            "has_c_hi": True,
            "c_hi_over_lo": 0.0,
            "drop_allowed": False,
            "degrade_allowed": False,
            "slack_ratio": 0.0,
        }
        self.assertEqual(self._col(base, "mc_has_c_hi"), 0.0)
        self.assertEqual(self._col(base, "mc_c_hi_scaled"), 0.0)
        self.assertEqual(self._col(zero_hi, "mc_c_hi_scaled"), 0.0)
        self.assertEqual(self._col(zero_hi, "mc_has_c_hi"), 1.0)


class TestColumnIndependence(_VersionGuard):
    def setUp(self):
        super().setUp()
        eo.set_obs_version(VERSION)
        self.res = _resources()
        idx = _index()
        self.idx = idx
        self.dl_start = idx[eo.DEADLINE_FEATURE_NAMES[0]]
        self.dl_end = self.dl_start + len(eo.DEADLINE_FEATURE_NAMES)

    def _encode(self, dag, ctx):
        return eo.encode_canonical_dag(
            dag, sorted(dag.tasks), resources=self.res, cycles_per_bit=CYC,
            mc_context=ctx,
        )

    def test_deadline_change_does_not_move_criticality_columns(self):
        ctx = _mc_context(3)
        loose = self._encode(_chain(3, deadline=1e6), ctx)
        tight = self._encode(_chain(3, deadline=1e-3), ctx)
        for name in MC_CRITICALITY_COLUMNS:
            np.testing.assert_array_equal(
                loose[:, self.idx[name]], tight[:, self.idx[name]],
                err_msg="deadline moved MC column %s" % name,
            )
        # ...and the deadline channels really did move (test is not vacuous)
        self.assertFalse(
            np.array_equal(
                loose[:, self.dl_start:self.dl_end], tight[:, self.dl_start:self.dl_end]
            )
        )

    def test_mc_change_does_not_move_deadline_columns(self):
        dag = _chain(3, deadline=5.0)
        base = self._encode(dag, _mc_context(3))
        other_ctx = copy.deepcopy(_mc_context(3))
        for i in range(3):
            other_ctx["per_task"][i]["c_lo_s"] = 0.05
            other_ctx["per_task"][i]["has_c_hi"] = True
            other_ctx["per_task"][i]["c_hi_s"] = 0.3
            other_ctx["per_task"][i]["drop_allowed"] = True
        other = self._encode(dag, other_ctx)
        np.testing.assert_array_equal(
            base[:, self.dl_start:self.dl_end], other[:, self.dl_start:self.dl_end]
        )
        self.assertFalse(
            np.array_equal(
                base[:, self.idx["mc_c_lo_scaled"]],
                other[:, self.idx["mc_c_lo_scaled"]],
            )
        )
        self.assertFalse(
            np.array_equal(
                base[:, self.idx["mc_drop_allowed"]],
                other[:, self.idx["mc_drop_allowed"]],
            )
        )


class TestModelAndAdapterFields(unittest.TestCase):
    def test_canonical_task_validation(self):
        CanonicalTask(task_id=0, compute_workload_bytes=1, task_output_bytes=1,
                      c_lo_s=0.5, c_hi_s=0.5)
        with self.assertRaises(ValueError):
            CanonicalTask(task_id=0, compute_workload_bytes=1, task_output_bytes=1,
                          c_lo_s=0.0)
        with self.assertRaises(ValueError):
            CanonicalTask(task_id=0, compute_workload_bytes=1, task_output_bytes=1,
                          c_lo_s=0.5, c_hi_s=0.4)

    def test_defaults_keep_legacy_positional_construction(self):
        task = CanonicalTask(0, 10, 4, 3, None, 5.0, "hard", "high", 2.0)
        self.assertIsNone(task.c_lo_s)
        self.assertIsNone(task.c_hi_s)
        self.assertFalse(task.drop_allowed_hi_mode)
        self.assertFalse(task.mc_mode_is_hi)

    def test_adapter_maps_mc_fields_with_getattr_defaults(self):
        class _Task:
            def __init__(self, **kw):
                self.processing_data_size = 10
                self.transmission_data_size = 4
                self.__dict__.update(kw)

        class _Graph:
            task_list = [
                _Task(c_lo_s=0.2, c_hi_s=0.4, drop_allowed_lo_mode=True,
                      drop_allowed_hi_mode=True, degrade_allowed_hi_mode=True,
                      mc_mode_is_hi=True),
                _Task(),  # no MC attributes at all -> schema defaults
            ]
            pre_task_sets = [[], [0]]
            edge_set = [[0, 0, 0, 4, 1, 1, 0]]

        dag = to_canonical_dag(_Graph())
        self.assertEqual(dag.tasks[0].c_lo_s, 0.2)
        self.assertEqual(dag.tasks[0].c_hi_s, 0.4)
        self.assertTrue(dag.tasks[0].drop_allowed_lo_mode)
        self.assertTrue(dag.tasks[0].drop_allowed_hi_mode)
        self.assertFalse(dag.tasks[0].degrade_allowed_lo_mode)
        self.assertTrue(dag.tasks[0].degrade_allowed_hi_mode)
        self.assertTrue(dag.tasks[0].mc_mode_is_hi)
        self.assertIsNone(dag.tasks[1].c_lo_s)
        self.assertIsNone(dag.tasks[1].c_hi_s)
        self.assertFalse(dag.tasks[1].mc_mode_is_hi)


class TestStatsFile(_VersionGuard):
    def test_default_stats_loads_the_new_file(self):
        eo.set_obs_version(VERSION)
        stats = eo.default_feature_stats()
        self.assertEqual(stats.mean.shape, (40,))
        self.assertEqual(stats.std.shape, (40,))
        self.assertEqual(tuple(stats.feature_names), eo.FEATURE_NAMES_AUTOMOTIVE_MC_V1)

    def test_stats_file_shape_and_tag(self):
        data = json.loads(STATS_FILE.read_text())
        self.assertEqual(len(data["feature_names"]), 40)
        self.assertEqual(data["feature_names"], list(eo.FEATURE_NAMES_AUTOMOTIVE_MC_V1))
        self.assertEqual(data["packed_dim"], 79)
        self.assertEqual(data["max_tasks"], 20)
        self.assertEqual(data["max_neigh"], 19)
        self.assertEqual(data["obs_version"], VERSION)
        self.assertEqual(len(data["mean"]), 40)
        self.assertEqual(len(data["std"]), 40)
        self.assertEqual(data["mean"][-9:], [0.0] * 9)
        self.assertEqual(data["std"][-9:], [1.0] * 9)

    def test_stats_rows_copied_verbatim_from_v3(self):
        v3 = json.loads((ROOT / "spec/encoder_feature_stats_v3.json").read_text())
        new = json.loads(STATS_FILE.read_text())
        self.assertEqual(new["mean"][:31], v3["mean"])
        self.assertEqual(new["std"][:31], v3["std"])

    def test_stats_tool_check_passes(self):
        proc = subprocess.run(
            [sys.executable, str(STATS_TOOL), "--check"],
            capture_output=True, text=True, cwd=str(ROOT),
            env={**os.environ, "PYTHONPATH": str(ROOT)},
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
