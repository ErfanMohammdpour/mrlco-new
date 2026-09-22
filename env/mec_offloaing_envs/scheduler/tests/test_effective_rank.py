#!/usr/bin/env python3
"""⑤b math tests for the effective-rank / separability probe (pure numpy)."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
for _name in ("gym", "gym.core"):
    if _name not in sys.modules:
        sys.modules[_name] = types.ModuleType(_name)
if not hasattr(sys.modules.get("gym.core", types.ModuleType("gym.core")), "Env"):
    sys.modules["gym.core"].Env = type("Env", (), {})

import numpy as np  # noqa: E402
import unittest  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "effective_rank_probe", str(ROOT / "spec" / "effective_rank_probe.py")
)
probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(probe)


class TestRankMath(unittest.TestCase):
    def test_rank_one_matrix(self):
        x = np.outer(np.arange(50.0), np.ones(6))
        stats = probe.effective_rank_stats(x)
        self.assertAlmostEqual(stats["effective_rank"], 1.0, places=6)
        self.assertAlmostEqual(stats["participation_ratio"], 1.0, places=6)
        self.assertEqual(stats["n90"], 1)

    def test_rank_equals_dim_when_isotropic(self):
        rng = np.random.RandomState(0)
        x = rng.normal(size=(4000, 30))       # full-rank isotropic noise
        stats = probe.effective_rank_stats(x)
        self.assertGreater(stats["effective_rank"], 20.0)
        self.assertLessEqual(stats["effective_rank"], 30.0 + 1e-6)

    def test_constant_columns_do_not_inflate_rank(self):
        rng = np.random.RandomState(1)
        x = np.concatenate([rng.normal(size=(500, 4)), np.zeros((500, 20))], axis=1)
        stats = probe.effective_rank_stats(x)
        self.assertLessEqual(stats["effective_rank"], 4.0 + 1e-6)
        self.assertEqual(stats["dims"][1], 4)     # constant columns dropped

    def test_ninety_percent_components_monotone(self):
        rng = np.random.RandomState(2)
        x = rng.normal(size=(800, 40)) @ np.diag(np.linspace(5.0, 0.05, 40))
        stats = probe.effective_rank_stats(x)
        self.assertLessEqual(stats["n90"], stats["n95"])
        self.assertLessEqual(stats["n95"], stats["n99"])

    def test_empty_matrix_is_safe(self):
        stats = probe.effective_rank_stats(np.zeros((10, 5)))
        self.assertEqual(stats["effective_rank"], 0.0)


class TestSeparabilityMath(unittest.TestCase):
    def _blobs(self, centres, n=120, dim=8, seed=0):
        rng = np.random.RandomState(seed)
        xs, ys = [], []
        for label, centre in enumerate(centres):
            xs.append(rng.normal(loc=centre, scale=0.4, size=(n, dim)))
            ys.append(np.full(n, label))
        return np.concatenate(xs), np.concatenate(ys)

    def test_ridge_probe_separates_well_separated_blobs(self):
        x, y = self._blobs([np.zeros(8), np.ones(8) * 6.0])
        res = probe.ridge_probe_accuracy(x, y)
        self.assertGreater(res["accuracy"], 0.95)
        self.assertAlmostEqual(res["chance"], 0.5, places=9)

    def test_ridge_probe_near_chance_for_overlapping_blobs(self):
        x, y = self._blobs([np.zeros(8), np.ones(8) * 0.05])
        res = probe.ridge_probe_accuracy(x, y)
        self.assertLess(res["accuracy"], 0.75)

    def test_fisher_ratio_orders_separability(self):
        # same dim for both cases: only the separation changes
        near, y_near = self._blobs([np.zeros(8), np.ones(8) * 0.2])
        far, y_far = self._blobs([np.zeros(8), np.ones(8) * 6.0])
        self.assertLess(probe.fisher_ratio(near, y_near), probe.fisher_ratio(far, y_far))

    def test_single_class_probe_returns_nan(self):
        res = probe.ridge_probe_accuracy(np.zeros((20, 3)), np.zeros(20, dtype=int))
        self.assertTrue(np.isnan(res["accuracy"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
