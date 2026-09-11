#!/usr/bin/env python3
"""Per-instance EAS budget math tests."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spec.eas_instance import BUDGET_PRESETS, budget_nk, summarize_rows  # noqa: E402


class TestEasInstance(unittest.TestCase):
    def test_budget_presets(self):
        for b, p in BUDGET_PRESETS.items():
            n, k = budget_nk(b)
            self.assertEqual(n * k, b)
            self.assertEqual(n, p["n_adapt"])
            self.assertEqual(k, p["k"])

    def test_summarize_gate(self):
        rows = []
        for i in range(4):
            rows.append(
                {
                    "eas": {
                        "T_greedy0": 600.0,
                        "T_best_among_samples": 500.0,
                        "T_greedy_after": 580.0,
                    },
                    "bok": {"T_best": 510.0, "T_greedy": 600.0},
                }
            )
        s = summarize_rows(rows, 32)
        self.assertAlmostEqual(s["delta_eas_minus_bok"], -10.0)
        self.assertTrue(s["gate_eas_better_by_5s"])
        self.assertEqual(s["frac_eas_beats_bok"], 1.0)


if __name__ == "__main__":
    unittest.main()
