#!/usr/bin/env python3
"""Discount-consistent telescoping reward tests (pure numpy).

The approved reward is

    r_t = J(s_{t-1}) - gamma * J(s_t),      J = w_L*L/L_scale (+ w_E*E/E_scale)

which for gamma == 1 is exactly the historical delta form, and for gamma < 1 is
potential-based shaping whose discounted sum telescopes to J_0 - gamma^N J_N.
"""

from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _stub_optional(name: str) -> None:
    if name in sys.modules:
        return
    try:
        __import__(name)
    except Exception:
        sys.modules[name] = types.ModuleType(name)


for _name in ("gym", "gym.core", "graphviz", "pydotplus", "pydotplus.graphviz"):
    _stub_optional(_name)
if not hasattr(sys.modules.get("gym.core", types.ModuleType("gym.core")), "Env"):
    sys.modules.setdefault("gym", types.ModuleType("gym"))
    sys.modules.setdefault("gym.core", types.ModuleType("gym.core"))
    sys.modules["gym.core"].Env = type("Env", (), {})
if not hasattr(sys.modules.get("graphviz", types.ModuleType("graphviz")), "Digraph"):
    sys.modules.setdefault("graphviz", types.ModuleType("graphviz"))
    sys.modules["graphviz"].Digraph = type("Digraph", (), {})

from env.mec_offloaing_envs.scheduler import (  # noqa: E402
    ResourceConfig,
    expected_episode_return,
    pure_location_plan,
    telescoping_token_rewards,
)

N = 6
GAMMA = 0.99


class _FakeTask:
    def __init__(self, proc: int, tx: int):
        self.processing_data_size = proc
        self.transmission_data_size = tx


class _FakeTG:
    def __init__(self, n: int = N):
        self.task_number = n
        self.task_list = [_FakeTask(1_572_864, 786_432) for _ in range(n)]
        self.prioritize_sequence = list(range(n))
        self.pre_task_sets = [set() if i == 0 else {i - 1} for i in range(n)]
        self.succ_task_sets = [set() if i == n - 1 else {i + 1} for i in range(n)]
        self.edge_set = [
            [i, i, 1_572_864, 786_432, i + 1, i + 1, 1_572_864] for i in range(n - 1)
        ]


def _mixed_plan(tg):
    """A deliberately mixed plan: alternate local / MEC / helper."""
    order = [int(t) for t in tg.prioritize_sequence]
    return [(tid, k % 3) for k, tid in enumerate(order)]


class TestDiscountConsistentShaping(unittest.TestCase):
    def setUp(self):
        self.tg = _FakeTG()
        self.res = ResourceConfig.from_frozen_yaml()
        self.plan = _mixed_plan(self.tg)

    def test_discount_one_matches_historical_delta_form(self):
        out = telescoping_token_rewards(self.tg, self.plan, self.res, discount=1.0)
        refs = out.refs
        expected = []
        for t in range(1, len(out.makespans)):
            dl = (out.makespans[t] - out.makespans[t - 1]) / refs.L_scale
            de = (out.energies[t] - out.energies[t - 1]) / refs.E_scale
            expected.append(-(0.5 * dl + 0.5 * de))
        for got, want in zip(out.rewards, expected):
            self.assertAlmostEqual(got, want, places=12)

    def test_default_discount_is_one(self):
        base = telescoping_token_rewards(self.tg, self.plan, self.res)
        explicit = telescoping_token_rewards(self.tg, self.plan, self.res, discount=1.0)
        self.assertEqual(base.rewards, explicit.rewards)

    def test_discounted_sum_telescopes_exactly(self):
        out = telescoping_token_rewards(self.tg, self.plan, self.res, discount=GAMMA)
        refs = out.refs
        j0 = 0.5 * out.makespans[0] / refs.L_scale + 0.5 * out.energies[0] / refs.E_scale
        jn = 0.5 * out.makespans[-1] / refs.L_scale + 0.5 * out.energies[-1] / refs.E_scale
        n = len(out.rewards)
        discounted = sum((GAMMA ** (t - 1)) * r for t, r in enumerate(out.rewards, start=1))
        self.assertAlmostEqual(discounted, j0 - (GAMMA ** n) * jn, places=10)

    def test_discounted_all_optimizers_agree_on_final_objective(self):
        """Any two plans with the same J_N must have the same discounted return."""
        other = [(tid, 1) for tid, _ in self.plan]
        a = telescoping_token_rewards(self.tg, self.plan, self.res, discount=GAMMA)
        b = telescoping_token_rewards(self.tg, other, self.res, discount=GAMMA)
        disc_a = sum((GAMMA ** (t - 1)) * r for t, r in enumerate(a.rewards, start=1))
        disc_b = sum((GAMMA ** (t - 1)) * r for t, r in enumerate(b.rewards, start=1))
        refs = a.refs

        def j(out):
            return 0.5 * out.final_makespan / refs.L_scale + 0.5 * out.final_energy / refs.E_scale

        n = len(a.rewards)
        self.assertAlmostEqual(disc_a - disc_b, (GAMMA ** n) * (j(b) - j(a)), places=10)

    def test_expected_episode_return_matches_numeric_sum(self):
        for gamma in (1.0, 0.95, GAMMA):
            out = telescoping_token_rewards(self.tg, self.plan, self.res, discount=gamma)
            closed = expected_episode_return(
                out.makespans, out.energies, out.refs, discount=gamma
            )
            if gamma == 1.0:
                numeric = float(sum(out.rewards))
            else:
                numeric = float(
                    sum((gamma ** (t - 1)) * r for t, r in enumerate(out.rewards, start=1))
                )
            self.assertAlmostEqual(closed, numeric, places=9)

    def test_diagnostic_latency_ref_still_consistent(self):
        out = telescoping_token_rewards(
            self.tg, self.plan, self.res, latency_ref="l_mec", discount=GAMMA
        )
        denom = out.refs.L_mec
        j0 = out.makespans[0] / denom
        jn = out.makespans[-1] / denom
        n = len(out.rewards)
        discounted = sum((GAMMA ** (t - 1)) * r for t, r in enumerate(out.rewards, start=1))
        self.assertAlmostEqual(discounted, j0 - (GAMMA ** n) * jn, places=10)

    def test_invalid_discount_rejected(self):
        for bad in (0.0, -0.5, 1.5):
            with self.assertRaises(ValueError):
                telescoping_token_rewards(self.tg, self.plan, self.res, discount=bad)


if __name__ == "__main__":
    unittest.main(verbosity=2)
