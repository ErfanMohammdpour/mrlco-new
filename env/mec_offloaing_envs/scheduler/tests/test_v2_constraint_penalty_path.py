#!/usr/bin/env python3
"""F14/F33: the penalty PATH on controlled feasible / violating / mixed fixtures.

The production gate used to require that some episode in EVERY run was penalised. That is not a
property of a valid constraint: with lambda = 0 the hinge penalty is EXACTLY zero however large
the violation, and a feasible production batch may legitimately pass with zero penalties. The
requirement was removed from the gate and is pinned here instead.

Rule under test (the real `ConstraintCosts.penalty` and the real
`V2ConstraintManager.apply_penalty`):
    penalty = sum_i lambda_i * max(0, g_i)          (hinge primal)
    lambda_{t+1} = clip(lambda_t + lr * mean_i g_i) (signed dual ascent)
"""
import sys
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env.mec_offloaing_envs.scheduler.constraints import ConstraintCosts  # noqa: E402
from spec.automotive_training.v2.constraints_v2 import V2ConstraintManager  # noqa: E402
from spec.objective_contract import SHAPING_DISCOUNT, discounted_return  # noqa: E402


def _costs(signed, names=("total_energy",), budgets=(1.0,), scales=(1.0,)):
    signed = tuple(float(x) for x in np.asarray(signed, dtype=float).reshape(-1))
    return ConstraintCosts(
        names=tuple(names),
        raw=tuple(s * sc + b for s, sc, b in zip(signed, scales, budgets)),
        budgets=tuple(budgets),
        scales=tuple(scales),
        signed=signed,
        violations=tuple(max(0.0, s) for s in signed),
    )


class _StubManager:
    """Carries just enough state to exercise the REAL `apply_penalty` body."""

    def __init__(self, penalty):
        self.controller = type("C", (), {"penalty": lambda _self, _c: float(penalty)})()
        self.applied_penalty = 0.0


class PenaltyRuleTests(unittest.TestCase):
    def test_lambda_zero_gives_exactly_zero_penalty_even_when_violating(self):
        c = _costs([3.0])
        self.assertEqual(c.total_violation, 3.0)
        self.assertEqual(c.penalty([0.0]), 0.0,
                         "lambda=0 must yield a zero hinge penalty however large the violation")

    def test_positive_lambda_times_violation(self):
        self.assertAlmostEqual(_costs([2.0]).penalty([0.25]), 0.5, places=12)

    def test_feasible_row_prices_to_zero(self):
        c = _costs([-1.5])
        self.assertEqual(c.total_violation, 0.0)
        self.assertEqual(c.penalty([0.25]), 0.0)

    def test_mixed_batch_prices_only_violating_rows(self):
        """A batch is a SET of per-episode cost objects, one entry per constraint."""
        batch = [_costs([-1.0]), _costs([0.0]), _costs([2.0])]
        total = sum(c.penalty([0.25]) for c in batch)
        self.assertAlmostEqual(total, 0.5, places=12)
        self.assertAlmostEqual(sum(c.total_violation for c in batch), 2.0, places=12)

    def test_multiple_channels_price_independently(self):
        c = _costs([2.0, 0.0], names=("total_energy", "ue_energy"),
                   budgets=(1.0, 1.0), scales=(1.0, 1.0))
        self.assertAlmostEqual(c.penalty([0.25, 0.5]), 0.5, places=12)

    def test_negative_lambda_is_rejected_by_the_manager(self):
        """The MANAGER validates the broadcast vector; a negative multiplier is refused."""
        stub = type("M", (), {})()
        stub.names = ("total_energy",)
        stub.controller = type("C", (), {"_lambdas": [0.0]})()
        with self.assertRaises(Exception):
            V2ConstraintManager.set_lambdas(stub, {"total_energy": -0.1})

    def test_lambda_vector_length_mismatch_is_rejected(self):
        stub = type("M", (), {})()
        stub.names = ("total_energy", "ue_energy")
        stub.controller = type("C", (), {"_lambdas": [0.0, 0.0]})()
        # a positional vector must match the constraint count exactly
        with self.assertRaises(Exception):
            V2ConstraintManager.set_lambdas(stub, [0.1])
        # an unknown NAME in a mapping is refused too
        with self.assertRaises(Exception):
            V2ConstraintManager.set_lambdas(stub, {"not_a_channel": 0.1})


class TerminalApplicationTests(unittest.TestCase):
    def test_charge_hits_only_the_terminal_token(self):
        m = _StubManager(penalty=0.4)
        rewards = [0.1] * 20
        before = list(rewards)
        applied = V2ConstraintManager.apply_penalty(
            m, rewards, _costs([2.0]), terminal_weight=SHAPING_DISCOUNT ** 19)
        self.assertAlmostEqual(applied, 0.4, places=12)
        self.assertEqual(rewards[:-1], before[:-1])
        self.assertLess(rewards[-1], before[-1])

    def test_charge_contributes_exactly_minus_penalty_to_the_discounted_return(self):
        m = _StubManager(penalty=0.37)
        rewards = [0.1] * 20
        before = list(rewards)
        applied = V2ConstraintManager.apply_penalty(
            m, rewards, _costs([2.0]), terminal_weight=SHAPING_DISCOUNT ** 19)
        delta = discounted_return(rewards) - discounted_return(before)
        self.assertAlmostEqual(delta, -applied, places=9,
                               msg="terminal charge is not exactly -penalty in the return")

    def test_inactive_costs_apply_nothing(self):
        m = _StubManager(penalty=99.0)          # would be charged IF the costs were active
        rewards = [0.1] * 5
        before = list(rewards)
        self.assertEqual(V2ConstraintManager.apply_penalty(m, rewards, ConstraintCosts()), 0.0)
        self.assertEqual(rewards, before)

    def test_equal_endpoints_with_and_without_penalty(self):
        """Zero and nonzero P, same endpoints: the only difference is exactly P."""
        base = [0.2] * 20
        with_p = list(base)
        V2ConstraintManager.apply_penalty(_StubManager(penalty=0.5), with_p, _costs([5.0]),
                                          terminal_weight=SHAPING_DISCOUNT ** 19)
        self.assertAlmostEqual(discounted_return(base) - discounted_return(with_p), 0.5,
                               places=9)

    def test_bad_terminal_weight_is_rejected(self):
        for bad in (0.0, -1.0, float("inf")):
            with self.assertRaises(Exception):
                V2ConstraintManager.apply_penalty(_StubManager(penalty=0.4), [0.1] * 3,
                                                  _costs([1.0]), terminal_weight=bad)


class GateSemanticsTests(unittest.TestCase):
    """F33: the run guard must NOT demand a penalised episode unconditionally."""

    def test_guard_does_not_require_a_penalised_episode(self):
        from spec.automotive_training.v2 import run_tf
        evidence = {
            "iterations_completed": 3, "next_iteration": 3, "iterations_requested": 3,
            "sampler_counters": {"support_calls": 3},
            "last_rollout": {"makespan_s": 0.5, "system_joules": 10.0},
            "constraints": {"updates": 3},
            "penalty": {"signed_rows_observed": 600, "episodes_with_nonzero_penalty": 0},
            "lambda_broadcast": {"targets": 11},
        }
        # a fully feasible run with zero penalties must PASS the structural guard
        run_tf._assert_real_run(evidence)

    def test_guard_still_requires_the_dual_to_be_live(self):
        from spec.automotive_training.v2 import run_tf
        base = {
            "iterations_completed": 3, "next_iteration": 3, "iterations_requested": 3,
            "sampler_counters": {"support_calls": 3},
            "last_rollout": {"makespan_s": 0.5, "system_joules": 10.0},
            "constraints": {"updates": 3},
            "penalty": {"signed_rows_observed": 600, "episodes_with_nonzero_penalty": 0},
            "lambda_broadcast": {"targets": 11},
        }
        for broken in ({"constraints": {"updates": 0}}, {"lambda_broadcast": {"targets": 0}},
                       {"penalty": {"signed_rows_observed": 0}}):
            bad = dict(base)
            bad.update(broken)
            with self.assertRaises(run_tf.V2RunError):
                run_tf._assert_real_run(bad)


if __name__ == "__main__":
    unittest.main(verbosity=2)
