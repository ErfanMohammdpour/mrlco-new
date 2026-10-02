#!/usr/bin/env python3
"""Blocking correctness gates for the automotive training loop (Gates B/C/D/E).

These tests encode the four defects found in the 5x500 pilot:
  B  the validation evaluator never copied the trained core into the scratch policy
     (`Seq2SeqPolicy` has no `assign_trainable`), so k0/k3 measured a stale policy;
  C  the Lagrangian multipliers were only written to the original env while rollouts
     run on `copy.deepcopy` clones, so the penalty never reached the reward;
  D1 the penalty was applied twice in the non-single-distribution step;
  D2 the dual batch was never reset, so the ascent used the whole history.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spec.automotive_training.automotive_constraints import (  # noqa: E402
    CONSTRAINT_NAMES,
    AutomotiveDualController,
    default_constraint_specs,
)
from spec.automotive_training.automotive_env import AutomotiveEnv, _constraint_view  # noqa: E402
from spec.automotive_training.automotive_loader import load_dataset  # noqa: E402
from spec.automotive_training.automotive_primary import (  # noqa: E402
    AutomotivePrimaryError,
    AutomotiveResourceCluster,
    AutomotiveMetaSampler,
    metrics_from_paths,
)

ZERO = {name: 0.0 for name in CONSTRAINT_NAMES}


def _graph_with_high():
    dataset = load_dataset()
    graphs = dataset.meta_train()
    return next(g for g in graphs if any(str(t.criticality) == "HIGH" for t in g.tasks))


PLAN_CANDIDATES = ([0] * 20, [2] * 20, [1] * 20,
                   [2, 2, 2, 2, 2, 1, 1, 1, 0, 0] * 2)


def _violating_case():
    """Deterministically find a (graph, plan, seed) whose penalty is non-zero.

    all-MEC is latency-optimal and often violates nothing, so the fixture must be
    searched rather than assumed.
    """
    dataset = load_dataset()
    graphs = [g for g in dataset.meta_train()
              if any(str(t.criticality) == "HIGH" for t in g.tasks)][:6]
    lam = {"C_GRAPH_HARD_DEADLINE": 1.0, "C_HI_TASK_TARDINESS": 1.0,
           "C_MED_TASK_TARDINESS": 1.0}
    for graph in graphs:
        for plan in PLAN_CANDIDATES:
            for seed in (5, 11, 17):
                env = _env(graph, seed=seed)
                env.set_constraint_lambdas(lam)
                env.reset()
                _o, _r, _d, info = env.step(np.array([plan]))
                penalty = float(info[2][0]["constraint_penalty"])
                if penalty > 0.0:
                    return graph, plan, seed, penalty
    raise AssertionError("no violating fixture found in the frozen meta-train split")


def _env(graph, seed=11, slots=1):
    env = AutomotiveEnv([graph], AutomotiveResourceCluster(), slots_per_task=slots,
                        base_seed=seed, mc_enabled=True)
    env.set_task({"dist_index": 0, "graph_indices": np.zeros(slots, dtype=np.int32)})
    return env


class TestPenaltyApplication(unittest.TestCase):
    """Gate D1: the Lagrangian penalty must be applied exactly once."""

    def test_penalty_applied_exactly_once(self):
        graph, plan, seed, _p = _violating_case()
        plan = np.array([plan])
        lam = {"C_GRAPH_HARD_DEADLINE": 0.7, "C_HI_TASK_TARDINESS": 0.9,
               "C_MED_TASK_TARDINESS": 0.3}

        free = _env(graph, seed=seed)
        free.set_constraint_lambdas(ZERO)
        free.reset()
        _obs, rewards_free, _d, _i = free.step(plan)

        taxed = _env(graph, seed=seed)
        taxed.set_constraint_lambdas(lam)
        taxed.reset()
        _obs2, rewards_taxed, _d2, info = taxed.step(plan)

        telemetry = info[2][0]
        penalty = float(telemetry["constraint_penalty"])
        self.assertGreater(penalty, 0.0, "fixture must violate at least one channel")
        expected = -penalty / float(graph.D_G_s)
        delta = float(rewards_taxed[-1][-1] - rewards_free[-1][-1])
        # rewards are stored as float32, so compare with a relative tolerance
        self.assertLessEqual(abs(delta - expected), 1e-5 * max(1.0, abs(expected)),
                             "penalty must be applied EXACTLY once")
        self.assertGreater(abs(delta - 2.0 * expected), 0.5 * abs(expected),
                           "the double-application regression would give 2x the penalty")
        self.assertAlmostEqual(taxed.last_penalty_applied, penalty, places=10)
        self.assertEqual(len(rewards_taxed), 1)      # one slot
        self.assertEqual(len(rewards_taxed[0]), 20)  # 20 tokens

    def test_telemetry_penalty_matches_channels(self):
        from spec.automotive_training.automotive_constraints import evaluate_constraints

        graph, plan, seed, _p = _violating_case()
        weights = {"C_GRAPH_HARD_DEADLINE": 0.5, "C_HI_TASK_TARDINESS": 0.25,
                   "C_MED_TASK_TARDINESS": 0.0}
        env = _env(graph, seed=seed)
        env.set_constraint_lambdas(weights)
        env.reset()
        _o, _r, _d, info = env.step(np.array([plan]))
        record = info[2][0]
        mc = env._slot_mc[0]
        channels = evaluate_constraints(_constraint_view(graph, mc),
                                        env._schedule(0, plan, mc)[0])
        expected = sum(weights[name] * float(channels[name]["violation"])
                       for name in CONSTRAINT_NAMES)
        self.assertAlmostEqual(float(record["constraint_penalty"]), expected, places=10)


class TestLambdaBroadcast(unittest.TestCase):
    """Gate C: multipliers must reach every env a rollout can run on."""

    def test_broadcast_reaches_iterative_clones(self):
        try:
            from samplers.vectorized_env_executor import MetaIterativeEnvExecutor
        except Exception as exc:  # pragma: no cover - heavy optional deps
            self.skipTest("executor import failed: %r" % (exc,))
        graph = _graph_with_high()
        env = _env(graph)
        executor = MetaIterativeEnvExecutor(env, meta_batch_size=2, envs_per_task=2,
                                            max_path_length=20)
        lam = {"C_GRAPH_HARD_DEADLINE": 0.4, "C_HI_TASK_TARDINESS": 0.6,
               "C_MED_TASK_TARDINESS": 0.1}
        updated = executor.set_constraint_lambdas(lam)
        self.assertEqual(updated, len(executor.envs))
        for clone in executor.envs:
            for name in CONSTRAINT_NAMES:
                self.assertAlmostEqual(clone.constraint_lambdas.get(name, 0.0), lam[name])
        # the trainer-side helper must reach the clones through the sampler too
        from spec.automotive_training.automotive_trainer import AutomotiveTrainerMixin

        class FakeSampler(object):
            vec_env = executor

        class Probe(AutomotiveTrainerMixin):
            pass

        probe = Probe()
        spec_map = default_constraint_specs()
        probe.auto_controller = AutomotiveDualController(
            specs={name: spec_map[name] for name in CONSTRAINT_NAMES}, dual_lr=0.05)
        probe.auto_env = env
        probe.meta_sampler = FakeSampler()
        targets = probe.broadcast_constraint_lambdas()
        self.assertGreater(targets, 1, "broadcast must include the executor clones")

    def test_penalty_reaches_the_reward_through_the_clones(self):
        graph, plan, seed, _p = _violating_case()
        plan = np.array([plan])
        lam = {"C_GRAPH_HARD_DEADLINE": 1.0, "C_HI_TASK_TARDINESS": 1.0,
               "C_MED_TASK_TARDINESS": 1.0}
        taxed = _env(graph, seed=seed)
        taxed.set_constraint_lambdas(lam)          # simulates a correct broadcast
        taxed.reset()
        _o, rewards, _d, info = taxed.step(plan)
        self.assertGreater(float(info[2][0]["constraint_penalty"]), 0.0)
        free = _env(graph, seed=seed)
        free.set_constraint_lambdas(ZERO)
        free.reset()
        _o2, base_rewards, _d2, _i2 = free.step(plan)
        self.assertLess(float(rewards[-1][-1]), float(base_rewards[-1][-1]),
                        "a broadcast lambda must make the reward strictly worse")


class TestDualBatch(unittest.TestCase):
    """Gate D2: one dual step per iteration, on that iteration's batch only."""

    def _controller(self):
        spec_map = default_constraint_specs()
        return AutomotiveDualController(
            specs={name: spec_map[name] for name in CONSTRAINT_NAMES}, dual_lr=0.5)

    def test_batch_resets_and_updates_from_current_batch_only(self):
        controller = self._controller()
        controller.observe({name: [1.0] * 10 for name in CONSTRAINT_NAMES})
        controller.dual_step()
        first = dict(controller.lambdas)
        self.assertAlmostEqual(controller.batch_size("C_GRAPH_HARD_DEADLINE"), 10)
        controller.reset_batch()
        self.assertEqual(controller.batch_size("C_GRAPH_HARD_DEADLINE"), 0)
        # a satisfied next batch must NOT move lambda (history would have moved it)
        controller.observe({name: [0.0] * 10 for name in CONSTRAINT_NAMES})
        diag = controller.dual_step()
        self.assertEqual(diag["updated"], [])
        for name in CONSTRAINT_NAMES:
            self.assertAlmostEqual(controller.lambdas[name], first[name])
        controller.reset_batch()
        # and a fresh violating batch moves it again from the new mean only
        controller.observe({name: [2.0] * 10 for name in CONSTRAINT_NAMES})
        diag = controller.dual_step()
        self.assertIn("C_GRAPH_HARD_DEADLINE", diag["updated"])


class TestMetricSemantics(unittest.TestCase):
    """Gate E: no misleading metric names/denominators."""

    def _telemetry(self, n_high, n_med, tardy_high, tardy_med, makespan=0.1):
        return {"makespan_s": makespan, "violation/C_GRAPH_HARD_DEADLINE": 0.01,
                "violation/C_HI_TASK_TARDINESS": 0.002 if tardy_high else 0.0,
                "violation/C_MED_TASK_TARDINESS": 0.003 if tardy_med else 0.0,
                "firm_miss_count": tardy_high + tardy_med, "mode": "HI",
                "mode_switch_count": 1, "dropped_task_count": 2, "high_preserved": 1.0,
                "n_tasks_high": n_high, "n_tasks_medium": n_med,
                "n_tardy_high_tasks": tardy_high, "n_tardy_medium_tasks": tardy_med}

    def test_task_level_rates_and_firm_denominator(self):
        paths = {0: [{"energy_telemetry": self._telemetry(4, 2, 1, 1)},
                     {"energy_telemetry": self._telemetry(4, 2, 0, 0)}]}
        metrics = metrics_from_paths(paths)
        # task-level: 1 tardy HIGH of 8 HIGH tasks across the batch
        self.assertAlmostEqual(metrics["high_task_tardiness_task_rate"], 1.0 / 8.0)
        self.assertAlmostEqual(metrics["medium_task_tardiness_task_rate"], 1.0 / 4.0)
        # graph incidence: 1 of 2 graphs
        self.assertAlmostEqual(metrics["graph_high_tardiness_incidence_rate"], 0.5)
        # firm miss denominator is the number of firm tasks (8 + 4), not 20 * graphs
        self.assertAlmostEqual(metrics["firm_task_miss_rate"], 2.0 / 12.0)
        self.assertNotIn("high_task_tardiness_rate", metrics,
                         "the graph-incidence alias must not be reported as a rate")

    def test_zero_firm_tasks_is_not_a_division_error(self):
        paths = {0: [{"energy_telemetry": self._telemetry(0, 0, 0, 0)}]}
        metrics = metrics_from_paths(paths)
        self.assertEqual(metrics["firm_task_miss_rate"], 0.0)
        self.assertEqual(metrics["high_task_tardiness_task_rate"], 0.0)


class TestSamplerCounters(unittest.TestCase):
    """Gate E: the counters must be exact for the whole run."""

    class _StubSampler(object):
        def __init__(self):
            self.total_samples = 0
            self.calls = []

        def obtain_samples(self, log=False, log_prefix=""):
            self.calls.append((log, log_prefix))
            path = {"rewards": np.zeros((1, 20), dtype=np.float32),
                    "energy_telemetry": {"rollout_seed": len(self.calls),
                                         "graph_index": 0, "constraint_penalty": 0.0}}
            return {0: [path]}

        def update_tasks(self):
            return [{"dist_index": 0, "graph_indices": np.zeros(2, dtype=np.int32)}]

        def __getattr__(self, item):
            raise AttributeError(item)

    def test_roles_follow_the_log_flag(self):
        stub = self._StubSampler()
        wrapper = AutomotiveMetaSampler(stub, meta_batch_size=1,
                                        trajectories_per_meta_task=1,
                                        tokens_per_trajectory=20)
        for _ in range(3):
            wrapper.obtain_samples(log=False, log_prefix="")
            wrapper.obtain_samples(log=True, log_prefix="")
        counters = wrapper.counters_snapshot()
        self.assertEqual(counters["support_calls"], 3)
        self.assertEqual(counters["query_calls"], 3)
        self.assertTrue(counters["counter_role_contract_ok"])
        self.assertEqual(counters["support_trajectories"], 3)
        self.assertEqual(counters["query_trajectories"], 3)


class TestProvenance(unittest.TestCase):
    def test_fingerprint_reports_git_sha_and_dirty_and_run_kind(self):
        from spec.automotive_training.automotive_fingerprint import (
            training_fingerprint,
        )

        fp = training_fingerprint(
            obs_version="automotive_mc_obs_v1",
            scheduler_axes={"timing_model": "physical_rates"},
            checkpoint_rule_sha="deadbeef",
            sampler_budget={"meta_batch_size": 10},
            run_kind="primary_500", outer_iterations=500)
        parts = fp["parts"]
        self.assertNotEqual(parts["git_sha"], "unknown")
        self.assertIn("code_dirty", parts)
        self.assertEqual(parts["run_kind"], "primary_500")
        self.assertEqual(int(parts["outer_iterations"]), 500)
        other = training_fingerprint(
            obs_version="automotive_mc_obs_v1",
            scheduler_axes={"timing_model": "physical_rates"},
            checkpoint_rule_sha="deadbeef",
            sampler_budget={"meta_batch_size": 10},
            run_kind="primary_250", outer_iterations=250)
        self.assertNotEqual(fp["fingerprint"], other["fingerprint"],
                            "the iteration budget must change the fingerprint")


if __name__ == "__main__":
    unittest.main(verbosity=2)
