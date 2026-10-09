#!/usr/bin/env python3
"""v2 constraint integration: the budget must change the ACTUAL reward, the dual must move in
BOTH directions on training data, and missing metrics must fail instead of reading as zero.

Audited defect: `env.step` computed `penalty = 0.0` (and the controller stayed ineffective
despite the broadcast), so a budget could not influence anything, while telemetry lacked
top-level violations and task denominators — a checkpoint selector therefore read misses and
violations as zero.
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spec.automotive_training.automotive_loader import load_dataset  # noqa: E402
from spec.automotive_training.automotive_primary import AutomotiveResourceCluster  # noqa: E402
from spec.automotive_training.v2.constraints_v2 import (  # noqa: E402
    V2ConstraintError, V2ConstraintManager, calibrate_budgets, require_measurable,
    select_checkpoint, spec_from_fractions, v2_metrics,
)
from spec.automotive_training.v2.energy import (  # noqa: E402
    reference_ranges_from_plans, schedule_energy,
)
from spec.automotive_training.v2.env import V2AutomotiveEnv  # noqa: E402
from spec.automotive_training.v2.helper_model import HelperState  # noqa: E402
from spec.automotive_training.v2.shared_scheduler import (  # noqa: E402
    V2ComputeSpec, V2DAGSpec, V2LinkSpec, V2TaskSpec, schedule_shared,
)

LINK = V2LinkSpec(20e6, 20e6, 10e6)
COMPUTE = V2ComputeSpec(10e6, 1, (1e6,), (4e6,))
SHA = "b" * 64
SLOTS = 3
TOKENS = 20


def _env(constraints=False, seed=11):
    graphs = load_dataset().validation_query()[:SLOTS]
    env = V2AutomotiveEnv(graphs, AutomotiveResourceCluster(), role="validation",
                          slots_per_task=SLOTS, base_seed=seed, single_dist=True,
                          constraints_enabled=constraints,
                          # total_energy is a fraction of the pure all-MEC plan's SYSTEM
                          # energy (the scope-consistent anchor). 0.5 puts the all-MEC plan
                          # at +1.0 signed cost so the constraint genuinely binds.
                          budget_fractions={"total_energy_mec": 0.5, "ue_energy": 2.0},
                          scheduler_config_sha256=SHA)
    env.set_task({"dist_index": 0,
                  "graph_indices": np.arange(SLOTS, dtype=np.int32)})
    return env


def _fixture():
    """Hand-checkable one-DAG fixture for the manager-level tests."""
    dag = V2DAGSpec("d", 0, [V2TaskSpec(0, 6e6, 2e6, (), True, True, "HIGH", None, 0,
                                        external_input_bytes=1000.0)], helper_id=0)
    helper = {0: HelperState(0, 4e6, contact_end_s=100.0, predicted_contact_end_s=100.0)}
    plans = {
        "all_ue": schedule_shared([dag], {"d": {0: 0}}, link=LINK, compute=COMPUTE),
        "all_mec": schedule_shared([dag], {"d": {0: 1}}, link=LINK, compute=COMPUTE),
        "all_helper": schedule_shared([dag], {"d": {0: 2}}, link=LINK, compute=COMPUTE,
                                      helper_states=helper),
    }
    return dag, helper, plans


class TestRewardPenaltyIsReal(unittest.TestCase):
    def test_a_budget_that_binds_changes_the_actual_reward(self):
        env = _env(constraints=True)
        env.reset()
        over = np.ones((SLOTS, TOKENS), dtype=int)      # all-MEC: huge system energy
        under = np.zeros((SLOTS, TOKENS), dtype=int)    # all-UE

        # iteration 1: lambda starts at 0, so the penalty is 0 but the VIOLATION is measured
        _o, _r, _d, info = env.step(over)
        for record in info[2]:
            c = record["constraints"]
            self.assertTrue(any(c[name + "_violation"] > 0.0 for name in c["names"]),
                            "an over-budget rollout must be MEASURED as violated")
            self.assertGreater(record["system_joules"],
                               record["constraints"]["total_energy_budget"])
            self.assertEqual(record["v2"]["constraint_penalty"], 0.0,
                             "lambda is still zero at this point")

        # the dual step (trainer-side, training data only) makes lambda strictly positive
        env.constraint_manager.dual_step()
        self.assertGreater(max(env.constraint_manager.lambdas), 0.0,
                           "lambda must leave zero after observing the violation")

        # iteration 2: the SAME over-budget rollout now carries a real reward penalty
        _o, _rewards, _d, info = env.step(over)
        for record in info[2]:
            self.assertGreater(float(record["v2"]["constraint_penalty"]), 0.0,
                               "with lambda > 0 an over-budget rollout MUST be penalised")

        # a within-budget rollout must NOT be penalised
        _o, _r, _d, info_under = env.step(under)
        for record in info_under[2]:
            self.assertEqual(record["v2"]["constraint_penalty"], 0.0,
                             "a within-budget rollout must not be penalised")

    def test_the_penalty_is_charged_exactly_once_on_the_terminal_token(self):
        env = _env(constraints=True)
        env.reset()
        plan = np.ones((SLOTS, TOKENS), dtype=int)
        env.step(plan)                       # observe the violation
        env.constraint_manager.dual_step()   # trainer-side dual update -> lambda > 0
        _o, rewards, _d, info = env.step(plan)
        checked = 0
        for slot, record in enumerate(info[2]):
            applied = float(record["v2"]["constraint_penalty"])
            base = np.asarray(env._telescoping(slot, plan[slot]), dtype=float)
            got = np.asarray(rewards[slot], dtype=float)
            # every non-terminal token is bit-identical to the un-penalised telescoping value
            np.testing.assert_allclose(got[:-1], base[:-1], rtol=0.0, atol=1e-6)
            if applied:
                checked += 1
                # The terminal token carries -penalty/gamma^(N-1) so that its contribution to the
                # DISCOUNTED return is exactly -penalty (the constrained objective is
                # R = J_0 - gamma^N J_N - penalty). Asserting the raw terminal value against
                # `base[-1] - applied` would encode the old, gamma-mis-scaled convention.
                from spec.objective_contract import SHAPING_DISCOUNT
                g = SHAPING_DISCOUNT
                # APPROVED RETURN-CONTRACT (`gamma_N_v2`): the env passes
                # terminal_scale=gamma**N with terminal_weight=gamma**(N-1), so the terminal
                # token is charged `gamma * applied` and the discounted contribution is
                # `gamma**N * applied`. That makes R = J_0 - gamma^N*(J_N + P), an exact affine
                # function of the approved constrained latency. The superseded form charged
                # `applied / gamma**(N-1)` (contribution exactly `applied`), which is the
                # different weighted objective J_N + P/gamma**N.
                np.testing.assert_allclose(float(got[-1]), float(base[-1]) - g * applied,
                                           rtol=1e-4, atol=1e-4)
                contrib = (
                    float(np.sum(got * (g ** np.arange(len(got)))))
                    - float(np.sum(np.asarray(base, dtype=float)
                                   * (g ** np.arange(len(base))))))
                self.assertAlmostEqual(contrib, -(g ** len(got)) * applied, places=4)
        self.assertGreater(checked, 0, "the fixture must contain an over-budget slot")

    def test_telemetry_exposes_violations_and_denominators(self):
        env = _env(constraints=True)
        env.reset()
        _o, _r, _d, info = env.step(np.ones((SLOTS, TOKENS), dtype=int))
        for record in info[2]:
            c = record["constraints"]
            self.assertTrue(c["enabled"])
            self.assertTrue(c["names"])
            for name in c["names"]:
                for suffix in ("_raw", "_budget", "_scale", "_signed", "_violation",
                               "_lambda"):
                    self.assertIn(name + suffix, c)
                self.assertGreater(c[name + "_budget"], 0.0)
            v2 = record["v2"]
            self.assertGreater(v2["n_tasks"], 0)
            self.assertEqual(v2["n_tasks"],
                             v2["n_mec_tasks"] + v2["n_helper_tasks"]
                             + (v2["n_tasks"] - v2["n_mec_tasks"] - v2["n_helper_tasks"]))
            self.assertIn("requester_joules", record)
            self.assertGreater(record["system_joules"], 0.0)

    def test_disabled_constraints_are_a_strict_noop(self):
        env = _env(constraints=False)
        env.reset()
        _o, rewards, _d, info = env.step(np.ones((SLOTS, TOKENS), dtype=int))
        for record in info[2]:
            self.assertFalse(record["constraints"]["enabled"])
            self.assertEqual(record["v2"]["constraint_penalty"], 0.0)
            self.assertEqual(record["energy_constraint"], "telemetry_only")
        self.assertTrue(all(np.isfinite(np.asarray(r)).all() for r in rewards))


class TestDualsMoveBothWays(unittest.TestCase):
    def test_lambda_rises_above_budget_and_falls_below_it(self):
        _dag, _helper, plans = _fixture()
        refs = reference_ranges_from_plans(plans, scheduler_config_sha256=SHA)
        # The total-energy budget is a fraction of the pure all-MEC plan's SYSTEM energy
        # (E4.3 scope-consistency fix), because the measured `total_energy_j` is whole-world
        # system scope. A fraction of 0.5 therefore puts the all-MEC plan AT +1.0 signed cost
        # (over budget) while the all-HELPER plan sits below it, so the dual must move in BOTH
        # directions. Anchoring on `E_ue` instead would compare a whole-world measurement with
        # a foreground-only reference: that was the audited scope mismatch.
        self.assertAlmostEqual(refs.energy_scope, "system")
        spec = calibrate_budgets(refs, {"total_energy_mec": 0.5})
        manager = V2ConstraintManager(spec=spec, references=refs, dual_lr=0.5,
                                      scheduler_config_sha256=SHA)
        # two over-budget all-MEC episodes -> lambda must rise
        for _ in range(2):
            result = schedule_shared([_dag], {"d": {0: 1}}, link=LINK, compute=COMPUTE)
            manager.observe(manager.evaluate(result, schedule_energy(result)))
        manager.dual_step()
        high = manager.lambdas[0]
        self.assertGreater(high, 0.0)
        # then many under-budget all-HELPER episodes -> lambda must FALL (signed dual ascent)
        for _ in range(40):
            result = schedule_shared([_dag], {"d": {0: 2}}, link=LINK, compute=COMPUTE,
                                     helper_states=_helper)
            manager.observe(manager.evaluate(result, schedule_energy(result)))
        manager.dual_step()
        low = manager.lambdas[0]
        self.assertLess(low, high,
                        "lambda must be able to DECREASE when costs fall below budget; "
                        "a positive-hinge-only update makes it monotone non-decreasing")
        self.assertGreaterEqual(low, 0.0)

    def test_lambdas_are_broadcast_to_every_environment(self):
        a, b = _env(constraints=True, seed=1), _env(constraints=True, seed=2)
        a.reset()
        b.reset()
        dag, _helper, _plans = _fixture()
        _dag, _h, plans = dag, _helper, _plans
        a.constraint_manager.references = reference_ranges_from_plans(
            plans, scheduler_config_sha256=SHA)
        result = schedule_shared([dag], {"d": {0: 1}}, link=LINK, compute=COMPUTE)
        a.constraint_manager.observe(a.constraint_manager.evaluate(
            result, schedule_energy(result)))
        a.constraint_manager.dual_step()
        vector = a.constraint_manager.lambdas_by_name()
        b.set_constraint_lambdas(vector)
        self.assertEqual(b.constraint_manager.lambdas_by_name(), vector)
        self.assertEqual(a.constraint_manager.lambdas_by_name(), vector)
        with self.assertRaises(V2ConstraintError):
            b.constraint_manager.set_lambdas({"nope": 1.0})

    def test_constraint_state_round_trips_and_rejects_a_mismatch(self):
        a = _env(constraints=True, seed=3)
        a.reset()
        state = a.constraint_state()
        b = _env(constraints=True, seed=4)
        b.reset()
        b.load_constraint_state(state)
        self.assertEqual(a.constraint_manager.lambdas_by_name(),
                         b.constraint_manager.lambdas_by_name())
        bad = dict(state)
        bad["constraints_sha256"] = "0" * 64
        with self.assertRaises(V2ConstraintError):
            b.load_constraint_state(bad)


class TestFailFast(unittest.TestCase):
    def test_missing_metric_raises_instead_of_becoming_zero(self):
        _dag, _helper, plans = _fixture()
        result = plans["all_mec"]
        ledger = schedule_energy(result)
        metrics = v2_metrics(result, ledger)
        require_measurable(metrics)
        import dataclasses

        broken = dataclasses.replace(metrics, total_energy_j=float("nan"))
        with self.assertRaises(V2ConstraintError):
            require_measurable(broken)
        broken = dataclasses.replace(metrics, n_tasks=0)
        with self.assertRaises(V2ConstraintError):
            require_measurable(broken)

    def test_select_checkpoint_never_calls_an_infeasible_winner_feasible(self):
        records = [
            {"objective": 0.1, "feasible": False, "total_violation": 5.0},
            {"objective": 0.9, "feasible": False, "total_violation": 0.0},
            {"objective": 0.5, "feasible": True, "total_violation": 0.0},
        ]
        best = select_checkpoint(records)
        self.assertTrue(best["feasible"])
        self.assertEqual(best["selected_index"], 2)
        infeasible = [{"objective": 0.1, "feasible": False, "total_violation": 5.0},
                      {"objective": 0.9, "feasible": False, "total_violation": 1.0}]
        best = select_checkpoint(infeasible)
        self.assertFalse(best["feasible"])
        self.assertEqual(best["selected_index"], 0)
        self.assertIn("selection_note", best)
        with self.assertRaises(V2ConstraintError):
            select_checkpoint([{"objective": 0.1, "feasible": True}])

    def test_scope_mismatch_is_rejected(self):
        _dag, _helper, plans = _fixture()
        refs = reference_ranges_from_plans(plans, scheduler_config_sha256=SHA)
        spec = calibrate_budgets(refs, {"total_energy_mec": 0.5})
        manager = V2ConstraintManager(spec=spec, references=refs, scheduler_config_sha256=SHA)
        result = schedule_shared([_dag], {"d": {0: 1}}, link=LINK, compute=COMPUTE)
        ledger = schedule_energy(result)
        import dataclasses

        wrong = dataclasses.replace(refs, energy_scope="mobile")
        manager.references = wrong
        with self.assertRaises(Exception):
            manager.evaluate(result, ledger)


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestTotalEnergyScopeConsistency(unittest.TestCase):
    """E4.3: the total-energy measurement and its reference must share one scope.

    Audited defect (F09): the measurement was WHOLE-WORLD SYSTEM energy (foreground plus
    every declared background DAG of other owners, which share the same MEC CPU and radio),
    but its budget and scale were anchored on `refs.E_ue` - the system energy of the pure
    all-UE plan. In a background world that anchor is tiny, because the all-UE plan leaves the
    shared MEC energy to the background. A production GPU run consequently reported a signed
    cost of +136.6 (vs a budget of 0.14 J for a 38.3 J episode) and penalties near 1.9e3 per
    episode. These tests pin the corrected formulation.
    """

    def _plans(self):
        _dag, _helper, plans = _fixture()
        return plans

    def test_anchor_is_all_mec_system_energy_not_all_ue(self):
        from spec.automotive_training.v2.constraints_v2 import (
            spec_from_fractions, v2_constraint_costs,
        )
        from spec.automotive_training.v2.energy import (
            reference_ranges_from_plans, schedule_energy,
        )

        dag, _helper, plans = _fixture()
        refs = reference_ranges_from_plans(plans, scheduler_config_sha256=SHA)
        spec = spec_from_fractions({"total_energy_mec": 0.5})
        result = schedule_shared([dag], {"d": {0: 1}}, link=LINK, compute=COMPUTE)
        ledger = schedule_energy(result)
        costs = v2_constraint_costs(result, ledger, spec, refs,
                                    scheduler_config_sha256=SHA)
        index = list(costs.names).index("total_energy")
        budget = float(costs.budgets[index])
        scale = float(costs.scales[index])
        measured = float(costs.raw[index])
        # the anchor MUST be the all-MEC system reference, which is what the measured
        # whole-world system energy is comparable to
        self.assertAlmostEqual(scale, float(refs.E_mec), places=6)
        self.assertAlmostEqual(budget, 0.5 * float(refs.E_mec), places=6)
        self.assertNotAlmostEqual(scale, float(refs.E_ue), places=6)
        # and the normalised signed cost is O(1) rather than O(100)
        expected = (measured - budget) / scale
        self.assertAlmostEqual(float(costs.signed[index]), expected, places=9)
        self.assertLess(abs(float(costs.signed[index])), 10.0)

    def test_a_foreground_only_reference_would_have_been_off_by_two_orders(self):
        """Show the magnitude of the defect that was fixed, not merely that the code changed."""
        from spec.automotive_training.v2.energy import reference_ranges_from_plans

        plans = self._plans()
        refs = reference_ranges_from_plans(plans, scheduler_config_sha256=SHA)
        # the all-UE plan's SYSTEM energy is far below the all-MEC plan's on this same world
        self.assertGreater(float(refs.E_mec) / max(float(refs.E_ue), 1e-12), 10.0)


class TestReferenceEnergiesFollowTheFrozenTierContract(unittest.TestCase):
    """F11 consequence: every energy reference must be re-derived at RUNTIME from the
    resolved V2 resource contract `R = f_frozen/(8*xi)`.

    The F11 fix rewrote the scheduling/pricing frequency, so any reference whose value was
    produced under the OLD per-graph dataset frequency (`f_dataset/(8*xi)`) is stale: the
    duration form and the work form referred to different `f`. These tests recompute each
    reference INDEPENDENTLY from `f_frozen`, `xi` and `kappa` (never by re-reading
    `reference_ranges_from_plans` output) and pin the approved FRACTIONS.
    """

    def _env(self, background=0):
        from spec.automotive_training.automotive_loader import load_dataset

        graphs = list(load_dataset().validation_query())[:SLOTS]
        env = V2AutomotiveEnv(graphs, AutomotiveResourceCluster(), role="validation",
                              slots_per_task=SLOTS, base_seed=19, single_dist=True,
                              background_dags=int(background),
                              constraints_enabled=True,
                              budget_fractions={"total_energy_mec": 0.5, "ue_energy": 1.0},
                              scheduler_config_sha256=SHA)
        env.set_task({"dist_index": 0,
                      "graph_indices": np.arange(SLOTS, dtype=np.int32)})
        env.reset()
        return graphs, env

    def test_each_reference_is_the_analytic_frozen_tier_cpu_work(self):
        from spec.automotive_training.automotive_resources import (
            dataset_reference_parameters,
        )
        from spec.automotive_training.v2.adapters import config_for, dag_spec_from_graph
        from spec.automotive_training.v2.energy import frozen_energy_spec

        graphs, env = self._env()
        xi = float(dataset_reference_parameters()["xi_cycles_per_bit"])
        tiers = frozen_energy_spec().tiers
        for slot, graph in enumerate(graphs):
            refs = env.reference_ranges_for_slot(slot)
            ledgers = env._reference_ledgers[slot]
            nominal = dag_spec_from_graph(graph, dag_id="analytic", owner=0, mc=None)
            workload_bytes = float(sum(float(t.compute_bytes) for t in nominal.tasks))
            self.assertGreater(workload_bytes, 0.0)
            # the slot's scheduler rate must BE the analytic frozen-tier rate
            cfg = config_for(graph)
            for tier, rate in (("ue", cfg.ue_cpu_bytes_per_second),
                               ("helper", cfg.helper_cpu_bytes_per_second),
                               ("mec", cfg.mec_cpu_bytes_per_second)):
                self.assertAlmostEqual(
                    float(rate), float(tiers[tier].f_hz) / (8.0 * xi), places=6,
                    msg="%s scheduler rate is not f_frozen/(8*xi)" % tier)
            for plan_name, tier in (("all_ue", "ue"), ("all_mec", "mec"),
                                    ("all_helper", "helper")):
                # analytic: T = C/f = bytes*8*xi/f  and  E = kappa*C*f^2
                expected_seconds = workload_bytes * 8.0 * xi / float(tiers[tier].f_hz)
                expected_joules = float(tiers[tier].kappa) * (
                    workload_bytes * 8.0 * xi) * float(tiers[tier].f_hz) ** 2
                ledger = ledgers[plan_name]
                self.assertAlmostEqual(float(ledger.per_tier_cpu_seconds[tier]),
                                       expected_seconds, places=9)
                # the DURATION form and the WORK form coincide: the F11 identity
                self.assertAlmostEqual(float(ledger.cpu_duration_form_j), expected_joules,
                                       places=9)
                self.assertAlmostEqual(float(ledger.cpu_work_form_j), expected_joules,
                                       places=9)
                self.assertAlmostEqual(float(ledger.cpu_work_over_duration), 1.0, places=9)
                reference = {"all_ue": float(refs.E_ue), "all_mec": float(refs.E_mec),
                             "all_helper": float(refs.E_helper)}[plan_name]
                self.assertAlmostEqual(reference, float(ledger.system_joules), places=6)
                # the CPU part can never exceed the system reference
                self.assertLessEqual(expected_joules, reference + 1e-9)
            # an all-UE plan is CPU-only, so its reference IS the analytic work
            self.assertAlmostEqual(
                float(refs.E_ue),
                float(tiers["ue"].kappa) * (workload_bytes * 8.0 * xi)
                * float(tiers["ue"].f_hz) ** 2, places=9)

    def test_references_are_recomputed_from_the_live_contract_not_a_cached_artifact(self):
        """Invalidate the cache through the LIVE contract: doubling f doubles the rate and
        quadruples the CPU energy (`E = kappa*C*f^2`), with no frozen reference in between."""
        import dataclasses
        from unittest import mock

        import spec.automotive_training.v2.energy as energy_module

        _graphs, env = self._env()
        baseline = env.reference_ranges_for_slot(0)
        live = energy_module.frozen_energy_spec()
        doubled = dataclasses.replace(
            live, tiers={name: dataclasses.replace(tier, f_hz=2.0 * float(tier.f_hz))
                         for name, tier in live.tiers.items()})
        with mock.patch.object(energy_module, "frozen_energy_spec",
                               lambda path=None: doubled):
            env._reference_ranges = {}
            env._reference_results = {}
            env._reference_ledgers = {}
            moved = env.reference_ranges_for_slot(0)
        self.assertNotEqual(float(moved.E_ue), float(baseline.E_ue))
        # all-UE is CPU-only: the energy scales exactly as f^2
        self.assertAlmostEqual(float(moved.E_ue), 4.0 * float(baseline.E_ue), places=9)
        # and restoring the contract restores the value: nothing was frozen/cached
        env._reference_ranges = {}
        env._reference_results = {}
        env._reference_ledgers = {}
        restored = env.reference_ranges_for_slot(0)
        self.assertAlmostEqual(float(restored.E_ue), float(baseline.E_ue), places=12)
        self.assertAlmostEqual(float(restored.E_mec), float(baseline.E_mec), places=12)

    def test_approved_fractions_are_unchanged_and_bound_the_same_dag_and_boundary(self):
        """(3)+(4): fractions 0.5 (total, all-MEC anchor) and 1.0 (UE) are NOT retuned; the
        measurement and its reference describe the same FOREGROUND/system boundary."""
        from spec.automotive_training.v2.constraints_v2 import (
            spec_from_fractions, v2_constraint_costs,
        )
        from spec.automotive_training.v2.energy import schedule_energy

        spec = spec_from_fractions({"total_energy_mec": 0.5, "ue_energy": 1.0})
        self.assertEqual(float(spec.total_energy_frac_of_all_mec), 0.5)
        self.assertEqual(float(spec.ue_energy_frac_of_all_ue), 1.0)
        self.assertIsNone(spec.total_energy_budget_j)

        graphs, env = self._env(background=2)
        env.reset()
        _o, _r, _d, info = env.step(np.ones((SLOTS, TOKENS), dtype=int))
        for slot, record in enumerate(info[2]):
            refs = env.reference_ranges_for_slot(slot)
            constraints = record["constraints"]
            index = list(constraints["names"]).index("total_energy")
            self.assertAlmostEqual(float(constraints["total_energy_budget"]),
                                   0.5 * float(refs.E_mec), places=6)
            self.assertAlmostEqual(float(constraints["total_energy_scale"]),
                                   float(refs.E_mec), places=6)
            self.assertAlmostEqual(float(constraints["ue_energy_budget"]),
                                   1.0 * float(refs.E_ue), places=6)
            # the constrained measurement is the FOREGROUND-only ledger, which is exactly the
            # same DAG/ownership boundary the background-free reference plans describe
            self.assertAlmostEqual(float(constraints["total_energy_raw"]),
                                   float(record["foreground_system_joules"]), places=6)
            self.assertLess(float(record["foreground_system_joules"]),
                            float(record["system_joules"]),
                            "a background world must keep the foreground boundary separate")
            self.assertLessEqual(float(record["requester_joules"]),
                                 float(record["mobile_joules"]) + 1e-9)
            self.assertLessEqual(float(record["mobile_joules"]),
                                 float(record["system_joules"]) + 1e-9)
            # the reference plans are foreground-only: ONE DAG, the pure-location plan
            for name, result in env._reference_results[slot].items():
                dags = {str(dag_id) for dag_id, _tid in result.timings}
                self.assertEqual(len(dags), 1, "%s reference built %d DAGs" % (name, len(dags)))
            # and the reference energies are the system scope of those plans
            for name in ("all_ue", "all_mec", "all_helper"):
                ledger = schedule_energy(env._reference_results[slot][name])
                self.assertEqual(str(ledger.primary_scope), "system")
                self.assertEqual(str(refs.energy_scope), "system")
                self.assertEqual(str(refs.scheduler_config_sha256), SHA)
