#!/usr/bin/env python3
"""②B tests: plan objective, cost channels, lexicographic selection, mask API."""

from __future__ import annotations

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

from env.mec_offloaing_envs.scheduler import (  # noqa: E402
    CanonicalDAG,
    CanonicalTask,
    ReferenceRanges,
    ResourceConfig,
    schedule,
)
from env.mec_offloaing_envs.scheduler.model import Location  # noqa: E402
from env.mec_offloaing_envs.scheduler.feasibility import (  # noqa: E402
    FeasibilityContext,
    ParentInput,
    all_actions_masked,
    feasible_actions,
    finish_lower_bound,
    heuristic_failure_must_not_mask,
    mask_vector,
    optimistic_lower_bound_ready,
    start_lower_bound,
    transfer_lower_bound,
)
from env.mec_offloaing_envs.scheduler.objective import (  # noqa: E402
    LATENCY_REF_ALL_MEC,
    LATENCY_REF_ALL_UE,
    LATENCY_REF_FIXED,
    SELECTION_METRIC_NAME,
    CheckpointCandidate,
    ObjectiveSpec,
    PlanObjective,
    choose_checkpoint,
    evaluate_plan_objective,
    objective_log_kvs,
    selection_key,
)

ORDER = [0, 1]
ACTIONS = [1, 0]
WORKLOAD = 4_166_700
BIG_OUTPUT = 4_166_700


def _resources(physical: bool = True):
    from env.mec_offloaing_envs.scheduler.energy_model import (
        MODEL_PHYSICAL,
        SCOPE_SYSTEM,
        EnergyModelSpec,
    )
    from env.mec_offloaing_envs.scheduler.resources import TIMING_LEGACY, TIMING_PHYSICAL

    spec = None
    if physical:
        spec = EnergyModelSpec.from_dict(
            {
                "model": MODEL_PHYSICAL,
                "energy_scope": SCOPE_SYSTEM,
                "cycles_per_bit": 300.0,
                "tiers": {
                    "ue": {"f_hz": 1.0e9, "kappa": 1.0e-27},
                    "helper": {"f_hz": 1.5e9, "kappa": 5.0e-27},
                    "mec": {"f_hz": 10.0e9, "kappa": 1.0e-27},
                },
                "radio": {"ue_tx_w": 1.0, "mec_tx_w": 3.162, "helper_tx_w": 1.0},
            }
        )
    base = ResourceConfig.from_frozen_yaml(model="legacy")
    return ResourceConfig(
        ue_cpu_bytes_per_second=base.ue_cpu_bytes_per_second,
        mec_cpu_bytes_per_second=base.mec_cpu_bytes_per_second,
        helper_cpu_bytes_per_second=base.helper_cpu_bytes_per_second,
        mec_uplink_bytes_per_second=base.mec_uplink_bytes_per_second,
        mec_downlink_bytes_per_second=base.mec_downlink_bytes_per_second,
        v2v_bytes_per_second=base.v2v_bytes_per_second,
        rho_ue=base.rho_ue,
        f_l=base.f_l,
        zeta=base.zeta,
        ptx_mec_w=base.ptx_mec_w,
        prx_mec_w=base.prx_mec_w,
        ptx_v2v_w=base.ptx_v2v_w,
        prx_v2v_w=base.prx_v2v_w,
        rho_helper=base.rho_helper,
        f_v2v=base.f_v2v,
        timing_model=TIMING_PHYSICAL if spec is not None else TIMING_LEGACY,
        radio_timing_model=TIMING_LEGACY,
        timing_tiers=spec,
        energy_model=spec,
    )


def _refs_from_dag(dag, resources, order):
    """Reference ranges for a CanonicalDAG (energy_api needs a legacy graph)."""
    metrics = {}
    for action in (0, 1, 2):
        out = schedule(dag, order, [action] * len(order), resources)
        metrics[action] = (out.makespan_seconds, out.energy.total_mobile_joules)
    return ReferenceRanges(
        L_ue=metrics[0][0], L_mec=metrics[1][0], L_helper=metrics[2][0],
        E_ue=metrics[0][1], E_mec=metrics[1][1], E_helper=metrics[2][1],
    )


def _dag(deadline0=None, dtype="none", deadline1=None):
    tasks = [
        CanonicalTask(
            task_id=0,
            compute_workload_bytes=WORKLOAD,
            task_output_bytes=BIG_OUTPUT,
            deadline_s=deadline0,
            deadline_type=dtype,
        ),
        CanonicalTask(
            task_id=1,
            compute_workload_bytes=1024,
            task_output_bytes=512,
            deadline_s=deadline1,
            deadline_type=dtype if deadline1 is not None else "none",
        ),
    ]
    return CanonicalDAG.from_records(tasks, [(0, 1, BIG_OUTPUT)])


class TestObjective(unittest.TestCase):
    def setUp(self):
        self.res = _resources(physical=True)
        self.refs = _refs_from_dag(_dag(), self.res, ORDER)
        self.result = schedule(_dag(), ORDER, ACTIONS, self.res)

    def test_J_is_latency_norm_plus_beta_soft_tardiness(self):
        spec = ObjectiveSpec(
            latency_ref=LATENCY_REF_ALL_UE,
            beta_soft=2.0,
            energy_budget_frac_of_all_ue=1.0,
        )
        obj = evaluate_plan_objective(self.result, self.refs, spec)
        self.assertAlmostEqual(obj.latency_norm, obj.latency_s / self.refs.L_ue, places=9)
        self.assertAlmostEqual(
            obj.J, obj.latency_norm + 2.0 * obj.soft_tardiness_norm, places=9
        )
        self.assertEqual(obj.soft_tardiness_norm, 0.0)   # no deadlines attached
        self.assertAlmostEqual(obj.J, obj.latency_norm, places=12)

    def test_latency_ref_choices(self):
        for ref, expected in (
            (LATENCY_REF_ALL_UE, self.refs.L_ue),
            (LATENCY_REF_ALL_MEC, self.refs.L_mec),
        ):
            spec = ObjectiveSpec(
                latency_ref=ref, energy_budget_frac_of_all_ue=1.0
            )
            obj = evaluate_plan_objective(self.result, self.refs, spec)
            self.assertAlmostEqual(obj.latency_ref_s, expected, places=6)
        spec = ObjectiveSpec(
            latency_ref=LATENCY_REF_FIXED,
            latency_denominator_s=100.0,
            energy_budget_frac_of_all_ue=1.0,
        )
        obj = evaluate_plan_objective(self.result, self.refs, spec)
        self.assertAlmostEqual(obj.latency_ref_s, 100.0, places=9)

    def test_energy_channel_is_capped_and_monotone(self):
        tight = ObjectiveSpec(
            latency_ref=LATENCY_REF_ALL_UE,
            energy_budget_frac_of_all_ue=0.001,
        )
        loose = ObjectiveSpec(
            latency_ref=LATENCY_REF_ALL_UE,
            energy_budget_j=1e12,
        )
        o_tight = evaluate_plan_objective(self.result, self.refs, tight)
        o_loose = evaluate_plan_objective(self.result, self.refs, loose)
        self.assertGreater(o_tight.c_E, 0.0)
        self.assertEqual(o_loose.c_E, 0.0)
        self.assertLessEqual(o_tight.c_E, tight.cost_cap)
        self.assertFalse(o_tight.energy_ok)
        self.assertTrue(o_loose.energy_ok)

    def test_hard_and_firm_channels(self):
        base = schedule(_dag(), ORDER, ACTIONS, self.res)
        deadline = 0.5 * base.tasks[0].availability_seconds
        hard = schedule(_dag(deadline0=deadline, dtype="hard"), ORDER, ACTIONS, self.res)
        spec = ObjectiveSpec(
            latency_ref=LATENCY_REF_ALL_UE, energy_budget_frac_of_all_ue=100.0
        )
        obj = evaluate_plan_objective(hard, self.refs, spec)
        self.assertEqual(obj.hard_miss_count, 1)
        self.assertEqual(obj.hard_task_count, 1)
        self.assertAlmostEqual(obj.c_H, 1.0, places=9)
        self.assertFalse(obj.hard_ok)
        self.assertFalse(obj.feasible)

        firm = schedule(_dag(deadline0=deadline, dtype="firm"), ORDER, ACTIONS, self.res)
        obj_f = evaluate_plan_objective(firm, self.refs, spec)
        self.assertEqual(obj_f.c_F, 1.0)
        self.assertEqual(obj_f.c_H, 0.0)
        self.assertFalse(obj_f.firm_ok)
        lenient = ObjectiveSpec(
            latency_ref=LATENCY_REF_ALL_UE,
            energy_budget_j=1e12,
            firm_miss_epsilon=1.0,
        )
        self.assertTrue(evaluate_plan_objective(firm, self.refs, lenient).firm_ok)

    def test_soft_tardiness_enters_J(self):
        base = schedule(_dag(), ORDER, ACTIONS, self.res)
        deadline = 0.5 * base.tasks[0].availability_seconds
        soft = schedule(_dag(deadline0=deadline, dtype="soft"), ORDER, ACTIONS, self.res)
        spec = ObjectiveSpec(
            latency_ref=LATENCY_REF_ALL_UE,
            beta_soft=1.0,
            energy_budget_j=1e12,
        )
        obj = evaluate_plan_objective(soft, self.refs, spec)
        self.assertGreater(obj.soft_tardiness_norm, 0.0)
        self.assertAlmostEqual(
            obj.J, obj.latency_norm + obj.soft_tardiness_norm, places=9
        )

    def test_log_kvs_have_all_channels(self):
        spec = ObjectiveSpec(
            latency_ref=LATENCY_REF_ALL_UE, energy_budget_frac_of_all_ue=1.0
        )
        kvs = objective_log_kvs(evaluate_plan_objective(self.result, self.refs, spec))
        for key in ("objective/J", "objective/c_E", "objective/c_H", "objective/c_F",
                    "objective/feasible", "objective/total_violation"):
            self.assertIn(key, kvs)

    def test_spec_validation(self):
        with self.assertRaises(ValueError):
            ObjectiveSpec(latency_ref="bogus", energy_budget_frac_of_all_ue=1.0)
        with self.assertRaises(ValueError):
            ObjectiveSpec(latency_ref=LATENCY_REF_FIXED, energy_budget_frac_of_all_ue=1.0)
        with self.assertRaises(ValueError):
            ObjectiveSpec(latency_ref=LATENCY_REF_ALL_UE)   # no energy budget
        with self.assertRaises(ValueError):
            ObjectiveSpec.from_dict({"latency_ref": "all_ue", "nope": 1})
        # hardening: XOR budget, strictly positive, hard epsilon frozen at 0
        with self.assertRaises(ValueError):
            ObjectiveSpec(
                latency_ref=LATENCY_REF_ALL_UE,
                energy_budget_j=10.0,
                energy_budget_frac_of_all_ue=0.5,
            )
        with self.assertRaises(ValueError):
            ObjectiveSpec(latency_ref=LATENCY_REF_ALL_UE, energy_budget_j=0.0)
        with self.assertRaises(ValueError):
            ObjectiveSpec(
                latency_ref=LATENCY_REF_ALL_UE,
                energy_budget_j=10.0,
                hard_miss_epsilon=0.05,
            )
        with self.assertRaises(NotImplementedError):
            ObjectiveSpec(
                latency_ref=LATENCY_REF_ALL_UE,
                energy_budget_j=10.0,
                chance_hard_epsilon=0.05,
            )

    def test_c_E_raw_is_reported_next_to_capped(self):
        spec = ObjectiveSpec(latency_ref=LATENCY_REF_ALL_UE, energy_budget_j=1e-6)
        obj = evaluate_plan_objective(self.result, self.refs, spec)
        self.assertGreater(obj.c_E_raw, spec.cost_cap)
        self.assertAlmostEqual(obj.c_E, spec.cost_cap, places=9)


class TestLexicographicSelection(unittest.TestCase):
    def _obj(self, *, J, c_E=0.0, c_H=0.0, c_F=0.0, eps_F=0.0):
        return PlanObjective(
            latency_s=J, latency_ref_s=1.0, latency_norm=J,
            soft_tardiness_s=0.0, soft_tardiness_norm=0.0, beta_soft=1.0, J=J,
            energy_system_j=1.0, energy_budget_j=1.0, c_E=c_E,
            hard_miss_count=1 if c_H > 0 else 0, hard_task_count=1, c_H=c_H,
            firm_miss_count=1 if c_F > 0 else 0, firm_task_count=1, c_F=c_F,
            hard_miss_epsilon=0.0, firm_miss_epsilon=eps_F,
        )

    def test_metric_name(self):
        self.assertEqual(SELECTION_METRIC_NAME, "lexicographic_feasible_then_J")

    def test_feasible_beats_better_J(self):
        """A slightly worse J with no violations wins over a great J that misses."""
        great_but_infeasible = CheckpointCandidate("great", self._obj(J=0.10, c_H=1.0))
        modest_feasible = CheckpointCandidate("modest", self._obj(J=0.40))
        choice = choose_checkpoint([great_but_infeasible, modest_feasible])
        self.assertTrue(choice.feasible)
        self.assertEqual(choice.winner, "modest")
        self.assertEqual(choice.ranked, ["modest", "great"])

    def test_energy_and_firm_gate(self):
        a = CheckpointCandidate("a", self._obj(J=0.2, c_E=0.3))
        b = CheckpointCandidate("b", self._obj(J=0.3))
        self.assertEqual(choose_checkpoint([a, b]).winner, "b")
        c = CheckpointCandidate("c", self._obj(J=0.25, c_F=0.5))
        d = CheckpointCandidate("d", self._obj(J=0.5))
        self.assertEqual(choose_checkpoint([c, d]).winner, "d")

    def test_none_feasible_reports_infeasible_without_winner(self):
        a = CheckpointCandidate("a", self._obj(J=0.1, c_H=1.0))
        b = CheckpointCandidate("b", self._obj(J=0.2, c_E=0.9))
        choice = choose_checkpoint([a, b])
        self.assertFalse(choice.feasible)
        self.assertIsNone(choice.winner)
        self.assertIn("NOT a scientific winner", choice.note)
        self.assertIsNotNone(choice.best_infeasible)

    def test_empty_candidate_list(self):
        choice = choose_checkpoint([])
        self.assertIsNone(choice.winner)
        self.assertFalse(choice.feasible)

    def test_selection_key_is_stable(self):
        feasible = self._obj(J=5.0)
        infeasible = self._obj(J=1.0, c_H=1.0)
        self.assertLess(selection_key(feasible), selection_key(infeasible))


class TestFeasibilityMask(unittest.TestCase):
    """Sound optimistic bound: parent-by-parent, parallel transfers, no phantom
    delivery for non-sinks (the ②B blocker fix)."""

    def setUp(self):
        self.res = _resources(physical=True)
        self.mec = Location.MEC
        self.ue = Location.UE

    def _ctx(self, **kw):
        base = dict(parents=(), deadline_s=None, is_sink=False)
        base.update(kw)
        return FeasibilityContext(**base)

    def test_no_deadline_masks_nothing(self):
        feas = feasible_actions(self._ctx(), self.res, workload_bytes=WORKLOAD)
        self.assertTrue(all(f.mask for f in feas))
        self.assertTrue(all(f.reason == "no_deadline" for f in feas))
        self.assertEqual(mask_vector(feas), [True, True, True])

    def test_impossible_deadline_masks_all(self):
        ctx = self._ctx(deadline_s=1e-6)
        feas = feasible_actions(ctx, self.res, workload_bytes=WORKLOAD)
        self.assertTrue(all_actions_masked(feas))
        self.assertTrue(all(f.reason == "lb_exceeds_deadline" for f in feas))
        self.assertEqual(mask_vector(feas), [False, False, False])

    def test_non_sink_has_no_delivery_term(self):
        """No phantom downlink: co-located successors can be free."""
        ctx = self._ctx(is_sink=False)
        for action in (0, 1, 2):
            ready = optimistic_lower_bound_ready(
                action, ctx, self.res, workload_bytes=WORKLOAD
            )
            finish = finish_lower_bound(
                action, ctx, self.res, workload_bytes=WORKLOAD
            )
            self.assertAlmostEqual(ready, finish, places=12)

    def test_parents_transfer_in_parallel_not_serialized(self):
        """max over parents, never a sum of inbound bytes."""
        big = ParentInput(source_location=self.mec, source_ready_s=5.0, bytes=4_000_000)
        ctx_one = self._ctx(parents=(big,))
        ctx_two = self._ctx(parents=(big, big))
        one = start_lower_bound(1, ctx_one, self.res)
        two = start_lower_bound(1, ctx_two, self.res)
        self.assertAlmostEqual(one, two, places=12)
        # and strictly below a serialized sum
        serial = 2.0 * one
        self.assertLess(two, serial - 1e-9)

    def test_start_uses_parent_compute_finish_only(self):
        """A parent's all-consumers-ready time is NOT required to start this child."""
        fast = ParentInput(source_location=self.mec, source_ready_s=1.0, bytes=1024)
        ctx = self._ctx(parents=(fast,))
        start = start_lower_bound(1, ctx, self.res)
        # MEC->MEC transfer is free, so the bound is exactly the parent's finish
        self.assertAlmostEqual(start, 1.0, places=12)

    def test_sink_pays_the_unavoidable_return_hop(self):
        ctx_non_sink = self._ctx(is_sink=False)
        ctx_sink = self._ctx(is_sink=True, return_bytes=BIG_OUTPUT)
        for action in (1, 2):   # remote sinks must return to the UE
            a = optimistic_lower_bound_ready(
                action, ctx_non_sink, self.res, workload_bytes=WORKLOAD
            )
            b = optimistic_lower_bound_ready(
                action, ctx_sink, self.res, workload_bytes=WORKLOAD
            )
            self.assertGreater(b, a)
        # a UE sink has no return hop
        self.assertAlmostEqual(
            optimistic_lower_bound_ready(0, ctx_non_sink, self.res, workload_bytes=WORKLOAD),
            optimistic_lower_bound_ready(0, ctx_sink, self.res, workload_bytes=WORKLOAD),
            places=12,
        )

    def test_transfer_lower_bound_is_the_route_sum(self):
        self.assertEqual(transfer_lower_bound(0, self.mec, self.ue, self.res), 0.0)
        self.assertEqual(transfer_lower_bound(1000, self.mec, self.mec, self.res), 0.0)
        dl = transfer_lower_bound(1000, self.mec, self.ue, self.res)
        self.assertAlmostEqual(
            dl, 1000 / self.res.mec_downlink_bytes_per_second, places=12
        )
        # MEC -> HELPER is a two-hop route: MEC_DL then V2V (sum, not min)
        two_hop = transfer_lower_bound(1000, self.mec, Location.HELPER, self.res)
        expected = (
            1000 / self.res.mec_downlink_bytes_per_second
            + 1000 / self.res.v2v_bytes_per_second
        )
        self.assertAlmostEqual(two_hop, expected, places=12)

    def test_masked_action_is_infeasible_for_every_sampled_suffix(self):
        """Stronger soundness check: no sampled suffix can rescue a masked action.

        The mask must hold for the whole family of continuations, not just for one
        test suffix.  We enumerate every pure/MEC/HELPER placement of the
        successors and require that the achieved availability stays above the
        deadline for every one of them.
        """
        import itertools

        # A -> {B, C}; A's deadline is checked on all-consumers-ready
        def build(dtype="hard", deadline=None):
            tasks = [
                CanonicalTask(
                    task_id=0, compute_workload_bytes=WORKLOAD,
                    task_output_bytes=BIG_OUTPUT, deadline_s=deadline,
                    deadline_type=dtype,
                ),
                CanonicalTask(task_id=1, compute_workload_bytes=1024, task_output_bytes=256),
                CanonicalTask(task_id=2, compute_workload_bytes=1024, task_output_bytes=256),
            ]
            return CanonicalDAG.from_records(tasks, [(0, 1, BIG_OUTPUT), (0, 2, BIG_OUTPUT)])

        dag = build()
        order = [0, 1, 2]
        # find a deadline that masks at least one action given parents=(none)
        ctx_free = self._ctx(deadline_s=0.0)
        feas = feasible_actions(ctx_free, self.res, workload_bytes=WORKLOAD)
        masked = {f.action for f in feas if not f.mask}
        self.assertTrue(masked, "no action masked at deadline 0")
        for action in masked:
            for suffix in itertools.product((0, 1, 2), repeat=2):
                actions = [action, suffix[0], suffix[1]]
                out = schedule(build(deadline=0.0), order, actions, self.res)
                self.assertGreater(
                    out.tasks[0].availability_seconds, 0.0,
                    msg="masked action %d rescued by suffix %s" % (action, suffix),
                )

    def test_lower_bound_never_exceeds_sampled_schedules(self):
        """LB <= achieved availability for every sampled continuation."""
        import itertools

        dag = CanonicalDAG.from_records(
            [
                CanonicalTask(
                    task_id=0, compute_workload_bytes=WORKLOAD,
                    task_output_bytes=BIG_OUTPUT,
                ),
                CanonicalTask(task_id=1, compute_workload_bytes=1024, task_output_bytes=256),
            ],
            [(0, 1, BIG_OUTPUT)],
        )
        for action in (0, 1, 2):
            ctx = self._ctx(is_sink=False)
            lb = optimistic_lower_bound_ready(
                action, ctx, self.res, workload_bytes=WORKLOAD
            )
            for suffix in itertools.product((0, 1, 2), repeat=1):
                out = schedule(dag, [0, 1], [action, suffix[0]], self.res)
                achieved = out.tasks[0].availability_seconds
                self.assertLessEqual(
                    lb, achieved + 1e-9 * max(1.0, abs(achieved)),
                    msg="LB above achieved availability for action %d suffix %s"
                    % (action, suffix),
                )

    def test_heuristic_failure_must_not_mask(self):
        self.assertEqual(
            heuristic_failure_must_not_mask(heuristic_found=False),
            "heuristic_only_not_used",
        )
        self.assertEqual(
            heuristic_failure_must_not_mask(heuristic_found=True), "not_proven_infeasible"
        )


class TestPerGraphAggregation(unittest.TestCase):
    """②B blocker #2: per-graph objective/constraints, then aggregate.

    Ratio-of-means is not allowed: mean energy under a mean budget can hide a
    per-episode violation, and E[L]/E[L_ref] != E[L_g/L_ref,g].
    """

    def setUp(self):
        self.res = _resources(physical=True)
        self.base = _dag()
        self.refs = _refs_from_dag(self.base, self.res, ORDER)
        self.result = schedule(self.base, ORDER, ACTIONS, self.res)

    def _plans(self, n=3, deadline=None, dtype="none"):
        out = []
        for _ in range(n):
            dag = _dag(deadline0=deadline, dtype=dtype)
            out.append((schedule(dag, ORDER, ACTIONS, self.res), self.refs))
        return out

    def test_J_is_mean_of_per_graph_J(self):
        from spec.objective_selection import objective_from_plans

        spec = ObjectiveSpec(latency_ref=LATENCY_REF_ALL_UE, energy_budget_j=1e12)
        agg = objective_from_plans(self._plans(3), spec)
        self.assertEqual(agg.n_graphs, 3)
        manual = sum(o.J for o in agg.per_graph) / 3.0
        self.assertAlmostEqual(agg.J, manual, places=12)
        # identical graphs -> mean equals the single-graph value
        self.assertAlmostEqual(agg.J, agg.per_graph[0].J, places=12)

    def test_grid_mean_differs_from_ratio_of_means(self):
        """Pins the bias the blocker was about."""
        from spec.objective_selection import aggregate_plan_objectives

        spec = ObjectiveSpec(latency_ref=LATENCY_REF_ALL_UE, energy_budget_j=1e12)
        objs = []
        for scale in (1.0, 4.0):
            refs_scaled = ReferenceRanges(
                L_ue=self.refs.L_ue * scale,
                L_mec=self.refs.L_mec * scale,
                L_helper=self.refs.L_helper * scale,
                E_ue=self.refs.E_ue,
                E_mec=self.refs.E_mec,
                E_helper=self.refs.E_helper,
            )
            objs.append(evaluate_plan_objective(self.result, refs_scaled, spec))
        agg = aggregate_plan_objectives(objs, spec)
        mean_of_ratios = sum(o.latency_norm for o in objs) / 2.0
        ratio_of_means = self.result.makespan_seconds / (
            (self.refs.L_ue * 1.0 + self.refs.L_ue * 4.0) / 2.0
        )
        self.assertAlmostEqual(agg.latency_norm_mean, mean_of_ratios, places=12)
        self.assertNotAlmostEqual(mean_of_ratios, ratio_of_means, places=6)

    def test_per_episode_energy_violation_is_not_hidden_by_averages(self):
        """Graph A over budget, graph B far under -> mean looks fine, gate must fail."""
        from spec.objective_selection import aggregate_plan_objectives

        # budget per graph via references: graph A tight, graph B huge
        tight = ReferenceRanges(
            L_ue=self.refs.L_ue, L_mec=self.refs.L_mec, L_helper=self.refs.L_helper,
            E_ue=1e-9, E_mec=1e-9, E_helper=1e-9,
        )
        loose = ReferenceRanges(
            L_ue=self.refs.L_ue, L_mec=self.refs.L_mec, L_helper=self.refs.L_helper,
            E_ue=1e9, E_mec=1e9, E_helper=1e9,
        )
        spec = ObjectiveSpec(
            latency_ref=LATENCY_REF_ALL_UE, energy_budget_frac_of_all_ue=0.5
        )
        objs = [
            evaluate_plan_objective(self.result, tight, spec),
            evaluate_plan_objective(self.result, loose, spec),
        ]
        agg = aggregate_plan_objectives(objs, spec)
        mean_energy = sum(o.energy_system_j for o in objs) / 2.0
        mean_budget = sum(o.energy_budget_j for o in objs) / 2.0
        self.assertLessEqual(mean_energy, mean_budget)      # averages look fine
        self.assertGreater(agg.c_E_max, 0.0)                # but A violates
        self.assertAlmostEqual(agg.energy_violation_rate, 0.5, places=12)
        self.assertFalse(agg.energy_ok)
        self.assertFalse(agg.feasible)

    def test_task_counts_are_accumulated_not_inferred(self):
        """Two hard tasks per graph must count 2, not 1 per graph."""
        from spec.objective_selection import objective_from_plans

        spec = ObjectiveSpec(latency_ref=LATENCY_REF_ALL_UE, energy_budget_j=1e12)
        plans = []
        for _ in range(2):
            dag = CanonicalDAG.from_records(
                [
                    CanonicalTask(
                        task_id=0, compute_workload_bytes=WORKLOAD,
                        task_output_bytes=BIG_OUTPUT, deadline_s=1e-6,
                        deadline_type="hard",
                    ),
                    CanonicalTask(
                        task_id=1, compute_workload_bytes=1024,
                        task_output_bytes=512, deadline_s=1e-6, deadline_type="hard",
                    ),
                ],
                [(0, 1, BIG_OUTPUT)],
            )
            plans.append((schedule(dag, ORDER, ACTIONS, self.res), self.refs))
        agg = objective_from_plans(plans, spec)
        self.assertEqual(agg.hard_task_total, 4)     # 2 graphs x 2 hard tasks
        self.assertEqual(agg.hard_miss_total, 4)
        self.assertAlmostEqual(agg.c_H, 1.0, places=12)
        self.assertFalse(agg.feasible)

    def test_firm_is_a_population_constraint(self):
        from spec.objective_selection import objective_from_plans

        spec = ObjectiveSpec(
            latency_ref=LATENCY_REF_ALL_UE, energy_budget_j=1e12,
            firm_miss_epsilon=1.0,
        )
        plans = self._plans(3, deadline=1e-6, dtype="firm")
        agg = objective_from_plans(plans, spec)
        self.assertAlmostEqual(agg.c_F, 1.0, places=12)
        self.assertTrue(agg.firm_ok)      # eps_F = 1.0 tolerates the population

    def test_selection_prefers_feasible_aggregate(self):
        from spec.objective_selection import (
            aggregate_plan_objectives,
            selection_decision,
        )

        spec = ObjectiveSpec(latency_ref=LATENCY_REF_ALL_UE, energy_budget_j=1e12)
        ok = aggregate_plan_objectives(
            [evaluate_plan_objective(self.result, self.refs, spec)], spec
        )
        tight_refs = ReferenceRanges(
            L_ue=self.refs.L_ue, L_mec=self.refs.L_mec, L_helper=self.refs.L_helper,
            E_ue=1e-9, E_mec=1e-9, E_helper=1e-9,
        )
        spec_e = ObjectiveSpec(
            latency_ref=LATENCY_REF_ALL_UE, energy_budget_frac_of_all_ue=0.5
        )
        bad = aggregate_plan_objectives(
            [evaluate_plan_objective(self.result, tight_refs, spec_e)], spec_e
        )
        decision = selection_decision([("bad", bad), ("ok", ok)])
        self.assertTrue(decision["feasible"])
        self.assertEqual(decision["winner"], "ok")

    def test_log_kvs_shape(self):
        from spec.objective_selection import objective_from_plans, objective_log_kvs

        spec = ObjectiveSpec(latency_ref=LATENCY_REF_ALL_UE, energy_budget_j=1e12)
        kvs = objective_log_kvs(objective_from_plans(self._plans(2), spec))
        for key in ("objective/J", "objective/energy_violation_rate",
                    "objective/c_E_max", "objective/c_E_raw_max",
                    "objective/hard_miss_total", "objective/feasible"):
            self.assertIn(key, kvs)


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestCandidatePanelBudget(unittest.TestCase):
    """③ support: B_E(rho) from a candidate panel, not from all-UE energy."""

    def test_endpoints_and_interpolation(self):
        from env.mec_offloaing_envs.scheduler.objective import candidate_panel_budget

        panel = [(500.0, 300.0), (600.0, 100.0), (700.0, 50.0)]  # (T, E_system)
        low = candidate_panel_budget(panel, 0.0)
        high = candidate_panel_budget(panel, 1.0)
        mid = candidate_panel_budget(panel, 0.5)
        self.assertAlmostEqual(low.e_min, 50.0, places=9)
        self.assertAlmostEqual(high.e_fast, 300.0, places=9)   # energy of fastest
        self.assertAlmostEqual(low.budget_j, 50.0, places=9)
        self.assertAlmostEqual(high.budget_j, 300.0, places=9)
        self.assertAlmostEqual(mid.budget_j, 175.0, places=9)

    def test_monotone_in_rho(self):
        from env.mec_offloaing_envs.scheduler.objective import candidate_panel_budget

        panel = [(500.0, 300.0), (700.0, 50.0)]
        values = [candidate_panel_budget(panel, r).budget_j for r in (0.0, 0.25, 0.5, 1.0)]
        self.assertEqual(values, sorted(values))

    def test_validation(self):
        from env.mec_offloaing_envs.scheduler.objective import candidate_panel_budget

        with self.assertRaises(ValueError):
            candidate_panel_budget([], 0.5)
        with self.assertRaises(ValueError):
            candidate_panel_budget([(1.0, 2.0)], 1.5)
        # a panel where the fastest plan is also the most expensive is legitimate
        self.assertGreater(
            candidate_panel_budget([(1.0, 10.0), (2.0, 5.0)], 0.0).budget_j, 0.0
        )
        with self.assertRaises(ValueError):
            candidate_panel_budget([(1.0, 5.0), (2.0, 10.0)], 0.5).__class__(
                e_min=10.0, e_fast=5.0, rho=0.5
            )
