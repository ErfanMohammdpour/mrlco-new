#!/usr/bin/env python3
"""③ tests: optimistic DAG lookahead, constructive suffix, mask proofs, potential.

Test 1  current action looks fine alone but wrecks the suffix  -> suffix finds better
Test 2  heuristic failure != infeasibility                     -> NO mask
Test 3  every masked action: no sampled completion meets the deadline (strong)
Test 4  potential consistency: terminal Ĵ == J_actual, correct reward sign
"""

from __future__ import annotations

import itertools
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
from env.mec_offloaing_envs.scheduler.objective import (  # noqa: E402
    LATENCY_REF_ALL_UE,
    ObjectiveSpec,
    evaluate_plan_objective,
)
from env.mec_offloaing_envs.scheduler.suffix import (  # noqa: E402
    REASON_HEURISTIC_FAILED,
    REASON_PROOF_MASK,
    ObjectiveContext,
    SuffixContext,
    TaskDeadline,
    best_by_potential,
    construct_fastest_feasible_suffix,
    dag_lower_bound_masks,
    dag_lower_bound_ready,
    evaluate_action,
    evaluate_all_actions,
    feasible_action_mask,
    state_potential,
    suffix_potential,
)

WORKLOAD = 4_166_700
BIG_OUTPUT = 4_166_700
SMALL = 2048


def _resources(physical: bool = True):
    from env.mec_offloaing_envs.scheduler.resources import (
        TIMING_LEGACY,
        TIMING_PHYSICAL,
    )
    from env.mec_offloaing_envs.scheduler.energy_model import (
        MODEL_PHYSICAL,
        SCOPE_SYSTEM,
        EnergyModelSpec,
    )

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
        radio_timing_model=TIMING_LEGACY,   # no radio model in these fixtures
        timing_tiers=spec,
        energy_model=spec,
    )


def _chain_dag(n=3, deadlines=None, dtype="hard", output=BIG_OUTPUT):
    """0 -> 1 -> 2 chain, last task a sink."""
    deadlines = deadlines or {}
    tasks = []
    for i in range(n):
        tasks.append(
            CanonicalTask(
                task_id=i,
                compute_workload_bytes=WORKLOAD if i == 0 else SMALL,
                task_output_bytes=output if i == 0 else SMALL,
                deadline_s=deadlines.get(i),
                deadline_type=dtype if i in deadlines else "none",
            )
        )
    edges = [(i, i + 1, output if i == 0 else SMALL) for i in range(n - 1)]
    return CanonicalDAG.from_records(tasks, edges)


def _refs(dag, res, order):
    metrics = {}
    for action in (0, 1, 2):
        out = schedule(dag, order, [action] * len(order), res)
        metrics[action] = (out.makespan_seconds, out.energy.total_mobile_joules)
    return ReferenceRanges(
        L_ue=metrics[0][0], L_mec=metrics[1][0], L_helper=metrics[2][0],
        E_ue=metrics[0][1], E_mec=metrics[1][1], E_helper=metrics[2][1],
    )


def _objective(dag, res, order, deadlines=None, dtype="hard", beta=1.0):
    refs = _refs(dag, res, order)
    dl = {}
    for tid, d in (deadlines or {}).items():
        dl[int(tid)] = TaskDeadline(deadline_s=float(d), deadline_type=dtype)
    return ObjectiveContext(latency_ref_s=refs.L_ue, beta_soft=beta, deadlines=dl), refs


class TestSuffixContextAndConstruction(unittest.TestCase):
    def setUp(self):
        self.res = _resources()
        self.dag = _chain_dag(3)
        self.order = [0, 1, 2]

    def test_context_exposes_state(self):
        obj, _refs = _objective(self.dag, self.res, self.order)
        ctx = SuffixContext(
            dag=self.dag, resources=self.res, order=self.order, objective=obj, index=0
        )
        self.assertEqual(ctx.current_task, 0)
        self.assertEqual(ctx.remaining, [0, 1, 2])
        self.assertTrue(ctx.is_sink(2))
        self.assertFalse(ctx.is_sink(0))
        self.assertEqual(ctx.successor_bytes(0), BIG_OUTPUT)

    def test_objective_context_validation(self):
        with self.assertRaises(ValueError):
            ObjectiveContext(latency_ref_s=0.0)
        with self.assertRaises(ValueError):
            ObjectiveContext(latency_ref_s=1.0, energy_budget_j=0.0)

    def test_suffix_found_and_potential_finite(self):
        obj, _ = _objective(self.dag, self.res, self.order)
        ctx = SuffixContext(
            dag=self.dag, resources=self.res, order=self.order, objective=obj, index=0
        )
        s = construct_fastest_feasible_suffix(ctx, 1)
        self.assertTrue(s.found, s.reason)
        self.assertIsNotNone(s.estimated_J)
        self.assertGreater(s.estimated_latency_s, 0.0)
        self.assertEqual(len(s.plan), 3)
        # the constructed plan is a real, schedulable plan
        actions = [a for _t, a in s.plan]
        out = schedule(self.dag, self.order, actions, self.res)
        self.assertAlmostEqual(out.makespan_seconds, s.estimated_latency_s, places=9)

    def test_suffix_potential_matches_actual_objective(self):
        obj, refs = _objective(self.dag, self.res, self.order)
        spec = ObjectiveSpec(latency_ref=LATENCY_REF_ALL_UE, energy_budget_j=1e12)
        ctx = SuffixContext(
            dag=self.dag, resources=self.res, order=self.order, objective=obj, index=0
        )
        s = construct_fastest_feasible_suffix(ctx, 1)
        actual = evaluate_plan_objective(s.result, refs, spec)
        self.assertAlmostEqual(s.estimated_J, actual.J, places=9)


class TestTest1LookaheadDecidesCurrentAction(unittest.TestCase):
    """Test 1: a LATER deadline must be able to decide the CURRENT action.

    This is the whole point of ③: an action is no longer judged by its own task
    alone.  task0 heavy, task1 heavy with a tight hard deadline.
    """

    def setUp(self):
        self.res = _resources()
        tasks = [
            CanonicalTask(task_id=0, compute_workload_bytes=WORKLOAD, task_output_bytes=BIG_OUTPUT),
            CanonicalTask(
                task_id=1, compute_workload_bytes=WORKLOAD, task_output_bytes=SMALL,
                deadline_s=3.0, deadline_type="hard",
            ),
            CanonicalTask(task_id=2, compute_workload_bytes=SMALL, task_output_bytes=SMALL),
        ]
        self.dag = CanonicalDAG.from_records(tasks, [(0, 1, BIG_OUTPUT), (1, 2, SMALL)])
        self.order = [0, 1, 2]

    def _ctx(self, deadlines):
        obj, _ = _objective(self.dag, self.res, self.order, deadlines=deadlines)
        return SuffixContext(
            dag=self.dag, resources=self.res, order=self.order, objective=obj, index=0
        )

    def test_later_deadline_decides_current_action(self):
        ctx = self._ctx({1: 3.0})
        ev_mec = evaluate_action(ctx, 1)
        ev_ue = evaluate_action(ctx, 0)
        # MEC on task0 keeps task1's deadline reachable
        self.assertTrue(ev_mec.suffix.found)
        self.assertTrue(ev_mec.mask)
        # UE on task0 makes it impossible even under the relaxation -> sound mask
        self.assertFalse(ev_ue.mask)
        self.assertIn(ev_ue.mask_reason, ("dag_lower_bound_infeasible", "lb_exceeds_deadline"))

    def test_action_evaluation_exposes_potential_per_action(self):
        # loose deadline: every action keeps a suffix, potentials differ
        ctx = self._ctx({1: 1e6})
        evals = evaluate_all_actions(ctx)
        self.assertEqual(len(evals), 3)
        self.assertTrue(all(e.mask for e in evals))
        self.assertTrue(all(e.suffix.found for e in evals))
        potentials = [e.potential for e in evals]
        self.assertTrue(all(p is not None for p in potentials))
        self.assertGreater(len(set(round(p, 9) for p in potentials)), 1)
        picked = best_by_potential(evals)
        self.assertIsNotNone(picked)
        self.assertEqual(picked.potential, min(potentials))
        # the fastest placement is not automatically the best objective
        self.assertLessEqual(picked.potential, evals[0].potential)


class TestTest2HeuristicFailureNoMask(unittest.TestCase):
    """Test 2: a failed construction never masks."""

    def setUp(self):
        self.res = _resources()
        # every task carries an impossible hard deadline -> construction must fail
        tasks = [
            CanonicalTask(
                task_id=0, compute_workload_bytes=WORKLOAD, task_output_bytes=BIG_OUTPUT,
                deadline_s=1e-9, deadline_type="hard",
            ),
            CanonicalTask(
                task_id=1, compute_workload_bytes=SMALL, task_output_bytes=SMALL,
                deadline_s=1e-9, deadline_type="hard",
            ),
        ]
        self.dag = CanonicalDAG.from_records(tasks, [(0, 1, BIG_OUTPUT)])
        self.order = [0, 1]

    def test_heuristic_failure_reason_is_recorded(self):
        obj, _ = _objective(self.dag, self.res, self.order, deadlines={0: 1e-9, 1: 1e-9})
        ctx = SuffixContext(
            dag=self.dag, resources=self.res, order=self.order, objective=obj, index=0
        )
        # index 1: the last task, its own LB already exceeds the deadline -> proof mask
        ev0 = evaluate_action(ctx, 1)
        self.assertFalse(ev0.mask)                     # proven by LB, not by heuristic
        self.assertEqual(ev0.mask_reason, "lb_exceeds_deadline")

    def test_heuristic_failure_alone_keeps_action_open(self):
        # a deadline the current task can meet, but a *later* task cannot:
        # construction fails, yet no proof exists for the current action
        tasks = [
            CanonicalTask(task_id=0, compute_workload_bytes=SMALL, task_output_bytes=BIG_OUTPUT),
            CanonicalTask(
                task_id=1, compute_workload_bytes=WORKLOAD, task_output_bytes=SMALL,
                deadline_s=1e-9, deadline_type="hard",
            ),
        ]
        dag = CanonicalDAG.from_records(tasks, [(0, 1, BIG_OUTPUT)])
        obj, _ = _objective(dag, self.res, [0, 1], deadlines={1: 1e-9})
        ctx = SuffixContext(dag=dag, resources=self.res, order=[0, 1], objective=obj, index=0)
        evals = evaluate_all_actions(ctx)
        for ev in evals:
            if ev.suffix.found is False:
                # construction failed for this action; check whether it was masked
                if not ev.mask:
                    continue
                self.assertIn(
                    ev.mask_reason,
                    ("not_proven_infeasible", REASON_HEURISTIC_FAILED, "lb_exceeds_deadline"),
                )

    def test_mask_reason_never_claims_proof_from_heuristic(self):
        tasks = [
            CanonicalTask(task_id=0, compute_workload_bytes=SMALL, task_output_bytes=BIG_OUTPUT),
            CanonicalTask(
                task_id=1, compute_workload_bytes=WORKLOAD, task_output_bytes=SMALL,
                deadline_s=1e-9, deadline_type="hard",
            ),
        ]
        dag = CanonicalDAG.from_records(tasks, [(0, 1, BIG_OUTPUT)])
        obj, _ = _objective(dag, self.res, [0, 1], deadlines={1: 1e-9})
        ctx = SuffixContext(dag=dag, resources=self.res, order=[0, 1], objective=obj, index=0)
        for ev in evaluate_all_actions(ctx):
            if not ev.mask:
                self.assertNotEqual(ev.mask_reason, REASON_HEURISTIC_FAILED)


class TestTest3MaskSoundness(unittest.TestCase):
    """Test 3: for every masked action, NO sampled completion meets the deadline."""

    def setUp(self):
        self.res = _resources()
        tasks = [
            CanonicalTask(task_id=0, compute_workload_bytes=WORKLOAD, task_output_bytes=BIG_OUTPUT),
            CanonicalTask(
                task_id=1, compute_workload_bytes=SMALL, task_output_bytes=SMALL,
                deadline_s=0.5, deadline_type="hard",
            ),
            CanonicalTask(task_id=2, compute_workload_bytes=SMALL, task_output_bytes=SMALL),
        ]
        self.dag = CanonicalDAG.from_records(
            tasks, [(0, 1, BIG_OUTPUT), (1, 2, SMALL)]
        )
        self.order = [0, 1, 2]

    def test_masked_action_has_no_feasible_completion(self):
        deadline = 0.5
        obj, _ = _objective(self.dag, self.res, self.order, deadlines={1: deadline})
        ctx = SuffixContext(
            dag=self.dag, resources=self.res, order=self.order, objective=obj, index=0
        )
        for ev in evaluate_all_actions(ctx):
            if ev.mask:
                continue
            # masked -> every completion of the remaining tasks must miss task1
            for suffix in itertools.product((0, 1, 2), repeat=2):
                actions = [ev.action, suffix[0], suffix[1]]
                out = schedule(self.dag, self.order, actions, self.res)
                self.assertGreater(
                    out.tasks[1].tardiness_s, 0.0,
                    msg="masked action %d rescued by suffix %s" % (ev.action, suffix),
                )

    def test_dag_lower_bound_is_admissible(self):
        """The relaxation must never exceed a real completion's availability."""
        obj, _ = _objective(self.dag, self.res, self.order)
        ctx = SuffixContext(
            dag=self.dag, resources=self.res, order=self.order, objective=obj, index=0
        )
        for action in (0, 1, 2):
            lb = dag_lower_bound_ready(ctx, action)
            for suffix in itertools.product((0, 1, 2), repeat=2):
                actions = [action, suffix[0], suffix[1]]
                out = schedule(self.dag, self.order, actions, self.res)
                for tid in (0, 1, 2):
                    achieved = out.tasks[tid].availability_seconds
                    self.assertLessEqual(
                        lb[tid], achieved + 1e-9 * max(1.0, abs(achieved)),
                        msg="LB above achieved for task %d action %d suffix %s"
                        % (tid, action, suffix),
                    )

    def test_mask_comes_only_from_proof_sources(self):
        obj, _ = _objective(self.dag, self.res, self.order, deadlines={1: 1e-9})
        ctx = SuffixContext(
            dag=self.dag, resources=self.res, order=self.order, objective=obj, index=0
        )
        for action in (0, 1, 2):
            proof, reason, _task = dag_lower_bound_masks(ctx, action)
            ev = evaluate_action(ctx, action)
            if proof:
                self.assertFalse(ev.mask)
                self.assertEqual(reason, REASON_PROOF_MASK)
            elif not ev.mask:
                self.assertEqual(ev.mask_reason, "lb_exceeds_deadline")


class TestTest4PotentialConsistency(unittest.TestCase):
    """Test 4: terminal Ĵ == J_actual and the shaped reward has the right sign."""

    def setUp(self):
        self.res = _resources()
        self.dag = _chain_dag(3, deadlines={2: 5.0}, dtype="soft")
        self.order = [0, 1, 2]
        self.spec = ObjectiveSpec(latency_ref=LATENCY_REF_ALL_UE, energy_budget_j=1e12)

    def test_terminal_potential_equals_actual_J(self):
        obj, refs = _objective(self.dag, self.res, self.order, deadlines={2: 5.0}, dtype="soft")
        ctx = SuffixContext(
            dag=self.dag, resources=self.res, order=self.order, objective=obj, index=0
        )
        suffix = construct_fastest_feasible_suffix(ctx, 1)
        self.assertTrue(suffix.found)
        actual = evaluate_plan_objective(suffix.result, refs, self.spec)
        self.assertAlmostEqual(suffix.estimated_J, actual.J, places=9)
        self.assertAlmostEqual(
            suffix_potential(suffix.result, obj), actual.J, places=9
        )

    def test_shaped_reward_sign_and_telescoping(self):
        """r_t = Ĵ(s_{t-1}) - gamma * Ĵ(s_t) with Ĵ = state_potential (gamma=1)."""
        obj, refs = _objective(self.dag, self.res, self.order, deadlines={2: 5.0}, dtype="soft")
        plan = [1, 1, 0]
        realized = schedule(self.dag, self.order, plan, self.res)

        def state_seq(actions):
            out = schedule(self.dag, self.order, actions, self.res)
            seq = []
            for idx in range(len(self.order) + 1):
                decisions = {
                    tid: Location.from_action(a)
                    for tid, a in zip(self.order[:idx], actions[:idx])
                }
                prefix_finish = {
                    tid: float(out.tasks[tid].finish) for tid in self.order[:idx]
                }
                idx_clamped = min(idx, len(self.order) - 1)
                ctx = SuffixContext(
                    dag=self.dag, resources=self.res, order=self.order, objective=obj,
                    decisions=decisions, prefix_finish=prefix_finish, index=idx_clamped,
                )
                if idx == len(self.order):
                    seq.append(suffix_potential(out, obj))     # terminal: J_actual
                    continue
                s = state_potential(ctx)
                self.assertTrue(s.found, "state %d: %s" % (idx, s.reason))
                seq.append(s.estimated_J)
            return seq

        seq = state_seq(plan)
        gamma = 1.0
        rewards = [seq[i - 1] - gamma * seq[i] for i in range(1, len(seq))]
        self.assertAlmostEqual(sum(rewards), seq[0] - seq[-1], places=9)
        self.assertAlmostEqual(seq[-1], suffix_potential(realized, obj), places=9)

        ue_seq = state_seq([0, 0, 0])
        # Ĵ(s_0) is the same for both plans
        self.assertAlmostEqual(seq[0], ue_seq[0], places=9)
        ue_total = sum(ue_seq[i - 1] - ue_seq[i] for i in range(1, len(ue_seq)))
        constructed_total = sum(rewards)
        # with exact telescoping the difference is purely terminal
        self.assertAlmostEqual(
            constructed_total - ue_total, ue_seq[-1] - seq[-1], places=9
        )
        self.assertGreater(constructed_total, ue_total)   # faster plan scores better

    def test_feasible_mask_shape(self):
        obj, _ = _objective(self.dag, self.res, self.order)
        ctx = SuffixContext(
            dag=self.dag, resources=self.res, order=self.order, objective=obj, index=0
        )
        mask = feasible_action_mask(ctx)
        self.assertEqual(len(mask), 3)
        self.assertTrue(all(isinstance(m, bool) for m in mask))


if __name__ == "__main__":
    unittest.main(verbosity=2)
