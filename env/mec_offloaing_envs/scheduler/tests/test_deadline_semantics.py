#!/usr/bin/env python3
"""②A — deadline semantics and measurement (no PPO/objective change).

The rule under test (approved):

    a task misses its deadline when its OUTPUT is not usable in time,
    i.e.  F_ready > deadline_s,   NOT when compute finishes late.

②A.1 fixes the availability notion:

    F_ready(i) = max_{j in succ(i)} delivery(i -> j)      (all consumers ready)
                 return-to-UE for a sink

A task with two successors — one on MEC that can read the output at 10 s and one
on UE that only gets it at 18 s — is NOT ready at 10 s.  `first_available`
(earliest consumer) is kept as a diagnostic only.
"""

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
    ResourceConfig,
    schedule,
)
from env.mec_offloaing_envs.scheduler.deadlines import assign_deadlines  # noqa: E402
from env.mec_offloaing_envs.scheduler.energy_model import (  # noqa: E402
    MODEL_PHYSICAL,
    SCOPE_SYSTEM,
)

WORKLOAD = 4_166_700          # ~1.0 s of MEC compute at f=10GHz, xi=300
BIG_OUTPUT = 4_166_700        # large downlink: ~4.5 s at the frozen DL rate
ORDER = [0, 1]
ACTIONS = [1, 0]              # task0 on MEC, task1 on UE


def _spec():
    from env.mec_offloaing_envs.scheduler.energy_model import EnergyModelSpec

    return EnergyModelSpec.from_dict(
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


def _resources(spec=None):
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
        energy_model=spec,
    )


def _dag(deadline0=None, dtype="none", weight=1.0, deadline1=None, cclass="medium"):
    tasks = [
        CanonicalTask(
            task_id=0,
            compute_workload_bytes=WORKLOAD,
            task_output_bytes=BIG_OUTPUT,
            external_input_bytes=0,
            deadline_s=deadline0,
            deadline_type=dtype,
            tardiness_weight=weight,
            criticality_class=cclass,
        ),
        CanonicalTask(
            task_id=1,
            compute_workload_bytes=1024,
            task_output_bytes=512,
            external_input_bytes=0,
            deadline_s=deadline1,
            deadline_type=dtype if deadline1 is not None else "none",
            tardiness_weight=weight,
            criticality_class=cclass,
        ),
    ]
    return CanonicalDAG.from_records(tasks, [(0, 1, BIG_OUTPUT)])


def _fanout_dag(deadline0=None, dtype="none", weight=1.0):
    """A -> B (MEC, ready immediately) and A -> C (UE, ready after a slow DL).

    This is the ②A.1 motivating case: earliest-consumer says "met", all-
    consumers-ready says "missed".
    """
    tasks = [
        CanonicalTask(
            task_id=0,
            compute_workload_bytes=WORKLOAD,
            task_output_bytes=BIG_OUTPUT,
            deadline_s=deadline0,
            deadline_type=dtype,
            tardiness_weight=weight,
        ),
        CanonicalTask(task_id=1, compute_workload_bytes=1024, task_output_bytes=256),
        CanonicalTask(task_id=2, compute_workload_bytes=1024, task_output_bytes=256),
    ]
    return CanonicalDAG.from_records(tasks, [(0, 1, 256), (0, 2, BIG_OUTPUT)])


FANOUT_ORDER = [0, 1, 2]
FANOUT_ACTIONS = [1, 1, 0]   # A on MEC, B on MEC (no hop), C on UE (slow DL)


class TestDeadlineSemantics(unittest.TestCase):
    def setUp(self):
        self.res = _resources(_spec())

    def test_no_deadlines_is_inert(self):
        out = schedule(_dag(), ORDER, ACTIONS, self.res)
        self.assertEqual(out.soft_tardiness_s, 0.0)
        self.assertEqual(out.hard_miss_count, 0)
        self.assertTrue(out.hard_feasible)
        for rec in out.tasks.values():
            self.assertEqual(rec.deadline_type, "none")
            self.assertEqual(rec.tardiness_s, 0.0)
        # measurement is additive: identical T/E to a graph built without the
        # ②A fields at all, on the SAME resource model
        plain = CanonicalDAG.from_records(
            [
                CanonicalTask(
                    task_id=0,
                    compute_workload_bytes=WORKLOAD,
                    task_output_bytes=BIG_OUTPUT,
                ),
                CanonicalTask(
                    task_id=1, compute_workload_bytes=1024, task_output_bytes=512
                ),
            ],
            [(0, 1, BIG_OUTPUT)],
        )
        base = schedule(plain, ORDER, ACTIONS, self.res)
        self.assertAlmostEqual(out.makespan_seconds, base.makespan_seconds, places=9)
        self.assertAlmostEqual(
            out.energy.total_system_joules, base.energy.total_system_joules, places=9
        )

    def test_availability_not_compute_finish(self):
        """The approved example: compute on time, transfer late => MISS."""
        base = schedule(_dag(), ORDER, ACTIONS, self.res)
        rec0 = base.tasks[0]
        self.assertLess(rec0.finish, 3.0)            # compute finishes early
        self.assertGreater(rec0.availability_seconds, 3.0)  # output lands late
        out = schedule(_dag(deadline0=3.0, dtype="hard"), ORDER, ACTIONS, self.res)
        self.assertGreater(out.tasks[0].tardiness_s, 0.0)
        self.assertEqual(out.hard_miss_count, 1)
        self.assertFalse(out.hard_feasible)
        self.assertAlmostEqual(
            out.tasks[0].tardiness_s,
            out.tasks[0].availability_seconds - 3.0,
            places=9,
        )

    def test_deadline_against_finish_would_have_passed(self):
        """Shows the semantics matter: same plan, deadline between finish and availability."""
        out = schedule(_dag(deadline0=3.0, dtype="hard"), ORDER, ACTIONS, self.res)
        rec0 = out.tasks[0]
        self.assertLess(rec0.finish, 3.0)       # a finish-based check would say OK
        self.assertTrue(rec0.missed)            # an availability-based check says MISS

    def test_meeting_the_deadline(self):
        out = schedule(_dag(deadline0=1e9, dtype="hard"), ORDER, ACTIONS, self.res)
        self.assertEqual(out.hard_miss_count, 0)
        self.assertTrue(out.hard_feasible)
        self.assertEqual(out.tasks[0].tardiness_s, 0.0)

    def test_soft_tardiness_weighted_by_tardiness_weight(self):
        a = schedule(_dag(deadline0=1.0, dtype="soft", weight=1.0), ORDER, ACTIONS, self.res)
        b = schedule(_dag(deadline0=1.0, dtype="soft", weight=2.5), ORDER, ACTIONS, self.res)
        self.assertGreater(a.soft_tardiness_s, 0.0)
        self.assertAlmostEqual(b.soft_tardiness_s / a.soft_tardiness_s, 2.5, places=9)
        self.assertAlmostEqual(
            a.soft_tardiness_normalized, a.soft_tardiness_s / 1.0, places=9
        )

    def test_firm_and_hard_channels_are_separate(self):
        out = schedule(_dag(deadline0=1.0, dtype="firm"), ORDER, ACTIONS, self.res)
        self.assertEqual(out.firm_miss_count, 1)
        self.assertEqual(out.hard_miss_count, 0)
        self.assertTrue(out.hard_feasible)
        self.assertGreater(out.firm_miss_rate, 0.0)

    def test_availability_never_precedes_finish(self):
        for actions in ([0, 0], [1, 0], [2, 0]):
            out = schedule(_dag(), ORDER, actions, self.res)
            for rec in out.tasks.values():
                self.assertGreaterEqual(
                    rec.availability_seconds + 1e-12, rec.finish
                )

    def test_sink_availability_is_the_return_to_ue(self):
        # task1 is the sink on UE: no return hops, availability == finish
        out = schedule(_dag(), ORDER, ACTIONS, self.res)
        self.assertAlmostEqual(
            out.tasks[1].availability_seconds, out.tasks[1].finish, places=9
        )

    def test_max_and_mean_tardiness(self):
        out = schedule(
            _dag(deadline0=1.0, dtype="soft", weight=1.0, deadline1=1e9),
            ORDER,
            ACTIONS,
            self.res,
        )
        self.assertGreater(out.max_tardiness_s, 0.0)
        self.assertGreaterEqual(out.max_tardiness_s, out.mean_tardiness_s)


class TestAllConsumersReadySemantics(unittest.TestCase):
    """②A.1: the primary deadline basis is max_j delivery(i->j)."""

    def setUp(self):
        self.res = _resources(_spec())

    def test_fanout_earliest_vs_all_consumers(self):
        out = schedule(_fanout_dag(), FANOUT_ORDER, FANOUT_ACTIONS, self.res)
        rec = out.tasks[0]
        self.assertLess(rec.first_available, rec.all_consumers_ready)
        # the fast consumer (B on MEC) sees the output at compute finish
        self.assertAlmostEqual(rec.first_available, rec.finish, places=9)
        # the slow consumer (C on UE) needs the downlink
        self.assertGreater(rec.all_consumers_ready, rec.finish)

    def test_deadline_between_the_two_arrivals_is_a_miss(self):
        base = schedule(_fanout_dag(), FANOUT_ORDER, FANOUT_ACTIONS, self.res)
        rec = base.tasks[0]
        # deadline strictly between first_available and all_consumers_ready
        deadline = 0.5 * (rec.first_available + rec.all_consumers_ready)
        self.assertGreater(deadline, rec.first_available)
        self.assertLess(deadline, rec.all_consumers_ready)

        out = schedule(
            _fanout_dag(deadline0=deadline, dtype="hard"),
            FANOUT_ORDER,
            FANOUT_ACTIONS,
            self.res,
        )
        r0 = out.tasks[0]
        # earliest-consumer semantics WOULD have passed...
        self.assertLess(r0.first_available, deadline)
        # ...but the primary basis (all consumers ready) correctly misses
        self.assertGreater(r0.availability_seconds, deadline)
        self.assertTrue(r0.missed)
        self.assertEqual(out.hard_miss_count, 1)
        self.assertFalse(out.hard_feasible)

    def test_basis_is_recorded(self):
        out = schedule(_fanout_dag(), FANOUT_ORDER, FANOUT_ACTIONS, self.res)
        self.assertEqual(out.deadline_basis, "all_consumers_ready")
        self.assertEqual(out.deadline_metrics()["deadline_basis_all_consumers_ready"], 1.0)

    def test_last_delivery_alias_still_works(self):
        out = schedule(_fanout_dag(), FANOUT_ORDER, FANOUT_ACTIONS, self.res)
        self.assertEqual(out.tasks[0].last_delivery, out.tasks[0].all_consumers_ready)

    def test_single_consumer_basis_equals_first_available(self):
        out = schedule(_dag(), ORDER, ACTIONS, self.res)
        for rec in out.tasks.values():
            self.assertAlmostEqual(
                rec.all_consumers_ready, rec.first_available, places=9
            )

    def test_criticality_class_is_separate_from_weight(self):
        out = schedule(
            _dag(deadline0=1.0, dtype="soft", weight=2.0, cclass="high"),
            ORDER,
            ACTIONS,
            self.res,
        )
        rec = out.tasks[0]
        self.assertEqual(rec.criticality_class, "high")
        self.assertAlmostEqual(rec.tardiness_weight, 2.0)
        # class alone changes nothing numerically
        out2 = schedule(
            _dag(deadline0=1.0, dtype="soft", weight=2.0, cclass="low"),
            ORDER,
            ACTIONS,
            self.res,
        )
        self.assertAlmostEqual(
            out.soft_tardiness_s, out2.soft_tardiness_s, places=12
        )

    def test_invalid_criticality_class_rejected(self):
        with self.assertRaises(ValueError):
            CanonicalTask(
                task_id=0,
                compute_workload_bytes=1024,
                task_output_bytes=512,
                criticality_class="critical",
            )


class _FakeGraph:
    """Minimal legacy-task-graph stand-in for `assign_deadlines`."""

    def __init__(self, n=4):
        class _T:
            def __init__(self):
                self.processing_data_size = 1_048_576
                self.transmission_data_size = 524_288
                self.deadline_s = None
                self.deadline_type = "none"
                self.criticality = 1.0

        self.task_number = n
        self.task_list = [_T() for _ in range(n)]
        self.prioritize_sequence = list(range(n))
        self.pre_task_sets = [set() if i == 0 else {i - 1} for i in range(n)]
        self.succ_task_sets = [set() if i == n - 1 else {i + 1} for i in range(n)]
        self.edge_set = [
            [i, i, 1_048_576, 524_288, i + 1, i + 1, 1_048_576] for i in range(n - 1)
        ]


class TestDeadlineRules(unittest.TestCase):
    def test_none_rule_is_inert(self):
        tg = _FakeGraph()
        assign_deadlines(tg, _resources(), rule="none")
        for task in tg.task_list:
            self.assertIsNone(task.deadline_s)
            self.assertEqual(task.deadline_type, "none")

    def test_uniform_rule_scales_with_reference(self):
        from env.mec_offloaing_envs.scheduler import compute_reference_ranges

        tg = _FakeGraph()
        res = _resources()
        assign_deadlines(tg, res, rule="uniform_of_all_mec", factor=1.5, deadline_type="soft")
        refs = compute_reference_ranges(tg, res)
        for task in tg.task_list:
            self.assertAlmostEqual(task.deadline_s, 1.5 * refs.L_mec, places=6)
            self.assertEqual(task.deadline_type, "soft")

    def test_per_task_rule_uses_availability(self):
        tg = _FakeGraph()
        res = _resources()
        assign_deadlines(tg, res, rule="per_task_slack_of_ready", factor=1.1,
                         deadline_type="hard", tardiness_weight=0.5,
                         tardiness_weight_by_task={0: 3.0})
        deadlines = [t.deadline_s for t in tg.task_list]
        self.assertEqual(len(set(deadlines)), len(deadlines))   # per-task
        self.assertTrue(all(d > 0 for d in deadlines))
        self.assertAlmostEqual(tg.task_list[0].tardiness_weight, 3.0)
        self.assertAlmostEqual(tg.task_list[1].tardiness_weight, 0.5)

    def test_invalid_rule_and_factor(self):
        tg = _FakeGraph()
        with self.assertRaises(ValueError):
            assign_deadlines(tg, _resources(), rule="bogus")
        with self.assertRaises(ValueError):
            assign_deadlines(tg, _resources(), rule="uniform_of_all_mec", factor=0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
