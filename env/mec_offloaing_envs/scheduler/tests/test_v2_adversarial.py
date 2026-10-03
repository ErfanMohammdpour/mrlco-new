#!/usr/bin/env python3
"""v2 stage-8 adversarial fixtures: extreme states must stay finite, invariant-clean and
monotone in the expected direction (no crash, no silent NaN, no impossible schedule)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spec.automotive_training.automotive_loader import load_dataset  # noqa: E402
from spec.automotive_training.v2.adapters import (  # noqa: E402
    compute_spec, dag_spec_from_graph, link_spec, plan_map_from_actions,
)
from spec.automotive_training.v2.helper_model import HelperState  # noqa: E402
from spec.automotive_training.v2.link_model import make_process  # noqa: E402
from spec.automotive_training.v2.reliability import make_gate, standby_required  # noqa: E402
from spec.automotive_training.v2.shared_scheduler import (  # noqa: E402
    V2ComputeSpec, V2DAGSpec, V2LinkSpec, V2TaskSpec, schedule_shared,
)


def _graph(i=0):
    return load_dataset().validation_query()[i]


def _sched(graph, actions, *, link=None, compute=None, background=0, **kw):
    link = link or link_spec(graph)
    compute = compute or compute_spec([graph])
    dag = dag_spec_from_graph(graph, dag_id="fg", owner=0, helper_id=0 if "helper_states" in kw else None)
    dags, plans = [dag], {"fg": plan_map_from_actions(graph, actions)}
    for b in range(int(background)):
        bg = dag_spec_from_graph(graph, dag_id="bg%d" % b, owner=0)
        dags.append(bg)
        plans[bg.dag_id] = plan_map_from_actions(graph, [1] * 20)
    res = schedule_shared(dags, plans, link=link, compute=compute, **kw)
    return res, float(res.completion_by_dag["fg"])


class TestExtremeStates(unittest.TestCase):
    def test_mec_overload_makes_local_win(self):
        graph = _graph()
        cached = {}
        one, _ = _sched(graph, [1] * 20, background=0)
        for b in range(32):
            cached.setdefault("plan", plan_map_from_actions(graph, [1] * 20))
        heavy, fg_heavy = _sched(graph, [1] * 20, background=32)
        local, fg_local = _sched(graph, [0] * 20, background=32)
        self.assertGreater(fg_heavy, fg_local, "32-way MEC contention must favour local")

    def test_idle_mec_beats_local_for_mec_heavy_graph(self):
        graph = _graph()
        mec, fg_mec = _sched(graph, [1] * 20)
        ue, fg_ue = _sched(graph, [0] * 20)
        self.assertLess(fg_mec, fg_ue)

    def test_helper_with_zero_contact_is_rejected_not_crashed(self):
        graph = _graph()
        state = {0: HelperState(0, compute_spec([graph]).helper_cpu_bytes_per_s[0],
                                contact_end_s=1e-9, predicted_contact_end_s=1e-9)}
        res, _fg = _sched(graph, [2] * 20, helper_states=state)
        self.assertGreaterEqual(res.queue_stats["helper_rejections_inadmissible"], 1)
        for tm in res.timings.values():
            self.assertTrue(np.isfinite(tm.finish_s))

    def test_poor_v2v_removes_helper_advantage(self):
        graph = _graph()
        base = link_spec(graph)
        good = V2LinkSpec(base.mec_ul_bytes_per_s, base.mec_dl_bytes_per_s,
                          base.v2v_bytes_per_s, base.direct_helper_v2i, base.shared_radio)
        poor = V2LinkSpec(base.mec_ul_bytes_per_s, base.mec_dl_bytes_per_s,
                          base.v2v_bytes_per_s * 0.1, base.direct_helper_v2i, base.shared_radio)
        state = {0: HelperState(0, compute_spec([graph]).helper_cpu_bytes_per_s[0],
                                contact_end_s=1e6, predicted_contact_end_s=1e6)}
        _r1, fg_good = _sched(graph, [2] * 20, link=good, helper_states=state)
        _r2, fg_poor = _sched(graph, [2] * 20, link=poor, helper_states=state)
        self.assertGreaterEqual(fg_poor, fg_good)

    def test_high_criticality_unreliable_link_goes_local(self):
        graph = _graph()
        res, _fg = _sched(graph, [1] * 20, reliability_gate=make_gate(),
                          reliability_evidence={"link_confidence": 0.5,
                                                "outage_fraction": 0.3},
                          standby_for=standby_required)
        self.assertEqual(res.queue_stats["reliability_rejections"] > 0, True)
        self.assertTrue(all(tm.location == "UE" for tm in res.timings.values()))

    def test_outage_episode_stays_finite_and_logs_wait(self):
        graph = _graph()
        process = make_process("degraded", 3)
        res, fg = _sched(graph, [1] * 20, link_process=process)
        self.assertTrue(np.isfinite(fg) and fg > 0.0)
        self.assertGreaterEqual(res.queue_stats["outage_wait_total_s"], 0.0)
        for tm in res.timings.values():
            self.assertGreaterEqual(tm.outage_wait_s, 0.0)

    def test_single_task_and_all_local_plans_are_consistent(self):
        task = V2TaskSpec(0, 5e6, 2e6, (), True, True, "HIGH", 1.0, 0, external_input_bytes=1000)
        dag = V2DAGSpec("solo", 0, [task], helper_id=0)
        state = {0: HelperState(0, 2e6, contact_end_s=1e6, predicted_contact_end_s=1e6)}
        for plan in ({0: 0}, {0: 1}, {0: 2}):
            res = schedule_shared([dag], {"solo": plan}, link=V2LinkSpec(20e6, 20e6, 10e6),
                                  compute=V2ComputeSpec(10e6, 1, (1e6,), (2e6,)),
                                  helper_states=state)
            self.assertTrue(np.isfinite(res.makespan_s))
            self.assertTrue(res.invariants["precedence_and_data_arrival"])
            self.assertTrue(res.invariants["no_double_booking"])

    def test_contention_ordering_is_monotone(self):
        graph = _graph()
        times = []
        for load in (0, 4, 16):
            _res, fg = _sched(graph, [1] * 20, background=load)
            times.append(fg)
        self.assertLessEqual(times[0], times[1] + 1e-9)
        self.assertLessEqual(times[1], times[2] + 1e-9)


if __name__ == "__main__":
    unittest.main(verbosity=2)
