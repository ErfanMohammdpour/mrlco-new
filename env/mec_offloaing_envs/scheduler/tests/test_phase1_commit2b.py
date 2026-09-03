#!/usr/bin/env python3
"""Commit 2B: validation, greedy-via-engine, architecture, real .gv integration."""

from __future__ import annotations

import math
import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env.mec_offloaing_envs.scheduler import (  # noqa: E402
    CanonicalDAG,
    CanonicalTask,
    ResourceConfig,
    greedy_plan,
    schedule,
    schedule_via_adapter,
    to_canonical_dag,
)


class TestNumericValidation(unittest.TestCase):
    def test_negative_output_rejected(self):
        with self.assertRaises(ValueError):
            CanonicalTask(1, 10, -1, 0)

    def test_negative_external_rejected(self):
        with self.assertRaises(ValueError):
            CanonicalTask(1, 10, 5, -3)

    def test_self_edge_rejected(self):
        tasks = [CanonicalTask(1, 10, 5, 10)]
        with self.assertRaises(ValueError):
            CanonicalDAG.from_records(tasks, [(1, 1, 5)])

    def test_negative_edge_rejected(self):
        tasks = [CanonicalTask(1, 10, 5, 10), CanonicalTask(2, 10, 5, 0)]
        with self.assertRaises(ValueError):
            CanonicalDAG.from_records(tasks, [(1, 2, -5)])

    def test_zero_rate_rejected(self):
        cfg = ResourceConfig.from_frozen_yaml()
        with self.assertRaises(ValueError):
            ResourceConfig(
                ue_cpu_bytes_per_second=0.0,
                mec_cpu_bytes_per_second=cfg.mec_cpu_bytes_per_second,
                helper_cpu_bytes_per_second=cfg.helper_cpu_bytes_per_second,
                mec_uplink_bytes_per_second=cfg.mec_uplink_bytes_per_second,
                mec_downlink_bytes_per_second=cfg.mec_downlink_bytes_per_second,
                v2v_bytes_per_second=cfg.v2v_bytes_per_second,
                rho_ue=cfg.rho_ue,
                f_l=cfg.f_l,
                zeta=cfg.zeta,
                ptx_mec_w=cfg.ptx_mec_w,
                prx_mec_w=cfg.prx_mec_w,
                ptx_v2v_w=cfg.ptx_v2v_w,
                prx_v2v_w=cfg.prx_v2v_w,
                rho_helper=cfg.rho_helper,
                f_v2v=cfg.f_v2v,
            )

    def test_nan_power_rejected(self):
        cfg = ResourceConfig.from_frozen_yaml()
        with self.assertRaises(ValueError):
            ResourceConfig(
                ue_cpu_bytes_per_second=cfg.ue_cpu_bytes_per_second,
                mec_cpu_bytes_per_second=cfg.mec_cpu_bytes_per_second,
                helper_cpu_bytes_per_second=cfg.helper_cpu_bytes_per_second,
                mec_uplink_bytes_per_second=cfg.mec_uplink_bytes_per_second,
                mec_downlink_bytes_per_second=cfg.mec_downlink_bytes_per_second,
                v2v_bytes_per_second=cfg.v2v_bytes_per_second,
                rho_ue=math.nan,
                f_l=cfg.f_l,
                zeta=cfg.zeta,
                ptx_mec_w=cfg.ptx_mec_w,
                prx_mec_w=cfg.prx_mec_w,
                ptx_v2v_w=cfg.ptx_v2v_w,
                prx_v2v_w=cfg.prx_v2v_w,
                rho_helper=cfg.rho_helper,
                f_v2v=cfg.f_v2v,
            )


class TestZeroDurationNotRecorded(unittest.TestCase):
    def test_zero_byte_same_location_no_interval(self):
        cfg = ResourceConfig.from_frozen_yaml()
        tasks = [
            CanonicalTask(1, 1048576, 0, 1048576),
            CanonicalTask(2, 1048576, 0, 0),
        ]
        dag = CanonicalDAG.from_records(tasks, [(1, 2, 0)])
        result = schedule(dag, [1, 2], ["UE", "UE"], cfg)
        self.assertTrue(all(iv.end > iv.start for iv in result.resource_intervals))
        self.assertEqual(result.transfers, [])


class TestGreedyUsesEngine(unittest.TestCase):
    def test_greedy_scores_with_schedule(self):
        class TG:
            task_number = 2
            task_list = [
                types.SimpleNamespace(processing_data_size=1048576, transmission_data_size=458752),
                types.SimpleNamespace(processing_data_size=1048576, transmission_data_size=458752),
            ]
            pre_task_sets = [set(), {0}]
            edge_set = [[0, 0, 1048576, 458752, 1, 1, 1048576]]
            prioritize_sequence = [0, 1]

        cfg = ResourceConfig.from_frozen_yaml()
        plan, result = greedy_plan(TG(), cfg)
        self.assertEqual(len(plan), 2)
        self.assertEqual({a for _, a in plan} <= {0, 1, 2}, True)
        direct, _, _ = schedule_via_adapter(TG(), plan, cfg)
        self.assertAlmostEqual(result.makespan_seconds, direct.makespan_seconds)


class TestNoLegacySchedulerMath(unittest.TestCase):
    FORBIDDEN = (
        "FT_cloud",
        "FT_ws",
        "FT_wr",
        "FT_v2v_dl",
        "v2v_channel_available_time",
        "ws_avaliable_time",
    )

    def test_offloading_env_and_evaluator(self):
        env_text = (ROOT / "env/mec_offloaing_envs/offloading_env.py").read_text()
        env_body = env_text.split("class OffloadingEnvironment", 1)[1]
        eval_text = (ROOT / "meta_evaluator.py").read_text()
        for label, text in (("OffloadingEnvironment", env_body), ("meta_evaluator", eval_text)):
            for token in self.FORBIDDEN:
                self.assertNotIn(
                    token,
                    text,
                    f"{label} still contains legacy calendar token {token}",
                )


class TestRealGvIntegration(unittest.TestCase):
    def test_parser_adapter_schedule(self):
        for name in ("gym", "gym.core"):
            if name not in sys.modules:
                sys.modules[name] = types.ModuleType(name)
        sys.modules["gym.core"].Env = type("Env", (), {})
        if "graphviz" not in sys.modules:
            gv = types.ModuleType("graphviz")
            gv.Digraph = type("Digraph", (), {})
            sys.modules["graphviz"] = gv
        for stub in ("pydotplus", "pydotplus.graphviz"):
            mod = sys.modules.get(stub)
            graphviz_mod = getattr(mod, "graphviz", None) if mod is not None else None
            has_parser = (
                mod is not None
                and (
                    hasattr(mod, "graph_from_dot_file")
                    or hasattr(graphviz_mod, "graph_from_dot_file")
                )
            )
            if mod is not None and not has_parser:
                sys.modules.pop(stub, None)
        try:
            import pydotplus.graphviz  # noqa: F401
        except ImportError:
            self.skipTest("pydotplus required for OffloadingTaskGraph parser integration")

        from env.mec_offloaing_envs.offloading_task_graph import OffloadingTaskGraph

        gv_path = ROOT / (
            "env/mec_offloaing_envs/data/meta_offloading_20/"
            "offload_random20_1/random.20.0.gv"
        )
        self.assertTrue(gv_path.exists(), gv_path)
        tg = OffloadingTaskGraph(str(gv_path))
        tg.prioritize_sequence = list(range(tg.task_number))
        dag = to_canonical_dag(tg)
        self.assertEqual(len(dag.tasks), 20)
        cfg = ResourceConfig.from_frozen_yaml()
        plan = [(i, 0) for i in range(tg.task_number)]
        result, deltas, energy = schedule_via_adapter(tg, plan, cfg)
        self.assertEqual(len(deltas), 20)
        self.assertAlmostEqual(sum(deltas), result.makespan_seconds, places=9)
        self.assertGreater(result.makespan_seconds, 0.0)
        self.assertTrue(all(e >= 0.0 for e in energy))


if __name__ == "__main__":
    unittest.main()
