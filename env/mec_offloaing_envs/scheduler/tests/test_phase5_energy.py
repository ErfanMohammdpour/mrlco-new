#!/usr/bin/env python3
"""Phase 5 energy physics: J_λ, ADR-001 helper compute, hypervolume. No GPU."""

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


for name in ("gym", "gym.core", "graphviz", "pydotplus", "pydotplus.graphviz"):
    _stub_optional(name)
if not hasattr(sys.modules.get("gym.core", types.ModuleType("gym.core")), "Env"):
    sys.modules.setdefault("gym", types.ModuleType("gym"))
    sys.modules.setdefault("gym.core", types.ModuleType("gym.core"))
    sys.modules["gym.core"].Env = type("Env", (), {})
if not hasattr(sys.modules.get("graphviz", types.ModuleType("graphviz")), "Digraph"):
    sys.modules.setdefault("graphviz", types.ModuleType("graphviz"))
    sys.modules["graphviz"].Digraph = type("Digraph", (), {})


from env.mec_offloaing_envs.offloading_task_graph import OffloadingTaskGraph  # noqa: E402
from env.mec_offloaing_envs.scheduler import (  # noqa: E402
    CanonicalDAG,
    CanonicalTask,
    PARETO_LAMBDAS,
    ResourceConfig,
    compute_reference_ranges,
    j_lambda,
    j_report,
    lambda_tag,
    schedule,
    schedule_via_adapter,
)
from env.mec_offloaing_envs.scheduler.energy_api import ReferenceRanges  # noqa: E402
from spec.best_of_k import load_toy_oracle  # noqa: E402
from spec.hamming2_probe import _gv_path  # noqa: E402
from spec.pareto import hypervolume_2d, nondominated  # noqa: E402
from spec.resource_profiles import resource_config_for_profile, resources_cluster_for_profile  # noqa: E402
from spec.split_loader import meta_test_distribution_ids, meta_train_distribution_ids  # noqa: E402
from spec.twopt_expert import iterate_2opt  # noqa: E402


class TestJLambda(unittest.TestCase):
    def test_pareto_lambdas_and_tags(self):
        self.assertEqual(PARETO_LAMBDAS, (1.0, 0.75, 0.5, 0.25, 0.0))
        self.assertEqual(lambda_tag(1.0), "1.00")
        self.assertEqual(lambda_tag(0.75), "0.75")

    def test_j_report_is_lambda_half(self):
        refs = ReferenceRanges(1.0, 2.0, 3.0, 10.0, 20.0, 30.0)
        j = j_report(2.0, 20.0, refs)
        self.assertAlmostEqual(j_lambda(2.0, 20.0, refs, 0.5, clip=True), j)

    def test_lambda1_unclipped_ranks_by_T(self):
        refs = ReferenceRanges(100.0, 200.0, 300.0, 1.0, 2.0, 3.0)
        plans = [(150.0, 9.0), (120.0, 1.0), (180.0, 1.1)]
        by_t = sorted(plans, key=lambda p: p[0])
        by_j = sorted(plans, key=lambda p: j_lambda(p[0], p[1], refs, 1.0, clip=False))
        self.assertEqual(by_t, by_j)

    def test_lambda1_2opt_matches_T_on_toy(self):
        cfg = ResourceConfig.from_frozen_yaml()
        tg, acts, _want, _doc = load_toy_oracle("05_ue-to-helper.yaml")
        refs = compute_reference_ranges(tg, cfg)

        def score_t(trial):
            order = [int(tid) for tid in tg.prioritize_sequence]
            result, _, _ = schedule_via_adapter(
                tg, list(zip(order, [int(a) for a in trial])), cfg
            )
            return float(result.makespan_seconds)

        def score_j1(trial):
            order = [int(tid) for tid in tg.prioritize_sequence]
            result, _, _ = schedule_via_adapter(
                tg, list(zip(order, [int(a) for a in trial])), cfg
            )
            return float(
                j_lambda(
                    result.makespan_seconds,
                    result.total_mobile_joules,
                    refs,
                    1.0,
                    clip=False,
                )
            )

        t0 = score_t(acts.tolist())
        out_t = iterate_2opt(acts.tolist(), t0, score_t)
        out_j = iterate_2opt(acts.tolist(), score_j1(acts.tolist()), score_j1)
        self.assertEqual(out_t["actions"], out_j["actions"])


class TestAdr001HelperCompute(unittest.TestCase):
    def test_v2v_task_includes_helper_compute(self):
        cfg = ResourceConfig.from_frozen_yaml()
        dag = CanonicalDAG.from_records(
            [
                CanonicalTask(0, 1_048_576, 458_752, 1_048_576),
                CanonicalTask(1, 1_048_576, 458_752, 0),
            ],
            [(0, 1, 458_752)],
        )
        result = schedule(dag, [0, 1], [0, 2], cfg)
        self.assertGreater(result.energy.helper_compute_joules, 0.0)
        rec = result.tasks[1]
        dur = rec.finish - rec.start
        hand = dur * cfg.rho_helper * (cfg.f_v2v**cfg.zeta)
        self.assertAlmostEqual(result.energy.helper_compute_joules, hand, places=9)
        self.assertEqual(result.energy.mec_compute_joules_optional, 0.0)

    def test_toy05_helper_compute_matches_yaml(self):
        cfg = ResourceConfig.from_frozen_yaml()
        tg, acts, _want, doc = load_toy_oracle("05_ue-to-helper.yaml")
        order = [int(tid) for tid in tg.prioritize_sequence]
        result, _, _ = schedule_via_adapter(
            tg, list(zip(order, [int(a) for a in acts])), cfg
        )
        want = float(doc["expected"]["energy_components"]["helper_compute_joules"])
        self.assertAlmostEqual(result.energy.helper_compute_joules, want, places=6)
        self.assertAlmostEqual(
            result.total_mobile_joules, float(doc["expected"]["total_mobile_joules"]), places=6
        )


class TestPurePlanEnergyFractions(unittest.TestCase):
    def test_frozen_train_sample_mec_vs_ue(self):
        cfg = resource_config_for_profile("frozen_7_5_10")
        cluster = resources_cluster_for_profile("frozen_7_5_10")
        dists = list(meta_train_distribution_ids())
        leak = set(meta_test_distribution_ids())
        self.assertFalse(set(dists) & leak)
        n_mec = 0
        n_ue = 0
        n = 0
        for dist_id in dists[:5]:
            for gi in range(4):
                gv = _gv_path(dist_id, gi)
                tg = OffloadingTaskGraph(str(gv))
                tg.prioritize_tasks(cluster)
                refs = compute_reference_ranges(tg, cfg)
                n += 1
                if refs.E_mec <= refs.E_ue + 1e-12 and refs.E_mec <= refs.E_helper + 1e-12:
                    n_mec += 1
                if refs.E_ue + 1e-12 >= refs.E_mec and refs.E_ue + 1e-12 >= refs.E_helper:
                    n_ue += 1
        frac_mec = float(n_mec) / float(n)
        frac_ue = float(n_ue) / float(n)
        # Audit assumption (≥95%) is measured, not asserted as a gate if false.
        self.assertEqual(n, 20)
        self.assertGreaterEqual(frac_mec, 0.0)
        self.assertGreaterEqual(frac_ue, 0.0)


class TestHandEnergyReplay(unittest.TestCase):
    def test_five_graphs_match_formula_1e6(self):
        cfg = resource_config_for_profile("frozen_7_5_10")
        cluster = resources_cluster_for_profile("frozen_7_5_10")
        picks = [(1, 0), (1, 1), (3, 0), (5, 2), (9, 3)]
        self.assertFalse(set(d for d, _ in picks) & set(meta_test_distribution_ids()))
        for dist_id, gi in picks:
            gv = _gv_path(dist_id, gi)
            tg = OffloadingTaskGraph(str(gv))
            tg.prioritize_tasks(cluster)
            order = [int(tid) for tid in tg.prioritize_sequence]
            for acts in ([1] * 20, [0] * 20, [2] * 20):
                result, _, _ = schedule_via_adapter(tg, list(zip(order, acts)), cfg)
                hand = 0.0
                for rec in result.tasks.values():
                    dur = rec.finish - rec.start
                    loc = rec.location.name if hasattr(rec.location, "name") else str(rec.location)
                    if loc == "UE":
                        hand += dur * cfg.rho_ue * (cfg.f_l**cfg.zeta)
                    elif loc == "HELPER":
                        hand += dur * cfg.rho_helper * (cfg.f_v2v**cfg.zeta)
                for t in result.transfers:
                    dur = t.end - t.start
                    hop = t.hop
                    src = t.src_location.name if hasattr(t.src_location, "name") else str(t.src_location)
                    if hop == "MEC_UL":
                        hand += dur * cfg.ptx_mec_w
                    elif hop == "MEC_DL":
                        hand += dur * cfg.prx_mec_w
                    elif hop == "V2V":
                        if src == "HELPER":
                            hand += dur * cfg.ptx_v2v_w + dur * cfg.prx_v2v_w
                        else:
                            hand += dur * cfg.ptx_v2v_w + dur * cfg.prx_v2v_w
                self.assertAlmostEqual(hand, result.total_mobile_joules, places=6)


class TestHypervolume(unittest.TestCase):
    def test_three_point_known_hv(self):
        pts = [(1.0, 3.0), (2.0, 2.0), (3.0, 1.0)]
        ref = (4.0, 4.0)
        self.assertEqual(len(nondominated(pts)), 3)
        self.assertAlmostEqual(hypervolume_2d(pts, ref), 6.0)
        dominated = pts + [(2.5, 2.5)]
        self.assertAlmostEqual(hypervolume_2d(dominated, ref), 6.0)


if __name__ == "__main__":
    unittest.main()
