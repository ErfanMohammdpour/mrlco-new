#!/usr/bin/env python3
"""Gate 14 PARITY: the training scheduler must reproduce the certification physics.

For selected frozen graphs and fixed placement plans, the automotive training
configuration is compared against the frozen M10 certification configuration
(`spec/automotive_mc_v1/certify_automotive_m10.py`, imported read-only) on:

    makespan, key task finish/availability times, transfer accounting, sink return

The training config uses the CO-PHYSICAL axes; the certification config used the
frozen legacy radio table with the SAME graph rates. R = B*eta with B = R_graph/eta
makes both consume the identical effective rate, so the schedules must agree.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from spec.automotive_training.automotive_dag import (  # noqa: E402
    canonical_dag, decoder_order,
)
from spec.automotive_training.automotive_env import AutomotiveEnv  # noqa: E402
from spec.automotive_training.automotive_primary import AutomotiveResourceCluster  # noqa: E402
from spec.automotive_training.automotive_resources import (  # noqa: E402
    co_physical_config_for_graph, version_axes,
)
from spec.automotive_training.automotive_loader import load_dataset  # noqa: E402

GRAPHS = ROOT / "env" / "mec_offloaing_envs" / "data" / "automotive_mc_v1" / "graphs.jsonl"
PLANS = {"all_UE": 0, "all_MEC": 1, "all_HELPER": 2}


def _records(limit: int = 6):
    out = []
    for line in GRAPHS.read_text().splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out[:: max(1, len(out) // limit)][:limit]


class TestSchedulerParity(unittest.TestCase):
    def setUp(self):
        from env.mec_offloaing_envs.scheduler.engine import schedule

        self.schedule = schedule
        self.records = _records()

    def _certification_config(self, record):
        import importlib.util

        path = ROOT / "spec" / "automotive_mc_v1" / "certify_automotive_m10.py"
        spec = importlib.util.spec_from_file_location("_frozen_m10_certifier", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["_frozen_m10_certifier"] = module
        spec.loader.exec_module(module)
        return module.to_resources(record)

    def test_makespan_finish_transfer_and_sink_parity(self):
        for record in self.records:
            dag = canonical_dag(record)
            order = decoder_order(record)
            training = co_physical_config_for_graph(
                record["resource"], source_sha256=record["canonical_sha256"])
            frozen = self._certification_config(record)
            for name, action in PLANS.items():
                actions = [action] * len(order)
                a = self.schedule(dag, order, actions, training)
                b = self.schedule(dag, order, actions, frozen)
                self.assertAlmostEqual(a.makespan_seconds, b.makespan_seconds, places=9,
                                       msg="%s/%s makespan" % (record["graph_id"], name))
                self.assertAlmostEqual(a.terminal_return_time, b.terminal_return_time, places=9,
                                       msg="%s/%s sink return" % (record["graph_id"], name))
                self.assertEqual(sorted(a.tasks), sorted(b.tasks))
                for tid in a.tasks:
                    self.assertAlmostEqual(a.tasks[tid].finish, b.tasks[tid].finish, places=9)
                    self.assertAlmostEqual(a.tasks[tid].all_consumers_ready,
                                           b.tasks[tid].all_consumers_ready, places=9)
                self.assertEqual([(t.hop, t.bytes, t.src_location.value, t.dst_location.value)
                                  for t in a.transfers],
                                 [(t.hop, t.bytes, t.src_location.value, t.dst_location.value)
                                  for t in b.transfers])
                for ta, tb in zip(a.transfers, b.transfers):
                    self.assertAlmostEqual(ta.start, tb.start, places=9)
                    self.assertAlmostEqual(ta.end, tb.end, places=9)

    def test_training_axes_are_co_physical_and_graph_specific(self):
        record = self.records[0]
        config = co_physical_config_for_graph(
            record["resource"], source_sha256=record["canonical_sha256"])
        axes = version_axes(config)
        self.assertEqual(axes["timing_model"], "physical_rates")
        self.assertEqual(axes["radio_timing_model"], "physical_rates")
        self.assertEqual(axes["energy_model"], "physical_v1")
        self.assertEqual(axes["radio_model"], "physical_v1")
        self.assertEqual(axes["energy_scope"], "system")
        other = co_physical_config_for_graph(
            self.records[1]["resource"],
            source_sha256=self.records[1]["canonical_sha256"])
        self.assertNotEqual(config.source_config_sha256, other.source_config_sha256)
        for hop, key in (("MEC_UL", "r_mec_ul_bps"), ("MEC_DL", "r_mec_dl_bps"),
                         ("V2V", "r_v2v_bps")):
            self.assertAlmostEqual(config.hop_rate(hop), record["resource"][key] / 8.0,
                                   places=6, msg=hop)

    def test_env_uses_the_graph_specific_config(self):
        from spec.automotive_training.mc_runtime import load_uncertainty

        dataset = load_dataset()
        graphs = dataset.meta_train()[:2]
        env = AutomotiveEnv(graphs, AutomotiveResourceCluster(), slots_per_task=1,
                            base_seed=0, uncertainty=load_uncertainty())
        self.assertNotEqual(env.configs[0].source_config_sha256,
                            env.configs[1].source_config_sha256)
        self.assertEqual(env.orders[0], decoder_order(graphs[0].as_record()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
