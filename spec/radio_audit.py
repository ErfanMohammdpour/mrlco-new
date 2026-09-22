#!/usr/bin/env python3
"""④ RADIO_MODEL_V1 audit: link parameters, effective rates, and the effect of
switching the radio model on transfer times and pure-plan makespans.

Usage:
    python spec/radio_audit.py --graphs 8 --dist 1
    python spec/radio_audit.py --graphs 8 --dist 1 --json out.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env.mec_offloaing_envs.offloading_task_graph import OffloadingTaskGraph  # noqa: E402
from env.mec_offloaing_envs.scheduler import (  # noqa: E402
    ResourceConfig,
    pure_location_plan,
    schedule_via_adapter,
)
from env.mec_offloaing_envs.scheduler.radio import (  # noqa: E402
    HOP_TO_LINK,
    RadioModelSpec,
)

DEFAULT_DATA = ROOT / "env" / "mec_offloaing_envs" / "data" / "meta_offloading_20"
MBPS_TO_BPS = 1024.0 * 1024.0 / 8.0


class _FrozenCluster:
    def __init__(self):
        self.mobile_process_capable = 1.0 * 1024 * 1024
        self.mec_process_capable = 10.0 * 1024 * 1024
        self.v2v_process_capable = 1.0 * 1024 * 1024
        self.bandwidth_up = 7.0
        self.bandwidth_dl = 7.0
        self.v2v_bandwidth = 5.0

    def up_transmission_cost(self, data):
        return data / (self.bandwidth_up * MBPS_TO_BPS)

    def dl_transmission_cost(self, data):
        return data / (self.bandwidth_dl * MBPS_TO_BPS)

    def v2v_transmission_cost(self, data):
        return data / (self.v2v_bandwidth * MBPS_TO_BPS)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dist", type=int, default=1)
    ap.add_argument("--graphs", type=int, default=8)
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()

    legacy = ResourceConfig.from_frozen_yaml(model="legacy")
    # radio-only: legacy compute physics, physical radio (isolates the radio change)
    radio_only = ResourceConfig.from_frozen_yaml(
        model="legacy", radio_model="physical_v1"
    )
    physical = ResourceConfig.from_frozen_yaml(model=None)

    print("=" * 80)
    print("RADIO MODEL AUDIT — legacy vs physical_v1")
    print("=" * 80)

    rows = {}
    print("\nper-link effective rate:")
    print("  %-8s %14s %10s %14s %14s" % ("hop", "bandwidth_hz", "eta", "physical B/s", "legacy B/s"))
    for hop, link in HOP_TO_LINK.items():
        spec = physical.radio_model.link(link)
        rows[hop] = {
            **spec.as_dict(),
            "legacy_bytes_per_second": legacy.hop_rate(hop),
            "physical_bytes_per_second": physical.hop_rate(hop),
            "speedup": physical.hop_rate(hop) / legacy.hop_rate(hop),
        }
        print(
            "  %-8s %14.3e %10s %14.6e %14.6e   (x%.3f)"
            % (hop, spec.bandwidth_hz, spec.spectral_efficiency,
               physical.hop_rate(hop), legacy.hop_rate(hop), rows[hop]["speedup"])
        )
    print("\n  assumed links (not literature-pinned): %s" % (physical.radio_model.assumptions() or "none"))

    # transfer time for a reference payload
    payload = 1_048_576
    print("\ntransfer time for %.3f MiB (s):" % (payload / 2**20))
    for hop in HOP_TO_LINK:
        t_leg = payload / legacy.hop_rate(hop)
        t_phys = payload / physical.hop_rate(hop)
        rows[hop]["transfer_s_legacy"] = t_leg
        rows[hop]["transfer_s_physical"] = t_phys
        print("  %-8s legacy %8.3f   physical %8.3f   (x%.3f)" % (hop, t_leg, t_phys, t_leg / t_phys))

    folder = DEFAULT_DATA / f"offload_random20_{args.dist}"
    files = sorted(folder.glob("random.20.*.gv"))[: args.graphs]
    makespans = {
        "legacy": {0: [], 1: [], 2: []},
        "radio_only": {0: [], 1: [], 2: []},
        "physical": {0: [], 1: [], 2: []},
    }
    if files:
        cluster = _FrozenCluster()
        for path in files:
            tg = OffloadingTaskGraph(str(path))
            tg.prioritize_tasks(cluster)
            order = [int(t) for t in tg.prioritize_sequence]
            for action in (0, 1, 2):
                for name, res in (
                    ("legacy", legacy),
                    ("radio_only", radio_only),
                    ("physical", physical),
                ):
                    out, _, _ = schedule_via_adapter(tg, pure_location_plan(order, action), res)
                    makespans[name][action].append(out.makespan_seconds)
        print("\npure-plan makespan medians (s), n=%d graphs:" % len(files))
        print("  %-10s %10s %10s %10s" % ("model", "all_UE", "all_MEC", "all_HELPER"))
        for name in ("legacy", "radio_only", "physical"):
            vals = [statistics.median(makespans[name][a]) for a in (0, 1, 2)]
            print("  %-10s %10.2f %10.2f %10.2f" % (name, *vals))
            rows[f"median_makespan_{name}"] = {
                "all_UE": vals[0], "all_MEC": vals[1], "all_HELPER": vals[2]
            }
        ratio = [
            statistics.median(makespans["legacy"][a]) / statistics.median(makespans["radio_only"][a])
            for a in (0, 1, 2)
        ]
        print("  radio-only speedup legacy/physical: UE x%.3f  MEC x%.3f  HELPER x%.3f" % tuple(ratio))
        rows["radio_only_makespan_speedup"] = {
            "all_UE": ratio[0], "all_MEC": ratio[1], "all_HELPER": ratio[2]
        }
    else:
        print("\n[skip] dataset not present")

    print(
        "\nNOTE: this changes deadline feasibility. Before freezing, the ③ suite "
        "(LB admissibility, mask soundness, suffix feasibility, potential) is "
        "re-run under physical_v1 — see tests/test_radio_model.py."
    )
    if args.json:
        Path(args.json).write_text(
            json.dumps({"radio": physical.radio_model.as_dict(), "audit": rows}, indent=2)
        )
        print("raw JSON -> %s" % args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
