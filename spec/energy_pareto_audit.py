#!/usr/bin/env python3
"""Pure-policy / Pareto audit for the physical energy model (CPU, numpy).

Answers the ① stop-condition question with numbers:

    After moving MEC compute energy inside the objective, does all-MEC still
    dominate every other pure plan on (T, E), or did a genuine trade-off appear?

Plan families scored (each by the canonical `schedule()`):
    all_UE, all_MEC, all_HELPER, greedy_from_mec (2 local-search passes), 2-opt
    (H1+H2 local search on makespan, started from greedy).

Reported per family: makespan, E_requester, E_mobile, E_system, and the implied
dynamic power kappa*f^3 per tier.  Dominance and Pareto-front membership are
counted across graphs and across all three accounting boundaries.

Usage:
    python spec/energy_pareto_audit.py --graphs 20 --dist 1
    python spec/energy_pareto_audit.py --graphs 10 --preset zhao --json out.json
"""

from __future__ import annotations

import argparse
import copy
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
    greedy_from_mec_plan,
    pure_location_plan,
    schedule_via_adapter,
)
from env.mec_offloaing_envs.scheduler.energy_model import (  # noqa: E402
    SCOPE_MOBILE,
    SCOPE_REQUESTER,
    SCOPE_SYSTEM,
)

DEFAULT_DATA = ROOT / "env" / "mec_offloaing_envs" / "data" / "meta_offloading_20"
MBPS_TO_BPS = 1024.0 * 1024.0 / 8.0
FAMILIES = ("all_UE", "all_MEC", "all_HELPER", "greedy_from_mec", "twopt")

# Parameter sensitivities — every value is a literature-verified anchor or an
# explicitly flagged assumption (see spec/frozen_experiment.yaml).
PRESETS = {
    "primary": {},                                        # Liu2023 pair as frozen
    "zhao": {"tiers": {"mec": {"kappa": 1.0e-28}}},       # Zhao2018 RSU kappa
    "f6": {"tiers": {"mec": {"f_hz": 6.0e9}}},            # Gu2025 RSU frequency
    "f4": {"tiers": {"mec": {"f_hz": 4.0e9}}},            # Zhang2022 MEC frequency
    "xi500": {"cycles_per_bit": 500.0},
    "xi1000": {"cycles_per_bit": 1000.0},                 # Michailidis2021 / Mao2016
    "ue28": {"tiers": {"ue": {"kappa": 1.0e-28}}},        # kappa_UE sweep low
    "ue26": {"tiers": {"ue": {"kappa": 1.0e-26}}},        # kappa_UE sweep high
    "rx": {"include_rx_energy": True, "radio": {"ue_rx_w": 0.2, "helper_rx_w": 0.1}},
}


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


def _apply_preset(doc: dict, preset: str) -> dict:
    out = copy.deepcopy(doc)
    patch = PRESETS[preset]
    for key, value in patch.items():
        if key == "tiers":
            out.setdefault(key, {})
            for tier, spec in value.items():
                out[key].setdefault(tier, {})
                out[key][tier].update(spec)
        elif key == "radio":
            out.setdefault(key, {})
            out[key].update(value)
        else:
            out[key] = value
    return out


def _resources_with(doc: dict) -> ResourceConfig:
    import yaml

    path = ROOT / "spec" / "frozen_experiment.yaml"
    base = yaml.safe_load(path.read_text())
    merged = dict(base)
    merged["energy_model"] = doc
    tmp = Path("/tmp/margo_energy_preset.yaml")
    tmp.write_text(yaml.safe_dump(merged))
    return ResourceConfig.from_frozen_yaml(tmp, model=None)


def _families(tg, resources, order):
    """Return {family: (T, E_req, E_mob, E_sys)}."""
    out = {}
    for action, label in ((0, "all_UE"), (1, "all_MEC"), (2, "all_HELPER")):
        res, _, _ = schedule_via_adapter(tg, pure_location_plan(order, action), resources)
        out[label] = res
    greedy_plan, greedy_res = greedy_from_mec_plan(tg, resources, max_passes=2)
    out["greedy_from_mec"] = greedy_res
    try:
        from spec.twopt_expert import iterate_2opt

        actions0 = [int(a) for _, a in greedy_plan]

        def score(actions):
            res, _, _ = schedule_via_adapter(
                tg, list(zip(order, [int(a) for a in actions])), resources
            )
            return float(res.makespan_seconds)

        report = iterate_2opt(actions0, greedy_res.makespan_seconds, score)
        best_actions = [int(a) for a in report["actions"]]
        res2, _, _ = schedule_via_adapter(
            tg, list(zip(order, best_actions)), resources
        )
        out["twopt"] = res2
    except Exception as exc:  # pragma: no cover - optional family
        print("[warn] 2-opt family unavailable: %s" % exc)
    return out


def _metrics(res):
    e = res.energy
    return {
        "T": float(res.makespan_seconds),
        "E_requester": float(e.total_requester_joules),
        "E_mobile": float(e.total_mobile_joules),
        "E_system": float(e.total_system_joules),
        "E_mec_compute": float(e.mec_compute_joules_optional),
        "E_mec_tx": float(e.mec_tx_joules_optional),
    }


SCOPE_KEY = {
    SCOPE_REQUESTER: "E_requester",
    SCOPE_MOBILE: "E_mobile",
    SCOPE_SYSTEM: "E_system",
}


def _dominates(a, b):
    """a dominates b on (T, E): no worse in both, strictly better in one."""
    return (
        a["T"] <= b["T"] + 1e-9
        and a["E"] <= b["E"] + 1e-9
        and (a["T"] < b["T"] - 1e-9 or a["E"] < b["E"] - 1e-9)
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dist", type=int, default=1)
    ap.add_argument("--graphs", type=int, default=20)
    ap.add_argument("--preset", choices=sorted(PRESETS), default="primary")
    ap.add_argument("--scope", choices=[SCOPE_REQUESTER, SCOPE_MOBILE, SCOPE_SYSTEM],
                    default=SCOPE_SYSTEM)
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()

    import yaml

    doc = yaml.safe_load((ROOT / "spec" / "frozen_experiment.yaml").read_text())
    model_doc = _apply_preset(doc["energy_model"], args.preset)
    model_doc["energy_scope"] = args.scope
    resources = _resources_with(model_doc)
    spec = resources.energy_model

    folder = DEFAULT_DATA / f"offload_random20_{args.dist}"
    files = sorted(folder.glob("random.20.*.gv"))[: args.graphs]
    if not files:
        print("no graphs in %s" % folder, file=sys.stderr)
        return 2

    cluster = _FrozenCluster()
    rows = []
    for path in files:
        tg = OffloadingTaskGraph(str(path))
        tg.prioritize_tasks(cluster)
        order = [int(t) for t in tg.prioritize_sequence]
        fams = _families(tg, resources, order)
        rows.append({"graph": path.name, "families": {k: _metrics(v) for k, v in fams.items()}})

    key = SCOPE_KEY[args.scope]
    names = [f for f in FAMILIES if all(f in r["families"] for r in rows)]

    print("=" * 84)
    print(
        "ENERGY PARETO AUDIT — model=%s preset=%s scope=%s dist=%s n=%d"
        % (spec.model, args.preset, args.scope, args.dist, len(rows))
    )
    print("=" * 84)
    print("\nimplied dynamic power P = kappa*f^3 per tier:")
    for tier in ("ue", "helper", "mec"):
        ts = spec.tier(tier)
        print(
            "  %-7s f=%6.2f GHz  kappa=%.1e  P=%8.2f W   (rate=%9.3e B/s)"
            % (tier, ts.f_hz / 1e9, ts.kappa, ts.implied_dynamic_power_w,
               ts.cpu_rate_bytes_per_second(spec.cycles_per_bit))
        )
    print("\nmedians over %d graphs (%s energy):" % (len(rows), args.scope))
    print("  %-15s %10s %14s %14s %14s" % ("family", "T (s)", key, "E_mec_cpu", "E_mec_tx"))
    for name in names:
        T = statistics.median([r["families"][name]["T"] for r in rows])
        E = statistics.median([r["families"][name][key] for r in rows])
        Ec = statistics.median([r["families"][name]["E_mec_compute"] for r in rows])
        Et = statistics.median([r["families"][name]["E_mec_tx"] for r in rows])
        print("  %-15s %10.2f %14.4g %14.4g %14.4g" % (name, T, E, Ec, Et))

    print("\ndominance counts on (T, %s) — how often row dominates column:" % key)
    header = "  %-15s" % "" + "".join("%13s" % n[:13] for n in names)
    print(header)
    dom = {}
    for a in names:
        line = "  %-15s" % a
        for b in names:
            if a == b:
                line += "%13s" % "-"
                continue
            count = sum(
                1
                for r in rows
                if _dominates(
                    {"T": r["families"][a]["T"], "E": r["families"][a][key]},
                    {"T": r["families"][b]["T"], "E": r["families"][b][key]},
                )
            )
            dom[f"{a}>{b}"] = count
            line += "%13s" % ("%d/%d" % (count, len(rows)))
        print(line)

    front = {n: 0 for n in names}
    for r in rows:
        pts = {n: {"T": r["families"][n]["T"], "E": r["families"][n][key]} for n in names}
        for n in names:
            if not any(_dominates(pts[m], pts[n]) for m in names if m != n):
                front[n] += 1
    print("\nPareto-front membership (graph counts):")
    for n in names:
        print("  %-15s %d/%d" % (n, front[n], len(rows)))

    verdict = (
        "TRADE-OFF: no family dominates all others"
        if all(front[n] > 0 for n in names)
        else "SOME FAMILY NEVER PARETO-OPTIMAL: %s"
        % ",".join(n for n in names if front[n] == 0)
    )
    # the ① stop-condition question, asked explicitly
    mec_dominates_all = all(
        dom.get("all_MEC>%s" % b, 0) == len(rows) for b in names if b != "all_MEC"
    )
    print(
        "\nMEC domination check (the ① stop condition): all_MEC dominates every "
        "other family in %s" % ("ALL graphs" if mec_dominates_all else "NOT all graphs")
    )
    print("  all_MEC Pareto-optimal in %d/%d graphs" % (front.get("all_MEC", 0), len(rows)))
    for b in names:
        if b != "all_MEC":
            print(
                "  all_MEC > %-15s %d/%d" % (b, dom.get("all_MEC>%s" % b, 0), len(rows))
            )
    print("\nVERDICT: %s" % verdict)

    if args.json:
        Path(args.json).write_text(
            json.dumps(
                {
                    "model": spec.as_dict(),
                    "preset": args.preset,
                    "scope": args.scope,
                    "dist": args.dist,
                    "rows": rows,
                    "dominance": dom,
                    "pareto_front_counts": front,
                    "verdict": verdict,
                },
                indent=2,
            )
        )
        print("raw JSON -> %s" % args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
