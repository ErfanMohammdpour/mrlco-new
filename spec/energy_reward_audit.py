#!/usr/bin/env python3
"""Energy / reward logic audit for MARGO-SPEC-v0.1 (CPU only, numpy).

Answers, with measured numbers instead of opinion:

  A. Are the three pure-location reference plans ordered the way the spec assumes,
     and is `E_ref_min` really all-MEC?
  B. Do ordinary mixed plans fall OUTSIDE the pure-location reference range
     `[L_ref_min, L_ref_max] x [E_ref_min, E_ref_max]`?  (If yes, `j_report`
     saturates at 0 for exactly the plans the policy is supposed to find.)
  C. What are the real magnitudes of the two reward terms
     `0.5*(L-L_ue)/L_scale` and `0.5*(E-E_ue)/E_scale`?  Is the 0.5/0.5 weighting
     what the physics implies, or does one term dominate the gradient?
  D. How small can `L_scale` / `E_scale` get?  A tiny scale turns token rewards
     into outliers.
  E. Token-credit quality of the telescoping reward: how often does a decided
     prefix make the plan WORSE (dL > 0), and how large is the largest single step?
  F. Energy composition: what fraction of `total_mobile_joules` is compute vs radio,
     and what is the MEC/UE energy ratio that biases the policy toward MEC?

Usage:
    python spec/energy_reward_audit.py --graphs 20 --dist 1
    python spec/energy_reward_audit.py --graphs 8 --dist 12 --json out.json
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
    attribute_energy_by_task,
    compute_reference_ranges,
    greedy_from_mec_plan,
    j_report,
    pure_location_plan,
    schedule_via_adapter,
    telescoping_token_rewards,
)
from env.mec_offloaing_envs.scheduler.energy_scope import (  # noqa: E402
    SCOPE_MOBILE,
    energy_scalar,
)

DEFAULT_DATA = ROOT / "env" / "mec_offloaing_envs" / "data" / "meta_offloading_20"
MBPS_TO_BPS = 1024.0 * 1024.0 / 8.0


class _FrozenCluster:
    """Minimal HEFT-priority cluster mirroring env Resources at frozen rates.

    Only the four attributes `OffloadingTaskGraph.prioritize_tasks` reads.
    """

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


def _plan_metrics(task_graph, resources, actions_by_pos):
    order = [int(tid) for tid in task_graph.prioritize_sequence]
    plan = list(zip(order, [int(a) for a in actions_by_pos]))
    result, _, _ = schedule_via_adapter(task_graph, plan, resources)
    return plan, result


def audit_graph(path: Path, resources: ResourceConfig, cluster: _FrozenCluster) -> dict:
    tg = OffloadingTaskGraph(str(path))
    tg.prioritize_tasks(cluster)
    n = int(tg.task_number)
    refs = compute_reference_ranges(tg, resources)

    # -- A: pure-location ordering -------------------------------------------
    pure_T = {"UE": refs.L_ue, "MEC": refs.L_mec, "HELPER": refs.L_helper}
    pure_E = {"UE": refs.E_ue, "MEC": refs.E_mec, "HELPER": refs.E_helper}
    t_argmin = min(pure_T, key=pure_T.get)
    e_argmin = min(pure_E, key=pure_E.get)

    # -- B/C: mixed plans vs the pure-location range -------------------------
    order = [int(tid) for tid in tg.prioritize_sequence]
    rows = {}
    _, pure_mec = _plan_metrics(tg, resources, [1] * n)
    rows["all_MEC"] = pure_mec
    _, g_result = greedy_from_mec_plan(tg, resources, max_passes=2)
    rows["greedy_from_mec"] = g_result

    plan_metrics = {}
    for name, result in rows.items():
        L = float(result.makespan_seconds)
        E = energy_scalar(result, scope=SCOPE_MOBILE)
        lat_term = 0.5 * (L - refs.L_ue) / refs.L_scale
        en_term = 0.5 * (E - refs.E_ue) / refs.E_scale
        plan_metrics[name] = {
            "L": L,
            "E": E,
            "lat_term": lat_term,
            "en_term": en_term,
            "below_L_ref_min": L < refs.L_ref_min - 1e-9,
            "below_E_ref_min": E < refs.E_ref_min - 1e-9,
            "j_report_clipped": j_report(L, E, refs),
            "j_report_unclipped": lat_term + en_term,
        }

    # -- D/E: token-reward behaviour on the greedy plan ----------------------
    g_plan, _ = greedy_from_mec_plan(tg, resources, max_passes=2)
    tel = telescoping_token_rewards(tg, g_plan, resources)
    dL = [tel.makespans[i] - tel.makespans[i - 1] for i in range(1, len(tel.makespans))]
    dE = [tel.energies[i] - tel.energies[i - 1] for i in range(1, len(tel.energies))]

    # -- F: energy composition of each pure plan -----------------------------
    energy_mix = {}
    for action, label in ((0, "UE"), (1, "MEC"), (2, "HELPER")):
        res, _, _ = schedule_via_adapter(
            tg, pure_location_plan([int(t) for t in tg.prioritize_sequence], action), resources
        )
        bd = attribute_energy_by_task(res, resources)
        energy_mix[label] = {
            "total": energy_scalar(res, scope=SCOPE_MOBILE),
            "sum_by_task": float(sum(bd.values())),
        }

    return {
        "graph": path.name,
        "tasks": n,
        "refs": {
            "L_ue": refs.L_ue,
            "L_mec": refs.L_mec,
            "L_helper": refs.L_helper,
            "E_ue": refs.E_ue,
            "E_mec": refs.E_mec,
            "E_helper": refs.E_helper,
            "L_scale": refs.L_scale,
            "E_scale": refs.E_scale,
            "L_ref_min": refs.L_ref_min,
            "L_ref_max": refs.L_ref_max,
            "E_ref_min": refs.E_ref_min,
            "E_ref_max": refs.E_ref_max,
        },
        "pure_argmin": {"latency": t_argmin, "energy": e_argmin},
        "plans": plan_metrics,
        "telescoping": {
            "return_sum": float(sum(tel.rewards)),
            "closed_form": float(
                -(
                    0.5 * (tel.makespans[-1] - tel.makespans[0]) / refs.L_scale
                    + 0.5 * (tel.energies[-1] - tel.energies[0]) / refs.E_scale
                )
            ),
            "steps_regressing_L": int(sum(1 for d in dL if d > 1e-9)),
            "steps_regressing_E": int(sum(1 for d in dE if d > 1e-9)),
            "max_abs_dL_over_Lscale": max((abs(d) / refs.L_scale for d in dL), default=0.0),
            "max_abs_dE_over_Escale": max((abs(d) / refs.E_scale for d in dE), default=0.0),
        },
        "energy_mix": energy_mix,
    }


def _pct(x, total):
    return 100.0 * float(x) / float(total) if total else 0.0


def main() -> int:
    import logging

    logging.getLogger("env.mec_offloaing_envs.scheduler.energy_api").setLevel(logging.ERROR)

    ap = argparse.ArgumentParser()
    ap.add_argument("--dist", type=int, default=1, help="distribution id (offload_random20_<id>)")
    ap.add_argument("--graphs", type=int, default=20, help="how many .gv files to audit")
    ap.add_argument("--json", type=str, default=None, help="write raw audit JSON here")
    args = ap.parse_args()

    folder = DEFAULT_DATA / f"offload_random20_{args.dist}"
    files = sorted(folder.glob("random.20.*.gv"))[: args.graphs]
    if not files:
        print(f"no graphs in {folder}", file=sys.stderr)
        return 2

    resources = ResourceConfig.from_frozen_yaml()
    cluster = _FrozenCluster()
    rows = [audit_graph(p, resources, cluster) for p in files]

    n_below_L = sum(1 for r in rows if r["plans"]["greedy_from_mec"]["below_L_ref_min"])
    n_below_E = sum(1 for r in rows if r["plans"]["greedy_from_mec"]["below_E_ref_min"])
    lat_terms = [r["plans"]["greedy_from_mec"]["lat_term"] for r in rows]
    en_terms = [r["plans"]["greedy_from_mec"]["en_term"] for r in rows]
    j_clip = [r["plans"]["greedy_from_mec"]["j_report_clipped"] for r in rows]
    j_unclip = [r["plans"]["greedy_from_mec"]["j_report_unclipped"] for r in rows]

    print("=" * 78)
    print(f"ENERGY / REWARD AUDIT — dist offload_random20_{args.dist}, n={len(rows)} graphs")
    print("=" * 78)

    print("\n[A] which pure location is the reference minimum?")
    for key in ("latency", "energy"):
        counts = {}
        for r in rows:
            counts[r["pure_argmin"][key]] = counts.get(r["pure_argmin"][key], 0) + 1
        print(f"    argmin {key:8s}: {counts}")

    print("\n[B] normal mixed plan (greedy_from_mec) vs the pure-location reference box")
    print(f"    below L_ref_min: {n_below_L}/{len(rows)}  ({_pct(n_below_L, len(rows)):.0f}%)")
    print(f"    below E_ref_min: {n_below_E}/{len(rows)}  ({_pct(n_below_E, len(rows)):.0f}%)")

    print("\n[C] the two reward terms on that plan (negative = improvement over all-UE)")
    print(
        "    lat_term  0.5*(L-L_ue)/L_scale : "
        f"min {min(lat_terms):+8.3f}  median {statistics.median(lat_terms):+8.3f}  max {max(lat_terms):+8.3f}"
    )
    print(
        "    en_term   0.5*(E-E_ue)/E_scale : "
        f"min {min(en_terms):+8.3f}  median {statistics.median(en_terms):+8.3f}  max {max(en_terms):+8.3f}"
    )
    ratio = [abs(a) / abs(b) for a, b in zip(lat_terms, en_terms) if abs(b) > 1e-12]
    if ratio:
        print(
            "    |lat_term|/|en_term|              : "
            f"min {min(ratio):.2f}  median {statistics.median(ratio):.2f}  max {max(ratio):.2f}"
        )
    print(
        "    j_report after clip vs unclipped   : "
        f"mean {statistics.mean(j_clip):+.3f} vs {statistics.mean(j_unclip):+.3f}"
    )

    # -- the decisive check: does the CLIPPED composite rank a better plan worse?
    inversions = 0
    ties = 0
    for r in rows:
        j_mec = r["plans"]["all_MEC"]["j_report_clipped"]
        j_gr = r["plans"]["greedy_from_mec"]["j_report_clipped"]
        t_mec = r["plans"]["all_MEC"]["L"]
        t_gr = r["plans"]["greedy_from_mec"]["L"]
        if t_gr < t_mec:  # greedy is strictly faster
            if j_gr > j_mec + 1e-12:
                inversions += 1
            elif abs(j_gr - j_mec) <= 1e-12:
                ties += 1
    print(
        "\n[C2] clipped J saturation check (all-MEC vs faster mixed plan): "
        f"{inversions}/{len(rows)} rank-worse (faster plan scores WORSE; clip saturation), "
        f"{ties}/{len(rows)} ties"
    )

    print("\n[D] reference-scale distribution (denominators of the reward)")
    for key in ("L_scale", "E_scale"):
        vals = sorted(r["refs"][key] for r in rows)
        print(
            f"    {key}: min {vals[0]:.3f}  p50 {vals[len(vals) // 2]:.3f}  max {vals[-1]:.3f}"
        )

    print("\n[E] telescoping token-credit quality (greedy plan, 20 steps)")
    reg_L = sum(r["telescoping"]["steps_regressing_L"] for r in rows)
    reg_E = sum(r["telescoping"]["steps_regressing_E"] for r in rows)
    steps = 20 * len(rows)
    print(f"    steps where the prefix makes L worse : {reg_L}/{steps} ({_pct(reg_L, steps):.0f}%)")
    print(f"    steps where the prefix makes E worse : {reg_E}/{steps} ({_pct(reg_E, steps):.0f}%)")
    print(
        "    worst single step, |dL|/L_scale      : "
        f"{max(r['telescoping']['max_abs_dL_over_Lscale'] for r in rows):.2f}"
    )
    print(
        "    worst single step, |dE|/E_scale      : "
        f"{max(r['telescoping']['max_abs_dE_over_Escale'] for r in rows):.2f}"
    )

    print("\n[F] pure-plan energy totals (joules) and MEC/UE ratio")
    for label in ("UE", "MEC", "HELPER"):
        vals = [r["energy_mix"][label]["total"] for r in rows]
        print(f"    E_{label:7s}: median {statistics.median(vals):10.3f}")
    ratios = [
        r["energy_mix"]["MEC"]["total"] / r["energy_mix"]["UE"]["total"]
        for r in rows
        if r["energy_mix"]["UE"]["total"] > 1e-12
    ]
    if ratios:
        print(
            "    E_MEC / E_UE                         : "
            f"min {min(ratios):.3f}  median {statistics.median(ratios):.3f}  max {max(ratios):.3f}"
        )

    mismatches = []
    for r in rows:
        for label in ("UE", "MEC", "HELPER"):
            if abs(r["energy_mix"][label]["total"] - r["energy_mix"][label]["sum_by_task"]) > 1e-6:
                mismatches.append((r["graph"], label))
    print(f"\n[G] per-task energy attribution == episode total: {'OK' if not mismatches else mismatches}")

    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2))
        print(f"\nraw JSON -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
