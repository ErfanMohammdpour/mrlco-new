#!/usr/bin/env python3
"""Spec-faithful toy scheduler for Phase 0 oracles. Not the production env."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

UE, MEC, HELPER = "UE", "MEC", "HELPER"
LOC = {0: UE, 1: MEC, 2: HELPER, "UE": UE, "MEC": MEC, "HELPER": HELPER}


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


class Cal:
    def __init__(self) -> None:
        self.t = 0.0

    def reserve(self, duration: float, earliest: float) -> tuple[float, float]:
        start = max(self.t, earliest)
        end = start + duration
        self.t = end
        return start, end


def route(pred_loc: str, succ_loc: str) -> list[str]:
    if pred_loc == succ_loc:
        return []
    table = {
        (UE, MEC): ["MEC_UL"],
        (UE, HELPER): ["V2V"],
        (MEC, UE): ["MEC_DL"],
        (MEC, HELPER): ["MEC_DL", "V2V"],
        (HELPER, UE): ["V2V"],
        (HELPER, MEC): ["V2V", "MEC_UL"],
    }
    return table[(pred_loc, succ_loc)]


def hop_rate(hop: str, rates: dict) -> float:
    return {
        "MEC_UL": rates["mec_uplink_bytes_per_second"],
        "MEC_DL": rates["mec_downlink_bytes_per_second"],
        "V2V": rates["v2v_bytes_per_second"],
    }[hop]


def hop_cal(hop: str, cals: dict) -> Cal:
    return {
        "MEC_UL": cals["MEC_UL"],
        "MEC_DL": cals["MEC_DL"],
        "V2V": cals["V2V_CHANNEL"],
    }[hop]


def cpu_rate(loc: str, rates: dict) -> float:
    return {
        UE: rates["ue_cpu_bytes_per_second"],
        MEC: rates["mec_cpu_bytes_per_second"],
        HELPER: rates["helper_cpu_bytes_per_second"],
    }[loc]


def cpu_cal(loc: str, cals: dict) -> Cal:
    return {UE: cals["UE_CPU"], MEC: cals["MEC_CPU"], HELPER: cals["HELPER_CPU"]}[loc]


def schedule(oracle: dict, rates: dict, power: dict, actions: list[str]) -> dict:
    nodes = {int(n["task_id"]): n for n in oracle["nodes"]}
    edges = oracle.get("edges", [])
    preds: dict[int, list[tuple[int, int]]] = {tid: [] for tid in nodes}
    succs: dict[int, list[int]] = {tid: [] for tid in nodes}
    seen = set()
    for e in edges:
        key = (int(e["src_task_id"]), int(e["dst_task_id"]), int(e["edge_output_bytes"]))
        if key in seen:
            continue
        seen.add(key)
        preds[key[1]].append((key[0], key[2]))
        succs[key[0]].append(key[1])

    order = []
    indeg = {tid: len(preds[tid]) for tid in nodes}
    stack = [tid for tid, d in indeg.items() if d == 0]
    while stack:
        u = stack.pop()
        order.append(u)
        for v in succs[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                stack.append(v)
    if len(order) != len(nodes):
        raise ValueError("cyclic toy graph")

    locs = {tid: LOC[actions[i]] for i, tid in enumerate(sorted(nodes))}
    if list(sorted(nodes)) != sorted(nodes):
        pass
    # actions aligned to sorted task_id
    tids = sorted(nodes)
    if len(actions) != len(tids):
        raise ValueError("action plan length mismatch")
    locs = {tid: LOC[actions[i]] for i, tid in enumerate(tids)}

    cals = {k: Cal() for k in ["UE_CPU", "MEC_UL", "MEC_CPU", "MEC_DL", "HELPER_CPU", "V2V_CHANNEL"]}
    energy = {
        "ue_local_cpu_joules": 0.0,
        "ue_mec_uplink_joules": 0.0,
        "ue_mec_downlink_joules": 0.0,
        "ue_v2v_tx_joules": 0.0,
        "ue_v2v_rx_joules": 0.0,
        "helper_compute_joules": 0.0,
        "helper_v2v_tx_joules": 0.0,
        "helper_v2v_rx_joules": 0.0,
    }
    finish = {}
    start = {}
    loc_out = {}
    reservations = []
    transfers = []

    def add_energy_hop(hop: str, duration: float, src_loc: str, dst_loc: str) -> None:
        if hop == "MEC_UL":
            energy["ue_mec_uplink_joules"] += duration * power["ptx_mec_w"]
        elif hop == "MEC_DL":
            energy["ue_mec_downlink_joules"] += duration * power["prx_mec_w"]
        elif hop == "V2V":
            # sender then receiver
            if src_loc == UE:
                energy["ue_v2v_tx_joules"] += duration * power["ptx_v2v_w"]
                energy["helper_v2v_rx_joules"] += duration * power["prx_v2v_w"]
            elif src_loc == HELPER:
                energy["helper_v2v_tx_joules"] += duration * power["ptx_v2v_w"]
                energy["ue_v2v_rx_joules"] += duration * power["prx_v2v_w"]
            else:
                # two-hop middle: MEC_DL already billed; V2V after DL is UE->HELPER
                energy["ue_v2v_tx_joules"] += duration * power["ptx_v2v_w"]
                energy["helper_v2v_rx_joules"] += duration * power["prx_v2v_w"]

    def move_bytes(nbytes: int, hops: list[str], earliest: float, src_loc: str) -> float:
        t = earliest
        cur = src_loc
        for hop in hops:
            dur = nbytes / hop_rate(hop, rates)
            s, e = hop_cal(hop, cals).reserve(dur, t)
            reservations.append({"resource": hop if hop != "V2V" else "V2V_CHANNEL", "start": s, "end": e})
            dst = HELPER if hop == "V2V" and cur == UE else (UE if hop == "V2V" and cur == HELPER else (MEC if hop == "MEC_UL" else UE))
            if hop == "MEC_UL":
                dst = MEC
            elif hop == "MEC_DL":
                dst = UE
            elif hop == "V2V":
                dst = HELPER if cur == UE else UE
            add_energy_hop(hop, dur, cur, dst)
            transfers.append({"hop": hop, "bytes": nbytes, "start": s, "end": e})
            t = e
            cur = dst
        return t

    for tid in order:
        node = nodes[tid]
        loc = locs[tid]
        ready = 0.0
        # root external input from UE
        ext = int(node.get("external_input_bytes", 0))
        if ext > 0:
            hops = route(UE, loc)
            ready = max(ready, move_bytes(ext, hops, 0.0, UE))
        for src, nbytes in preds[tid]:
            hops = route(loc_out[src], loc)
            ready = max(ready, move_bytes(nbytes, hops, finish[src], loc_out[src]))
        dur = int(node["compute_workload_bytes"]) / cpu_rate(loc, rates)
        s, e = cpu_cal(loc, cals).reserve(dur, ready)
        reservations.append({"resource": f"{loc}_CPU", "start": s, "end": e, "task_id": tid})
        start[tid] = s
        finish[tid] = e
        loc_out[tid] = loc
        if loc == UE:
            energy["ue_local_cpu_joules"] += dur * power["rho_ue"] * (power["f_l"] ** power["zeta"])
        elif loc == HELPER:
            energy["helper_compute_joules"] += dur * power["rho_helper"] * (power["f_v2v"] ** power["zeta"])

    sinks = [tid for tid in nodes if not succs[tid]]
    result_at_ue = 0.0
    for tid in sinks:
        out_b = int(nodes[tid]["task_output_bytes"])
        hops = route(loc_out[tid], UE)
        result_at_ue = max(result_at_ue, move_bytes(out_b, hops, finish[tid], loc_out[tid]))
        if not hops:
            result_at_ue = max(result_at_ue, finish[tid])

    total_ue = (
        energy["ue_local_cpu_joules"]
        + energy["ue_mec_uplink_joules"]
        + energy["ue_mec_downlink_joules"]
        + energy["ue_v2v_tx_joules"]
        + energy["ue_v2v_rx_joules"]
    )
    total_helper = (
        energy["helper_compute_joules"]
        + energy["helper_v2v_tx_joules"]
        + energy["helper_v2v_rx_joules"]
    )
    return {
        "start": start,
        "finish": finish,
        "output_location": loc_out,
        "terminal_return_time": result_at_ue,
        "makespan_seconds": result_at_ue,
        "energy": energy,
        "total_ue_joules": total_ue,
        "total_helper_joules": total_helper,
        "total_mobile_joules": total_ue + total_helper,
        "reservations": reservations,
        "transfers": transfers,
    }


def almost(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(b)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle-dir", default=str(Path(__file__).parent))
    args = ap.parse_args()
    oracle_dir = Path(args.oracle_dir)
    frozen = yaml.safe_load((oracle_dir.parent / "frozen_experiment.yaml").read_text())
    rates = frozen["resource_rates"]
    power = frozen["power"]
    failed = 0
    for path in sorted(oracle_dir.glob("*.yaml")):
        if path.name in {"resources.yaml"}:
            continue
        doc = load_yaml(path)
        actions = [LOC[a] for a in doc["actions"]]
        got = schedule(doc, rates, power, actions)
        exp = doc.get("expected", {})
        ok = True
        msgs = []
        if "makespan_seconds" in exp and not almost(got["makespan_seconds"], exp["makespan_seconds"]):
            ok = False
            msgs.append(f"makespan got={got['makespan_seconds']} expected={exp['makespan_seconds']}")
        if "total_mobile_joules" in exp and not almost(got["total_mobile_joules"], exp["total_mobile_joules"]):
            ok = False
            msgs.append(f"energy got={got['total_mobile_joules']} expected={exp['total_mobile_joules']}")
        status = "PASS" if ok else "FAIL"
        if not ok:
            failed += 1
        print(f"{status} {path.name} makespan={got['makespan_seconds']:.9g} E={got['total_mobile_joules']:.9g}")
        for m in msgs:
            print("  ", m)
    if failed:
        raise SystemExit(failed)
    print("ALL toy oracles with expected fields PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
