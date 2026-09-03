#!/usr/bin/env python3
"""Spec-faithful toy scheduler for Phase 0 oracles. Not the production env."""

from __future__ import annotations

import argparse
import heapq
from pathlib import Path

import yaml

UE, MEC, HELPER = "UE", "MEC", "HELPER"
LOC = {0: UE, 1: MEC, 2: HELPER, "UE": UE, "MEC": MEC, "HELPER": HELPER}

REQUIRED_EXPECTED = {
    "makespan_seconds",
    "total_mobile_joules",
    "total_ue_joules",
    "total_helper_joules",
    "terminal_return_time",
    "task_intervals",
    "transfers",
    "resource_intervals",
    "energy_components",
    "topo_order",
    "output_residency",
}

# Ready-queue / topo-ready tie-break (normative for v0.1 toy oracles):
#   ready_time -> decoder_order -> task_id
# For Kahn topo construction, all sources share ready_time=0; decoder_order is
# the index in the fixed action plan (sorted task_id for these toys).


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


def topo_order(nodes: dict, preds: dict, succs: dict, decoder_order: dict[int, int]) -> list[int]:
    """Deterministic Kahn: always pop min by (decoder_order, task_id)."""
    indeg = {tid: len(preds[tid]) for tid in nodes}
    heap: list[tuple[int, int]] = []
    for tid, d in indeg.items():
        if d == 0:
            heapq.heappush(heap, (decoder_order[tid], tid))
    order: list[int] = []
    while heap:
        _, u = heapq.heappop(heap)
        order.append(u)
        for v in sorted(succs[u]):
            indeg[v] -= 1
            if indeg[v] == 0:
                heapq.heappush(heap, (decoder_order[v], v))
    if len(order) != len(nodes):
        raise ValueError("cyclic toy graph")
    return order


def assert_no_overlap(intervals: list[dict]) -> None:
    by_res: dict[str, list[tuple[float, float]]] = {}
    for it in intervals:
        by_res.setdefault(it["resource"], []).append((float(it["start"]), float(it["end"])))
    for res, spans in by_res.items():
        spans_sorted = sorted(spans)
        for i in range(1, len(spans_sorted)):
            prev_end = spans_sorted[i - 1][1]
            cur_start = spans_sorted[i][0]
            if cur_start + 1e-12 < prev_end:
                raise ValueError(f"overlap on {res}: {spans_sorted[i - 1]} vs {spans_sorted[i]}")


def assert_precedence(finish: dict, start: dict, preds: dict, loc_out: dict, locs: dict) -> None:
    for dst, plist in preds.items():
        for src, _nbytes in plist:
            # successor may wait on transfer after pred finish; start >= finish is necessary
            # when same location (zero transfer). Cross-location may start later only.
            if loc_out[src] == locs[dst]:
                if start[dst] + 1e-12 < finish[src]:
                    raise ValueError(f"precedence violated {src}->{dst}")


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

    tids = sorted(nodes)
    if len(actions) != len(tids):
        raise ValueError("action plan length mismatch")
    locs = {tid: LOC[actions[i]] for i, tid in enumerate(tids)}
    decoder_order = {tid: i for i, tid in enumerate(tids)}
    order = topo_order(nodes, preds, succs, decoder_order)

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
    resource_intervals: list[dict] = []
    transfers: list[dict] = []

    def add_energy_hop(hop: str, duration: float, src_loc: str) -> None:
        if hop == "MEC_UL":
            energy["ue_mec_uplink_joules"] += duration * power["ptx_mec_w"]
        elif hop == "MEC_DL":
            energy["ue_mec_downlink_joules"] += duration * power["prx_mec_w"]
        elif hop == "V2V":
            if src_loc == UE:
                energy["ue_v2v_tx_joules"] += duration * power["ptx_v2v_w"]
                energy["helper_v2v_rx_joules"] += duration * power["prx_v2v_w"]
            elif src_loc == HELPER:
                energy["helper_v2v_tx_joules"] += duration * power["ptx_v2v_w"]
                energy["ue_v2v_rx_joules"] += duration * power["prx_v2v_w"]
            else:
                energy["ue_v2v_tx_joules"] += duration * power["ptx_v2v_w"]
                energy["helper_v2v_rx_joules"] += duration * power["prx_v2v_w"]

    def move_bytes(nbytes: int, hops: list[str], earliest: float, src_loc: str, edge_src: int | None, edge_dst: int | None) -> float:
        t = earliest
        cur = src_loc
        for hop_i, hop in enumerate(hops):
            dur = nbytes / hop_rate(hop, rates)
            s, e = hop_cal(hop, cals).reserve(dur, t)
            res_name = "V2V_CHANNEL" if hop == "V2V" else hop
            resource_intervals.append({"resource": res_name, "start": s, "end": e})
            if hop == "MEC_UL":
                dst = MEC
            elif hop == "MEC_DL":
                dst = UE
            else:
                dst = HELPER if cur == UE else UE
            add_energy_hop(hop, dur, cur)
            transfers.append(
                {
                    "hop": hop,
                    "hop_index": hop_i,
                    "bytes": nbytes,
                    "start": s,
                    "end": e,
                    "src_location": cur,
                    "dst_location": dst,
                    "src_task_id": edge_src,
                    "dst_task_id": edge_dst,
                }
            )
            t = e
            cur = dst
        return t

    for tid in order:
        node = nodes[tid]
        loc = locs[tid]
        ready = 0.0
        ext = int(node.get("external_input_bytes", 0))
        if ext > 0:
            hops = route(UE, loc)
            ready = max(ready, move_bytes(ext, hops, 0.0, UE, None, tid))
        for src, nbytes in sorted(preds[tid], key=lambda x: (decoder_order[x[0]], x[0])):
            hops = route(loc_out[src], loc)
            ready = max(ready, move_bytes(nbytes, hops, finish[src], loc_out[src], src, tid))
        dur = int(node["compute_workload_bytes"]) / cpu_rate(loc, rates)
        s, e = cpu_cal(loc, cals).reserve(dur, ready)
        resource_intervals.append({"resource": f"{loc}_CPU", "start": s, "end": e, "task_id": tid})
        start[tid] = s
        finish[tid] = e
        loc_out[tid] = loc
        if loc == UE:
            energy["ue_local_cpu_joules"] += dur * power["rho_ue"] * (power["f_l"] ** power["zeta"])
        elif loc == HELPER:
            energy["helper_compute_joules"] += dur * power["rho_helper"] * (power["f_v2v"] ** power["zeta"])

    sinks = [tid for tid in nodes if not succs[tid]]
    result_at_ue = 0.0
    for tid in sorted(sinks, key=lambda x: (decoder_order[x], x)):
        out_b = int(nodes[tid]["task_output_bytes"])
        hops = route(loc_out[tid], UE)
        if hops:
            result_at_ue = max(result_at_ue, move_bytes(out_b, hops, finish[tid], loc_out[tid], tid, None))
        else:
            result_at_ue = max(result_at_ue, finish[tid])

    assert_no_overlap(resource_intervals)
    assert_precedence(finish, start, preds, loc_out, locs)

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

    task_intervals = {
        str(tid): {"start": start[tid], "finish": finish[tid], "location": loc_out[tid]} for tid in tids
    }

    return {
        "makespan_seconds": result_at_ue,
        "total_mobile_joules": total_ue + total_helper,
        "total_ue_joules": total_ue,
        "total_helper_joules": total_helper,
        "terminal_return_time": result_at_ue,
        "task_intervals": task_intervals,
        "transfers": transfers,
        "resource_intervals": resource_intervals,
        "energy_components": dict(energy),
        "topo_order": order,
        "output_residency": {str(tid): loc_out[tid] for tid in tids},
    }


def almost(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(b)))


def compare_expected(got: dict, exp: dict) -> list[str]:
    msgs: list[str] = []
    missing = REQUIRED_EXPECTED - set(exp.keys())
    if missing:
        msgs.append(f"missing required expected keys: {sorted(missing)}")
        return msgs

    if not almost(got["makespan_seconds"], exp["makespan_seconds"]):
        msgs.append(f"makespan got={got['makespan_seconds']} expected={exp['makespan_seconds']}")
    if not almost(got["total_mobile_joules"], exp["total_mobile_joules"]):
        msgs.append(f"energy got={got['total_mobile_joules']} expected={exp['total_mobile_joules']}")
    if not almost(got["total_ue_joules"], exp["total_ue_joules"]):
        msgs.append(f"total_ue_joules got={got['total_ue_joules']} expected={exp['total_ue_joules']}")
    if not almost(got["total_helper_joules"], exp["total_helper_joules"]):
        msgs.append(f"total_helper_joules got={got['total_helper_joules']} expected={exp['total_helper_joules']}")
    if not almost(got["terminal_return_time"], exp["terminal_return_time"]):
        msgs.append(
            f"terminal_return_time got={got['terminal_return_time']} expected={exp['terminal_return_time']}"
        )

    if list(got.get("topo_order", [])) != list(exp["topo_order"]):
        msgs.append(f"topo_order got={got.get('topo_order')} expected={exp['topo_order']}")

    for tid, loc in exp["output_residency"].items():
        g = (got.get("output_residency") or {}).get(str(tid))
        if g != loc:
            msgs.append(f"output_residency[{tid}] got={g} expected={loc}")

    for tid, iv in exp["task_intervals"].items():
        g = got["task_intervals"].get(str(tid))
        if g is None:
            msgs.append(f"missing task_intervals[{tid}]")
            continue
        if not almost(g["start"], iv["start"]) or not almost(g["finish"], iv["finish"]):
            msgs.append(f"task_intervals[{tid}] got={g} expected={iv}")
        if "location" in iv and g["location"] != iv["location"]:
            msgs.append(f"task location[{tid}] got={g['location']} expected={iv['location']}")

    if len(got["transfers"]) != len(exp["transfers"]):
        msgs.append(f"transfer count got={len(got['transfers'])} expected={len(exp['transfers'])}")
    else:
        for i, (g, e) in enumerate(zip(got["transfers"], exp["transfers"])):
            for k in ("hop", "bytes", "src_location", "dst_location", "hop_index", "src_task_id", "dst_task_id"):
                if k in e and g.get(k) != e.get(k):
                    msgs.append(f"transfers[{i}].{k} got={g.get(k)} expected={e.get(k)}")
            for k in ("start", "end"):
                if k in e and not almost(g[k], e[k]):
                    msgs.append(f"transfers[{i}].{k} got={g[k]} expected={e[k]}")

    if len(got["resource_intervals"]) != len(exp["resource_intervals"]):
        msgs.append(
            f"resource_intervals count got={len(got['resource_intervals'])} expected={len(exp['resource_intervals'])}"
        )
    else:
        for i, (g, e) in enumerate(zip(got["resource_intervals"], exp["resource_intervals"])):
            if g.get("resource") != e.get("resource"):
                msgs.append(f"resource_intervals[{i}].resource got={g.get('resource')} expected={e.get('resource')}")
            for k in ("start", "end"):
                if not almost(g[k], e[k]):
                    msgs.append(f"resource_intervals[{i}].{k} got={g[k]} expected={e[k]}")
            for k in ("task_id", "hop"):
                if k in e and g.get(k) != e.get(k):
                    msgs.append(f"resource_intervals[{i}].{k} got={g.get(k)} expected={e.get(k)}")

    for k, v in exp["energy_components"].items():
        if k not in got["energy_components"] or not almost(got["energy_components"][k], v):
            msgs.append(
                f"energy_components[{k}] got={got['energy_components'].get(k)} expected={v}"
            )
    return msgs


def expected_from_schedule(got: dict) -> dict:
    return {
        "makespan_seconds": got["makespan_seconds"],
        "total_mobile_joules": got["total_mobile_joules"],
        "total_ue_joules": got["total_ue_joules"],
        "total_helper_joules": got["total_helper_joules"],
        "terminal_return_time": got["terminal_return_time"],
        "task_intervals": got["task_intervals"],
        "transfers": got["transfers"],
        "resource_intervals": got["resource_intervals"],
        "energy_components": got["energy_components"],
        "topo_order": got["topo_order"],
        "output_residency": got["output_residency"],
    }


def schedule_production(oracle: dict) -> dict:
    """Run the same oracle YAML through the production engine (oracle consumes engine)."""
    import sys

    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from env.mec_offloaing_envs.scheduler import (
        CanonicalDAG,
        CanonicalTask,
        ResourceConfig,
        schedule as production_schedule,
    )

    tasks = [
        CanonicalTask(
            task_id=int(n["task_id"]),
            compute_workload_bytes=int(n["compute_workload_bytes"]),
            task_output_bytes=int(n["task_output_bytes"]),
            external_input_bytes=int(n.get("external_input_bytes", 0)),
        )
        for n in oracle["nodes"]
    ]
    raw_edges = [
        (int(e["src_task_id"]), int(e["dst_task_id"]), int(e["edge_output_bytes"]))
        for e in oracle.get("edges", [])
    ]
    dag = CanonicalDAG.from_records(tasks, raw_edges)
    order = sorted(dag.tasks)
    actions = [LOC[a] for a in oracle["actions"]]
    result = production_schedule(dag, order, actions, ResourceConfig.from_frozen_yaml())

    transfers = []
    for t in result.transfers:
        transfers.append(
            {
                "hop": t.hop,
                "hop_index": t.hop_index,
                "bytes": t.bytes,
                "start": t.start,
                "end": t.end,
                "src_location": t.src_location.value,
                "dst_location": t.dst_location.value,
                "src_task_id": t.src_task_id,
                "dst_task_id": t.dst_task_id,
            }
        )
    resource_intervals = []
    for iv in result.resource_intervals:
        item = {"resource": iv.resource, "start": iv.start, "end": iv.end}
        if iv.task_id is not None:
            item["task_id"] = iv.task_id
        if iv.hop is not None:
            item["hop"] = iv.hop
        resource_intervals.append(item)

    energy = {
        k: v
        for k, v in result.energy.as_dict().items()
        if k
        not in {
            "total_ue_joules",
            "total_helper_joules",
            "total_mobile_joules",
            "mec_compute_joules_optional",
            "total_system_joules_optional",
        }
    }
    return {
        "makespan_seconds": result.makespan_seconds,
        "total_mobile_joules": result.total_mobile_joules,
        "total_ue_joules": result.energy.total_ue_joules,
        "total_helper_joules": result.energy.total_helper_joules,
        "terminal_return_time": result.terminal_return_time,
        "task_intervals": {
            str(tid): {
                "start": rec.start,
                "finish": rec.finish,
                "location": rec.output_location.value,
            }
            for tid, rec in result.tasks.items()
        },
        "transfers": transfers,
        "resource_intervals": resource_intervals,
        "energy_components": energy,
        "topo_order": result.topo_order,
        "output_residency": {
            str(tid): rec.output_location.value for tid, rec in result.tasks.items()
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle-dir", default=str(Path(__file__).parent))
    ap.add_argument(
        "--engine",
        choices=["reference", "production"],
        default="reference",
        help="reference = toy checker scheduler; production = env/.../scheduler engine",
    )
    ap.add_argument(
        "--write-expected",
        action="store_true",
        help="overwrite expected blocks from deterministic scheduler (Phase 0 freeze aid)",
    )
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
        if args.engine == "production":
            got = schedule_production(doc)
        else:
            got = schedule(doc, rates, power, actions)
        if args.write_expected:
            if args.engine != "reference":
                raise SystemExit("--write-expected only allowed with --engine reference")
            doc["expected"] = expected_from_schedule(got)
            path.write_text(yaml.safe_dump(doc, sort_keys=False, default_flow_style=False))
            print(f"WROTE {path.name}")
            continue
        exp = doc.get("expected")
        if not isinstance(exp, dict):
            failed += 1
            print(f"FAIL {path.name} missing expected block")
            continue
        msgs = compare_expected(got, exp)
        ok = not msgs
        status = "PASS" if ok else "FAIL"
        if not ok:
            failed += 1
        print(
            f"{status} {path.name} engine={args.engine} "
            f"makespan={got['makespan_seconds']:.9g} E={got['total_mobile_joules']:.9g}"
        )
        for m in msgs:
            print("  ", m)
    if args.write_expected:
        print("wrote expected blocks for all oracles")
        return 0
    if failed:
        raise SystemExit(failed)
    print(f"ALL toy oracles PASS (engine={args.engine})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
