#!/usr/bin/env python3
"""M8 deterministic generator + materializer for MARGO-AUTOMOTIVE-MC-v1.

The generator is a pure function of the frozen artifacts:
    (application_templates, task_semantics, workload_model_v2, resource_profiles,
     sla_registry_v2, criticality_policy, generation_config) x (family, seed, cell)

No previous milestone is mutated, no deadline is chosen from a schedule, and no
payload is resized to fit a link. Materializing twice into two clean directories is
byte-identical (proved by `validate_automotive_m8.py`).

CLI:
    python3 automotive_generator.py [--output DIR]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

import automotive_dataset as ad  # noqa: E402

GENERATOR_VERSION = "automotive_generator_v1"
MATERIALIZED_FILES = (
    "graphs.jsonl",
    "dataset_manifest.jsonl",
    "dataset_manifest.sha256",
    "graphs.sha256",
    "dataset_card.json",
    "generation_config.yaml",
)


def _cm():
    return ad._load_module("criticality_model")


def cells(cfg: dict) -> list[dict]:
    wl = list(cfg["cells"]["workload_regime"])
    sl = list(cfg["cells"]["sla_regime"])
    rl = list(cfg["cells"]["resource_level"])
    out = []
    for w in wl:
        for s in sl:
            for r in rl:
                out.append({
                    "workload_regime": w,
                    "sla_regime": s,
                    "resource_level": r,
                    "cell_id": cfg["cells"]["cell_id_format"].format(
                        workload_regime=w, sla_regime=s, resource_level=r),
                })
    return sorted(out, key=lambda c: c["cell_id"])


def parent_seed(docs: dict, family_id: str, seed_index: int) -> int:
    cfg = docs["generation_config.yaml"]
    material = f"parent|{cfg['dataset_version']}|{family_id}|{seed_index}"
    return int(ad.sha256_text(material)[:18], 16)


def graph_seed(docs: dict, parent: int, cell_id: str) -> int:
    return int(ad.sha256_text(f"graph|{parent}|{cell_id}")[:18], 16)


def _payload_classes(docs: dict) -> list[str]:
    return list(docs["task_semantics.yaml"]["payload_classes"])


def generate_graph(docs: dict, family_id: str, seed_index: int, cell: dict) -> dict:
    cfg = docs["generation_config.yaml"]
    sem = docs["task_semantics.yaml"]
    w2 = docs["workload_model_v2.yaml"]
    res = docs["resource_profiles.yaml"]
    sla2 = docs["sla_registry_v2.yaml"]
    pol = docs["criticality_policy.yaml"]
    cm = _cm()
    tpl = ad.templates_by_family(docs)[family_id]

    pseed = parent_seed(docs, family_id, seed_index)
    gseed = graph_seed(docs, pseed, cell["cell_id"])
    abbrev = cfg["cells"]["family_abbrev"][family_id]
    graph_id = cfg["cells"]["graph_id_format"].format(
        family_abbrev=abbrev, seed_index=seed_index, cell_id=cell["cell_id"])

    rng_w = ad.HashRNG(f"{gseed}|workload")
    rng_p = ad.HashRNG(f"{gseed}|profile")
    rng_r = ad.HashRNG(f"{gseed}|rates")
    rng_d = ad.HashRNG(f"{gseed}|deadline")

    # ---- workload (t_ref -> W_i), fixed motif partition stays invariant ------
    ref = w2["reference_compute_model"]
    f_ref = float(ref["f_ref_hz"])
    xi = float(ref["xi_cycles_per_bit"])
    fixed_roles = set(cfg["workload_draw"].get("fixed_roles") or [])
    w_band = cfg["cells"]["workload_regime"][cell["workload_regime"]]
    t_ref: dict[int, float] = {}
    workload_quantiles: dict[int, float | None] = {}
    rule_by_role: dict[str, dict] = {}
    for task in tpl["tasks"]:
        role = str(task["semantic_role"])
        if role not in rule_by_role:
            rule_by_role[role] = cm.resolve_demand_rule(w2, role)
        demand = rule_by_role[role]
        if role in fixed_roles or demand["kind"] == "fixed":
            t_ref[int(task["task_id"])] = float(demand["value_s"])
            workload_quantiles[int(task["task_id"])] = None
        else:
            lo, hi = demand["range_s"]
            u = rng_w.uniform(w_band[0], w_band[1])
            t_ref[int(task["task_id"])] = lo + u * (hi - lo)
            workload_quantiles[int(task["task_id"])] = u
    for tid, value in t_ref.items():
        if value <= 0:
            raise ad.DatasetError(f"{graph_id}: t_ref[{tid}] must be positive")

    # ---- resource profile + rates -------------------------------------------
    compat = list(sem["families"][family_id]["resource_profile_compatibility"])
    profile_name = rng_p.choice(compat)
    profile = res["profiles"][profile_name]
    r_band = cfg["cells"]["resource_level"][cell["resource_level"]]
    rates = {}
    for key in sorted(profile):
        spec = profile[key]
        if not isinstance(spec, dict) or "range" not in spec:
            continue
        rates[key] = rng_r.uniform(spec["range"][0], spec["range"][1])
    for key in ("f_ue_hz", "f_helper_hz", "f_mec_hz", "r_mec_ul_bps",
                "r_mec_dl_bps", "r_v2v_bps"):
        if key not in rates or rates[key] <= 0:
            raise ad.DatasetError(f"{graph_id}: resource rate {key} missing/invalid")

    # ---- graph deadline (frozen before any scheduling) ----------------------
    sla_fam = sla2["deadline_registry"]["families"][family_id]
    d_lo, d_hi = (float(v) for v in sla_fam["D_f_s_range"])
    d_band = cfg["cells"]["sla_regime"][cell["sla_regime"]]
    d_q = rng_d.uniform(d_band[0], d_band[1])
    D_G = d_lo + d_q * (d_hi - d_lo)
    P_f = float(sla2["period_registry"]["families"][family_id]["P_f_s"])
    deadline = ad.deadlines_with_D_G(tpl, w2, sla2, D_G)

    # ---- payload per class (compute and payload never mix) ------------------
    payload_draws = {cls: ad.payload_bytes_for_class(
        cfg, w2, cls, ad.HashRNG(f"{gseed}|payload|{cls}")) for cls in _payload_classes(docs)}

    # ---- criticality + budgets + mode semantics ----------------------------
    table = cm.build_task_table(pol, w2, tpl)
    preds, succs = ad.predecessors(tpl), ad.successors(tpl)
    sink_ids = set(ad.sinks(tpl))
    role_out_class = {r["semantic_role"]: (r["payload_out"] or [None])[0]
                      for r in sem["roles"]}
    order = ad.topo_order(tpl)
    tasks = []
    for tid in order:
        role = str(next(t["semantic_role"] for t in tpl["tasks"] if int(t["task_id"]) == tid))
        row = table[tid]
        w_bytes = int(round(t_ref[tid] * f_ref / (8.0 * xi)))
        if w_bytes <= 0:
            raise ad.DatasetError(f"{graph_id}/{tid}: W_i must be > 0")
        if tid in sink_ids:
            cls = role_out_class[role]
            out_bytes = int(payload_draws[cls]["bytes"]) if cls else 0
        else:
            outs = [e for e in tpl["edges"] if int(e["src"]) == tid]
            out_bytes = max(int(payload_draws[e["payload_class"]]["bytes"]) for e in outs)
        dl = deadline["tasks"][tid]
        tasks.append({
            "task_id": tid,
            "semantic_role": role,
            "motif_id": str(next(t["motif_id"] for t in tpl["tasks"] if int(t["task_id"]) == tid)),
            "family_id": family_id,
            "lineage_id": f"{tpl['template_id']}#{tid}",
            "criticality": row["criticality"],
            "criticality_source_rule": row["source_rule"],
            "motif_role": row["motif_role"],
            "motif_floor_class": row["motif_floor_class"],
            "safety_scope": row["safety_scope"],
            "compute_workload_bytes": w_bytes,
            "t_ref_s": t_ref[tid],
            "t_ref_rule_id": row["demand_class_id"],
            "t_ref_evidence_class": (
                "measured_cpu_anchor" if row["demand_class_id"] == "planning_motif_split"
                else "source_calibrated_synthetic"),
            "empirical_execution_budget_lo_s": row["empirical_execution_budget_lo_s"],
            "empirical_execution_budget_hi_s": row["empirical_execution_budget_hi_s"],
            "budget_hi_applicable": row["budget_hi_applicable"],
            "budget_hi_status": row["budget_hi_status"],
            "budget_rule_id": row["budget_rule_id"],
            "budget_evidence_class": row["budget_evidence_class"],
            "wcet_claim": False,
            "drop_allowed_lo_mode": row["drop_degrade_lo"]["drop_allowed"],
            "drop_allowed_hi_mode": row["drop_degrade_hi"]["drop_allowed"],
            "degrade_allowed_lo_mode": row["drop_degrade_lo"]["degrade_allowed"],
            "degrade_allowed_hi_mode": row["drop_degrade_hi"]["degrade_allowed"],
            "task_output_bytes": out_bytes,
            "external_input_bytes": 0,
            "output_payload_class": role_out_class[role] if tid in sink_ids else None,
            "E_s": dl["E_s"],
            "L_s": dl["L_s"],
            "deadline_s": dl["deadline_s"],
            "slack_s": dl["slack_s"],
            "deadline_type": cfg["deadline_type"]["task_level"],
            "tardiness_weight": float(cfg["deadline_type"]["tardiness_weight"]),
            "is_root": not preds[tid],
            "is_sink": tid in sink_ids,
            "predecessors": preds[tid],
            "successors": succs[tid],
        })

    edges = []
    for k, e in enumerate(sorted(tpl["edges"], key=lambda x: (int(x["src"]), int(x["dst"])))):
        cls = e["payload_class"]
        edges.append({
            "edge_id": f"e{k:03d}",
            "src": int(e["src"]),
            "dst": int(e["dst"]),
            "payload_class": cls,
            "payload_model_ref": e["payload_model_ref"],
            "payload_bytes": int(payload_draws[cls]["bytes"]),
            "payload_rule_id": payload_draws[cls]["rule_id"],
            "payload_evidence_class": payload_draws[cls]["evidence_class"],
            "provenance_or_rule_ref": e["provenance_or_rule_ref"],
        })

    counts = cm.criticality_mixture(table)
    overrun = sorted(t["task_id"] for t in tasks
                     if t["criticality"] == "HIGH"
                     and t["t_ref_s"] > t["empirical_execution_budget_lo_s"] + 1e-15)
    raw_draws = {
        "workload_quantiles": {str(k): v for k, v in sorted(workload_quantiles.items())},
        "resource_rates": {k: rates[k] for k in sorted(rates)},
        "deadline_quantile": d_q,
        "payload": {k: v for k, v in sorted(payload_draws.items())},
    }
    raw_fingerprint = {
        "dataset_version": cfg["dataset_version"],
        "generator_version": GENERATOR_VERSION,
        "graph_id": graph_id,
        "family_id": family_id,
        "template_id": tpl["template_id"],
        "seed_index": seed_index,
        "cell": cell,
        "parent_seed": pseed,
        "graph_seed": gseed,
        "draws": raw_draws,
    }

    record = {
        "schema_version": ad.SCHEMA_VERSION,
        "dataset_version": cfg["dataset_version"],
        "generator_version": GENERATOR_VERSION,
        "graph_id": graph_id,
        "seed_index": seed_index,
        "parent_seed": pseed,
        "graph_seed": gseed,
        "cell_id": cell["cell_id"],
        "application_family": family_id,
        "safety_scope": table[0]["safety_scope"],
        "template_id": tpl["template_id"],
        "template_lineage": tpl["template_lineage"],
        "semantic_signature": tpl["semantic_signature"],
        "topology_regime": cfg["topology_regimes"][tpl["template_id"]],
        "topology_signature": ad.topology_signature(tpl),
        "workload_regime": cell["workload_regime"],
        "resource_level": cell["resource_level"],
        "resource_profile": profile_name,
        "resource_profile_rule_id": profile["rule_id"],
        "resource_profile_evidence_class": profile["evidence_class"],
        "resource": {k: rates[k] for k in sorted(rates)},
        "sla_id": sla_fam["deadline_rule_id"],
        "sla_regime": cell["sla_regime"],
        "sla_class": sla_fam.get("sla_class"),
        "P_f_s": P_f,
        "D_f_s_range": [d_lo, d_hi],
        "D_G_s": D_G,
        "deadline_status": deadline["status"],
        "criticality_policy_id": pol["policy_id"],
        "criticality_counts": counts,
        "criticality_mixture": ",".join(f"{c}:{counts[c]}" for c in cm.CLASSES),
        "task_count": len(tasks),
        "edge_count": len(edges),
        "raw_draws": raw_draws,
        "tasks": tasks,
        "edges": edges,
        "mode_semantics": {
            "initial_mode": pol["mode_semantics"]["initial_mode"],
            "modes": list(pol["mode_semantics"]["modes"]),
            "switch_rule_id": pol["mode_semantics"]["switch_rule_id"],
            "hi_exit": pol["mode_semantics"]["hi_exit"],
            "hi_to_lo": pol["mode_semantics"]["hi_to_lo"],
            "high_never_silently_dropped": True,
            "drop_degrade_policy": pol["mode_semantics"]["drop_degrade_policy"],
            "predicted_lo_overrun_high_tasks": overrun,
        },
        "provenance": ad.base_provenance(docs),
    }
    record["raw_sha256"] = ad.sha256_json(raw_fingerprint)
    record["canonical_sha256"] = ad.sha256_json(record)
    return record


def generate_all(docs: dict) -> list[dict]:
    cfg = docs["generation_config.yaml"]
    graphs = []
    for family_id in sorted(docs["task_semantics.yaml"]["families"]):
        for seed_index in cfg["seeds"]["parent_seed_indices"]:
            for cell in cells(cfg):
                graphs.append(generate_graph(docs, family_id, int(seed_index), cell))
    graphs.sort(key=lambda g: g["graph_id"])
    return graphs


def dataset_manifest_record(g: dict) -> dict:
    return {
        "dataset_version": g["dataset_version"],
        "graph_id": g["graph_id"],
        "application_family": g["application_family"],
        "template_id": g["template_id"],
        "template_lineage": g["template_lineage"],
        "semantic_signature": g["semantic_signature"],
        "topology_regime": g["topology_regime"],
        "topology_signature": g["topology_signature"],
        "parent_seed": g["parent_seed"],
        "graph_seed": g["graph_seed"],
        "cell_id": g["cell_id"],
        "workload_regime": g["workload_regime"],
        "resource_level": g["resource_level"],
        "resource_profile": g["resource_profile"],
        "sla_id": g["sla_id"],
        "sla_regime": g["sla_regime"],
        "criticality_policy_id": g["criticality_policy_id"],
        "criticality_mixture": g["criticality_mixture"],
        "D_G_s": g["D_G_s"],
        "task_count": g["task_count"],
        "edge_count": g["edge_count"],
        "raw_sha256": g["raw_sha256"],
        "canonical_sha256": g["canonical_sha256"],
    }


def factor_distributions(graphs: list[dict]) -> dict:
    keys = ("application_family", "template_lineage", "topology_regime", "workload_regime",
            "resource_profile", "resource_level", "sla_regime", "criticality_mixture",
            "safety_scope")
    out = {}
    for key in keys:
        counts: dict[str, int] = {}
        for g in graphs:
            counts[str(g[key])] = counts.get(str(g[key]), 0) + 1
        out[key] = dict(sorted(counts.items()))
    return out


def materialize(output_dir: Path, docs: dict | None = None) -> dict:
    docs = docs or ad.load_inputs()
    graphs = generate_all(docs)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name in MATERIALIZED_FILES:
        p = out / name
        if p.exists():
            p.unlink()

    graphs_text = "".join(ad.canonical_json(g) + "\n" for g in graphs)
    (out / "graphs.jsonl").write_text(graphs_text)
    manifest_text = "".join(ad.canonical_json(dataset_manifest_record(g)) + "\n" for g in graphs)
    (out / "dataset_manifest.jsonl").write_text(manifest_text)
    graphs_sha = ad.sha256_text(graphs_text)
    manifest_sha = ad.sha256_text(manifest_text)
    (out / "graphs.sha256").write_text(graphs_sha + "\n")
    (out / "dataset_manifest.sha256").write_text(manifest_sha + "\n")
    shutil.copyfile(ad.SPEC_DIR / "generation_config.yaml", out / "generation_config.yaml")

    card = {
        "dataset_version": docs["generation_config.yaml"]["dataset_version"],
        "generator_version": GENERATOR_VERSION,
        "schema_version": ad.SCHEMA_VERSION,
        "graph_count": len(graphs),
        "family_counts": factor_distributions(graphs)["application_family"],
        "factor_distributions": factor_distributions(graphs),
        "criticality_counts_total": _total_criticality(graphs),
        "high_task_total": sum(int(g["criticality_counts"].get("HIGH", 0)) for g in graphs),
        "task_total": sum(g["task_count"] for g in graphs),
        "deadline_count": sum(1 for g in graphs for t in g["tasks"]
                              if float(t["deadline_s"]) > 0),
        "graphs_sha256": graphs_sha,
        "dataset_manifest_sha256": manifest_sha,
        "input_sha256": {k: v for k, v in sorted(ad.inputs_sha256(docs).items())},
        "immutable_inputs": ["workload_model_v2.yaml", "sla_registry_v2.yaml",
                             "resource_profiles.yaml", "application_templates.yaml",
                             "task_semantics.yaml"],
        "deterministic": True,
    }
    (out / "dataset_card.json").write_text(json.dumps(card, indent=2, sort_keys=True) + "\n")
    return {"output_dir": str(out), "graph_count": len(graphs), "graphs": graphs,
            "graphs_sha256": graphs_sha, "dataset_manifest_sha256": manifest_sha, "card": card}


def _total_criticality(graphs: list[dict]) -> dict:
    total = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
    for g in graphs:
        for k, v in g["criticality_counts"].items():
            total[k] = total.get(k, 0) + int(v)
    return total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default=str(ad.DATASET_DIR))
    args = ap.parse_args()
    result = materialize(Path(args.output))
    print(json.dumps({"output_dir": result["output_dir"],
                      "graph_count": result["graph_count"],
                      "graphs_sha256": result["graphs_sha256"],
                      "dataset_manifest_sha256": result["dataset_manifest_sha256"]},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
