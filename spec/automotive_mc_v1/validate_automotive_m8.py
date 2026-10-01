#!/usr/bin/env python3
"""M8 validator: deterministic materialization + generator invariants. Exit != 0.

Writes `m8_materialization_report.json`. It refuses a stale on-disk dataset: if the
committed `graphs.jsonl` differs from a fresh in-memory generation, the milestone is
red until the generator is rerun.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

import automotive_dataset as ad  # noqa: E402
import automotive_generator as ag  # noqa: E402


def determinism(docs: dict) -> dict:
    tmp_a = Path(tempfile.mkdtemp(prefix="mcv1_det_a_"))
    tmp_b = Path(tempfile.mkdtemp(prefix="mcv1_det_b_"))
    try:
        ra = ag.materialize(tmp_a, docs)
        rb = ag.materialize(tmp_b, docs)
        same_files = all((tmp_a / n).read_bytes() == (tmp_b / n).read_bytes()
                         for n in ag.MATERIALIZED_FILES)
        checks = {
            "graph_counts_identical": ra["graph_count"] == rb["graph_count"],
            "canonical_hashes_identical": [g["canonical_sha256"] for g in ra["graphs"]] ==
                                          [g["canonical_sha256"] for g in rb["graphs"]],
            "raw_hashes_identical": [g["raw_sha256"] for g in ra["graphs"]] ==
                                    [g["raw_sha256"] for g in rb["graphs"]],
            "manifest_content_identical": (tmp_a / "dataset_manifest.jsonl").read_bytes() ==
                                          (tmp_b / "dataset_manifest.jsonl").read_bytes(),
            "semantic_annotations_identical": all(
                a["criticality_mixture"] == b["criticality_mixture"]
                and a["mode_semantics"] == b["mode_semantics"]
                and a["provenance"] == b["provenance"]
                for a, b in zip(ra["graphs"], rb["graphs"])),
            "lineage_fields_identical": all(
                a["template_lineage"] == b["template_lineage"]
                and a["parent_seed"] == b["parent_seed"]
                and a["topology_signature"] == b["topology_signature"]
                and a["workload_regime"] == b["workload_regime"]
                and a["resource_profile"] == b["resource_profile"]
                and a["resource_level"] == b["resource_level"]
                and a["sla_regime"] == b["sla_regime"]
                and a["criticality_mixture"] == b["criticality_mixture"]
                for a, b in zip(ra["graphs"], rb["graphs"])),
            "materialized_files_byte_identical": same_files,
            "graphs_sha256": ra["graphs_sha256"],
            "dataset_manifest_sha256": ra["dataset_manifest_sha256"],
        }
        checks["status"] = "PASS" if all(v for k, v in checks.items()
                                         if k not in ("graphs_sha256", "dataset_manifest_sha256",
                                                      "status")) else "FAIL"
        return checks
    finally:
        shutil.rmtree(tmp_a, ignore_errors=True)
        shutil.rmtree(tmp_b, ignore_errors=True)


def generator_invariants(docs: dict, graphs: list[dict]) -> tuple[list[str], dict]:
    v: list[str] = []
    sem = docs["task_semantics.yaml"]
    cfg = docs["generation_config.yaml"]
    w2 = docs["workload_model_v2.yaml"]
    f_ref = float(w2["reference_compute_model"]["f_ref_hz"])
    xi = float(w2["reference_compute_model"]["xi_cycles_per_bit"])
    classes = set(sem["payload_classes"])
    models = sem["payload_models"]
    stats = {"graphs": len(graphs), "tasks": 0, "edges": 0}
    high_total = 0
    safety_graphs = 0

    for g in graphs:
        gid = g["graph_id"]
        v.extend(ad.validate_graph_record(g))
        stats["tasks"] += len(g["tasks"])
        stats["edges"] += len(g["edges"])
        if len(g["tasks"]) != 20:
            v.append(f"{gid}: exactly 20 tasks required")
        # refs resolve
        for e in g["edges"]:
            if e["payload_class"] not in classes:
                v.append(f"{gid}: unknown payload class {e['payload_class']!r}")
            if e["payload_model_ref"] != models.get(e["payload_class"]):
                v.append(f"{gid}: payload_model_ref does not resolve for {e['payload_class']!r}")
        for t in g["tasks"]:
            role = next((r for r in sem["roles"] if r["semantic_role"] == t["semantic_role"]), None)
            if role is None:
                v.append(f"{gid}/{t['task_id']}: unknown semantic role")
                continue
            if t["motif_id"] != role["motif_role"]:
                v.append(f"{gid}/{t['task_id']}: motif_id disagrees with task_semantics")
            if t["lineage_id"] != f"{g['template_id']}#{t['task_id']}":
                v.append(f"{gid}/{t['task_id']}: lineage_id does not resolve")
        # workload recomputes from t_ref (and is action-invariant by construction)
        for t in g["tasks"]:
            expect = int(round(float(t["t_ref_s"]) * f_ref / (8.0 * xi)))
            if int(t["compute_workload_bytes"]) != expect:
                v.append(f"{gid}/{t['task_id']}: W_i does not recompute from t_ref")
            for forbidden in ("location", "action", "placement", "chosen_tier"):
                if forbidden in t:
                    v.append(f"{gid}/{t['task_id']}: workload record leaks placement field {forbidden!r}")
        # deadlines recompute from the frozen D_G
        recomputed = ad.deadlines_with_D_G(_template(docs, g), w2, docs["sla_registry_v2.yaml"],
                                           float(g["D_G_s"]))
        for t in g["tasks"]:
            r = recomputed["tasks"][int(t["task_id"])]
            for field, key in (("E_s", "E_s"), ("L_s", "L_s"), ("deadline_s", "deadline_s"),
                               ("slack_s", "slack_s")):
                if abs(float(t[field]) - float(r[key])) > 1e-15:
                    v.append(f"{gid}/{t['task_id']}: {field} is not the frozen M6-v2 construction")
        # D_G is the pre-declared draw
        sla = docs["sla_registry_v2.yaml"]["deadline_registry"]["families"][g["application_family"]]
        lo, hi = (float(x) for x in sla["D_f_s_range"])
        expected_dg = lo + float(g["raw_draws"]["deadline_quantile"]) * (hi - lo)
        if abs(float(g["D_G_s"]) - expected_dg) > 1e-12:
            v.append(f"{gid}: D_G is not the declared deadline draw")
        # MC budgets + HIGH existence
        if g["safety_scope"] == "safety":
            safety_graphs += 1
            if int(g["criticality_counts"].get("HIGH", 0)) <= 0:
                v.append(f"{gid}: safety-family graph has no HIGH task")
        high_total += int(g["criticality_counts"].get("HIGH", 0))
        for t in g["tasks"]:
            if t["criticality"] == "HIGH":
                if not t["budget_hi_applicable"] or t["empirical_execution_budget_hi_s"] is None:
                    v.append(f"{gid}/{t['task_id']}: HIGH lacks a HI budget")
                elif float(t["empirical_execution_budget_hi_s"]) < float(t["empirical_execution_budget_lo_s"]):
                    v.append(f"{gid}/{t['task_id']}: C_HI < C_LO")
            elif t["budget_hi_applicable"]:
                v.append(f"{gid}/{t['task_id']}: non-HIGH claims an HI budget")
            if t["wcet_claim"]:
                v.append(f"{gid}/{t['task_id']}: empirical budget must not be called WCET")

    # provenance coverage: every numeric leaf belongs to a declared field group
    allowed_task = set(ad.NUMERIC_TASK_FIELDS) | {"empirical_execution_budget_hi_s"}
    allowed_edge = set(ad.NUMERIC_EDGE_FIELDS)
    allowed_graph = {"seed_index", "parent_seed", "graph_seed", "task_count", "edge_count",
                     "P_f_s", "D_G_s"}
    uncovered = set()
    for g in graphs[::11]:
        for key, value in g.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            if key not in allowed_graph:
                uncovered.add(f"{g['graph_id']}:graph.{key}")
        for t in g["tasks"]:
            for key, value in t.items():
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    continue
                if key not in allowed_task:
                    uncovered.add(f"{g['graph_id']}:task.{key}")
        for e in g["edges"]:
            for key, value in e.items():
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    continue
                if key not in allowed_edge:
                    uncovered.add(f"{g['graph_id']}:edge.{key}")
    if uncovered:
        v.append(f"numeric fields outside the declared schema: {sorted(uncovered)[:5]}")

    # criticality cannot come from topology: recompute from role/family and compare
    import criticality_model as cm
    for g in graphs[:40] + graphs[-5:]:
        table = cm.build_task_table(docs["criticality_policy.yaml"], w2, _template(docs, g))
        for t in g["tasks"]:
            if t["criticality"] != table[int(t["task_id"])]["criticality"]:
                v.append(f"{g['graph_id']}/{t['task_id']}: criticality is not the semantic function")

    # payload class correctness and pairwise-disjoint bands
    bands = ad.realised_payload_bands(graphs)
    if set(bands) != classes:
        v.append(f"payload classes missing from the population: {sorted(classes - set(bands))}")
    bl = sorted(bands)
    for i, a in enumerate(bl):
        for b in bl[i + 1:]:
            if not (bands[a][1] < bands[b][0] or bands[b][1] < bands[a][0]):
                v.append(f"payload bands overlap: {a}{bands[a]} and {b}{bands[b]}")

    if len(graphs) != int(cfg["grid"]["counts_per_family"]) * 4:
        v.append(f"graph count {len(graphs)} != 4 x {cfg['grid']['counts_per_family']}")
    if high_total == 0:
        v.append("no HIGH task exists in the materialized population")
    stats["high_task_total"] = high_total
    stats["safety_family_graphs"] = safety_graphs
    stats["payload_bands"] = bands
    return v, stats


def _template(docs: dict, g: dict) -> dict:
    for tpl in docs["application_templates.yaml"]["templates"]:
        if tpl["family_id"] == g["application_family"]:
            return tpl
    raise ad.DatasetError(f"unknown family {g['application_family']!r}")


def build_report() -> dict:
    docs = ad.load_inputs()
    graphs = ag.generate_all(docs)
    det = determinism(docs)
    violations, stats = generator_invariants(docs, graphs)
    if det["status"] != "PASS":
        violations.append("materialization is not deterministic")

    # the committed dataset must not be stale
    stale = None
    graphs_path = ad.DATASET_DIR / "graphs.jsonl"
    if graphs_path.exists():
        on_disk = graphs_path.read_text()
        fresh = "".join(ad.canonical_json(g) + "\n" for g in graphs)
        if on_disk != fresh:
            stale = ad.sha256_text(on_disk) != ad.sha256_text(fresh)
            violations.append("on-disk graphs.jsonl is stale; rerun the generator")
    else:
        violations.append("dataset not materialized")

    report = {
        "schema_version": "m8_materialization_report_v1",
        "milestone": "M8",
        "generator_version": ag.GENERATOR_VERSION,
        "deterministic_regeneration": det["status"],
        "determinism": det,
        "graph_count": len(graphs),
        "stats": stats,
        "stale_on_disk": stale,
        "dataset_dir": str(ad.DATASET_DIR),
        "generator_sha256": ad.sha256_file(BASE / "automotive_generator.py"),
        "generation_config_sha256": ad.sha256_file(BASE / "generation_config.yaml"),
        "dataset_manifest_sha256": ad.sha256_text(
            "".join(ad.canonical_json(ag.dataset_manifest_record(g)) + "\n" for g in graphs)),
        "status": "PASS" if not violations else "FAIL",
        "violations": violations,
    }
    (BASE / "m8_materialization_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> int:
    rep = build_report()
    print(json.dumps({"status": rep["status"],
                      "graph_count": rep["graph_count"],
                      "deterministic_regeneration": rep["deterministic_regeneration"],
                      "stale_on_disk": rep["stale_on_disk"],
                      "high_task_total": rep["stats"]["high_task_total"],
                      "violations": rep["violations"][:10]}, indent=2, sort_keys=True))
    return 1 if rep["violations"] else 0


if __name__ == "__main__":
    sys.exit(main())
