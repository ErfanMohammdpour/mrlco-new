#!/usr/bin/env python3
"""Shared library for MARGO-AUTOMOTIVE-MC-v1 (M8/M9/M10).

Design rules enforced here:

* EVERY draw is a deterministic function of a sha256-counter PRNG seeded by a
  canonical string. No `random`, no wall-clock, no time, no dict/FS ordering.
* The immutable M4/M5-v2/M6-v2 inputs are read from `spec/automotive_mc_v1/` and are
  never written by any function in this module or in the generator/certifier.
* `D_G` is computed by `deadlines_with_D_G`, which is the frozen M6-v2 construction
  evaluated at a per-graph pre-declared deadline draw. At `D_G = midpoint` it must
  reproduce `deadline_model_v2.generate_deadlines_v2` exactly (tested).
* `W_i` never feeds `B_e`; `B_e` is per edge and action-invariant.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_DIR = REPO_ROOT / "spec" / "automotive_mc_v1"
DATASET_VERSION = "MARGO-AUTOMOTIVE-MC-v1"
DATASET_DIR = REPO_ROOT / "env" / "mec_offloaing_envs" / "data" / "automotive_mc_v1"
SCHEMA_VERSION = "automotive_mc_v1_graph"

INPUT_FILES = (
    "application_templates.yaml",
    "task_semantics.yaml",
    "workload_model_v2.yaml",
    "resource_profiles.yaml",
    "sla_registry_v2.yaml",
    "criticality_policy.yaml",
    "generation_config.yaml",
    "SOURCE_REGISTRY.yaml",
    "REQUIRED_PARAMETER_MANIFEST.yaml",
    "registry_pin.json",
)
# NOTE: the split policy is deliberately NOT a generation input: M8 must be
# materializable without M9 (it is pinned separately in provenance.json).

# DATASET_CONTRACT.md §2 vocabulary, plus the dataset-card label `derived`
# ("computed from verified inputs by a stated rule"), which the contract's own
# W_i = t_ref * f_ref / (8 * xi) construction needs. The union is recorded in the
# provenance audit so the two vocabularies are never silently conflated.
PROVENANCE_LABELS = (
    "measured",
    "standard-derived",
    "trace-fitted",
    "source-calibrated-synthetic",
    "synthetic",
    "derived",
)
CONTRACT_PROVENANCE_LABELS = ("measured", "standard-derived", "trace-fitted",
                              "source-calibrated-synthetic", "synthetic")
CARD_EXTRA_PROVENANCE_LABELS = ("derived",)


class DatasetError(ValueError):
    """Input refused. Never silently repaired."""


# --------------------------------------------------------------------------- #
# Determinism primitives
# --------------------------------------------------------------------------- #
def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha256_json(obj) -> str:
    return sha256_text(canonical_json(obj))


class HashRNG:
    """sha256-counter PRNG. Deterministic across Python versions and platforms."""

    def __init__(self, material: str):
        self.material = str(material)
        self.counter = 0

    def _word(self) -> int:
        digest = hashlib.sha256(f"{self.material}|{self.counter}".encode("utf-8")).digest()
        self.counter += 1
        return int.from_bytes(digest[:8], "big")

    def unit(self) -> float:
        return self._word() / float(1 << 64)

    def uniform(self, lo: float, hi: float) -> float:
        lo, hi = float(lo), float(hi)
        if not (math.isfinite(lo) and math.isfinite(hi)) or hi < lo:
            raise DatasetError(f"invalid uniform band [{lo}, {hi}]")
        return lo + self.unit() * (hi - lo)

    def choice(self, seq):
        seq = list(seq)
        if not seq:
            raise DatasetError("cannot choose from an empty sequence")
        return seq[self._word() % len(seq)]


# --------------------------------------------------------------------------- #
# Immutable inputs
# --------------------------------------------------------------------------- #
_MODULE_CACHE: dict = {}


def _load_module(name: str):
    if name in _MODULE_CACHE:
        return _MODULE_CACHE[name]
    spec = importlib.util.spec_from_file_location(name, SPEC_DIR / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    _MODULE_CACHE[name] = mod
    return mod


def load_inputs(spec_dir: Path | None = None) -> dict:
    base = Path(spec_dir) if spec_dir else SPEC_DIR
    docs = {}
    for name in INPUT_FILES:
        path = base / name
        if not path.exists():
            continue
        text = path.read_text()
        docs[name] = json.loads(text) if name.endswith(".json") else yaml.safe_load(text)
    return docs


def inputs_sha256(docs: dict) -> dict:
    return {name: sha256_json(doc) for name, doc in sorted(docs.items())}


def templates_by_family(docs: dict) -> dict:
    return {t["family_id"]: t for t in docs["application_templates.yaml"]["templates"]}


# --------------------------------------------------------------------------- #
# Graph structure helpers
# --------------------------------------------------------------------------- #
def task_ids(template: dict) -> list[int]:
    return sorted(int(t["task_id"]) for t in template["tasks"])


def predecessors(template: dict) -> dict:
    preds = {i: [] for i in task_ids(template)}
    for e in template["edges"]:
        preds[int(e["dst"])].append(int(e["src"]))
    for i in preds:
        preds[i].sort()
    return preds


def successors(template: dict) -> dict:
    succs = {i: [] for i in task_ids(template)}
    for e in template["edges"]:
        succs[int(e["src"])].append(int(e["dst"]))
    for i in succs:
        succs[i] = sorted(set(succs[i]))
    return succs


def topo_order(template: dict) -> list[int]:
    preds = predecessors(template)
    succs = successors(template)
    indeg = {i: len(preds[i]) for i in task_ids(template)}
    ready = sorted(i for i, d in indeg.items() if d == 0)
    order: list[int] = []
    while ready:
        node = ready.pop(0)
        order.append(node)
        for nxt in succs[node]:
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                ready.append(nxt)
                ready.sort()
    if len(order) != len(indeg):
        raise DatasetError("template contains a cycle")
    return order


def sinks(template: dict) -> list[int]:
    succs = successors(template)
    return sorted(i for i in task_ids(template) if not succs[i])


def topology_signature(template: dict) -> str:
    edges = sorted((int(e["src"]), int(e["dst"]), str(e["payload_class"]))
                   for e in template["edges"])
    return sha256_json({"task_count": len(task_ids(template)), "edges": edges})


# --------------------------------------------------------------------------- #
# M6-v2 deadline construction at a supplied D_G
# --------------------------------------------------------------------------- #
def deadlines_with_D_G(template: dict, w2: dict, sla2: dict, D_G: float) -> dict:
    """Frozen M6-v2 E/L/d/slack construction evaluated at an explicit D_G.

    Kappa * CP is never used: D_G enters as the family application-E2E requirement.
    """
    dm2 = _load_module("deadline_model_v2")
    durations = dm2.task_reference_durations(template, w2)
    lb, _ref = dm2.edge_lower_bound_costs(template, w2)
    ids = sorted(durations)
    succ = {i: [] for i in ids}
    pred = {i: [] for i in ids}
    for (s, d) in lb:
        if s not in durations or d not in durations:
            raise DatasetError(f"edge {s}->{d} is not a task")
        succ[s].append(d)
        pred[d].append(s)
    order = topo_order(template)
    D_G = float(D_G)
    if not math.isfinite(D_G) or D_G <= 0:
        raise DatasetError(f"D_G must be positive and finite, got {D_G!r}")

    ES, EF = {}, {}
    for i in order:
        es = 0.0
        for p in pred[i]:
            es = max(es, EF[p] + lb.get((p, i), 0.0))
        ES[i] = es
        EF[i] = es + durations[i][1]
    LF, LS = {}, {}
    for i in reversed(order):
        lf = D_G if not succ[i] else min(LS[j] - lb.get((i, j), 0.0) for j in succ[i])
        LF[i] = lf
        LS[i] = lf - durations[i][1]

    alpha = float((sla2.get("subdeadline_rule") or {}).get("alpha_f", 0.5))
    stress = (max(EF.values()) > D_G + 1e-12) or any(EF[i] > LF[i] + 1e-12 for i in ids)
    status = "infeasible_reference_lower_bound" if stress else "ok"
    tasks: dict[int, dict] = {}
    for i in ids:
        E, L = EF[i], LF[i]
        if E <= L + 1e-12:
            slack = L - E
            d = E + alpha * max(0.0, slack)
        else:
            # A reference-tier stress case has no admissible subdeadline. It is
            # REPRESENTED with zero slack and flagged; it is never rewritten into a
            # feasible-looking deadline and never used to redefine D_G.
            L, slack, d = E, 0.0, E
        tasks[i] = {"task_id": i, "semantic_role": durations[i][0], "c_ref_s": durations[i][1],
                    "E_s": E, "L_s": L, "deadline_s": d, "slack_s": max(0.0, slack),
                    "deadline_stress": bool(stress)}
    return {"D_G_s": D_G, "alpha_f": alpha, "status": status,
            "deadline_stress": bool(stress), "tasks": tasks}


def reference_tier_critical_path(template: dict, w2: dict) -> float:
    dm2 = _load_module("deadline_model_v2")
    durations = dm2.task_reference_durations(template, w2)
    order = topo_order(template)
    pred = predecessors(template)
    finish: dict[int, float] = {}
    for i in order:
        start = max((finish[p] for p in pred[i]), default=0.0)
        finish[i] = start + durations[i][1]
    return max(finish.values())


# --------------------------------------------------------------------------- #
# Payload construction (per class, per graph)
# --------------------------------------------------------------------------- #
PAYLOAD_RULE_IDS = {
    "camera": "RAW-FRAME-SIZE-V1",
    "lidar": "LIDAR-POINTS-V1",
    "radar": "RADAR-DETECTIONS-V1",
    "feature_tensor": "FEATURE-TENSOR-V1",
    "object_list": "OBJECT-LIST-V1",
    "trajectory": "TRAJECTORY-V1",
    "control": "CONTROL-MESSAGE-V1",
    "v2x_message": "V2X-PAYLOAD-USE-CASE-V1",
}
PAYLOAD_REGISTRY_BINDING = {
    "camera": ["payload_raw_camera_v1"],
    "lidar": ["payload_lidar_v1"],
    "radar": ["payload_radar_v1"],
    "feature_tensor": ["payload_feature_tensor_v1"],
    "object_list": ["payload_object_list_v1"],
    "trajectory": ["payload_trajectory_v1"],
    "control": ["payload_control_v1"],
    "v2x_message": ["cooperative_collision_avoidance_payload",
                    "emergency_trajectory_alignment_payload"],
}
V2X_USE_CASES = ("cooperative_collision_avoidance_payload",
                 "emergency_trajectory_alignment_payload")


def payload_bytes_for_class(cfg: dict, w2: dict, class_name: str, rng: HashRNG) -> dict:
    bands = cfg["payload_bands"]["classes"]
    if class_name not in bands:
        raise DatasetError(f"payload class {class_name!r} has no generation band")
    spec = bands[class_name]
    nominal = float(w2["payload_models"][class_name]["nominal_construction_bytes"])
    if class_name == "v2x_message":
        use_case = rng.choice(list(spec["use_cases"]))
        return {"bytes": int(spec["bytes"]), "parameter": "use_case_selection",
                "use_case": use_case, "rule_id": PAYLOAD_RULE_IDS[class_name],
                "evidence_class": "standard-derived"}
    if spec["parameter"] == "scale_factor":
        scale = rng.uniform(*spec["band"])
        value = int(math.ceil(nominal * scale - 1e-9))
        return {"bytes": value, "parameter": "scale_factor", "scale_factor": scale,
                "rule_id": PAYLOAD_RULE_IDS[class_name],
                "evidence_class": "source_calibrated_synthetic"}
    if spec["parameter"] == "n_objects":
        n = int(round(rng.uniform(*spec["band"])))
        value = int(n * int(spec["bytes_per_record"]))
        return {"bytes": value, "parameter": "n_objects", "n_objects": n,
                "bytes_per_record": int(spec["bytes_per_record"]),
                "rule_id": PAYLOAD_RULE_IDS[class_name],
                "evidence_class": "source_calibrated_synthetic"}
    if spec["parameter"] == "n_waypoints":
        n = int(round(rng.uniform(*spec["band"])))
        value = int(n * int(spec["bytes_per_waypoint"]))
        return {"bytes": value, "parameter": "n_waypoints", "n_waypoints": n,
                "bytes_per_waypoint": int(spec["bytes_per_waypoint"]),
                "rule_id": PAYLOAD_RULE_IDS[class_name],
                "evidence_class": "source_calibrated_synthetic"}
    raise DatasetError(f"unknown payload parameter {spec['parameter']!r}")


def realised_payload_bands(graphs: list[dict]) -> dict:
    bands: dict[str, list[int]] = {}
    for g in graphs:
        for e in g["edges"]:
            bands.setdefault(e["payload_class"], []).append(int(e["payload_bytes"]))
    return {k: [min(v), max(v)] for k, v in bands.items() if v}


# --------------------------------------------------------------------------- #
# Provenance map (one entry per generated field group)
# --------------------------------------------------------------------------- #
def base_provenance(docs: dict) -> dict:
    w2 = docs["workload_model_v2.yaml"]
    ref = w2["reference_compute_model"]
    cfg = docs["generation_config.yaml"]
    return {
        "t_ref_fixed_planning_motif": {
            "label": "source-calibrated-synthetic",
            "rule_id": "WORKLOAD-PARTITION-PLANNING-V1",
            "registry_binding": ["ref_time_frenet_1thread",
                                 "ref_time_validator_single_thread",
                                 "ref_time_trajectory_single_thread"],
            "source_id": "OBI-2026-OJIES",
            "note": "measured planning-MOTIF aggregate 86.83 ms partitioned by frozen weights",
        },
        "t_ref_range_roles": {
            "label": "source-calibrated-synthetic",
            "rule_id": cfg["workload_draw"]["rule_id"],
            "parent_rule_ids": [r["rule_id"] for r in w2["task_class_rules"]],
            "registry_binding": [],
            "note": "drawn inside the frozen workload_model_v2 t_ref_s_range of the role class",
        },
        "compute_workload_bytes": {
            "label": "derived",
            "rule_id": "WORKLOAD-FORMULA-V2",
            "registry_binding": ["ref_tier_i9_12900hx", "cycles_per_bit_xu"],
            "formula": ref["formula"],
            "note": "W_i is a task property; it never feeds B_e",
        },
        "payload_bytes": {
            "label": "source-calibrated-synthetic",
            "rule_id": cfg["payload_bands"]["rule_id"],
            "registry_binding": sorted({b for v in PAYLOAD_REGISTRY_BINDING.values() for b in v}),
            "note": "per-class construction band around the frozen M5-v2 nominal; v2x is standard-derived",
        },
        "resource_rates": {
            "label": "source-calibrated-synthetic",
            "rule_id": cfg["resource_draw"]["rule_id"],
            "parent_rule_ids": [p["rule_id"] for p in docs["resource_profiles.yaml"]["profiles"].values()],
            "registry_binding": ["capacity_mec_ul_v1", "capacity_mec_dl_v1", "capacity_v2v_v1",
                                 "compute_ue_profile_v1", "compute_mec_profile_v1",
                                 "compute_helper_profile_v1"],
        },
        "graph_deadline": {
            "label": "source-calibrated-synthetic",
            "rule_id": cfg["deadline_draw"]["rule_id"],
            "parent_rule_ids": ["E2E-SLA-V2"],
            "registry_binding": ["e2e_sla_perception_planning_control_v1",
                                 "e2e_sla_cooperative_perception_v1",
                                 "e2e_sla_localization_prediction_planning_v1",
                                 "e2e_sla_mapping_background_v1"],
            "measurement_scope": "application_e2e",
            "semantic_role": "requirement",
        },
        "subdeadlines": {
            "label": "derived",
            "rule_id": "SUBDEADLINE-ALPHA-V2",
            "registry_binding": [],
            "required_parameter_id": "subdeadline_allocation_method",
            "evidence_source_ids": ["ECRTS-2024-DAG-TIME-CONSTRAINTS"],
        },
        "criticality": {
            "label": "source-calibrated-synthetic",
            "rule_id": docs["criticality_policy.yaml"]["policy_id"],
            "registry_binding": [],
            "required_parameter_id": "criticality_role_taxonomy",
            "evidence_source_ids": ["AUTOWARE-OFFICIAL-DOCS", "ECLIPSE-APP4MC-AMALTHEA"],
        },
        "execution_budgets": {
            "label": "source-calibrated-synthetic",
            "rule_ids": ["MC-BUDGET-RANGE-V1", "MC-BUDGET-FIXED-V1"],
            "registry_binding": [],
            "required_parameter_id": "mixed_criticality_budget_model",
            "evidence_source_ids": ["VESTAL-2007-RTSS"],
            "note": "execution-demand uncertainty, not a safety level and not WCET",
        },
        "mode_semantics": {
            "label": "source-calibrated-synthetic",
            "rule_id": docs["criticality_policy.yaml"]["mode_semantics"]["switch_rule_id"],
            "registry_binding": [],
        },
        "topology": {
            "label": "synthetic",
            "rule_id": "M4-TEMPLATE-FROZEN-V1",
            "registry_binding": [],
            "note": "edges are template-frozen; the generator never rewires a semantic payload edge",
        },
    }


# --------------------------------------------------------------------------- #
# Schema validation of a materialized graph (used by M8 tests and M10 audit)
# --------------------------------------------------------------------------- #
NUMERIC_TASK_FIELDS = (
    "task_id", "compute_workload_bytes", "t_ref_s", "task_output_bytes",
    "external_input_bytes", "empirical_execution_budget_lo_s", "E_s", "L_s",
    "deadline_s", "slack_s", "tardiness_weight",
)
NUMERIC_EDGE_FIELDS = ("src", "dst", "payload_bytes")
GRAPH_SCALAR_FIELDS = (
    "schema_version", "dataset_version", "graph_id", "parent_seed", "graph_seed",
    "application_family", "template_id", "template_lineage", "semantic_signature",
    "topology_regime", "workload_regime", "resource_profile", "resource_level",
    "sla_id", "sla_regime", "criticality_policy_id", "criticality_mixture",
    "task_count", "edge_count", "D_G_s", "P_f_s", "raw_sha256", "canonical_sha256",
)


def validate_graph_record(g: dict) -> list[str]:
    v: list[str] = []
    for key in GRAPH_SCALAR_FIELDS:
        if key not in g:
            v.append(f"{g.get('graph_id', '?')}: missing scalar field {key}")
    if g.get("dataset_version") != DATASET_VERSION:
        v.append(f"{g.get('graph_id')}: wrong dataset_version {g.get('dataset_version')!r}")
    if len(g.get("tasks") or []) != 20:
        v.append(f"{g.get('graph_id')}: task count != 20")
    if int(g.get("task_count", -1)) != len(g.get("tasks") or []):
        v.append(f"{g.get('graph_id')}: task_count disagrees with tasks")
    if int(g.get("edge_count", -1)) != len(g.get("edges") or []):
        v.append(f"{g.get('graph_id')}: edge_count disagrees with edges")
    ids = [int(t["task_id"]) for t in g["tasks"]]
    if sorted(ids) != list(range(20)):
        v.append(f"{g.get('graph_id')}: task ids are not 0..19")
    # acyclicity + reference resolution
    edges = {(int(e["src"]), int(e["dst"])) for e in g["edges"]}
    for (s, d) in edges:
        if s not in ids or d not in ids:
            v.append(f"{g.get('graph_id')}: edge {s}->{d} has an unknown endpoint")
    order = [t["task_id"] for t in g["tasks"]]
    rank = {tid: k for k, tid in enumerate(order)}
    for (s, d) in edges:
        if rank.get(s, -1) >= rank.get(d, -1):
            v.append(f"{g.get('graph_id')}: task order is not topological for {s}->{d}")
    # payload/compute separation
    for t in g["tasks"]:
        if int(t["compute_workload_bytes"]) <= 0:
            v.append(f"{g.get('graph_id')}/{t['task_id']}: W_i must be > 0")
    for e in g["edges"]:
        if int(e["payload_bytes"]) <= 0:
            v.append(f"{g.get('graph_id')}: edge payload must be > 0")
        # B_e is per edge and comes from the frozen per-class construction; it is not
        # derived from W_i and it is not resized per link.
        drawn = (((g.get("raw_draws") or {}).get("payload") or {})
                 .get(e["payload_class"], {}).get("bytes"))
        if drawn is None or int(drawn) != int(e["payload_bytes"]):
            v.append(f"{g.get('graph_id')}: edge {e['edge_id']} payload is not the declared class draw")
    if not {e["payload_class"] for e in g["edges"]} <= set(
            ((g.get("raw_draws") or {}).get("payload") or {})):
        v.append(f"{g.get('graph_id')}: an edge uses a payload class with no declared draw")
    # MC budgets
    for t in g["tasks"]:
        if float(t["empirical_execution_budget_lo_s"]) <= 0:
            v.append(f"{g.get('graph_id')}/{t['task_id']}: C_LO must be > 0")
        if t["criticality"] == "HIGH":
            hi = t.get("empirical_execution_budget_hi_s")
            if hi is None or float(hi) < float(t["empirical_execution_budget_lo_s"]):
                v.append(f"{g.get('graph_id')}/{t['task_id']}: HIGH C_HI < C_LO or missing")
        elif t.get("empirical_execution_budget_hi_s") is not None:
            v.append(f"{g.get('graph_id')}/{t['task_id']}: non-HIGH must have C_HI not_applicable")
        if float(t["E_s"]) > float(t["deadline_s"]) + 1e-12:
            v.append(f"{g.get('graph_id')}/{t['task_id']}: E > d")
        if float(t["deadline_s"]) > float(t["L_s"]) + 1e-12:
            v.append(f"{g.get('graph_id')}/{t['task_id']}: d > L")
        if float(t["slack_s"]) < -1e-12:
            v.append(f"{g.get('graph_id')}/{t['task_id']}: negative slack")
    # HIGH exists per safety family
    mix = g.get("criticality_counts") or {}
    if g.get("safety_scope") == "safety" and int(mix.get("HIGH", 0)) <= 0:
        v.append(f"{g.get('graph_id')}: safety family graph has no HIGH task")
    # canonical hash recomputation
    # canonical hash covers the whole frozen record except the hash field itself
    body = {k: v for k, v in g.items() if k != "canonical_sha256"}
    if sha256_json(body) != g.get("canonical_sha256"):
        v.append(f"{g.get('graph_id')}: canonical hash does not recompute")
    return v
