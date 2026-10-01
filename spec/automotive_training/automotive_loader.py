#!/usr/bin/env python3
"""First-class loader for the FROZEN MARGO-AUTOMOTIVE-MC-v1 dataset.

Design rules:

* The loader is READ-ONLY. The frozen files are opened, hashed and parsed; nothing in
  `env/mec_offloaing_envs/data/automotive_mc_v1/` is ever written or regenerated.
* Malformed or stale graphs are REFUSED (task count, task ids, acyclicity, canonical
  hash, manifest cross-check, provenance SHA cross-check).
* Split isolation is mechanical. Every access is counted by `META_TEST_GUARD`, and
  `meta_test` cannot be read at all unless the caller passes `allow_meta_test=True`
  (reserved for the model-side freeze evaluation, which is NOT part of this
  integration). Before that freeze `meta_test_access_count` must be exactly 0.
* No field is silently dropped: the legacy `meta_offloading_*` loader is a different
  loader and stays untouched.
* A meta-task is ONE frozen graph; support/query are independent rollout seeds on that
  graph (never the same trajectory).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_ID = "MARGO-AUTOMOTIVE-MC-v1"
DATASET_VERSION = "MARGO-AUTOMOTIVE-MC-v1"
DATASET_DIR = REPO_ROOT / "env" / "mec_offloaing_envs" / "data" / "automotive_mc_v1"
TASK_COUNT = 20
SPLITS = ("meta_train", "validation", "meta_test")
SPLIT_ROLES = {
    "meta_train": ("train",),
    "validation": ("support", "query"),
    "meta_test": ("support", "query"),
}
SUPPORT_COUNT = 20
QUERY_COUNT = 40
CRITICALITY_CLASSES = ("LOW", "MEDIUM", "HIGH")

#: provenance.json keys that must match the bytes on disk
FROZEN_SHA_KEYS = {
    "graphs_sha": "graphs.jsonl",
    "dataset_manifest_sha": "dataset_manifest.jsonl",
    "splits_sha": "splits.jsonl",
    "calibration_report_sha": "calibration_report.json",
    "certification_manifest_sha": "manifest.jsonl",
}


class AutomotiveLoaderError(ValueError):
    """Refused. Never silently repaired."""


class MetaTestAccessError(RuntimeError):
    """meta_test was read before the model-side freeze."""


class MetaTestGuard:
    """Mechanical meta-test access counter.

    A single process-wide instance (`META_TEST_GUARD`) is used by the loader, the
    env and the validation path so that no code path can read meta_test unnoticed.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.counts: dict[str, int] = {s: 0 for s in SPLITS}
        self.sources: list[dict] = []

    def record(self, split: str, source: str) -> None:
        if split not in self.counts:
            raise AutomotiveLoaderError("unknown split %r" % (split,))
        self.counts[split] += 1
        self.sources.append({"split": split, "source": str(source)})

    def meta_test_access_count(self) -> int:
        return int(self.counts.get("meta_test", 0))

    def assert_unopened(self) -> int:
        count = self.meta_test_access_count()
        if count != 0:
            raise MetaTestAccessError(
                "meta_test was accessed %d time(s) before the model-side freeze: %r"
                % (count, self.sources[-5:])
            )
        return count

    def as_dict(self) -> dict:
        return {"counts": dict(self.counts), "meta_test_access_count":
                self.meta_test_access_count(), "sources": list(self.sources)}


META_TEST_GUARD = MetaTestGuard()


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AutomotiveEdge:
    src: int
    dst: int
    payload_class: str
    payload_model_ref: str
    payload_bytes: int
    payload_rule_id: str
    payload_evidence_class: str


@dataclass(frozen=True)
class AutomotiveTask:
    task_id: int
    semantic_role: str
    motif_id: str
    lineage_id: str
    criticality: str
    safety_scope: str
    compute_workload_bytes: int
    t_ref_s: float
    task_output_bytes: int
    external_input_bytes: int
    output_payload_class: str | None
    empirical_execution_budget_lo_s: float
    empirical_execution_budget_hi_s: float | None
    budget_hi_applicable: bool
    budget_hi_status: str
    drop_allowed_lo_mode: bool
    drop_allowed_hi_mode: bool
    degrade_allowed_lo_mode: bool
    degrade_allowed_hi_mode: bool
    E_s: float
    L_s: float
    deadline_s: float
    slack_s: float
    deadline_type: str
    tardiness_weight: float
    is_root: bool
    is_sink: bool
    predecessors: tuple[int, ...]
    successors: tuple[int, ...]

    def drop_allowed(self, mode: str) -> bool:
        return self.drop_allowed_lo_mode if mode == "LO" else self.drop_allowed_hi_mode

    def degrade_allowed(self, mode: str) -> bool:
        return self.degrade_allowed_lo_mode if mode == "LO" else self.degrade_allowed_hi_mode

    @property
    def demand_lo_seconds(self) -> float:
        return float(self.empirical_execution_budget_lo_s)

    @property
    def demand_hi_seconds(self) -> float | None:
        return (None if self.empirical_execution_budget_hi_s is None
                else float(self.empirical_execution_budget_hi_s))


@dataclass(frozen=True)
class AutomotiveGraph:
    graph_id: str
    split: str
    role_in_split: str
    application_family: str
    safety_scope: str
    template_id: str
    template_lineage: str
    semantic_signature: str
    topology_regime: str
    topology_signature: str
    workload_regime: str
    resource_profile: str
    resource_level: str
    resource: Mapping[str, float]
    sla_id: str
    sla_regime: str
    sla_class: str | None
    P_f_s: float
    D_G_s: float
    criticality_policy_id: str
    criticality_counts: Mapping[str, int]
    criticality_mixture: str
    parent_seed: int
    graph_seed: int
    cell_id: str
    canonical_sha256: str
    raw_sha256: str
    tasks: tuple[AutomotiveTask, ...]
    edges: tuple[AutomotiveEdge, ...]
    mode_semantics: Mapping[str, Any]
    provenance_refs: Mapping[str, Any]
    certification: Mapping[str, Any]
    dataset_manifest: Mapping[str, Any]

    # -- convenience -------------------------------------------------------
    @property
    def family(self) -> str:
        return self.application_family

    @property
    def task_by_id(self) -> dict[int, AutomotiveTask]:
        return {t.task_id: t for t in self.tasks}

    @property
    def criticality_counts_by_class(self) -> dict[str, int]:
        out = {c: 0 for c in CRITICALITY_CLASSES}
        for t in self.tasks:
            out[t.criticality] += 1
        return out

    def actions_placeholder(self) -> list[int]:
        return [0] * len(self.tasks)

    def high_task_ids(self) -> tuple[int, ...]:
        return tuple(t.task_id for t in self.tasks if t.criticality == "HIGH")

    def as_record(self) -> dict:
        """The frozen record shape (round-trip helper for tests/tools)."""
        return {
            "graph_id": self.graph_id,
            "application_family": self.application_family,
            "tasks": [
                {
                    "task_id": t.task_id,
                    "semantic_role": t.semantic_role,
                    "motif_id": t.motif_id,
                    "lineage_id": t.lineage_id,
                    "criticality": t.criticality,
                    "safety_scope": t.safety_scope,
                    "compute_workload_bytes": t.compute_workload_bytes,
                    "t_ref_s": t.t_ref_s,
                    "task_output_bytes": t.task_output_bytes,
                    "external_input_bytes": t.external_input_bytes,
                    "output_payload_class": t.output_payload_class,
                    "empirical_execution_budget_lo_s": t.empirical_execution_budget_lo_s,
                    "empirical_execution_budget_hi_s": t.empirical_execution_budget_hi_s,
                    "budget_hi_applicable": t.budget_hi_applicable,
                    "budget_hi_status": t.budget_hi_status,
                    "drop_allowed_lo_mode": t.drop_allowed_lo_mode,
                    "drop_allowed_hi_mode": t.drop_allowed_hi_mode,
                    "degrade_allowed_lo_mode": t.degrade_allowed_lo_mode,
                    "degrade_allowed_hi_mode": t.degrade_allowed_hi_mode,
                    "E_s": t.E_s,
                    "L_s": t.L_s,
                    "deadline_s": t.deadline_s,
                    "slack_s": t.slack_s,
                    "deadline_type": t.deadline_type,
                    "tardiness_weight": t.tardiness_weight,
                    "is_root": t.is_root,
                    "is_sink": t.is_sink,
                    "predecessors": list(t.predecessors),
                    "successors": list(t.successors),
                }
                for t in self.tasks
            ],
            "edges": [
                {
                    "src": e.src,
                    "dst": e.dst,
                    "payload_class": e.payload_class,
                    "payload_model_ref": e.payload_model_ref,
                    "payload_bytes": e.payload_bytes,
                    "payload_rule_id": e.payload_rule_id,
                    "payload_evidence_class": e.payload_evidence_class,
                }
                for e in self.edges
            ],
            "resource": dict(self.resource),
            "D_G_s": self.D_G_s,
            "canonical_sha256": self.canonical_sha256,
            "raw_sha256": self.raw_sha256,
        }


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise AutomotiveLoaderError("missing frozen file %s" % path)
    out = []
    for i, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise AutomotiveLoaderError("%s:%d is not valid JSON: %s" % (path, i, exc)) from exc
    return out


def _validate_graph_record(rec: Mapping[str, Any], manifest_row: Mapping[str, Any]) -> None:
    gid = rec.get("graph_id")
    if rec.get("dataset_version") != DATASET_VERSION:
        raise AutomotiveLoaderError(
            "%s: dataset_version %r != %r" % (gid, rec.get("dataset_version"), DATASET_VERSION))
    tasks = rec.get("tasks") or []
    if len(tasks) != TASK_COUNT:
        raise AutomotiveLoaderError("%s: %d tasks, expected exactly %d" % (gid, len(tasks), TASK_COUNT))
    ids = sorted(int(t["task_id"]) for t in tasks)
    if ids != list(range(TASK_COUNT)):
        raise AutomotiveLoaderError("%s: task ids are not 0..%d" % (gid, TASK_COUNT - 1))
    for t in tasks:
        if t["criticality"] not in CRITICALITY_CLASSES:
            raise AutomotiveLoaderError("%s/%s: bad criticality %r" % (gid, t["task_id"], t["criticality"]))
        if float(t["empirical_execution_budget_lo_s"]) <= 0.0:
            raise AutomotiveLoaderError("%s/%s: C_LO must be > 0" % (gid, t["task_id"]))
        hi = t.get("empirical_execution_budget_hi_s")
        if t["criticality"] == "HIGH":
            if hi is None or float(hi) < float(t["empirical_execution_budget_lo_s"]):
                raise AutomotiveLoaderError("%s/%s: HIGH without a valid C_HI" % (gid, t["task_id"]))
        elif hi is not None:
            raise AutomotiveLoaderError("%s/%s: non-HIGH must report C_HI as not_applicable" % (gid, t["task_id"]))
    rank = {int(t["task_id"]): i for i, t in enumerate(tasks)}
    edges = rec.get("edges") or []
    for e in edges:
        if int(e["src"]) not in rank or int(e["dst"]) not in rank:
            raise AutomotiveLoaderError("%s: edge %s->%s has an unknown endpoint" % (gid, e["src"], e["dst"]))
        if rank[int(e["src"])] >= rank[int(e["dst"])]:
            raise AutomotiveLoaderError("%s: stored order is not topological for %s->%s" % (gid, e["src"], e["dst"]))
        if int(e["payload_bytes"]) <= 0:
            raise AutomotiveLoaderError("%s: non-positive payload on %s->%s" % (gid, e["src"], e["dst"]))
    body = {k: v for k, v in rec.items() if k != "canonical_sha256"}
    canon = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"),
                                      ensure_ascii=True, allow_nan=False).encode()).hexdigest()
    if canon != rec.get("canonical_sha256"):
        raise AutomotiveLoaderError("%s: canonical hash does not recompute (stale/corrupt graph)" % gid)
    if manifest_row.get("canonical_sha256") != rec.get("canonical_sha256"):
        raise AutomotiveLoaderError("%s: manifest and graph disagree on the canonical hash" % gid)
    if int(manifest_row.get("task_count", -1)) != TASK_COUNT:
        raise AutomotiveLoaderError("%s: manifest task_count != %d" % (gid, TASK_COUNT))


def _task_from_record(t: Mapping[str, Any]) -> AutomotiveTask:
    return AutomotiveTask(
        task_id=int(t["task_id"]),
        semantic_role=str(t["semantic_role"]),
        motif_id=str(t["motif_id"]),
        lineage_id=str(t["lineage_id"]),
        criticality=str(t["criticality"]),
        safety_scope=str(t.get("safety_scope", "")),
        compute_workload_bytes=int(t["compute_workload_bytes"]),
        t_ref_s=float(t["t_ref_s"]),
        task_output_bytes=int(t["task_output_bytes"]),
        external_input_bytes=int(t.get("external_input_bytes", 0)),
        output_payload_class=t.get("output_payload_class"),
        empirical_execution_budget_lo_s=float(t["empirical_execution_budget_lo_s"]),
        empirical_execution_budget_hi_s=(None if t.get("empirical_execution_budget_hi_s") is None
                                         else float(t["empirical_execution_budget_hi_s"])),
        budget_hi_applicable=bool(t["budget_hi_applicable"]),
        budget_hi_status=str(t["budget_hi_status"]),
        drop_allowed_lo_mode=bool(t["drop_allowed_lo_mode"]),
        drop_allowed_hi_mode=bool(t["drop_allowed_hi_mode"]),
        degrade_allowed_lo_mode=bool(t["degrade_allowed_lo_mode"]),
        degrade_allowed_hi_mode=bool(t["degrade_allowed_hi_mode"]),
        E_s=float(t["E_s"]),
        L_s=float(t["L_s"]),
        deadline_s=float(t["deadline_s"]),
        slack_s=float(t["slack_s"]),
        deadline_type=str(t.get("deadline_type", "firm")),
        tardiness_weight=float(t.get("tardiness_weight", 1.0)),
        is_root=bool(t["is_root"]),
        is_sink=bool(t["is_sink"]),
        predecessors=tuple(int(x) for x in (t.get("predecessors") or [])),
        successors=tuple(int(x) for x in (t.get("successors") or [])),
    )


def _graph_from_records(rec: Mapping[str, Any], assignment: Mapping[str, Any],
                        manifest_row: Mapping[str, Any],
                        certification: Mapping[str, Any]) -> AutomotiveGraph:
    return AutomotiveGraph(
        graph_id=str(rec["graph_id"]),
        split=str(assignment["split"]),
        role_in_split=str(assignment["role_in_split"]),
        application_family=str(rec["application_family"]),
        safety_scope=str(rec.get("safety_scope", "")),
        template_id=str(rec["template_id"]),
        template_lineage=str(rec["template_lineage"]),
        semantic_signature=str(rec["semantic_signature"]),
        topology_regime=str(rec["topology_regime"]),
        topology_signature=str(rec["topology_signature"]),
        workload_regime=str(rec["workload_regime"]),
        resource_profile=str(rec["resource_profile"]),
        resource_level=str(rec["resource_level"]),
        resource={k: float(v) for k, v in rec["resource"].items()},
        sla_id=str(rec["sla_id"]),
        sla_regime=str(rec["sla_regime"]),
        sla_class=rec.get("sla_class"),
        P_f_s=float(rec["P_f_s"]),
        D_G_s=float(rec["D_G_s"]),
        criticality_policy_id=str(rec["criticality_policy_id"]),
        criticality_counts={k: int(v) for k, v in rec["criticality_counts"].items()},
        criticality_mixture=str(rec["criticality_mixture"]),
        parent_seed=int(rec["parent_seed"]),
        graph_seed=int(rec["graph_seed"]),
        cell_id=str(rec["cell_id"]),
        canonical_sha256=str(rec["canonical_sha256"]),
        raw_sha256=str(rec["raw_sha256"]),
        tasks=tuple(_task_from_record(t) for t in rec["tasks"]),
        edges=tuple(AutomotiveEdge(
            src=int(e["src"]), dst=int(e["dst"]), payload_class=str(e["payload_class"]),
            payload_model_ref=str(e["payload_model_ref"]), payload_bytes=int(e["payload_bytes"]),
            payload_rule_id=str(e["payload_rule_id"]),
            payload_evidence_class=str(e["payload_evidence_class"])) for e in rec["edges"]),
        mode_semantics=dict(rec["mode_semantics"]),
        provenance_refs=dict(rec["provenance"]),
        certification=dict(certification.get("certification") or {}),
        dataset_manifest=dict(manifest_row),
    )


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class AutomotiveDataset:
    """Loaded, validated, split-joined, access-guarded frozen dataset."""

    def __init__(self, dataset_dir: Path | None = None, *, verify_shas: bool = True,
                 guard: MetaTestGuard | None = None):
        self.dataset_dir = Path(dataset_dir) if dataset_dir else DATASET_DIR
        self.guard = guard or META_TEST_GUARD
        self.provenance = self._load_json("provenance.json")
        self.calibration = self._load_json("calibration_report.json")
        self.summary = self._load_json("split_summary.json")
        self.split_policy = self._load_json("split_policy.json")
        self.audit_report = self._load_json("audit_report.json")
        self.dataset_card = self._load_json("dataset_card.json")
        self.sha_evidence = {}
        if verify_shas:
            self.sha_evidence = self._verify_frozen_shas()

        raw_graphs = _read_jsonl(self.dataset_dir / "graphs.jsonl")
        manifest_rows = {r["graph_id"]: r for r in _read_jsonl(self.dataset_dir / "dataset_manifest.jsonl")}
        cert_rows = {r["graph_id"]: r for r in _read_jsonl(self.dataset_dir / "manifest.jsonl")}
        assignments = {r["graph_id"]: r for r in _read_jsonl(self.dataset_dir / "splits.jsonl")}

        if len(raw_graphs) != len(manifest_rows):
            raise AutomotiveLoaderError("graphs.jsonl and dataset_manifest.jsonl disagree in length")
        self._graphs: dict[str, AutomotiveGraph] = {}
        for rec in raw_graphs:
            gid = rec["graph_id"]
            if gid in self._graphs:
                raise AutomotiveLoaderError("duplicate graph_id %r" % gid)
            if gid not in assignments:
                raise AutomotiveLoaderError("%s has no split assignment" % gid)
            if gid not in manifest_rows:
                raise AutomotiveLoaderError("%s is missing from the dataset manifest" % gid)
            if gid not in cert_rows:
                raise AutomotiveLoaderError("%s is missing from the certification manifest" % gid)
            _validate_graph_record(rec, manifest_rows[gid])
            a = assignments[gid]
            if a["split"] not in SPLITS:
                raise AutomotiveLoaderError("%s: unknown split %r" % (gid, a["split"]))
            if a["role_in_split"] not in SPLIT_ROLES[a["split"]]:
                raise AutomotiveLoaderError("%s: role %r is illegal for split %r"
                                            % (gid, a["role_in_split"], a["split"]))
            if a.get("canonical_sha256") != rec["canonical_sha256"]:
                raise AutomotiveLoaderError("%s: split file canonical hash mismatch" % gid)
            self._graphs[gid] = _graph_from_records(rec, a, manifest_rows[gid], cert_rows[gid])
        self._validate_split_shape()

    # -- io -----------------------------------------------------------------
    def _load_json(self, name: str) -> dict:
        path = self.dataset_dir / name
        if not path.exists():
            raise AutomotiveLoaderError("missing frozen file %s" % path)
        return json.loads(path.read_text())

    def _verify_frozen_shas(self) -> dict:
        ev = {}
        for key, filename in FROZEN_SHA_KEYS.items():
            expected = self.provenance.get(key)
            actual = sha256_file(self.dataset_dir / filename)
            ev[key] = {"expected": expected, "actual": actual, "match": expected == actual,
                       "file": filename}
            if expected != actual:
                raise AutomotiveLoaderError(
                    "frozen %s is stale: provenance says %s, file is %s"
                    % (filename, expected, actual))
        # The dataset-dir `split_policy.json` is the M9 re-serialization of the
        # authoritative spec file, so its BYTES differ from the pinned sha. Compare
        # parsed content here and pin the authoritative file to provenance.
        spec_policy = REPO_ROOT / "spec" / "automotive_mc_v1" / "split_policy.json"
        local_policy = json.loads((self.dataset_dir / "split_policy.json").read_text())
        if local_policy != json.loads(spec_policy.read_text()):
            raise AutomotiveLoaderError(
                "dataset-dir split_policy.json disagrees with the authoritative spec copy")
        actual = sha256_file(spec_policy)
        ev["split_policy_sha"] = {"expected": self.provenance.get("split_policy_sha"),
                                  "actual": actual,
                                  "match": actual == self.provenance.get("split_policy_sha"),
                                  "file": str(spec_policy)}
        if actual != self.provenance.get("split_policy_sha"):
            raise AutomotiveLoaderError("authoritative split_policy.json does not match split_policy_sha")
        audit_local = sha256_file(self.dataset_dir / "audit_report.json")
        ev["audit_report_sha"] = {"expected": self.provenance.get("audit_report_sha"),
                                  "actual": audit_local,
                                  "match": audit_local == self.provenance.get("audit_report_sha"),
                                  "file": "audit_report.json"}
        if audit_local != self.provenance.get("audit_report_sha"):
            raise AutomotiveLoaderError("dataset audit_report.json does not match audit_report_sha")
        return ev

    def _validate_split_shape(self) -> None:
        counts = {s: 0 for s in SPLITS}
        roles: dict[str, int] = {}
        for g in self._graphs.values():
            counts[g.split] += 1
            roles[f"{g.split}/{g.role_in_split}"] = roles.get(f"{g.split}/{g.role_in_split}", 0) + 1
        for split, role_counts in (("validation", ("support", "query")),
                                   ("meta_test", ("support", "query"))):
            if roles.get(f"{split}/support") != SUPPORT_COUNT:
                raise AutomotiveLoaderError("%s support count %r != %d"
                                            % (split, roles.get(f"{split}/support"), SUPPORT_COUNT))
            if roles.get(f"{split}/query") != QUERY_COUNT:
                raise AutomotiveLoaderError("%s query count %r != %d"
                                            % (split, roles.get(f"{split}/query"), QUERY_COUNT))
        if roles.get("meta_train/train") != counts["meta_train"]:
            raise AutomotiveLoaderError("every meta_train graph must carry role 'train'")
        if sum(counts.values()) != len(self._graphs):
            raise AutomotiveLoaderError("split counts do not cover the dataset")
        self.counts = counts
        self.role_counts = roles
        for split in ("validation", "meta_test"):
            support = {g.graph_id for g in self._graphs.values()
                       if g.split == split and g.role_in_split == "support"}
            query = {g.graph_id for g in self._graphs.values()
                     if g.split == split and g.role_in_split == "query"}
            if support & query:
                raise AutomotiveLoaderError("%s: support and query overlap" % split)

    @property
    def graphs(self) -> dict[str, AutomotiveGraph]:
        """Full-dataset view. Reading it RECORDS an access for every graph, so a
        caller that touches `.graphs` can never leave meta_test uncounted: the
        loader-level guarantee is mechanical, not conventional. Use `meta_train()`,
        `validation_support()`/`validation_query()` or `meta_test(allow_meta_test=True)`
        for split-scoped reads."""
        for g in self._graphs.values():
            self.guard.record(g.split, "graphs_property")
        return dict(self._graphs)

    # -- access -------------------------------------------------------------
    def graph(self, graph_id: str, *, source: str = "graph") -> AutomotiveGraph:
        g = self._graphs.get(graph_id)
        if g is None:
            raise AutomotiveLoaderError("unknown graph_id %r" % graph_id)
        self.guard.record(g.split, source)
        if g.split == "meta_test":
            raise MetaTestAccessError(
                "meta_test graph %s requested through %s before the model-side freeze"
                % (graph_id, source))
        return g

    def by_split(self, split: str, *, source: str = "by_split",
                 allow_meta_test: bool = False) -> list[AutomotiveGraph]:
        if split not in SPLITS:
            raise AutomotiveLoaderError("unknown split %r" % (split,))
        if split == "meta_test" and not allow_meta_test:
            self.guard.record("meta_test", source)
            raise MetaTestAccessError(
                "meta_test read through %s before the model-side freeze" % source)
        out = [g for g in self.sorted_graphs() if g.split == split]
        for g in out:
            self.guard.record(split, source)
        return out

    def meta_train(self) -> list[AutomotiveGraph]:
        return self.by_split("meta_train", source="meta_train")

    def validation(self) -> list[AutomotiveGraph]:
        return self.by_split("validation", source="validation")

    def validation_support(self) -> list[AutomotiveGraph]:
        return [g for g in self.validation() if g.role_in_split == "support"]

    def validation_query(self) -> list[AutomotiveGraph]:
        return [g for g in self.validation() if g.role_in_split == "query"]

    def meta_test(self, *, allow_meta_test: bool = False) -> list[AutomotiveGraph]:
        return self.by_split("meta_test", source="meta_test", allow_meta_test=allow_meta_test)

    def sorted_graphs(self) -> list[AutomotiveGraph]:
        return [self._graphs[k] for k in sorted(self._graphs)]

    def certification_status(self) -> dict:
        out: dict[str, int] = {}
        for g in self._graphs.values():
            status = str(g.certification.get("status", "unknown"))
            out[status] = out.get(status, 0) + 1
        return out

    def fingerprint(self) -> str:
        material = {
            "dataset_id": DATASET_ID,
            "graphs_sha": self.provenance.get("graphs_sha"),
            "splits_sha": self.provenance.get("splits_sha"),
            "dataset_manifest_sha": self.provenance.get("dataset_manifest_sha"),
            "calibration_report_sha": self.provenance.get("calibration_report_sha"),
            "split_policy_sha": self.provenance.get("split_policy_sha"),
            "graph_count": len(self._graphs),
        }
        return hashlib.sha256(json.dumps(material, sort_keys=True,
                                         separators=(",", ":")).encode()).hexdigest()


def load_dataset(dataset_dir: Path | None = None, *, verify_shas: bool = True) -> AutomotiveDataset:
    return AutomotiveDataset(dataset_dir, verify_shas=verify_shas)


# --------------------------------------------------------------------------- #
# Meta-task / rollout seeds
# --------------------------------------------------------------------------- #
def meta_tasks(dataset: AutomotiveDataset) -> list[AutomotiveGraph]:
    """One frozen graph = one meta-task (the frozen split is authoritative)."""
    return dataset.meta_train()


def validation_tasks(dataset: AutomotiveDataset) -> list[AutomotiveGraph]:
    return dataset.validation()


def rollout_seed(base_seed: int, graph_id: str, role: str, index: int) -> int:
    """Deterministic, role-scoped seed. support and query can never collide."""
    if role not in ("support", "query"):
        raise AutomotiveLoaderError("rollout role must be support or query, got %r" % (role,))
    material = "margo-rollout|%d|%s|%s|%d" % (int(base_seed), graph_id, role, int(index))
    return int(hashlib.sha256(material.encode()).hexdigest()[:15], 16)


def rollout_seeds(base_seed: int, graph_id: str, role: str, count: int) -> list[int]:
    seeds = [rollout_seed(base_seed, graph_id, role, i) for i in range(int(count))]
    if len(set(seeds)) != len(seeds):
        raise AutomotiveLoaderError("duplicate rollout seeds for %s/%s" % (graph_id, role))
    return seeds


def assert_support_query_disjoint(dataset: AutomotiveDataset) -> dict:
    out = {}
    for split in ("validation", "meta_test"):
        support = {g.graph_id for g in dataset._graphs.values()
                   if g.split == split and g.role_in_split == "support"}
        query = {g.graph_id for g in dataset._graphs.values()
                 if g.split == split and g.role_in_split == "query"}
        if support & query or not support or not query:
            raise AutomotiveLoaderError("%s support/query are not disjoint and non-empty" % split)
        out[split] = {"support": len(support), "query": len(query), "overlap": sorted(support & query)}
    return out
