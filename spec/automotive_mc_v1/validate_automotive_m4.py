#!/usr/bin/env python3
"""M4 validator: semantic roles, families, payload classes, templates. Exit != 0.

Validates `task_semantics.yaml` always, and `application_templates.yaml` when it
exists. Templates are the 20-task compositions; the semantics file alone is a
valid intermediate state, but a template that violates any rule fails loudly.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

BASE = Path(__file__).resolve().parent
SEMANTICS = BASE / "task_semantics.yaml"
TEMPLATES = BASE / "application_templates.yaml"
REQUIRED_TEMPLATE_TASKS = 20
FORBIDDEN_PAYLOAD = "raw_sensor"


def validate_semantics(doc: dict) -> tuple[list[str], dict]:
    v: list[str] = []
    classes = list(doc.get("payload_classes") or [])
    models = doc.get("payload_models") or {}
    families = doc.get("families") or {}
    roles = doc.get("roles") or []
    if len(families) != 4:
        v.append(f"expected exactly 4 frozen families, found {len(families)}")
    if FORBIDDEN_PAYLOAD in classes:
        v.append(f"payload class {FORBIDDEN_PAYLOAD!r} is forbidden")
    for cls in classes:
        if cls not in models:
            v.append(f"payload class {cls!r} has no payload_model_ref")
    names = set()
    for role in roles:
        name = role.get("semantic_role")
        if not name:
            v.append("role without semantic_role")
            continue
        if name in names:
            v.append(f"duplicate semantic role {name!r}")
        names.add(name)
        for field in ("motif_role", "families", "payload_in", "payload_out"):
            if role.get(field) is None:
                v.append(f"role {name}: missing {field}")
        for fam in role.get("families") or []:
            if fam not in families:
                v.append(f"role {name}: unknown family {fam!r}")
        for cls in (role.get("payload_in") or []) + (role.get("payload_out") or []):
            if cls not in classes:
                v.append(f"role {name}: unknown payload class {cls!r}")
        if "criticality_class" in role or "criticality" in role:
            v.append(f"role {name}: criticality must not be assigned in the semantics layer")
    for fam, spec in families.items():
        for field in ("function", "provenance_refs", "required_motifs", "sla_rule_ref",
                      "criticality_policy_ref", "resource_profile_compatibility"):
            if not spec.get(field):
                v.append(f"family {fam}: missing {field}")
    rules = doc.get("rules") or {}
    if not rules.get("cooperative_edges_must_be_v2x"):
        v.append("rules must require cooperative edges to be real V2X")
    return v, {"roles": len(roles), "families": len(families), "payload_classes": len(classes)}


def validate_templates(doc: dict, semantics: dict) -> tuple[list[str], dict]:
    v: list[str] = []
    classes = set(semantics.get("payload_classes") or [])
    models = semantics.get("payload_models") or {}
    roles = {r["semantic_role"] for r in (semantics.get("roles") or [])}
    families = set(semantics.get("families") or {})
    templates = doc.get("templates") or []
    for tpl in templates:
        tid = tpl.get("template_id", "?")
        fam = tpl.get("family_id")
        if fam not in families:
            v.append(f"{tid}: unknown family {fam!r}")
        tasks = tpl.get("tasks") or []
        if len(tasks) != REQUIRED_TEMPLATE_TASKS:
            v.append(f"{tid}: task_count {len(tasks)} != {REQUIRED_TEMPLATE_TASKS}")
        ids = [t.get("task_id") for t in tasks]
        if len(set(ids)) != len(ids):
            v.append(f"{tid}: duplicate task ids")
        for t in tasks:
            for field in ("semantic_role", "motif_id", "family_id", "lineage_id"):
                if t.get(field) is None:
                    v.append(f"{tid}/{t.get('task_id')}: missing {field}")
            if t.get("semantic_role") not in roles:
                v.append(f"{tid}/{t.get('task_id')}: unknown semantic_role {t.get('semantic_role')!r}")
            if t.get("family_id") != fam:
                v.append(f"{tid}/{t.get('task_id')}: family_id disagrees with template")
            if "criticality_class" in t:
                v.append(f"{tid}/{t.get('task_id')}: criticality must not be baked into the template")
        known = set(ids)
        graph: dict = {i: [] for i in ids}
        for e in tpl.get("edges") or []:
            src, dst = e.get("src"), e.get("dst")
            if src not in known or dst not in known:
                v.append(f"{tid}: invalid edge endpoint {src}->{dst}")
                continue
            cls = e.get("payload_class")
            if cls not in classes:
                v.append(f"{tid}: unknown payload_class {cls!r}")
            if cls == FORBIDDEN_PAYLOAD:
                v.append(f"{tid}: forbidden payload class {FORBIDDEN_PAYLOAD!r}")
            if e.get("payload_model_ref") != models.get(cls):
                v.append(f"{tid}: payload_model_ref does not resolve for class {cls!r}")
            if not e.get("provenance_or_rule_ref"):
                v.append(f"{tid}: edge {src}->{dst} lacks provenance/rule ref")
            graph.setdefault(src, []).append(dst)
        # acyclicity (Kahn)
        indeg = {i: 0 for i in ids}
        for src, outs in graph.items():
            for dst in outs:
                indeg[dst] = indeg.get(dst, 0) + 1
        queue = [i for i, d in indeg.items() if d == 0]
        seen = 0
        while queue:
            node = queue.pop()
            seen += 1
            for nxt in graph.get(node, []):
                indeg[nxt] -= 1
                if indeg[nxt] == 0:
                    queue.append(nxt)
        if seen != len(ids):
            v.append(f"{tid}: template contains a cycle")
    return v, {"templates": len(templates),
               "materializable_templates": sum(1 for t in templates
                                               if len(t.get("tasks") or []) == REQUIRED_TEMPLATE_TASKS)}


def main() -> int:
    semantics = yaml.safe_load(SEMANTICS.read_text())
    sv, sstats = validate_semantics(semantics)
    tv: list[str] = []
    tstats = {"templates": 0, "materializable_templates": 0}
    if TEMPLATES.exists():
        tv, tstats = validate_templates(yaml.safe_load(TEMPLATES.read_text()), semantics)
    violations = sv + tv
    print(json.dumps({**sstats, **tstats,
                      "status": "PASS" if not violations else "FAIL",
                      "violations": violations}, indent=2, sort_keys=True))
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
