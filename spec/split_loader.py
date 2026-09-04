"""Frozen split accessors. Training may only see meta_train distributions."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

SPEC = Path(__file__).resolve().parent
ROOT = SPEC.parent
MANIFEST = SPEC / "dataset_manifest.jsonl"
SPLIT_POLICY = SPEC / "split_policy.json"

GRAPH_PREFIX_RE = re.compile(r"offload_random20_(\d+)/random\.20\.(\d+)\.gv$")

TRAIN_ONLY_ROLES = frozenset({"meta_train"})
HELD_OUT_SUPPORT = frozenset({"validation_support", "meta_test_support"})
HELD_OUT_QUERY = frozenset({"validation_query", "meta_test_query"})


def load_split_policy() -> dict:
    return json.loads(SPLIT_POLICY.read_text())


def load_manifest_rows() -> list[dict]:
    rows = []
    with MANIFEST.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_graph_ref(relative_path: str) -> tuple[int, int]:
    match = GRAPH_PREFIX_RE.search(relative_path.replace("\\", "/"))
    if match is None:
        raise ValueError(f"cannot parse distribution/graph from {relative_path}")
    return int(match.group(1)), int(match.group(2))


def graph_prefix(distribution_id: int) -> str:
    return (
        "./env/mec_offloaing_envs/data/meta_offloading_20/"
        f"offload_random20_{int(distribution_id)}/random.20."
    )


def rows_by_role() -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in load_manifest_rows():
        grouped[str(row["role"])].append(row)
    return grouped


def meta_train_distribution_ids() -> list[int]:
    policy = load_split_policy()
    ids = [int(x) for x in policy["meta_train_distribution_ids"]]
    if len(ids) != 15:
        raise ValueError(f"expected 15 meta_train distributions, got {len(ids)}")
    return ids


def validation_distribution_ids() -> list[int]:
    return [int(x) for x in load_split_policy()["validation_distribution_ids"]]


def meta_test_distribution_ids() -> list[int]:
    return [int(x) for x in load_split_policy()["meta_test_distribution_ids"]]


def meta_train_graph_prefixes() -> list[str]:
    return [graph_prefix(dist_id) for dist_id in meta_train_distribution_ids()]


def graph_indices_for_role(distribution_id: int, role: str) -> list[int]:
    indices = []
    for row in load_manifest_rows():
        if int(row["distribution_id"]) != int(distribution_id):
            continue
        if str(row["role"]) != role:
            continue
        _, graph_idx = parse_graph_ref(row["relative_path"])
        indices.append(graph_idx)
    indices.sort()
    return indices


def assert_train_prefixes(paths: list[str]) -> None:
    allowed_ids = set(meta_train_distribution_ids())
    forbidden_ids = set(validation_distribution_ids()) | set(meta_test_distribution_ids())
    if len(paths) != 15:
        raise ValueError(f"trainer must load 15 meta_train prefixes, got {len(paths)}")
    seen = []
    for path in paths:
        match = re.search(r"offload_random20_(\d+)", path.replace("\\", "/"))
        if match is None:
            raise ValueError(f"cannot parse distribution id from {path}")
        dist_id = int(match.group(1))
        if dist_id in forbidden_ids:
            raise ValueError(f"held-out distribution leaked into trainer: {path}")
        if dist_id not in allowed_ids:
            raise ValueError(f"non-meta_train prefix in trainer: {path}")
        seen.append(dist_id)
    if set(seen) != allowed_ids:
        raise ValueError(f"trainer prefixes must be exactly the 15 meta_train ids, got {sorted(seen)}")
