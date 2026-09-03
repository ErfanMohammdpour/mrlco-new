#!/usr/bin/env python3
"""Fit encoder z-score statistics on meta_train graphs only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env.mec_offloaing_envs.offloading_task_graph import OffloadingTaskGraph  # noqa: E402
from env.mec_offloaing_envs.scheduler.adapter import to_canonical_dag  # noqa: E402
from env.mec_offloaing_envs.scheduler.encoder_obs import (  # noqa: E402
    MAX_NEIGH,
    fit_feature_stats,
    raw_node_features,
    spec_source_hashes,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "spec" / "dataset_manifest.jsonl",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "spec" / "encoder_feature_stats.json",
    )
    args = parser.parse_args()

    rows = []
    n_graphs = 0
    max_in = 0
    max_out = 0
    other_roles = 0
    with args.manifest.open() as handle:
        for line in handle:
            rec = json.loads(line)
            role = rec.get("role")
            if role != "meta_train":
                if role in {
                    "validation_support",
                    "validation_query",
                    "meta_test_support",
                    "meta_test_query",
                }:
                    other_roles += 1
                continue
            path = ROOT / rec["relative_path"]
            dag = to_canonical_dag(OffloadingTaskGraph(str(path)))
            order = sorted(dag.tasks)
            raw = raw_node_features(dag, order)
            rows.append(raw)
            n_graphs += 1
            max_in = max(max_in, int(rec.get("max_indegree_unique", 0)))
            max_out = max(max_out, int(rec.get("max_outdegree_unique", 0)))

    if not rows:
        print("no meta_train graphs in manifest", file=sys.stderr)
        return 1
    if max_in > MAX_NEIGH or max_out > MAX_NEIGH:
        print(
            f"corpus degree {max_in}/{max_out} exceeds MAX_NEIGH={MAX_NEIGH}",
            file=sys.stderr,
        )
        return 1

    manifest_hash, split_hash = spec_source_hashes(ROOT / "spec")
    stats = fit_feature_stats(
        rows,
        n_graphs=n_graphs,
        max_indegree_unique=max_in,
        max_outdegree_unique=max_out,
        dataset_manifest_sha256=manifest_hash,
        split_policy_sha256=split_hash,
    )
    args.out.write_text(json.dumps(stats.to_json_dict(), indent=2) + "\n")
    print(
        f"wrote {args.out} n_graphs={n_graphs} n_nodes={stats.n_nodes} "
        f"held_out_role_rows_ignored={other_roles} "
        f"max_in={max_in} max_out={max_out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
