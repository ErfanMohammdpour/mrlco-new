#!/usr/bin/env python3
"""Fit encoder z-score statistics on meta_train graphs only.

v1: structural features only → spec/encoder_feature_stats.json
v2: structural + resource log features over meta_train profiles
    → spec/encoder_feature_stats_v2.json (does not touch v1)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env.mec_offloaing_envs.offloading_task_graph import OffloadingTaskGraph  # noqa: E402
from env.mec_offloaing_envs.scheduler.adapter import to_canonical_dag  # noqa: E402
from env.mec_offloaing_envs.scheduler import encoder_obs as eo  # noqa: E402
from env.mec_offloaing_envs.scheduler.encoder_obs import (  # noqa: E402
    MAX_NEIGH,
    fit_feature_stats,
    raw_node_features,
    set_obs_version,
    spec_source_hashes,
)
from spec.resource_profiles import (  # noqa: E402
    resource_feature_vector,
    role_profile_ids,
)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def compute_meta_train_stats_v1(root: Path, manifest: Path):
    set_obs_version("v1")
    rows = []
    n_graphs = 0
    max_in = 0
    max_out = 0
    with manifest.open() as handle:
        for line in handle:
            rec = json.loads(line)
            if rec.get("role") != "meta_train":
                continue
            path = root / rec["relative_path"]
            dag = to_canonical_dag(OffloadingTaskGraph(str(path)))
            order = sorted(dag.tasks)
            rows.append(raw_node_features(dag, order))
            n_graphs += 1
            max_in = max(max_in, int(rec.get("max_indegree_unique", 0)))
            max_out = max(max_out, int(rec.get("max_outdegree_unique", 0)))
    if not rows:
        raise ValueError("no meta_train graphs in manifest")
    if max_in > MAX_NEIGH or max_out > MAX_NEIGH:
        raise ValueError("corpus degree %s/%s exceeds MAX_NEIGH=%s" % (max_in, max_out, MAX_NEIGH))
    manifest_hash, split_hash = spec_source_hashes(root / "spec")
    return fit_feature_stats(
        rows,
        n_graphs=n_graphs,
        max_indegree_unique=max_in,
        max_outdegree_unique=max_out,
        dataset_manifest_sha256=manifest_hash,
        split_policy_sha256=split_hash,
    )


def compute_meta_train_stats_v2(root: Path, manifest: Path):
    """Fit on meta_train graphs × meta_train resource profiles only."""
    set_obs_version("v2")
    if eo.OBS_VERSION != "v2" or eo.FEATURE_DIM != 15:
        raise RuntimeError(
            "obs v2 not active after set_obs_version (version=%s dim=%s)"
            % (eo.OBS_VERSION, eo.FEATURE_DIM)
        )
    profiles = list(role_profile_ids("meta_train"))
    heldout = list(role_profile_ids("validation_heldout")) + list(
        role_profile_ids("meta_test_heldout")
    )
    train_vecs = {pid: resource_feature_vector(pid) for pid in profiles}
    held_vecs = {pid: resource_feature_vector(pid) for pid in heldout}

    rows = []
    n_graphs = 0
    max_in = 0
    max_out = 0
    with manifest.open() as handle:
        for line in handle:
            rec = json.loads(line)
            if rec.get("role") != "meta_train":
                continue
            path = root / rec["relative_path"]
            dag = to_canonical_dag(OffloadingTaskGraph(str(path)))
            order = sorted(dag.tasks)
            for pid in profiles:
                rows.append(raw_node_features(dag, order, resource_vec=train_vecs[pid]))
            n_graphs += 1
            max_in = max(max_in, int(rec.get("max_indegree_unique", 0)))
            max_out = max(max_out, int(rec.get("max_outdegree_unique", 0)))
    if not rows:
        raise ValueError("no meta_train graphs in manifest")
    if max_in > MAX_NEIGH or max_out > MAX_NEIGH:
        raise ValueError("corpus degree %s/%s exceeds MAX_NEIGH=%s" % (max_in, max_out, MAX_NEIGH))
    manifest_hash, split_hash = spec_source_hashes(root / "spec")
    stats = fit_feature_stats(
        rows,
        n_graphs=n_graphs * len(profiles),
        max_indegree_unique=max_in,
        max_outdegree_unique=max_out,
        dataset_manifest_sha256=manifest_hash,
        split_policy_sha256=split_hash,
    )
    import numpy as np

    stacked = np.concatenate([np.asarray(r, dtype=np.float64) for r in rows], axis=0)
    res_min = stacked[:, 11:15].min(axis=0)
    res_max = stacked[:, 11:15].max(axis=0)
    held_ok = True
    held_report = {}
    for pid, vec in held_vecs.items():
        arr = np.asarray(vec, dtype=np.float64)
        inside = bool(np.all(arr >= res_min - 1e-9) and np.all(arr <= res_max + 1e-9))
        held_report[pid] = {
            "vec": [float(x) for x in arr],
            "inside_train_range": inside,
        }
        held_ok = held_ok and inside
    meta = {
        "obs_version": "v2",
        "feature_names": list(eo.FEATURE_NAMES),
        "feature_dim": int(eo.FEATURE_DIM),
        "n_unique_graphs": int(n_graphs),
        "n_train_profiles": int(len(profiles)),
        "train_profiles": profiles,
        "resource_log_min": [float(x) for x in res_min],
        "resource_log_max": [float(x) for x in res_max],
        "heldout_interpolation_ok": bool(held_ok),
        "heldout_resource_check": held_report,
    }
    return stats, meta


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "spec" / "dataset_manifest.jsonl",
    )
    parser.add_argument(
        "--obs-version",
        choices=("v1", "v2"),
        default="v1",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="default: encoder_feature_stats.json (v1) or encoder_feature_stats_v2.json (v2)",
    )
    args = parser.parse_args()
    v1_path = ROOT / "spec" / "encoder_feature_stats.json"
    v1_before = _sha256_bytes(v1_path.read_bytes()) if v1_path.is_file() else None

    if args.obs_version == "v1":
        out = args.out or v1_path
        stats = compute_meta_train_stats_v1(ROOT, args.manifest)
        meta = None
    else:
        out = args.out or (ROOT / "spec" / "encoder_feature_stats_v2.json")
        stats, meta = compute_meta_train_stats_v2(ROOT, args.manifest)

    out.write_text(json.dumps(stats.to_json_dict(), indent=2) + "\n")
    if meta is not None:
        meta_path = out.with_name(out.stem + "_meta.json")
        meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
        print(
            "v2_meta heldout_ok=%s resource_log_min=%s max=%s"
            % (meta["heldout_interpolation_ok"], meta["resource_log_min"], meta["resource_log_max"])
        )

    v1_after = _sha256_bytes(v1_path.read_bytes()) if v1_path.is_file() else None
    if v1_before is not None and v1_after != v1_before:
        raise SystemExit("REFUSING: encoder_feature_stats.json (v1) hash changed")
    print(
        "wrote %s n_graphs=%s n_nodes=%s dim=%d obs=%s v1_sha=%s"
        % (
            out,
            stats.n_graphs,
            stats.n_nodes,
            len(stats.mean),
            args.obs_version,
            v1_after,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
