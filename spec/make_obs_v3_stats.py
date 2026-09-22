#!/usr/bin/env python3
"""Derive `encoder_feature_stats_v3.json` from the frozen v2 statistics.

The 16 v3 deadline/feasibility features are bounded by construction (0/1 flags
and clipped ratios), so they are NOT z-scored: their stats rows are identity
(mean 0, std 1).  That keeps v3 internally consistent without inventing corpus
statistics, and the v1/v2 standardized columns are copied verbatim.

Usage:
    python spec/make_obs_v3_stats.py            # writes spec/encoder_feature_stats_v3.json
    python spec/make_obs_v3_stats.py --check    # verify the file is up to date
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SPEC = ROOT / "spec"
V2 = SPEC / "encoder_feature_stats_v2.json"
V3 = SPEC / "encoder_feature_stats_v3.json"


def build() -> dict:
    from env.mec_offloaing_envs.scheduler import encoder_obs as eo

    v2 = json.loads(V2.read_text())
    v2_names = list(v2["feature_names"])
    v3_names = list(eo.FEATURE_NAMES_V3)
    if v2_names + list(eo.DEADLINE_FEATURE_NAMES) != v3_names:
        raise SystemExit("v3 schema is not v2 + DEADLINE_FEATURE_NAMES; refusing to guess")

    mean = list(v2["mean"]) + [0.0] * len(eo.DEADLINE_FEATURE_NAMES)
    std = list(v2["std"]) + [1.0] * len(eo.DEADLINE_FEATURE_NAMES)
    standardize = [n for n in v3_names if n not in eo.NON_STANDARDIZED_FEATURES]
    return {
        "spec_version": v2.get("spec_version", "MARGO-SPEC-v0.1"),
        "split_version": v2.get("split_version", "MARGO-SPLIT-v1"),
        "obs_version": "v3",
        "role": v2["role"],
        "n_graphs": int(v2["n_graphs"]),
        "n_nodes": int(v2["n_nodes"]),
        "feature_names": v3_names,
        "standardize": standardize,
        "non_standardized": list(eo.NON_STANDARDIZED_FEATURES),
        "mean": mean,
        "std": std,
        "max_indegree_unique": int(v2.get("max_indegree_unique", 0)),
        "max_outdegree_unique": int(v2.get("max_outdegree_unique", 0)),
        "max_tasks": int(eo.MAX_TASKS),
        "max_neigh": int(eo.MAX_NEIGH),
        "packed_dim": int(eo.FEATURE_DIM and len(v3_names) + 2 * eo.MAX_NEIGH + 1),
        "std_eps": float(eo.STD_EPS),
        "hash_normalization": "canonical_lf",
        "dataset_manifest_sha256": v2["dataset_manifest_sha256"],
        "split_policy_sha256": v2["split_policy_sha256"],
        "derivation": (
            "v3 = frozen v2 statistics + identity rows for the 16 bounded "
            "deadline/feasibility features (derived, not refit)"
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    doc = build()
    text = json.dumps(doc, indent=2) + "\n"
    if args.check:
        if not V3.exists() or V3.read_text() != text:
            print("encoder_feature_stats_v3.json is STALE; rerun without --check")
            return 1
        print("encoder_feature_stats_v3.json up to date")
        return 0
    V3.write_text(text)
    print("wrote %s (dim=%d)" % (V3, len(doc["feature_names"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
