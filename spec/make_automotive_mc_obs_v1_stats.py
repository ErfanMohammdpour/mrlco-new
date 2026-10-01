#!/usr/bin/env python3
"""Derive `encoder_feature_stats_automotive_mc_v1.json` from the frozen v3 stats.

`automotive_mc_obs_v1` is an append-only extension of v3 for the frozen
MARGO-AUTOMOTIVE-MC-v1 dataset. The 9 appended mixed-criticality features
(C_LO/C_HI budgets, C_HI presence, C_HI/C_LO ratio, drop/degrade permission,
mode, slack ratio, deadline/D_G) are bounded by construction (0/1 flags and
clipped ratios), so they are NOT z-scored: their stats rows are identity
(mean 0, std 1). The 31 v3 rows are copied verbatim, so the standardized
v1/v2/v3 columns stay byte-identical and no corpus refit is needed.

Usage:
    python spec/make_automotive_mc_obs_v1_stats.py            # writes the json
    python spec/make_automotive_mc_obs_v1_stats.py --check    # staleness check
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
V3 = SPEC / "encoder_feature_stats_v3.json"
OUT = SPEC / "encoder_feature_stats_automotive_mc_v1.json"
OBS_VERSION = "automotive_mc_obs_v1"


def build() -> dict:
    from env.mec_offloaing_envs.scheduler import encoder_obs as eo

    v3 = json.loads(V3.read_text())
    v3_names = list(v3["feature_names"])
    names = list(eo.FEATURE_NAMES_AUTOMOTIVE_MC_V1)
    mc_names = list(eo.MC_FEATURE_NAMES)
    if v3_names + mc_names != names:
        raise SystemExit(
            "automotive_mc_obs_v1 schema is not v3 + MC_FEATURE_NAMES; "
            "refusing to guess"
        )

    mean = list(v3["mean"]) + [0.0] * len(mc_names)
    std = list(v3["std"]) + [1.0] * len(mc_names)
    standardize = [n for n in names if n not in eo.NON_STANDARDIZED_FEATURES]
    return {
        "spec_version": v3.get("spec_version", "MARGO-SPEC-v0.1"),
        "split_version": v3.get("split_version", "MARGO-SPLIT-v1"),
        "obs_version": OBS_VERSION,
        "role": v3["role"],
        "n_graphs": int(v3["n_graphs"]),
        "n_nodes": int(v3["n_nodes"]),
        "feature_names": names,
        "standardize": standardize,
        "non_standardized": list(eo.NON_STANDARDIZED_FEATURES),
        "mean": mean,
        "std": std,
        "max_indegree_unique": int(v3.get("max_indegree_unique", 0)),
        "max_outdegree_unique": int(v3.get("max_outdegree_unique", 0)),
        "max_tasks": int(eo.MAX_TASKS),
        "max_neigh": int(eo.MAX_NEIGH),
        "packed_dim": int(len(names) + 2 * eo.MAX_NEIGH + 1),
        "std_eps": float(eo.STD_EPS),
        "hash_normalization": "canonical_lf",
        "dataset_manifest_sha256": v3["dataset_manifest_sha256"],
        "split_policy_sha256": v3["split_policy_sha256"],
        "derivation": (
            "automotive_mc_obs_v1 = frozen v3 statistics + identity rows for the "
            "9 bounded mixed-criticality features (derived, not refit)"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    doc = build()
    text = json.dumps(doc, indent=2) + "\n"
    if args.check:
        if not OUT.exists() or OUT.read_text() != text:
            print(
                "encoder_feature_stats_automotive_mc_v1.json is STALE; "
                "rerun without --check"
            )
            return 1
        print("encoder_feature_stats_automotive_mc_v1.json up to date")
        return 0
    OUT.write_text(text)
    print("wrote %s (dim=%d)" % (OUT, len(doc["feature_names"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
