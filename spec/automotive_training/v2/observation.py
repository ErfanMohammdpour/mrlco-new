#!/usr/bin/env python3
"""v2 observation extension contract (additive; the frozen v1 schema is untouched).

The v2 context is a 12-vector per graph (see `v2.env.V2_CONTEXT_FIELDS`). Two encodings are
possible and both are defined here so the choice is explicit:

* `pack_v2_row(v1_row, context)` - APPEND 12 columns to a packed v1 row. Pure function, no
  dependency on the frozen obs packer, used by tests and by any tool that needs the v2
  tensor today. The v1 prefix is preserved bit-for-bit.
* `write_v2_stats_file()` - generate `encoder_feature_stats_automotive_v2.json` (52
  entries: the 40 frozen automotive entries copied verbatim + 12 identity entries with
  mean 0.0 / std 1.0, because the v2 context columns are already bounded). This is the
  artifact the frozen packer's stats validation will need once
  `automotive_v2_obs_v1` is added to `encoder_obs.set_obs_version`.

NOT yet wired into the frozen packer: adding the version there changes FEATURE_DIM/
PACKED_DIM at import time, so it belongs to the next stage together with the trainer bridge.
Until then the policy input stays the v1 schema and `v2_context()` is telemetry only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np

from spec.automotive_training.v2.context_fields import V2_CONTEXT_FIELDS

V2_OBS_VERSION = "automotive_v2_obs_v1"
V1_OBS_VERSION = "automotive_mc_obs_v1"
V1_FEATURE_DIM = 40
V1_PACKED_DIM = 79
V2_CONTEXT_DIM = len(V2_CONTEXT_FIELDS)
V2_FEATURE_DIM = V1_FEATURE_DIM + V2_CONTEXT_DIM
V2_PACKED_DIM = V1_PACKED_DIM + V2_CONTEXT_DIM

_SPEC_DIR = Path(__file__).resolve().parents[2]
V1_STATS_PATH = _SPEC_DIR / "encoder_feature_stats_automotive_mc_v1.json"
V2_STATS_PATH = _SPEC_DIR / "encoder_feature_stats_automotive_v2.json"


class V2ObservationError(RuntimeError):
    """Raised on shape/contract violations of the v2 observation extension."""


def v2_feature_names(v1_feature_names: Sequence[str] | None = None) -> tuple:
    """The 52 feature names: the 40 frozen names followed by the 12 v2 context names."""
    if v1_feature_names is None:
        v1_feature_names = _v1_feature_names()
    if len(v1_feature_names) != V1_FEATURE_DIM:
        raise V2ObservationError("expected %d v1 feature names, got %d"
                                 % (V1_FEATURE_DIM, len(v1_feature_names)))
    names = tuple("%s_%s" % (V2_OBS_VERSION, f) for f in V2_CONTEXT_FIELDS)
    return tuple(v1_feature_names) + names


def _v1_feature_names() -> tuple:
    from env.mec_offloaing_envs.scheduler import encoder_obs

    previous = encoder_obs.OBS_VERSION
    try:
        encoder_obs.set_obs_version(V1_OBS_VERSION)
        return tuple(encoder_obs.FEATURE_NAMES)
    finally:
        encoder_obs.set_obs_version(previous)


#: packed layout is [FEATURE_DIM | fw | bw | mask], so the context columns are INSERTED
#: at the end of the feature block (not appended after the mask) - that is the only layout
#: the graph2seq `unpack` can consume when FEATURE_DIM = 52.
MAX_NEIGH = (V1_PACKED_DIM - V1_FEATURE_DIM - 1) // 2
assert V1_FEATURE_DIM + 2 * MAX_NEIGH + 1 == V1_PACKED_DIM
assert V2_FEATURE_DIM + 2 * MAX_NEIGH + 1 == V2_PACKED_DIM


def pack_v2_row(v1_row: np.ndarray, context: np.ndarray) -> np.ndarray:
    """Insert the 12 v2 context columns at the end of the feature block.

    `v1_row` is [N, 79] = [features(40) | fw(19) | bw(19) | mask(1)]; the result is
    [N, 91] = [features(40) | ctx(12) | fw(19) | bw(19) | mask(1)]. The v1 feature columns,
    the neighbor tables and the mask are preserved bit-for-bit.
    """
    row = np.asarray(v1_row, dtype=np.float32)
    ctx = np.asarray(context, dtype=np.float32).reshape(-1)
    if row.ndim != 2 or row.shape[-1] != V1_PACKED_DIM:
        raise V2ObservationError("v1 row must be [N, %d], got %s"
                                 % (V1_PACKED_DIM, (row.shape,)))
    if ctx.shape[0] != V2_CONTEXT_DIM:
        raise V2ObservationError("context must have %d entries, got %d"
                                 % (V2_CONTEXT_DIM, ctx.shape[0]))
    if not np.all(np.isfinite(ctx)):
        raise V2ObservationError("v2 context must be finite")
    features = row[:, :V1_FEATURE_DIM]
    tail = row[:, V1_FEATURE_DIM:]
    block = np.tile(ctx, (row.shape[0], 1))
    return np.concatenate([features, block, tail], axis=-1)


def split_v2_row(v2_row: np.ndarray) -> tuple:
    """Inverse of `pack_v2_row`: ([N, 40] features, [N, 12] context, [N, 39] tail)."""
    arr = np.asarray(v2_row, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[-1] != V2_PACKED_DIM:
        raise V2ObservationError("v2 row must be [N, %d], got %s"
                                 % (V2_PACKED_DIM, (arr.shape,)))
    return (arr[:, :V1_FEATURE_DIM], arr[:, V1_FEATURE_DIM:V2_FEATURE_DIM],
            arr[:, V2_FEATURE_DIM:])


def v1_form_of_v2_row(v2_row: np.ndarray) -> np.ndarray:
    """Recover the equivalent v1 packed row (drops the context block)."""
    features, _ctx, tail = split_v2_row(v2_row)
    return np.concatenate([features, tail], axis=-1)


def write_v2_stats_file(path: Path | None = None) -> Path:
    """Build the 52-entry stats artifact from the frozen 40-entry one."""
    target = Path(path or V2_STATS_PATH)
    doc = json.loads(V1_STATS_PATH.read_text())
    names = list(doc["feature_names"])
    if len(names) != V1_FEATURE_DIM:
        raise V2ObservationError("frozen stats file has %d names, expected %d"
                                 % (len(names), V1_FEATURE_DIM))
    new_names = ["%s_%s" % (V2_OBS_VERSION, f) for f in V2_CONTEXT_FIELDS]
    if names[0].startswith(V2_OBS_VERSION):
        raise V2ObservationError("frozen stats file already carries v2 names")
    doc["feature_names"] = names + new_names
    doc["mean"] = list(doc["mean"]) + [0.0] * V2_CONTEXT_DIM
    doc["std"] = list(doc["std"]) + [1.0] * V2_CONTEXT_DIM
    doc["obs_version"] = V2_OBS_VERSION
    doc["derived_from"] = V1_STATS_PATH.name
    doc["note"] = ("v2 context columns are bounded/identity-normalised, so mean 0 / std 1 "
                   "is the documented identity entry")
    target.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    return target


def validate_stats_file(path: Path | None = None) -> dict:
    doc = json.loads(Path(path or V2_STATS_PATH).read_text())
    if len(doc["feature_names"]) != V2_FEATURE_DIM:
        raise V2ObservationError("stats file must have %d feature names" % V2_FEATURE_DIM)
    if len(doc["mean"]) != V2_FEATURE_DIM or len(doc["std"]) != V2_FEATURE_DIM:
        raise V2ObservationError("stats file mean/std must have %d entries" % V2_FEATURE_DIM)
    return {"feature_dim": len(doc["feature_names"]), "obs_version": doc.get("obs_version"),
            "derived_from": doc.get("derived_from")}
