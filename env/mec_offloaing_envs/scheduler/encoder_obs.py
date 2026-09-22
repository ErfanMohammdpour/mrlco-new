"""Phase 2 encoder observations: canonical DAG adjacency + meta-train features.

Packed layout per decoder position:
  [FEATURE_DIM node features | MAX_NEIGH fw (successors) | MAX_NEIGH bw (predecessors) | mask]

Neighbor slots are decoder indices, padded with PAD_INDEX. No self-loop.
MAX_NEIGH = MAX_TASKS - 1 so a 20-node DAG can hold degree 19. Overflow raises.
Feature z-score uses frozen meta_train statistics only.

Obs version:
  v1 (default): FEATURE_DIM=11, PACKED_DIM=50 — Phases 1–3.
  v2: FEATURE_DIM=15, PACKED_DIM=54 — Phase 4 resource axis.
  v3: FEATURE_DIM=31, PACKED_DIM=70 — deadline + feasibility awareness (⑤a).
      Adds, per task: deadline presence/type one-hot, slack ratio against the
      optimistic bound, criticality class one-hot, tardiness weight, and per
      ACTION the static optimistic ready bound plus its feasibility flag.  Without
      these the policy is structurally blind to deadlines and the shield becomes
      the only source of feasibility information — see
      reports/PRE_PPO_ARCHITECTURE_REVIEW.md.
  Select with env MARGO_OBS_VERSION=v2|v3 before importing policies, or call
  set_obs_version.  v1/v2 stay byte-identical; only v3 needs a resource model.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .adapter import to_canonical_dag
from .model import CanonicalDAG

# Spec `model.task_count` = 20. Capacity is graph-theoretic, not corpus stats.
MAX_TASKS = 20
MAX_NEIGH = MAX_TASKS - 1
PAD_INDEX = -1
STD_EPS = 1e-12
ENCODER_DROPOUT = 0.0
GNN_LAYERS = 2
NEIGHBORHOOD = "predecessor_and_successor"
DIRECTION_COMBINE = "sum"
AGGREGATOR = "masked_mean"
SELF_NEIGHBOR_CONCAT = True

FEATURE_NAMES_V1: tuple[str, ...] = (
    "compute_workload_bytes",
    "task_output_bytes",
    "external_input_bytes",
    "incoming_edge_bytes",
    "outgoing_edge_bytes",
    "indegree",
    "outdegree",
    "decoder_index",
    "depth",
    "is_root",
    "is_sink",
)
RESOURCE_FEATURE_NAMES: tuple[str, ...] = (
    "log_ul_bps",
    "log_v2v_bps",
    "log_mec_cpu",
    "log_ue_cpu",
)
FEATURE_NAMES_V2: tuple[str, ...] = FEATURE_NAMES_V1 + RESOURCE_FEATURE_NAMES

# --- v3: deadline / criticality / feasibility awareness ---------------------
# All of these are bounded by construction (0/1, clipped ratios), so they are NOT
# z-scored: their frozen stats rows are identity. That also keeps the v3 stats
# file derivable from the v2 one without inventing corpus statistics.
DEADLINE_FEATURE_NAMES: tuple[str, ...] = (
    "has_deadline",
    "deadline_is_soft",
    "deadline_is_firm",
    "deadline_is_hard",
    "slack_ratio_min_lb",
    "criticality_low",
    "criticality_medium",
    "criticality_high",
    "tardiness_weight_scaled",
    "return_hop_over_min_lb",
    "lb_ue_log1p",
    "feasible_ue",
    "lb_mec_log1p",
    "feasible_mec",
    "lb_helper_log1p",
    "feasible_helper",
)
FEATURE_NAMES_V3: tuple[str, ...] = FEATURE_NAMES_V2 + DEADLINE_FEATURE_NAMES
NON_STANDARDIZED_FEATURES: tuple[str, ...] = (
    "is_root",
    "is_sink",
) + DEADLINE_FEATURE_NAMES

# Mutable active schema (default v1). Policies import these names at load time —
# set MARGO_OBS_VERSION before importing graph2seq / policies for v2 jobs.
FEATURE_NAMES: tuple[str, ...] = FEATURE_NAMES_V1
STANDARDIZE_FEATURES: frozenset[str] = frozenset(
    name for name in FEATURE_NAMES if name not in NON_STANDARDIZED_FEATURES
)
FEATURE_DIM = len(FEATURE_NAMES)
PACKED_DIM = FEATURE_DIM + 2 * MAX_NEIGH + 1
OBS_VERSION = "v1"

_SPEC_DIR = Path(__file__).resolve().parents[3] / "spec"
_DEFAULT_STATS_PATH = _SPEC_DIR / "encoder_feature_stats.json"
_DEFAULT_STATS_PATH_V2 = _SPEC_DIR / "encoder_feature_stats_v2.json"
_DEFAULT_STATS_PATH_V3 = _SPEC_DIR / "encoder_feature_stats_v3.json"
_STATS_CACHE = None  # type: ignore[var-annotated]
_STATS_CACHE_VERSION: str | None = None


class EncoderGraphError(ValueError):
    """Invalid DAG, decoder order, or neighbor degree for encoder packing."""


def set_obs_version(version: str) -> None:
    """Switch FEATURE_DIM / PACKED_DIM. Call before policy import for v2 train/eval."""
    global FEATURE_NAMES, STANDARDIZE_FEATURES, FEATURE_DIM, PACKED_DIM, OBS_VERSION
    global _STATS_CACHE, _STATS_CACHE_VERSION
    version = str(version).lower().strip()
    if version not in ("v1", "v2", "v3"):
        raise EncoderGraphError("obs version must be v1, v2 or v3, got %r" % version)
    if version == "v1":
        FEATURE_NAMES = FEATURE_NAMES_V1
    elif version == "v2":
        FEATURE_NAMES = FEATURE_NAMES_V2
    else:
        FEATURE_NAMES = FEATURE_NAMES_V3
    STANDARDIZE_FEATURES = frozenset(
        name for name in FEATURE_NAMES if name not in NON_STANDARDIZED_FEATURES
    )
    FEATURE_DIM = len(FEATURE_NAMES)
    PACKED_DIM = FEATURE_DIM + 2 * MAX_NEIGH + 1
    OBS_VERSION = version
    _STATS_CACHE = None
    _STATS_CACHE_VERSION = None


def _boot_obs_version_from_env() -> None:
    ver = os.environ.get("MARGO_OBS_VERSION", "v1").strip().lower() or "v1"
    if ver != "v1":
        set_obs_version(ver)


_boot_obs_version_from_env()


@dataclass(frozen=True)
class FeatureStats:
    feature_names: tuple[str, ...]
    mean: np.ndarray
    std: np.ndarray
    n_graphs: int
    n_nodes: int
    role: str = "meta_train"
    max_indegree_unique: int = 0
    max_outdegree_unique: int = 0
    dataset_manifest_sha256: str = ""
    split_policy_sha256: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "mean", np.asarray(self.mean, dtype=np.float64).reshape(-1))
        object.__setattr__(self, "std", np.asarray(self.std, dtype=np.float64).reshape(-1))
        if tuple(self.feature_names) != FEATURE_NAMES:
            raise EncoderGraphError(
                f"feature_names mismatch: {self.feature_names} != {FEATURE_NAMES}"
            )
        if self.mean.shape != (FEATURE_DIM,) or self.std.shape != (FEATURE_DIM,):
            raise EncoderGraphError("mean/std must have shape [FEATURE_DIM]")
        if self.role != "meta_train":
            raise EncoderGraphError("encoder stats role must be meta_train")
        if np.any(self.std[list(_standardize_indices())] < STD_EPS):
            raise EncoderGraphError("standardize std below STD_EPS")
        if not np.all(np.isfinite(self.mean)) or not np.all(np.isfinite(self.std)):
            raise EncoderGraphError("encoder stats mean/std must be finite")
        if self.n_graphs > 0 and (
            len(self.dataset_manifest_sha256) != 64 or len(self.split_policy_sha256) != 64
        ):
            raise EncoderGraphError("frozen encoder stats must pin manifest and split_policy hashes")

    def standardize(self, raw: np.ndarray) -> np.ndarray:
        out = np.asarray(raw, dtype=np.float32).copy()
        for i in _standardize_indices():
            out[..., i] = (out[..., i] - self.mean[i]) / self.std[i]
        return out

    @classmethod
    def identity(cls) -> FeatureStats:
        std = np.ones(FEATURE_DIM, dtype=np.float64)
        return cls(
            feature_names=FEATURE_NAMES,
            mean=np.zeros(FEATURE_DIM, dtype=np.float64),
            std=std,
            n_graphs=0,
            n_nodes=0,
            role="meta_train",
        )

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "spec_version": "MARGO-SPEC-v0.1",
            "split_version": "MARGO-SPLIT-v1",
            "role": self.role,
            "n_graphs": int(self.n_graphs),
            "n_nodes": int(self.n_nodes),
            "feature_names": list(self.feature_names),
            "standardize": [n for n in self.feature_names if n in STANDARDIZE_FEATURES],
            "mean": [float(x) for x in self.mean],
            "std": [float(x) for x in self.std],
            "max_indegree_unique": int(self.max_indegree_unique),
            "max_outdegree_unique": int(self.max_outdegree_unique),
            "max_tasks": MAX_TASKS,
            "max_neigh": MAX_NEIGH,
            "packed_dim": PACKED_DIM,
            "std_eps": STD_EPS,
            "hash_normalization": "canonical_lf",
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "split_policy_sha256": self.split_policy_sha256,
        }


def _standardize_indices() -> list[int]:
    return [i for i, name in enumerate(FEATURE_NAMES) if name in STANDARDIZE_FEATURES]


def sha256_canonical_text(path: str | Path) -> str:
    """SHA-256 after CRLF/CR → LF. Working-copy line endings must not change the pin."""
    data = Path(path).read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    return sha256_canonical_text(path)


def spec_source_hashes(root: Path | None = None) -> tuple[str, str]:
    base = Path(root) if root is not None else _DEFAULT_STATS_PATH.parent
    return sha256_canonical_text(base / "dataset_manifest.jsonl"), sha256_canonical_text(
        base / "split_policy.json"
    )


def require_neighbor_capacity(count: int, label: str) -> None:
    if count > MAX_NEIGH:
        raise EncoderGraphError(
            f"{label} degree {count} exceeds MAX_NEIGH={MAX_NEIGH} "
            f"(MAX_TASKS={MAX_TASKS}); silent truncation forbidden"
        )


def require_spec_task_count(n: int) -> None:
    if int(n) != MAX_TASKS:
        raise EncoderGraphError(
            f"encoder graph has {n} tasks; spec model.task_count / MAX_TASKS={MAX_TASKS}"
        )


def validate_neighbor_indices(fw: np.ndarray, bw: np.ndarray, n: int) -> None:
    for name, arr in (("fw", fw), ("bw", bw)):
        a = np.asarray(arr, dtype=np.int32)
        if a.ndim != 2 or a.shape[1] != MAX_NEIGH:
            raise EncoderGraphError(f"{name} must be [N, MAX_NEIGH]")
        if a.shape[0] != n:
            raise EncoderGraphError(f"{name} rows {a.shape[0]} != seq_len {n}")
        valid = a != PAD_INDEX
        if np.any(a[valid] < 0) or np.any(a[valid] >= n):
            raise EncoderGraphError(
                f"{name} neighbor index outside [0, {n}); silent wrap into TF forbidden"
            )


def load_feature_stats(path: str | Path | None = None) -> FeatureStats:
    stats_path = Path(path) if path is not None else _DEFAULT_STATS_PATH
    data = json.loads(stats_path.read_text())
    if data.get("role") != "meta_train":
        raise EncoderGraphError("encoder_feature_stats.json role must be meta_train")
    if tuple(data.get("feature_names", ())) != FEATURE_NAMES:
        raise EncoderGraphError("encoder stats feature_names/order mismatch")
    if int(data.get("max_tasks", -1)) != MAX_TASKS:
        raise EncoderGraphError("encoder stats max_tasks must equal MAX_TASKS")
    if int(data.get("max_neigh", -1)) != MAX_NEIGH:
        raise EncoderGraphError("encoder stats max_neigh must equal MAX_NEIGH")
    if data.get("hash_normalization") != "canonical_lf":
        raise EncoderGraphError("encoder stats hash_normalization must be canonical_lf")
    manifest_hash, split_hash = spec_source_hashes(stats_path.parent)
    if data.get("dataset_manifest_sha256") != manifest_hash:
        raise EncoderGraphError("encoder stats dataset_manifest_sha256 stale")
    if data.get("split_policy_sha256") != split_hash:
        raise EncoderGraphError("encoder stats split_policy_sha256 stale")
    return FeatureStats(
        feature_names=tuple(data["feature_names"]),
        mean=np.asarray(data["mean"], dtype=np.float64),
        std=np.asarray(data["std"], dtype=np.float64),
        n_graphs=int(data["n_graphs"]),
        n_nodes=int(data["n_nodes"]),
        role=str(data["role"]),
        max_indegree_unique=int(data.get("max_indegree_unique", 0)),
        max_outdegree_unique=int(data.get("max_outdegree_unique", 0)),
        dataset_manifest_sha256=str(data["dataset_manifest_sha256"]),
        split_policy_sha256=str(data["split_policy_sha256"]),
    )


def default_feature_stats() -> FeatureStats:
    global _STATS_CACHE, _STATS_CACHE_VERSION
    if _STATS_CACHE is None or _STATS_CACHE_VERSION != OBS_VERSION:
        if OBS_VERSION == "v3":
            path = _DEFAULT_STATS_PATH_V3
        elif OBS_VERSION == "v2":
            path = _DEFAULT_STATS_PATH_V2
        else:
            path = _DEFAULT_STATS_PATH
        _STATS_CACHE = load_feature_stats(path)
        _STATS_CACHE_VERSION = OBS_VERSION
    return _STATS_CACHE


def reset_feature_stats_cache() -> None:
    global _STATS_CACHE, _STATS_CACHE_VERSION
    _STATS_CACHE = None
    _STATS_CACHE_VERSION = None


def resource_log_vector_from_cluster(resource_cluster: Any) -> np.ndarray:
    """[log UL_bps, log V2V_bps, log MEC_CPU, log UE_CPU] from OffloadingEnvironment Resources."""
    mbps_to_bps = 1024.0 * 1024.0 / 8.0
    ul = float(resource_cluster.bandwidth_up) * mbps_to_bps
    v2v_bw = float(getattr(resource_cluster, "v2v_bandwidth", 5.0))
    v2v = v2v_bw * mbps_to_bps
    mec = float(resource_cluster.mec_process_capable)
    ue = float(resource_cluster.mobile_process_capable)
    for name, val in (("ul", ul), ("v2v", v2v), ("mec", mec), ("ue", ue)):
        if not (val > 0.0) or not math.isfinite(val):
            raise EncoderGraphError("bad resource rate %s=%s" % (name, val))
    return np.asarray(
        [math.log(ul), math.log(v2v), math.log(mec), math.log(ue)],
        dtype=np.float64,
    )


def resource_log_vector_from_config(resources: Any) -> np.ndarray:
    """[log UL_bps, log V2V_bps, log MEC_CPU, log UE_CPU] from a ResourceConfig.

    Mirrors `resource_log_vector_from_cluster` so obs v2/v3 can be built from the
    scheduler config directly (radio- and energy-model aware).
    """
    from .model import Location

    values = (
        float(resources.hop_rate("MEC_UL")),
        float(resources.hop_rate("V2V")),
        float(resources.cpu_rate(Location.MEC)),
        float(resources.cpu_rate(Location.UE)),
    )
    for name, val in zip(("ul", "v2v", "mec", "ue"), values):
        if not (val > 0.0) or not math.isfinite(val):
            raise EncoderGraphError("bad resource rate %s=%s" % (name, val))
    return np.asarray([math.log(v) for v in values], dtype=np.float64)


def resource_ctx_from_vec(resource_vec: Sequence[float], stats: FeatureStats | None = None) -> np.ndarray:
    """Z-scored resource 4-vector for readout-side conditioning. Shape [4]."""
    if stats is None:
        stats = default_feature_stats()
    raw = np.asarray(resource_vec, dtype=np.float64).reshape(4)
    if OBS_VERSION != "v2":
        raise EncoderGraphError("resource_ctx requires obs v2")
    idxs = [FEATURE_NAMES.index(n) for n in RESOURCE_FEATURE_NAMES]
    out = np.zeros(4, dtype=np.float32)
    for j, i in enumerate(idxs):
        out[j] = (raw[j] - stats.mean[i]) / stats.std[i]
    return out


def _decoder_ids(decoder_order: Sequence[Any]) -> list[int]:
    return [int(x) for x in list(decoder_order)]


def _task_depths(dag: CanonicalDAG) -> dict[int, int]:
    preds = dag.predecessors()
    memo: dict[int, int] = {}

    def depth(tid: int) -> int:
        if tid in memo:
            return memo[tid]
        incoming = preds[tid]
        if not incoming:
            memo[tid] = 0
            return 0
        memo[tid] = 1 + max(depth(edge.src_task_id) for edge in incoming)
        return memo[tid]

    return {tid: depth(tid) for tid in dag.tasks}


def feasibility_channel_indices() -> tuple[int, int, int]:
    """Indices of (feasible_ue, feasible_mec, feasible_helper) in the ACTIVE schema.

    Used by the policy to build the static mask directly from the observation, so
    that rollout and update construct the identical mask with no extra plumbing.
    Raises for v1/v2, which have no feasibility channels.
    """
    if OBS_VERSION != "v3":
        raise EncoderGraphError(
            "feasibility channels exist only in obs v3 (active %r)" % OBS_VERSION
        )
    return tuple(
        FEATURE_NAMES.index(name)
        for name in ("feasible_ue", "feasible_mec", "feasible_helper")
    )


def _deadline_block(
    dag: CanonicalDAG,
    order: Sequence[int],
    bounds: Any,
    resources: Any,
    cycles_per_bit: float | None,
) -> np.ndarray:
    """[N, len(DEADLINE_FEATURE_NAMES)] bounded deadline/feasibility features."""
    from .model import Location
    from .static_bounds import ACTION_LOCATIONS, deadline_vector

    n = len(order)
    out = np.zeros((n, len(DEADLINE_FEATURE_NAMES)), dtype=np.float64)
    idx = {name: i for i, name in enumerate(DEADLINE_FEATURE_NAMES)}
    dl = deadline_vector(dag, order)
    deadlines = [d for d, _t in dl]
    slack = bounds.min_slack_ratio(deadlines)
    feas = bounds.feasible_by_deadline(deadlines)
    scale = max(float(bounds.max_ready_lb), 1e-9)

    for pos, tid in enumerate(order):
        task = dag.tasks[int(tid)]
        deadline, dtype = dl[pos]
        row = out[pos]
        if deadline is None or dtype == "none":
            row[idx["has_deadline"]] = 0.0
        else:
            row[idx["has_deadline"]] = 1.0
            if dtype in ("soft", "firm", "hard"):
                row[idx["deadline_is_%s" % dtype]] = 1.0
        row[idx["slack_ratio_min_lb"]] = slack[pos]
        cls = str(task.criticality_class)
        if cls in ("low", "medium", "high"):
            row[idx["criticality_%s" % cls]] = 1.0
        # log1p keeps the coefficient bounded for large weights
        row[idx["tardiness_weight_scaled"]] = float(
            min(1.0, math.log1p(max(0.0, float(task.tardiness_weight))) / math.log1p(10.0))
        )
        # unavoidable return cost for a sink, relative to the best achievable ready time
        out_bytes = int(task.task_output_bytes)
        if bounds.is_sink[pos]:
            cheapest_return = min(
                _transfer_lb(out_bytes, loc, Location.UE, resources)
                for loc in ACTION_LOCATIONS
            )
            row[idx["return_hop_over_min_lb"]] = float(
                min(4.0, cheapest_return / max(bounds.min_ready_lb[pos], 1e-9))
            )
        for action, name in ((0, "ue"), (1, "mec"), (2, "helper")):
            lb = float(bounds.ready_lb[pos][action])
            row[idx["lb_%s_log1p" % name]] = float(
                min(4.0, math.log1p(max(0.0, lb) / scale * 10.0))
            )
            row[idx["feasible_%s" % name]] = 1.0 if feas[pos][action] else 0.0
    return out


def _transfer_lb(nbytes: int, src: Any, dst: Any, resources: Any) -> float:
    from .feasibility import transfer_lower_bound

    return transfer_lower_bound(int(nbytes), src, dst, resources)


def raw_node_features(
    dag: CanonicalDAG,
    decoder_order: Sequence[Any],
    resource_vec: Sequence[float] | None = None,
    resources: Any = None,
    cycles_per_bit: float | None = None,
) -> np.ndarray:
    order = _decoder_ids(decoder_order)
    if len(order) != len(set(order)):
        raise EncoderGraphError("decoder_order has duplicate task ids")
    if set(order) != set(dag.tasks):
        raise EncoderGraphError("decoder_order must be a permutation of DAG task ids")

    preds = dag.predecessors()
    succs = dag.successors()
    depths = _task_depths(dag)
    rows = np.zeros((len(order), FEATURE_DIM), dtype=np.float64)
    name_index = {name: i for i, name in enumerate(FEATURE_NAMES)}

    for pos, tid in enumerate(order):
        task = dag.tasks[tid]
        incoming = preds[tid]
        outgoing_ids = succs[tid]
        incoming_bytes = int(sum(edge.edge_output_bytes for edge in incoming))
        outgoing_bytes = int(
            sum(
                edge.edge_output_bytes
                for edge in dag.edges
                if edge.src_task_id == tid
            )
        )
        indeg = len(incoming)
        outdeg = len(outgoing_ids)
        require_neighbor_capacity(indeg, f"task {tid} in")
        require_neighbor_capacity(outdeg, f"task {tid} out")
        row = rows[pos]
        row[name_index["compute_workload_bytes"]] = task.compute_workload_bytes
        row[name_index["task_output_bytes"]] = task.task_output_bytes
        row[name_index["external_input_bytes"]] = task.external_input_bytes
        row[name_index["incoming_edge_bytes"]] = incoming_bytes
        row[name_index["outgoing_edge_bytes"]] = outgoing_bytes
        row[name_index["indegree"]] = indeg
        row[name_index["outdegree"]] = outdeg
        row[name_index["decoder_index"]] = pos
        row[name_index["depth"]] = depths[tid]
        row[name_index["is_root"]] = 1.0 if indeg == 0 else 0.0
        row[name_index["is_sink"]] = 1.0 if outdeg == 0 else 0.0

    if OBS_VERSION in ("v2", "v3"):
        if resource_vec is None:
            raise EncoderGraphError("obs %s requires resource_vec [4]" % OBS_VERSION)
        rv = np.asarray(resource_vec, dtype=np.float64).reshape(4)
        if not np.all(np.isfinite(rv)):
            raise EncoderGraphError("resource_vec must be finite")
        for j, name in enumerate(RESOURCE_FEATURE_NAMES):
            rows[:, name_index[name]] = rv[j]
    elif resource_vec is not None:
        raise EncoderGraphError("resource_vec only valid for obs v2/v3")

    if OBS_VERSION == "v3":
        if resources is None:
            raise EncoderGraphError(
                "obs v3 requires the scheduler ResourceConfig (static bounds for "
                "deadline/feasibility features)"
            )
        from .static_bounds import static_action_bounds

        bounds = static_action_bounds(
            dag, order, resources, cycles_per_bit=cycles_per_bit
        )
        rows[:, name_index[DEADLINE_FEATURE_NAMES[0]]:] = _deadline_block(
            dag, order, bounds, resources, cycles_per_bit
        )
    return rows


def neighbor_index_tables(
    dag: CanonicalDAG, decoder_order: Sequence[Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    order = _decoder_ids(decoder_order)
    tid_to_pos = {tid: pos for pos, tid in enumerate(order)}
    preds = dag.predecessors()
    succs = dag.successors()
    n = len(order)
    fw = np.full((n, MAX_NEIGH), PAD_INDEX, dtype=np.int32)
    bw = np.full((n, MAX_NEIGH), PAD_INDEX, dtype=np.int32)
    fw_len = np.zeros((n,), dtype=np.int32)
    bw_len = np.zeros((n,), dtype=np.int32)

    for pos, tid in enumerate(order):
        fw_ids = sorted(tid_to_pos[sid] for sid in succs[tid])
        bw_ids = sorted(tid_to_pos[edge.src_task_id] for edge in preds[tid])
        require_neighbor_capacity(len(fw_ids), f"decoder pos {pos} fw")
        require_neighbor_capacity(len(bw_ids), f"decoder pos {pos} bw")
        fw_len[pos] = len(fw_ids)
        bw_len[pos] = len(bw_ids)
        if fw_ids:
            fw[pos, : len(fw_ids)] = fw_ids
        if bw_ids:
            bw[pos, : len(bw_ids)] = bw_ids
    validate_neighbor_indices(fw, bw, n)
    return fw, bw, fw_len, bw_len


def pack_observation(
    features: np.ndarray,
    fw: np.ndarray,
    bw: np.ndarray,
    node_mask: np.ndarray | None = None,
) -> np.ndarray:
    n = features.shape[0]
    if node_mask is None:
        mask = np.ones((n, 1), dtype=np.float32)
    else:
        mask = np.asarray(node_mask, dtype=np.float32).reshape(n, 1)
    fw = np.asarray(fw, dtype=np.int32)
    bw = np.asarray(bw, dtype=np.int32)
    validate_neighbor_indices(fw, bw, n)
    packed = np.concatenate(
        [
            np.asarray(features, dtype=np.float32),
            np.asarray(fw, dtype=np.float32),
            np.asarray(bw, dtype=np.float32),
            mask,
        ],
        axis=1,
    )
    if packed.shape[1] != PACKED_DIM:
        raise EncoderGraphError(f"packed dim {packed.shape[1]} != {PACKED_DIM}")
    return packed


def unpack_observation(packed: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    arr = np.asarray(packed)
    if arr.shape[-1] != PACKED_DIM:
        raise EncoderGraphError(f"packed last dim {arr.shape[-1]} != {PACKED_DIM}")
    features = arr[..., :FEATURE_DIM]
    fw = arr[..., FEATURE_DIM : FEATURE_DIM + MAX_NEIGH]
    bw = arr[..., FEATURE_DIM + MAX_NEIGH : FEATURE_DIM + 2 * MAX_NEIGH]
    mask = arr[..., -1]
    return features, fw, bw, mask


def global_neighbor_indices(local_adj: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map decoder-local neighbor indices to batched global indices.

    local_adj: [batch, seq_len, MAX_NEIGH], PAD_INDEX padded.
    Returns (global_adj [batch*seq_len, MAX_NEIGH], lengths [batch*seq_len]).
    Pad slots become dummy index = batch*seq_len (zero embedding row).
    """
    adj = np.asarray(local_adj, dtype=np.int32)
    if adj.ndim != 3 or adj.shape[2] != MAX_NEIGH:
        raise EncoderGraphError("local_adj must be [batch, seq_len, MAX_NEIGH]")
    batch, seq_len, _ = adj.shape
    dummy = batch * seq_len
    valid = adj >= 0
    offsets = (np.arange(batch, dtype=np.int32) * seq_len).reshape(batch, 1, 1)
    global_adj = np.where(valid, adj + offsets, dummy).astype(np.int32)
    lengths = valid.sum(axis=2).reshape(-1).astype(np.int32)
    return global_adj.reshape(dummy, MAX_NEIGH), lengths


def reachability_mask(dag: CanonicalDAG, decoder_order: Sequence[Any] | None = None) -> np.ndarray:
    """Directed reachability in decoder-index space: ancestor ∪ descendant ∪ self.

    Returns float32 [n, n] with 1.0 on allowed transformer pairs. No padding.
    """
    if decoder_order is None:
        decoder_order = sorted(dag.tasks)
    order = _decoder_ids(decoder_order)
    n = len(order)
    if set(order) != set(dag.tasks):
        raise EncoderGraphError("decoder_order must be a permutation of DAG task ids")
    tid_to_pos = {tid: i for i, tid in enumerate(order)}
    fwd = np.zeros((n, n), dtype=np.float32)
    succs = dag.successors()
    for tid in order:
        i = tid_to_pos[tid]
        stack = [tid_to_pos[sid] for sid in succs[tid]]
        seen = set(stack)
        while stack:
            j = stack.pop()
            fwd[i, j] = 1.0
            src_tid = order[j]
            for sid in succs[src_tid]:
                k = tid_to_pos[sid]
                if k not in seen:
                    seen.add(k)
                    stack.append(k)
    eye = np.eye(n, dtype=np.float32)
    return np.maximum(np.maximum(fwd, fwd.T), eye)


def packed_edge_set(packed: np.ndarray, decoder_order: Sequence[Any]) -> set[tuple[int, int]]:
    order = _decoder_ids(decoder_order)
    _, fw, _, mask = unpack_observation(packed)
    edges: set[tuple[int, int]] = set()
    for i, src in enumerate(order):
        if mask[i] < 0.5:
            continue
        for slot in fw[i]:
            j = int(slot)
            if j == PAD_INDEX:
                continue
            if j < 0 or j >= len(order):
                raise EncoderGraphError(f"neighbor index {j} out of decoder range")
            edges.add((src, order[j]))
    return edges


def encode_canonical_dag(
    dag: CanonicalDAG,
    decoder_order: Sequence[Any],
    stats: FeatureStats | None = None,
    enforce_task_count: bool = False,
    resource_vec: Sequence[float] | None = None,
    resource_cluster: Any = None,
    resources: Any = None,
    cycles_per_bit: float | None = None,
) -> np.ndarray:
    order = _decoder_ids(decoder_order)
    if enforce_task_count:
        require_spec_task_count(len(order))
    if stats is None:
        stats = default_feature_stats()
    if resource_vec is None and resource_cluster is not None:
        resource_vec = resource_log_vector_from_cluster(resource_cluster)
    if resource_vec is None and resources is not None:
        resource_vec = resource_log_vector_from_config(resources)
    raw = raw_node_features(
        dag,
        order,
        resource_vec=resource_vec,
        resources=resources,
        cycles_per_bit=cycles_per_bit,
    )
    features = stats.standardize(raw)
    fw, bw, _, _ = neighbor_index_tables(dag, order)
    return pack_observation(features, fw, bw)


def encode_task_graph(
    task_graph: Any,
    decoder_order: Sequence[Any],
    stats: FeatureStats | None = None,
    resource_vec: Sequence[float] | None = None,
    resource_cluster: Any = None,
    resources: Any = None,
    cycles_per_bit: float | None = None,
) -> np.ndarray:
    dag = to_canonical_dag(task_graph)
    if resources is None and resource_cluster is not None:
        from .adapter import resource_config_from_cluster

        resources = resource_config_from_cluster(resource_cluster)
    return encode_canonical_dag(
        dag,
        decoder_order,
        stats=stats,
        enforce_task_count=True,
        resource_vec=resource_vec,
        resource_cluster=resource_cluster,
        resources=resources,
        cycles_per_bit=cycles_per_bit,
    )


def fit_feature_stats(
    raw_rows: Iterable[np.ndarray],
    n_graphs: int,
    max_indegree_unique: int,
    max_outdegree_unique: int,
    dataset_manifest_sha256: str = "",
    split_policy_sha256: str = "",
) -> FeatureStats:
    stacked = np.concatenate([np.asarray(r, dtype=np.float64) for r in raw_rows], axis=0)
    if stacked.ndim != 2 or stacked.shape[1] != FEATURE_DIM:
        raise EncoderGraphError("raw feature rows must be [N, FEATURE_DIM]")
    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0)
    std = np.maximum(std, STD_EPS)
    if n_graphs > 0 and not dataset_manifest_sha256:
        dataset_manifest_sha256, split_policy_sha256 = spec_source_hashes()
    return FeatureStats(
        feature_names=FEATURE_NAMES,
        mean=mean,
        std=std,
        n_graphs=int(n_graphs),
        n_nodes=int(stacked.shape[0]),
        role="meta_train",
        max_indegree_unique=int(max_indegree_unique),
        max_outdegree_unique=int(max_outdegree_unique),
        dataset_manifest_sha256=dataset_manifest_sha256,
        split_policy_sha256=split_policy_sha256,
    )
