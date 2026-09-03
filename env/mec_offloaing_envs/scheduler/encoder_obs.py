"""Phase 2 encoder observations: canonical DAG adjacency + meta-train features.

Packed layout per decoder position:
  [FEATURE_DIM node features | MAX_NEIGH fw (successors) | MAX_NEIGH bw (predecessors) | mask]

Neighbor slots are decoder indices, padded with PAD_INDEX. No self-loop.
MAX_NEIGH = MAX_TASKS - 1 so a 20-node DAG can hold degree 19. Overflow raises.
Feature z-score uses frozen meta_train statistics only.
"""

from __future__ import annotations

import hashlib
import json
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

FEATURE_NAMES: tuple[str, ...] = (
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
STANDARDIZE_FEATURES: frozenset[str] = frozenset(
    name for name in FEATURE_NAMES if name not in ("is_root", "is_sink")
)
FEATURE_DIM = len(FEATURE_NAMES)
PACKED_DIM = FEATURE_DIM + 2 * MAX_NEIGH + 1

_DEFAULT_STATS_PATH = (
    Path(__file__).resolve().parents[3] / "spec" / "encoder_feature_stats.json"
)
_STATS_CACHE: FeatureStats | None = None


class EncoderGraphError(ValueError):
    """Invalid DAG, decoder order, or neighbor degree for encoder packing."""


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
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "split_policy_sha256": self.split_policy_sha256,
        }


def _standardize_indices() -> list[int]:
    return [i for i, name in enumerate(FEATURE_NAMES) if name in STANDARDIZE_FEATURES]


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def spec_source_hashes(root: Path | None = None) -> tuple[str, str]:
    base = Path(root) if root is not None else _DEFAULT_STATS_PATH.parent
    return sha256_file(base / "dataset_manifest.jsonl"), sha256_file(base / "split_policy.json")


def require_neighbor_capacity(count: int, label: str) -> None:
    if count > MAX_NEIGH:
        raise EncoderGraphError(
            f"{label} degree {count} exceeds MAX_NEIGH={MAX_NEIGH} "
            f"(MAX_TASKS={MAX_TASKS}); silent truncation forbidden"
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
    global _STATS_CACHE
    if _STATS_CACHE is None:
        _STATS_CACHE = load_feature_stats()
    return _STATS_CACHE


def reset_feature_stats_cache() -> None:
    global _STATS_CACHE
    _STATS_CACHE = None


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


def raw_node_features(dag: CanonicalDAG, decoder_order: Sequence[Any]) -> np.ndarray:
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
) -> np.ndarray:
    if stats is None:
        stats = default_feature_stats()
    raw = raw_node_features(dag, decoder_order)
    features = stats.standardize(raw)
    fw, bw, _, _ = neighbor_index_tables(dag, decoder_order)
    return pack_observation(features, fw, bw)


def encode_task_graph(
    task_graph: Any,
    decoder_order: Sequence[Any],
    stats: FeatureStats | None = None,
) -> np.ndarray:
    return encode_canonical_dag(to_canonical_dag(task_graph), decoder_order, stats=stats)


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
