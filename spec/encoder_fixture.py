"""Fixed packed-obs fixture for meanagg regression. No TensorFlow."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from env.mec_offloaing_envs.scheduler.encoder_obs import (
    MAX_TASKS,
    FeatureStats,
    encode_canonical_dag,
    reachability_mask,
)
from env.mec_offloaing_envs.scheduler.model import CanonicalDAG, CanonicalTask

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"
PACKED_PATH = FIXTURE_DIR / "meanagg_packed.npy"
OUTPUT_PATH = FIXTURE_DIR / "meanagg_outputs.npy"
REACH_PATH = FIXTURE_DIR / "meanagg_reach.npy"
SEED = 0
HIDDEN = 128


def _tasks(n: int) -> list[CanonicalTask]:
    out = []
    for tid in range(n):
        out.append(
            CanonicalTask(
                task_id=tid,
                compute_workload_bytes=10 + 3 * tid,
                task_output_bytes=4 + tid,
                external_input_bytes=3 if tid == 0 else 0,
            )
        )
    return out


def chain_dag(n: int = MAX_TASKS) -> CanonicalDAG:
    edges = [(i, i + 1, 7 + i) for i in range(n - 1)]
    return CanonicalDAG.from_records(_tasks(n), edges)


def fork_join_dag() -> CanonicalDAG:
    """0 → {1,2} → 3, remaining nodes a tail chain 4…19."""
    n = MAX_TASKS
    edges = [(0, 1, 5), (0, 2, 6), (1, 3, 7), (2, 3, 8)]
    for i in range(3, n - 1):
        edges.append((i, i + 1, 9))
    return CanonicalDAG.from_records(_tasks(n), edges)


def build_packed_batch() -> tuple[np.ndarray, np.ndarray]:
    stats = FeatureStats.identity()
    dags = [chain_dag(), fork_join_dag()]
    packed = []
    reach = []
    order = list(range(MAX_TASKS))
    for dag in dags:
        packed.append(encode_canonical_dag(dag, order, stats=stats))
        reach.append(reachability_mask(dag, order))
    return np.stack(packed, axis=0).astype(np.float32), np.stack(reach, axis=0).astype(np.float32)


def save_packed_fixture(path: Path | None = None) -> Path:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    packed, reach = build_packed_batch()
    out = Path(path) if path is not None else PACKED_PATH
    np.save(str(out), packed)
    np.save(str(REACH_PATH), reach)
    return out
