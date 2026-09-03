#!/usr/bin/env python3
"""Phase 2 encoder contract gate.

Exit 0 when encoder tests + wiring checks PASS.
Does NOT close Phase 2. Does NOT create a tag. Does NOT train.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

SPEC = Path(__file__).parent.resolve()
ROOT = SPEC.parent.resolve()
STATUS_MD = SPEC / "PHASE2_STATUS.md"
STATS = SPEC / "encoder_feature_stats.json"
ENCODER_OBS_PY = ROOT / "env" / "mec_offloaing_envs" / "scheduler" / "encoder_obs.py"
ENCODER_PY = ROOT / "policies" / "graph2seq_encoder.py"
SILENT_SLICE = re.compile(
    r"(\[:MAX_NEIGH\]|\[0:MAX_NEIGH\]|\[:6\]|\[0:6\]|fw_ids\[:MAX_NEIGH\]|bw_ids\[:MAX_NEIGH\])"
)
AGG_PY = ROOT / "policies" / "graph2seq_modules" / "aggregators.py"
TASK_GRAPH_PY = ROOT / "env" / "mec_offloaing_envs" / "offloading_task_graph.py"
TRAINER_PY = ROOT / "meta_trainer.py"
EVAL_PY = ROOT / "meta_evaluator.py"


def block(reasons: list[str], msg: str) -> None:
    reasons.append(msg)


def run_capture(cmd: list[str]) -> subprocess.CompletedProcess:
    print("+", " ".join(cmd))
    return subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True)


def check_status_doc(reasons: list[str]) -> None:
    text = STATUS_MD.read_text()
    if not re.search(r"^Status:\s*IN PROGRESS\b", text, re.M):
        block(reasons, "PHASE2_STATUS.md must say Status: IN PROGRESS until Phase 2 closes")
    if "phase1-freeze-v0.1" not in text:
        block(reasons, "PHASE2_STATUS.md must name phase1-freeze-v0.1")
    if "Do not move or rewrite" not in text:
        block(reasons, "PHASE2_STATUS.md must forbid rewriting the Phase 1 tag")


def check_no_clique(reasons: list[str]) -> None:
    text = ENCODER_PY.read_text()
    if "tf.tile(tf.expand_dims(seq_indices" in text:
        block(reasons, "graph2seq_encoder.py still builds a clique adjacency")
    if "sample_size_per_layer = seq_len" in text:
        block(reasons, "encoder still samples the full sequence as neighbors")
    if "packed_to_graph" not in text:
        block(reasons, "encoder must unpack packed DAG adjacency")


def check_masked_mean(reasons: list[str]) -> None:
    text = AGG_PY.read_text()
    if "tf.sequence_mask(neigh_len" not in text:
        block(reasons, "MeanAggregator must mask padded neighbors")
    mean_block = text.split("class MeanAggregator", 1)[-1].split("class MaxPoolingAggregator", 1)[0]
    if "tf.reduce_mean(neigh_vecs, axis=1)" in mean_block:
        block(reasons, "MeanAggregator still reduce_mean-s over padded neighbors")


def check_neighbor_capacity(reasons: list[str]) -> None:
    text = ENCODER_OBS_PY.read_text()
    if "MAX_NEIGH = MAX_TASKS - 1" not in text:
        block(reasons, "MAX_NEIGH must be MAX_TASKS - 1, not a corpus-fitted cap")
    if "def require_neighbor_capacity" not in text:
        block(reasons, "encoder_obs.py must fail-fast on degree overflow")
    for path in (ENCODER_OBS_PY, ENCODER_PY, TASK_GRAPH_PY):
        src = path.read_text()
        hit = SILENT_SLICE.search(src)
        if hit:
            block(reasons, f"silent neighbor truncation in {path.relative_to(ROOT)}: {hit.group(0)}")


def check_encode_path(reasons: list[str]) -> None:
    text = TASK_GRAPH_PY.read_text()
    if "encode_task_graph" not in text:
        block(reasons, "encode_point_sequence_with_ranking_and_cost must call encode_task_graph")
    trainer = TRAINER_PY.read_text()
    evaluator = EVAL_PY.read_text()
    if "obs_dim=20" in trainer:
        block(reasons, "meta_trainer.py still hardcodes obs_dim=20")
    if "obs_dim=env.input_dim" not in trainer:
        block(reasons, "meta_trainer.py must use env.input_dim")
    if "obs_dim=20" in evaluator:
        block(reasons, "meta_evaluator.py still hardcodes obs_dim=20")
    if "obs_dim=env.input_dim" not in evaluator:
        block(reasons, "meta_evaluator.py must use env.input_dim")


def check_stats(reasons: list[str]) -> None:
    if not STATS.is_file():
        block(reasons, "spec/encoder_feature_stats.json missing")
        return
    data = json.loads(STATS.read_text())
    if data.get("role") != "meta_train":
        block(reasons, "encoder stats role must be meta_train")
    n_train = 0
    with (SPEC / "dataset_manifest.jsonl").open() as handle:
        for line in handle:
            rec = json.loads(line)
            if rec.get("role") == "meta_train":
                n_train += 1
    if int(data.get("n_graphs", -1)) != n_train:
        block(reasons, "encoder stats n_graphs must equal meta_train manifest count")
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from env.mec_offloaing_envs.scheduler.encoder_obs import (  # noqa: E402
        FEATURE_NAMES,
        MAX_NEIGH,
        MAX_TASKS,
        spec_source_hashes,
    )

    if tuple(data.get("feature_names", ())) != FEATURE_NAMES:
        block(reasons, "encoder stats feature_names/order mismatch")
    if int(data.get("max_tasks", -1)) != MAX_TASKS:
        block(reasons, "encoder stats max_tasks must equal MAX_TASKS")
    if int(data.get("max_neigh", -1)) != MAX_NEIGH:
        block(reasons, "encoder stats max_neigh must equal MAX_NEIGH")
    manifest_hash, split_hash = spec_source_hashes(SPEC)
    if data.get("dataset_manifest_sha256") != manifest_hash:
        block(reasons, "encoder stats dataset_manifest_sha256 stale")
    if data.get("split_policy_sha256") != split_hash:
        block(reasons, "encoder stats split_policy_sha256 stale")


def check_encoder_tests(reasons: list[str]) -> None:
    proc = run_capture(
        [
            sys.executable,
            "-m",
            "unittest",
            "env.mec_offloaing_envs.scheduler.tests.test_phase2_encoder",
            "-v",
        ]
    )
    out = proc.stdout + "\n" + proc.stderr
    for line in out.splitlines():
        if line.startswith(("test_", "OK", "FAILED", "ERROR", "Ran ", "=")):
            print(line)
        if "SKIP" in line or "skipped" in line.lower():
            block(reasons, f"encoder test skipped: {line.strip()}")
    if proc.returncode != 0:
        sys.stdout.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        block(reasons, f"Phase 2 encoder unittest failed (rc={proc.returncode})")


def main() -> int:
    reasons: list[str] = []
    print("=== Phase 2 encoder gate ===")
    check_status_doc(reasons)
    check_no_clique(reasons)
    check_masked_mean(reasons)
    check_neighbor_capacity(reasons)
    check_encode_path(reasons)
    check_stats(reasons)
    check_encoder_tests(reasons)

    print("\nGATE SNAPSHOT")
    print("PHASE2_STATUS.md:     checked")
    print("No clique adj:        checked")
    print("Masked mean:          checked")
    print("Neighbor capacity:    checked")
    print("Encode / obs_dim:     checked")
    print("meta_train stats:     checked")
    print("Encoder unit tests:   checked")

    if reasons:
        print("\nPhase 2 encoder: BLOCKED")
        for r in reasons:
            print(f"  - {r}")
        return 1

    print("\nPhase 2 encoder: PASS")
    print("Phase 2 closure: NOT CLAIMED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
