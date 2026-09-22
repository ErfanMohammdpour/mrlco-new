#!/usr/bin/env python3
"""⑤b — effective-rank / separability probe for the MARGO observation pipeline.

Three measurements (all requested before deciding on a wider decoder):

  T1  rank of the observation -> encoder representation
        effective rank (spectral entropy), participation ratio, and the number of
        components needed for 90/95/99% of the variance.

  T2  the same rank with the deadline block ZEROED
        if the rank barely moves, the deadline features are not enriching the
        representation and a wider decoder cannot help.

  T3  action separability (the measurement that actually matters)
        label every graph by its best pure action (min makespan), then ask whether
        the representation separates those classes: Fisher between/within ratio,
        centroid distances, and a closed-form ridge linear probe with k-fold CV
        accuracy against the 1/3 chance level.  A second probe separates the
        deadline REGIME (loose vs tight) — that is the direct test that deadline
        information is present and linearly usable.

Modes
  --mode obs          raw + standardized observation features (numpy only)
  --mode structural   numpy replication of the Graph2Seq aggregation path with
                      fixed random weights -> h[256] and state[128].  This is a
                      STRUCTURAL CAPACITY probe (what the architecture can carry
                      before training), not the trained representation.
  --mode activations  analyze real activations dumped by
                      spec/dump_encoder_activations.py (--npz file)

Usage
    python spec/effective_rank_probe.py --mode obs --graphs 120
    python spec/effective_rank_probe.py --mode structural --graphs 120
    python spec/effective_rank_probe.py --mode activations --npz acts.npz
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from env.mec_offloaing_envs.offloading_task_graph import OffloadingTaskGraph  # noqa: E402
from env.mec_offloaing_envs.scheduler import (  # noqa: E402
    ResourceConfig,
    pure_location_plan,
    schedule_via_adapter,
)
from env.mec_offloaing_envs.scheduler.adapter import to_canonical_dag  # noqa: E402
from env.mec_offloaing_envs.scheduler.deadlines import assign_deadlines  # noqa: E402

DEFAULT_DATA = ROOT / "env" / "mec_offloaing_envs" / "data" / "meta_offloading_20"
MBPS_TO_BPS = 1024.0 * 1024.0 / 8.0
DEADLINE_BLOCK_START = 15  # features 0..14 are v2; 15.. are the v3 deadline block


class _FrozenCluster:
    def __init__(self):
        self.mobile_process_capable = 1.0 * 1024 * 1024
        self.mec_process_capable = 10.0 * 1024 * 1024
        self.v2v_process_capable = 1.0 * 1024 * 1024
        self.bandwidth_up = 7.0
        self.bandwidth_dl = 7.0
        self.v2v_bandwidth = 5.0

    def up_transmission_cost(self, data):
        return data / (self.bandwidth_up * MBPS_TO_BPS)

    def dl_transmission_cost(self, data):
        return data / (self.bandwidth_dl * MBPS_TO_BPS)

    def v2v_transmission_cost(self, data):
        return data / (self.v2v_bandwidth * MBPS_TO_BPS)


# ---------------------------------------------------------------------------
# rank math
# ---------------------------------------------------------------------------
def effective_rank_stats(x: np.ndarray) -> dict:
    """Spectral statistics of a 2-D matrix of representations."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("expected a 2-D matrix")
    x = x - x.mean(axis=0, keepdims=True)
    # drop all-zero columns (must not inflate the rank)
    norms = np.linalg.norm(x, axis=0)
    x = x[:, norms > 1e-12]
    if x.shape[1] == 0:
        return {"dims": [int(x.shape[0]), 0], "effective_rank": 0.0,
                "participation_ratio": 0.0, "n90": 0, "n95": 0, "n99": 0,
                "top1_share": 0.0, "singular_values": []}
    sv = np.linalg.svd(x, compute_uv=False)
    energy = sv ** 2
    total = float(energy.sum())
    if total <= 0.0:
        return {"dims": list(x.shape), "effective_rank": 0.0,
                "participation_ratio": 0.0, "n90": 0, "n95": 0, "n99": 0,
                "top1_share": 0.0, "singular_values": []}
    p = energy / total
    entropy = float(-(p * np.log(p + 1e-300)).sum())
    cumulative = np.cumsum(p)

    def n_for(share):
        return int(np.searchsorted(cumulative, share) + 1)

    return {
        "dims": [int(x.shape[0]), int(x.shape[1])],
        "effective_rank": float(math.exp(entropy)),
        "participation_ratio": float(1.0 / float((p ** 2).sum())),
        "n90": n_for(0.90),
        "n95": n_for(0.95),
        "n99": n_for(0.99),
        "top1_share": float(p[0]),
        "singular_values": [float(v) for v in sv[:20]],
    }


def ridge_probe_accuracy(x: np.ndarray, y: np.ndarray, folds: int = 5, lam: float = 1.0) -> dict:
    """Closed-form ridge one-hot classifier with k-fold CV (no sklearn)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    n, d = x.shape
    classes = np.unique(y)
    if len(classes) < 2 or n < folds * 2:
        return {"accuracy": float("nan"), "chance": 1.0 / max(1, len(classes)), "n": int(n)}
    mu, sd = x.mean(axis=0), x.std(axis=0)
    sd[sd < 1e-9] = 1.0
    xs = (x - mu) / sd
    order = np.arange(n)
    rng = np.random.RandomState(0)
    rng.shuffle(order)
    xs, y = xs[order], y[order]
    hits = 0
    for k in range(folds):
        test = np.arange(k, n, folds)
        train = np.setdiff1d(np.arange(n), test)
        xt, yt = xs[train], y[train]
        onehot = np.zeros((len(train), len(classes)))
        for i, label in enumerate(yt):
            onehot[i, np.searchsorted(classes, label)] = 1.0
        a = xt.T @ xt + lam * np.eye(d)
        w = np.linalg.solve(a, xt.T @ onehot)
        pred = np.argmax(xs[test] @ w, axis=1)
        truth = np.array([np.searchsorted(classes, v) for v in y[test]])
        hits += int((pred == truth).sum())
    return {
        "accuracy": float(hits) / float(n),
        "chance": 1.0 / float(len(classes)),
        "n": int(n),
        "classes": [int(c) for c in classes],
    }


def fisher_ratio(x: np.ndarray, y: np.ndarray) -> float:
    """Between-class / within-class scatter ratio (trace form)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    sd = x.std(axis=0)
    sd[sd < 1e-9] = 1.0
    x = (x - x.mean(axis=0)) / sd
    overall = x.mean(axis=0)
    between = 0.0
    within = 0.0
    for label in np.unique(y):
        sel = x[y == label]
        if len(sel) == 0:
            continue
        centre = sel.mean(axis=0)
        between += len(sel) * float(((centre - overall) ** 2).sum())
        within += float(((sel - centre) ** 2).sum())
    return float(between / within) if within > 1e-12 else float("inf")


# ---------------------------------------------------------------------------
# structural surrogate of the Graph2Seq path (numpy, fixed random weights)
# ---------------------------------------------------------------------------
def structural_encoder(packed: np.ndarray, seed: int = 0, hidden: int = 128) -> dict:
    """Masked-mean Graph2Seq surrogate mirroring the frozen architecture.

    Layer shapes follow spec/CURRENT_ARCHITECTURE.md §5:
      embed(features -> 128)
      per direction (fw = successors, bw = predecessors), 2 layers:
        layer0: concat(self 128, neigh 128) -> 256
        layer1: concat(self 256, neigh 256) -> 256
      h = relu(fw_hidden + bw_hidden)              # [B, 20, 256]
      readout: concat(mean, max, attn) 768 -> 256 -> 128

    Weights are FIXED RANDOM (seeded), so the measured rank is a STRUCTURAL
    CAPACITY bound for the untrained architecture — not the trained model.
    """
    packed = np.asarray(packed, dtype=np.float64)
    if packed.ndim != 3:
        raise ValueError("packed obs must be [B, 20, PACKED_DIM]")
    b, n, packed_dim = packed.shape
    feature_dim = packed_dim - 2 * 19 - 1
    feats = packed[:, :, :feature_dim]
    # The real pipeline z-scores the node features; without it the readout tanh
    # saturates on raw byte magnitudes and the surrogate collapses to rank 0
    # (observed during development). Standardize here to match the real path.
    mu = feats.reshape(-1, feature_dim).mean(axis=0)
    sd = feats.reshape(-1, feature_dim).std(axis=0)
    sd[sd < 1e-9] = 1.0
    feats = (feats - mu) / sd
    fw = packed[:, :, feature_dim:feature_dim + 19].astype(np.int64)
    bw = packed[:, :, feature_dim + 19:feature_dim + 38].astype(np.int64)
    node_mask = (packed[:, :, -1] > 0.5).astype(np.float64)
    rng = np.random.RandomState(seed)
    out_dim = 2 * hidden

    def dense(x, out_features, scale):
        w = rng.normal(0.0, scale / math.sqrt(max(1, x.shape[-1])), (x.shape[-1], out_features))
        return x @ w

    embed = np.maximum(dense(feats, hidden, 1.0), 0.0)          # [B, 20, 128]

    def neighbour_mean(x, idx):
        rows = []
        for t in range(x.shape[1]):
            slots = idx[:, t, :]
            valid = slots >= 0
            safe = np.where(valid, slots, 0)
            gathered = x[np.arange(x.shape[0])[:, None], safe]   # [B, 19, d]
            gathered = gathered * valid[:, :, None]
            counts = np.maximum(valid.sum(axis=1, keepdims=True), 1)
            rows.append(gathered.sum(axis=1) / counts)
        return np.stack(rows, axis=1)

    def run_direction(x, idx):
        neigh = neighbour_mean(x, idx)
        y = np.maximum(dense(np.concatenate([x, neigh], axis=-1), out_dim, 1.0), 0.0)
        neigh2 = neighbour_mean(y, idx)
        return np.maximum(
            dense(np.concatenate([y, neigh2], axis=-1), out_dim, 1.0), 0.0
        )

    fw_h = run_direction(embed, fw)
    bw_h = run_direction(embed, bw)
    node_h = np.maximum(fw_h + bw_h, 0.0) * node_mask[:, :, None]   # [B, 20, 256]

    mask3 = node_mask[:, :, None]
    denom = np.maximum(mask3.sum(axis=1), 1e-8)
    mean_pool = (node_h * mask3).sum(axis=1) / denom
    neg = (1.0 - mask3) * (-1e9)
    max_pool = np.max(node_h + neg, axis=1)
    attn_logits = dense(node_h, 1, 1.0) + neg                        # [B, 20, 1]
    attn_logits = attn_logits - attn_logits.max(axis=1, keepdims=True)
    attn = np.exp(attn_logits)
    attn = attn / np.maximum(attn.sum(axis=1, keepdims=True), 1e-12)
    attn_pool = (node_h * attn).sum(axis=1)
    pooled = np.concatenate([mean_pool, max_pool, attn_pool], axis=-1)   # [B, 768]
    proj256 = np.tanh(dense(pooled, 256, 1.0))
    state128 = dense(proj256, 128, 1.0)
    return {"node_h": node_h, "proj256": proj256, "state128": state128}


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def build_dataset(dist: int, graphs: int, deadline_factor: float | None):
    """Return list of dicts with packed obs (loose/tight), label, regime."""
    resources = ResourceConfig.from_frozen_yaml(model="physical_v1")
    cluster = _FrozenCluster()
    folder = DEFAULT_DATA / f"offload_random20_{dist}"
    files = sorted(folder.glob("random.20.*.gv"))[:graphs]
    if not files:
        raise SystemExit("no graphs in %s" % folder)

    from env.mec_offloaing_envs.scheduler import encoder_obs as eo

    eo.set_obs_version("v3")
    rows = []
    for path in files:
        tg = OffloadingTaskGraph(str(path))
        tg.prioritize_tasks(cluster)
        order = [int(t) for t in tg.prioritize_sequence]
        # labels: best pure action by makespan (on the deadline-free graph)
        best_action, best_t = None, float("inf")
        for action in (0, 1, 2):
            out, _, _ = schedule_via_adapter(tg, pure_location_plan(order, action), resources)
            if out.makespan_seconds < best_t:
                best_t, best_action = out.makespan_seconds, action
        obs_loose = eo.encode_canonical_dag(
            to_canonical_dag(tg), order, resources=resources, cycles_per_bit=300.0
        )
        tight = None
        if deadline_factor is not None:
            tg2 = OffloadingTaskGraph(str(path))
            tg2.prioritize_tasks(cluster)
            assign_deadlines(
                tg2, resources, rule="uniform_of_all_mec",
                factor=float(deadline_factor), deadline_type="hard",
            )
            tight = eo.encode_canonical_dag(
                to_canonical_dag(tg2), order, resources=resources, cycles_per_bit=300.0
            )
        from env.mec_offloaing_envs.scheduler import greedy_from_mec_plan

        teacher_plan, _ = greedy_from_mec_plan(tg, resources, max_passes=2)
        teacher = [a for _tid, a in teacher_plan]        # aligned to decoder order
        rows.append({"graph": path.name, "order": order, "loose": obs_loose,
                     "tight": tight, "label": int(best_action), "best_T": best_t,
                     "teacher": np.asarray(teacher, dtype=np.int64)})
    return rows, resources


def flatten(rows, key):
    """[G, 20, PACKED] -> [G*20, PACKED]"""
    return np.concatenate([r[key] for r in rows], axis=0)


def stack(rows, key):
    """[G, 20, PACKED]"""
    return np.stack([r[key] for r in rows], axis=0)


def teacher_labels(rows):
    return np.concatenate([r["teacher"] for r in rows], axis=0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("obs", "structural", "activations"), default="obs")
    ap.add_argument("--dist", type=int, default=1)
    ap.add_argument("--graphs", type=int, default=120)
    ap.add_argument("--deadline-factor", type=float, default=1.05)
    ap.add_argument("--npz", type=str, default=None)
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()

    out: dict = {"mode": args.mode, "dist": args.dist, "graphs": args.graphs}
    print("=" * 78)
    print("EFFECTIVE-RANK PROBE — mode=%s dist=%d graphs=%d" % (args.mode, args.dist, args.graphs))
    print("=" * 78)

    if args.mode == "activations":
        if not args.npz:
            raise SystemExit("--mode activations requires --npz")
        data = np.load(args.npz)
        for key in data.files:
            arr = data[key]
            if arr.ndim >= 2:
                flat = arr.reshape(-1, arr.shape[-1])
                out[key] = effective_rank_stats(flat)
                s = out[key]
                print(
                    "  %-28s dims=%s  erank=%.1f  PR=%.1f  n90=%d n95=%d"
                    % (key, s["dims"], s["effective_rank"], s["participation_ratio"],
                       s["n90"], s["n95"])
                )
        if "labels" in data.files and "state128" in data.files:
            x = data["state128"].reshape(-1, data["state128"].shape[-1])
            y = np.asarray(data["labels"]).reshape(-1)
            n = min(len(x), len(y))
            out["action_probe"] = ridge_probe_accuracy(x[:n], y[:n])
            out["action_fisher"] = fisher_ratio(x[:n], y[:n])
            print("  action probe: acc=%.3f (chance %.3f)"
                  % (out["action_probe"]["accuracy"], out["action_probe"]["chance"]))
        if args.json:
            Path(args.json).write_text(json.dumps(out, indent=2))
        return 0

    rows, resources = build_dataset(args.dist, args.graphs, args.deadline_factor)
    labels = np.array([r["label"] for r in rows])
    t_labels = teacher_labels(rows)
    print("\nbest-pure-action label distribution (per graph): %s"
          % {int(k): int((labels == k).sum()) for k in np.unique(labels)})
    print("teacher action distribution (per task, n=%d): %s"
          % (len(t_labels), {int(k): int((t_labels == k).sum()) for k in np.unique(t_labels)}))
    if len(np.unique(labels)) == 1:
        print("  NOTE: every graph's best PURE action is the same (MEC); per-graph "
              "action separability is therefore degenerate. Per-TASK teacher labels "
              "are used for the separability probe.")

    if args.mode == "obs":
        for name, key, zero_block in (
            ("obs v3 raw", "loose", False),
            ("obs v3 deadline block ZEROED", "loose", True),
        ):
            x = flatten(rows, key).astype(np.float64)
            if zero_block:
                x = x.copy()
                x[:, DEADLINE_BLOCK_START:] = 0.0
            stats = effective_rank_stats(x)
            out[name] = stats
            print("  %-32s dims=%s erank=%.1f PR=%.1f n90=%d n95=%d"
                  % (name, stats["dims"], stats["effective_rank"],
                     stats["participation_ratio"], stats["n90"], stats["n95"]))
        # T2: does the deadline block change the observation rank / regime separability?
        x_loose = stack(rows, "loose")
        x_tight = stack(rows, "tight")
        diff = np.abs(x_loose - x_tight)
        out["deadline_block_effect"] = {
            "max_abs_diff": float(diff.max()),
            "mean_abs_diff": float(diff.mean()),
            "graphs_changed": int((diff.max(axis=(1, 2)) > 1e-9).sum()),
            "graphs_total": int(diff.shape[0]),
        }
        print("  deadline loose-vs-tight: graphs changed %d/%d  max|Δ|=%.4f"
              % (out["deadline_block_effect"]["graphs_changed"],
                 out["deadline_block_effect"]["graphs_total"],
                 out["deadline_block_effect"]["max_abs_diff"]))
        # regime probe on the deadline block only
        block_loose = x_loose[:, :, DEADLINE_BLOCK_START:].reshape(-1, len_ := (x_loose.shape[2] - DEADLINE_BLOCK_START))
        block_tight = x_tight[:, :, DEADLINE_BLOCK_START:].reshape(-1, len_)
        y_regime = np.concatenate([np.zeros(block_loose.shape[0]), np.ones(block_tight.shape[0])])
        x_regime = np.concatenate([block_loose, block_tight], axis=0)
        out["regime_probe"] = ridge_probe_accuracy(x_regime, y_regime.astype(int))
        out["regime_fisher"] = fisher_ratio(x_regime, y_regime.astype(int))
        print("  deadline-regime probe (loose vs tight) on deadline block: acc=%.3f"
              % out["regime_probe"]["accuracy"])

    else:  # structural
        packed_loose = stack(rows, "loose")
        packed_tight = stack(rows, "tight")
        out["labels"] = [int(v) for v in labels]
        out["teacher_labels"] = [int(v) for v in t_labels]
        for name, packed in (("loose", packed_loose), ("tight", packed_tight)):
            res = structural_encoder(packed, seed=0)
            node_h = res["node_h"].reshape(-1, res["node_h"].shape[-1])
            stats_node = effective_rank_stats(node_h)
            stats_256 = effective_rank_stats(res["proj256"])
            stats_128 = effective_rank_stats(res["state128"])
            out[f"structural_{name}"] = {
                "node_h_256": stats_node, "proj256": stats_256, "state128": stats_128,
            }
            print("\n  [%s] node_h(%d-d): erank=%.1f PR=%.1f n90=%d"
                  % (name, stats_node["dims"][1], stats_node["effective_rank"],
                     stats_node["participation_ratio"], stats_node["n90"]))
            print("  [%s] proj 256-d : erank=%.1f PR=%.1f n90=%d"
                  % (name, stats_256["effective_rank"], stats_256["participation_ratio"],
                     stats_256["n90"]))
            print("  [%s] state 128-d: erank=%.1f PR=%.1f n90=%d  (per-graph state)"
                  % (name, stats_128["effective_rank"], stats_128["participation_ratio"],
                     stats_128["n90"]))
            # T3a: per-TASK teacher-action separability on node embeddings
            probe_task = ridge_probe_accuracy(node_h, t_labels)
            fisher_task = fisher_ratio(node_h, t_labels)
            out[f"structural_{name}"]["task_action_probe"] = probe_task
            out[f"structural_{name}"]["task_action_fisher"] = fisher_task
            print("  [%s] per-task teacher-action probe: acc=%.3f (chance %.3f) Fisher=%.3f"
                  % (name, probe_task["accuracy"], probe_task["chance"], fisher_task))
            # T3b: per-graph best-pure-action separability (may be degenerate)
            per_graph = res["state128"]           # [G, 128]
            probe = ridge_probe_accuracy(per_graph, labels)
            out[f"structural_{name}"]["graph_action_probe"] = probe
            print("  [%s] per-graph best-pure-action probe: acc=%.3f (chance %.3f, %d classes)"
                  % (name, probe["accuracy"], probe["chance"], len(np.unique(labels))))

        # deadine effect on the structural state: recompute tight with same weights
        res_loose = structural_encoder(packed_loose, seed=0)
        res_tight = structural_encoder(packed_tight, seed=0)
        delta = np.abs(res_loose["state128"] - res_tight["state128"])
        out["structural_deadline_delta"] = {
            "max_abs_diff": float(delta.max()),
            "mean_abs_diff": float(delta.mean()),
        }
        print("\n  structural state: loose-vs-tight  max|Δ|=%.5f  mean|Δ|=%.5f"
              % (out["structural_deadline_delta"]["max_abs_diff"],
                 out["structural_deadline_delta"]["mean_abs_diff"]))

    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2))
        print("\nraw JSON -> %s" % args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
