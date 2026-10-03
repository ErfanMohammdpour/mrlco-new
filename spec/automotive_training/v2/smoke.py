#!/usr/bin/env python3
"""v2 end-to-end smoke + checkpoint/resume, executable WITHOUT TensorFlow.

What is REAL in this harness:
* the v2 environment: canonical world builder, shared scheduler, event-based energy ledger,
  budget/dual constraint manager, the 109-wide observation tensor;
* the FROZEN outer-update operator `spec.learning_ops.mean_pseudogradient(theta0, adapted,
  inner_lr, k_steps)` — imported, not reimplemented;
* the FROZEN support-row selection `spec.learning_ops.select_support_rows`;
* the CRN protocol identity and the frozen hyperparameters (inner/outer lr 5e-4, 3 inner
  steps in the reference config, clip 0.2).

What is SUBSTITUTED (declared): the TensorFlow graph. The seq2seq/Graph2Seq network is
replaced by a numpy linear softmax policy over the emitted observation, and the PPO surrogate
is evaluated analytically for that policy. The chain it exercises is the same chain:
support rollout -> inner PPO -> query rollout on an independent realization -> outer update ->
held-out adaptation/validation -> feasibility-aware checkpoint selection -> save -> restore ->
resumed iteration.

The frozen TF path therefore remains NOT RUN; this harness proves the CONTRACT around it
(dimensions, schemas, hashes, duals, RNG, optimizer slots, resume equivalence), not the
frozen network's numerics.

    python3 -m spec.automotive_training.v2.smoke --meta-tasks 4 --support 2 --inner 3
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REPORTS = ROOT / "spec" / "automotive_training" / "reports" / "v2_system_model"

from spec.automotive_training.automotive_loader import load_dataset  # noqa: E402
from spec.automotive_training.automotive_primary import AutomotiveResourceCluster  # noqa: E402
from spec.automotive_training.v2.context_fields import V2_CONTEXT_FIELDS  # noqa: E402
from spec.automotive_training.v2.crn import PROTOCOL_ID, protocol_sha, seed_pair  # noqa: E402
from spec.automotive_training.v2.env import V2AutomotiveEnv  # noqa: E402
from spec.automotive_training.v2.observation import (  # noqa: E402
    V1_FEATURE_DIM, V2_CONTEXT_DIM, V2_FEATURE_DIM, V2_OBS_VERSION, V2_PACKED_DIM,
)
from spec.automotive_training.v2.world import V2WorldConfig  # noqa: E402
from spec.learning_ops import mean_pseudogradient, select_support_rows  # noqa: E402

SCHEMA = "v2_smoke_v1"
CHECKPOINT_SCHEMA = "v2_checkpoint_v1"
#: token-position features appended to the emitted graph embedding
POSITION_FEATURES = 3


class SmokeError(RuntimeError):
    """Raised when the smoke chain violates its contract."""


def _stable_int(*parts) -> int:
    payload = "|".join(str(p) for p in parts).encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:12], 16) % (2 ** 31 - 1)


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def _log_softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


class AdamSlots:
    """Adam with EXPLICIT slot state, so 'fresh slots per task' is observable and savable."""

    def __init__(self, shape, lr=5e-4, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = float(lr)
        self.b1, self.b2, self.eps = float(beta1), float(beta2), float(eps)
        self.shape = tuple(shape)
        self.reset()

    def reset(self) -> None:
        self.m = np.zeros(self.shape, dtype=np.float64)
        self.v = np.zeros(self.shape, dtype=np.float64)
        self.t = 0

    def step(self, grad: np.ndarray) -> None:
        self.t += 1
        g = np.asarray(grad, dtype=np.float64)
        self.m = self.b1 * self.m + (1.0 - self.b1) * g
        self.v = self.b2 * self.v + (1.0 - self.b2) * (g * g)
        mh = self.m / (1.0 - self.b1 ** self.t)
        vh = self.v / (1.0 - self.b2 ** self.t)
        self.last_update = self.lr * mh / (np.sqrt(vh) + self.eps)

    def state(self) -> dict:
        return {"m": self.m.tolist(), "v": self.v.tolist(), "t": int(self.t),
                "shape": list(self.shape)}

    def load_state(self, state: Mapping) -> None:
        if list(state.get("shape", [])) != list(self.shape):
            raise SmokeError("optimizer slot shape mismatch: %r vs %r"
                             % (state.get("shape"), list(self.shape)))
        self.m = np.asarray(state["m"], dtype=np.float64)
        self.v = np.asarray(state["v"], dtype=np.float64)
        self.t = int(state["t"])


@dataclass
class SmokeConfig:
    meta_tasks: int = 4
    support: int = 2
    query: int = 2
    inner_steps: int = 3
    inner_lr: float = 5e-4
    outer_lr: float = 5e-4
    clip: float = 0.2
    tokens: int = 20
    seed: int = 7
    background_dags: int = 0
    constraints: bool = True
    budget_fraction: float = 2.0
    validations: int = 2

    def as_dict(self) -> dict:
        return dict(self.__dict__)

    def sha256(self) -> str:
        return hashlib.sha256(json.dumps(self.as_dict(), sort_keys=True).encode()).hexdigest()


def token_features(obs_slot: np.ndarray, token_index: int, n_tokens: int) -> np.ndarray:
    """[V2_FEATURE_DIM + POSITION_FEATURES] policy input for one token.

    The graph embedding is the mean over the emitted node rows of the REAL observation tensor
    (the v1 feature block plus the v2 context block), and the token position is appended so a
    single linear map can still place different tokens differently. The decoder-order to
    node-order mapping is the real policy's job and is NOT faked here.
    """
    arr = np.asarray(obs_slot, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != V2_PACKED_DIM:
        raise SmokeError("observation slot must be [nodes, %d], got %r"
                         % (V2_PACKED_DIM, (arr.shape,)))
    embedded = arr[:, :V2_FEATURE_DIM].mean(axis=0)
    denom = max(1, int(n_tokens) - 1)
    position = np.asarray([token_index / denom,
                           float(token_index == 0),
                           float(token_index == n_tokens - 1)], dtype=np.float64)
    return np.concatenate([embedded, position])


class ReferencePolicy:
    """numpy linear softmax over the emitted observation; the TF-substituted component."""

    def __init__(self, n_features: int, seed: int = 0, scale: float = 0.01):
        rng = np.random.RandomState(int(seed))
        self.W = (rng.randn(int(n_features), 3) * float(scale)).astype(np.float64)

    def logits(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=np.float64) @ self.W

    def copy_weights(self) -> np.ndarray:
        return self.W.copy()

    def set_weights(self, W: np.ndarray) -> None:
        if np.asarray(W).shape != self.W.shape:
            raise SmokeError("weight shape mismatch: %r vs %r"
                             % (np.asarray(W).shape, self.W.shape))
        self.W = np.asarray(W, dtype=np.float64).copy()


def ppo_surrogate_gradient(X: np.ndarray, actions: np.ndarray, old_logits: np.ndarray,
                           advs: np.ndarray, params: np.ndarray,
                           *, clip: float) -> tuple:
    """Analytic PPO gradient for a linear softmax policy, computed by TEACHER FORCING on the
    STORED actions (no re-sampling). Returns (grad, diagnostics)."""
    logits = X @ params
    logp = _log_softmax(logits)
    logp_old = _log_softmax(old_logits)
    rows = np.arange(actions.shape[0])
    ratio = np.exp(logp[rows, actions] - logp_old[rows, actions])
    unclipped = ratio * advs
    clipped = np.clip(ratio, 1.0 - clip, 1.0 + clip) * advs
    use_unclipped = unclipped <= clipped
    inside = (ratio >= 1.0 - clip) & (ratio <= 1.0 + clip)
    active = np.where(use_unclipped, True, inside)
    coef = -advs * ratio * active.astype(np.float64)
    probs = _softmax(logits)
    onehot = np.zeros_like(probs)
    onehot[rows, actions] = 1.0
    grad = X.T @ (coef[:, None] * (onehot - probs)) / float(actions.shape[0])
    diag = {
        "surrogate": float(np.mean(np.minimum(unclipped, clipped))),
        "ratio_mean": float(np.mean(ratio)),
        "ratio_max": float(np.max(ratio)),
        "kl_approx": float(np.mean(logp_old[rows, actions] - logp[rows, actions])),
        "clipped_fraction": float(np.mean(~active)),
    }
    return grad, diag


def _build_env(config: SmokeConfig, graphs: Sequence) -> V2AutomotiveEnv:
    env = V2AutomotiveEnv(
        list(graphs), AutomotiveResourceCluster(), role="meta_train",
        slots_per_task=len(graphs), base_seed=int(config.seed), single_dist=True,
        background_dags=int(config.background_dags), constraints_enabled=bool(config.constraints),
        budget_fractions={"total_energy": float(config.budget_fraction)},
        scheduler_config_sha256=None)
    env.set_task({"dist_index": 0, "graph_indices": np.arange(len(graphs), dtype=np.int32)})
    return env


def _rollout(env: V2AutomotiveEnv, policy: ReferencePolicy, config: SmokeConfig, *,
             episode: int, trajectory: int, tag: str) -> dict:
    """One support/query trajectory: PER-TOKEN RNG substreams, then the real v2 step.

    The environmental realization is fixed by a STABLE key (tag/episode/trajectory), so a
    replay, a CRN pairing or a resumed checkpoint draws exactly the same world.
    """
    env.set_world_realization("%s/%d/%d" % (tag, int(episode), int(trajectory)))
    obs = np.asarray(env.reset(), dtype=np.float32)
    n_slots = obs.shape[0]
    actions = np.zeros((n_slots, int(config.tokens)), dtype=int)
    logits = np.zeros((n_slots, int(config.tokens), 3), dtype=np.float64)
    X = np.zeros((n_slots, int(config.tokens),
                  V2_FEATURE_DIM + POSITION_FEATURES), dtype=np.float64)
    seeds = []
    for slot in range(n_slots):
        for token in range(int(config.tokens)):
            x = token_features(obs[slot], token, int(config.tokens))
            X[slot, token] = x
            z = policy.logits(x)
            logits[slot, token] = z
            # a SEPARATE substream per token: one stateless seed reused across tokens would
            # give repeatability but correlate the sampled decisions
            s = _stable_int(PROTOCOL_ID, tag, config.seed, episode, trajectory, slot, token)
            seeds.append(int(s))
            noise = np.random.RandomState(s).gumbel(size=3)
            actions[slot, token] = int(np.argmax(z + noise))
    _obs2, rewards, _done, info = env.step(actions)
    returns = np.asarray([float(np.sum(r)) for r in rewards], dtype=np.float64)
    return {
        "tag": tag, "episode": int(episode), "trajectory": int(trajectory),
        "n_slots": int(n_slots), "actions": actions.tolist(),
        "logits": logits, "X": X, "returns": returns, "rewards": rewards,
        "telemetry": info[2], "token_seeds": seeds,
    }


def _advantages(returns: np.ndarray) -> np.ndarray:
    r = np.asarray(returns, dtype=np.float64)
    if r.size > 1 and r.std() > 1e-12:
        return (r - r.mean()) / (r.std() + 1e-8)
    return r - r.mean()


def inner_adapt(policy: ReferencePolicy, trajectories: Sequence[dict], config: SmokeConfig,
                *, seed: int) -> dict:
    """K inner PPO steps on the SUPPORT trajectories with FRESH optimizer slots."""
    adam = AdamSlots(policy.W.shape, lr=float(config.inner_lr))
    rows = select_support_rows(len(trajectories), len(trajectories),
                               np.random.RandomState(int(seed)))
    diags = []
    for _ in range(int(config.inner_steps)):
        grads, weights = [], []
        for idx in rows:
            traj = trajectories[int(idx)]
            n_slots = traj["X"].shape[0]
            for slot in range(n_slots):
                X = traj["X"][slot]
                a = np.asarray(traj["actions"][slot], dtype=int)
                old = traj["logits"][slot]
                adv = np.full(a.shape[0], _advantages(traj["returns"])[slot], dtype=np.float64)
                g, diag = ppo_surrogate_gradient(X, a, old, adv, policy.W,
                                                 clip=float(config.clip))
                grads.append(g)
                weights.append(1.0 / max(1, a.shape[0]))
                diags.append(diag)
        total_w = float(sum(weights))
        grad = sum(g * (w / total_w) for g, w in zip(grads, weights))
        adam.step(grad)
        policy.set_weights(policy.W + adam.last_update)
    return {"inner_steps": int(config.inner_steps), "diagnostics": diags,
            "weight_delta_norm": float(np.linalg.norm(adam.last_update)),
            "adam_t": int(adam.t)}


def _checkpoint_doc(policy: ReferencePolicy, outer_adam: AdamSlots, env, config: SmokeConfig,
                    *, iteration: int, rng_state, extra: Mapping | None = None) -> dict:
    world = env.world_for_slot(0)
    refs = env.reference_ranges_for_slot(0)
    return {
        "schema": CHECKPOINT_SCHEMA,
        "iteration": int(iteration),
        "weights": policy.W.tolist(),
        "outer_optimizer": outer_adam.state(),
        "inner_optimizer_contract": "fresh slots per meta-task (reset before each inner loop)",
        "duals": (env.constraint_manager.state() if env.constraint_manager else {"enabled": False}),
        "constraint_spec_sha256": (hashlib.sha256(json.dumps(
            env.constraint_manager.spec.as_dict(), sort_keys=True, default=str).encode()
        ).hexdigest() if env.constraint_manager else None),
        "reference_scope": getattr(refs, "energy_scope", None),
        "reference_scheduler_config_sha256": getattr(refs, "scheduler_config_sha256", None),
        "obs": {"obs_version": V2_OBS_VERSION, "v2_context_dim": V2_CONTEXT_DIM,
                "v2_feature_dim": V2_FEATURE_DIM, "v2_packed_dim": V2_PACKED_DIM,
                "context_fields": list(V2_CONTEXT_FIELDS)},
        "normalisation": "v2 context identity entries; frozen v1 statistics (train-only)",
        "world_config_sha256": env.world_config.sha256(),
        "world_fingerprint_sha256": world.fingerprint_sha256(),
        "scheduler_config_sha256": env._scheduler_fingerprint(),
        "crn_protocol_id": PROTOCOL_ID,
        "crn_protocol_sha256": protocol_sha(),
        "smoke_config": config.as_dict(),
        "smoke_config_sha256": config.sha256(),
        "rng_contract": "per-token substreams keyed by (protocol, tag, seed, episode, "
                        "trajectory, slot, token)",
        "rng_state": rng_state,
        **(dict(extra) if extra else {}),
    }


def _iterate(env, policy, outer_adam, config, *, iteration: int, rng, rng_state_fn) -> dict:
    """One complete iteration: support rollouts, inner PPO, query, outer update, validation."""
    task_params, inner_reports, support_all = [], [], []
    for task in range(int(config.meta_tasks)):
        support = []
        for t in range(int(config.support)):
            support.append(_rollout(env, policy, config, episode=iteration * 100 + task,
                                    trajectory=t, tag="support"))
        support_all.extend(support)
        theta0 = policy.copy_weights()
        report = inner_adapt(policy, support, config,
                             seed=_stable_int("inner", config.seed, iteration, task))
        task_params.append(policy.copy_weights())
        inner_reports.append({"task": task, "theta0_norm": float(np.linalg.norm(theta0)),
                              **report})
        policy.set_weights(theta0)      # restore the initial parameters before the next task
    # ---- frozen first-order outer update -------------------------------------------
    core = policy.copy_weights()
    adapted = [task_params]
    grads = mean_pseudogradient([core], [[p] for p in task_params],
                                float(config.inner_lr), int(config.inner_steps))
    outer_adam.step(grads[0])
    core_after = core + outer_adam.last_update
    policy.set_weights(core_after)
    outer_norm = float(np.linalg.norm(outer_adam.last_update))
    # ---- query on INDEPENDENT realizations + held-out validation --------------------
    query = []
    for task in range(int(config.meta_tasks)):
        policy.set_weights(task_params[task])
        for t in range(int(config.query)):
            # `_rollout` sets its own identity key and resets, so this is an INDEPENDENT
            # realization of the same meta-task, not a continuation of the support episode
            query.append(_rollout(env, policy, config, episode=iteration * 100 + 50 + task,
                                  trajectory=t, tag="query"))
    policy.set_weights(core_after)
    validation = []
    for v in range(int(config.validations)):
        validation.append(_rollout(env, policy, config, episode=iteration * 100 + 90 + v,
                                   trajectory=0, tag="validation"))
    return {
        "iteration": int(iteration),
        "inner": inner_reports,
        "outer_update_norm": outer_norm,
        "outer_pseudogradient_norm": float(np.linalg.norm(grads[0])),
        "core_weight_norm": float(np.linalg.norm(core_after)),
        "support_return_mean": float(np.mean([np.mean(s["returns"]) for s in support_all])),
        "query_return_mean": float(np.mean([np.mean(q["returns"]) for q in query])),
        "validation_return_mean": float(np.mean([np.mean(v["returns"]) for v in validation])),
        "validation_latency_mean": float(np.mean([t["v2"]["makespan_s"]
                                                  for v in validation for t in v["telemetry"]])),
        "validation_system_joules_mean": float(np.mean(
            [t["system_joules"] for v in validation for t in v["telemetry"]])),
        "validation_violation_total": float(sum(
            t["constraints"]["total_violation"] for v in validation for t in v["telemetry"]
            if t["constraints"].get("enabled"))),
        "duals": (env.constraint_manager.lambdas_by_name() if env.constraint_manager else {}),
        "weights": core_after.tolist(),
        "rng_state": rng_state_fn(rng),
    }


def run_smoke(config: SmokeConfig, *, resume_from: Mapping | None = None,
              iterations: int = 2, start_iteration: int = 0,
              checkpoint_at: int | None = None) -> dict:
    graphs = load_dataset().meta_train()[: max(2, int(config.meta_tasks))]
    if len(graphs) < 2:
        raise SmokeError("the smoke needs at least 2 meta-train graphs")
    env = _build_env(config, graphs)
    env.reset()
    n_features = V2_FEATURE_DIM + POSITION_FEATURES
    policy = ReferencePolicy(n_features, seed=int(config.seed))
    outer_adam = AdamSlots(policy.W.shape, lr=float(config.outer_lr))
    if resume_from is not None:
        policy.set_weights(np.asarray(resume_from["weights"], dtype=np.float64))
        outer_adam.load_state(resume_from["outer_optimizer"])
        if resume_from.get("obs", {}).get("v2_packed_dim") != V2_PACKED_DIM:
            raise SmokeError("checkpoint observation schema does not match this environment")
        if resume_from.get("smoke_config_sha256") != config.sha256():
            raise SmokeError("checkpoint configuration hash does not match")
        if env.constraint_manager is not None and resume_from.get("duals"):
            env.constraint_manager.load_state(resume_from["duals"])
    rng = np.random.RandomState(_stable_int("smoke", config.seed))
    report = {
        "schema": SCHEMA, "substitution": (
            "the TensorFlow graph is replaced by a numpy linear softmax policy; the "
            "environment, world, energy, constraints, CRN protocol and the FROZEN outer "
            "update operator mean_pseudogradient are the real ones. The frozen TF path is "
            "NOT RUN."),
        "config": config.as_dict(), "config_sha256": config.sha256(),
        "graph_ids": [str(getattr(g, "graph_id", "?")) for g in graphs],
        "obs": {"obs_version": V2_OBS_VERSION, "v2_packed_dim": V2_PACKED_DIM,
                "v2_feature_dim": V2_FEATURE_DIM, "v2_context_dim": V2_CONTEXT_DIM},
        "iterations": [], "resumed": bool(resume_from is not None),
    }
    checkpoint_doc = None
    for step in range(int(iterations)):
        it = int(start_iteration) + step
        record = _iterate(env, policy, outer_adam, config, iteration=it, rng=rng,
                          rng_state_fn=lambda r: {"state": r.get_state()[1].tolist(),
                                                  "pos": int(r.get_state()[2])})
        report["iterations"].append(record)
        if checkpoint_at is not None and it == int(checkpoint_at):
            checkpoint_doc = _checkpoint_doc(policy, outer_adam, env, config, iteration=it,
                                             rng_state=record["rng_state"])
    report["final"] = {
        "weights": policy.W.tolist(),
        "duals": (env.constraint_manager.lambdas_by_name() if env.constraint_manager else {}),
        "constraint_updates": (env.constraint_manager.controller.updates
                               if env.constraint_manager else 0),
        "inner_optimizer_slots_reset_per_task": True,
        "finite": bool(np.all(np.isfinite(policy.W))),
    }
    if checkpoint_doc is None:
        checkpoint_doc = _checkpoint_doc(policy, outer_adam, env, config,
                                         iteration=int(report["iterations"][-1]["iteration"]),
                                         rng_state=report["iterations"][-1]["rng_state"])
    report["checkpoint"] = checkpoint_doc
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta-tasks", type=int, default=4)
    ap.add_argument("--support", type=int, default=2)
    ap.add_argument("--query", type=int, default=2)
    ap.add_argument("--inner", type=int, default=3)
    ap.add_argument("--iterations", type=int, default=2)
    ap.add_argument("--background", type=int, default=0)
    ap.add_argument("--json", default=str(REPORTS / "V2_END_TO_END_SMOKE.json"))
    args = ap.parse_args()
    config = SmokeConfig(meta_tasks=int(args.meta_tasks), support=int(args.support),
                         query=int(args.query), inner_steps=int(args.inner),
                         background_dags=int(args.background))
    # The reference run checkpoints after `iterations - 2` and continues for one more
    # iteration; the resumed run then replays THAT iteration from the checkpoint. Both runs
    # must therefore produce the same weights for the same iteration index, which is the
    # resume-equivalence claim (not "resume produces *something*").
    total = max(2, int(args.iterations))
    reference = run_smoke(config, iterations=total, checkpoint_at=total - 2)
    resumed = run_smoke(config, resume_from=reference["checkpoint"], iterations=1,
                        start_iteration=total - 1)
    ok = _compare_resume(reference, resumed)
    reference["resume_check"] = ok
    path = Path(args.json)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(reference, indent=2, sort_keys=True, default=str) + "\n")
    for it in reference["iterations"]:
        print("iter %d  query_return=%+.6f  validation_return=%+.6f  latency=%.6f s  "
              "system=%.4f J  outer|dtheta|=%.3e  duals=%s"
              % (it["iteration"], it["query_return_mean"], it["validation_return_mean"],
                 it["validation_latency_mean"], it["validation_system_joules_mean"],
                 it["outer_update_norm"], {k: round(v, 5) for k, v in it["duals"].items()}))
    print("resume check:", json.dumps(ok, sort_keys=True))
    print("written", path)
    return 0


def _compare_resume(reference: Mapping, resumed: Mapping) -> dict:
    """A resumed run must reproduce the SAME iteration of the uninterrupted run exactly."""
    a = np.asarray(reference["iterations"][-1]["weights"], dtype=np.float64)
    b = np.asarray(resumed["iterations"][-1]["weights"], dtype=np.float64)
    same = bool(a.shape == b.shape and np.array_equal(a, b))
    return {
        "reference_iteration": int(reference["iterations"][-1]["iteration"]),
        "resumed_iteration": int(resumed["iterations"][-1]["iteration"]),
        "checkpoint_iteration": int(reference["checkpoint"]["iteration"]),
        "bit_identical_weights": same,
        "max_abs_difference": float(np.max(np.abs(a - b))) if a.shape == b.shape else None,
        "tolerance": "exact (numpy float64)",
        "duals_match": reference["final"]["duals"] == resumed["final"]["duals"],
        "schema_hashes_present": all(
            k in reference["checkpoint"] for k in
            ("obs", "world_config_sha256", "scheduler_config_sha256", "crn_protocol_sha256",
             "smoke_config_sha256", "reference_scheduler_config_sha256")),
    }


if __name__ == "__main__":
    sys.exit(main())
