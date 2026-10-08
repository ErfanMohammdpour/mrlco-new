#!/usr/bin/env python3
"""First-divergence tracing for fresh-process continuation (F33).

Two uninterrupted controls on the fixed CPU configuration are BIT-IDENTICAL (measured: 0.0000%
difference in every per-iteration record, identical lambdas to 15 significant digits). Any
control-vs-resume difference is therefore a real state-restoration defect, not backend noise, and
it can be localised by fingerprinting state at successive boundaries and diffing.

Boundaries, in the order they are consumed:
  B1 saved/restored  -- every TF variable (name -> value digest), the dual state, the counters,
                        the environment identity (primary + every clone), and every RNG stream
  B2 worlds          -- per slot: graph/order identity, MC survivors, helper contact, link trace,
                        background DAGs, deadlines, realization identity
  B3 observations    -- the packed observation array
  B4 inference       -- logits, values, seed pairs, sampled actions
  B5 optimization    -- old likelihoods/values, advantages/returns, minibatch order, inner/outer
  B6 costs           -- per-episode energy, signed costs, applied penalty, dual update

`digest` is a sha256 over a canonical byte representation, so equal digests mean equal values.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

SCHEMA = "v2_state_trace_v1"


def digest(obj) -> str:
    """Stable sha256 over a value's canonical bytes (floats via their exact repr)."""
    h = hashlib.sha256()
    _feed(h, obj, 0)
    return h.hexdigest()[:32]


def _feed(h, obj, depth):
    if depth > 8:
        h.update(b"<depth>")
        return
    if obj is None:
        h.update(b"<none>")
    elif isinstance(obj, bool):
        h.update(b"true" if obj else b"false")
    elif isinstance(obj, int):
        h.update(b"i" + str(int(obj)).encode())
    elif isinstance(obj, float):
        h.update(b"f" + (repr(float(obj)) if math.isfinite(obj) else str(obj)).encode())
    elif isinstance(obj, str):
        h.update(b"s" + obj.encode("utf-8", "replace"))
    elif isinstance(obj, (bytes, bytearray)):
        h.update(b"b" + bytes(obj))
    elif isinstance(obj, dict):
        h.update(b"{")
        for k in sorted(obj, key=str):
            _feed(h, k, depth + 1)
            _feed(h, obj[k], depth + 1)
        h.update(b"}")
    elif isinstance(obj, (list, tuple)):
        h.update(b"[")
        for item in obj:
            _feed(h, item, depth + 1)
        h.update(b"]")
    else:
        # numpy arrays and anything else with tobytes()
        tb = getattr(obj, "tobytes", None)
        if callable(tb):
            h.update(b"a" + str(getattr(obj, "shape", None)).encode())
            try:
                h.update(tb())
            except Exception:
                h.update(b"<unreadable>")
        else:
            h.update(b"r" + repr(obj).encode("utf-8", "replace"))


# --------------------------------------------------------------------------- #
# B1: saved/restored state
# --------------------------------------------------------------------------- #
_SLOT_TAGS = ("beta1_power", "beta2_power", "Adam", "adam", "/m", "/v")


def all_variable_digests(sess) -> tuple:
    """(every variable's digest, the slot subset) in ONE batched session fetch.

    A per-variable `sess.run` costs ~1000 round trips per boundary, which made a traced run
    unusably slow (13 minutes without finishing one iteration). One fetch of the whole list
    keeps the trace cheap enough to leave enabled.
    """
    import tensorflow as tf
    graph = tf.compat.v1.get_default_graph()
    variables = list(graph.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES))
    if not variables:
        return {}, {}
    try:
        values = sess.run(variables)
    except Exception:                                              # pragma: no cover
        values = [None] * len(variables)
    every: dict = {}
    slots: dict = {}
    for v, value in zip(variables, values):
        d = "ERROR:unfetchable" if value is None else digest(value)
        every[v.name] = d
        if any(tag in v.name for tag in _SLOT_TAGS):
            slots[v.name] = d
    every["<count>"] = len(variables)
    return every, slots


def tf_variable_digests(sess) -> dict:
    """name -> value digest for EVERY global variable (weights, slots, beta powers)."""
    return all_variable_digests(sess)[0]


def optimiser_slot_digests(sess) -> dict:
    """Only the Adam slots and beta powers, so a slot-specific defect is visible at a glance."""
    return all_variable_digests(sess)[1]


def rng_digests(trainer) -> dict:
    import numpy as np
    out = {}
    algo = getattr(trainer, "algo", None)
    rng = getattr(algo, "rng", None) if algo is not None else None
    if rng is not None and hasattr(rng, "get_state"):
        st = rng.get_state()
        out["algo.rng"] = digest((str(st[0]), st[1].tolist(), int(st[2]), int(st[3]),
                                  float(st[4])))
    st = np.random.get_state()
    out["numpy.global"] = digest((str(st[0]), st[1].tolist(), int(st[2]), int(st[3]),
                                 float(st[4])))
    try:
        import random as _random
        out["python.random"] = digest(repr(_random.getstate()))
    except Exception:                                              # pragma: no cover
        pass
    return out


def env_identity(env) -> dict:
    """Primary env + the base env's realization state."""
    if env is None:
        return {}
    base = getattr(env, "base", None)
    return {
        "episodes": getattr(env, "episodes", None),
        "world_key": getattr(env, "_world_key", None),
        "reset_count": getattr(base, "reset_count", None) if base is not None else None,
        "realization_epoch": (getattr(base, "realization_epoch", None)
                              if base is not None else None),
        "task_id": getattr(base, "task_id", None) if base is not None else None,
        "graph_indices": (list(map(int, getattr(base, "graph_indices", []) or []))
                          if base is not None else None),
        "policy_task_id": getattr(env, "task_id", None),
    }


def clone_identity(trainer) -> list:
    """Per-clone counter state, positionally (the workers reset independently)."""
    from spec.automotive_training.v2.run_tf import _sampler_clone_envs
    out = []
    for c in _sampler_clone_envs(trainer):
        base = getattr(c, "base", None)
        out.append({
            "episodes": getattr(c, "episodes", None),
            "reset_count": getattr(base, "reset_count", None) if base is not None else None,
            "realization_epoch": (getattr(base, "realization_epoch", None)
                                  if base is not None else None),
            "task_id": getattr(base, "task_id", None) if base is not None else None,
        })
    return out


def trainer_counters(trainer) -> dict:
    keys = ("started_iterations", "completed_iterations", "last_completed_iteration",
            "next_iteration", "outer_updates_completed", "dual_updates_completed",
            "validations_completed", "current_iteration", "best_val_objective",
            "auto_penalty_episodes", "auto_penalty_episodes_nonzero", "auto_penalty_sum",
            "auto_v2_signed_rows", "auto_lambda_broadcast_targets")
    out = {}
    for k in keys:
        v = getattr(trainer, k, None)
        out[k] = v if isinstance(v, (int, float, str, bool, type(None))) else repr(v)
    return out


def dual_state(trainer) -> dict:
    m = getattr(trainer, "auto_v2_constraint_manager", None)
    if m is None:
        return {}
    out = {"lambdas": m.lambdas_by_name(),
           "updates": int(getattr(m.controller, "updates", 0))}
    batch = getattr(m.controller, "batch", None)
    if batch is not None:
        try:
            out["batch_size"] = len(batch)
            out["batch"] = list(batch)
        except Exception:                                          # pragma: no cover
            out["batch"] = repr(batch)
    return out


def sampler_counters(trainer) -> dict:
    for attr in ("sampler", "meta_sampler"):
        s = getattr(trainer, attr, None)
        c = getattr(s, "counters", None)
        if c:
            return {str(k): int(v) for k, v in dict(c).items()}
    return {}


def current_session():
    """The default TF session, so the training loop can trace without holding a handle."""
    try:
        import tensorflow as tf
        return tf.compat.v1.get_default_session()
    except Exception:                                              # pragma: no cover
        return None


def boundary1_default(trainer) -> dict:
    sess = current_session()
    if sess is None:
        return {"boundary": "B1_saved_restored", "error": "no default session"}
    return boundary1(sess, trainer)


def boundary1(sess, trainer) -> dict:
    if sess is None:
        every, slots = {}, {}
    else:
        every, slots = all_variable_digests(sess)
    return {
        "boundary": "B1_saved_restored",
        "tf.variables": every,
        "tf.slots": slots,
        "trainer": trainer_counters(trainer),
        "dual": dual_state(trainer),
        "env": env_identity(getattr(trainer, "env", None)),
        "clones": clone_identity(trainer),
        "rng": rng_digests(trainer),
        "sampler": sampler_counters(trainer),
    }


# --------------------------------------------------------------------------- #
# B2: the immutable worlds the next iteration will see
# --------------------------------------------------------------------------- #
def _world_trace(world) -> dict:
    fg = getattr(world, "foreground", None)
    out = {
        "world_id": getattr(world, "world_id", None),
        "dataset_graph_id": getattr(world, "dataset_graph_id", None),
        "fingerprint": (world.fingerprint_sha256()
                        if hasattr(world, "fingerprint_sha256") else None),
        "foreground_dag": getattr(fg, "dag_id", None),
    }
    if fg is not None:
        tasks = list(getattr(fg, "tasks", []) or [])
        out["fg_tasks"] = [(int(t.task_id), float(t.compute_bytes), float(t.output_bytes),
                            int(t.owner), str(t.criticality), t.deadline_s) for t in tasks]
        out["fg_predecessors"] = {int(t.task_id): [int(p) for p in (t.predecessors or [])]
                                  for t in tasks}
    bg = list(getattr(world, "background_ids", []) or [])
    out["background_ids"] = sorted(str(b) for b in bg)
    helpers = getattr(world, "helpers", None) or {}
    out["helpers"] = {int(k): {"cpu": float(getattr(v, "cpu_bytes_per_s", 0.0)),
                               "busy_until": float(getattr(v, "busy_until_s", 0.0)),
                               "contact_start": float(getattr(v, "contact_start_s", 0.0)),
                               "contact_end": float(getattr(v, "contact_end_s", 0.0))}
                      for k, v in dict(helpers).items()}
    return out


def boundary2(env, link_process, slot_bounds=None) -> dict:
    """Per-slot world identity. `link_process` contributes its realized trace digest."""
    out = {"boundary": "B2_worlds", "slots": []}
    if env is None:
        return out
    n = len(getattr(env, "last_v2_context", []) or []) or 0
    for flat in range(n):
        try:
            world = env.world_for_slot(flat)
        except Exception as exc:
            out["slots"].append({"slot": flat, "error": "%s" % (type(exc).__name__,)})
            continue
        rec = _world_trace(world)
        rec["slot"] = flat
        mc = None
        try:
            mc = env.base._slot_mc[flat] if env.base._slot_mc else None
        except Exception:                                          # pragma: no cover
            pass
        if mc is not None:
            rec["mc"] = {"final_mode": mc.get("final_mode"),
                         "executed": sorted(int(t) for t in mc.get("executed_task_ids", [])),
                         "dropped": sorted(int(t) for t in mc.get("dropped_task_ids", [])),
                         "capped_to_hi": sorted(int(t) for t in mc.get("capped_to_hi", [])),
                         "high_preserved": bool(mc.get("high_preserved"))}
        out["slots"].append(rec)
    if link_process is not None:
        links = getattr(link_process, "links", ())
        out["link_trace"] = {str(l): digest(link_process._realized[l]) for l in links
                             if hasattr(link_process, "_realized") and l in link_process._realized}
        out["link_horizon_s"] = (link_process.horizon_s()
                                 if hasattr(link_process, "horizon_s") else None)
    return out


def boundary3(obs) -> dict:
    import numpy as np
    arr = np.asarray(obs)
    return {"boundary": "B3_observations", "shape": list(arr.shape),
            "digest": digest(arr), "sum": float(arr.sum()), "absmax": float(np.abs(arr).max())
            if arr.size else 0.0}


def boundary4(logits=None, values=None, seeds=None, actions=None, extra=None) -> dict:
    out = {"boundary": "B4_inference"}
    if logits is not None:
        out["logits"] = digest(logits)
    if values is not None:
        out["values"] = digest(values)
    if seeds is not None:
        out["seed_pairs"] = digest(seeds)
    if actions is not None:
        out["actions"] = digest(actions)
    if extra:
        out.update(extra)
    return out


def boundary6(env, trainer) -> dict:
    """Per-episode cost/energy records plus the dual state after the update."""
    out = {"boundary": "B6_costs"}
    ledgers = list(getattr(env, "last_foreground_ledger", []) or [])
    out["fg_ledgers"] = [{"system": float(getattr(x, "system_joules", 0.0)),
                          "requester": float(getattr(x, "requester_joules", 0.0))}
                         for x in ledgers if x is not None]
    tel = list(getattr(env, "last_telemetry", []) or [])
    out["telemetry"] = [{"penalty": float(getattr(t, "constraint_penalty", 0.0)),
                         "system_joules": float(getattr(t, "system_joules", 0.0)),
                         "violations": {str(k): float(v) for k, v in
                                        dict(getattr(t, "constraint_violations", {}) or {}).items()}}
                        for t in tel]
    out["dual"] = dual_state(trainer)
    costs = getattr(env, "last_constraint_costs", None)
    if costs is not None and getattr(costs, "active", False):
        out["costs"] = {str(k): float(v) for k, v in dict(costs.as_dict()).items()}
    return out


# --------------------------------------------------------------------------- #
class StateTracer:
    """Append boundary records to a JSONL file; disabled unless a path is given."""

    def __init__(self, path):
        self.path = Path(path) if path else None
        self.written = 0

    def record(self, label, payload):
        if self.path is None:
            return
        rec = {"schema": SCHEMA, "label": str(label),
               "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "payload": payload}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as fh:
            fh.write(json.dumps(rec, sort_keys=True, default=str) + "\n")
        self.written += 1
