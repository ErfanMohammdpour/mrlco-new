#!/usr/bin/env python3
"""Common random numbers (CRN) for the v2 evaluation protocol.

Problem: pairing the MC realization is not enough. PPO optimises the expected return under
the *sampling* policy, so any k-vs-k0 or checkpoint-vs-checkpoint comparison must also use
the SAME policy-sampling randomness. Consecutive stateful TF `Categorical.sample()` calls do
not provide that: the RNG stream advances differently for every k and every checkpoint.

Protocol (frozen by `PROTOCOL_ID`):

    action_t = argmax(logits_t + G_t)          (Gumbel-max, exact categorical sampling)
    G_t ~ Gumbel(0, 1) derived deterministically from
          (PROTOCOL_ID, replicate r, sample s, graph_id, token_index t, action a)

The same `(r, s, graph, t, a)` therefore yields the SAME noise for k0, k1, k2, k3, k5 and
for every compared checkpoint, while different `s` gives an independent draw. The seed pair
per (r, s, graph) is what the TF sampler receives (see the `crn_seed` placeholder in
`policies/meta_seq2seq_policy.py`), so the design works for both the numpy reference and the
TF path.

Nested sampling: `S` stochastic samples are nested inside each of the `R` MC realizations,
so aggregation must first average over `s` within a replicate and then average the replicate
means; the standard error is computed on the replicate means (R independent units).
"""

from __future__ import annotations

import hashlib
import math
import statistics
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

PROTOCOL_ID = "automotive_crn_gumbel_v1"
DEFAULT_R_SELECT = 5
DEFAULT_S_SELECT = 5


class CRNError(RuntimeError):
    """Raised on invalid CRN configuration or use."""


def seed_pair(protocol_id: str, replicate: int, sample: int, graph_id: str) -> tuple:
    """Deterministic 2-int seed for (protocol, replicate, sample, graph)."""
    key = "%s|r%d|s%d|%s" % (protocol_id, int(replicate), int(sample), graph_id)
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    a = int.from_bytes(digest[0:4], "big") % (2 ** 31 - 1)
    b = int.from_bytes(digest[4:8], "big") % (2 ** 31 - 1)
    return a, b


def gumbel_noise(protocol_id: str, replicate: int, sample: int, graph_id: str,
                 tokens: int, vocab: int = 3) -> np.ndarray:
    """[tokens, vocab] Gumbel(0,1) noise, deterministic in the CRN key."""
    a, b = seed_pair(protocol_id, replicate, sample, graph_id)
    rng = np.random.RandomState(a ^ (b << 1))
    u = np.clip(rng.random_sample((int(tokens), int(vocab))), 1e-12, 1.0 - 1e-12)
    return -np.log(-np.log(u))


def gumbel_argmax(logits: np.ndarray, noise: np.ndarray, *, mask: np.ndarray | None = None) -> np.ndarray:
    """action_t = argmax(logits_t + G_t) with optional -inf masking."""
    logits = np.asarray(logits, dtype=np.float64)
    noise = np.asarray(noise, dtype=np.float64)
    if logits.shape != noise.shape:
        raise CRNError("logits %s and noise %s shapes differ" % (logits.shape, noise.shape))
    scores = logits + noise
    if mask is not None:
        scores = np.where(np.asarray(mask) > 0.0, scores, -np.inf)
    return np.argmax(scores, axis=-1)


def categorical_probabilities(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    z = z - np.max(z, axis=-1, keepdims=True)
    p = np.exp(z)
    return p / np.sum(p, axis=-1, keepdims=True)


@dataclass
class NestedAggregate:
    replicates: int
    samples_per_replicate: int
    mean: float
    stderr: float
    per_replicate_means: list
    per_sample_means: list

    def as_dict(self) -> dict:
        return {"replicates": self.replicates, "samples_per_replicate": self.samples_per_replicate,
                "mean": self.mean, "stderr": self.stderr,
                "per_replicate_means": list(self.per_replicate_means),
                "per_sample_means": list(self.per_sample_means)}


def nested_aggregate(values: Sequence[Sequence[float]]) -> NestedAggregate:
    """`values[r][s]` -> mean over samples within a replicate, then over replicates."""
    if not values or not values[0]:
        raise CRNError("nested_aggregate needs a non-empty R x S grid")
    widths = {len(row) for row in values}
    if len(widths) != 1:
        raise CRNError("ragged nested samples: %s" % sorted(widths))
    per_rep = [statistics.fmean(row) for row in values]
    per_sample = [statistics.fmean([row[s] for row in values]) for s in range(len(values[0]))]
    mean = statistics.fmean(per_rep)
    stderr = (statistics.stdev(per_rep) / math.sqrt(len(per_rep))) if len(per_rep) > 1 else 0.0
    return NestedAggregate(replicates=len(values), samples_per_replicate=len(values[0]),
                           mean=mean, stderr=stderr, per_replicate_means=per_rep,
                           per_sample_means=per_sample)


def paired_delta(a: Sequence[float], b: Sequence[float]) -> dict:
    """Paired per-replicate difference a-b (same CRN key) with its own standard error."""
    if len(a) != len(b):
        raise CRNError("paired_delta needs equal lengths")
    diffs = [float(x) - float(y) for x, y in zip(a, b)]
    mean = statistics.fmean(diffs) if diffs else 0.0
    stderr = (statistics.stdev(diffs) / math.sqrt(len(diffs))) if len(diffs) > 1 else 0.0
    return {"mean": mean, "stderr": stderr, "n": len(diffs), "deltas": diffs}


def frozen_protocol() -> dict:
    return {
        "protocol_id": PROTOCOL_ID,
        "r_select": DEFAULT_R_SELECT,
        "s_select": DEFAULT_S_SELECT,
        "action_rule": "argmax(logits_t + G_t) with G ~ Gumbel(0,1)",
        "key": "(protocol_id, replicate, sample, graph_id, token_index, action_index)",
        "aggregation": "mean over samples within a replicate, then mean over replicates; "
                       "stderr on the replicate means (R independent units)",
        "tf_seed": "stateless seed pair per (replicate, sample, graph) fed to the sampler",
    }


def protocol_sha() -> str:
    import json
    return hashlib.sha256(json.dumps(frozen_protocol(), sort_keys=True).encode()).hexdigest()
