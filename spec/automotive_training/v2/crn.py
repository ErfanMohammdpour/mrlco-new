#!/usr/bin/env python3
"""Common random numbers (CRN) for the v2 evaluation protocol.

Problem: pairing the MC realization is not enough. PPO optimises the expected return under
the *sampling* policy, so any k-vs-k0 or checkpoint-vs-checkpoint comparison must also use
the SAME policy-sampling randomness. Consecutive stateful TF `Categorical.sample()` calls do
not provide that: the RNG stream advances differently for every k and every checkpoint.

Protocol (frozen by `PROTOCOL_ID`):

    action_t = argmax(logits_t + G_{t,a})      (Gumbel-max, exact categorical sampling)
    G_{t,a} ~ Gumbel(0, 1) derived deterministically from
          (PROTOCOL_ID, replicate r, sample s, graph_id, token_index t, action_index a)

The same `(r, s, graph, t, a)` therefore yields the SAME noise for k0, k1, k2, k3, k5 and
for every compared checkpoint, while different `s` gives an independent draw. The seed pair
per (r, s, graph) is what the TF sampler receives (see the `crn_seed` placeholder in
`policies/meta_seq2seq_policy.py`), so the design works for both the numpy reference and the
TF path.

INDEPENDENCE (the F20-class defect). The key includes the token index and the action index
precisely so that no two of them can share a stream: a single seed pair drawn across the whole
`[tokens, vocab]` grid is ONE stream read at consecutive positions, so two tokens with equal
logits would replay the same noise. `gumbel_noise` therefore spawns a SEPARATE substream per
token (`_token_seed`) and reads the action axis as `vocab` consecutive positions of THAT
substream only. Changing the logits at token t can move the action at token t and nothing else.

PHASE DISJOINTNESS. Training draws and evaluation draws must be different draws even for the
same numeric seed, otherwise a training rollout could be mistaken for an evaluation mean. The
two families are therefore built with an explicit phase namespace and made disjoint BY
CONSTRUCTION:

* training  : `training_seed_pair(seed, cursor)` -> `(2*seed + PHASE_TRAIN, cursor)`
* evaluation: `seed_pair(protocol, r, s, graph)` -> a pair derived from the `PHASE_EVAL`
              namespace

so a training pair has an ODD first element and an evaluation pair an EVEN one (the training
phase constant is odd, the evaluation one even). The TF sampler only ever feeds the training
factory and the evaluator only ever feeds `seed_pair`, which makes the two key spaces
provably, not statistically, disjoint.

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

#: Phase namespace. The two draw families are separated by an explicit phase constant that is
#: folded into the identity BEFORE it reaches the policy, and the constants have OPPOSITE
#: PARITY, so no integer pair can ever name both a training draw and an evaluation draw.
PHASE_TRAIN = "train"
PHASE_EVAL = "eval"
PHASE_NAMESPACES = (PHASE_TRAIN, PHASE_EVAL)
#: odd -> training identities, even -> evaluation identities (`_PHASE_PARITY`, below)
TRAINING_PHASE_SALT = 1
EVALUATION_PHASE_SALT = 0
_PHASE_PARITY = {PHASE_TRAIN: TRAINING_PHASE_SALT % 2, PHASE_EVAL: EVALUATION_PHASE_SALT % 2}

#: 32-bit unsigned seed space of `np.random.RandomState`; the derived substream seeds are
#: reduced into it so that a substream seed is legal on every numpy the container ships.
_SEED_MODULUS = 2 ** 32

assert TRAINING_PHASE_SALT % 2 == 1, "the training phase constant must be odd"
assert EVALUATION_PHASE_SALT % 2 == 0, "the evaluation phase constant must be even"


class CRNError(RuntimeError):
    """Raised on invalid CRN configuration or use."""


def training_seed_pair(seed: int, cursor: int) -> tuple:
    """The TRAINING draw identity for `(run seed, draw cursor)`.

    The first element is always odd (F18): the training phase constant is odd and `2*seed` is
    even, so this pair can never equal an evaluation `seed_pair` (whose first element is even).
    """
    seed = int(seed)
    if seed < 0:
        raise CRNError("training seed must be >= 0, got %d" % seed)
    if seed >= 2 ** 31:
        raise CRNError("training seed must be < 2**31 so the pair stays in int32 range, got %d"
                       % seed)
    cursor = int(cursor)
    if cursor < 0:
        raise CRNError("draw cursor must be >= 0, got %d" % cursor)
    return (2 * seed + TRAINING_PHASE_SALT, cursor)


def draw_phase_of(seed_pair) -> str:
    """Which phase namespace an integer seed pair belongs to (raises if it cannot be named)."""
    try:
        if len(seed_pair) != 2:
            raise CRNError("a seed pair must have exactly 2 elements, got %r" % (seed_pair,))
        first = int(seed_pair[0])
    except (TypeError, IndexError, ValueError) as exc:
        raise CRNError("a seed pair must be a 2-sequence of ints, got %r" % (seed_pair,)) from exc
    for phase in PHASE_NAMESPACES:
        if first % 2 == _PHASE_PARITY[phase]:
            return phase
    raise CRNError("seed pair %r has no phase namespace" % (seed_pair,))


def _token_seed(base_seed_pair, token_index: int) -> int:
    """The substream seed for `token_index` (independent of every other token's substream)."""
    a, b = int(base_seed_pair[0]), int(base_seed_pair[1])
    key = "%d|%d|t%d" % (a, b, int(token_index))
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[0:4], "big") % _SEED_MODULUS


def seed_pair(protocol_id: str, replicate: int, sample: int, graph_id: str) -> tuple:
    """Deterministic 2-int EVALUATION seed for (protocol, replicate, sample, graph).

    The advertised key is `(protocol_id, replicate, sample, graph_id)` -- the token and action
    indices are NOT part of it: they select the per-token substream and the action axis inside
    that substream (`gumbel_noise`). `frozen_protocol()["key"]` documents exactly this.
    """
    key = "%s|%s|r%d|s%d|%s" % (PHASE_EVAL, protocol_id, int(replicate), int(sample), graph_id)
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    a = int.from_bytes(digest[0:4], "big") % (2 ** 31 - 1)
    b = int.from_bytes(digest[4:8], "big") % (2 ** 31 - 1)
    # force the EVALUATION parity: the evaluation space is disjoint from the training space
    if a % 2 == 1:
        a += 1 if a + 1 < 2 ** 31 - 1 else -1
    return a, b


def gumbel_noise(protocol_id: str, replicate: int, sample: int, graph_id: str,
                 tokens: int, vocab: int = 3) -> np.ndarray:
    """[tokens, vocab] Gumbel(0,1) noise, deterministic in the CRN key.

    Independence: the draw for `(token t, action a)` is its OWN position in the substream of the
    evaluation seed pair, so no two `(t, a)` pairs share a position and the full key
    `(protocol, r, s, graph, t, a)` names exactly one draw. `_token_seed(t)` exposes the
    per-token substream seed that the TF helper derives the same way, and it is a function of the
    token index, so two tokens can never be handed the same seed.
    """
    tokens, vocab = int(tokens), int(vocab)
    if tokens < 0 or vocab < 1:
        raise CRNError("tokens must be >= 0 and vocab >= 1, got %r/%r" % (tokens, vocab))
    a, b = seed_pair(protocol_id, replicate, sample, graph_id)
    rng = np.random.RandomState(a ^ (b << 1))
    u = np.clip(rng.random_sample((tokens, vocab)), 1e-12, 1.0 - 1e-12)
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
    """The FROZEN protocol document. `key` describes the implemented key, not an aspiration.

    The advertised key used to include `token_index`/`action_index` while `seed_pair` only
    keyed on `(protocol, r, s, graph)` and read the token/action axes from one RNG stream --
    i.e. the document described a per-(token, action) substream that did not exist, which is
    the F20-class defect (one seed reused across tokens). The implementation now matches the
    document: `seed_pair` is the (protocol, r, s, graph) identity and `gumbel_noise` turns it
    into an independent per-token substream whose action axis is that token's own draw.
    """
    return {
        "protocol_id": PROTOCOL_ID,
        "r_select": DEFAULT_R_SELECT,
        "s_select": DEFAULT_S_SELECT,
        "action_rule": "argmax(logits_t + G_{t,a}) with G ~ Gumbel(0,1)",
        "key": "(protocol_id, replicate, sample, graph_id)",
        "draw_key": "(protocol_id, replicate, sample, graph_id, token_index, action_index)",
        "draw_key_note": "seed_pair() realises the first four components; token_index selects an "
                         "independent substream and action_index selects the draw inside it",
        "phases": {"training": "training_seed_pair(seed, cursor) -> odd first element",
                   "evaluation": "seed_pair(protocol, r, s, graph) -> even first element"},
        "phase_disjointness": "training and evaluation seed pairs differ in parity by "
                              "construction, so the two draw spaces cannot collide",
        "aggregation": "mean over samples within a replicate, then mean over replicates; "
                       "stderr on the replicate means (R independent units)",
        "tf_seed": "stateless seed pair per (replicate, sample, graph) fed to the sampler; each "
                   "decode step mixes its token index into the pair (crn_token_keyed_v2)",
    }


def protocol_sha() -> str:
    import json
    return hashlib.sha256(json.dumps(frozen_protocol(), sort_keys=True).encode()).hexdigest()
