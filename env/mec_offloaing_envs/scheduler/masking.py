"""⑥b — masked action distribution: reference semantics (numpy).

Single source of truth for how feasibility enters the policy distribution. The TF
graph mirrors exactly these rules; this module exists so the semantics are
testable without TensorFlow.

Rules
  1. mask BEFORE softmax: `masked = where(feasible, logits, NEG)` with
     NEG = -1e9 (never -inf: softmax/log-softmax gradients turn into NaN).
  2. probability of an infeasible action is exactly 0.
  3. log-probabilities for the PPO ratio come from the MASKED distribution, both
     old (rollout) and new (update). With a fixed mask, ratio == 1 by construction.
  4. entropy is computed over the VALID actions only.
  5. dead-end guard: if no action is feasible the mask is dropped (all allowed)
     and the event is counted; the distribution is never empty.

The mask itself must be a pure function of the state (observation / prefix), so
that rollout and update reconstruct the same mask.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

NEG_LARGE = -1e9
N_ACTIONS = 3

# --- mask mode -----------------------------------------------------------------
# "off"     : legacy behaviour, no masking anywhere (byte-exact reproduction)
# "static"  : mask derived from the obs v3 feasibility channels (state-only)
# "runtime" : static base AND an env-provided shield mask, stored with the batch
MASK_MODE_OFF = "off"
MASK_MODE_STATIC = "static"
MASK_MODE_RUNTIME = "runtime"
MASK_MODES = (MASK_MODE_OFF, MASK_MODE_STATIC, MASK_MODE_RUNTIME)
MASK_MODE_ENV_VAR = "MARGO_MASK_MODE"
DEFAULT_MASK_MODE = MASK_MODE_OFF


def resolve_mask_mode(value: Any = None, env: Any = None) -> str:
    """Resolve the mask mode from an explicit value, else $MARGO_MASK_MODE, else off."""
    if value is None:
        environ = os.environ if env is None else env
        value = environ.get(MASK_MODE_ENV_VAR, DEFAULT_MASK_MODE)
    mode = str(value).strip().lower() or DEFAULT_MASK_MODE
    if mode not in MASK_MODES:
        raise ValueError(
            "mask mode must be one of %s, got %r" % (", ".join(MASK_MODES), value)
        )
    return mode


def mask_mode_active(mode: Any = None) -> bool:
    return resolve_mask_mode(mode) != MASK_MODE_OFF


def intersect_masks(base: Any, shield: Any) -> np.ndarray:
    """AND of two masks that must agree on shape (no silent broadcasting)."""
    base = normalise_mask(base)
    shield = normalise_mask(shield)
    if base.shape != shield.shape:
        raise ValueError(
            "mask shapes differ: %s vs %s" % (base.shape, shield.shape)
        )
    return np.logical_and(base, shield)


def observation_mask(observations: Any, mode: Any = None) -> np.ndarray | None:
    """Static mask read from obs v3 feasibility channels, or None when mode is off.

    Ordering is (UE, MEC, HELPER) == action ids (0, 1, 2), matching
    `Location.to_action()`. Raises for obs v1/v2, which carry no feasibility
    channels: a missing mask must fail loudly, never degrade to "no mask".
    """
    resolved = resolve_mask_mode(mode)
    if resolved == MASK_MODE_OFF:
        return None
    from .encoder_obs import feasibility_channel_indices

    return mask_from_observation(observations, feasibility_channel_indices())


@dataclass(frozen=True)
class MaskedDistribution:
    """Masked categorical distribution for one decision point."""

    probabilities: np.ndarray      # [..., 3]
    log_probabilities: np.ndarray  # [..., 3]
    entropy: np.ndarray            # [...] over valid actions only
    valid_count: np.ndarray        # [...]
    all_invalid: np.ndarray        # [...] bool, True => guard fired

    @property
    def any_all_invalid(self) -> bool:
        return bool(np.any(self.all_invalid))


def normalise_mask(feasible: Any, n_actions: int = N_ACTIONS) -> np.ndarray:
    feasible = np.asarray(feasible, dtype=bool)
    if feasible.shape[-1] != n_actions:
        raise ValueError("mask last dimension must be %d" % n_actions)
    return feasible


def dead_end_guard(feasible: Any) -> tuple[np.ndarray, np.ndarray]:
    """Drop the mask where nothing is feasible. Returns (mask, all_invalid)."""
    feasible = normalise_mask(feasible)
    all_invalid = ~feasible.any(axis=-1)
    guarded = np.where(all_invalid[..., None], True, feasible)
    return guarded, all_invalid


def apply_mask(logits: Any, feasible: Any, neg: float = NEG_LARGE) -> np.ndarray:
    """masked = where(feasible, logits, neg); dead-end rows are left unmasked."""
    logits = np.asarray(logits, dtype=np.float64)
    if logits.shape[-1] != N_ACTIONS:
        raise ValueError("logits last dimension must be %d" % N_ACTIONS)
    guarded, _all_invalid = dead_end_guard(feasible)
    if logits.shape[:-1] != guarded.shape[:-1]:
        raise ValueError("logits and mask batch shapes differ")
    return np.where(guarded, logits, float(neg))


def masked_softmax(logits: Any, feasible: Any, neg: float = NEG_LARGE) -> np.ndarray:
    masked = apply_mask(logits, feasible, neg=neg)
    shifted = masked - masked.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.maximum(exp.sum(axis=-1, keepdims=True), 1e-300)


def masked_log_softmax(logits: Any, feasible: Any, neg: float = NEG_LARGE) -> np.ndarray:
    masked = apply_mask(logits, feasible, neg=neg)
    shifted = masked - masked.max(axis=-1, keepdims=True)
    log_z = np.log(np.maximum(np.exp(shifted).sum(axis=-1, keepdims=True), 1e-300))
    return shifted - log_z


def entropy_valid(logits: Any, feasible: Any, neg: float = NEG_LARGE) -> np.ndarray:
    """Entropy over the valid set: -sum_{a valid} p log p (in nats)."""
    guarded, _ = dead_end_guard(feasible)
    p = masked_softmax(logits, feasible, neg=neg)
    safe = np.where(p > 0.0, p, 1.0)
    terms = np.where(guarded, -p * np.log(safe), 0.0)
    return terms.sum(axis=-1)


def distribution(logits: Any, feasible: Any, neg: float = NEG_LARGE) -> MaskedDistribution:
    guarded, all_invalid = dead_end_guard(feasible)
    p = masked_softmax(logits, feasible, neg=neg)
    logp = masked_log_softmax(logits, feasible, neg=neg)
    return MaskedDistribution(
        probabilities=p,
        log_probabilities=logp,
        entropy=entropy_valid(logits, feasible, neg=neg),
        valid_count=guarded.sum(axis=-1).astype(np.int64),
        all_invalid=all_invalid,
    )


def sample(logits: Any, feasible: Any, rng: np.random.RandomState, neg: float = NEG_LARGE) -> np.ndarray:
    """Sample actions from the masked distribution (never an infeasible action)."""
    p = masked_softmax(logits, feasible, neg=neg)
    flat_p = p.reshape(-1, p.shape[-1])
    draws = np.array([rng.choice(flat_p.shape[-1], p=row) for row in flat_p])
    return draws.reshape(p.shape[:-1])


def likelihood_ratio(
    action: Any,
    old_logits: Any,
    new_logits: Any,
    feasible: Any,
    new_feasible: Any | None = None,
) -> np.ndarray:
    """exp(new_logp - old_logp) from the MASKED distributions.

    `feasible` is the mask stored with the batch (used for the rollout log-prob).
    `new_feasible` defaults to it; pass a different mask only to study the hazard
    of recomputing the mask at update time (ratio collapses for actions that
    became infeasible).
    """
    action = np.asarray(action, dtype=np.int64)
    old_logp = masked_log_softmax(old_logits, feasible)[..., :]
    new_logp = masked_log_softmax(
        new_logits, feasible if new_feasible is None else new_feasible
    )[..., :]
    idx = action[..., None]
    return np.exp(np.take_along_axis(new_logp, idx, axis=-1) - np.take_along_axis(old_logp, idx, axis=-1)).squeeze(-1)


def mask_from_observation(packed: Any, feasible_indices: Sequence[int]) -> np.ndarray:
    """Static feasibility mask read from the obs v3 feasibility channels.

    Deterministic function of the observation only, which is what makes it safe
    to use both at rollout and at update time without extra plumbing.
    """
    packed = np.asarray(packed, dtype=np.float64)
    if packed.ndim < 2:
        raise ValueError("packed obs must be at least 2-D")
    idx = list(feasible_indices)
    if len(idx) != N_ACTIONS:
        raise ValueError("expected %d feasibility channels, got %d" % (N_ACTIONS, len(idx)))
    return np.stack([packed[..., i] > 0.5 for i in idx], axis=-1)
