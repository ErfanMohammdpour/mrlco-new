"""⑥b metric core: mask / policy / critic rates over rollout tokens.

Pure numpy so the definitions are testable without TensorFlow. The TF side only
supplies four rollout arrays and never recomputes anything:

    pre_guard   the feasibility mask STORED during rollout   [N, 3] bool
    actions     sampled actions from the same sess.run       [N]    int
    raw_logits  UNMASKED decoder logits from that sess.run   [N, 3] float
    values      sample_vf from that sess.run                 [N]    float

Definitions (frozen by the audit):
    valid_count = pre_guard.sum(-1)
    dead_end    = valid_count == 0
    post_guard  = where(dead_end, True, pre_guard)

    mask/active_rate           = mean(any(~pre_guard, axis=-1))
    mask/forced_rate           = mean(valid_count == 1)
    mask/all_invalid_rate      = mean(valid_count == 0)      # BEFORE the guard
    policy/invalid_action_rate = mean(action outside post_guard)
    policy/argmax_masked_rate  = count(valid_count > 0 and raw argmax is closed
                                       by pre_guard) / total_tokens
    policy/entropy_valid       = mean entropy over the post_guard support (nats)
    critic/value_abs_max       = max |values| over the iteration

Conventions that are easy to get wrong, so they are stated once:
  * dead-end rows are excluded from argmax_masked_rate's numerator (the guard
    deliberately drops the shield there) but still count in the denominator,
    which is every token;
  * a dead-end row is not an invalid action: post_guard is all-True, so the
    sampled action is always valid and invalid_action_rate cannot punish it;
  * entropy is never normalised by log(valid_count);
  * argmax comes from the RAW logits: a masked argmax is unrecoverable;
  * in `off` mode there is no mask, the five control rates are defined as 0.0,
    and entropy/value are still real.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from .masking import N_ACTIONS, entropy_valid

METRIC_KEYS = (
    "mask/active_rate",
    "mask/forced_rate",
    "mask/all_invalid_rate",
    "policy/invalid_action_rate",
    "policy/argmax_masked_rate",
    "policy/entropy_valid",
    "critic/value_abs_max",
)

# the five rates that must be exactly 0.0 in off mode / no-deadline runs
RATE_KEYS = METRIC_KEYS[:5]


@dataclass(frozen=True)
class MaskMetricAccumulator:
    """Token counts, not averages: merging paths/tasks stays exact."""

    tokens: int = 0
    active: int = 0
    forced: int = 0
    all_invalid: int = 0
    invalid_actions: int = 0
    argmax_masked: int = 0
    entropy_sum: float = 0.0
    value_abs_max: float = 0.0

    def __add__(self, other: "MaskMetricAccumulator") -> "MaskMetricAccumulator":
        if not isinstance(other, MaskMetricAccumulator):
            return NotImplemented
        return MaskMetricAccumulator(
            tokens=self.tokens + other.tokens,
            active=self.active + other.active,
            forced=self.forced + other.forced,
            all_invalid=self.all_invalid + other.all_invalid,
            invalid_actions=self.invalid_actions + other.invalid_actions,
            argmax_masked=self.argmax_masked + other.argmax_masked,
            entropy_sum=self.entropy_sum + other.entropy_sum,
            value_abs_max=max(self.value_abs_max, other.value_abs_max),
        )

    def finalize(self) -> dict[str, float]:
        if self.tokens <= 0:
            return {key: 0.0 for key in METRIC_KEYS}
        n = float(self.tokens)
        return {
            "mask/active_rate": self.active / n,
            "mask/forced_rate": self.forced / n,
            "mask/all_invalid_rate": self.all_invalid / n,
            "policy/invalid_action_rate": self.invalid_actions / n,
            "policy/argmax_masked_rate": self.argmax_masked / n,
            "policy/entropy_valid": self.entropy_sum / n,
            "critic/value_abs_max": float(self.value_abs_max),
        }


def merge(accumulators: Iterable[MaskMetricAccumulator]) -> MaskMetricAccumulator:
    total = MaskMetricAccumulator()
    for acc in accumulators:
        if acc is None:
            continue
        total = total + acc
    return total


def _as_tokens(array: Any, name: str, width: int | None = None) -> np.ndarray:
    arr = np.asarray(array)
    if arr.size == 0:
        raise ValueError("%s is empty: nothing to aggregate" % name)
    if width is not None and arr.shape[-1] != width:
        raise ValueError(
            "%s last dimension must be %d, got %s" % (name, width, arr.shape[-1])
        )
    return arr


def accumulate(
    pre_guard: Any = None,
    actions: Any = None,
    raw_logits: Any = None,
    values: Any = None,
    *,
    require_mask: bool = False,
) -> MaskMetricAccumulator:
    """Fold one rollout batch (any leading shape) into a token accumulator.

    `require_mask=True` (an active mask mode) makes a missing or mis-shaped mask
    a hard error instead of silently reporting zeros.
    """
    if raw_logits is None:
        raise ValueError(
            "raw_logits are required: capture sample_decoder_logits_raw in the "
            "rollout sess.run, the masked logits cannot be inverted"
        )
    raw = _as_tokens(raw_logits, "raw_logits", N_ACTIONS).astype(np.float64)
    flat_raw = raw.reshape(-1, N_ACTIONS)
    actions = _as_tokens(actions, "actions")
    if actions.shape != raw.shape[:-1]:
        raise ValueError(
            "actions shape %s != logits shape %s (drop the action axis)"
            % (actions.shape, raw.shape[:-1])
        )
    flat_actions = actions.reshape(-1).astype(np.int64)
    values = _as_tokens(values, "values")
    if values.shape != actions.shape:
        raise ValueError(
            "values shape %s != actions shape %s" % (values.shape, actions.shape)
        )
    flat_values = values.reshape(-1).astype(np.float64)

    if pre_guard is None:
        if require_mask:
            raise ValueError(
                "mask mode is active but the batch has no feasible mask; the "
                "rollout mask must be stored, never recomputed"
            )
        guard = np.ones(flat_raw.shape, dtype=bool)
        masked_mode = False
    else:
        raw_guard = _as_tokens(pre_guard, "pre_guard", N_ACTIONS)
        # validate BEFORE astype(bool): NaN/inf silently become True, and a
        # non-binary value would be silently reinterpreted as "open"
        if not np.all(np.isfinite(raw_guard)):
            raise ValueError("pre_guard contains non-finite values")
        if not np.all((raw_guard == 0) | (raw_guard == 1)):
            raise ValueError(
                "pre_guard must contain only 0/1 or bool values"
            )
        guard = raw_guard.astype(bool)
        if guard.shape != raw.shape:
            raise ValueError(
                "mask shape %s != logits shape %s" % (guard.shape, raw.shape)
            )
        guard = guard.reshape(-1, N_ACTIONS)
        masked_mode = True

    if not np.all(np.isfinite(flat_raw)):
        raise ValueError("raw_logits contain non-finite values")
    if not np.all(np.isfinite(flat_values)):
        raise ValueError("values contain non-finite values")
    if flat_actions.min() < 0 or flat_actions.max() >= N_ACTIONS:
        raise ValueError("actions out of range for %d actions" % N_ACTIONS)

    valid_count = guard.sum(axis=-1)
    dead_end = valid_count == 0
    post_guard = np.where(dead_end[:, None], True, guard)

    # sampled action validity is judged against the POST-guard support, so a
    # dead-end row (guard dropped) can never be counted as invalid
    rows = np.arange(flat_actions.shape[0])
    sampled_valid = post_guard[rows, flat_actions]
    invalid_actions = int((~sampled_valid).sum())

    raw_argmax = np.argmax(flat_raw, axis=-1)
    argmax_closed = ~guard[rows, raw_argmax]
    # denominator is every token; all-invalid rows are excluded from numerator
    argmax_masked = int((argmax_closed & ~dead_end).sum())

    entropy = entropy_valid(flat_raw, post_guard)
    if not np.all(np.isfinite(entropy)):
        raise ValueError("entropy is non-finite")

    return MaskMetricAccumulator(
        tokens=int(flat_actions.shape[0]),
        active=int((~guard).any(axis=-1).sum()) if masked_mode else 0,
        forced=int((valid_count == 1).sum()) if masked_mode else 0,
        all_invalid=int(dead_end.sum()) if masked_mode else 0,
        invalid_actions=invalid_actions,
        argmax_masked=argmax_masked if masked_mode else 0,
        entropy_sum=float(entropy.sum()),
        value_abs_max=float(np.max(np.abs(flat_values))),
    )


def rates(accumulator: MaskMetricAccumulator) -> dict[str, float]:
    """finalize() plus a non-finite guard: a bad metric must never be logged."""
    out = accumulator.finalize()
    bad = [k for k, v in out.items() if not np.isfinite(v)]
    if bad:
        raise ValueError("non-finite metrics: %s" % ", ".join(sorted(bad)))
    for key in METRIC_KEYS:
        if key not in out:
            raise ValueError("missing metric %s" % key)
    return out
