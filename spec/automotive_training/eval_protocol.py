#!/usr/bin/env python3
"""Frozen automotive evaluation protocol (v1).

Fixes the three protocol defects found in the 5x500 pilot evaluation:

1. **Pairing.** k=0, k=3 and every baseline must see the SAME mixed-criticality
   realization for the same graph. The realization seed is
   `hash(domain, base_seed, graph_id, slot, epoch)`, so the evaluator pins an explicit
   `realization_epoch` per replicate instead of letting `reset_count` (which differs
   between the k0 and k3 paths) and different base seeds decide the draw.
2. **Multi-realization.** A single realization per graph is a single sample. The frozen
   protocol uses `SELECT_REPLICATES = 5` realization pairs for in-training validation and
   checkpoint selection, and `REPORT_REPLICATES = 20` pairs for the final report.
3. **Deterministic decoding.** The primary inference for reported numbers is greedy
   (argmax) decoding over the same support adaptation; the stochastic policy is kept as a
   secondary robustness result.

The realization seeds are committed here BEFORE any campaign, so no seed can be chosen
after seeing a result.
"""

from __future__ import annotations

from spec.automotive_training.v2.compat import fmean  # noqa: E402
import hashlib
import json
from typing import Any, Mapping, Sequence

PROTOCOL_ID = "automotive_eval_protocol_v1"

#: in-training validation / checkpoint selection panel
SELECT_REPLICATES = 5
#: final report panel (also used for the adaptation sweep and the salvage re-evaluation)
REPORT_REPLICATES = 20

#: primary and secondary inference
PRIMARY_DECODING = "deterministic"
SECONDARY_DECODING = "stochastic"

#: support-adaptation budgets to sweep on the frozen checkpoints
ADAPTATION_K_STEPS = (0, 1, 2, 3, 5)

#: committed realization base seeds: seed_r = SEED_BASE + r * SEED_STRIDE
SEED_BASE = 1000
SEED_STRIDE = 17
REALIZATION_SEEDS = tuple(SEED_BASE + SEED_STRIDE * r for r in range(REPORT_REPLICATES))

BASELINE_CANDIDATES = ("all_UE", "all_MEC", "all_HELPER", "heft_reference_v2",
                       "greedy_coordinate_descent_MC")


class EvalProtocolError(RuntimeError):
    """Raised when a caller asks for a protocol variant that is not frozen."""


def realization_seeds(n_replicates: int) -> tuple[int, ...]:
    n = int(n_replicates)
    if n < 1 or n > REPORT_REPLICATES:
        raise EvalProtocolError(
            "replicates must be in 1..%d (frozen protocol), got %r" % (REPORT_REPLICATES, n))
    return REALIZATION_SEEDS[:n]


def decoding_flags(decoding: str) -> dict:
    if decoding == PRIMARY_DECODING:
        return {"greedy": True, "decoding": decoding}
    if decoding == SECONDARY_DECODING:
        return {"greedy": False, "decoding": decoding}
    raise EvalProtocolError("unknown decoding %r (frozen: %s/%s)"
                            % (decoding, PRIMARY_DECODING, SECONDARY_DECODING))


def frozen_protocol() -> dict:
    return {
        "protocol_id": PROTOCOL_ID,
        "select_replicates": SELECT_REPLICATES,
        "report_replicates": REPORT_REPLICATES,
        "primary_decoding": PRIMARY_DECODING,
        "secondary_decoding": SECONDARY_DECODING,
        "adaptation_k_steps": list(ADAPTATION_K_STEPS),
        "realization_seeds": list(REALIZATION_SEEDS),
        "seed_base": SEED_BASE,
        "seed_stride": SEED_STRIDE,
        "baseline_candidates": list(BASELINE_CANDIDATES),
        "pairing": ("same (base_seed, realization_epoch) for k0, k3 and every baseline in a "
                    "replicate; the epoch pins the MC draw independently of reset_count"),
        "aggregation": "mean/std over replicates; per-graph candidate-oracle regret reported",
    }


def protocol_sha() -> str:
    blob = json.dumps(frozen_protocol(), sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def candidate_oracle(makespans: Mapping[str, float]) -> tuple[str, float]:
    """Best candidate of the frozen panel: min over BASELINE_CANDIDATES."""
    present = [(float(makespans[name]), name) for name in BASELINE_CANDIDATES
               if name in makespans and makespans[name] is not None]
    if not present:
        raise EvalProtocolError("no baseline candidate available for the oracle")
    value, name = min(present)
    return name, value


def regret(model_makespan: float, makespans: Mapping[str, float]) -> float:
    """MARGO makespan minus the candidate oracle on the same graph+realization."""
    _name, best = candidate_oracle(makespans)
    return float(model_makespan) - float(best)


def aggregate(per_replicate: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> dict:
    """Mean/std/min/max over replicates for the requested metric keys."""
    import statistics

    out: dict[str, Any] = {"replicates": len(per_replicate)}
    for key in keys:
        values = [float(r[key]) for r in per_replicate if r.get(key) is not None]
        if not values:
            out[key] = None
            continue
        out[key] = fmean(values)
        out[key + "_std"] = statistics.pstdev(values) if len(values) > 1 else 0.0
        out[key + "_min"] = min(values)
        out[key + "_max"] = max(values)
    return out
