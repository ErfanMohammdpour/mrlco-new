#!/usr/bin/env python3
"""The ONE v2 decision-time context contract (leaf module: imports nothing).

Both the environment (`v2.env`) and the frozen encoder (`encoder_obs`) read the field names
from here. Duplicating the list is how the two would silently drift apart and produce a
tensor whose width no longer matches its declared schema, so there is exactly one source and
every dimension is derived from `len(V2_CONTEXT_FIELDS)`.

EVERY entry must be observable at plan time. Nothing in this contract may compare an estimate
against a realized future, and no entry may read a step after `t_now`.
"""

from __future__ import annotations

#: Prefix applied to each field name inside the encoder's feature-name registry.
V2_CONTEXT_NAME_PREFIX = "automotive_v2_obs_v1_"

#: The historical first 12 entries keep their order; additions follow.
V2_CONTEXT_FIELDS: tuple = (
    # -- estimated link state and how much it can be trusted --
    "est_ul", "est_dl", "est_v2v",
    "conf_ul", "conf_dl", "conf_v2v",
    # helper contact window: REMAINING seconds and committed BUSY SECONDS (not a fraction)
    "helper_contact_remaining_s", "helper_busy_s",
    # reliability class epsilon of this graph and the criticality mix
    "epsilon_class", "criticality_high_share", "criticality_medium_share",
    "mec_workers",
    # -- additions: age of the observed evidence, PER-NODE epsilons --
    "est_age_ul_s", "est_age_dl_s", "est_age_v2v_s",
    "epsilon_node_min", "epsilon_node_mean", "epsilon_node_max",
    # -- decision-time queue/load of the three pure-location reference plans --
    "est_wait_mec_s", "est_wait_ue_s", "est_wait_helper_s",
    # -- decision-time energy estimates of those same plans (system scope) --
    "est_energy_ue_j", "est_energy_mec_j", "est_energy_helper_j",
    # -- objective conditioning: active energy budget ratio and the broadcast duals --
    "energy_budget_ratio",
    "lambda_ue_energy", "lambda_total_energy", "lambda_helper_energy",
    # -- helper contact slack and the declared competitor load --
    "contact_slack", "background_dags",
)


def v2_context_feature_names(prefix: str = V2_CONTEXT_NAME_PREFIX) -> tuple:
    """Encoder feature names, derived from the single field list."""
    return tuple("%s%s" % (prefix, field) for field in V2_CONTEXT_FIELDS)


def v2_context_dim() -> int:
    return len(V2_CONTEXT_FIELDS)


__all__ = ["V2_CONTEXT_FIELDS", "V2_CONTEXT_NAME_PREFIX",
           "v2_context_dim", "v2_context_feature_names"]
