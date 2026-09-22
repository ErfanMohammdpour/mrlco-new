"""⑥b metric smoke: the seven mask/policy/critic metrics on real TF.

Runs on kish (TF 1.15). Builds the policy, rolls it out on synthetic v3
observations through the SAME code path the trainer uses, and folds the
`last_feasible_mask` / `last_raw_logits` / actions / values captured in that one
sess.run into the accumulator from `scheduler.mask_metrics`.

    python -m spec.mask_metric_smoke --mask-mode off
    python -m spec.mask_metric_smoke --mask-mode static

Acceptance (from the audit):
  * no-deadline and all-valid: the five control rates are exactly 0.0 in BOTH
    modes, entropy is finite, value_abs_max is finite and bounded;
  * hard synthetic:          active_rate > 0 and invalid_action_rate == 0;
  * forced synthetic:        forced_rate > 0;
  * dead-end synthetic:      all_invalid_rate > 0, invalid_action_rate == 0 and
                             every logit/value/entropy is finite.

It also cross-checks the metric against the TF distribution it describes: the
fraction of sampled actions with `sample_pi == 0` must equal
`policy/invalid_action_rate`, and the mean entropy of `sample_pi` must equal
`policy/entropy_valid`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

RATE_KEYS = (
    "mask/active_rate",
    "mask/forced_rate",
    "mask/all_invalid_rate",
    "policy/invalid_action_rate",
    "policy/argmax_masked_rate",
)


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mask-mode", default="static", choices=("off", "static"))
    parser.add_argument("--obs-version", default="v3")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--slots", type=int, default=20)
    return parser.parse_args(argv)


# --------------------------------------------------------------------------- #
# synthetic scenarios: (hard deadline?, feasibility pattern per slot)
# --------------------------------------------------------------------------- #
SCENARIOS = {
    # every action proof-feasible -> nothing may be shielded
    "all_valid": dict(hard=True, pattern=lambda i: (True, True, True)),
    # a hard deadline with MEC proof-infeasible on some slots -> real shield
    "hard_partial": dict(hard=True, pattern=lambda i: (True, i % 3 != 0, True)),
    # exactly one feasible action on odd slots -> forced decisions
    "forced": dict(hard=True, pattern=lambda i: (True, True, True) if i % 2 == 0 else (False, True, False)),
    # nothing feasible on some slots -> all-invalid rows, guard must fire
    "dead_end": dict(hard=True, pattern=lambda i: (True, True, True) if i % 4 else (False, False, False)),
    # proof-infeasible but NO deadline: soft/none deadlines must not be shielded
    "no_deadline": dict(hard=False, pattern=lambda i: (False, False, False)),
}


def build_obs(eo, hard, pattern, batch, slots, seed):
    rng = np.random.RandomState(seed)
    obs = rng.normal(size=(batch, slots, eo.PACKED_DIM)).astype(np.float32)
    channels = eo.feasibility_channel_indices()
    proof = np.zeros((batch, slots, 3), dtype=bool)
    for i in range(slots):
        proof[:, i, :] = np.asarray(pattern(i), dtype=bool)
    for a, ch in enumerate(channels):
        obs[..., ch] = proof[..., a].astype(np.float32)
    obs[..., eo.hard_deadline_channel_index()] = 1.0 if hard else 0.0
    return obs, proof


def expected_mask(hard, proof):
    """Shield mask: proof feasibility only under a HARD deadline."""
    if not hard:
        return np.ones_like(proof)
    return proof


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    os.environ["MARGO_OBS_VERSION"] = str(args.obs_version)
    os.environ["MARGO_MASK_MODE"] = str(args.mask_mode)

    import tensorflow as tf

    from env.mec_offloaing_envs.scheduler import encoder_obs as eo
    from env.mec_offloaing_envs.scheduler import mask_metrics
    from policies.meta_seq2seq_policy import Seq2SeqPolicy

    checks = {}
    failures = []

    def check(name, ok, detail=None):
        checks[name] = {"ok": bool(ok), "detail": detail}
        if not ok:
            failures.append(name)

    if eo.OBS_VERSION != "v3":
        print(json.dumps({"error": "needs v3", "obs_version": eo.OBS_VERSION}, indent=2))
        return 2

    policy = Seq2SeqPolicy(
        eo.PACKED_DIM,
        encoder_units=16,
        decoder_units=16,
        vocab_size=3,
        name="metric_smoke_policy",
        encoder_type="meanagg",
        readout_type="triple",
    )
    network = policy.network
    mode = getattr(policy, "mask_mode", "off")
    masking_on = policy.feasible_mask is not None
    check(
        "placeholder_presence_matches_mode",
        masking_on == (mode != "off"),
        {"mask_mode": mode, "placeholder": masking_on},
    )

    results = {}
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())

        for name, cfg in SCENARIOS.items():
            obs, proof = build_obs(
                eo, cfg["hard"], cfg["pattern"], args.batch, args.slots, args.seed
            )
            actions, _logits, values = policy.get_actions(obs)
            stored = policy.last_feasible_mask
            raw = policy.last_raw_logits
            check(
                "raw_logits_are_captured[%s]" % name,
                raw is not None and np.asarray(raw).shape == obs.shape[:2] + (3,),
                None if raw is None else list(np.asarray(raw).shape),
            )
            pre_guard = None if stored is None else np.asarray(stored).astype(bool)
            if masking_on:
                want = expected_mask(cfg["hard"], proof)
                check("stored_mask_matches_expectation[%s]" % name, bool(np.array_equal(pre_guard, want)))

            try:
                acc = mask_metrics.accumulate(
                    pre_guard=pre_guard,
                    actions=actions,
                    raw_logits=raw,
                    values=values,
                    require_mask=masking_on,
                )
                out = mask_metrics.rates(acc)
            except ValueError as exc:
                # a non-finite or ill-shaped metric is a failure, but the run
                # still has to emit evidence for the other scenarios
                check("metrics_computable[%s]" % name, False, str(exc))
                results[name] = {"error": str(exc)}
                continue
            check("metrics_computable[%s]" % name, True)
            results[name] = {k: float(v) for k, v in out.items()}

            # --- cross-check against the TF distribution of THAT trajectory ---
            # sample_pi must come from the rollout run: the decoder samples its
            # own next input, so a second sess.run would replay a different path
            sample_pi = np.asarray(policy.last_sample_pi)
            check(
                "sample_pi_shape[%s]" % name,
                sample_pi.shape == obs.shape[:2] + (3,),
                list(sample_pi.shape),
            )
            picked = np.take_along_axis(sample_pi, actions[..., None], axis=-1)[..., 0]
            zero_prob_rate = float(np.mean(picked == 0.0))
            check(
                "invalid_action_rate_matches_zero_probability[%s]" % name,
                abs(zero_prob_rate - out["policy/invalid_action_rate"]) < 1e-9,
                {"from_sample_pi": zero_prob_rate, "metric": out["policy/invalid_action_rate"]},
            )
            p = sample_pi
            safe = np.where(p > 0.0, p, 1.0)
            tf_entropy = float(np.mean(-np.where(p > 0.0, p * np.log(safe), 0.0).sum(axis=-1)))
            # the metric evaluates the same formula in float64 from float32
            # logits; the graph's pi is float32, hence the 1e-4 tolerance
            check(
                "entropy_matches_tf_distribution[%s]" % name,
                abs(tf_entropy - out["policy/entropy_valid"]) < 1e-4,
                {"from_sample_pi": tf_entropy, "metric": out["policy/entropy_valid"]},
            )
            check(
                "logits_values_finite[%s]" % name,
                bool(np.all(np.isfinite(np.asarray(raw)))) and bool(np.all(np.isfinite(values)))
                and np.isfinite(out["policy/entropy_valid"]),
            )
            check(
                "entropy_in_range[%s]" % name,
                0.0 <= out["policy/entropy_valid"] <= float(np.log(3.0)) + 1e-6,
                out["policy/entropy_valid"],
            )
            check(
                "value_abs_max_reasonable[%s]" % name,
                0.0 <= out["critic/value_abs_max"] < 1e3,
                out["critic/value_abs_max"],
            )
            # monotonicity invariants: each implies the token was active
            eps = 1e-9
            check(
                "sub_rates_within_active[%s]" % name,
                out["mask/forced_rate"] <= out["mask/active_rate"] + eps
                and out["mask/all_invalid_rate"] <= out["mask/active_rate"] + eps
                and out["policy/argmax_masked_rate"] <= out["mask/active_rate"] + eps,
                {
                    "active": out["mask/active_rate"],
                    "forced": out["mask/forced_rate"],
                    "all_invalid": out["mask/all_invalid_rate"],
                    "argmax_masked": out["policy/argmax_masked_rate"],
                },
            )

    # ---------------- acceptance rules ----------------
    zero_rate_scenarios = ("all_valid", "no_deadline")
    for name in zero_rate_scenarios:
        for key in RATE_KEYS:
            check(
                "zero_rate[%s][%s]" % (name, key),
                results[name][key] == 0.0,
                results[name][key],
            )
    if mode == "static":
        # these three scenarios only carry a shield in static mode; in off mode
        # there is no mask, so the five rates are zero by definition (checked below)
        check(
            "hard_partial_active",
            results.get("hard_partial", {}).get("mask/active_rate", 0.0) > 0.0,
            results.get("hard_partial", {}).get("mask/active_rate"),
        )
        check(
            "forced_rate_positive",
            results.get("forced", {}).get("mask/forced_rate", 0.0) > 0.0,
            results.get("forced", {}).get("mask/forced_rate"),
        )
        check(
            "dead_end_all_invalid_positive",
            results.get("dead_end", {}).get("mask/all_invalid_rate", 0.0) > 0.0,
            results.get("dead_end", {}).get("mask/all_invalid_rate"),
        )
    # invalid actions are a bug in every mode
    for name in SCENARIOS:
        check(
            "no_invalid_action[%s]" % name,
            results.get(name, {}).get("policy/invalid_action_rate", 1.0) == 0.0,
            results.get(name, {}).get("policy/invalid_action_rate"),
        )
    if mode == "off":
        for name in SCENARIOS:
            for key in RATE_KEYS:
                check(
                    "off_mode_zero[%s][%s]" % (name, key),
                    results.get(name, {}).get(key, 1.0) == 0.0,
                    results.get(name, {}).get(key),
                )

    report = {
        "mask_mode": mode,
        "obs_version": eo.OBS_VERSION,
        "obs_dim": int(eo.PACKED_DIM),
        "metrics": results,
        "metric_keys": list(mask_metrics.METRIC_KEYS),
        "checks": checks,
        "failures": failures,
    }
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
