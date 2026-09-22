"""⑥b TF smoke test for the masked policy + masked PPO update.

Runs on a machine with TensorFlow 1.15 (kish). The local dev box has no TF, so
this script is the executable half of the interface contract described in
reports/v0.3-audit/MASKED_PPO_INTERFACE_6b.md.

Usage
    MARGO_OBS_VERSION=v3 MARGO_MASK_MODE=static \
        python -m spec.masked_ppo_smoke            # static shield path
    MARGO_OBS_VERSION=v3 MARGO_MASK_MODE=off \
        python -m spec.masked_ppo_smoke            # legacy no-op path

Exit code 0 means every invariant held. Any failure prints a JSON report naming
the offending check and exits non-zero, so it can gate a commit.
"""

from __future__ import annotations

import json
import sys

import numpy as np


def main(argv=None):
    import tensorflow as tf

    from env.mec_offloaing_envs.scheduler import encoder_obs as eo
    from env.mec_offloaing_envs.scheduler import masking as masking_mod
    from policies.meta_seq2seq_policy import Seq2SeqPolicy

    if eo.OBS_VERSION != "v3":
        print(
            json.dumps(
                {
                    "error": "this smoke test needs the v3 schema",
                    "obs_version": eo.OBS_VERSION,
                    "hint": "run with MARGO_OBS_VERSION=v3",
                },
                indent=2,
            )
        )
        return 2

    checks = {}
    failures = []

    def check(name, ok, detail=None):
        checks[name] = {"ok": bool(ok), "detail": detail}
        if not ok:
            failures.append(name)

    policy = Seq2SeqPolicy(
        eo.PACKED_DIM,
        encoder_units=16,
        decoder_units=16,
        vocab_size=3,
        name="smoke_policy",
        encoder_type="meanagg",
        readout_type="triple",
    )
    network = policy.network
    mask_mode = getattr(policy, "mask_mode", "off")
    masking_on = policy.feasible_mask is not None
    check(
        "placeholder_presence_matches_mode",
        masking_on == (mask_mode != "off"),
        {"mask_mode": mask_mode, "placeholder": masking_on},
    )

    channels = eo.feasibility_channel_indices()

    def synthetic_obs(batch=2, slots=20, seed=0):
        rng = np.random.RandomState(seed)
        obs = rng.normal(size=(batch, slots, eo.PACKED_DIM)).astype(np.float32)
        feasible = np.ones((batch, slots, 3), dtype=bool)
        feasible[0, ::3, 1] = False      # MEC infeasible on every third slot
        feasible[1, 1::4, 2] = False     # HELPER infeasible on some slots
        for a, ch in enumerate(channels):
            obs[..., ch] = feasible[..., a].astype(np.float32)
        return obs, feasible

    obs, feasible = synthetic_obs()
    obs_dead = obs.copy()
    obs_dead[1, 0, list(channels)] = 0.0     # one all-infeasible slot

    def mask_feed(observations, feasible_mask=None):
        if not masking_on:
            return {}
        if feasible_mask is None:
            feasible_mask = masking_mod.observation_mask(observations, mode=mask_mode)
        return policy.mask_feed(policy.check_feasible_mask(feasible_mask, observations))

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())

        actions, logits, values = policy.get_actions(obs)
        check("action_shape", actions.shape == obs.shape[:2], list(actions.shape))
        check("logits_shape", logits.shape == obs.shape[:2] + (3,), list(logits.shape))
        check("values_shape", values.shape == obs.shape[:2], list(values.shape))
        picked = np.take_along_axis(feasible, actions[..., None], axis=-1)[..., 0]
        check("sampled_actions_are_feasible", bool(np.all(picked)), "infeasible action sampled")
        if masking_on:
            check(
                "stored_mask_is_the_applied_mask",
                bool(np.array_equal(policy.last_feasible_mask, feasible)),
            )
        else:
            check("stored_mask_is_the_applied_mask", policy.last_feasible_mask is None)

        probs = sess.run(network.sample_pi, feed_dict=mask_feed(obs))
        if masking_on:
            check(
                "infeasible_probability_is_zero",
                float(np.max(probs[~feasible])) == 0.0,
                float(np.max(probs[~feasible])),
            )
            check(
                "probabilities_sum_to_one",
                bool(np.allclose(probs.sum(axis=-1), 1.0, atol=1e-5)),
            )
            valid_entropy = sess.run(
                network.entropy(), feed_dict=mask_feed(obs)
            )
            check("entropy_finite", bool(np.all(np.isfinite(valid_entropy))))
        else:
            check("legacy_softmax_untouched", bool(np.all(probs > 0.0)))

        # ratio identity: identical logits under the SAME mask -> ratio exactly 1
        old_logits = tf.compat.v1.placeholder(
            dtype=tf.float32, shape=[None, None, 3], name="smoke_old_logits"
        )
        acts_ph = tf.compat.v1.placeholder(dtype=tf.int32, shape=[None, None], name="smoke_acts")
        ratio = policy.distribution.likelihood_ratio_sym(
            acts_ph, old_logits, network.decoder_logits
        )
        base_feed = {
            policy.obs: obs,
            policy.decoder_inputs: np.zeros(obs.shape[:2], dtype=np.int32),
            policy.decoder_targets: actions,
            policy.decoder_full_length: np.full((obs.shape[0],), obs.shape[1], dtype=np.int32),
            old_logits: logits,
            acts_ph: actions,
        }
        base_feed.update(mask_feed(obs))
        ratio_value = sess.run(ratio, feed_dict=base_feed)
        deviation = float(np.max(np.abs(ratio_value - 1.0)))
        check("ratio_identity_with_fixed_mask", deviation < 1e-5, deviation)

        # dead-end guard: guard fires, distribution stays finite
        _, logits_dead, _ = policy.get_actions(obs_dead)
        check("dead_end_logits_finite", bool(np.all(np.isfinite(logits_dead))))
        if masking_on:
            dead = sess.run(
                network.sample_dead_end_rows,
                feed_dict={
                    policy.obs: obs_dead,
                    policy.decoder_full_length: np.full(
                        (obs_dead.shape[0],), obs_dead.shape[1], dtype=np.int32
                    ),
                    **mask_feed(obs_dead),
                },
            )
            check("dead_end_guard_counts_row", int(dead) >= 1, int(dead))

        # log-prob of a deliberately infeasible action is ~ -1e9 (finite)
        if masking_on:
            action_grid = np.zeros(obs.shape[:2], dtype=np.int32)
            first_infeasible = np.argmax(~feasible, axis=-1)
            has_infeasible = ~feasible.all(axis=-1)
            action_grid[has_infeasible] = first_infeasible[has_infeasible]
            logp = sess.run(
                policy.distribution.log_likelihood_sym(acts_ph, network.sample_decoder_logits),
                feed_dict={
                    policy.obs: obs,
                    policy.decoder_full_length: np.full(
                        (obs.shape[0],), obs.shape[1], dtype=np.int32
                    ),
                    acts_ph: action_grid,
                    **mask_feed(obs),
                },
            )
            worst = float(np.min(logp[has_infeasible]))
            check("infeasible_action_logp_is_neg_large", worst <= -1e8, worst)

        # gradient sanity on the PPO surrogate
        advs_ph = tf.compat.v1.placeholder(
            dtype=tf.float32, shape=[None, None], name="smoke_advs"
        )
        surr = -tf.reduce_mean(ratio * advs_ph)
        grads = [
            g
            for g in tf.gradients(surr, network.get_trainable_variables())
            if g is not None
        ]
        grad_feed = dict(base_feed)
        grad_feed[advs_ph] = np.ones(obs.shape[:2], dtype=np.float32)
        grad_values = sess.run(grads, feed_dict=grad_feed)
        check(
            "gradients_finite",
            all(bool(np.all(np.isfinite(g))) for g in grad_values),
            "non-finite gradient",
        )

    report = {
        "mask_mode": mask_mode,
        "obs_version": eo.OBS_VERSION,
        "obs_dim": int(eo.PACKED_DIM),
        "checks": checks,
        "failures": failures,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
