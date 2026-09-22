"""⑥b TF smoke test for the masked policy + masked PPO update.

Runs on a machine with TensorFlow 1.15 (kish). The local dev box has no TF, so
this script is the executable half of the interface contract in
reports/v0.3-audit/MASKED_PPO_INTERFACE_6b.md.

Usage
    python -m spec.masked_ppo_smoke --mask-mode off
    python -m spec.masked_ppo_smoke --mask-mode static
    python -m spec.masked_ppo_smoke --mask-mode runtime   # must fail loudly

Exit code 0 means every invariant held. Any failure prints a JSON report naming
the offending check and exits non-zero, so it can gate a commit.

What it covers (post-audit P0):
  * shapes, feasibility of sampled actions, stored-mask identity;
  * zero probability / -1e9 log-probability for infeasible actions;
  * ratio identity with the CORRECT teacher-forced prefix (shifted actions);
  * dead-end guard;
  * value head magnitude (the -1e9 critic contamination regression);
  * an actual optimizer step: finite losses, non-empty finite grads, parameters move;
  * off-mode parity: the mask node is the identity on the legacy path;
  * runtime mode refuses to silently degrade to a static shield.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mask-mode", default="static", choices=("off", "static", "runtime"))
    parser.add_argument("--obs-version", default="v3")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--slots", type=int, default=20)
    parser.add_argument("--batch", type=int, default=2)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    # Must happen before the policy / encoder modules are imported.
    os.environ["MARGO_OBS_VERSION"] = str(args.obs_version)
    os.environ["MARGO_MASK_MODE"] = str(args.mask_mode)

    import tensorflow as tf

    from env.mec_offloaing_envs.scheduler import encoder_obs as eo
    from env.mec_offloaing_envs.scheduler import masking as masking_mod
    from policies.meta_seq2seq_policy import Seq2SeqPolicy

    checks = {}
    failures = []

    def check(name, ok, detail=None):
        checks[name] = {"ok": bool(ok), "detail": detail}
        if not ok:
            failures.append(name)

    if eo.OBS_VERSION != "v3":
        print(json.dumps({"error": "needs the v3 schema", "obs_version": eo.OBS_VERSION}, indent=2))
        return 2

    channels = eo.feasibility_channel_indices()
    hard_channel = eo.hard_deadline_channel_index()

    def synthetic_obs(batch, slots, seed):
        """Random packed obs with a hard deadline on row 0 and a soft one on row 1.

        Row 0: hard deadline, MEC infeasible on every third slot -> real shield.
        Row 1: SOFT deadline with the same proof-infeasible slots -> must stay
               fully unmasked (soft lateness is priced, not forbidden).
        """
        rng = np.random.RandomState(seed)
        obs = rng.normal(size=(batch, slots, eo.PACKED_DIM)).astype(np.float32)
        feasible = np.ones((batch, slots, 3), dtype=bool)
        # BOTH rows carry the same proof-infeasibility; only the deadline TYPE
        # differs, so the shield must differ while the features do not.
        feasible[:, ::3, 1] = False
        for a, ch in enumerate(channels):
            obs[..., ch] = feasible[..., a].astype(np.float32)
        obs[..., hard_channel] = 0.0
        if batch > 0:
            obs[0, :, hard_channel] = 1.0                      # row 0: hard
        if batch > 1:
            obs[1, :, hard_channel] = 0.0                      # row 1: soft
        return obs, feasible

    obs, proof_feasible = synthetic_obs(args.batch, args.slots, args.seed)
    expected_mask = np.logical_or(
        proof_feasible, np.logical_not((obs[..., hard_channel] > 0.5)[..., None])
    )
    obs_dead = obs.copy()
    obs_dead[0, 0, list(channels)] = 0.0                       # all-infeasible hard slot

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

    def mask_feed(observations, feasible_mask=None):
        if not masking_on:
            return {}
        if feasible_mask is None:
            feasible_mask = masking_mod.observation_mask(observations, mode=mask_mode)
        return policy.mask_feed(policy.check_feasible_mask(feasible_mask, observations))

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())

        # ---- runtime mode must refuse to degrade into a static shield ----
        if mask_mode == "runtime":
            # Runtime mode is expected to fail loudly. This script catches that
            # ValueError and exits 0, so the shell must NOT swallow its status.
            caught = None
            try:
                masking_mod.observation_mask(obs, mode="runtime")
            except ValueError as exc:
                caught = exc
            check(
                "runtime_mask_is_not_derived_from_obs",
                caught is not None,
                {
                    "exception": type(caught).__name__ if caught is not None else None,
                    "message": str(caught) if caught is not None else "no error raised",
                },
            )
            report = {
                "mask_mode": mask_mode,
                "obs_version": eo.OBS_VERSION,
                "obs_dim": int(eo.PACKED_DIM),
                "expected": "ValueError from observation_mask(mode='runtime')",
                "checks": checks,
                "failures": failures,
            }
            print(json.dumps(report, indent=2, sort_keys=True, default=str))
            return 1 if failures else 0

        actions, logits, values = policy.get_actions(obs)
        check("action_shape", actions.shape == obs.shape[:2], list(actions.shape))
        check("logits_shape", logits.shape == obs.shape[:2] + (3,), list(logits.shape))
        check("values_shape", values.shape == obs.shape[:2], list(values.shape))
        if masking_on:
            check(
                "shield_is_hard_deadline_only",
                bool(np.array_equal(np.asarray(policy.last_feasible_mask).astype(bool), expected_mask)),
                {
                    "masked_rows_hard": int((~expected_mask[0]).sum()),
                    "masked_rows_soft": int((~policy.last_feasible_mask[1]).sum()),
                },
            )
            picked = np.take_along_axis(
                np.asarray(policy.last_feasible_mask).astype(bool),
                actions[..., None],
                axis=-1,
            )[..., 0]
        else:
            check("stored_mask_is_the_applied_mask", policy.last_feasible_mask is None)
            picked = np.ones(actions.shape, dtype=bool)
        check("sampled_actions_are_feasible", bool(np.all(picked)), "infeasible action sampled")

        # ---- off-mode parity: the mask node is the identity ----
        raw_logits, masked_logits = sess.run(
            [network.sample_decoder_logits_raw, network.sample_decoder_logits],
            feed_dict=mask_feed(obs) if masking_on else {},
        )
        if not masking_on:
            check("off_mode_mask_is_identity", bool(np.array_equal(raw_logits, masked_logits)))

        sample_pi, sample_q, sample_vf = sess.run(
            [network.sample_pi, network.sample_q, network.sample_vf],
            feed_dict=mask_feed(obs),
        )
        # value magnitude: -1e9 logits inside the critic would push this to ~1e8
        check(
            "critic_value_scale_is_sane",
            float(np.max(np.abs(sample_vf))) < 1e3,
            float(np.max(np.abs(sample_vf))),
        )
        check(
            "value_is_masked_support_dot_raw_q",
            bool(np.allclose(sample_vf, (sample_pi * sample_q).sum(axis=-1), atol=1e-3)),
        )
        if masking_on:
            check(
                "infeasible_probability_is_zero",
                float(np.max(sample_pi[~np.asarray(policy.last_feasible_mask).astype(bool)])) == 0.0,
            )
            check(
                "probabilities_sum_to_one",
                bool(np.allclose(sample_pi.sum(axis=-1), 1.0, atol=1e-5)),
            )

        # ---- ratio identity with the correct teacher-forced prefix ----
        shift_actions = np.concatenate(
            [np.zeros((actions.shape[0], 1), dtype=np.int32), actions[:, :-1]], axis=1
        )
        old_logits = tf.compat.v1.placeholder(
            dtype=tf.float32, shape=[None, None, 3], name="smoke_old_logits"
        )
        acts_ph = tf.compat.v1.placeholder(dtype=tf.int32, shape=[None, None], name="smoke_acts")
        ratio = policy.distribution.likelihood_ratio_sym(
            acts_ph, old_logits, network.decoder_logits
        )
        base_feed = {
            policy.obs: obs,
            policy.decoder_inputs: shift_actions,
            policy.decoder_targets: actions,
            policy.decoder_full_length: np.full((obs.shape[0],), obs.shape[1], dtype=np.int32),
            old_logits: logits,
            acts_ph: actions,
        }
        base_feed.update(mask_feed(obs))
        ratio_value = sess.run(ratio, feed_dict=base_feed)
        deviation = float(np.max(np.abs(ratio_value - 1.0)))
        check(
            "ratio_identity_before_first_update",
            deviation < 1e-4,
            {"max_abs_deviation": deviation},
        )

        # ---- dead-end guard ----
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

        # ---- log-prob of a deliberately infeasible action is ~ -1e9 ----
        if masking_on:
            action_grid = np.zeros(obs.shape[:2], dtype=np.int32)
            infeasible = ~np.asarray(policy.last_feasible_mask).astype(bool)
            has_infeasible = infeasible.any(axis=-1)
            action_grid[has_infeasible] = np.argmax(infeasible, axis=-1)[has_infeasible]
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
            check("infeasible_action_logp_is_neg_large", float(np.min(logp[has_infeasible])) <= -1e8)

        # ---- an ACTUAL PPO-shaped optimizer step ----
        advs_ph = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, None], name="smoke_advs")
        old_v_ph = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, None], name="smoke_old_v")
        ret_ph = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, None], name="smoke_ret")
        clip_value = 0.2

        ratio_train = policy.distribution.likelihood_ratio_sym(
            acts_ph, old_logits, network.decoder_logits
        )
        clipped = tf.minimum(
            ratio_train * advs_ph,
            tf.clip_by_value(ratio_train, 1.0 - clip_value, 1.0 + clip_value) * advs_ph,
        )
        surr_obj = -tf.reduce_mean(clipped)
        vpred = network.vf
        vpredclipped = old_v_ph + tf.clip_by_value(vpred - old_v_ph, -clip_value, clip_value)
        vf_loss = 0.5 * tf.reduce_mean(
            tf.maximum(tf.square(vpred - ret_ph), tf.square(vpredclipped - ret_ph))
        )
        total_loss = surr_obj + 0.5 * vf_loss
        params = network.get_trainable_variables()
        # keep (grad, var) pairs together: filtering the list would misalign them
        grads_and_vars = [
            (g, v) for g, v in zip(tf.gradients(total_loss, params), params) if g is not None
        ]
        grads = [g for g, _v in grads_and_vars]
        check("gradient_list_is_non_empty", len(grads) > 0, {"n_grads": len(grads)})
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=5e-4, name="smoke_adam")
        train_op = optimizer.apply_gradients(grads_and_vars)
        # ONLY the Adam slots are initialised below: a global re-init would wipe
        # the policy weights that produced old_logits / old_v, making the step
        # inconsistent with the rollout it is supposed to replay.
        slot_init = tf.compat.v1.variables_initializer(optimizer.variables())

        train_feed = {
            policy.obs: obs,
            policy.decoder_inputs: shift_actions,
            policy.decoder_targets: actions,
            policy.decoder_full_length: np.full((obs.shape[0],), obs.shape[1], dtype=np.int32),
            old_logits: logits,
            acts_ph: actions,
            advs_ph: np.ones(obs.shape[:2], dtype=np.float32),
            old_v_ph: values,
            ret_ph: np.zeros(obs.shape[:2], dtype=np.float32),
        }
        train_feed.update(mask_feed(obs))
        sess.run(slot_init)
        before = sess.run(params)
        # grads BEFORE the step, in their own run, so the update cannot race them
        grads_before = sess.run(grads, feed_dict=train_feed)
        _, losses = sess.run([train_op, [surr_obj, vf_loss, total_loss]], feed_dict=train_feed)
        after = sess.run(params)
        check(
            "losses_finite",
            all(bool(np.isfinite(v)) for v in losses),
            [float(v) for v in losses],
        )
        check(
            "value_loss_is_sane",
            float(abs(losses[1])) < 1e6,
            float(losses[1]),
        )
        check(
            "gradients_finite",
            all(bool(np.all(np.isfinite(g))) for g in grads_before),
            "non-finite gradient",
        )
        check(
            "parameters_moved",
            any(not np.array_equal(a, b) for a, b in zip(before, after)),
        )

    report = {
        "mask_mode": mask_mode,
        "obs_version": eo.OBS_VERSION,
        "obs_dim": int(eo.PACKED_DIM),
        "checks": checks,
        "failures": failures,
    }
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
