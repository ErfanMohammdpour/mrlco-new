"""Few-shot from π_BC. Diagnostic only. Not the frozen 3500 primary.

Three adaptations on held-out support, report query:
  k0: copy π_BC, no inner
  k3_ppo: frozen inner PPO (expect occupancy damage)
  k3_ce: encoder-frozen CE on greedy-from-MEC support labels
"""

from __future__ import annotations

import numpy as np

from spec.bc_greedy_mec import BC_MAX_PASSES
from spec.eval_protocol import query_metrics_from_samples, require_sliced_task
from spec.split_loader import support_query_tasks

CE_STEPS = 3
CE_LR = 5e-4


def is_encoder_var_name(name):
    return "encoder" in str(name)


def encoder_frozen_adapt_vars(trainable):
    frozen = [v for v in trainable if is_encoder_var_name(v.name)]
    adapt = [v for v in trainable if not is_encoder_var_name(v.name)]
    if not frozen:
        raise ValueError("encoder-frozen CE found no encoder vars")
    if not adapt:
        raise ValueError("encoder-frozen CE found no decoder/head vars")
    return frozen, adapt


def action_mix(samples_data):
    acts = np.asarray(samples_data["actions"], dtype=np.int32)
    flat = acts.reshape(-1)
    counts = np.bincount(flat, minlength=3).astype(np.float64)
    counts = counts / max(float(counts.sum()), 1.0)
    n_non = np.sum(acts != 1, axis=-1).astype(np.float64)
    return {
        "local_frac": float(counts[0]),
        "mec_frac": float(counts[1]),
        "v2v_frac": float(counts[2]),
        "n_nonmec_mean": float(np.mean(n_non)),
        "n_nonmec_p50": float(np.median(n_non)),
    }


def _json_metrics(metrics):
    out = {}
    for key, val in metrics.items():
        if isinstance(val, (np.floating, np.integer)):
            out[key] = float(val)
        elif isinstance(val, np.ndarray):
            out[key] = val.tolist()
        else:
            out[key] = val
    return out


def attach_mix(metrics, samples_data):
    metrics.update(action_mix(samples_data))
    return _json_metrics(metrics)


def support_expert_obs_acts(env, support_task):
    from env.mec_offloaing_envs.scheduler import greedy_from_mec_plan

    env.set_task(require_sliced_task(support_task))
    graphs = env._slice_current(env.task_graphs_batchs)
    enc = np.asarray(env._slice_current(env.encoder_batchs), dtype=np.float32)
    obs_rows = []
    act_rows = []
    for g_i, tg in enumerate(graphs):
        plan, _result = greedy_from_mec_plan(tg, env.scheduler_resources, max_passes=BC_MAX_PASSES)
        actions = [int(a) for _, a in plan]
        if len(actions) != int(tg.task_number):
            raise ValueError("support expert plan length mismatch")
        obs_rows.append(np.asarray(enc[g_i], dtype=np.float32))
        act_rows.append(np.asarray(actions, dtype=np.int32))
    obs = np.stack(obs_rows, axis=0)
    acts = np.stack(act_rows, axis=0)
    return obs, acts


def encoder_frozen_ce_ops(policy, name):
    import tensorflow as tf

    loss = tf.reduce_mean(policy.network.neglogp())
    opt = tf.compat.v1.train.AdamOptimizer(CE_LR, name=name)
    frozen, adapt = encoder_frozen_adapt_vars(policy.get_trainable_variables())
    train = opt.minimize(loss, var_list=adapt)
    return {
        "loss": loss,
        "train": train,
        "opt": opt,
        "n_adapt": len(adapt),
        "n_frozen": len(frozen),
    }


def build_scratch_held_out_evaluator(env, source_policy, name, seed):
    import numpy as np
    from baselines.vf_baseline import ValueFunctionBaseline
    from meta_algos.held_out_eval import HeldOutQueryEvaluator
    from meta_algos.ppo_offloading import PPO
    from policies.meta_seq2seq_policy import Seq2SeqPolicy
    from samplers.seq2seq_sampler import Seq2SeqSampler
    from samplers.seq2seq_sampler_process import Seq2SeSamplerProcessor

    policy = Seq2SeqPolicy(
        obs_dim=env.input_dim,
        encoder_units=128,
        decoder_units=128,
        vocab_size=3,
        name=name,
    )
    sampler = Seq2SeqSampler(
        env,
        policy,
        rollouts_per_meta_task=1,
        max_path_length=20000,
        envs_per_task=None,
        parallel=False,
    )
    processor = Seq2SeSamplerProcessor(
        baseline=ValueFunctionBaseline(),
        discount=0.99,
        gae_lambda=0.95,
        normalize_adv=True,
        positive_adv=False,
    )
    ppo = PPO(
        policy=policy,
        meta_sampler=sampler,
        meta_sampler_process=processor,
        lr=5e-4,
        num_inner_grad_steps=CE_STEPS,
        clip_value=0.2,
        max_grad_norm=0.5,
        rng=np.random.RandomState(int(seed)),
        support_select="random",
    )
    return HeldOutQueryEvaluator(
        env=env,
        policy=policy,
        sampler=sampler,
        processor=processor,
        ppo=ppo,
        source_policy=source_policy,
        ppo_batch_size=20,
    )


def run_ce_steps(sess, policy, loss_op, train_op, opt, obs, acts, n_steps=CE_STEPS):
    import tensorflow as tf

    n_steps = int(n_steps)
    if n_steps != CE_STEPS:
        raise ValueError("few-shot CE steps must be %d, got %s" % (CE_STEPS, n_steps))
    slot = opt.variables()
    if slot:
        sess.run(tf.compat.v1.variables_initializer(slot))
    n, n_tok = acts.shape
    shift = np.concatenate([np.zeros((n, 1), dtype=np.int32), acts[:, :-1]], axis=1)
    fl = np.full((n,), n_tok, dtype=np.int32)
    losses = []
    for _ in range(n_steps):
        _, lv = sess.run(
            [train_op, loss_op],
            feed_dict={
                policy.obs: obs,
                policy.decoder_inputs: shift,
                policy.decoder_targets: acts,
                policy.decoder_full_length: fl,
            },
        )
        losses.append(float(lv))
    return losses


def evaluate_query(evaluator, query_task, k_steps, adapt_name, distribution_id):
    evaluator._activate(query_task)
    query_paths = evaluator.sampler.obtain_samples(log=False, log_prefix="query_")
    query_data = evaluator.processor.process_samples(query_paths, log=False, log_prefix="query_")
    greedy = evaluator.env.greedy_solution_for_current_task()
    metrics = query_metrics_from_samples(query_data)
    metrics["k_steps"] = int(k_steps)
    metrics["adapt"] = adapt_name
    metrics["distribution_id"] = int(distribution_id)
    if evaluator.env.resource_cluster.use_energy:
        _, greedy_latency, greedy_energy = greedy
        metrics["query_greedy_latency"] = float(np.mean(greedy_latency))
        metrics["query_greedy_energy"] = float(np.mean(greedy_energy))
    else:
        _, greedy_latency = greedy
        metrics["query_greedy_latency"] = float(np.mean(greedy_latency))
    return attach_mix(metrics, query_data)


def evaluate_one_adapt(evaluator, env_index, distribution_id, adapt, sess, ce_bundle=None):
    from meta_algos.variable_io import assign_trainable

    assign_trainable(evaluator.source_policy, evaluator.policy, sess=sess)
    support_task, query_task = support_query_tasks(env_index, distribution_id)
    if adapt == "k0":
        pass
    elif adapt == "k3_ppo":
        evaluator._activate(support_task)
        support_paths = evaluator.sampler.obtain_samples(log=False, log_prefix="")
        support_data = evaluator.processor.process_samples(support_paths, log=False, log_prefix="")
        evaluator.ppo.UpdatePPOTarget(support_data, batch_size=evaluator.ppo_batch_size, k_steps=3)
    elif adapt == "k3_ce":
        if ce_bundle is None:
            raise ValueError("k3_ce needs ce_bundle")
        obs, acts = support_expert_obs_acts(evaluator.env, support_task)
        run_ce_steps(
            sess,
            evaluator.policy,
            ce_bundle["loss"],
            ce_bundle["train"],
            ce_bundle["opt"],
            obs,
            acts,
            n_steps=CE_STEPS,
        )
    else:
        raise ValueError("unknown adapt %r" % (adapt,))
    k_steps = 0 if adapt == "k0" else 3
    return evaluate_query(evaluator, query_task, k_steps, adapt, distribution_id)


def evaluate_split(evaluator, sess, ce_bundle, split_name):
    dist_ids = list(evaluator.env.distribution_ids)
    out = {"split": split_name, "n_distributions": len(dist_ids)}
    for adapt in ("k0", "k3_ppo", "k3_ce"):
        rows = []
        for env_index, dist_id in enumerate(dist_ids):
            row = evaluate_one_adapt(
                evaluator, env_index, dist_id, adapt, sess, ce_bundle=ce_bundle
            )
            rows.append(row)
            print(
                "bc_fewshot %s %s dist=%s T=%.1f greedy=%.1f mix L/M/V=%.3f/%.3f/%.3f n_non_p50=%.1f"
                % (
                    split_name,
                    adapt,
                    dist_id,
                    row["query_mean_latency"],
                    row["query_greedy_latency"],
                    row["local_frac"],
                    row["mec_frac"],
                    row["v2v_frac"],
                    row["n_nonmec_p50"],
                )
            )
        out[adapt] = {
            "query_mean_latency": float(np.mean([r["query_mean_latency"] for r in rows])),
            "query_greedy_latency": float(np.mean([r["query_greedy_latency"] for r in rows])),
            "local_frac": float(np.mean([r["local_frac"] for r in rows])),
            "mec_frac": float(np.mean([r["mec_frac"] for r in rows])),
            "v2v_frac": float(np.mean([r["v2v_frac"] for r in rows])),
            "n_nonmec_p50": float(np.mean([r["n_nonmec_p50"] for r in rows])),
            "per_distribution": rows,
        }
    k0 = out["k0"]["query_mean_latency"]
    ppo = out["k3_ppo"]["query_mean_latency"]
    ce = out["k3_ce"]["query_mean_latency"]
    mix_shift = abs(out["k3_ppo"]["local_frac"] - out["k0"]["local_frac"]) + abs(
        out["k3_ppo"]["mec_frac"] - out["k0"]["mec_frac"]
    )
    out["ppo_destroyed"] = bool(ppo > k0 * 1.15 and mix_shift > 0.15)
    out["ce_helps"] = bool(ce + 1e-9 < k0)
    print(
        "bc_fewshot_split %s k0=%.1f k3_ppo=%.1f k3_ce=%.1f ppo_destroyed=%s ce_helps=%s"
        % (split_name, k0, ppo, ce, out["ppo_destroyed"], out["ce_helps"])
    )
    return out
