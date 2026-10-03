#!/usr/bin/env python3
"""v2 train/val stack: the frozen policy/sampler/MRLCO chain on the v2 environment.

Mirrors `build_automotive_primary_stack` exactly (same budgets, same algorithm, same
validation evaluator) and changes only the environment family and the observation version:

    env class          AutomotiveEnv      -> V2AutomotiveEnv
    obs version        automotive_mc_obs_v1 -> automotive_v2_obs_v1  (52/91)
    dynamics           v1 engine          -> v2 shared scheduler (queueing, links,
                                             helper occupancy/contact, reliability gate)
    evaluation protocol v1 protocol        -> CRN Gumbel protocol (automotive_crn_gumbel_v1)

The frozen v1 budgets are enforced identically (meta_batch 10, support 20, 3 inner applies).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from spec.automotive_training.automotive_primary import (
    CO_PHYSICAL_AXES, CONSTRAINT_NAMES, DUAL_LR, LEGACY_MIXED_AXES,
    META_BATCH_SIZE, SUPPORT_TRAJECTORIES_PER_META_TASK, TOKENS_PER_TRAJECTORY,
    VALIDATION_QUERY_COUNT, VALIDATION_SUPPORT_COUNT, AutomotiveDualAdapter,
    AutomotiveDualController, AutomotiveMetaSampler, AutomotivePrimaryError,
    AutomotiveResourceCluster, _validation_policy, config_fingerprint,
    decoding_flags, default_constraint_specs,
)
from spec.automotive_training.v2.crn import DEFAULT_R_SELECT, DEFAULT_S_SELECT, PROTOCOL_ID
from spec.automotive_training.v2.env import V2AutomotiveEnv
from spec.automotive_training.v2.observation import V2_OBS_VERSION

V2_ALLOWED_LINK_REGIMES = ("stable", "moderate", "degraded")


def build_automotive_v2_stack(*, seed: int, n_itr: int, ckpt_dir: str,
                              dataset_dir: str | None = None,
                              reward_mode: str = "latency_only",
                              use_energy: bool = True,
                              constraints: Sequence[str] = CONSTRAINT_NAMES,
                              constraint_dual_lr: float = DUAL_LR,
                              meta_batch_size: int = META_BATCH_SIZE,
                              support_trajectories: int = SUPPORT_TRAJECTORIES_PER_META_TASK,
                              run_kind: str = "v2_train",
                              decoding: str = "deterministic",
                              link_regime: str = "stable",
                              mec_workers: int = 1,
                              reliability: bool = False,
                              helper_contact_mean_s: float = 2.0,
                              helper_contact_cv: float = 0.5,
                              helper_busy_s: float = 0.0,
                              r_select: int = DEFAULT_R_SELECT,
                              s_select: int = DEFAULT_S_SELECT,
                              background_dags: int = 0):
    """Build (Trainer, MRLCO) on the frozen dataset with the v2 system model."""
    if str(link_regime) not in V2_ALLOWED_LINK_REGIMES:
        raise AutomotivePrimaryError("link_regime must be one of %s"
                                     % (V2_ALLOWED_LINK_REGIMES,))
    if int(mec_workers) < 1:
        raise AutomotivePrimaryError("mec_workers must be >= 1")
    if int(meta_batch_size) != META_BATCH_SIZE:
        raise AutomotivePrimaryError(
            "MRLCO requires meta_batch_size == %d (frozen)" % META_BATCH_SIZE)
    if int(support_trajectories) != SUPPORT_TRAJECTORIES_PER_META_TASK:
        raise AutomotivePrimaryError(
            "MRLCO requires support_trajectories == %d (frozen)"
            % SUPPORT_TRAJECTORIES_PER_META_TASK)

    os.environ["MARGO_OBS_VERSION"] = V2_OBS_VERSION
    from env.mec_offloaing_envs.scheduler import encoder_obs

    encoder_obs.set_obs_version(V2_OBS_VERSION)

    import tensorflow as tf
    from policies.meta_seq2seq_policy import MetaSeq2SeqPolicy
    from samplers.seq2seq_meta_sampler import Seq2SeqMetaSampler
    from samplers.seq2seq_meta_sampler_process import Seq2SeqMetaSamplerProcessor
    from baselines.vf_baseline import ValueFunctionBaseline
    from meta_algos.MRLCO import MRLCO

    from spec.automotive_training.automotive_loader import (
        DATASET_ID as LOADER_DATASET_ID, META_TEST_GUARD, load_dataset,
    )
    from spec.automotive_training.automotive_primary import DATASET_ID

    if LOADER_DATASET_ID != DATASET_ID:
        raise AutomotivePrimaryError("loader/dataset id mismatch")
    seed = int(seed)
    np.random.seed(seed)
    tf.compat.v1.set_random_seed(seed)

    dataset = load_dataset(dataset_dir) if dataset_dir else load_dataset()
    META_TEST_GUARD.assert_unopened()
    train_graphs = dataset.meta_train()
    val_support = dataset.validation_support()
    val_query = dataset.validation_query()
    if len(val_support) != VALIDATION_SUPPORT_COUNT or len(val_query) != VALIDATION_QUERY_COUNT:
        raise AutomotivePrimaryError("the frozen validation split must be 20 support / 40 query")

    spec_map = default_constraint_specs()
    requested = list(constraints) if constraints else list(CONSTRAINT_NAMES)
    enabled = [name for name in requested if name in spec_map]
    controller = AutomotiveDualController(specs={k: spec_map[k] for k in enabled},
                                          dual_lr=float(constraint_dual_lr))
    adapter = AutomotiveDualAdapter(controller, enabled)
    cluster = AutomotiveResourceCluster(energy_config={}, constraint_controller=adapter)

    def make_env(role: str) -> V2AutomotiveEnv:
        env = V2AutomotiveEnv(train_graphs, cluster, role=role,
                              slots_per_task=int(support_trajectories), base_seed=seed,
                              link_regime=str(link_regime), mec_workers=int(mec_workers),
                              reliability=bool(reliability),
                              helper_contact_mean_s=float(helper_contact_mean_s),
                              helper_contact_cv=float(helper_contact_cv),
                              helper_busy_s=float(helper_busy_s),
                              background_dags=int(background_dags))
        env.constraint_controller = adapter
        env.constraint_spec = None
        return env

    env = make_env("meta_train")
    _greedy_plan, greedy_finish, _greedy_energy = env.greedy_solution()
    baseline = ValueFunctionBaseline()
    meta_policy = MetaSeq2SeqPolicy(meta_batch_size=int(meta_batch_size),
                                    obs_dim=env.input_dim, encoder_units=128,
                                    decoder_units=128, vocab_size=3)
    sampler = Seq2SeqMetaSampler(env, meta_policy, rollouts_per_meta_task=1,
                                 meta_batch_size=int(meta_batch_size),
                                 max_path_length=TOKENS_PER_TRAJECTORY, parallel=False)
    budgeted = AutomotiveMetaSampler(
        sampler, meta_batch_size=int(meta_batch_size),
        trajectories_per_meta_task=int(support_trajectories),
        tokens_per_trajectory=TOKENS_PER_TRAJECTORY)
    processor = Seq2SeqMetaSamplerProcessor(baseline=baseline, discount=0.99,
                                            gae_lambda=0.95, normalize_adv=True,
                                            positive_adv=False)
    processor.mask_mode = getattr(meta_policy, "mask_mode", None)
    algo = MRLCO(policy=meta_policy, meta_sampler=budgeted, meta_sampler_process=processor,
                 inner_lr=5e-4, outer_lr=5e-4, meta_batch_size=int(meta_batch_size),
                 num_inner_grad_steps=3, clip_value=0.2, value_clip_epsilon=0.2,
                 support_trajectories=int(support_trajectories),
                 ppo_batch_size_trajectories=int(support_trajectories),
                 rng=np.random.RandomState(seed), support_select="random")

    def v2_env_factory(graphs, slots, seed, single_dist):
        """Validation env factory: v2 dynamics, v2 obs version, no constraint controller."""
        return V2AutomotiveEnv(list(graphs), AutomotiveResourceCluster(), role="validation",
                               slots_per_task=int(slots), base_seed=int(seed),
                               single_dist=bool(single_dist),
                               link_regime=str(link_regime), mec_workers=int(mec_workers),
                               reliability=bool(reliability),
                               helper_contact_mean_s=float(helper_contact_mean_s),
                               helper_contact_cv=float(helper_contact_cv),
                               helper_busy_s=float(helper_busy_s),
                               background_dags=int(background_dags))

    flags = decoding_flags(decoding)
    held_out = V2HeldOutEvaluator(
        support_graphs=val_support, query_graphs=val_query,
        policy=_validation_policy(env, decoding, flags),
        source_policy=meta_policy.core_policy,
        ppo_batch_size=int(support_trajectories),
        env_factory=v2_env_factory,
        link_regime=str(link_regime), mec_workers=int(mec_workers),
        reliability=bool(reliability), r_select=int(r_select), s_select=int(s_select))

    from meta_trainer import Trainer
    from spec.automotive_training.automotive_trainer import AutomotiveTrainerMixin

    AutomotiveTrainerImpl = type("AutomotiveV2Trainer", (AutomotiveTrainerMixin, Trainer), {})
    trainer = AutomotiveTrainerImpl(
        algo=algo, env=env, sampler=budgeted, sample_processor=processor,
        policy=meta_policy, n_itr=int(n_itr), greedy_finish_time=greedy_finish,
        start_itr=0, inner_batch_size=int(support_trajectories),
        print_action_choices=False, action_print_interval=0, seed=seed,
        validation_interval=50, held_out_evaluator=held_out, ckpt_dir=ckpt_dir,
        audit_writer=None)
    trainer.auto_spec_map = {k: spec_map[k] for k in enabled}
    trainer.auto_controller = controller
    trainer.auto_dataset_fingerprint = dataset.fingerprint()
    trainer.auto_sampler = budgeted
    trainer.auto_scheduler_fingerprint = config_fingerprint(env.configs[0])
    trainer.auto_axes = dict(CO_PHYSICAL_AXES)
    trainer.auto_legacy_axes = dict(LEGACY_MIXED_AXES)
    trainer.auto_env = env
    trainer.auto_run_dir = Path(ckpt_dir).resolve().parent
    trainer.auto_run_kind = str(run_kind)
    trainer.auto_method_id = "margo_automotive_v2_%s" % str(run_kind)
    trainer.auto_decoding = str(decoding)
    trainer.auto_protocol_id = PROTOCOL_ID
    trainer.auto_obs_version = V2_OBS_VERSION
    trainer.auto_v2_system = {"link_regime": str(link_regime), "mec_workers": int(mec_workers),
                              "background_dags": int(background_dags),
                              "reliability": bool(reliability),
                              "helper_contact_mean_s": float(helper_contact_mean_s),
                              "helper_contact_cv": float(helper_contact_cv),
                              "helper_busy_s": float(helper_busy_s)}
    trainer.auto_crn = {"protocol_id": PROTOCOL_ID, "r_select": int(r_select),
                        "s_select": int(s_select)}
    return trainer, algo


from spec.automotive_training.automotive_primary import AutomotiveHeldOutEvaluator  # noqa: E402


class V2HeldOutEvaluator(AutomotiveHeldOutEvaluator):
    """Held-out evaluator carrying the v2 system configuration and the CRN protocol."""

    def __init__(self, *args, link_regime: str = "stable", mec_workers: int = 1,
                 reliability: bool = False, r_select: int = DEFAULT_R_SELECT,
                 s_select: int = DEFAULT_S_SELECT, **kwargs):
        super().__init__(*args, **kwargs)
        #: proof-of-routing flag: the injected factory must be the one this stack built
        self.uses_injected_env_factory = bool(kwargs.get("env_factory") is not None)
        self.v2_system = {"link_regime": link_regime, "mec_workers": int(mec_workers),
                          "reliability": bool(reliability)}
        self.crn_protocol = {"protocol_id": PROTOCOL_ID, "r_select": int(r_select),
                             "s_select": int(s_select)}
        self.obs_version = V2_OBS_VERSION
