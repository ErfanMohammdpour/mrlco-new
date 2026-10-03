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
from spec.automotive_training.v2.adapters import config_for
from spec.automotive_training.v2.constraints_v2 import constraints_fingerprint
from spec.automotive_training.v2.crn import DEFAULT_R_SELECT, DEFAULT_S_SELECT, PROTOCOL_ID
from spec.automotive_training.v2.env import V2AutomotiveEnv
from spec.automotive_training.v2.observation import V2_OBS_VERSION

V2_ALLOWED_LINK_REGIMES = ("stable", "moderate", "degraded")


def _iter_v2_telemetry(paths):
    """Yield every v2 telemetry record carried by the rollout paths."""
    if not paths:
        return
    for task_paths in paths.values():
        for path in task_paths:
            telemetry = path.get("energy_telemetry")
            if isinstance(telemetry, Mapping):
                yield telemetry
            elif isinstance(telemetry, (list, tuple)):
                for record in telemetry:
                    if isinstance(record, Mapping):
                        yield record


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
                              background_dags: int = 0,
                              background_policy: str = "all_mec",
                              constraints_enabled: bool = True,
                              budget_fractions: Mapping | None = None):
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

    fractions = dict(budget_fractions or {"total_energy": 0.5, "ue_energy": 1.0})

    def make_env(role: str) -> V2AutomotiveEnv:
        env = V2AutomotiveEnv(train_graphs, cluster, role=role,
                              slots_per_task=int(support_trajectories), base_seed=seed,
                              link_regime=str(link_regime), mec_workers=int(mec_workers),
                              reliability=bool(reliability),
                              helper_contact_mean_s=float(helper_contact_mean_s),
                              helper_contact_cv=float(helper_contact_cv),
                              helper_busy_s=float(helper_busy_s),
                              background_dags=int(background_dags),
                              background_policy=str(background_policy),
                              # THE constraint channel the v2 environment actually penalises on
                              constraints_enabled=bool(constraints_enabled),
                              budget_fractions=fractions,
                              scheduler_config_sha256=config_fingerprint(
                                  config_for(train_graphs[0])))
        if constraints_enabled and env.constraint_manager is not None:
            # The v2 manager is both the penalty authority (`env.step` applies ITS lambda) and
            # the object the frozen training loop dual-steps, so the SAME multipliers drive the
            # reward and the ascent. The legacy v1 adapter stays attached only for the
            # deadline-channel telemetry the v2 stack still emits.
            env.constraint_controller = env.constraint_manager
        else:
            env.constraint_controller = None
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
        """Validation env factory: v2 dynamics and the SAME canonical world builder."""
        return V2AutomotiveEnv(list(graphs), AutomotiveResourceCluster(), role="validation",
                               slots_per_task=int(slots), base_seed=int(seed),
                               single_dist=bool(single_dist),
                               link_regime=str(link_regime), mec_workers=int(mec_workers),
                               reliability=bool(reliability),
                               helper_contact_mean_s=float(helper_contact_mean_s),
                               helper_contact_cv=float(helper_contact_cv),
                               helper_busy_s=float(helper_busy_s),
                               background_dags=int(background_dags),
                               background_policy=str(background_policy))

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

    class AutomotiveV2Trainer(AutomotiveTrainerMixin, Trainer):
        """v2 trainer: the v2 SIGNED costs feed the v2 manager that applies the penalty.

        The inherited observer reads `violation/<NAME>` from the telemetry, which the v2 stack
        now emits, but the channel it must drive is the v2 manager (the object whose lambda
        `env.step` actually subtracts from the reward). Overriding here keeps one authority for
        the penalty and the dual ascent.
        """

        def constraint_observer(self, samples_data=None, task_specs=None, paths=None) -> dict:
            manager = getattr(self, "auto_v2_constraint_manager", None)
            if manager is None:
                return super().constraint_observer(samples_data, task_specs, paths)
            rows, penalties = [], []
            records = list(_iter_v2_telemetry(paths))
            # Stash the ACTUAL rollout telemetry: the vectorised executor runs the rollouts on
            # `copy.deepcopy(env)` clones, so neither `trainer.env` nor the sampler tree exposes
            # `last_telemetry`; the rollout PATH is the production carrier of these records.
            self.auto_last_telemetry_records = records[-8:]
            self.auto_telemetry_records_seen = int(
                getattr(self, "auto_telemetry_records_seen", 0)) + len(records)
            measured = [r for r in records if r.get("system_joules") is not None]
            if measured:
                self.auto_energy_last_iteration = float(
                    sum(float(r["system_joules"]) for r in measured) / len(measured))
                self.auto_requester_last_iteration = float(
                    sum(float(r.get("requester_joules", 0.0)) for r in measured) / len(measured))
                self.auto_violation_last_iteration = float(sum(
                    float((r.get("constraints") or {}).get("%s_violation" % n, 0.0))
                    for r in measured
                    for n in (r.get("constraints") or {}).get("names", [])) / len(measured))
            for record in records:
                block = record.get("constraints")
                if not isinstance(block, Mapping) or not block.get("enabled"):
                    continue
                names = list(block.get("names") or [])
                rows.append({name: float(block.get("%s_signed" % name, 0.0))
                             for name in names})
                penalties.append(float(record.get("constraint_penalty", 0.0) or 0.0))
            accepted = manager.observe_signed(rows)
            self.auto_v2_signed_rows = int(getattr(self, "auto_v2_signed_rows", 0)) + accepted
            self.auto_penalty_episodes = int(getattr(self, "auto_penalty_episodes", 0)) + len(penalties)
            self.auto_penalty_episodes_nonzero = int(
                getattr(self, "auto_penalty_episodes_nonzero", 0)
                + sum(1 for value in penalties if abs(value) > 1e-12))
            return {name: [row.get(name, 0.0) for row in rows]
                    for name in (rows[0] if rows else {})}

        def broadcast_constraint_lambdas(self) -> int:
            """Broadcast the v2 manager's multipliers — the ONE dual owner.

            The inherited implementation reads `self.auto_controller`, which still held the
            legacy v1 DEADLINE-channel adapter, so it pushed channel names the v2 manager does
            not know and the broadcast raised. The v2 manager is the authority whose lambda
            `env.step` subtracts, so it is broadcast directly to the environment and to every
            executor clone.
            """
            manager = getattr(self, "auto_v2_constraint_manager", None)
            if manager is None:
                return super().broadcast_constraint_lambdas()
            vector = manager.lambdas_by_name()
            updated = 0
            env = getattr(self, "auto_env", None)
            if env is not None and hasattr(env, "set_constraint_lambdas"):
                env.set_constraint_lambdas(vector)
                updated += 1
            sampler = getattr(self, "meta_sampler", None) or getattr(self, "sampler", None)
            executor = getattr(sampler, "vec_env", None)
            if executor is not None and hasattr(executor, "set_constraint_lambdas"):
                updated += int(executor.set_constraint_lambdas(vector))
            self.auto_lambda_broadcast_targets = updated
            self.auto_lambda_broadcast_values = dict(vector)
            if updated == 0:
                raise RuntimeError(
                    "no environment accepted the v2 constraint multipliers; the Lagrangian "
                    "feedback cannot reach the rollouts")
            return updated

    AutomotiveTrainerImpl = AutomotiveV2Trainer
    trainer = AutomotiveTrainerImpl(
        algo=algo, env=env, sampler=budgeted, sample_processor=processor,
        policy=meta_policy, n_itr=int(n_itr), greedy_finish_time=greedy_finish,
        start_itr=0, inner_batch_size=int(support_trajectories),
        print_action_choices=False, action_print_interval=0, seed=seed,
        validation_interval=50, held_out_evaluator=held_out, ckpt_dir=ckpt_dir,
        audit_writer=None)
    trainer.auto_spec_map = {k: spec_map[k] for k in enabled}
    if env.constraint_manager is not None:
        # single authority: everything the frozen loop calls a "controller" is the object whose
        # lambda the environment actually applies
        trainer.auto_controller = env.constraint_manager
    trainer.auto_v2_constraint_manager = env.constraint_manager
    trainer.auto_v2_constraints = {
        "enabled": bool(constraints_enabled and env.constraint_manager is not None),
        "budget_fractions": dict(fractions),
        "names": list(env.constraint_manager.names) if env.constraint_manager else [],
        "lambdas": (env.constraint_manager.lambdas_by_name()
                    if env.constraint_manager else {}),
        "spec_sha256": (constraints_fingerprint(env.constraint_manager.spec)
                        if env.constraint_manager else None),
        "objectives": {"primary": "constrained_latency",
                       "weighted_ablation": "w_T*T/T_ref + w_E*E/E_ref"},
    }
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
                              "background_policy": str(background_policy),
                              "constraints_enabled": bool(constraints_enabled),
                              "budget_fractions": dict(fractions),
                              "reliability": bool(reliability),
                              "helper_contact_mean_s": float(helper_contact_mean_s),
                              "helper_contact_cv": float(helper_contact_cv),
                              "helper_busy_s": float(helper_busy_s)}
    trainer.auto_crn = {"protocol_id": PROTOCOL_ID, "r_select": int(r_select),
                        "s_select": int(s_select)}
    return trainer, algo


from spec.automotive_training.automotive_primary import AutomotiveHeldOutEvaluator  # noqa: E402


class V2HeldOutEvaluator(AutomotiveHeldOutEvaluator):
    """v2 held-out evaluator: every baseline is scored on the CANONICAL v2 world.

    The inherited `baseline_panel` calls `query_env._schedule(...)`, which `V2AutomotiveEnv`
    does not define — `__getattr__` then delegated it to the FROZEN V1 ENGINE, so a v2 run
    silently scored its baselines with v1 dynamics. `baseline_panel` is overridden here to
    score each plan on the same v2 world/scheduler/energy model as the policy, and
    `_schedule` is trapped while that panel runs so any remaining delegation raises instead of
    silently returning v1 numbers.
    """

    def baseline_panel(self, query_env, epoch: int | None = None) -> dict:
        """v2 baselines in the SAME schema the frozen consumer reads.

        The inherited panel called `query_env._schedule(...)`; `V2AutomotiveEnv` does not define
        it, so `__getattr__` delegated the call to the FROZEN V1 ENGINE and the run silently
        scored its baselines with v1 dynamics. Here every candidate is scored on the canonical
        v2 world by the v2 scheduler, and the entry schema (`all_UE`/`all_MEC`/`all_HELPER`/
        `greedy_coordinate_descent_MC`/`heft_reference_v2`, `heft_nominal_s`, `heft_plan`,
        `greedy_plan`, `_name`, `candidate_oracle_s`) matches the frozen consumer exactly.
        """
        from spec.automotive_training.automotive_primary import candidate_oracle
        from spec.automotive_training.v2.adapters import plan_map_from_actions, pure_plan
        from spec.automotive_training.v2.eval_loop import EvalProtocol, baseline_plans, _score
        from spec.automotive_training.v2.heft_bridge import heft_plan_for_graph
        from spec.automotive_training.v2.link_model import make_process
        from spec.automotive_training.v2.world import V2WorldConfig, build_world

        if epoch is not None and epoch in self._panel_cache:
            return self._panel_cache[epoch]
        trap = getattr(query_env, "set_v1_schedule_trap", None)
        if trap is None:
            raise RuntimeError(
                "the v2 evaluator requires a V2AutomotiveEnv able to trap `_schedule`; "
                "refusing to score baselines on a path that can reach the v1 engine")
        graphs = list(self.query_graphs)
        protocol = EvalProtocol(r_select=int(self.crn_protocol["r_select"]),
                               s_select=int(self.crn_protocol["s_select"]),
                               background_dags=int(self.v2_system.get("background_dags", 0)))
        panel: dict = {}
        seed_base = int(epoch if epoch is not None else 0)
        for index, graph in enumerate(graphs):
            trap(True)
            try:
                world = build_world(
                    graph, slot_id=index, mc=None,
                    config=V2WorldConfig(
                        background_dags=int(protocol.background_dags),
                        mec_workers=int(self.v2_system.get("mec_workers", 1))),
                    helper_seed=seed_base + index)
                link_process = None
                if str(self.v2_system.get("link_regime", "stable")) != "stable":
                    link_process = make_process(str(self.v2_system["link_regime"]),
                                                seed=seed_base + index)
                entry: dict = {}
                for name, action in (("all_UE", 0), ("all_MEC", 1), ("all_HELPER", 2)):
                    entry[name] = float(_score(world, pure_plan(graph, action),
                                               link_process=link_process)["episode_latency_s"])
                plans = baseline_plans(world, graph, protocol)
                entry["greedy_coordinate_descent_MC"] = float(
                    _score(world, plans["greedy_cd"], link_process=link_process)
                    ["episode_latency_s"])
                heft_actions, heft_nominal = heft_plan_for_graph(graph)
                entry["heft_reference_v2"] = float(
                    _score(world, plan_map_from_actions(graph, list(heft_actions)),
                           link_process=link_process)["episode_latency_s"])
                entry["heft_nominal_s"] = float(heft_nominal)
                entry["heft_plan"] = [int(a) for a in heft_actions]
                entry["greedy_plan"] = [int(plans["greedy_cd"][t])
                                        for t in sorted(plans["greedy_cd"])]
                entry["_name"], entry["candidate_oracle_s"] = candidate_oracle(entry)
                entry["scoring"] = "v2_canonical_world_foreground_episode_latency"
                panel[index] = entry
            finally:
                trap(False)
        if epoch is not None:
            self._panel_cache[epoch] = panel
        return panel

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
        self.baseline_scoring = "v2_canonical_world"
