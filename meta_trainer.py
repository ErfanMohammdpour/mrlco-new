import os
import tensorflow as tf
import numpy as np
import time
from utils import logger

from env.mec_offloaing_envs.scheduler.mask_metrics import merge as merge_mask_metrics
from env.mec_offloaing_envs.scheduler.mask_metrics import rates as mask_metric_rates
from spec.eval_protocol import protocol_log_kvs
from spec.train_audit import health_verdict, task_spec_records

FROZEN_PPO_BATCH = 20
FROZEN_K_STEPS = 3
FROZEN_VALIDATION_INTERVAL = 50


class Trainer(object):
    def __init__(self, algo,
                env,
                sampler,
                sample_processor,
                policy,
                n_itr,
                greedy_finish_time,
                start_itr=0,
                inner_batch_size=20,
                save_interval=100,
                print_action_choices=False,
                action_print_interval=10,
                seed=0,
                validation_interval=50,
                held_out_evaluator=None,
                ckpt_dir="./meta_model_inner_step1",
                write_training_report=False,
                audit_writer=None,
                critic_warmup_iters=0,
                bc_policy=None):
        if int(inner_batch_size) != FROZEN_PPO_BATCH:
            raise ValueError("v0.1 inner_batch_size / ppo_batch_size_trajectories must be 20")
        if int(validation_interval) != FROZEN_VALIDATION_INTERVAL:
            raise ValueError("v0.1 validation_interval must be 50")
        if held_out_evaluator is None:
            raise ValueError("v0.1 trainer requires held_out_evaluator (validation every 50 outer iters)")
        self.algo = algo
        self.env = env
        self.sampler = sampler
        self.sampler_processor = sample_processor
        self.policy = policy
        self.n_itr = n_itr
        self.start_itr = start_itr
        self.inner_batch_size = int(inner_batch_size)
        self.greedy_finish_time = greedy_finish_time
        self.save_interval = save_interval
        self.print_action_choices = print_action_choices
        self.action_print_interval = action_print_interval
        self.seed = int(seed)
        self.validation_interval = int(validation_interval)
        self.held_out_evaluator = held_out_evaluator
        self.best_val_composite = None
        # ②B-2 lexicographic selection state (None -> legacy composite path)
        self.best_selection_key = None
        self.objective_spec = getattr(self, "objective_spec", None)
        self.objective_reference_summary = None
        self.ckpt_dir = ckpt_dir
        self.write_training_report = bool(write_training_report)
        self.audit_writer = audit_writer
        self.critic_warmup_iters = int(critic_warmup_iters)
        self.bc_policy = bc_policy
        os.makedirs(self.ckpt_dir, exist_ok=True)

    def _mask_metrics(self, samples_data):
        """Token-weighted ⑥b metrics over every meta task of this iteration."""
        accumulators = [
            s.get("mask_accumulator") for s in samples_data
            if isinstance(s, dict) and s.get("mask_accumulator") is not None
        ]
        if not accumulators:
            raise RuntimeError(
                "no mask_accumulator in the processed samples: the ⑥b metrics "
                "must be computed from the stored rollout mask and raw logits"
            )
        return mask_metric_rates(merge_mask_metrics(accumulators))

    def _audit(self, itr, name, payload=None):
        if self.audit_writer is None:
            return
        self.audit_writer.stage(itr, name, payload or {})

    def _core_vars(self):
        sess = tf.compat.v1.get_default_session()
        return sess.run(self.policy.core_policy.get_trainable_variables())

    def _task_vars(self):
        sess = tf.compat.v1.get_default_session()
        out = []
        for task_id in range(self.algo.meta_batch_size):
            out.append(sess.run(self.policy.meta_policies[task_id].get_trainable_variables()))
        return out

    def _param_l2(self, a, b):
        total = 0.0
        for x, y in zip(a, b):
            d = np.asarray(x, dtype=np.float64) - np.asarray(y, dtype=np.float64)
            total += float(np.sum(d * d))
        return total ** 0.5

    def _ckpt_path(self, name):
        return os.path.join(self.ckpt_dir, name)

    def _log_protocol_fields(self, itr, k_steps):
        for key, value in protocol_log_kvs(
            seed=self.seed,
            k_steps=k_steps,
            outer_update_count=itr + 1,
        ).items():
            logger.logkv(key, value)

    def _run_validation(self, itr):
        if self.held_out_evaluator is None:
            raise RuntimeError("validation_interval elapsed but held_out_evaluator is missing")
        k0 = self.held_out_evaluator.evaluate_all(k_steps=0)
        k3 = self.held_out_evaluator.evaluate_all(k_steps=3)
        logger.logkv("validation_query_composite_objective_k0", k0["validation_query_composite_objective"])
        logger.logkv("validation_query_composite_objective", k3["validation_query_composite_objective"])
        logger.logkv("validation_query_mean_latency_k0", k0["query_mean_latency"])
        logger.logkv("validation_query_mean_latency_k3", k3["query_mean_latency"])
        logger.logkv("checkpoint_selection_metric", "validation_query_composite_objective")
        composite = k3["validation_query_composite_objective"]
        # --- ②B: plan-level objective + lexicographic feasibility gate -------
        # objective_mode="off" (default) keeps the legacy composite path exactly.
        # "log_only" logs J and the constraint channels without changing the winner.
        # "lexicographic" applies: feasible checkpoints first, then min J.
        objective_mode = str(getattr(self, "objective_mode", "off"))
        lexicographic_winner = None
        if objective_mode != "off" and self.objective_spec is not None:
            from spec.objective_selection import objective_log_kvs

            obj = self._validation_plan_objective(k3)
            if obj is None:
                logger.logkv("objective/unavailable", 1)
                if objective_mode == "lexicographic":
                    # never fall back to the legacy scalar silently
                    logger.logkv("checkpoint_is_best_val_lexicographic", 0)
            else:
                for key, value in objective_log_kvs(obj).items():
                    logger.logkv(key, value)
                if objective_mode == "lexicographic":
                    key_now = obj.selection_key()
                    if self.best_selection_key is None or key_now < self.best_selection_key:
                        self.best_selection_key = key_now
                        lexicographic_winner = True
                    else:
                        lexicographic_winner = False
                    logger.logkv("checkpoint_is_best_val_lexicographic",
                                 1 if lexicographic_winner else 0)
        if objective_mode == "lexicographic" and lexicographic_winner is not None:
            save = bool(lexicographic_winner)
        else:
            save = self.best_val_composite is None or composite > self.best_val_composite
        if save:
            self.best_val_composite = max(
                composite, self.best_val_composite if self.best_val_composite is not None else composite
            )
            self.policy.core_policy.save_variables(
                save_path=self._ckpt_path("meta_model_best_val.ckpt")
            )
            logger.logkv("checkpoint_is_best_val", 1)
        else:
            logger.logkv("checkpoint_is_best_val", 0)
        self.algo.sync_task_policies_from_core()
        return k0, k3

    def _validation_plan_objective(self, metrics):
        """②B: per-graph objective aggregation for one validation measurement.

        Requires per-graph payloads: `validation_per_graph_plans` = list of
        (ScheduleResult, ReferenceRanges). Ratio-of-means is deliberately NOT
        supported: mean energy under mean budget can hide a per-episode violation.
        Returns None when the evaluator does not provide per-graph results, in
        which case lexicographic selection refuses to pick a winner.
        """
        from spec.objective_selection import objective_from_plans

        plans = metrics.get("validation_per_graph_plans")
        if not plans:
            return None
        return objective_from_plans(plans, self.objective_spec)

    def train(self):
        """MRLCO training: inner k=3 on support, one outer mean-PG, val every 50."""

        start_time = time.time()
        avg_ret = []
        avg_loss = []
        avg_latencies = []
        
        policy_losses_all = []
        value_losses_all = []
        greedy_latencies_all = []
        avg_energies = []
        for itr in range(self.start_itr, self.n_itr):
            itr_start_time = time.time()
            logger.log("\n ---------------- Iteration %d ----------------" % itr)
            if self.audit_writer is not None:
                self.audit_writer.begin_iter(itr)
            self.algo.sync_task_policies_from_core()
            theta0 = self._core_vars() if self.audit_writer is not None else None
            self._audit(itr, "sync_task_policies_from_core")
            logger.log("Sampling set of tasks/goals for this meta-batch...")

            task_specs = self.sampler.update_tasks()
            self._audit(itr, "sample_tasks", {"tasks": task_spec_records(task_specs, self.env)})
            if self.audit_writer is not None:
                self.audit_writer.dump_graphs(itr, self.env, task_specs)
            paths = self.sampler.obtain_samples(log=False, log_prefix='')
            ppo_summary = None
            if self.audit_writer is not None:
                _, ppo_summary = self.audit_writer.dump_paths(
                    itr, "trajs_ppo.jsonl", paths, task_specs, self.env
                )
                self._audit(itr, "obtain_samples_ppo", ppo_summary)

            if self.print_action_choices and (self.action_print_interval == 0 or itr == 0 or itr % self.action_print_interval == 0):
                all_actions = []
                for task_paths in paths.values():
                    for path in task_paths:
                        if 'actions' in path:
                            actions = path['actions']
                            if isinstance(actions, np.ndarray):
                                all_actions.extend(actions.flatten())
                            else:
                                all_actions.extend(actions)
                
                if len(all_actions) > 0:
                    all_actions = np.array(all_actions)
                    action_counts = {
                        'Local (0)': np.sum(all_actions == 0),
                        'MEC (1)': np.sum(all_actions == 1),
                        'V2V (2)': np.sum(all_actions == 2)
                    }
                    total = len(all_actions)
                    print(f"\n[Action Choices - Iteration {itr}]")
                    print(f"  Total actions: {total}")
                    for action_name, count in action_counts.items():
                        percentage = (count / total * 100) if total > 0 else 0
                        print(f"  {action_name}: {count} ({percentage:.1f}%)")

            greedy_run_time = []
            for spec in task_specs:
                dist_index = spec["dist_index"] if isinstance(spec, dict) else spec
                greedy_run_time.append(self.greedy_finish_time[dist_index])
            logger.logkv('Average greedy latency,', np.mean(greedy_run_time))
            greedy_latencies_all.append(np.mean(greedy_run_time))

            logger.log("Processing samples...")
            samples_data = self.sampler_processor.process_samples(paths, log=False, log_prefix='')
            # ⑥b metrics: merge the per-meta-task accumulators (token counts, so
            # unequal path lengths cannot skew the rates) and log once per itr.
            mask_metrics = self._mask_metrics(samples_data)
            self._audit(itr, "process_samples_ppo", {
                "n_tasks": len(samples_data),
                "adv_mean": float(np.mean([np.mean(s["advantages"]) for s in samples_data])),
                "adv_std": float(np.mean([np.std(s["advantages"]) for s in samples_data])),
                "return_mean": float(np.mean([np.mean(s["returns"]) for s in samples_data])),
            })

            update_mode = "publication"
            if float(getattr(self.algo, "bc_kl_coef", 0.0) or 0.0) > 0:
                if itr < self.critic_warmup_iters:
                    update_mode = "vf_only"
                else:
                    update_mode = "kl_bc"
            ppo_kwargs = {"batch_size": self.inner_batch_size}
            if update_mode != "publication":
                ppo_kwargs["update_mode"] = update_mode
                ppo_kwargs["bc_policy"] = self.bc_policy
            policy_losses, value_losses = self.algo.UpdatePPOTarget(samples_data, **ppo_kwargs)
            for metric_name, metric_value in mask_metrics.items():
                logger.logkv(metric_name, float(metric_value))
            inner_payload = {
                "policy_losses": policy_losses,
                "value_losses": value_losses,
                "mask_metrics": mask_metrics,
                "policy_loss_mean": float(np.mean(policy_losses)),
                "value_loss_mean": float(np.mean(value_losses)),
                "k_steps": FROZEN_K_STEPS,
                "inner_update_mode": update_mode,
            }
            kl_vals = getattr(self.algo, "last_kl_bc", None)
            if kl_vals:
                inner_payload["kl_bc_mean"] = float(np.mean(kl_vals))
                logger.logkv("kl_bc_mean", inner_payload["kl_bc_mean"])
            logger.logkv("inner_update_mode", update_mode)
            self._audit(itr, "inner_ppo", inner_payload)

            print("average task losses: ", np.mean(policy_losses))
            avg_loss.append(np.mean(policy_losses))
            policy_losses_all.append(np.mean(policy_losses))

            print("average value losses: ", np.mean(value_losses))
            value_losses_all.append(np.mean(value_losses))

            logger.log("Evaluating adapted task policies on a fresh support sample")
            new_paths = self.sampler.obtain_samples(log=True, log_prefix='')
            eval_summary = None
            if self.audit_writer is not None:
                _, eval_summary = self.audit_writer.dump_paths(
                    itr, "trajs_eval.jsonl", new_paths, task_specs, self.env
                )
                self._audit(itr, "obtain_samples_eval", eval_summary)
            new_samples_data = self.sampler_processor.process_samples(new_paths, log="all", log_prefix='')
            self._audit(itr, "process_samples_eval")

            logger.log("Optimizing policy...")
            adapted = self._task_vars() if self.audit_writer is not None else None
            self.algo.UpdateMetaPolicy()
            if self.audit_writer is not None:
                theta1 = self._core_vars()
                self._audit(itr, "outer_update", {
                    "core_delta_l2": self._param_l2(theta0, theta1),
                    "mean_adapt_l2": float(np.mean([self._param_l2(theta0, th) for th in adapted])),
                })

            ret = np.array([])
            for i in range(len(new_samples_data)):
                ret = np.concatenate((ret, np.sum(new_samples_data[i]['rewards'], axis=-1)), axis=-1)

            avg_reward = np.mean(ret)

            latency = np.array([])
            for i in range(len(new_samples_data)):
                latency = np.concatenate((latency, new_samples_data[i]['finish_time']), axis=-1)

            avg_latency = np.mean(latency)
            avg_latencies.append(avg_latency)

            if self.env.resource_cluster.use_energy:
                energy = np.array([])
                for i in range(len(new_samples_data)):
                    if 'energy' in new_samples_data[i]:
                        energy = np.concatenate((energy, np.sum(new_samples_data[i]['energy'], axis=-1)), axis=-1)
                if len(energy) > 0:
                    avg_energy = np.mean(energy)
                    print(f"Average energy per iteration {itr}: {avg_energy:.4f}")
                    logger.logkv('Average energy,', avg_energy)
                    avg_energies.append(avg_energy)
                else:
                    print(f"Average energy per iteration {itr}: 0.0 (no energy data)")
                    avg_energies.append(0.0)
            else:
                avg_energies.append(None)

            logger.logkv('Itr', itr)
            logger.logkv('Average reward, ', avg_reward)
            logger.logkv('Average latency,', avg_latency)
            logger.logkv('split_role', 'meta_train_support')
            self._log_protocol_fields(itr, FROZEN_K_STEPS)

            # --- Lagrangian dual ascent (constrained V2V/energy budgets) ----
            # The env observed every trajectory's signed constraint cost during
            # this iteration; one dual step per outer iteration closes the loop.
            # Inert (no log keys, no state change) when constraints are off.
            controller = getattr(self.env, "constraint_controller", None)
            if controller is not None and controller.spec.enabled:
                diag = controller.dual_step()
                for key, value in diag.items():
                    logger.logkv(key, value)
                last_costs = getattr(self.env, "last_constraint_costs", None)
                if last_costs is not None and last_costs.active:
                    logger.logkv("constraint/total_violation", last_costs.total_violation)
                    self._audit(itr, "constraints", {
                        "lambdas": {n: l for n, l in zip(controller.names, controller.lambdas)},
                        "costs": last_costs.as_dict(),
                    })

            if itr % self.validation_interval == 0:
                k0, k3 = self._run_validation(itr)
                self._audit(itr, "validation", {"k0": k0, "k3": k3})

            logger.dumpkvs()
            avg_ret.append(avg_reward)

            if self.audit_writer is not None:
                health_src = eval_summary if eval_summary is not None else (ppo_summary or {})
                health = health_verdict(
                    health_src,
                    inner_policy_losses=policy_losses,
                    inner_value_losses=value_losses,
                    k_steps=FROZEN_K_STEPS,
                )
                self.audit_writer.finish_iter(itr, {
                    "itr": itr,
                    "wall_s": time.time() - itr_start_time,
                    "avg_reward": float(avg_reward),
                    "avg_latency": float(avg_latency),
                    "ppo_sample": ppo_summary,
                    "eval_sample": eval_summary,
                    "inner": inner_payload,
                    "health": health,
                })
                logger.log("audit health itr %d ok=%s flags=%s" % (itr, health["ok"], health["flags"]))

            if itr % self.save_interval == 0:
                self.policy.core_policy.save_variables(
                    save_path=self._ckpt_path("meta_model_" + str(itr) + ".ckpt")
                )

        self.policy.core_policy.save_variables(
            save_path=self._ckpt_path("meta_model_final.ckpt")
        )

        if self.write_training_report:
            try:
                from automated_reporting import create_training_report
                print("\n==================== GENERATING AUTOMATED REPORT ====================")
                additional_metrics = {
                    'policy_losses': policy_losses_all,
                    'value_losses': value_losses_all,
                    'greedy_latencies': greedy_latencies_all
                }

                if self.env.resource_cluster.use_energy and len(avg_energies) > 0:
                    energy_values = [e for e in avg_energies if e is not None]
                    if len(energy_values) > 0:
                        additional_metrics['average_energy'] = energy_values
                        print(f"Added energy metrics to report ({len(energy_values)} iterations)")

                report_dir = create_training_report(
                    avg_ret=avg_ret,
                    avg_loss=avg_loss,
                    avg_latencies=avg_latencies,
                    additional_metrics=additional_metrics
                )
                print(f"Report generated successfully at: {report_dir}")
                print("=====================================================================\n")
            except Exception as e:
                print(f"WARNING: Failed to generate automated report: {str(e)}")
                print("Training completed successfully but report generation failed.")

        return avg_ret, avg_loss, avg_latencies


def build_frozen_primary_stack(seed=0, n_itr=3500, ckpt_dir="./meta_model_inner_step1",
                               audit_writer=None, print_action_choices=None,
                               parallel=False, reward_mode="publication",
                               learning_mode="publication",
                               bc_kl_coef=0.0, critic_warmup_iters=0,
                               vocab_size=3, use_energy=True,
                               constraints=None, constraint_dual_lr=0.05,
                               shaping_discount=0.99,
                               reference_range="candidate_panel",
                               objective_mode="off",
                               objective_spec=None):
    """Frozen v0.1 train+val stack. Caller must set CUDA_VISIBLE_DEVICES before importing TF."""
    from env.mec_offloaing_envs.offloading_env import Resources
    from env.mec_offloaing_envs.offloading_env import OffloadingEnvironment
    from policies.meta_seq2seq_policy import MetaSeq2SeqPolicy
    from samplers.seq2seq_meta_sampler import Seq2SeqMetaSampler
    from samplers.seq2seq_meta_sampler_process import Seq2SeqMetaSamplerProcessor
    from baselines.vf_baseline import ValueFunctionBaseline
    from meta_algos.MRLCO import MRLCO
    from spec.split_loader import (
        assert_held_out_prefixes,
        assert_train_prefixes,
        meta_train_graph_prefixes,
        validation_graph_prefixes,
    )
    from policies.meta_seq2seq_policy import Seq2SeqPolicy
    from samplers.seq2seq_sampler import Seq2SeqSampler
    from samplers.seq2seq_sampler_process import Seq2SeSamplerProcessor
    from meta_algos.ppo_offloading import PPO
    from meta_algos.held_out_eval import HeldOutQueryEvaluator

    SEED = int(seed)
    import random
    random.seed(SEED)
    np.random.seed(SEED)
    tf.compat.v1.set_random_seed(SEED)

    META_BATCH_SIZE = 10
    K_STEPS = 3
    PPO_BATCH_SIZE = 20
    SUPPORT_GRAPHS = 20
    VALIDATION_INTERVAL = 50
    PRINT_ACTION_CHOICES = False if print_action_choices is None else bool(print_action_choices)
    ACTION_PRINT_INTERVAL = 0
    vocab_size = int(vocab_size)
    if vocab_size not in (2, 3):
        raise ValueError("vocab_size must be 2 or 3, got %r" % (vocab_size,))
    USE_ENERGY = bool(use_energy)
    if learning_mode not in ("publication", "pomo_elite", "bc_greedy_mec", "kl_bc_ppo"):
        raise ValueError(
            "learning_mode must be publication, pomo_elite, bc_greedy_mec, or kl_bc_ppo, got %r"
            % (learning_mode,)
        )
    pomo_elite = learning_mode == "pomo_elite"
    support_select = "elite" if pomo_elite else "random"
    kl_coef = float(bc_kl_coef)
    warmup = int(critic_warmup_iters)
    if learning_mode == "kl_bc_ppo":
        if kl_coef <= 0:
            raise ValueError("kl_bc_ppo requires bc_kl_coef > 0")
        if warmup < 1:
            raise ValueError("kl_bc_ppo requires critic_warmup_iters >= 1")
    elif kl_coef != 0.0 or warmup != 0:
        raise ValueError("bc_kl_coef/critic_warmup_iters only valid for kl_bc_ppo")

    ENERGY_CONFIG = {
        'use_energy': USE_ENERGY,
        'reward_mode': str(reward_mode or "publication"),
        'energy_weight': 0.5,
        'latency_weight': 0.5,
        'rho': 1.0,
        'f_l': 1.0,
        'zeta': 2.0,
        'ptx': 0.1,
        'prx': 0.05,
        'ptx_v2v': 0.06,
        'prx_v2v': 0.03,
        'rho_v2v': 0.7,
        'f_v2v': 1.0,
        'normalize_energy': True,
        # Discount-consistent shaping: r_t = J_{t-1} - gamma * J_t with the SAME
        # gamma as PPO/GAE below, so the shaped return aligns with the final
        # schedule objective. 1.0 = historical delta-form rewards.
        'shaping_discount': float(shaping_discount),
        # Normalization reference bounds:
        #   "candidate_panel" (default, corrected): three pure plans + greedy_from_mec
        #   "pure_location"  (MARGO-SPEC-v0.1): reproduces the pre-fix numbers
        'reference_range': str(reference_range),
    }
    if not 0.0 < float(shaping_discount) <= 1.0:
        raise ValueError("shaping_discount must be in (0, 1], got %r" % (shaping_discount,))
    if str(reference_range) not in ("candidate_panel", "pure_location"):
        raise ValueError(
            "reference_range must be candidate_panel or pure_location, got %r"
            % (reference_range,)
        )

    # Constrained-MDP layer (V2V + energy budgets). None / mode="off" keeps the
    # unconstrained primary path byte-for-byte unchanged.
    constraint_controller = None
    if constraints is not None:
        from env.mec_offloaing_envs.scheduler.constraints import (
            ConstraintController,
            ConstraintSpec,
        )

        spec = (
            constraints
            if isinstance(constraints, ConstraintSpec)
            else ConstraintSpec.from_dict(dict(constraints))
        )
        if spec.enabled:
            constraint_controller = ConstraintController(
                spec=spec, dual_lr=float(constraint_dual_lr)
            )
            ENERGY_CONFIG['constraints'] = spec.as_dict()
            print(
                "[constraints] lagrangian mode, budgets=%s, dual_lr=%s"
                % (list(spec.active_names), constraint_dual_lr)
            )

    resource_cluster = Resources(mec_process_capable=(10.0 * 1024 * 1024),
                                 mobile_process_capable=(1.0 * 1024 * 1024),
                                 bandwidth_up=7.0, bandwidth_dl=7.0,
                                 v2v_process_capable=(1.0 * 1024 * 1024),
                                 v2v_bandwidth=5.0,
                                 use_energy=USE_ENERGY,
                                 energy_config=ENERGY_CONFIG)
    resource_cluster.constraint_controller = constraint_controller

    train_paths = meta_train_graph_prefixes()
    assert_train_prefixes(train_paths)

    env = OffloadingEnvironment(resource_cluster=resource_cluster,
                                batch_size=100,
                                graph_number=100,
                                graph_file_paths=train_paths,
                                time_major=False)
    env.support_graphs_per_task = SUPPORT_GRAPHS
    if vocab_size == 2:
        env.greedy_actions = (0, 1)

    greedy_result = env.greedy_solution()
    if env.resource_cluster.use_energy:
        action, greedy_finish_time, greedy_energy = greedy_result
    else:
        action, greedy_finish_time = greedy_result

    baseline = ValueFunctionBaseline()
    meta_policy = MetaSeq2SeqPolicy(meta_batch_size=META_BATCH_SIZE, obs_dim=env.input_dim, encoder_units=128, decoder_units=128,
                                    vocab_size=vocab_size)
    sampler = Seq2SeqMetaSampler(
        env=env,
        policy=meta_policy,
        rollouts_per_meta_task=1,
        meta_batch_size=META_BATCH_SIZE,
        max_path_length=20000,
        parallel=bool(parallel),
    )
    sample_processor = Seq2SeqMetaSamplerProcessor(baseline=baseline,
                                                   discount=0.99,
                                                   gae_lambda=0.95,
                                                   normalize_adv=not pomo_elite,
                                                   positive_adv=False)
    sample_processor.pomo_elite = pomo_elite
    sample_processor.pomo_n_instances = SUPPORT_GRAPHS
    # ⑥b: the processor must know whether a missing mask is an error
    sample_processor.mask_mode = getattr(meta_policy, "mask_mode", None)
    algo = MRLCO(policy=meta_policy,
                         meta_sampler=sampler,
                         meta_sampler_process=sample_processor,
                         inner_lr=5e-4,
                         outer_lr=5e-4,
                         meta_batch_size=META_BATCH_SIZE,
                         num_inner_grad_steps=K_STEPS,
                         clip_value=0.2,
                         value_clip_epsilon=0.2,
                         support_trajectories=SUPPORT_GRAPHS,
                         ppo_batch_size_trajectories=PPO_BATCH_SIZE,
                         rng=np.random.RandomState(SEED),
                         support_select=support_select,
                         bc_kl_coef=kl_coef)
    bc_policy = None
    if learning_mode == "kl_bc_ppo":
        bc_policy = Seq2SeqPolicy(
            obs_dim=env.input_dim,
            encoder_units=128,
            decoder_units=128,
            vocab_size=vocab_size,
            name="bc_frozen",
        )

    val_paths = validation_graph_prefixes()
    assert_held_out_prefixes(val_paths, "validation")
    val_env = OffloadingEnvironment(resource_cluster=resource_cluster,
                                    batch_size=100,
                                    graph_number=100,
                                    graph_file_paths=val_paths,
                                    time_major=False)
    if vocab_size == 2:
        val_env.greedy_actions = (0, 1)
    val_policy = Seq2SeqPolicy(obs_dim=env.input_dim,
                               encoder_units=128,
                               decoder_units=128,
                               vocab_size=vocab_size,
                               name="validation_policy")
    val_sampler = Seq2SeqSampler(val_env,
                                 val_policy,
                                 rollouts_per_meta_task=1,
                                 max_path_length=20000,
                                 envs_per_task=None,
                                 parallel=False)
    val_processor = Seq2SeSamplerProcessor(baseline=ValueFunctionBaseline(),
                                           discount=0.99,
                                           gae_lambda=0.95,
                                           normalize_adv=not pomo_elite,
                                           positive_adv=False)
    val_processor.pomo_elite = pomo_elite
    val_processor.pomo_n_instances = SUPPORT_GRAPHS
    val_processor.mask_mode = getattr(val_policy, "mask_mode", None)
    val_ppo = PPO(policy=val_policy,
                  meta_sampler=val_sampler,
                  meta_sampler_process=val_processor,
                  lr=5e-4,
                  num_inner_grad_steps=K_STEPS,
                  clip_value=0.2,
                  max_grad_norm=0.5,
                  rng=np.random.RandomState(SEED + 1),
                  support_select=support_select)
    held_out = HeldOutQueryEvaluator(
        env=val_env,
        policy=val_policy,
        sampler=val_sampler,
        processor=val_processor,
        ppo=val_ppo,
        source_policy=meta_policy.core_policy,
        ppo_batch_size=PPO_BATCH_SIZE,
    )
    trainer = Trainer(algo=algo,
                        env=env,
                        sampler=sampler,
                        sample_processor=sample_processor,
                        policy=meta_policy,
                        n_itr=int(n_itr),
                        greedy_finish_time=greedy_finish_time,
                        start_itr=0,
                        inner_batch_size=PPO_BATCH_SIZE,
                        print_action_choices=PRINT_ACTION_CHOICES,
                        action_print_interval=ACTION_PRINT_INTERVAL,
                        seed=SEED,
                        validation_interval=VALIDATION_INTERVAL,
                        held_out_evaluator=held_out,
                        ckpt_dir=ckpt_dir,
                        audit_writer=audit_writer,
                        critic_warmup_iters=warmup,
                        bc_policy=bc_policy)
    # ②B: attach the plan-level objective config (log_only / lexicographic).
    if str(objective_mode) not in ("off", "log_only", "lexicographic"):
        raise ValueError(
            "objective_mode must be off, log_only, or lexicographic; got %r"
            % (objective_mode,)
        )
    trainer.objective_mode = str(objective_mode)
    if objective_spec is not None:
        from env.mec_offloaing_envs.scheduler.objective import ObjectiveSpec

        trainer.objective_spec = (
            objective_spec
            if isinstance(objective_spec, ObjectiveSpec)
            else ObjectiveSpec.from_dict(dict(objective_spec))
        )
        print(
            "[objective] mode=%s spec=%s (per-graph evaluation required; "
            "the evaluator must emit validation_per_graph_plans)"
            % (trainer.objective_mode, trainer.objective_spec.as_dict())
        )
    return trainer, algo


if __name__ == "__main__":
    # Legacy CPU entry. Phase 4 GPU train goes through spec/phase4_campaign.py only.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    logger.configure(dir="./meta_offloading20_log-inner_step1/", format_strs=['stdout', 'log', 'csv'])

    # MARGO_CONSTRAINTS=/abs/path/to/constraints.yaml (or an empty value / "off")
    from spec.constraints_config import constraints_from_env

    constraint_spec, dual_lr = constraints_from_env()
    if constraint_spec is not None:
        print("[constraints] loaded from %s" % os.environ.get("MARGO_CONSTRAINTS"))

    trainer, algo = build_frozen_primary_stack(
        seed=0,
        n_itr=3500,
        ckpt_dir="./meta_model_inner_step1",
        constraints=constraint_spec,
        constraint_dual_lr=dual_lr,
    )
    with tf.compat.v1.Session() as sess:
        sess.run(tf.global_variables_initializer())
        algo.sync_task_policies_from_core()
        trainer.train()


