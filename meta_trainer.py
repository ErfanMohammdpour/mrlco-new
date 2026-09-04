import tensorflow as tf
import numpy as np
import time
from utils import logger
from automated_reporting import create_training_report

from spec.eval_protocol import protocol_log_kvs

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
                held_out_evaluator=None):
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
        if self.best_val_composite is None or composite > self.best_val_composite:
            self.best_val_composite = composite
            self.policy.core_policy.save_variables(
                save_path="./meta_model_inner_step1/meta_model_best_val.ckpt"
            )
            logger.logkv("checkpoint_is_best_val", 1)
        else:
            logger.logkv("checkpoint_is_best_val", 0)
        self.algo.sync_task_policies_from_core()
        return k0, k3

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
            self.algo.sync_task_policies_from_core()
            logger.log("Sampling set of tasks/goals for this meta-batch...")

            task_specs = self.sampler.update_tasks()
            paths = self.sampler.obtain_samples(log=False, log_prefix='')

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

            policy_losses, value_losses = self.algo.UpdatePPOTarget(samples_data, batch_size=self.inner_batch_size )

            print("average task losses: ", np.mean(policy_losses))
            avg_loss.append(np.mean(policy_losses))
            policy_losses_all.append(np.mean(policy_losses))

            print("average value losses: ", np.mean(value_losses))
            value_losses_all.append(np.mean(value_losses))

            logger.log("Evaluating adapted task policies on a fresh support sample")
            new_paths = self.sampler.obtain_samples(log=True, log_prefix='')
            new_samples_data = self.sampler_processor.process_samples(new_paths, log="all", log_prefix='')

            logger.log("Optimizing policy...")
            self.algo.UpdateMetaPolicy()

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

            if itr % self.validation_interval == 0:
                self._run_validation(itr)

            logger.dumpkvs()
            avg_ret.append(avg_reward)

            if itr % self.save_interval == 0:
                self.policy.core_policy.save_variables(save_path="./meta_model_inner_step1/meta_model_"+str(itr)+".ckpt")

        self.policy.core_policy.save_variables(save_path="./meta_model_inner_step1/meta_model_final.ckpt")

        try:
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


if __name__ == "__main__":
    from env.mec_offloaing_envs.offloading_env import Resources
    from env.mec_offloaing_envs.offloading_env import OffloadingEnvironment
    from policies.meta_seq2seq_policy import MetaSeq2SeqPolicy
    from samplers.seq2seq_meta_sampler import Seq2SeqMetaSampler
    from samplers.seq2seq_meta_sampler_process import Seq2SeqMetaSamplerProcessor
    from baselines.vf_baseline import ValueFunctionBaseline
    from meta_algos.MRLCO import MRLCO

    import os
    if os.environ.get("MARGO_ALLOW_GPU") != "1":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    from spec.split_loader import (
        assert_held_out_prefixes,
        assert_train_prefixes,
        meta_train_graph_prefixes,
        validation_graph_prefixes,
    )
    from policies.meta_seq2seq_policy import MetaSeq2SeqPolicy, Seq2SeqPolicy
    from samplers.seq2seq_sampler import Seq2SeqSampler
    from samplers.seq2seq_sampler_process import Seq2SeSamplerProcessor
    from meta_algos.ppo_offloading import PPO
    from meta_algos.held_out_eval import HeldOutQueryEvaluator

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    logger.configure(dir="./meta_offloading20_log-inner_step1/", format_strs=['stdout', 'log', 'csv'])

    SEED = 0
    np.random.seed(SEED)
    tf.compat.v1.set_random_seed(SEED)

    META_BATCH_SIZE = 10
    K_STEPS = 3
    PPO_BATCH_SIZE = 20
    SUPPORT_GRAPHS = 20
    VALIDATION_INTERVAL = 50
    
    # Control flags for printing
    PRINT_ACTION_CHOICES = True  # Set to True to print action choices (0=local, 1=MEC, 2=V2V)
    ACTION_PRINT_INTERVAL = 0   # Print action choices every N iterations (0 = every iteration)

    # ========== ENERGY CONFIGURATION ==========
    # Set to True to enable energy optimization alongside latency
    USE_ENERGY = True
    
    ENERGY_CONFIG = {
        'use_energy': USE_ENERGY,
        'energy_weight': 0.5,      # Weight for energy in combined reward
        'latency_weight': 0.5,     # Weight for latency in combined reward
        'rho': 1.0,                # Computation energy coefficient
        'f_l': 1.0,                # Local CPU frequency (normalized)
        'zeta': 2.0,               # CPU frequency exponent
        'ptx': 0.1,                # Transmission power (Watts)
        'prx': 0.05,               # Reception power (Watts)
        # V2V-specific parameters
        'ptx_v2v': 0.06,           # V2V transmission power (Watts, typically < ptx)
        'prx_v2v': 0.03,           # V2V reception power (Watts, typically < prx)
        'rho_v2v': 0.7,            # V2V computation energy coefficient (70% of local)
        'f_v2v': 1.0,              # V2V CPU frequency (normalized, same as local)
        'normalize_energy': True,   # Whether to normalize energy rewards
    }
    # ==========================================
    
    resource_cluster = Resources(mec_process_capable=(10.0 * 1024 * 1024),
                                 mobile_process_capable=(1.0 * 1024 * 1024),
                                 bandwidth_up=7.0, bandwidth_dl=7.0,
                                 v2v_process_capable=(1.0 * 1024 * 1024),  # Same as UE
                                 v2v_bandwidth=5.0,  # Lower than MEC
                                 use_energy=USE_ENERGY,
                                 energy_config=ENERGY_CONFIG)

    train_paths = meta_train_graph_prefixes()
    assert_train_prefixes(train_paths)

    env = OffloadingEnvironment(resource_cluster=resource_cluster,
                                batch_size=100,
                                graph_number=100,
                                graph_file_paths=train_paths,
                                time_major=False)
    env.support_graphs_per_task = SUPPORT_GRAPHS

    # Get greedy solution (with energy if enabled)
    greedy_result = env.greedy_solution()
    if env.resource_cluster.use_energy:
        action, greedy_finish_time, greedy_energy = greedy_result
        # Flatten finish times and energy for averaging
        flat_finish_times = [item for sublist in greedy_finish_time for item in sublist]
        flat_energy = [item for sublist in greedy_energy for item in sublist]
        print("avg greedy solution latency: ", np.mean(flat_finish_times))
        print("avg greedy solution energy: ", np.mean(flat_energy))
    else:
        action, greedy_finish_time = greedy_result
        # Flatten finish times for averaging
        flat_finish_times = [item for sublist in greedy_finish_time for item in sublist]
        print("avg greedy solution: ", np.mean(flat_finish_times))
    print()
    finish_time = env.get_all_mec_execute_time()
    print("avg all remote solution: ", np.mean(finish_time))
    print()
    finish_time = env.get_all_locally_execute_time()
    print("avg all local solution: ", np.mean(finish_time))
    print()
    finish_time = env.get_all_v2v_execute_time()
    print("avg all V2V solution: ", np.mean(finish_time))
    print()

    baseline = ValueFunctionBaseline()

    meta_policy = MetaSeq2SeqPolicy(meta_batch_size=META_BATCH_SIZE, obs_dim=env.input_dim, encoder_units=128, decoder_units=128,
                                    vocab_size=3)

    sampler = Seq2SeqMetaSampler(
        env=env,
        policy=meta_policy,
        rollouts_per_meta_task=1,  # This batch_size is confusing
        meta_batch_size=META_BATCH_SIZE,
        max_path_length=20000,
        parallel=False,
    )

    sample_processor = Seq2SeqMetaSamplerProcessor(baseline=baseline,
                                                   discount=0.99,
                                                   gae_lambda=0.95,
                                                   normalize_adv=True,
                                                   positive_adv=False)
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
                         rng=np.random.RandomState(SEED))

    val_paths = validation_graph_prefixes()
    assert_held_out_prefixes(val_paths, "validation")
    val_env = OffloadingEnvironment(resource_cluster=resource_cluster,
                                    batch_size=100,
                                    graph_number=100,
                                    graph_file_paths=val_paths,
                                    time_major=False)
    val_policy = Seq2SeqPolicy(obs_dim=env.input_dim,
                               encoder_units=128,
                               decoder_units=128,
                               vocab_size=3,
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
                                           normalize_adv=True,
                                           positive_adv=False)
    val_ppo = PPO(policy=val_policy,
                  meta_sampler=val_sampler,
                  meta_sampler_process=val_processor,
                  lr=5e-4,
                  num_inner_grad_steps=K_STEPS,
                  clip_value=0.2,
                  max_grad_norm=0.5,
                  rng=np.random.RandomState(SEED + 1))
    held_out = HeldOutQueryEvaluator(
        env=val_env,
        policy=val_policy,
        sampler=val_sampler,
        processor=val_processor,
        ppo=val_ppo,
        source_policy=meta_policy.core_policy,
        ppo_batch_size=PPO_BATCH_SIZE,
    )

    trainer = Trainer(algo = algo,
                        env=env,
                        sampler=sampler,
                        sample_processor=sample_processor,
                        policy=meta_policy,
                        n_itr=3500,
                        greedy_finish_time= greedy_finish_time,
                        start_itr=0,
                        inner_batch_size=PPO_BATCH_SIZE,
                        print_action_choices=PRINT_ACTION_CHOICES,
                        action_print_interval=ACTION_PRINT_INTERVAL,
                        seed=SEED,
                        validation_interval=VALIDATION_INTERVAL,
                        held_out_evaluator=held_out)

    with tf.compat.v1.Session() as sess:
        sess.run(tf.global_variables_initializer())
        algo.sync_task_policies_from_core()
        avg_ret, avg_loss, avg_latencies = trainer.train()


