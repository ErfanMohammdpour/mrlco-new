import tensorflow as tf
import numpy as np
import time
from utils import logger
from automated_reporting import create_training_report

class Trainer(object):
    def __init__(self,algo,
                env,
                sampler,
                sample_processor,
                policy,
                n_itr,
                greedy_finish_time,
                start_itr=0,
                inner_batch_size = 500,
                save_interval = 100,
                print_action_choices=False,
                action_print_interval=10):
        self.algo = algo
        self.env = env
        self.sampler = sampler
        self.sampler_processor = sample_processor
        self.policy = policy
        self.n_itr = n_itr
        self.start_itr = start_itr
        self.inner_batch_size = inner_batch_size
        self.greedy_finish_time = greedy_finish_time
        self.save_interval = save_interval
        self.print_action_choices = print_action_choices
        self.action_print_interval = action_print_interval

    def train(self):
        """
        Implement the MRLCO training process for task offloading problem
        """

        start_time = time.time()
        avg_ret = []
        avg_loss = []
        avg_latencies = []
        
        # Additional metrics for comprehensive reporting
        policy_losses_all = []
        value_losses_all = []
        greedy_latencies_all = []
        for itr in range(self.start_itr, self.n_itr):
            itr_start_time = time.time()
            logger.log("\n ---------------- Iteration %d ----------------" % itr)
            logger.log("Sampling set of tasks/goals for this meta-batch...")

            task_ids = self.sampler.update_tasks()
            paths = self.sampler.obtain_samples(log=False, log_prefix='')

            #print("sampled path length is: ", len(paths[0]))

            # Print action choices (0=local, 1=MEC, 2=V2V)
            if self.print_action_choices and (self.action_print_interval == 0 or itr == 0 or itr % self.action_print_interval == 0):
                all_actions = []
                for task_paths in paths.values():  # paths is OrderedDict, iterate over values (lists)
                    for path in task_paths:  # Each task_paths is a list of dictionaries
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

            greedy_run_time = [self.greedy_finish_time[x] for x in task_ids]
            logger.logkv('Average greedy latency,', np.mean(greedy_run_time))
            greedy_latencies_all.append(np.mean(greedy_run_time))

            """ ----------------- Processing Samples ---------------------"""
            logger.log("Processing samples...")
            samples_data = self.sampler_processor.process_samples(paths, log=False, log_prefix='')

            """ ------------------- Inner Policy Update --------------------"""
            policy_losses, value_losses = self.algo.UpdatePPOTarget(samples_data, batch_size=self.inner_batch_size )

            #print("task losses: ", losses)
            print("average task losses: ", np.mean(policy_losses))
            avg_loss.append(np.mean(policy_losses))
            policy_losses_all.append(np.mean(policy_losses))

            print("average value losses: ", np.mean(value_losses))
            value_losses_all.append(np.mean(value_losses))

            """ ------------------ Resample from updated sub-task policy ------------"""
            print("Evaluate the one-step update for sub-task policy")
            new_paths = self.sampler.obtain_samples(log=True, log_prefix='')
            new_samples_data = self.sampler_processor.process_samples(new_paths, log="all", log_prefix='')

            """ ------------------ Outer Policy Update ---------------------"""
            logger.log("Optimizing policy...")
            self.algo.UpdateMetaPolicy()

            """ ------------------- Logging Stuff --------------------------"""

            ret = np.array([])
            for i in range(5):
                ret = np.concatenate((ret, np.sum(new_samples_data[i]['rewards'], axis=-1)), axis=-1)

            avg_reward = np.mean(ret)

            latency = np.array([])
            for i in range(5):
                latency = np.concatenate((latency, new_samples_data[i]['finish_time']), axis=-1)

            avg_latency = np.mean(latency)
            avg_latencies.append(avg_latency)


            logger.logkv('Itr', itr)
            logger.logkv('Average reward, ', avg_reward)
            logger.logkv('Average latency,', avg_latency)

            logger.dumpkvs()
            avg_ret.append(avg_reward)

            if itr % self.save_interval == 0:
                self.policy.core_policy.save_variables(save_path="./meta_model_inner_step1/meta_model_"+str(itr)+".ckpt")

        self.policy.core_policy.save_variables(save_path="./meta_model_inner_step1/meta_model_final.ckpt")

        # Generate automated report
        try:
            print("\n==================== GENERATING AUTOMATED REPORT ====================")
            additional_metrics = {
                'policy_losses': policy_losses_all,
                'value_losses': value_losses_all,
                'greedy_latencies': greedy_latencies_all
            }
            
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

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    logger.configure(dir="./meta_offloading20_log-inner_step1/", format_strs=['stdout', 'log', 'csv'])

    print("********* inner_batch_size = 10  and meta_batch_size = 5 ya Ali *********")

    META_BATCH_SIZE = 5
    
    # Control flags for printing
    PRINT_ACTION_CHOICES = True  # Set to True to print action choices (0=local, 1=MEC, 2=V2V)
    ACTION_PRINT_INTERVAL = 0   # Print action choices every N iterations (0 = every iteration)

    resource_cluster = Resources(mec_process_capable=(10.0 * 1024 * 1024),
                                 mobile_process_capable=(1.0 * 1024 * 1024),
                                 bandwidth_up=7.0, bandwidth_dl=7.0,
                                 v2v_process_capable=(1.0 * 1024 * 1024),  # Same as UE
                                 v2v_bandwidth=5.0)  # Lower than MEC

    env = OffloadingEnvironment(resource_cluster=resource_cluster,
                                batch_size=100,
                                graph_number=100,
                                graph_file_paths=[
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_1/random.20.",
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_2/random.20.",
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_3/random.20.",
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_4/random.20.",
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_5/random.20.",
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_6/random.20.",
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_7/random.20.",
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_8/random.20.",
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_9/random.20.",
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_13/random.20.",
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_14/random.20.",
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_15/random.20.",
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_16/random.20.",
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_17/random.20.",
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_18/random.20.",
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_19/random.20.",
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_20/random.20.",
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_21/random.20.",
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_22/random.20.",
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_23/random.20.",
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_24/random.20.",
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_25/random.20.",
                                ],
                                time_major=False)

    action, greedy_finish_time = env.greedy_solution()
    print("avg greedy solution: ", np.mean(greedy_finish_time))
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

    meta_policy = MetaSeq2SeqPolicy(meta_batch_size=META_BATCH_SIZE, obs_dim=20, encoder_units=128, decoder_units=128,
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
                         inner_lr=5e-4,  # Increased from 5e-4 to improve learning
                         outer_lr=5e-4,  # Increased from 5e-4 to improve learning
                         meta_batch_size=META_BATCH_SIZE,
                         num_inner_grad_steps=1,
                         clip_value = 0.2)

    trainer = Trainer(algo = algo,
                        env=env,
                        sampler=sampler,
                        sample_processor=sample_processor,
                        policy=meta_policy,
                        n_itr=3500,
                        greedy_finish_time= greedy_finish_time,
                        start_itr=0,
                        inner_batch_size=10,
                        print_action_choices=PRINT_ACTION_CHOICES,
                        action_print_interval=ACTION_PRINT_INTERVAL)

    with tf.compat.v1.Session() as sess:
        sess.run(tf.global_variables_initializer())
        avg_ret, avg_loss, avg_latencies = trainer.train()


