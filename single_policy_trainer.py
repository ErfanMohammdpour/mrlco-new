import tensorflow as tf
import numpy as np
import time
from utils import logger
from automated_reporting import create_training_report

class SinglePolicyTrainer(object):
    def __init__(self, algo, env, sampler, sample_processor, policy, n_itr, 
                 greedy_finish_time, start_itr=0, batch_size=500, save_interval=100):
        self.algo = algo
        self.env = env
        self.sampler = sampler
        self.sampler_processor = sample_processor
        self.policy = policy
        self.n_itr = n_itr
        self.start_itr = start_itr
        self.batch_size = batch_size
        self.greedy_finish_time = greedy_finish_time
        self.save_interval = save_interval

    def train(self):
        """
        Implement the single-policy RL training process for task offloading problem
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
            logger.log("Sampling trajectories for this iteration...")

            # Sample trajectories from the environment
            paths = self.sampler.obtain_samples(log=False, log_prefix='')

            # Calculate greedy baseline for comparison (on current task)
            if hasattr(self.env, 'greedy_solution'):
                # Get current task info
                current_task = self.env.get_current_task_id() if hasattr(self.env, 'get_current_task_id') else 0
                total_tasks = self.env.get_total_tasks() if hasattr(self.env, 'get_total_tasks') else 1
                
                # Calculate greedy solution for current task
                _, greedy_times = self.env.greedy_solution()
                avg_greedy_time = np.mean(greedy_times) if greedy_times else 0.0
                logger.logkv('Average greedy latency,', avg_greedy_time)
                logger.logkv('Current task,', current_task)
                logger.logkv('Total tasks,', total_tasks)
                greedy_latencies_all.append(avg_greedy_time)

            """ ----------------- Processing Samples ---------------------"""
            logger.log("Processing samples...")
            samples_data = self.sampler_processor.process_samples(paths, log=False, log_prefix='')

            """ ------------------- Policy Update --------------------"""
            policy_losses, value_losses = self.algo.UpdatePPOTarget(samples_data, batch_size=self.batch_size)

            print("average policy loss: ", np.mean(policy_losses))
            avg_loss.append(np.mean(policy_losses))
            policy_losses_all.append(np.mean(policy_losses))

            print("average value loss: ", np.mean(value_losses))
            value_losses_all.append(np.mean(value_losses))

            """ ------------------- Logging Stuff --------------------------"""
            ret = np.sum(samples_data['rewards'], axis=-1)
            avg_reward = np.mean(ret)
            logger.logkv('Itr', itr)
            logger.logkv('Average reward, ', avg_reward)

            latency = samples_data['finish_time']
            avg_latency = np.mean(latency)
            avg_latencies.append(avg_latency)
            logger.logkv('Average latency,', avg_latency)

            logger.dumpkvs()
            avg_ret.append(avg_reward)

            if itr % self.save_interval == 0:
                self.policy.save_variables(save_path="./single_policy_model/single_policy_"+str(itr)+".ckpt")

        self.policy.save_variables(save_path="./single_policy_model/single_policy_final.ckpt")

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
    from policies.meta_seq2seq_policy import Seq2SeqPolicy
    from samplers.seq2seq_sampler import Seq2SeqSampler
    from samplers.seq2seq_sampler_process import Seq2SeqSamplerProcessor
    from baselines.vf_baseline import ValueFunctionBaseline
    from meta_algos.ppo_offloading import PPO

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    logger.configure(dir="./single_policy_offloading_log/", format_strs=['stdout', 'log', 'csv'])

    BATCH_SIZE = 100  # Single batch size instead of meta-batch

    resource_cluster = Resources(mec_process_capable=(10.0 * 1024 * 1024),
                                 mobile_process_capable=(1.0 * 1024 * 1024),
                                 bandwidth_up=7.0, bandwidth_dl=7.0)

    env = OffloadingEnvironment(resource_cluster=resource_cluster,
                                batch_size=BATCH_SIZE,
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

    baseline = ValueFunctionBaseline()

    # Use single policy instead of meta-policy
    policy = Seq2SeqPolicy(obs_dim=17, encoder_units=128, decoder_units=128, vocab_size=2)

    sampler = Seq2SeqSampler(
        env=env,
        policy=policy,
        rollouts_per_task=10,  # More rollouts for better efficiency
        max_path_length=1000,  # Reduced for faster training
        parallel=False,
    )

    sample_processor = Seq2SeqSamplerProcessor(baseline=baseline,
                                               discount=0.99,
                                               gae_lambda=0.95,
                                               normalize_adv=True,
                                               positive_adv=False)
    
    algo = PPO(policy=policy,
               meta_sampler=sampler,
               meta_sampler_process=sample_processor,
               lr=5e-4,
               num_inner_grad_steps=4,
               clip_value=0.2)

    trainer = SinglePolicyTrainer(algo=algo,
                                  env=env,
                                  sampler=sampler,
                                  sample_processor=sample_processor,
                                  policy=policy,
                                  n_itr=3500,
                                  greedy_finish_time=greedy_finish_time,
                                  start_itr=0,
                                  batch_size=1000)

    with tf.compat.v1.Session() as sess:
        sess.run(tf.global_variables_initializer())
        avg_ret, avg_loss, avg_latencies = trainer.train()
