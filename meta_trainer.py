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
                save_interval = 100):
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

    def train(self):
        """
        Implement the FOMAML training process for task offloading problem
        """

        start_time = time.time()
        avg_ret = []
        avg_loss = []
        avg_latencies = []
        
        # Additional metrics for comprehensive reporting
        policy_losses_all = []
        value_losses_all = []
        greedy_latencies_all = []
        meta_losses_all = []
        
        for itr in range(self.start_itr, self.n_itr):
            itr_start_time = time.time()
            logger.log("\n ---------------- Iteration %d ----------------" % itr)
            logger.log("Sampling set of tasks/goals for this meta-batch...")

            task_ids = self.sampler.update_tasks()
            paths = self.sampler.obtain_samples(log=False, log_prefix='')

            greedy_run_time = [self.greedy_finish_time[x] for x in task_ids]
            logger.logkv('Average greedy latency,', np.mean(greedy_run_time))
            greedy_latencies_all.append(np.mean(greedy_run_time))

            """ ----------------- Split into Support/Query Sets ---------------------"""
            logger.log("Splitting data into support and query sets...")
            support_paths, query_paths = self.sampler.split_support_query(paths, support_ratio=0.7)

            """ ----------------- Processing Support Samples ---------------------"""
            logger.log("Processing support samples...")
            support_samples_data = self.sampler_processor.process_samples(support_paths, log=False, log_prefix='')

            """ ----------------- Processing Query Samples ---------------------"""
            logger.log("Processing query samples...")
            query_samples_data = self.sampler_processor.process_samples(query_paths, log=False, log_prefix='')

            """ ------------------- Inner Loop: Task Adaptation --------------------"""
            logger.log("Performing task adaptation (inner loop)...")
            adapted_policies = []
            task_policy_losses = []
            task_value_losses = []
            
            # Debug: Print information about support samples
            print(f"Support samples data length: {len(support_samples_data)}")
            for i, data in enumerate(support_samples_data):
                if data and 'observations' in data:
                    print(f"  Task {i}: {data['observations'].shape[0]} samples")
                else:
                    print(f"  Task {i}: No valid data")
            
            for task_id in range(self.algo.meta_batch_size):
                if task_id < len(support_samples_data) and support_samples_data[task_id] is not None:
                    policy_losses, value_losses = self.algo.adapt_task(
                        support_samples_data[task_id], task_id, batch_size=self.inner_batch_size)
                    task_policy_losses.append(policy_losses)
                    task_value_losses.append(value_losses)
                    adapted_policies.append(f"adapted_policy_{task_id}")  # Placeholder for adapted policy
                else:
                    # Handle case where we have fewer tasks than meta_batch_size
                    print(f"Warning: No support data for task {task_id}")
                    task_policy_losses.append([0.0])
                    task_value_losses.append([0.0])
                    adapted_policies.append(f"adapted_policy_{task_id}")

            # Log inner loop losses
            avg_policy_loss = np.mean([np.mean(losses) for losses in task_policy_losses])
            avg_value_loss = np.mean([np.mean(losses) for losses in task_value_losses])
            
            print("average inner loop policy losses: ", avg_policy_loss)
            print("average inner loop value losses: ", avg_value_loss)
            
            avg_loss.append(avg_policy_loss)
            policy_losses_all.append(avg_policy_loss)
            value_losses_all.append(avg_value_loss)

            """ ------------------- Outer Loop: Meta-Update --------------------"""
            logger.log("Performing meta-update (outer loop)...")
            
            # Debug: Print information about query samples
            print(f"Query samples data length: {len(query_samples_data)}")
            for i, data in enumerate(query_samples_data):
                if data and 'observations' in data:
                    print(f"  Task {i}: {data['observations'].shape[0]} samples")
                else:
                    print(f"  Task {i}: No valid data")
            
            # Evaluate adapted policies on query sets
            query_losses = []
            for task_id in range(self.algo.meta_batch_size):
                if task_id < len(query_samples_data) and query_samples_data[task_id] is not None:
                    query_loss = self.algo.evaluate_adapted_policy(query_samples_data[task_id], task_id)
                    query_losses.append(query_loss)
                else:
                    print(f"Warning: No query data for task {task_id}")
                    query_losses.append(0.0)
            
            # Perform meta-update
            self.algo.meta_update(adapted_policies, query_losses)
            
            # Log meta-loss
            avg_meta_loss = np.mean(query_losses)
            meta_losses_all.append(avg_meta_loss)
            print("average meta-loss: ", avg_meta_loss)

            """ ------------------- Logging Stuff --------------------------"""
            # Compute average rewards from query sets
            ret = np.array([])
            for i in range(min(5, len(query_samples_data))):
                ret = np.concatenate((ret, np.sum(query_samples_data[i]['rewards'], axis=-1)), axis=-1)

            if len(ret) > 0:
                avg_reward = np.mean(ret)
            else:
                avg_reward = 0.0

            # Compute average latencies from query sets
            latency = np.array([])
            for i in range(min(5, len(query_samples_data))):
                latency = np.concatenate((latency, query_samples_data[i]['finish_time']), axis=-1)

            if len(latency) > 0:
                avg_latency = np.mean(latency)
            else:
                avg_latency = 0.0
                
            avg_latencies.append(avg_latency)

            logger.logkv('Itr', itr)
            logger.logkv('Average reward, ', avg_reward)
            logger.logkv('Average latency,', avg_latency)
            logger.logkv('Average meta-loss,', avg_meta_loss)

            logger.dumpkvs()
            avg_ret.append(avg_reward)

            if itr % self.save_interval == 0:
                self.policy.core_policy.save_variables(save_path="./meta_model_fomaml/meta_model_"+str(itr)+".ckpt")

        self.policy.core_policy.save_variables(save_path="./meta_model_fomaml/meta_model_final.ckpt")

        # Generate automated report
        try:
            print("\n==================== GENERATING AUTOMATED REPORT ====================")
            additional_metrics = {
                'policy_losses': policy_losses_all,
                'value_losses': value_losses_all,
                'greedy_latencies': greedy_latencies_all,
                'meta_losses': meta_losses_all
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
    # Clear any existing graph
    tf.reset_default_graph()
    
    from env.mec_offloaing_envs.offloading_env import Resources
    from env.mec_offloaing_envs.offloading_env import OffloadingEnvironment
    from policies.meta_seq2seq_policy import MetaSeq2SeqPolicy
    from samplers.seq2seq_meta_sampler import Seq2SeqMetaSampler
    from samplers.seq2seq_meta_sampler_process import Seq2SeqMetaSamplerProcessor
    from baselines.vf_baseline import ValueFunctionBaseline
    from meta_algos.FOMAML import FOMAML

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    logger.configure(dir="./meta_offloading20_log_fomaml/", format_strs=['stdout', 'log', 'csv'])

    META_BATCH_SIZE = 10

    resource_cluster = Resources(mec_process_capable=(10.0 * 1024 * 1024),
                                 mobile_process_capable=(1.0 * 1024 * 1024),
                                 bandwidth_up=7.0, bandwidth_dl=7.0)

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

    baseline = ValueFunctionBaseline()

    meta_policy = MetaSeq2SeqPolicy(meta_batch_size=META_BATCH_SIZE, obs_dim=17, encoder_units=128, decoder_units=128,
                                    vocab_size=2)

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
    algo = FOMAML(policy=meta_policy,
                         meta_sampler=sampler,
                         meta_sampler_process=sample_processor,
                         inner_lr=5e-4,  # Inner loop learning rate
                         outer_lr=5e-4,  # Outer loop learning rate
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
                        inner_batch_size=1000)

    with tf.compat.v1.Session() as sess:
        sess.run(tf.global_variables_initializer())
        avg_ret, avg_loss, avg_latencies = trainer.train()


