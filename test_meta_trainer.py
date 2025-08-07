import tensorflow as tf
import numpy as np
import time
from utils import logger
from automated_reporting import create_training_report

# Enable GPU if available
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Found {len(gpus)} GPU(s)")
    except RuntimeError as e:
        print(e)

# Print TF version
print(f"TensorFlow version: {tf.__version__}")

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
            # Limit to available samples (may be less than 5 for small batch)
            num_samples = min(5, len(new_samples_data))
            for i in range(num_samples):
                ret = np.concatenate((ret, np.sum(new_samples_data[i]['rewards'], axis=-1)), axis=-1)

            avg_reward = np.mean(ret) if len(ret) > 0 else 0.0

            latency = np.array([])
            for i in range(num_samples):
                latency = np.concatenate((latency, new_samples_data[i]['finish_time']), axis=-1)

            avg_latency = np.mean(latency) if len(latency) > 0 else 0.0
            avg_latencies.append(avg_latency)


            logger.logkv('Itr', itr)
            logger.logkv('Average reward, ', avg_reward)
            logger.logkv('Average latency,', avg_latency)

            logger.dumpkvs()
            avg_ret.append(avg_reward)

            if itr % self.save_interval == 0:
                self.policy.core_policy.save_variables(save_path="./meta_model_inner_step1/meta_model_"+str(itr)+".ckpt")

        self.policy.core_policy.save_variables(save_path="./meta_model_inner_step1/meta_model_final.ckpt")

        return avg_ret, avg_loss, avg_latencies


if __name__ == "__main__":
    from env.mec_offloading_envs.offloading_env import Resources
    from env.mec_offloading_envs.offloading_env import OffloadingEnvironment
    from policies.meta_seq2seq_policy import MetaSeq2SeqPolicy
    from samplers.seq2seq_meta_sampler import Seq2SeqMetaSampler
    from samplers.seq2seq_meta_sampler_process import Seq2SeqMetaSamplerProcessor
    from baselines.vf_baseline import ValueFunctionBaseline
    from meta_algos.MRLCO import MRLCO

    # Disable TensorFlow logging
    tf.get_logger().setLevel('ERROR')
    logger.configure(dir="./test_log/", format_strs=['stdout', 'log', 'csv'])

    # REDUCED FOR TESTING
    META_BATCH_SIZE = 2  # Reduced from 10 for testing
    
    print("Initializing environment...")
    resource_cluster = Resources(mec_process_capable=(10.0 * 1024 * 1024),
                                 mobile_process_capable=(1.0 * 1024 * 1024),
                                 bandwidth_up=7.0, bandwidth_dl=7.0)

    env = OffloadingEnvironment(resource_cluster=resource_cluster,
                                batch_size=10,  # Reduced from 100 for testing
                                graph_number=10,  # Reduced from 100 for testing
                                graph_file_paths=[
                                    "./env/mec_offloading_envs/data/meta_offloading_20/offload_random20_1/random.20.",
                                    "./env/mec_offloading_envs/data/meta_offloading_20/offload_random20_2/random.20.",
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

    print("Creating baseline...")
    baseline = ValueFunctionBaseline()

    print("Creating meta policy...")
    meta_policy = MetaSeq2SeqPolicy(meta_batch_size=META_BATCH_SIZE, obs_dim=17, encoder_units=128, decoder_units=128,
                                    vocab_size=2)

    print("Creating sampler...")
    sampler = Seq2SeqMetaSampler(
        env=env,
        policy=meta_policy,
        rollouts_per_meta_task=1,
        meta_batch_size=META_BATCH_SIZE,
        max_path_length=20000,
        parallel=False,
    )

    print("Creating sample processor...")
    sample_processor = Seq2SeqMetaSamplerProcessor(baseline=baseline,
                                                   discount=0.99,
                                                   gae_lambda=0.95,
                                                   normalize_adv=True,
                                                   positive_adv=False)
    
    print("Creating MRLCO algorithm...")
    algo = MRLCO(policy=meta_policy,
                         meta_sampler=sampler,
                         meta_sampler_process=sample_processor,
                         inner_lr=1e-3,
                         outer_lr=1e-3,
                         meta_batch_size=META_BATCH_SIZE,
                         num_inner_grad_steps=1,
                         clip_value = 0.3)

    print("Creating trainer...")
    trainer = Trainer(algo = algo,
                        env=env,
                        sampler=sampler,
                        sample_processor=sample_processor,
                        policy=meta_policy,
                        n_itr=3,  # Only 3 iterations for testing!
                        greedy_finish_time= greedy_finish_time,
                        start_itr=0,
                        inner_batch_size=50)  # Reduced from 1000 for testing

    print("\nStarting training loop...")
    print("=" * 80)
    
    # TF2: No session needed - just run the training
    avg_ret, avg_loss, avg_latencies = trainer.train()
    
    print("\n" + "=" * 80)
    print("Training completed successfully!")
    print(f"Final average return: {avg_ret[-1] if avg_ret else 'N/A'}")
    print(f"Final average loss: {avg_loss[-1] if avg_loss else 'N/A'}")
    print(f"Final average latency: {avg_latencies[-1] if avg_latencies else 'N/A'}")