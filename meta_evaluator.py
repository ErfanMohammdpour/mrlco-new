import tensorflow as tf
import numpy as np
import time
from utils import logger
from utils.gpu import setup_gpu_and_strategy, log_tensor_device, ensure_tensor_conversion

class Trainer():
    _first_batch = True  # Class variable for tracking first batch
    def __init__(self,algo,
                env,
                sampler,
                sample_processor,
                policy,
                n_itr,
                batch_size=500,
                start_itr=0,
                num_inner_grad_steps=3):
        self.algo = algo
        self.env = env
        self.sampler = sampler
        self.sampler_processor = sample_processor
        self.policy = policy
        self.n_itr = n_itr
        self.start_itr = start_itr
        self.num_inner_grad_steps = num_inner_grad_steps
        self.batch_size = batch_size

    def train(self):
        """
        Implement the repilte algorithm for ppo reinforcement learning
        """
        start_time = time.time()
        avg_ret = []
        avg_pg_loss = []
        avg_vf_loss = []

        avg_latencies = []
        for itr in range(self.start_itr, self.n_itr):
            itr_start_time = time.time()
            logger.log("\n ---------------- Iteration %d ----------------" % itr)
            logger.log("Sampling set of tasks/goals for this meta-batch...")

            paths = self.sampler.obtain_samples(log=True, log_prefix='')

            """ ----------------- Processing Samples ---------------------"""
            logger.log("Processing samples...")
            samples_data = self.sampler_processor.process_samples(paths, log='all', log_prefix='')

            """ ------------------- Inner Policy Update --------------------"""
            policy_losses, value_losses = self.algo.UpdatePPOTarget(samples_data, batch_size=self.batch_size)

            #print("task losses: ", losses)
            print("average policy losses: ", np.mean(policy_losses))
            avg_pg_loss.append(np.mean(policy_losses))

            print("average value losses: ", np.mean(value_losses))
            avg_vf_loss.append(np.mean(value_losses))

            """ ------------------- Logging Stuff --------------------------"""

            ret = np.sum(samples_data['rewards'], axis=-1)
            avg_reward = np.mean(ret)

            latency = samples_data['finish_time']
            avg_latency = np.mean(latency)

            avg_latencies.append(avg_latency)


            logger.logkv('Itr', itr)
            logger.logkv('Average reward, ', avg_reward)
            logger.logkv('Average latency,', avg_latency)
            logger.dumpkvs()
            avg_ret.append(avg_reward)

        return avg_ret, avg_pg_loss,avg_vf_loss, avg_latencies

if __name__ == "__main__":
    # Note: PPO algorithm uses TF2 eager execution, so we don't disable it here
    
    from env.mec_offloaing_envs.offloading_env import Resources
    from env.mec_offloaing_envs.offloading_env import OffloadingEnvironment
    from policies.meta_seq2seq_policy import  Seq2SeqPolicy
    from samplers.seq2seq_sampler import Seq2SeqSampler
    from samplers.seq2seq_sampler_process import Seq2SeSamplerProcessor
    from baselines.vf_baseline import ValueFunctionBaseline
    from meta_algos.ppo_offloading import PPO
    from utils import utils, logger
    
    # Set up GPU strategy before creating any models
    print("\n========== Setting up GPU/CPU strategy ==========\n")
    strategy, device_info = setup_gpu_and_strategy()

    logger.configure(dir="./meta_evaluate_ppo_log/task_offloading", format_strs=['stdout', 'log', 'csv'])

    resource_cluster = Resources(mec_process_capable=(10.0 * 1024 * 1024),
                                 mobile_process_capable=(1.0 * 1024 * 1024),
                                 bandwidth_up=7.0, bandwidth_dl=7.0)

    env = OffloadingEnvironment(resource_cluster=resource_cluster,
                                batch_size=100,
                                graph_number=100,
                                graph_file_paths=[
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_12/random.20."
                                    ],
                                time_major=False)

    print("calculate baseline solution======")

    env.set_task(0)
    action, finish_time = env.greedy_solution()
    target_batch, task_finish_time_batch = env.get_reward_batch_step_by_step(action[env.task_id],
                                          env.task_graphs_batchs[env.task_id],
                                          env.max_running_time_batchs[env.task_id],
                                          env.min_running_time_batchs[env.task_id])
    discounted_reward = []
    for reward_path in target_batch:
        discounted_reward.append(utils.discount_cumsum(reward_path, 1.0)[0])

    print("avg greedy solution: ", np.mean(discounted_reward))
    print("avg greedy solution: ", np.mean(task_finish_time_batch))
    print("avg greedy solution: ", np.mean(finish_time))

    print()
    finish_time = env.get_all_mec_execute_time()
    print("avg all remote solution: ", np.mean(finish_time))
    print()
    finish_time = env.get_all_locally_execute_time()
    print("avg all local solution: ", np.mean(finish_time))

    # Build all models within the strategy scope
    with strategy.scope():
        policy = Seq2SeqPolicy(obs_dim=17,
                               encoder_units=128,
                               decoder_units=128,
                               vocab_size=2,
                               name="core_policy")

        sampler = Seq2SeqSampler(env,
                                 policy,
                                 rollouts_per_meta_task=1,
                                 max_path_length=40000,
                                 envs_per_task=None,
                                 parallel=False)

        baseline = ValueFunctionBaseline()

        sample_processor = Seq2SeSamplerProcessor(baseline=baseline,
                                                  discount=0.99,
                                                  gae_lambda=0.95,
                                                  normalize_adv=True,
                                                  positive_adv=False)
        algo = PPODistributed(policy=policy,
                   meta_sampler=sampler,
                   meta_sampler_process=sample_processor,
                   lr=1e-4,
                   num_inner_grad_steps=3,
                   clip_value=0.2,
                   max_grad_norm=None,
                   strategy=strategy)

        # define the trainer of ppo to evaluate the performance of the trained meta policy for new tasks.
        trainer = Trainer(algo=algo,
                          env=env,
                          sampler=sampler,
                          sample_processor=sample_processor,
                          policy=policy,
                          n_itr=21,
                          start_itr=0,
                          batch_size=500,
                          num_inner_grad_steps=3)

    # Since PPO uses eager execution, we don't need sessions
    # Just verify GPU placement with a simple operation
    print("\n========== Verifying GPU placement ==========\n")
    # Create a test computation to verify GPU is being used
    if strategy.num_replicas_in_sync > 1:
        # Test distributed computation
        @tf.function
        def distributed_matmul():
            a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
            b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
            return tf.matmul(a, b)
        
        result = strategy.run(distributed_matmul)
        print(f"Test distributed computation executed successfully across {strategy.num_replicas_in_sync} devices")
    else:
        with tf.device(device_info['selected_devices'][0]):
            test_computation = tf.constant([[1.0, 2.0], [3.0, 4.0]])
            test_result = tf.matmul(test_computation, test_computation)
            log_tensor_device(test_result, "Test computation result")
            print(f"Test computation executed successfully on selected device")
    
    # Load checkpoint 
    from io.checkpointing import load_checkpoint
    load_checkpoint(policy, "./meta_model_inner_step1/meta_model_final.ckpt")
    
    # Run training/evaluation
    avg_ret, avg_pg_loss, avg_vf_loss, avg_latencies = trainer.train()


