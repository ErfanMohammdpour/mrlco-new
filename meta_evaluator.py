import tensorflow as tf
import numpy as np
import time
from utils import logger

class Trainer():
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
        avg_greedy_latencies = []  # Track greedy solution latencies
        avg_greedy_energies = []   # Track greedy solution energies (if enabled)
        
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

            """ ------------------- Compute Greedy Solution --------------------"""
            # Compute greedy solution for comparison
            self.env.set_task(0)  # Ensure we're evaluating the correct task
            greedy_result = self.env.greedy_solution()
            
            avg_greedy_energy = None
            if self.env.resource_cluster.use_energy:
                greedy_action, greedy_finish_time, greedy_energy = greedy_result
                # Flatten finish times and energy for averaging
                flat_greedy_finish_times = [item for sublist in greedy_finish_time for item in sublist]
                flat_greedy_energy = [item for sublist in greedy_energy for item in sublist]
                avg_greedy_latency = np.mean(flat_greedy_finish_times)
                avg_greedy_energy = np.mean(flat_greedy_energy)
                avg_greedy_latencies.append(avg_greedy_latency)
                avg_greedy_energies.append(avg_greedy_energy)
            else:
                greedy_action, greedy_finish_time = greedy_result
                # Flatten finish times for averaging
                flat_greedy_finish_times = [item for sublist in greedy_finish_time for item in sublist]
                avg_greedy_latency = np.mean(flat_greedy_finish_times)
                avg_greedy_latencies.append(avg_greedy_latency)

            """ ------------------- Logging Stuff --------------------------"""

            ret = np.sum(samples_data['rewards'], axis=-1)
            avg_reward = np.mean(ret)

            latency = samples_data['finish_time']
            avg_latency = np.mean(latency)

            avg_latencies.append(avg_latency)


            logger.logkv('Itr', itr)
            logger.logkv('Average reward, ', avg_reward)
            logger.logkv('Average latency,', avg_latency)
            logger.logkv('Greedy latency,', avg_greedy_latency)
            
            # Log energy if enabled
            if self.env.resource_cluster.use_energy and 'energy' in samples_data:
                avg_energy = np.mean(np.sum(samples_data['energy'], axis=-1))
                print(f"Average energy per iteration {itr}: {avg_energy:.4f}")
                print(f"Greedy energy per iteration {itr}: {avg_greedy_energy:.4f}")
                logger.logkv('Average energy,', avg_energy)
                logger.logkv('Greedy energy,', avg_greedy_energy)
            
            print(f"Policy latency: {avg_latency:.4f}, Greedy latency: {avg_greedy_latency:.4f}")
            
            logger.dumpkvs()
            avg_ret.append(avg_reward)

        return avg_ret, avg_pg_loss,avg_vf_loss, avg_latencies

if __name__ == "__main__":
    from env.mec_offloaing_envs.offloading_env import Resources
    from env.mec_offloaing_envs.offloading_env import OffloadingEnvironment
    from policies.meta_seq2seq_policy import  Seq2SeqPolicy
    from samplers.seq2seq_sampler import Seq2SeqSampler
    from samplers.seq2seq_sampler_process import Seq2SeSamplerProcessor
    from baselines.vf_baseline import ValueFunctionBaseline
    from meta_algos.ppo_offloading import PPO
    from utils import utils, logger

    logger.configure(dir="./meta_evaluate_ppo_log/task_offloading", format_strs=['stdout', 'log', 'csv'])

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
        'normalize_energy': True,   # Whether to normalize energy rewards
    }
    # ==========================================

    resource_cluster = Resources(mec_process_capable=(10.0 * 1024 * 1024),
                                 mobile_process_capable=(1.0 * 1024 * 1024),
                                 bandwidth_up=7.0, bandwidth_dl=7.0,
                                 use_energy=USE_ENERGY,
                                 energy_config=ENERGY_CONFIG)

    env = OffloadingEnvironment(resource_cluster=resource_cluster,
                                batch_size=100,
                                graph_number=100,
                                graph_file_paths=[
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_12/random.20."
                                    ],
                                time_major=False)

    print("calculate baseline solution======")

    env.set_task(0)
    # Get greedy solution (with energy if enabled)
    greedy_result = env.greedy_solution()
    if env.resource_cluster.use_energy:
        action, finish_time, greedy_energy = greedy_result
        # Flatten finish times and energy for averaging
        flat_finish_times = [item for sublist in finish_time for item in sublist]
        flat_energy = [item for sublist in greedy_energy for item in sublist]
        print("avg greedy solution latency: ", np.mean(flat_finish_times))
        print("avg greedy solution energy: ", np.mean(flat_energy))
    else:
        action, finish_time = greedy_result
        # Flatten finish times for averaging
        flat_finish_times = [item for sublist in finish_time for item in sublist]
        print("avg greedy solution: ", np.mean(flat_finish_times))
    
    # Get reward batch (with energy if enabled)
    reward_result = env.get_reward_batch_step_by_step(action[env.task_id],
                                          env.task_graphs_batchs[env.task_id],
                                          env.max_running_time_batchs[env.task_id],
                                          env.min_running_time_batchs[env.task_id])
    if env.resource_cluster.use_energy:
        target_batch, task_finish_time_batch, energy_batch = reward_result
    else:
        target_batch, task_finish_time_batch = reward_result
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
    algo = PPO(policy=policy,
               meta_sampler=sampler,
               meta_sampler_process=sample_processor,
               lr=1e-4,
               num_inner_grad_steps=3,
               clip_value=0.2,
               max_grad_norm=None)

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

    with tf.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        policy.load_variables(load_path="./meta_model_inner_step1/meta_model_2700.ckpt")
        avg_ret, avg_pg_loss, avg_vf_loss, avg_latencies = trainer.train()


