import tensorflow as tf
import numpy as np
import time
from utils import logger
from automated_reporting import create_training_report

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
        avg_energies = []          # Track policy energies (if enabled)
        
        # Energy configuration (same as in your project)
        ENERGY_CONFIG = {
            'rho': 1.0,
            'f_l': 1.0,
            'zeta': 2.0,
            'ptx': 0.1,
            'prx': 0.05,
            'latency_weight': 0.5,
            'energy_weight': 0.5,
            'normalize_energy': True,
        }
        
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
                # Calculate greedy energy using method (for consistency)
                avg_greedy_energy = self._calculate_greedy_energy(greedy_action, ENERGY_CONFIG)
                avg_greedy_energies.append(avg_greedy_energy)

            """ ------------------- Logging Stuff --------------------------"""

            ret = np.sum(samples_data['rewards'], axis=-1)
            avg_reward = np.mean(ret)

            latency = samples_data['finish_time']
            avg_latency = np.mean(latency)
            avg_latencies.append(avg_latency)

            # Calculate policy energy consumption
            if self.env.resource_cluster.use_energy:
                avg_energy = self._calculate_policy_energy(samples_data, ENERGY_CONFIG)
                avg_energies.append(avg_energy)
            else:
                avg_energies.append(None)

            logger.logkv('Itr', itr)
            logger.logkv('Average reward, ', avg_reward)
            logger.logkv('Average latency,', avg_latency)
            logger.logkv('Greedy latency,', avg_greedy_latency)
            
            # Log energy if enabled
            if self.env.resource_cluster.use_energy:
                logger.logkv('Average energy,', avg_energy)
                logger.logkv('Greedy energy,', avg_greedy_energy)
                
                # Print energy report after each epoch
                print(f"\n========== EPOCH {itr} ENERGY REPORT ==========")
                print(f"Policy Average Energy: {avg_energy:.6f} Joules")
                print(f"Greedy Average Energy: {avg_greedy_energy:.6f} Joules")
                print(f"Energy Ratio (Policy/Greedy): {avg_energy/avg_greedy_energy:.4f}" if avg_greedy_energy > 0 else "Energy Ratio: N/A")
                print(f"Policy Average Latency: {avg_latency:.6f}")
                print(f"Greedy Average Latency: {avg_greedy_latency:.6f}")
                print(f"===============================================\n")
            
            logger.dumpkvs()
            avg_ret.append(avg_reward)

        # Generate comprehensive report
        try:
            print("\n==================== GENERATING AUTOMATED REPORT ====================")
            additional_metrics = {
                'policy_losses': avg_pg_loss,
                'value_losses': avg_vf_loss,
                'greedy_latencies': avg_greedy_latencies,
                'average_energy': avg_energies,
                'greedy_energy': avg_greedy_energies
            }
            
            report_dir = create_training_report(
                avg_ret=avg_ret,
                avg_loss=avg_pg_loss,
                avg_latencies=avg_latencies,
                additional_metrics=additional_metrics
            )
            print(f"Report generated successfully at: {report_dir}")
            print("=====================================================================\n")
        except Exception as e:
            print(f"WARNING: Failed to generate automated report: {str(e)}")
            import traceback
            traceback.print_exc()
            print("Training completed successfully but report generation failed.")

        return avg_ret, avg_pg_loss, avg_vf_loss, avg_latencies
    
    def _calculate_policy_energy(self, samples_data, energy_config):
        """
        Calculate energy consumption for policy actions.
        Uses the same energy model as mrlco-new project:
        - Local execution: T_l * rho * (f_l ^ zeta)
        - Offloading (MEC/V2V): T_ul * ptx + T_dl * prx
        """
        # Get actions and finish times
        actions = samples_data['actions']  # Shape: [batch_size, seq_len]
        finish_times = samples_data['finish_time']  # Shape: [batch_size]
        
        total_energy = 0.0
        env = self.env
        
        # Calculate energy for each trajectory
        for i in range(len(finish_times)):
            action_seq = actions[i]
            finish_time = finish_times[i]
            
            # Get the task graph for this trajectory
            task_graph = env.task_graphs_batchs[env.task_id][i % len(env.task_graphs_batchs[env.task_id])]
            
            # Build plan from actions
            plan = []
            for idx, action in enumerate(action_seq):
                if idx < len(task_graph.prioritize_sequence):
                    task_id = task_graph.prioritize_sequence[idx]
                    plan.append((task_id, int(action)))
            
            # Calculate energy using environment's scheduling cost method
            # We'll simulate the scheduling to get execution times
            energy_sum = 0.0
            
            for task_id, action in plan:
                if task_id < len(task_graph.task_list):
                    task = task_graph.task_list[task_id]
                    
                    if action == 0:  # Local execution
                        # Calculate local execution time
                        T_l = env.resource_cluster.locally_execution_cost(task.processing_data_size)
                        # Energy: T_l * rho * (f_l ^ zeta)
                        energy = T_l * energy_config['rho'] * (energy_config['f_l'] ** energy_config['zeta'])
                    elif action == 1:  # MEC offloading
                        # Calculate transmission times
                        T_ul = env.resource_cluster.up_transmission_cost(task.processing_data_size)
                        T_dl = env.resource_cluster.dl_transmission_cost(task.transmission_data_size)
                        # Energy: T_ul * ptx + T_dl * prx
                        energy = T_ul * energy_config['ptx'] + T_dl * energy_config['prx']
                    elif action == 2:  # V2V offloading
                        # Calculate V2V transmission times
                        T_v2v_ul = env.resource_cluster.v2v_transmission_cost(task.processing_data_size)
                        T_v2v_dl = env.resource_cluster.v2v_transmission_cost(task.transmission_data_size)
                        # V2V execution time on helper vehicle
                        T_v2v_exec = env.resource_cluster.v2v_execution_cost(task.processing_data_size)
                        
                        # V2V transmission energy (uses V2V-specific parameters)
                        ptx_v2v = energy_config.get('ptx_v2v', energy_config['ptx'] * 0.6)
                        prx_v2v = energy_config.get('prx_v2v', energy_config['prx'] * 0.6)
                        transmission_energy = T_v2v_ul * ptx_v2v + T_v2v_dl * prx_v2v
                        
                        # V2V computation energy (less than local)
                        rho_v2v = energy_config.get('rho_v2v', energy_config['rho'] * 0.7)
                        f_v2v = energy_config.get('f_v2v', energy_config['f_l'])
                        computation_energy = T_v2v_exec * rho_v2v * (f_v2v ** energy_config['zeta'])
                        
                        # Total V2V energy
                        energy = transmission_energy + computation_energy
                    else:
                        energy = 0.0
                    
                    energy_sum += energy
            
            total_energy += energy_sum
        
        return total_energy / len(finish_times) if len(finish_times) > 0 else 0.0
    
    def _calculate_greedy_energy(self, greedy_action, energy_config):
        """
        Calculate energy consumption for greedy solution.
        Uses the same energy model as mrlco-new project:
        - Local execution: T_l * rho * (f_l ^ zeta)
        - Offloading (MEC/V2V): T_ul * ptx + T_dl * prx
        """
        if not greedy_action or len(greedy_action) == 0:
            return 0.0
        
        total_energy = 0.0
        env = self.env
        
        # Process each task graph batch
        for batch_idx, task_batch in enumerate(greedy_action):
            if batch_idx < len(env.task_graphs_batchs):
                task_graphs = env.task_graphs_batchs[batch_idx]
                
                for plan_idx, plan in enumerate(task_batch):
                    if plan_idx < len(task_graphs):
                        task_graph = task_graphs[plan_idx]
                        energy_sum = 0.0
                        
                        # plan is a list of (task_id, action) tuples
                        for task_id, action in plan:
                            if task_id < len(task_graph.task_list):
                                task = task_graph.task_list[task_id]
                                
                                if action == 0:  # Local execution
                                    # Calculate local execution time
                                    T_l = env.resource_cluster.locally_execution_cost(task.processing_data_size)
                                    # Energy: T_l * rho * (f_l ^ zeta)
                                    energy = T_l * energy_config['rho'] * (energy_config['f_l'] ** energy_config['zeta'])
                                elif action == 1:  # MEC offloading
                                    # Calculate transmission times
                                    T_ul = env.resource_cluster.up_transmission_cost(task.processing_data_size)
                                    T_dl = env.resource_cluster.dl_transmission_cost(task.transmission_data_size)
                                    # Energy: T_ul * ptx + T_dl * prx
                                    energy = T_ul * energy_config['ptx'] + T_dl * energy_config['prx']
                                elif action == 2:  # V2V offloading
                                    # Calculate V2V transmission times
                                    T_v2v_ul = env.resource_cluster.v2v_transmission_cost(task.processing_data_size)
                                    T_v2v_dl = env.resource_cluster.v2v_transmission_cost(task.transmission_data_size)
                                    # V2V execution time on helper vehicle
                                    T_v2v_exec = env.resource_cluster.v2v_execution_cost(task.processing_data_size)
                                    
                                    # V2V transmission energy (uses V2V-specific parameters)
                                    ptx_v2v = energy_config.get('ptx_v2v', energy_config['ptx'] * 0.6)
                                    prx_v2v = energy_config.get('prx_v2v', energy_config['prx'] * 0.6)
                                    transmission_energy = T_v2v_ul * ptx_v2v + T_v2v_dl * prx_v2v
                                    
                                    # V2V computation energy (less than local)
                                    rho_v2v = energy_config.get('rho_v2v', energy_config['rho'] * 0.7)
                                    f_v2v = energy_config.get('f_v2v', energy_config['f_l'])
                                    computation_energy = T_v2v_exec * rho_v2v * (f_v2v ** energy_config['zeta'])
                                    
                                    # Total V2V energy
                                    energy = transmission_energy + computation_energy
                                else:
                                    energy = 0.0
                                
                                energy_sum += energy
                        
                        total_energy += energy_sum
        
        # Average across all task graphs
        total_plans = sum(len(batch) for batch in greedy_action)
        return total_energy / total_plans if total_plans > 0 else 0.0

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

    # Energy configuration - enable energy optimization
    ENERGY_CONFIG = {
        'rho': 1.0,           # Local computation energy coefficient
        'f_l': 1.0,           # Local CPU frequency (normalized)
        'zeta': 2.0,          # CPU frequency exponent
        'ptx': 0.1,           # MEC transmission power (Watts)
        'prx': 0.05,          # MEC reception power (Watts)
        'ptx_v2v': 0.06,      # V2V transmission power (Watts, typically < ptx)
        'prx_v2v': 0.03,      # V2V reception power (Watts, typically < prx)
        'rho_v2v': 0.7,       # V2V computation energy coefficient (70% of local)
        'f_v2v': 1.0,         # V2V CPU frequency (normalized, same as local)
        'latency_weight': 0.5, # Weight for latency in combined reward
        'energy_weight': 0.5,  # Weight for energy in combined reward
        'normalize_energy': True,  # Whether to normalize energy rewards
    }

    resource_cluster = Resources(mec_process_capable=(10.0 * 1024 * 1024),
                                 mobile_process_capable=(1.0 * 1024 * 1024),
                                 bandwidth_up=7.0, bandwidth_dl=7.0,
                                 v2v_process_capable=(1.0 * 1024 * 1024),  # Same as UE
                                 v2v_bandwidth=5.0,  # Lower than MEC
                                 use_energy=True,  # Enable energy optimization
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
    print()
    finish_time = env.get_all_v2v_execute_time()
    print("avg all V2V solution: ", np.mean(finish_time))

    policy = Seq2SeqPolicy(obs_dim=20,
                           encoder_units=128,
                           decoder_units=128,
                           vocab_size=3,
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
        policy.load_variables(load_path="./meta_model_inner_step1/meta_model_final.ckpt")
        avg_ret, avg_pg_loss, avg_vf_loss, avg_latencies = trainer.train()


