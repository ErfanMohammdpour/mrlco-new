from env.base import MetaEnv
from env.mec_offloaing_envs.offloading_task_graph import OffloadingTaskGraph

from samplers.vectorized_env_executor import MetaIterativeEnvExecutor
import numpy as np
import os

class Resources(object):
    """
    This class denotes the MEC server, Mobile devices, and V2V helper vehicles (computation resources)

    Args:
        mec_process_capable: computation capacity of the MEC server
        mobile_process_capable: computation capacity of the mobile device
        bandwidth_up: wireless uplink band width
        bandwidth_dl: wireless downlink band width
        v2v_process_capable: computation capacity of V2V helper vehicle (default: same as mobile)
        v2v_bandwidth: V2V communication bandwidth (default: 5.0 Mbps)
        use_energy: boolean flag to enable energy optimization (default: False)
        energy_config: dictionary with energy configuration parameters
    """

    def __init__(self, mec_process_capable,
                  mobile_process_capable, bandwidth_up = 7.0, bandwidth_dl = 7.0,
                  v2v_process_capable=None, v2v_bandwidth=5.0,
                  use_energy=False, energy_config=None):
        self.mec_process_capable = mec_process_capable
        self.mobile_process_capable = mobile_process_capable
        self.mobile_process_available_time = 0.0
        self.mec_process_available_time = 0.0

        self.bandwidth_up = bandwidth_up
        self.bandwidth_dl = bandwidth_dl

        # V2V parameters (Phase 1: same capacity as UE, lower bandwidth than MEC)
        if v2v_process_capable is None:
            v2v_process_capable = mobile_process_capable  # Same as UE
        self.v2v_process_capable = v2v_process_capable
        self.v2v_bandwidth = v2v_bandwidth
        self.v2v_process_available_time = 0.0
        self.v2v_channel_available_time = 0.0
        
        # Energy extension parameters
        self.use_energy = use_energy
        if energy_config is None:
            energy_config = self._default_energy_config()
        self.energy_config = energy_config
        
        # Store energy weights if enabled
        if self.use_energy:
            self.latency_weight = energy_config.get('latency_weight', 0.5)
            self.energy_weight = energy_config.get('energy_weight', 0.5)
        else:
            self.latency_weight = 1.0
            self.energy_weight = 0.0

    def up_transmission_cost(self, data):
        rate = self.bandwidth_up * (1024.0 * 1024.0 / 8.0)

        transmission_time = data / rate

        return transmission_time

    def reset(self):
        self.mec_process_available_time = 0.0
        self.mobile_process_available_time = 0.0
        self.v2v_process_available_time = 0.0
        self.v2v_channel_available_time = 0.0

    def dl_transmission_cost(self, data):
        rate = self.bandwidth_dl * (1024.0 * 1024.0 / 8.0)
        transmission_time = data / rate

        return transmission_time

    def locally_execution_cost(self, data):
        return self._computation_cost(data, self.mobile_process_capable)

    def mec_execution_cost(self, data):
        return self._computation_cost(data, self.mec_process_capable)

    def v2v_transmission_cost(self, data):
        """V2V transmission cost (Phase 1: same model as MEC, no distance factor)"""
        rate = self.v2v_bandwidth * (1024.0 * 1024.0 / 8.0)
        transmission_time = data / rate
        return transmission_time

    def v2v_execution_cost(self, data):
        """V2V execution cost"""
        return self._computation_cost(data, self.v2v_process_capable)

    def _computation_cost(self, data, processing_power):
        computation_time = data / processing_power

        return computation_time
    
    def _default_energy_config(self):
        """Default energy configuration parameters"""
        return {
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
    
    def compute_local_energy(self, execution_time):
        """Compute energy consumption for local execution
        
        Args:
            execution_time: Local execution time
            
        Returns:
            Energy consumption (0.0 if use_energy=False)
        """
        if not self.use_energy:
            return 0.0
        return execution_time * self.energy_config['rho'] * \
               (self.energy_config['f_l'] ** self.energy_config['zeta'])
    
    def compute_transmission_energy(self, uplink_time, downlink_time):
        """Compute energy consumption for MEC transmission
        
        Args:
            uplink_time: Uplink transmission time
            downlink_time: Downlink transmission time
            
        Returns:
            Energy consumption (0.0 if use_energy=False)
        """
        if not self.use_energy:
            return 0.0
        return (uplink_time * self.energy_config['ptx'] + 
                downlink_time * self.energy_config['prx'])
    
    def compute_v2v_transmission_energy(self, uplink_time, downlink_time):
        """Compute energy consumption for V2V transmission
        
        Args:
            uplink_time: V2V uplink transmission time
            downlink_time: V2V downlink transmission time
            
        Returns:
            Energy consumption (0.0 if use_energy=False)
        """
        if not self.use_energy:
            return 0.0
        
        # Use V2V-specific transmission parameters (separate from MEC)
        ptx_v2v = self.energy_config.get('ptx_v2v', self.energy_config['ptx'] * 0.6)
        prx_v2v = self.energy_config.get('prx_v2v', self.energy_config['prx'] * 0.6)
        
        return (uplink_time * ptx_v2v + downlink_time * prx_v2v)
    
    def compute_v2v_energy(self, execution_time):
        """Compute energy consumption for V2V execution on helper vehicle
        
        Args:
            execution_time: V2V execution time on helper vehicle
            
        Returns:
            Energy consumption (0.0 if use_energy=False)
        """
        if not self.use_energy:
            return 0.0
        
        # Use reduced coefficient compared to local (e.g., 0.6-0.8 of local rho)
        rho_v2v = self.energy_config.get('rho_v2v', self.energy_config['rho'] * 0.7)
        f_v2v = self.energy_config.get('f_v2v', self.energy_config['f_l'])
        zeta = self.energy_config['zeta']
        
        return execution_time * rho_v2v * (f_v2v ** zeta)

class OffloadingEnvironment(MetaEnv):
    def __init__(self, resource_cluster, batch_size,
                 graph_number,
                 graph_file_paths, time_major):
        self.resource_cluster = resource_cluster
        self.task_graphs_batchs = []
        self.encoder_batchs = []
        self.encoder_lengths = []
        self.decoder_full_lengths = []
        self.max_running_time_batchs = []
        self.min_running_time_batchs = []
        self.graph_file_paths = graph_file_paths

        # load all the task graphs into the evnironment
        for graph_file_path in graph_file_paths:
            encoder_batchs, encoder_lengths, task_graph_batchs, decoder_full_lengths, max_running_time_batchs, min_running_time_batchs = \
                self.generate_point_batch_for_random_graphs(batch_size, graph_number, graph_file_path, time_major)

            self.encoder_batchs += encoder_batchs
            self.encoder_lengths += encoder_lengths
            self.task_graphs_batchs += task_graph_batchs
            self.decoder_full_lengths += decoder_full_lengths
            self.max_running_time_batchs += max_running_time_batchs
            self.min_running_time_batchs += min_running_time_batchs

        self.total_task = len(self.encoder_batchs)
        self.optimal_solution = -1
        self.task_id = -1
        self.time_major = time_major
        self.input_dim = np.array(encoder_batchs[0]).shape[-1]

        # set the file paht of task graphs.
        self.graph_file_paths = graph_file_paths
        self.graph_number = graph_number


        self.local_exe_time = self.get_all_locally_execute_time()
        self.mec_exe_time = self.get_all_mec_execute_time()

    def sample_tasks(self, n_tasks):
        """
        Samples task of the meta-environment

        Args:
            n_tasks (int) : number of different meta-tasks needed

        Returns:
            tasks (list) : an (n_tasks) length list of tasks
        """
        return np.random.choice(np.arange(self.total_task), n_tasks, replace=False)

    def merge_graphs(self):
        encoder_batchs = []
        encoder_lengths = []
        task_graphs_batchs = []
        decoder_full_lengths =[]
        max_running_time_batchs = []
        min_running_time_batchs = []

        for encoder_batch, encoder_length, task_graphs_batch, \
            decoder_full_length, max_running_time_batch, \
            min_running_time_batch in zip(self.encoder_batchs, self.encoder_lengths,
                                          self.task_graphs_batchs, self.decoder_full_lengths,
                                          self.max_running_time_batchs, self.min_running_time_batchs):
            encoder_batchs += encoder_batch.tolist()
            encoder_lengths += encoder_length.tolist()
            task_graphs_batchs += task_graphs_batch
            decoder_full_lengths += decoder_full_length.tolist()
            max_running_time_batchs += max_running_time_batch
            min_running_time_batchs += min_running_time_batch

        self.encoder_batchs = np.array([encoder_batchs])
        self.encoder_lengths = np.array([encoder_lengths])
        self.task_graphs_batchs = [task_graphs_batchs]
        self.decoder_full_lengths = np.array([decoder_full_lengths])
        self.max_running_time_batchs = np.array([max_running_time_batchs])
        self.min_running_time_batchs = np.array([min_running_time_batchs])

    def set_task(self, task):
        """
        Sets the specified task to the current environment

        Args:
            task: task of the meta-learning environment
        """
        self.task_id = task

    def get_task(self):
        """
        Gets the task that the agent is performing in the current environment

        Returns:
            task: task of the meta-learning environment
        """
        return self.graph_file_paths[self.task_id]

    def step(self, action):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.

        Accepts an action and returns a tuple (observation, reward, done, info).

        Args:
            action (object): an action provided by the agent

        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        plan_batch = []
        task_graph_batch = self.task_graphs_batchs[self.task_id]
        max_running_time_batch = self.max_running_time_batchs[self.task_id]
        min_running_time_batch = self.min_running_time_batchs[self.task_id]

        for action_sequence, task_graph in zip(action, task_graph_batch):
            plan_sequence = []

            for action, task_id in zip(action_sequence, task_graph.prioritize_sequence):
                plan_sequence.append((task_id, action))

            plan_batch.append(plan_sequence)

        # Get rewards and optionally energy
        result = self.get_reward_batch_step_by_step(plan_batch,
                                                     task_graph_batch,
                                                     max_running_time_batch,
                                                     min_running_time_batch)
        
        if self.resource_cluster.use_energy:
            reward_batch, task_finish_time, energy_batch = result
            # Include energy in info for logging (same format as original project)
            info = (task_finish_time, energy_batch)
        else:
            reward_batch, task_finish_time = result
            info = task_finish_time

        done = True
        observation = np.array(self.encoder_batchs[self.task_id])

        return observation, reward_batch, done, info

    def reset(self):
        """Resets the state of the environment and returns an initial observation.

        Returns:
            observation (object): the initial observation.
        """
        # reset the resource environment.
        self.resource_cluster.reset()

        return np.array(self.encoder_batchs[self.task_id])

    def render(self, mode='human'):
        pass

    def generate_point_batch_for_random_graphs(self, batch_size, graph_number, graph_file_path, time_major):
        encoder_list = []
        task_graph_list = []

        encoder_batchs = []
        encoder_lengths = []
        task_graph_batchs = []
        decoder_full_lengths = []

        max_running_time_vector = []
        min_running_time_vector = []

        max_running_time_batchs = []
        min_running_time_batchs = []

        for i in range(graph_number):
            task_graph = OffloadingTaskGraph(graph_file_path + str(i) + '.gv')
            task_graph_list.append(task_graph)

            max_time, min_time = self.calculate_max_min_runningcost(task_graph.max_data_size,
                                                                    task_graph.min_data_size)
            max_running_time_vector.append(max_time)
            min_running_time_vector.append(min_time)

            # the scheduling sequence will also store in self.'prioritize_sequence'
            scheduling_sequence = task_graph.prioritize_tasks(self.resource_cluster)

            task_encode = np.array(task_graph.encode_point_sequence_with_ranking_and_cost(scheduling_sequence,
                                                                                          self.resource_cluster), dtype=np.float32)
            encoder_list.append(task_encode)

        for i in range(int(graph_number / batch_size)):
            start_batch_index = i * batch_size
            end_batch_index = (i + 1) * batch_size

            task_encode_batch = encoder_list[start_batch_index:end_batch_index]
            if time_major:
                task_encode_batch = np.array(task_encode_batch).swapaxes(0, 1)
                sequence_length = np.asarray([task_encode_batch.shape[0]] * task_encode_batch.shape[1])
            else:
                task_encode_batch = np.array(task_encode_batch)
                sequence_length = np.asarray([task_encode_batch.shape[1]] * task_encode_batch.shape[0])

            decoder_full_lengths.append(sequence_length)
            encoder_lengths.append(sequence_length)
            encoder_batchs.append(task_encode_batch)

            task_graph_batch = task_graph_list[start_batch_index:end_batch_index]
            task_graph_batchs.append(task_graph_batch)
            max_running_time_batchs.append(max_running_time_vector[start_batch_index:end_batch_index])
            min_running_time_batchs.append(min_running_time_vector[start_batch_index:end_batch_index])

        return encoder_batchs, encoder_lengths, task_graph_batchs, \
               decoder_full_lengths, max_running_time_batchs, \
               min_running_time_batchs

    def calculate_max_min_runningcost(self, max_data_size, min_data_size):
        max_time = max( [self.resource_cluster.up_transmission_cost(max_data_size),
                         self.resource_cluster.dl_transmission_cost(max_data_size),
                         self.resource_cluster.locally_execution_cost(max_data_size),
                         self.resource_cluster.v2v_transmission_cost(max_data_size)] )

        min_time = self.resource_cluster.mec_execution_cost(min_data_size)

        return max_time, min_time

    def get_scheduling_cost_step_by_step(self, plan, task_graph):
        cloud_available_time = 0.0
        ws_available_time =0.0
        local_available_time = 0.0
        # V2V availability (separate from MEC)
        v2v_available_time = 0.0
        v2v_channel_available_time = 0.0

        # running time on local processor
        T_l = [0] * task_graph.task_number
        # running time on sending channel
        T_ul = [0] * task_graph.task_number
        #running time on receiving channel
        T_dl = [0] * task_graph.task_number
        # V2V transmission times
        T_v2v_ul = [0] * task_graph.task_number
        T_v2v_dl = [0] * task_graph.task_number

        # finish time on cloud for each task
        FT_cloud = [0] * task_graph.task_number
        # finish time on sending channel for each task
        FT_ws = [0] * task_graph.task_number
        # finish time locally for each task
        FT_locally = [0] * task_graph.task_number
        # finish time recieving channel for each task
        FT_wr = [0] * task_graph.task_number
        # V2V finish times
        FT_v2v_ul = [0] * task_graph.task_number
        FT_v2v_exec = [0] * task_graph.task_number
        FT_v2v_dl = [0] * task_graph.task_number
        current_FT = 0.0
        return_latency = []
        return_energy = []

        for item in plan:
            i = item[0]
            task = task_graph.task_list[i]
            x = item[1]

            # locally scheduling
            if x == 0:
                if len(task_graph.pre_task_sets[i]) != 0:
                    start_time = max(local_available_time,
                                     max([max(FT_locally[j], FT_wr[j], FT_v2v_dl[j]) for j in task_graph.pre_task_sets[i]]))
                else:
                    start_time = local_available_time

                T_l[i] = self.resource_cluster.locally_execution_cost(task.processing_data_size)
                FT_locally[i] = start_time + T_l[i]
                local_available_time = FT_locally[i]

                task_finish_time = FT_locally[i]

                # calculate the energy consumption
                if self.resource_cluster.use_energy:
                    energy_consumption = self.resource_cluster.compute_local_energy(T_l[i])
                    return_energy.append(energy_consumption)
                else:
                    return_energy.append(0.0)
            # MEC scheduling
            elif x == 1:
                if len(task_graph.pre_task_sets[i]) != 0:
                    ws_start_time = max(ws_available_time,
                                        max([max(FT_locally[j], FT_ws[j])  for j in task_graph.pre_task_sets[i]]))

                    T_ul[i] = self.resource_cluster.up_transmission_cost(task.processing_data_size)
                    ws_finish_time = ws_start_time + T_ul[i]
                    FT_ws[i] = ws_finish_time
                    ws_available_time = ws_finish_time

                    cloud_start_time = max( cloud_available_time,
                                            max([max(FT_ws[i], FT_cloud[j]) for j in task_graph.pre_task_sets[i]]))
                    cloud_finish_time = cloud_start_time + self.resource_cluster.mec_execution_cost(task.processing_data_size)
                    FT_cloud[i] = cloud_finish_time
                    # print("task {}, Cloud finish time {}".format(i, FT_cloud[i]))
                    cloud_available_time = cloud_finish_time

                    wr_start_time = FT_cloud[i]
                    T_dl[i] = self.resource_cluster.dl_transmission_cost(task.transmission_data_size)
                    wr_finish_time = wr_start_time + T_dl[i]
                    FT_wr[i] = wr_finish_time

                    # calculate the energy consumption
                    if self.resource_cluster.use_energy:
                        energy_consumption = self.resource_cluster.compute_transmission_energy(T_ul[i], T_dl[i])
                        return_energy.append(energy_consumption)
                    else:
                        return_energy.append(0.0)

                else:
                    ws_start_time = ws_available_time
                    T_ul[i] = self.resource_cluster.up_transmission_cost(task.processing_data_size)
                    ws_finish_time = ws_start_time + T_ul[i]
                    FT_ws[i] = ws_finish_time

                    cloud_start_time = max(cloud_available_time, FT_ws[i])
                    cloud_finish_time = cloud_start_time + self.resource_cluster.mec_execution_cost(task.processing_data_size)
                    FT_cloud[i] = cloud_finish_time
                    cloud_available_time = cloud_finish_time

                    wr_start_time = FT_cloud[i]
                    T_dl[i] = self.resource_cluster.dl_transmission_cost(task.transmission_data_size)
                    wr_finish_time = wr_start_time + T_dl[i]
                    FT_wr[i] = wr_finish_time

                    # calculate the energy consumption
                    if self.resource_cluster.use_energy:
                        energy_consumption = self.resource_cluster.compute_transmission_energy(T_ul[i], T_dl[i])
                        return_energy.append(energy_consumption)
                    else:
                        return_energy.append(0.0)

                task_finish_time = wr_finish_time
            # V2V scheduling
            elif x == 2:
                # V2V uses a shared half-duplex channel: uplink and downlink cannot occur simultaneously
                # Channel must be free before starting uplink, and free again before starting downlink
                
                # Step 1: Determine when V2V uplink can start
                # Must wait for: (a) channel availability, (b) predecessor task completions
                if len(task_graph.pre_task_sets[i]) != 0:
                    # Predecessors can finish on local (FT_locally), MEC (FT_wr), or V2V (FT_v2v_dl)
                    # Note: For V2V predecessors, we use FT_v2v_dl (downlink completion) because
                    # results must be received back at the source vehicle before next task can start
                    v2v_ul_start_time = max(v2v_channel_available_time,
                                            max([max(FT_locally[j], FT_wr[j], FT_v2v_dl[j]) 
                                                 for j in task_graph.pre_task_sets[i]]))
                else:
                    v2v_ul_start_time = v2v_channel_available_time

                # Step 2: V2V uplink transmission (uses shared channel)
                T_v2v_ul[i] = self.resource_cluster.v2v_transmission_cost(task.processing_data_size)
                v2v_ul_finish_time = v2v_ul_start_time + T_v2v_ul[i]
                FT_v2v_ul[i] = v2v_ul_finish_time
                # Channel becomes available after uplink completes
                v2v_channel_available_time = v2v_ul_finish_time

                # Step 3: V2V execution (on helper vehicle, independent of channel)
                # Must wait for: (a) helper vehicle availability, (b) uplink completion, (c) predecessor dependencies
                if len(task_graph.pre_task_sets[i]) != 0:
                    # Predecessors can finish locally (FT_locally), on MEC (FT_wr), or on V2V (FT_v2v_exec)
                    # Note: For execution, we use FT_v2v_exec (not FT_v2v_dl) because execution
                    # can start as soon as the task finishes processing, before results are transmitted back
                    v2v_exec_start_time = max(v2v_available_time, FT_v2v_ul[i],
                                              max([max(FT_locally[j], FT_wr[j], FT_v2v_exec[j]) 
                                                   for j in task_graph.pre_task_sets[i]]))
                else:
                    v2v_exec_start_time = max(v2v_available_time, FT_v2v_ul[i])
                
                exec_time = self.resource_cluster.v2v_execution_cost(task.processing_data_size)
                v2v_exec_finish_time = v2v_exec_start_time + exec_time
                FT_v2v_exec[i] = v2v_exec_finish_time
                v2v_available_time = v2v_exec_finish_time

                # Step 4: V2V downlink transmission (uses shared channel, must wait for channel availability)
                # Channel may have been used by other tasks' uplinks/downlinks since uplink completed
                # Downlink can start when: (a) execution completes, (b) channel is free
                v2v_dl_start_time = max(v2v_exec_finish_time, v2v_channel_available_time)
                T_v2v_dl[i] = self.resource_cluster.v2v_transmission_cost(task.transmission_data_size)
                v2v_dl_finish_time = v2v_dl_start_time + T_v2v_dl[i]
                FT_v2v_dl[i] = v2v_dl_finish_time
                # Channel becomes available after downlink completes
                v2v_channel_available_time = v2v_dl_finish_time

                task_finish_time = v2v_dl_finish_time

                # calculate the energy consumption for V2V (transmission + computation)
                if self.resource_cluster.use_energy:
                    # Transmission energy (uplink + downlink) - uses V2V-specific parameters
                    transmission_energy = self.resource_cluster.compute_v2v_transmission_energy(T_v2v_ul[i], T_v2v_dl[i])
                    
                    # Computation energy on helper vehicle (less than local)
                    computation_energy = self.resource_cluster.compute_v2v_energy(exec_time)
                    
                    # Total V2V energy
                    energy_consumption = transmission_energy + computation_energy
                    return_energy.append(energy_consumption)
                else:
                    return_energy.append(0.0)
            else:
                # Invalid action value
                raise ValueError(f"Invalid action value: {x}. Expected 0 (local), 1 (MEC), or 2 (V2V)")

            # print("task  {} finish time is {}".format(i , task_finish_time))
            delta_make_span = max(task_finish_time, current_FT) - current_FT
            current_FT = max(task_finish_time, current_FT)
            return_latency.append(delta_make_span)

        # Return based on energy flag for backward compatibility
        if self.resource_cluster.use_energy:
            return return_latency, current_FT, return_energy
        else:
            return return_latency, current_FT

    def score_func(self, cost, max_time, min_time):
        """Score function that handles both scalars and lists/arrays.
        
        For lists/arrays, computes element-wise scoring.
        """
        cost = np.asarray(cost)
        return -(cost - min_time) / (max_time - min_time)
    
    def _compute_energy_bounds(self, task_graph, max_time, min_time):
        """Compute theoretical min/max energy consumption for normalization
        
        Args:
            task_graph: Task graph object
            max_time: Maximum running time (for reference)
            min_time: Minimum running time (for reference)
            
        Returns:
            tuple: (max_energy, min_energy) or (0.0, 0.0) if energy disabled
        """
        if not self.resource_cluster.use_energy:
            return 0.0, 0.0
        
        # Max energy: All tasks executed locally
        max_energy = sum([
            self.resource_cluster.compute_local_energy(
                task.processing_data_size / self.resource_cluster.mobile_process_capable
            ) for task in task_graph.task_list
        ])
        
        # Min energy: All tasks offloaded (consider both MEC and V2V)
        min_energy = 0.0
        for task in task_graph.task_list:
            # MEC transmission energy (MEC has no computation energy on device)
            mec_ul_time = self.resource_cluster.up_transmission_cost(task.processing_data_size)
            mec_dl_time = self.resource_cluster.dl_transmission_cost(task.transmission_data_size)
            mec_energy = self.resource_cluster.compute_transmission_energy(mec_ul_time, mec_dl_time)
            
            # V2V energy (transmission + computation on helper vehicle)
            v2v_ul_time = self.resource_cluster.v2v_transmission_cost(task.processing_data_size)
            v2v_dl_time = self.resource_cluster.v2v_transmission_cost(task.transmission_data_size)
            v2v_exec_time = self.resource_cluster.v2v_execution_cost(task.processing_data_size)
            
            # Use V2V-specific transmission energy method (separate parameters)
            v2v_transmission_energy = self.resource_cluster.compute_v2v_transmission_energy(v2v_ul_time, v2v_dl_time)
            v2v_computation_energy = self.resource_cluster.compute_v2v_energy(v2v_exec_time)
            v2v_energy = v2v_transmission_energy + v2v_computation_energy
            
            # Use minimum of MEC and V2V
            min_energy += min(mec_energy, v2v_energy)
        
        return max_energy, min_energy

    def get_reward_batch_step_by_step(self, action_sequence_batch, task_graph_batch,
                                      max_running_time_batch, min_running_time_batch):
        target_batch = []
        task_finish_time_batch = []
        energy_batch = []  # NEW: Energy batch for logging
        
        for i in range(len(action_sequence_batch)):
            max_running_time = max_running_time_batch[i]
            min_running_time = min_running_time_batch[i]

            task_graph = task_graph_batch[i]
            self.resource_cluster.reset()
            plan = action_sequence_batch[i]
            
            # Get latency and optionally energy
            if self.resource_cluster.use_energy:
                cost, task_finish_time, energy = self.get_scheduling_cost_step_by_step(plan, task_graph)
                
                # Compute energy bounds for normalization
                max_energy, min_energy = self._compute_energy_bounds(
                    task_graph, max_running_time, min_running_time)
                
                # Sum energy to get total energy consumption
                total_energy = np.sum(energy) if isinstance(energy, (list, np.ndarray)) else energy
                
                # Normalize energy (handle edge case where max == min)
                if max_energy > min_energy:
                    total_energy_score = self.score_func(total_energy, max_energy, min_energy)
                    # Distribute energy score proportionally across steps
                    if len(energy) > 0 and total_energy > 0:
                        energy_proportions = np.array(energy) / total_energy
                        energy_score = total_energy_score * energy_proportions
                    elif len(energy) > 0:
                        # If total_energy is 0, distribute score equally
                        energy_score = np.full_like(energy, total_energy_score / len(energy), dtype=float)
                    else:
                        energy_score = np.array([total_energy_score])
                else:
                    # If no variation, set to zero
                    energy_score = np.zeros_like(energy)
                
                # Normalize latency - cost is incremental latencies, normalize element-wise
                latency_score = self.score_func(cost, max_running_time, min_running_time)
                
                # Combine rewards
                combined_score = (self.resource_cluster.latency_weight * latency_score + 
                                self.resource_cluster.energy_weight * energy_score)
                
                target_batch.append(combined_score)
                energy_batch.append(energy)
            else:
                # Original behavior - backward compatible
                cost, task_finish_time = self.get_scheduling_cost_step_by_step(plan, task_graph)
                latency = self.score_func(cost, max_running_time, min_running_time)
                score = np.array(latency)
                target_batch.append(score)
                energy_batch.append([])  # Empty for backward compatibility
            
            task_finish_time_batch.append(task_finish_time)

        target_batch = np.array(target_batch)
        
        # Return based on energy flag for backward compatibility
        if self.resource_cluster.use_energy:
            return target_batch, task_finish_time_batch, energy_batch
        else:
            return target_batch, task_finish_time_batch

    def greedy_solution(self):
        result_plan = []
        finish_time_batchs = []
        energy_batchs = []  # Track energy for greedy solution
        
        for task_graph_batch in self.task_graphs_batchs:
            plan_batchs = []
            finish_time_plan = []
            energy_plan = []  # Energy per task graph
            
            for task_graph in task_graph_batch:
                cloud_available_time = 0.0
                ws_available_time = 0.0
                local_available_time = 0.0
                # V2V availability (separate from MEC)
                v2v_available_time = 0.0
                v2v_channel_available_time = 0.0

                # finish time on cloud for each task
                FT_cloud = [0] * task_graph.task_number
                # finish time on sending channel for each task
                FT_ws = [0] * task_graph.task_number
                # finish time locally for each task
                FT_locally = [0] * task_graph.task_number
                # finish time recieving channel for each task
                FT_wr = [0] * task_graph.task_number
                # V2V finish times
                FT_v2v_ul = [0] * task_graph.task_number
                FT_v2v_exec = [0] * task_graph.task_number
                FT_v2v_dl = [0] * task_graph.task_number
                
                # Energy tracking
                total_energy = 0.0
                T_l = [0] * task_graph.task_number
                T_ul = [0] * task_graph.task_number
                T_dl = [0] * task_graph.task_number
                T_v2v_ul = [0] * task_graph.task_number
                T_v2v_dl = [0] * task_graph.task_number
                
                plan = []

                for i in task_graph.prioritize_sequence:
                    task = task_graph.task_list[i]

                    # calculate the local finish time
                    if len(task_graph.pre_task_sets[i]) != 0:
                        start_time = max(local_available_time,
                                         max([max(FT_locally[j], FT_wr[j], FT_v2v_dl[j]) for j in task_graph.pre_task_sets[i]]))
                    else:
                        start_time = local_available_time

                    local_running_time = self.resource_cluster.locally_execution_cost(task.processing_data_size)
                    FT_locally[i] = start_time + local_running_time
                    T_l[i] = local_running_time

                    # calculate the MEC finish time
                    if len(task_graph.pre_task_sets[i]) != 0:
                        ws_start_time = max(ws_available_time,
                                            max([max(FT_locally[j], FT_ws[j]) for j in task_graph.pre_task_sets[i]]))
                        T_ul[i] = self.resource_cluster.up_transmission_cost(task.processing_data_size)
                        FT_ws[i] = ws_start_time + T_ul[i]
                        cloud_start_time = max(cloud_available_time,
                                               max([max(FT_ws[i], FT_cloud[j]) for j in task_graph.pre_task_sets[i]]))
                        cloud_finish_time = cloud_start_time + self.resource_cluster.mec_execution_cost(
                            task.processing_data_size)
                        FT_cloud[i] = cloud_finish_time
                        # print("task {}, Cloud finish time {}".format(i, FT_cloud[i]))
                        wr_start_time = FT_cloud[i]
                        T_dl[i] = self.resource_cluster.dl_transmission_cost(task.transmission_data_size)
                        wr_finish_time = wr_start_time + T_dl[i]
                        FT_wr[i] = wr_finish_time
                    else:
                        ws_start_time = ws_available_time
                        T_ul[i] = self.resource_cluster.up_transmission_cost(task.processing_data_size)
                        ws_finish_time = ws_start_time + T_ul[i]
                        FT_ws[i] = ws_finish_time

                        cloud_start_time = max(cloud_available_time, FT_ws[i])
                        FT_cloud[i] = cloud_start_time + self.resource_cluster.mec_execution_cost(
                            task.processing_data_size)
                        T_dl[i] = self.resource_cluster.dl_transmission_cost(task.transmission_data_size)
                        FT_wr[i] = FT_cloud[i] + T_dl[i]

                    # Calculate the V2V finish time
                    # V2V uses a shared half-duplex channel: uplink and downlink cannot occur simultaneously
                    # Note: In greedy solution, we calculate finish times assuming current resource availability,
                    # then update resources only if this task is selected
                    
                    # Step 1: V2V uplink start time
                    # Must wait for: (a) channel availability, (b) predecessor task completions
                    if len(task_graph.pre_task_sets[i]) != 0:
                        # Predecessors can finish on local (FT_locally), MEC (FT_wr), or V2V (FT_v2v_dl)
                        v2v_ul_start_time = max(v2v_channel_available_time,
                                                max([max(FT_locally[j], FT_wr[j], FT_v2v_dl[j]) 
                                                     for j in task_graph.pre_task_sets[i]]))
                    else:
                        v2v_ul_start_time = v2v_channel_available_time
                    
                    # Step 2: V2V uplink transmission
                    T_v2v_ul[i] = self.resource_cluster.v2v_transmission_cost(task.processing_data_size)
                    FT_v2v_ul[i] = v2v_ul_start_time + T_v2v_ul[i]
                    # Channel becomes available after uplink completes (for this task's perspective)
                    # Note: In reality, channel may be used by other tasks between uplink and downlink
                    v2v_channel_after_ul = FT_v2v_ul[i]
                    
                    # Step 3: V2V execution start time
                    # Must wait for: (a) helper availability, (b) uplink completion, (c) predecessor dependencies
                    v2v_exec_start_time = max(v2v_available_time, FT_v2v_ul[i])
                    if len(task_graph.pre_task_sets[i]) != 0:
                        # Predecessors can finish locally (FT_locally), on MEC (FT_wr), or on V2V (FT_v2v_exec)
                        v2v_exec_start_time = max(v2v_exec_start_time,
                                                  max([max(FT_locally[j], FT_wr[j], FT_v2v_exec[j]) 
                                                       for j in task_graph.pre_task_sets[i]]))
                    
                    # Step 4: V2V execution
                    exec_time = self.resource_cluster.v2v_execution_cost(task.processing_data_size)
                    FT_v2v_exec[i] = v2v_exec_start_time + exec_time
                    
                    # Step 5: V2V downlink transmission
                    # Must wait for: (a) execution completion, (b) channel availability
                    # Channel availability is max of: (i) channel after this task's uplink, (ii) current global channel state
                    # This accounts for potential channel usage by other tasks processed earlier
                    v2v_dl_start_time = max(FT_v2v_exec[i], 
                                           max(v2v_channel_after_ul, v2v_channel_available_time))
                    T_v2v_dl[i] = self.resource_cluster.v2v_transmission_cost(task.transmission_data_size)
                    FT_v2v_dl[i] = v2v_dl_start_time + T_v2v_dl[i]

                    # Compare all three options and choose the best
                    t_local = FT_locally[i]
                    t_mec = FT_wr[i]
                    t_v2v = FT_v2v_dl[i]
                    
                    if t_local <= t_mec and t_local <= t_v2v:
                        action = 0  # Local execution
                        local_available_time = FT_locally[i]
                        
                        # Compute energy for local execution
                        if self.resource_cluster.use_energy:
                            total_energy += self.resource_cluster.compute_local_energy(T_l[i])
                        
                        FT_wr[i] = 0.0
                        FT_cloud[i] = 0.0
                        FT_ws[i] = 0.0
                        FT_v2v_ul[i] = 0.0
                        FT_v2v_exec[i] = 0.0
                        FT_v2v_dl[i] = 0.0
                        T_ul[i] = 0.0
                        T_dl[i] = 0.0
                        T_v2v_ul[i] = 0.0
                        T_v2v_dl[i] = 0.0
                    elif t_mec <= t_v2v:
                        action = 1  # MEC offloading
                        FT_locally[i] = 0.0
                        cloud_available_time = FT_cloud[i]
                        ws_available_time = FT_ws[i]
                        
                        # Compute energy for MEC offloading
                        if self.resource_cluster.use_energy:
                            total_energy += self.resource_cluster.compute_transmission_energy(T_ul[i], T_dl[i])
                        
                        FT_v2v_ul[i] = 0.0
                        FT_v2v_exec[i] = 0.0
                        FT_v2v_dl[i] = 0.0
                        T_l[i] = 0.0
                        T_v2v_ul[i] = 0.0
                        T_v2v_dl[i] = 0.0
                    else:
                        action = 2  # V2V offloading
                        FT_locally[i] = 0.0
                        FT_wr[i] = 0.0
                        FT_cloud[i] = 0.0
                        FT_ws[i] = 0.0
                        
                        # Compute energy for V2V offloading
                        if self.resource_cluster.use_energy:
                            # V2V transmission energy (uses V2V-specific parameters)
                            transmission_energy = self.resource_cluster.compute_v2v_transmission_energy(T_v2v_ul[i], T_v2v_dl[i])
                            # V2V computation energy (on helper vehicle, less than local)
                            computation_energy = self.resource_cluster.compute_v2v_energy(exec_time)
                            # Total V2V energy
                            total_energy += transmission_energy + computation_energy
                        
                        # Update V2V resource availability
                        v2v_available_time = FT_v2v_exec[i]  # Helper vehicle becomes available after execution
                        # Channel becomes available after downlink completes
                        # Note: We use max() here because other tasks may have used the channel
                        # between when we calculated v2v_dl_start_time and now
                        v2v_channel_available_time = max(v2v_channel_available_time, FT_v2v_dl[i])
                        T_l[i] = 0.0
                        T_ul[i] = 0.0
                        T_dl[i] = 0.0
                    plan.append((i, action))

                finish_time = max( max(FT_wr), max(FT_locally), max(FT_v2v_dl) )
                plan_batchs.append(plan)
                finish_time_plan.append(finish_time)
                
                # Store energy for this task graph
                if self.resource_cluster.use_energy:
                    energy_plan.append(total_energy)
                else:
                    energy_plan.append(0.0)

            finish_time_batchs.append(finish_time_plan)
            result_plan.append(plan_batchs)
            energy_batchs.append(energy_plan)

        # Return based on energy flag for backward compatibility
        if self.resource_cluster.use_energy:
            return result_plan, finish_time_batchs, energy_batchs
        else:
            return result_plan, finish_time_batchs

    def calculate_optimal_solution(self):
        # Finding the optimal solution via exhausting search the solution space.
        # Updated for 3 actions: 0 (local), 1 (MEC), 2 (V2V)
        def exhaustion_plans(n):
            plan_batch = []

            for i in range(3**n):  # 3 actions: local, MEC, V2V
                plan = []
                num = i
                # Convert to base-3 representation
                for _ in range(n):
                    plan.append(num % 3)
                    num //= 3
                plan.reverse()  # Reverse to get correct order
                plan_batch.append(plan)
            return plan_batch

        n = self.task_graphs_batchs[0][0].task_number
        plan_batch = exhaustion_plans(n)

        print("exhausted plan size: ", len(plan_batch))

        task_graph_optimal_costs = []
        optimal_plan = []

        for task_graph_batch in self.task_graphs_batchs:
            task_graph_batch_cost = []
            for task_graph in task_graph_batch:
                plans_costs = []
                prioritize_plan = []

                for plan in plan_batch:
                    plan_sequence = []
                    for action, task_id in zip(plan, task_graph.prioritize_sequence):
                        plan_sequence.append((task_id, action))

                    cos, task_finish_time = self.get_scheduling_cost_step_by_step(plan_sequence, task_graph)
                    plans_costs.append(task_finish_time)

                    prioritize_plan.append(plan_sequence)

                graph_min_cost = min(plans_costs)

                optimal_plan.append(prioritize_plan[np.argmin(plans_costs)])

                task_graph_batch_cost.append(graph_min_cost)

            print("task_graph_batch cost shape is {}".format(np.array(task_graph_batch_cost).shape))
            avg_minimal_cost = np.mean(task_graph_batch_cost)

            task_graph_optimal_costs.append(avg_minimal_cost)

        self.optimal_solution = task_graph_optimal_costs
        return task_graph_optimal_costs, optimal_plan

    def get_running_cost(self, action_sequence_batch, task_graph_batch):
        cost_batch = []
        energy_batch = []
        for action_sequence, task_graph in zip(action_sequence_batch,
                                               task_graph_batch):
            plan_sequence = []

            for action, task_id in zip(action_sequence,
                                       task_graph.prioritize_sequence):
                plan_sequence.append((task_id, action))

                _, task_finish_time = self.get_scheduling_cost_step_by_step(plan_sequence, task_graph)

            cost_batch.append(task_finish_time)

        return cost_batch

    def get_all_locally_execute_time(self):
        running_cost = []
        for task_graph_batch, encode_batch in zip(self.task_graphs_batchs, self.encoder_batchs):
            batch_size = encode_batch.shape[0]
            sequence_length = encode_batch.shape[1]

            scheduling_action = np.zeros(shape=(batch_size, sequence_length), dtype=np.int32)
            running_cost_batch = self.get_running_cost(scheduling_action, task_graph_batch)
            running_cost.append(np.mean(running_cost_batch))

        return running_cost

    def get_all_mec_execute_time(self):
        running_cost = []

        for task_graph_batch, encode_batch in zip(self.task_graphs_batchs, self.encoder_batchs):
            batch_size = encode_batch.shape[0]
            sequence_length = encode_batch.shape[1]

            scheduling_action = np.ones(shape=(batch_size, sequence_length), dtype=np.int32)
            running_cost_batch = self.get_running_cost(scheduling_action, task_graph_batch)

            running_cost.append(np.mean(running_cost_batch))

        return running_cost

    def get_all_v2v_execute_time(self):
        """Calculate finish time when all tasks are offloaded to V2V helper"""
        running_cost = []

        for task_graph_batch, encode_batch in zip(self.task_graphs_batchs, self.encoder_batchs):
            batch_size = encode_batch.shape[0]
            sequence_length = encode_batch.shape[1]

            scheduling_action = np.full(shape=(batch_size, sequence_length), fill_value=2, dtype=np.int32)
            running_cost_batch = self.get_running_cost(scheduling_action, task_graph_batch)

            running_cost.append(np.mean(running_cost_batch))

        return running_cost

    def greedy_solution_for_current_task(self):
        result_plan, finish_time_batchs = self.greedy_solution()

        return result_plan[self.task_id], finish_time_batchs[self.task_id]



