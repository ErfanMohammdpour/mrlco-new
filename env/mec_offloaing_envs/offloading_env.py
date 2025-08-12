from env.base import MetaEnv
from env.mec_offloaing_envs.offloading_task_graph import OffloadingTaskGraph

from samplers.vectorized_env_executor import MetaIterativeEnvExecutor
import numpy as np
import os

class Resources(object):
    """
    This class denotes the MEC server and Mobile devices (computation resources)

    Args:
        mec_process_capable: computation capacity of the MEC server
        mobile_process_capable: computation capacity of the mobile device
        bandwidth_up: wireless uplink band width
        bandwidth_dl: wireless downlink band width
    """

    def __init__(self, mec_process_capable,
                  mobile_process_capable, bandwidth_up = 7.0, bandwidth_dl = 7.0):
        self.mec_process_capble = mec_process_capable
        self.mobile_process_capable = mobile_process_capable
        self.mobile_process_avaliable_time = 0.0
        self.mec_process_avaliable_time = 0.0

        self.bandwidth_up = bandwidth_up
        self.bandwidth_dl = bandwidth_dl

    def up_transmission_cost(self, data):
        rate = self.bandwidth_up * (1024.0 * 1024.0 / 8.0)

        transmission_time = data / rate

        return transmission_time

    def reset(self):
        self.mec_process_avaliable_time = 0.0
        self.mobile_process_avaliable_time = 0.0

    def dl_transmission_cost(self, data):
        rate = self.bandwidth_dl * (1024.0 * 1024.0 / 8.0)
        transmission_time = data / rate

        return transmission_time

    def locally_execution_cost(self, data):
        return self._computation_cost(data, self.mobile_process_capable)

    def mec_execution_cost(self, data):
        return self._computation_cost(data, self.mec_process_capble)

    def _computation_cost(self, data, processing_power):
        computation_time = data / processing_power

        return computation_time

class OffloadingEnvironment(MetaEnv):
    def __init__(self, resource_cluster, batch_size,
                 graph_number,
                 graph_file_paths, time_major,
                 use_difference_reward=True,
                 reward_clip_range=(-2.0, 2.0),
                 epsilon=1e-9,
                 cp_shaping_enabled=True,
                 cp_discount=0.99,
                 cp_coefficient=1.0,
                 cp_normalize_mode="none",
                 cp_scale=None,
                 small_threshold=1e-9):
        self.resource_cluster = resource_cluster
        self.task_graphs_batchs = []
        self.encoder_batchs = []
        self.encoder_lengths = []
        self.decoder_full_lengths = []
        self.max_running_time_batchs = []
        self.min_running_time_batchs = []
        self.graph_file_paths = graph_file_paths
        
        # Difference reward configuration
        self.use_difference_reward = use_difference_reward
        self.reward_clip_range = reward_clip_range
        self.epsilon = epsilon
        
        # Critical-path shaping configuration
        self.cp_shaping_enabled = cp_shaping_enabled
        self.cp_discount = cp_discount
        self.cp_coefficient = cp_coefficient
        self.cp_normalize_mode = cp_normalize_mode
        self.cp_scale = cp_scale
        self.small_threshold = small_threshold
        
        # Diagnostic tracking
        self.diagnostic_data = {
            'delta_loc': [],
            'delta_rem': [],
            'delta_chosen': [],
            'delta_best': [],
            'delta_worst': [],
            'r_main': [],
            'r_norm': [],
            'action_is_best': [],
            'cp_before': [],
            'cp_after': [],
            'cp_reduction': [],
            'shaping_contribution': [],
            'shaping_ratio': [],
            'final_step_score': []
        }

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

        reward_batch, task_finish_time= self.get_reward_batch_step_by_step(plan_batch,
                                                  task_graph_batch,
                                                  max_running_time_batch,
                                                  min_running_time_batch)

        done = True
        observation = np.array(self.encoder_batchs[self.task_id])
        info = task_finish_time

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
                         self.resource_cluster.locally_execution_cost(max_data_size)] )

        min_time = self.resource_cluster.mec_execution_cost(min_data_size)

        return max_time, min_time

    def dry_run_preview(self, task_id, action, task_graph, current_state):
        """
        Non-mutating preview of scheduling decision.
        Returns delta_makespan for the given action without modifying any state.
        
        Args:
            task_id: ID of the task to schedule
            action: 0 for local, 1 for remote
            task_graph: The task graph object
            current_state: Dict containing current scheduling state
                - 'makespan_now': current makespan
                - 'local_avail': local CPU availability time
                - 'mec_avail': MEC CPU availability time  
                - 'ws_avail': uplink channel availability time
                - 'FT_locally': finish times for local execution
                - 'FT_wr': finish times for remote execution (after download)
        
        Returns:
            delta_makespan: Incremental makespan for this action
        """
        task = task_graph.task_list[task_id]
        
        # Get predecessor finish times
        pred_finish_times = []
        for pred_id in task_graph.pre_task_sets[task_id]:
            pred_ft = max(current_state['FT_locally'][pred_id], 
                         current_state['FT_wr'][pred_id])
            pred_finish_times.append(pred_ft)
        
        max_pred_ft = max(pred_finish_times) if pred_finish_times else 0.0
        
        if action == 0:  # Local execution
            # Calculate local execution timing
            start_time = max(current_state['local_avail'], max_pred_ft)
            exec_time = self.resource_cluster.locally_execution_cost(task.processing_data_size)
            finish_time = start_time + exec_time
            
        else:  # Remote execution (action == 1)
            # Uplink phase
            ul_start = max(current_state['ws_avail'], max_pred_ft)
            ul_time = self.resource_cluster.up_transmission_cost(task.processing_data_size)
            ul_finish = ul_start + ul_time
            
            # MEC processing phase
            mec_start = max(current_state['mec_avail'], ul_finish)
            mec_time = self.resource_cluster.mec_execution_cost(task.processing_data_size)
            mec_finish = mec_start + mec_time
            
            # Downlink phase
            dl_time = self.resource_cluster.dl_transmission_cost(task.transmission_data_size)
            finish_time = mec_finish + dl_time
        
        # Calculate incremental makespan
        candidate_makespan = max(current_state['makespan_now'], finish_time)
        delta_makespan = candidate_makespan - current_state['makespan_now']
        
        return delta_makespan
    
    def compute_remaining_critical_path(self, scheduled_tasks, task_graph):
        """
        Compute the critical-path time for remaining (unscheduled) tasks.
        This is a non-mutating computation.
        
        Args:
            scheduled_tasks: Set of task IDs already scheduled
            task_graph: The task graph object
            
        Returns:
            cp_time: Maximum critical path time over remaining tasks
        """
        # Identify remaining tasks
        remaining_tasks = set(range(task_graph.task_number)) - scheduled_tasks
        
        if not remaining_tasks:
            return 0.0
        
        # Compute node durations for remaining tasks
        node_durations = {}
        for task_id in remaining_tasks:
            task = task_graph.task_list[task_id]
            
            # Local execution time
            local_time = self.resource_cluster.locally_execution_cost(task.processing_data_size)
            
            # Remote execution time (uplink + edge + downlink)
            uplink_time = self.resource_cluster.up_transmission_cost(task.processing_data_size)
            edge_time = self.resource_cluster.mec_execution_cost(task.processing_data_size)
            downlink_time = self.resource_cluster.dl_transmission_cost(task.transmission_data_size)
            remote_time = uplink_time + edge_time + downlink_time
            
            # Best single-node duration
            node_durations[task_id] = min(local_time, remote_time)
        
        # Build topological order for remaining tasks
        # We need to process in topological order to compute accumulated paths
        in_degree = {task_id: 0 for task_id in remaining_tasks}
        for task_id in remaining_tasks:
            for succ_id in task_graph.suc_task_sets[task_id]:
                if succ_id in remaining_tasks:
                    in_degree[succ_id] += 1
        
        # Initialize accumulated path values
        accumulated = {task_id: 0.0 for task_id in remaining_tasks}
        
        # Process in topological order
        queue = [task_id for task_id in remaining_tasks if in_degree[task_id] == 0]
        processed = set()
        
        while queue:
            task_id = queue.pop(0)
            processed.add(task_id)
            
            # Compute accumulated value for this task
            max_pred_accumulated = 0.0
            for pred_id in task_graph.pre_task_sets[task_id]:
                if pred_id in remaining_tasks:
                    max_pred_accumulated = max(max_pred_accumulated, accumulated[pred_id])
            
            accumulated[task_id] = node_durations[task_id] + max_pred_accumulated
            
            # Update successors
            for succ_id in task_graph.suc_task_sets[task_id]:
                if succ_id in remaining_tasks:
                    in_degree[succ_id] -= 1
                    if in_degree[succ_id] == 0:
                        queue.append(succ_id)
        
        # Return maximum accumulated value (critical path)
        return max(accumulated.values()) if accumulated else 0.0
    
    def get_scheduling_cost_step_by_step(self, plan, task_graph):
        cloud_avaliable_time = 0.0
        ws_avaliable_time =0.0
        local_avaliable_time = 0.0

        # running time on local processor
        T_l = [0] * task_graph.task_number
        # running time on sending channel
        T_ul = [0] * task_graph.task_number
        #running time on receiving channel
        T_dl = [0] * task_graph.task_number


        # finish time on cloud for each task
        FT_cloud = [0] * task_graph.task_number
        # finish time on sending channel for each task
        FT_ws = [0] * task_graph.task_number
        # finish time locally for each task
        FT_locally = [0] * task_graph.task_number
        # finish time recieving channel for each task
        FT_wr = [0] * task_graph.task_number
        current_FT = 0.0
        total_energy = 0.0
        return_latency = []
        return_energy = []

        for item in plan:
            i = item[0]
            task = task_graph.task_list[i]
            x = item[1]

            # locally scheduling
            if x == 0:
                if len(task_graph.pre_task_sets[i]) != 0:
                    start_time = max(local_avaliable_time,
                                     max([max(FT_locally[j], FT_wr[j]) for j in task_graph.pre_task_sets[i]]))
                else:
                    start_time = local_avaliable_time

                T_l[i] = self.resource_cluster.locally_execution_cost(task.processing_data_size)
                FT_locally[i] = start_time + T_l[i]
                local_avaliable_time = FT_locally[i]

                task_finish_time = FT_locally[i]

                # calculate the energy consumption
                #energy_consumption = T_l[i] * self.rho * (self.f_l ** self.zeta)
            # mcc scheduling
            else:
                if len(task_graph.pre_task_sets[i]) != 0:
                    ws_start_time = max(ws_avaliable_time,
                                        max([max(FT_locally[j], FT_ws[j])  for j in task_graph.pre_task_sets[i]]))

                    T_ul[i] = self.resource_cluster.up_transmission_cost(task.processing_data_size)
                    ws_finish_time = ws_start_time + T_ul[i]
                    FT_ws[i] = ws_finish_time
                    ws_avaliable_time = ws_finish_time

                    cloud_start_time = max( cloud_avaliable_time,
                                            max([max(FT_ws[i], FT_cloud[j]) for j in task_graph.pre_task_sets[i]]))
                    cloud_finish_time = cloud_start_time + self.resource_cluster.mec_execution_cost(task.processing_data_size)
                    FT_cloud[i] = cloud_finish_time
                    # print("task {}, Cloud finish time {}".format(i, FT_cloud[i]))
                    cloud_avaliable_time = cloud_finish_time

                    wr_start_time = FT_cloud[i]
                    T_dl[i] = self.resource_cluster.dl_transmission_cost(task.transmission_data_size)
                    wr_finish_time = wr_start_time + T_dl[i]
                    FT_wr[i] = wr_finish_time

                    # calculate the energy consumption
                    #energy_consumption = T_ul[i] * self.ptx + T_dl[i] * self.prx

                else:
                    ws_start_time = ws_avaliable_time
                    T_ul[i] = self.resource_cluster.up_transmission_cost(task.processing_data_size)
                    ws_finish_time = ws_start_time + T_ul[i]
                    FT_ws[i] = ws_finish_time

                    cloud_start_time = max(cloud_avaliable_time, FT_ws[i])
                    cloud_finish_time = cloud_start_time + self.resource_cluster.mec_execution_cost(task.processing_data_size)
                    FT_cloud[i] = cloud_finish_time
                    cloud_avaliable_time = cloud_finish_time

                    wr_start_time = FT_cloud[i]
                    T_dl[i] = self.resource_cluster.dl_transmission_cost(task.transmission_data_size)
                    wr_finish_time = wr_start_time + T_dl[i]
                    FT_wr[i] = wr_finish_time

                    # calculate the energy consumption
                    #energy_consumption = T_ul[i] * self.ptx + T_dl[i] * self.prx

                task_finish_time = wr_finish_time

            # print("task  {} finish time is {}".format(i , task_finish_time))
            delta_make_span = max(task_finish_time, current_FT) - current_FT
            current_FT = max(task_finish_time, current_FT)
            return_latency.append(delta_make_span)

        return return_latency, current_FT
    
    def get_scheduling_cost_with_difference_reward(self, plan, task_graph):
        """
        Compute scheduling cost with difference reward scheme.
        Returns both step-by-step rewards and final makespan.
        """
        cloud_avaliable_time = 0.0
        ws_avaliable_time = 0.0
        local_avaliable_time = 0.0
        
        # Finish time arrays
        FT_cloud = [0] * task_graph.task_number
        FT_ws = [0] * task_graph.task_number
        FT_locally = [0] * task_graph.task_number
        FT_wr = [0] * task_graph.task_number
        
        current_makespan = 0.0
        return_rewards = []
        scheduled_tasks = set()
        
        # Clear diagnostics for new episode
        for key in self.diagnostic_data:
            self.diagnostic_data[key] = []
        
        for step_idx, item in enumerate(plan):
            task_id = item[0]
            action = item[1]
            
            # Create current state for dry-run preview
            current_state = {
                'makespan_now': current_makespan,
                'local_avail': local_avaliable_time,
                'mec_avail': cloud_avaliable_time,
                'ws_avail': ws_avaliable_time,
                'FT_locally': FT_locally.copy(),
                'FT_wr': FT_wr.copy()
            }
            
            # Preview both options
            delta_loc = self.dry_run_preview(task_id, 0, task_graph, current_state)
            delta_rem = self.dry_run_preview(task_id, 1, task_graph, current_state)
            
            # Determine which action was chosen
            delta_chosen = delta_loc if action == 0 else delta_rem
            delta_best = min(delta_loc, delta_rem)
            delta_worst = max(delta_loc, delta_rem)
            
            # Compute difference reward
            r_main = delta_best - delta_chosen
            
            # Normalize per step
            if abs(delta_worst - delta_best) > self.epsilon:
                r_norm = r_main / (delta_worst - delta_best)
            else:
                r_norm = 0.0
            
            # Clip base reward to [-1, 1]
            r_norm = np.clip(r_norm, -1.0, 1.0)
            
            # Compute critical-path shaping if enabled
            shaping_contribution = 0.0
            cp_before = 0.0
            cp_after = 0.0
            
            if self.cp_shaping_enabled:
                # Compute CP before action
                cp_before = self.compute_remaining_critical_path(scheduled_tasks, task_graph)
                
                # Compute CP after action (with task_id added to scheduled)
                scheduled_after = scheduled_tasks | {task_id}
                cp_after = self.compute_remaining_critical_path(scheduled_after, task_graph)
                
                # Apply normalization if needed
                if self.cp_normalize_mode == "divide_by_p75" and self.cp_scale is not None and self.cp_scale > 0:
                    cp_before_normalized = cp_before / self.cp_scale
                    cp_after_normalized = cp_after / self.cp_scale
                else:
                    cp_before_normalized = cp_before
                    cp_after_normalized = cp_after
                
                # Compute shaping contribution
                shaping_contribution = self.cp_coefficient * (cp_before_normalized - self.cp_discount * cp_after_normalized)
                
                # Assert invariants
                assert cp_after <= cp_before + self.small_threshold, \
                    f"CP increased: cp_before={cp_before}, cp_after={cp_after} at step {step_idx}"
            
            # Combine base reward with shaping
            final_step_score = r_norm + shaping_contribution
            
            # Clip final score to reward_clip_range
            final_step_score = np.clip(final_step_score, self.reward_clip_range[0], self.reward_clip_range[1])
            
            # Store diagnostics
            self.diagnostic_data['delta_loc'].append(delta_loc)
            self.diagnostic_data['delta_rem'].append(delta_rem)
            self.diagnostic_data['delta_chosen'].append(delta_chosen)
            self.diagnostic_data['delta_best'].append(delta_best)
            self.diagnostic_data['delta_worst'].append(delta_worst)
            self.diagnostic_data['r_main'].append(r_main)
            self.diagnostic_data['r_norm'].append(r_norm)
            self.diagnostic_data['cp_before'].append(cp_before)
            self.diagnostic_data['cp_after'].append(cp_after)
            self.diagnostic_data['cp_reduction'].append(cp_before - cp_after)
            self.diagnostic_data['shaping_contribution'].append(shaping_contribution)
            shaping_ratio = abs(shaping_contribution) / (abs(r_norm) + self.small_threshold)
            self.diagnostic_data['shaping_ratio'].append(shaping_ratio)
            self.diagnostic_data['final_step_score'].append(final_step_score)
            
            # Check if action matches best option
            action_is_best = False
            if abs(delta_loc - delta_rem) < self.epsilon:
                action_is_best = True  # Both equally good
            elif action == 0 and delta_loc < delta_rem:
                action_is_best = True
            elif action == 1 and delta_rem < delta_loc:
                action_is_best = True
            self.diagnostic_data['action_is_best'].append(action_is_best)
            
            # Now execute the actual action and update state
            task = task_graph.task_list[task_id]
            
            if action == 0:  # Local execution
                if len(task_graph.pre_task_sets[task_id]) != 0:
                    start_time = max(local_avaliable_time,
                                   max([max(FT_locally[j], FT_wr[j]) 
                                        for j in task_graph.pre_task_sets[task_id]]))
                else:
                    start_time = local_avaliable_time
                
                T_l = self.resource_cluster.locally_execution_cost(task.processing_data_size)
                FT_locally[task_id] = start_time + T_l
                local_avaliable_time = FT_locally[task_id]
                task_finish_time = FT_locally[task_id]
                
            else:  # Remote execution
                if len(task_graph.pre_task_sets[task_id]) != 0:
                    ws_start_time = max(ws_avaliable_time,
                                      max([max(FT_locally[j], FT_ws[j]) 
                                           for j in task_graph.pre_task_sets[task_id]]))
                    
                    T_ul = self.resource_cluster.up_transmission_cost(task.processing_data_size)
                    ws_finish_time = ws_start_time + T_ul
                    FT_ws[task_id] = ws_finish_time
                    ws_avaliable_time = ws_finish_time
                    
                    cloud_start_time = max(cloud_avaliable_time,
                                         max([max(FT_ws[task_id], FT_cloud[j]) 
                                              for j in task_graph.pre_task_sets[task_id]]))
                    cloud_finish_time = cloud_start_time + self.resource_cluster.mec_execution_cost(task.processing_data_size)
                    FT_cloud[task_id] = cloud_finish_time
                    cloud_avaliable_time = cloud_finish_time
                    
                    wr_start_time = FT_cloud[task_id]
                    T_dl = self.resource_cluster.dl_transmission_cost(task.transmission_data_size)
                    wr_finish_time = wr_start_time + T_dl
                    FT_wr[task_id] = wr_finish_time
                    
                else:
                    ws_start_time = ws_avaliable_time
                    T_ul = self.resource_cluster.up_transmission_cost(task.processing_data_size)
                    ws_finish_time = ws_start_time + T_ul
                    FT_ws[task_id] = ws_finish_time
                    
                    cloud_start_time = max(cloud_avaliable_time, FT_ws[task_id])
                    cloud_finish_time = cloud_start_time + self.resource_cluster.mec_execution_cost(task.processing_data_size)
                    FT_cloud[task_id] = cloud_finish_time
                    cloud_avaliable_time = cloud_finish_time
                    
                    wr_start_time = FT_cloud[task_id]
                    T_dl = self.resource_cluster.dl_transmission_cost(task.transmission_data_size)
                    wr_finish_time = wr_start_time + T_dl
                    FT_wr[task_id] = wr_finish_time
                
                task_finish_time = wr_finish_time
            
            # Update makespan
            current_makespan = max(task_finish_time, current_makespan)
            
            # Mark task as scheduled
            scheduled_tasks.add(task_id)
            
            # Return the final step score (includes shaping if enabled)
            return_rewards.append(final_step_score)
        
        # Final assertion: at episode end, remaining CP should be 0
        if self.cp_shaping_enabled:
            final_cp = self.compute_remaining_critical_path(scheduled_tasks, task_graph)
            assert final_cp <= self.small_threshold, \
                f"Non-zero critical path at episode end: {final_cp}"
        
        return return_rewards, current_makespan

    def score_func(self, cost, max_time, min_time):
        return -(cost - min_time) / (max_time - min_time)

    def get_reward_batch_step_by_step(self, action_sequence_batch, task_graph_batch,
                                      max_running_time_batch, min_running_time_batch):
        target_batch = []
        task_finish_time_batch = []
        
        for i in range(len(action_sequence_batch)):
            task_graph = task_graph_batch[i]
            self.resource_cluster.reset()
            plan = action_sequence_batch[i]
            
            if self.use_difference_reward:
                # Use new difference reward scheme
                rewards, task_finish_time = self.get_scheduling_cost_with_difference_reward(plan, task_graph)
                score = np.array(rewards)
            else:
                # Use original reward scheme
                max_running_time = max_running_time_batch[i]
                min_running_time = min_running_time_batch[i]
                cost, task_finish_time = self.get_scheduling_cost_step_by_step(plan, task_graph)
                latency = self.score_func(cost, max_running_time, min_running_time)
                score = np.array(latency)
            
            target_batch.append(score)
            task_finish_time_batch.append(task_finish_time)

        target_batch = np.array(target_batch)
        return target_batch, task_finish_time_batch

    def greedy_solution(self):
        result_plan = []
        finish_time_batchs = []
        for task_graph_batch in self.task_graphs_batchs:
            plan_batchs = []
            finish_time_plan = []
            for task_graph in task_graph_batch:
                cloud_avaliable_time = 0.0
                ws_avaliable_time = 0.0
                local_avaliable_time = 0.0

                # finish time on cloud for each task
                FT_cloud = [0] * task_graph.task_number
                # finish time on sending channel for each task
                FT_ws = [0] * task_graph.task_number
                # finish time locally for each task
                FT_locally = [0] * task_graph.task_number
                # finish time recieving channel for each task
                FT_wr = [0] * task_graph.task_number
                plan = []

                for i in task_graph.prioritize_sequence:
                    task = task_graph.task_list[i]

                    # calculate the local finish time
                    if len(task_graph.pre_task_sets[i]) != 0:
                        start_time = max(local_avaliable_time,
                                         max([max(FT_locally[j], FT_wr[j]) for j in task_graph.pre_task_sets[i]]))
                    else:
                        start_time = local_avaliable_time

                    local_running_time = self.resource_cluster.locally_execution_cost(task.processing_data_size)
                    FT_locally[i] = start_time + local_running_time

                    # calculate the remote finish time
                    if len(task_graph.pre_task_sets[i]) != 0:
                        ws_start_time = max(ws_avaliable_time,
                                            max([max(FT_locally[j], FT_ws[j]) for j in task_graph.pre_task_sets[i]]))
                        FT_ws[i] = ws_start_time + self.resource_cluster.up_transmission_cost(task.processing_data_size)
                        cloud_start_time = max(cloud_avaliable_time,
                                               max([max(FT_ws[i], FT_cloud[j]) for j in task_graph.pre_task_sets[i]]))
                        cloud_finish_time = cloud_start_time + self.resource_cluster.mec_execution_cost(
                            task.processing_data_size)
                        FT_cloud[i] = cloud_finish_time
                        # print("task {}, Cloud finish time {}".format(i, FT_cloud[i]))
                        wr_start_time = FT_cloud[i]
                        wr_finish_time = wr_start_time + self.resource_cluster.dl_transmission_cost(task.transmission_data_size)
                        FT_wr[i] = wr_finish_time
                    else:
                        ws_start_time = ws_avaliable_time
                        ws_finish_time = ws_start_time + self.resource_cluster.up_transmission_cost(task.processing_data_size)
                        FT_ws[i] = ws_finish_time

                        cloud_start_time = max(cloud_avaliable_time, FT_ws[i])
                        FT_cloud[i] = cloud_start_time + self.resource_cluster.mec_execution_cost(
                            task.processing_data_size)
                        FT_wr[i] = FT_cloud[i] + self.resource_cluster.dl_transmission_cost(task.transmission_data_size)

                    if FT_locally[i] < FT_wr[i]:
                        action = 0
                        local_avaliable_time = FT_locally[i]
                        FT_wr[i] = 0.0
                        FT_cloud[i] = 0.0
                        FT_ws[i] = 0.0
                    else:
                        action = 1
                        FT_locally[i] = 0.0
                        cloud_avaliable_time = FT_cloud[i]
                        ws_avaliable_time = FT_ws[i]
                    plan.append((i, action))

                finish_time = max( max(FT_wr), max(FT_locally) )
                plan_batchs.append(plan)
                finish_time_plan.append(finish_time)

            finish_time_batchs.append(finish_time_plan)
            result_plan.append(plan_batchs)

        return result_plan, finish_time_batchs

    def calculate_optimal_solution(self):
        # Finding the optimal solution via exhausting search the solution space.
        def exhaustion_plans(n):
            plan_batch = []

            for i in range(2**n):
                plan_str = bin(i)
                plan = []

                for x in plan_str[2:]:
                    plan.append(int(x))

                while len(plan) < n:
                    plan.insert(0, 0)
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

    def greedy_solution_for_current_task(self):
        result_plan, finish_time_batchs = self.greedy_solution()

        return result_plan[self.task_id], finish_time_batchs[self.task_id]
    
    def get_diagnostic_summary(self):
        """
        Returns summary statistics of diagnostic data for logging.
        """
        if not self.diagnostic_data['r_norm']:
            return {}
        
        summary = {
            'avg_delta_loc': np.mean(self.diagnostic_data['delta_loc']),
            'avg_delta_rem': np.mean(self.diagnostic_data['delta_rem']),
            'avg_delta_chosen': np.mean(self.diagnostic_data['delta_chosen']),
            'avg_delta_best': np.mean(self.diagnostic_data['delta_best']),
            'avg_r_main': np.mean(self.diagnostic_data['r_main']),
            'avg_r_norm': np.mean(self.diagnostic_data['r_norm']),
            'fraction_best_action': np.mean(self.diagnostic_data['action_is_best']),
            'fraction_positive_reward': np.mean([r > 0 for r in self.diagnostic_data['r_norm']])
        }
        
        # Add CP-related diagnostics if shaping is enabled
        if self.cp_shaping_enabled and self.diagnostic_data['cp_before']:
            summary.update({
                'avg_cp_before': np.mean(self.diagnostic_data['cp_before']),
                'avg_cp_after': np.mean(self.diagnostic_data['cp_after']),
                'avg_cp_reduction': np.mean(self.diagnostic_data['cp_reduction']),
                'avg_shaping_contribution': np.mean(self.diagnostic_data['shaping_contribution']),
                'avg_shaping_ratio': np.mean(self.diagnostic_data['shaping_ratio']),
                'avg_final_step_score': np.mean(self.diagnostic_data['final_step_score']),
                'max_shaping_ratio': np.max(self.diagnostic_data['shaping_ratio']),
                'fraction_positive_final': np.mean([r > 0 for r in self.diagnostic_data['final_step_score']])
            })
        
        return summary



