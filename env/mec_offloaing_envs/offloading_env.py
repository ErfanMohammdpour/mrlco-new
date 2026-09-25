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
                  use_energy=False, energy_config=None, scheduler_config=None):
        # resolved scheduler config (timing/accounting/scope axes). None keeps the
        # legacy rebuild path alive; primary training paths require it.
        self.scheduler_config = scheduler_config
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
        
        # Store energy weights if enabled — v0.1 publication freeze is 0.5/0.5.
        # Diagnostic latency_over_all_mec is not publication; skip the freeze check.
        self.reward_mode = str(energy_config.get("reward_mode", "publication"))
        if self.reward_mode == "latency_over_all_mec":
            self.latency_weight = 1.0
            self.energy_weight = 0.0
        elif self.reward_mode == "latency_only":
            # E3.1 primary: no energy term, so the legacy weight pair is latency-only
            self.latency_weight = 1.0
            self.energy_weight = 0.0
        elif self.use_energy:
            from env.mec_offloaing_envs.scheduler.energy_api import require_publication_weights

            lw = float(energy_config.get("latency_weight", 0.5))
            ew = float(energy_config.get("energy_weight", 0.5))
            self.latency_weight, self.energy_weight = require_publication_weights(lw, ew)
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
        from env.mec_offloaing_envs.scheduler import resource_config_from_cluster

        self.scheduler_resources = resource_config_from_cluster(resource_cluster)
        self.task_graphs_batchs = []
        self.encoder_batchs = []
        self.encoder_lengths = []
        self.decoder_full_lengths = []
        self.max_running_time_batchs = []
        self.min_running_time_batchs = []
        self.graph_file_paths = graph_file_paths

        # --- constrained-MDP layer (V2V / energy budgets) -------------------
        # Off by default: `ConstraintSpec()` has mode="off" and every reward
        # path below is then bit-identical to the unconstrained training path.
        from env.mec_offloaing_envs.scheduler.constraints import (
            ConstraintController,
            ConstraintSpec,
        )

        self.constraint_spec = ConstraintSpec.from_config(
            getattr(resource_cluster, "energy_config", None)
        )
        self.constraint_controller = getattr(resource_cluster, "constraint_controller", None)
        if self.constraint_spec.enabled and self.constraint_controller is None:
            self.constraint_controller = ConstraintController(spec=self.constraint_spec)
        if self.constraint_controller is not None:
            # keep the two objects consistent whatever order they were built in
            self.constraint_spec = self.constraint_controller.spec
        self.last_constraint_costs = None
        self.last_constraint_penalty = 0.0

        # 4.2c scoped telemetry: opt-in. OFF by default, so the legacy info tuple
        # and every sampler path are unchanged unless the trainer asks for it.
        self.energy_telemetry_enabled = bool(
            (getattr(resource_cluster, "energy_config", None) or {}).get(
                "energy_telemetry", False
            )
        )
        self.last_energy_telemetry = None
        # 4.2b/E4.2 validation objective channel: opt-in, off by default.
        self.validation_plans_enabled = bool(
            (getattr(resource_cluster, "energy_config", None) or {}).get(
                "validation_plans", False
            )
        )
        self.last_validation_plans = None

        # Discount-consistent telescoping: r_t = J_{t-1} - shaping_discount * J_t.
        # Must equal the PPO discount so that the shaped return stays aligned with
        # the final schedule objective (sum_t gamma^(t-1) r_t = J_0 - gamma^N J_N).
        # 1.0 reproduces the historical delta-form rewards exactly.
        self.shaping_discount = float(
            (getattr(resource_cluster, "energy_config", None) or {}).get(
                "shaping_discount", 1.0
            )
        )

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
        self.graph_indices = None
        self.support_graphs_per_task = 20
        self.time_major = time_major
        self.input_dim = np.array(encoder_batchs[0]).shape[-1]

        # set the file paht of task graphs.
        self.graph_file_paths = graph_file_paths
        self.greedy_actions = None  # None = publication ternary (0, 1, 2)
        from spec.split_loader import parse_distribution_id

        self.distribution_ids = [parse_distribution_id(path) for path in graph_file_paths]
        self.graph_number = graph_number


        self.local_exe_time = self.get_all_locally_execute_time()
        self.mec_exe_time = self.get_all_mec_execute_time()

    def sample_tasks(self, n_tasks):
        """
        Samples task of the meta-environment

        Args:
            n_tasks (int) : number of different meta-tasks needed

        Returns:
            tasks (list) : an (n_tasks) length list of {dist_index, graph_indices}
        """
        dist_ids = np.random.choice(np.arange(self.total_task), n_tasks, replace=False)
        tasks = []
        for dist_id in dist_ids:
            n_graphs = len(self.task_graphs_batchs[int(dist_id)])
            k = min(int(self.support_graphs_per_task), n_graphs)
            if k < n_graphs:
                graph_indices = np.random.choice(n_graphs, k, replace=False)
            else:
                graph_indices = np.arange(n_graphs)
            tasks.append({"dist_index": int(dist_id), "graph_indices": np.asarray(graph_indices, dtype=np.int32)})
        return tasks

    def _slice_current(self, batch):
        item = batch[self.task_id]
        if self.graph_indices is None:
            return item
        idx = np.asarray(self.graph_indices, dtype=np.int32)
        if isinstance(item, np.ndarray):
            return item[idx]
        return [item[int(i)] for i in idx]

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
        if isinstance(task, dict):
            self.task_id = int(task["dist_index"])
            self.graph_indices = np.asarray(task["graph_indices"], dtype=np.int32)
        else:
            self.task_id = int(task)
            self.graph_indices = None

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
        task_graph_batch = self._slice_current(self.task_graphs_batchs)
        max_running_time_batch = self._slice_current(self.max_running_time_batchs)
        min_running_time_batch = self._slice_current(self.min_running_time_batchs)

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
            # Include energy in info for logging. The legacy 2-tuple stays intact;
            # telemetry is appended as a third element only when it was enabled.
            if self.energy_telemetry_enabled and self.last_energy_telemetry is not None:
                info = (task_finish_time, energy_batch, self.last_energy_telemetry)
            else:
                info = (task_finish_time, energy_batch)
        else:
            reward_batch, task_finish_time = result
            info = task_finish_time

        done = True
        observation = np.array(self._slice_current(self.encoder_batchs))

        return observation, reward_batch, done, info

    def reset(self):
        """Resets the state of the environment and returns an initial observation.

        Returns:
            observation (object): the initial observation.
        """
        # reset the resource environment.
        self.resource_cluster.reset()

        return np.array(self._slice_current(self.encoder_batchs))

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
        """Legacy entrypoint: delegates all scheduling math to the canonical engine."""
        from env.mec_offloaing_envs.scheduler import schedule_via_adapter

        result, latency_deltas, energy_list = schedule_via_adapter(
            task_graph=task_graph,
            plan=plan,
            resources=self.scheduler_resources,
        )
        if self.resource_cluster.use_energy:
            return latency_deltas, result.makespan_seconds, energy_list
        return latency_deltas, result.makespan_seconds

    def score_func(self, cost, max_time, min_time):
        """Score function that handles both scalars and lists/arrays.
        
        For lists/arrays, computes element-wise scoring.
        """
        cost = np.asarray(cost)
        return -(cost - min_time) / (max_time - min_time)
    
    def get_reference_ranges(self, task_graph, scope=None):
        """Episode-local L/E ranges; cached by content key, not object id.

        `reference_range` in `energy_config` selects the construction:
          "pure_location"   -> all_UE / all_MEC / all_HELPER  (MARGO-SPEC-v0.1)
          "candidate_panel" -> plus greedy_from_mec (fixes the clipped-j_report
                               inversion; costs one extra local search per graph,
                               so it is cached here and never recomputed per step)

        `scope` (E3.2) is the boundary the caller needs; per-consumer references
        are fetched per scope and the cache key binds scope + fingerprint. It
        defaults to the declared primary scope (system). A system consumer never
        receives a mobile object: the hit is re-validated in
        `get_or_build_reference_ranges`.
        """
        from env.mec_offloaing_envs.scheduler.energy_cache import (
            get_or_build_reference_ranges,
            primary_scope_of,
        )

        cache = getattr(self, "_refs_cache", None)
        if cache is None:
            cache = {}
            self._refs_cache = cache
        mode = str(
            (getattr(self.resource_cluster, "energy_config", None) or {}).get(
                "reference_range", "pure_location"
            )
        )
        energy_scope = primary_scope_of(self.scheduler_resources) if scope is None else str(scope)
        return get_or_build_reference_ranges(
            cache,
            graph=task_graph,
            order=[int(tid) for tid in task_graph.prioritize_sequence],
            reference_mode=mode,
            energy_scope=energy_scope,
            resources=self.scheduler_resources,
            panel_max_passes=2,
        )

    def get_reward_batch_step_by_step(self, action_sequence_batch, task_graph_batch,
                                      max_running_time_batch, min_running_time_batch):
        """Post-hoc telescoping rewards (OBJECTIVE §6); max/min batch args unused."""
        from env.mec_offloaing_envs.scheduler.reward import (
            LATENCY_REF_L_MEC,
            REWARD_MODE_PUBLICATION,
            REWARD_MODE_LATENCY_ONLY,
            telescoping_token_rewards,
        )

        target_batch = []
        task_finish_time_batch = []
        energy_batch = []
        telemetry_batch = [] if self.energy_telemetry_enabled else None
        # `use_energy` still drives LOGGING (per-task mobile energy + telemetry);
        # the reward's energy term is its own decision (E3.1: latency-only primary).
        log_energy = bool(self.resource_cluster.use_energy)
        reward_mode = str(getattr(self.resource_cluster, "reward_mode", REWARD_MODE_LATENCY_ONLY))
        if reward_mode not in (REWARD_MODE_PUBLICATION, REWARD_MODE_LATENCY_ONLY, "latency_over_all_mec"):
            raise ValueError("unknown reward_mode %r" % (reward_mode,))
        reward_include_energy = log_energy and reward_mode == REWARD_MODE_PUBLICATION
        latency_ref = LATENCY_REF_L_MEC if reward_mode == "latency_over_all_mec" else "l_scale"
        # E3.2: the primary path is a SYSTEM consumer; legacy publication is the
        # only mobile one (byte-exact reproduction).
        from env.mec_offloaing_envs.scheduler.energy_cache import primary_scope_of
        from env.mec_offloaing_envs.scheduler.energy_scope import SCOPE_MOBILE

        ref_scope = (
            SCOPE_MOBILE
            if reward_mode == REWARD_MODE_PUBLICATION
            else primary_scope_of(self.scheduler_resources)
        )

        validation_batch = [] if self.validation_plans_enabled else None
        constraint_penalty_applied = 0.0

        for i in range(len(action_sequence_batch)):
            task_graph = task_graph_batch[i]
            self.resource_cluster.reset()
            plan = action_sequence_batch[i]
            refs = self.get_reference_ranges(task_graph, ref_scope)

            duals = (
                self.constraint_controller.lambdas
                if self.constraint_controller is not None
                else None
            )
            out = telescoping_token_rewards(
                task_graph,
                plan,
                self.scheduler_resources,
                include_energy=reward_include_energy,
                reward_mode=reward_mode,
                compute_j_report=False,
                latency_ref=latency_ref,
                refs=refs,
                constraints=self.constraint_spec,
                duals=duals,
                discount=getattr(self, "shaping_discount", 1.0),
            )
            if out.constraint_costs.active:
                self.last_constraint_costs = out.constraint_costs
                constraint_penalty_applied += float(out.constraint_penalty)
                if self.constraint_controller is not None:
                    self.constraint_controller.observe(out.constraint_costs)
            target_batch.append(np.asarray(out.rewards, dtype=float))
            task_finish_time_batch.append(out.final_makespan)
            if log_energy:
                energy_batch.append(out.final_per_task_energy)
            else:
                energy_batch.append([])
            if telemetry_batch is not None:
                # pure post-processing of the SAME result the reward came from:
                # no second schedule, no replay.
                from env.mec_offloaing_envs.scheduler.energy_telemetry import (
                    build_energy_telemetry,
                )

                telemetry_batch.append(
                    build_energy_telemetry(out.final_result, self.scheduler_resources)
                )
            if validation_batch is not None:
                # E4.2: keep the SAME result + reference for the validation
                # objective channel. No replay: the result is the reward's own.
                validation_batch.append(
                    self._validation_plan_record(task_graph, plan, out.final_result, refs)
                )

        self.last_energy_telemetry = telemetry_batch
        self.last_validation_plans = validation_batch
        self.last_constraint_penalty = float(constraint_penalty_applied)

        target_batch = np.array(target_batch, dtype=object)
        # Prefer numeric ndarray when all sequences share length.
        try:
            target_batch = np.asarray(target_batch.tolist(), dtype=float)
        except Exception:
            pass

        if log_energy:
            return target_batch, task_finish_time_batch, energy_batch
        return target_batch, task_finish_time_batch

    def _validation_plan_record(self, task_graph, plan, result, refs):
        """One graph's (identity, order, plan, result, system refs, fingerprint).

        Fail-loud: the result must record the resolved scheduler fingerprint and
        the reference must be the SYSTEM one E3.2 requires. A mobile or missing
        object raises instead of reaching the objective.
        """
        from env.mec_offloaing_envs.scheduler.energy_cache import (
            scheduling_graph_fingerprint,
        )
        from env.mec_offloaing_envs.scheduler.energy_scope import (
            SCOPE_SYSTEM,
            require_reference_scope,
        )
        from env.mec_offloaing_envs.scheduler.resources import resolved_config_sha256

        order = [int(t) for t in task_graph.prioritize_sequence]
        config_sha = str(resolved_config_sha256(self.scheduler_resources))
        result_sha = str(getattr(result, "scheduler_config_sha256", "") or "")
        if not result_sha or result_sha != config_sha:
            raise ValueError(
                "validation plan result fingerprint %r != resolved %r"
                % (result_sha[:12], config_sha[:12])
            )
        scope = require_reference_scope(
            refs,
            expected_scope=SCOPE_SYSTEM,
            expected_scheduler_config_sha256=config_sha,
        )
        return {
            "graph_fingerprint": scheduling_graph_fingerprint(task_graph, order),
            "order": order,
            "plan": [[int(t), int(a)] for t, a in plan],
            "scheduler_config_sha256": result_sha,
            "energy_scope": scope,
            "result": result,
            "reference_ranges": refs,
        }

    def validation_plan_payload(self):
        """`(result, system refs)` per current-task graph — the SAME results the
        reward used. Returns None when the channel is disabled or no rollout ran.
        """
        if not self.last_validation_plans:
            return None
        return [
            (record["result"], record["reference_ranges"])
            for record in self.last_validation_plans
        ]

    def validation_plan_identities(self):
        """Serializable identity of the validation plans (no ScheduleResult)."""
        if not self.last_validation_plans:
            return None
        return [
            {
                key: record[key]
                for key in (
                    "graph_fingerprint",
                    "order",
                    "plan",
                    "scheduler_config_sha256",
                    "energy_scope",
                )
            }
            for record in self.last_validation_plans
        ]

    def greedy_solution(self):
        """Greedy plan search; each candidate is scored by the canonical engine."""
        from env.mec_offloaing_envs.scheduler.energy_scope import (
            SCOPE_MOBILE,
            energy_scalar,
        )
        from env.mec_offloaing_envs.scheduler.greedy import greedy_plan

        result_plan = []
        finish_time_batchs = []
        energy_batchs = []

        for task_graph_batch in self.task_graphs_batchs:
            plan_batchs = []
            finish_time_plan = []
            energy_plan = []
            for task_graph in task_graph_batch:
                plan, scheduled = greedy_plan(
                    task_graph,
                    self.scheduler_resources,
                    actions=getattr(self, "greedy_actions", None),
                )
                plan_batchs.append(plan)
                finish_time_plan.append(scheduled.makespan_seconds)
                energy_plan.append(energy_scalar(scheduled, scope=SCOPE_MOBILE))
            finish_time_batchs.append(finish_time_plan)
            result_plan.append(plan_batchs)
            energy_batchs.append(energy_plan)

        if self.resource_cluster.use_energy:
            return result_plan, finish_time_batchs, energy_batchs
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

            # Call once after building the complete plan_sequence
            # Handle both energy-enabled and energy-disabled cases
            if self.resource_cluster.use_energy:
                _, task_finish_time, _ = self.get_scheduling_cost_step_by_step(plan_sequence, task_graph)
            else:
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
        greedy_result = self.greedy_solution()
        
        if self.resource_cluster.use_energy:
            result_plan, finish_time_batchs, energy_batchs = greedy_result
            return self._slice_current(result_plan), self._slice_current(finish_time_batchs), self._slice_current(energy_batchs)
        else:
            result_plan, finish_time_batchs = greedy_result
            return self._slice_current(result_plan), self._slice_current(finish_time_batchs)



