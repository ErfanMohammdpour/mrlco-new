from env.base import MetaEnv
from env.mec_offloaing_envs.offloading_task_graph import OffloadingTaskGraph

from samplers.vectorized_env_executor import MetaIterativeEnvExecutor
import numpy as np
import os


class Resources(object):
    """
    MEC server & Mobile device resources.

    Args:
        mec_process_capable: MEC compute capacity (bytes/sec)
        mobile_process_capable: mobile compute capacity (bytes/sec)
        bandwidth_up: uplink bandwidth (Mbps)
        bandwidth_dl: downlink bandwidth (Mbps)
    """
    def __init__(self, mec_process_capable,
                 mobile_process_capable, bandwidth_up=7.0, bandwidth_dl=7.0):
        self.mec_process_capble = mec_process_capable
        self.mobile_process_capable = mobile_process_capable
        self.mobile_process_avaliable_time = 0.0
        self.mec_process_avaliable_time = 0.0

        self.bandwidth_up = bandwidth_up
        self.bandwidth_dl = bandwidth_dl

    def up_transmission_cost(self, data):
        rate = self.bandwidth_up * (1024.0 * 1024.0 / 8.0)
        return data / rate

    def reset(self):
        self.mec_process_avaliable_time = 0.0
        self.mobile_process_avaliable_time = 0.0

    def dl_transmission_cost(self, data):
        rate = self.bandwidth_dl * (1024.0 * 1024.0 / 8.0)
        return data / rate

    def locally_execution_cost(self, data):
        return self._computation_cost(data, self.mobile_process_capable)

    def mec_execution_cost(self, data):
        return self._computation_cost(data, self.mec_process_capble)

    def _computation_cost(self, data, processing_power):
        return data / processing_power


class OffloadingEnvironment(MetaEnv):
    def __init__(self, resource_cluster, batch_size,
                 graph_number,
                 graph_file_paths, time_major,
                 # === NEW: Lookahead counterfactual reward config ===
                 use_cf_lookahead=True,
                 cf_norm_eps=1e-6,
                 lookahead_every_k=1,        # do lookahead every k steps (1 = every step)
                 use_terminal_bonus=False,
                 terminal_bonus_weight=0.1):
        self.resource_cluster = resource_cluster
        self.task_graphs_batchs = []
        self.encoder_batchs = []
        self.encoder_lengths = []
        self.decoder_full_lengths = []
        self.max_running_time_batchs = []
        self.min_running_time_batchs = []
        self.graph_file_paths = graph_file_paths

        # load all graphs
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

        # paths
        self.graph_file_paths = graph_file_paths
        self.graph_number = graph_number

        # quick refs
        self.local_exe_time = self.get_all_locally_execute_time()
        self.mec_exe_time = self.get_all_mec_execute_time()

        # === NEW: reward config ===
        self.use_cf_lookahead = bool(use_cf_lookahead)
        self.cf_norm_eps = float(cf_norm_eps)
        self.lookahead_every_k = int(lookahead_every_k)
        self.use_terminal_bonus = bool(use_terminal_bonus)
        self.terminal_bonus_weight = float(terminal_bonus_weight)

    # ---------- MetaEnv API ----------
    def sample_tasks(self, n_tasks):
        return np.random.choice(np.arange(self.total_task), n_tasks, replace=False)

    def merge_graphs(self):
        encoder_batchs, encoder_lengths, task_graphs_batchs = [], [], []
        decoder_full_lengths, max_running_time_batchs, min_running_time_batchs = [], [], []

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
        self.task_id = task

    def get_task(self):
        return self.graph_file_paths[self.task_id]

    def step(self, action):
        """
        Given an action sequence for each graph in the current task-batch,
        returns (observation, reward_batch, done, info).
        reward_batch is [B, T] step-wise rewards.
        """
        plan_batch = []
        task_graph_batch = self.task_graphs_batchs[self.task_id]
        max_running_time_batch = self.max_running_time_batchs[self.task_id]
        min_running_time_batch = self.min_running_time_batchs[self.task_id]

        for action_sequence, task_graph in zip(action, task_graph_batch):
            plan_sequence = []
            for a, task_id in zip(action_sequence, task_graph.prioritize_sequence):
                plan_sequence.append((task_id, a))
            plan_batch.append(plan_sequence)

        rewards, finish_times = self.get_reward_batch_step_by_step(
            plan_batch, task_graph_batch, max_running_time_batch, min_running_time_batch
        )

        done = True
        observation = np.array(self.encoder_batchs[self.task_id])
        info = finish_times
        return observation, rewards, done, info

    def reset(self):
        self.resource_cluster.reset()
        return np.array(self.encoder_batchs[self.task_id])

    def render(self, mode='human'):
        pass

    # ---------- Data prep ----------
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

            # prioritize sequence is stored in the graph
            scheduling_sequence = task_graph.prioritize_tasks(self.resource_cluster)

            task_encode = np.array(task_graph.encode_point_sequence_with_ranking_and_cost(
                scheduling_sequence, self.resource_cluster
            ), dtype=np.float32)
            encoder_list.append(task_encode)

        for i in range(int(graph_number / batch_size)):
            s = i * batch_size
            e = (i + 1) * batch_size
            task_encode_batch = encoder_list[s:e]
            if time_major:
                task_encode_batch = np.array(task_encode_batch).swapaxes(0, 1)
                sequence_length = np.asarray([task_encode_batch.shape[0]] * task_encode_batch.shape[1])
            else:
                task_encode_batch = np.array(task_encode_batch)
                sequence_length = np.asarray([task_encode_batch.shape[1]] * task_encode_batch.shape[0])

            decoder_full_lengths.append(sequence_length)
            encoder_lengths.append(sequence_length)
            encoder_batchs.append(task_encode_batch)

            task_graph_batch = task_graph_list[s:e]
            task_graph_batchs.append(task_graph_batch)
            max_running_time_batchs.append(max_running_time_vector[s:e])
            min_running_time_batchs.append(min_running_time_vector[s:e])

        return (encoder_batchs, encoder_lengths, task_graph_batchs,
                decoder_full_lengths, max_running_time_batchs, min_running_time_batchs)

    def calculate_max_min_runningcost(self, max_data_size, min_data_size):
        max_time = max([
            self.resource_cluster.up_transmission_cost(max_data_size),
            self.resource_cluster.dl_transmission_cost(max_data_size),
            self.resource_cluster.locally_execution_cost(max_data_size)
        ])
        min_time = self.resource_cluster.mec_execution_cost(min_data_size)
        return max_time, min_time

    # ---------- Core simulator + Lookahead Counterfactual ----------
    def _snapshot_state(self, cloud_avaliable_time, ws_avaliable_time, local_avaliable_time,
                        FT_cloud, FT_ws, FT_locally, FT_wr, current_FT):
        """Light snapshot for counterfactual lookahead."""
        return {
            "cloud": cloud_avaliable_time,
            "ws": ws_avaliable_time,
            "local": local_avaliable_time,
            "FT_cloud": FT_cloud[:],
            "FT_ws": FT_ws[:],
            "FT_locally": FT_locally[:],
            "FT_wr": FT_wr[:],
            "current_FT": current_FT
        }

    def _apply_one_task(self, snap, task_graph, task_id, action):
        """
        Apply one task (task_id) with 'action' (0 local / 1 remote) on a snapshot,
        update the snapshot in-place, and return the finish time of that task.
        """
        cloud_avaliable_time = snap["cloud"]
        ws_avaliable_time = snap["ws"]
        local_avaliable_time = snap["local"]
        FT_cloud = snap["FT_cloud"]
        FT_ws = snap["FT_ws"]
        FT_locally = snap["FT_locally"]
        FT_wr = snap["FT_wr"]
        current_FT = snap["current_FT"]

        task = task_graph.task_list[task_id]

        # compute candidate local/remote finish times from the snapshot state
        # --- LOCAL candidate ---
        if len(task_graph.pre_task_sets[task_id]) != 0:
            local_start = max(local_avaliable_time,
                              max([max(FT_locally[j], FT_wr[j]) for j in task_graph.pre_task_sets[task_id]]))
        else:
            local_start = local_avaliable_time
        local_finish = local_start + self.resource_cluster.locally_execution_cost(task.processing_data_size)

        # --- REMOTE candidate ---
        if len(task_graph.pre_task_sets[task_id]) != 0:
            ws_start = max(ws_avaliable_time,
                           max([max(FT_locally[j], FT_ws[j]) for j in task_graph.pre_task_sets[task_id]]))
            ws_finish = ws_start + self.resource_cluster.up_transmission_cost(task.processing_data_size)
            cloud_start = max(cloud_avaliable_time,
                              max([max(ws_finish, FT_cloud[j]) for j in task_graph.pre_task_sets[task_id]]))
            cloud_finish = cloud_start + self.resource_cluster.mec_execution_cost(task.processing_data_size)
            remote_finish = cloud_finish + self.resource_cluster.dl_transmission_cost(task.transmission_data_size)
        else:
            ws_start = ws_avaliable_time
            ws_finish = ws_start + self.resource_cluster.up_transmission_cost(task.processing_data_size)
            cloud_start = max(cloud_avaliable_time, ws_finish)
            cloud_finish = cloud_start + self.resource_cluster.mec_execution_cost(task.processing_data_size)
            remote_finish = cloud_finish + self.resource_cluster.dl_transmission_cost(task.transmission_data_size)

        if action == 0:
            # choose LOCAL
            FT_locally[task_id] = local_finish
            snap["local"] = FT_locally[task_id]
            # reset remote siblings
            FT_ws[task_id] = 0.0; FT_cloud[task_id] = 0.0; FT_wr[task_id] = 0.0
            task_finish_time = local_finish
        else:
            # choose REMOTE
            FT_ws[task_id] = ws_finish
            snap["ws"] = FT_ws[task_id]
            FT_cloud[task_id] = cloud_finish
            snap["cloud"] = FT_cloud[task_id]
            FT_wr[task_id] = remote_finish
            # reset local sibling
            FT_locally[task_id] = 0.0
            task_finish_time = remote_finish

        snap["current_FT"] = max(task_finish_time, current_FT)
        return task_finish_time

    def _greedy_finish_remaining(self, snap, task_graph):
        """
        From a given snapshot (some tasks may already be finished),
        greedily schedule remaining tasks until the end.
        Returns final makespan (T_final).
        """
        FT_cloud = snap["FT_cloud"]
        FT_ws = snap["FT_ws"]
        FT_locally = snap["FT_locally"]
        FT_wr = snap["FT_wr"]

        finished = set([i for i in range(task_graph.task_number)
                        if (FT_locally[i] > 0.0 or FT_wr[i] > 0.0)])
        for task_id in task_graph.prioritize_sequence:
            if task_id in finished:
                continue
            # evaluate both choices and pick the one with smaller finish time
            # make temporary copies to probe both choices cheaply
            # (lightweight: just duplicate snapshot and apply once)
            snap_local = self._snapshot_state(snap["cloud"], snap["ws"], snap["local"],
                                              FT_cloud, FT_ws, FT_locally, FT_wr, snap["current_FT"])
            fin_local = self._apply_one_task(snap_local, task_graph, task_id, action=0)

            snap_remote = self._snapshot_state(snap["cloud"], snap["ws"], snap["local"],
                                               FT_cloud, FT_ws, FT_locally, FT_wr, snap["current_FT"])
            fin_remote = self._apply_one_task(snap_remote, task_graph, task_id, action=1)

            if fin_local <= fin_remote:
                # commit local choice into main snapshot
                self._apply_one_task(snap, task_graph, task_id, action=0)
            else:
                self._apply_one_task(snap, task_graph, task_id, action=1)

        # final makespan
        return max(max(snap["FT_wr"]), max(snap["FT_locally"]))

    def get_scheduling_cost_step_by_step(self, plan, task_graph,
                                         want_cf_lookahead=False,
                                         cf_norm_eps=1e-6,
                                         lookahead_every_k=1):
        """
        Simulate a given plan step-by-step. If want_cf_lookahead=True,
        also compute per-step lookahead counterfactual rewards with greedy completion.

        Returns:
          - if want_cf_lookahead=False: (delta_list, final_makespan)
          - if want_cf_lookahead=True : (delta_list, final_makespan, cf_long_list)
        """
        cloud_avaliable_time = 0.0
        ws_avaliable_time = 0.0
        local_avaliable_time = 0.0

        T_l = [0] * task_graph.task_number
        T_ul = [0] * task_graph.task_number
        T_dl = [0] * task_graph.task_number

        FT_cloud = [0] * task_graph.task_number
        FT_ws = [0] * task_graph.task_number
        FT_locally = [0] * task_graph.task_number
        FT_wr = [0] * task_graph.task_number

        current_FT = 0.0
        delta_list = []
        cf_long_list = [] if want_cf_lookahead else None

        for step_idx, item in enumerate(plan):
            i = item[0]
            task = task_graph.task_list[i]
            x = item[1]  # 0 local, 1 remote

            # candidate local/remote finish times from the CURRENT main state
            # --- LOCAL candidate ---
            if len(task_graph.pre_task_sets[i]) != 0:
                local_start = max(local_avaliable_time,
                                  max([max(FT_locally[j], FT_wr[j]) for j in task_graph.pre_task_sets[i]]))
            else:
                local_start = local_avaliable_time
            local_finish = local_start + self.resource_cluster.locally_execution_cost(task.processing_data_size)

            # --- REMOTE candidate ---
            if len(task_graph.pre_task_sets[i]) != 0:
                ws_start = max(ws_avaliable_time,
                               max([max(FT_locally[j], FT_ws[j]) for j in task_graph.pre_task_sets[i]]))
                ws_finish = ws_start + self.resource_cluster.up_transmission_cost(task.processing_data_size)
                cloud_start = max(cloud_avaliable_time,
                                  max([max(ws_finish, FT_cloud[j]) for j in task_graph.pre_task_sets[i]]))
                cloud_finish = cloud_start + self.resource_cluster.mec_execution_cost(task.processing_data_size)
                remote_finish = cloud_finish + self.resource_cluster.dl_transmission_cost(task.transmission_data_size)
            else:
                ws_start = ws_avaliable_time
                ws_finish = ws_start + self.resource_cluster.up_transmission_cost(task.processing_data_size)
                cloud_start = max(cloud_avaliable_time, ws_finish)
                cloud_finish = cloud_start + self.resource_cluster.mec_execution_cost(task.processing_data_size)
                remote_finish = cloud_finish + self.resource_cluster.dl_transmission_cost(task.transmission_data_size)

            delta_local = max(local_finish, current_FT) - current_FT
            delta_remote = max(remote_finish, current_FT) - current_FT

            # --- Lookahead counterfactual (long-horizon via greedy completion) ---
            if want_cf_lookahead and (step_idx % max(1, lookahead_every_k) == 0):
                # snapshot BEFORE committing the real action
                snap = self._snapshot_state(cloud_avaliable_time, ws_avaliable_time, local_avaliable_time,
                                            FT_cloud, FT_ws, FT_locally, FT_wr, current_FT)

                # Branch 1: commit REAL action for this task, then greedy-complete remaining
                snap_sel = self._snapshot_state(**snap) if isinstance(snap, dict) else snap
                self._apply_one_task(snap_sel, task_graph, i, action=x)
                T_sel = self._greedy_finish_remaining(snap_sel, task_graph)

                # Branch 2: commit ALT action (1-x) for this task, then greedy-complete remaining
                snap_alt = self._snapshot_state(**snap) if isinstance(snap, dict) else snap
                self._apply_one_task(snap_alt, task_graph, i, action=1 - x)
                T_alt = self._greedy_finish_remaining(snap_alt, task_graph)

                denom = max(cf_norm_eps, (T_sel + T_alt))
                r_cf_long = - (T_sel - T_alt) / denom
                cf_long_list.append(r_cf_long)

            # --- Commit REAL action into the MAIN trajectory ---
            if x == 0:
                # LOCAL
                FT_locally[i] = local_finish
                local_avaliable_time = FT_locally[i]
                FT_ws[i] = 0.0; FT_cloud[i] = 0.0; FT_wr[i] = 0.0
                task_finish_time = local_finish
                delta_selected = delta_local
            else:
                # REMOTE
                FT_ws[i] = ws_finish
                ws_avaliable_time = FT_ws[i]
                FT_cloud[i] = cloud_finish
                cloud_avaliable_time = FT_cloud[i]
                FT_wr[i] = remote_finish
                FT_locally[i] = 0.0
                task_finish_time = remote_finish
                delta_selected = delta_remote

            # record Δmakespan of this step for potential legacy uses
            delta_list.append(delta_selected)

            # advance current makespan
            current_FT = max(task_finish_time, current_FT)

        if want_cf_lookahead:
            return delta_list, current_FT, cf_long_list
        else:
            return delta_list, current_FT

    def score_func(self, cost, max_time, min_time):
        """Legacy normalization (kept for compatibility)."""
        return -(cost - min_time) / (max_time - min_time)

    def compute_simple_baseline(self, task_graph):
        """Cheap baseline: min(all-local, all-remote). Used only for tiny terminal bonus."""
        saved_mobile = self.resource_cluster.mobile_process_avaliable_time
        saved_mec = self.resource_cluster.mec_process_avaliable_time

        self.resource_cluster.reset()
        all_local_plan = [(i, 0) for i in task_graph.prioritize_sequence]
        _, all_local_time = self.get_scheduling_cost_step_by_step(all_local_plan, task_graph)

        self.resource_cluster.reset()
        all_remote_plan = [(i, 1) for i in task_graph.prioritize_sequence]
        _, all_remote_time = self.get_scheduling_cost_step_by_step(all_remote_plan, task_graph)

        self.resource_cluster.mobile_process_avaliable_time = saved_mobile
        self.resource_cluster.mec_process_avaliable_time = saved_mec

        return min(all_local_time, all_remote_time)

    def get_reward_batch_step_by_step(self, action_sequence_batch, task_graph_batch,
                                      max_running_time_batch, min_running_time_batch):
        """
        Returns:
            rewards_batch: np.array [B, T] of per-step rewards
            task_finish_time_batch: list[float] final makespans
        """
        rewards_batch = []
        task_finish_time_batch = []

        for i in range(len(action_sequence_batch)):
            task_graph = task_graph_batch[i]
            self.resource_cluster.reset()
            plan = action_sequence_batch[i]

            if self.use_cf_lookahead:
                deltas, task_finish_time, cf_long = self.get_scheduling_cost_step_by_step(
                    plan, task_graph,
                    want_cf_lookahead=True,
                    cf_norm_eps=self.cf_norm_eps,
                    lookahead_every_k=self.lookahead_every_k
                )
                rewards = np.array(cf_long, dtype=np.float32)
            else:
                # fallback (legacy normalized -ΔT)
                deltas, task_finish_time = self.get_scheduling_cost_step_by_step(
                    plan, task_graph,
                    want_cf_lookahead=False
                )
                rewards = self.score_func(
                    np.array(deltas, dtype=np.float32),
                    max_running_time_batch[i], min_running_time_batch[i]
                ).astype(np.float32)

            # optional tiny terminal bonus for episode scaling
            if self.use_terminal_bonus and rewards.size > 0:
                B = self.compute_simple_baseline(task_graph)
                tb = 0.0 if B <= 0 else self.terminal_bonus_weight * (B - task_finish_time) / B
                rewards[-1] += float(tb)

            rewards_batch.append(rewards)
            task_finish_time_batch.append(task_finish_time)

        return np.array(rewards_batch, dtype=np.float32), task_finish_time_batch

    # ---------- Helpers for diagnostics/greedy ----------
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

                FT_cloud = [0] * task_graph.task_number
                FT_ws = [0] * task_graph.task_number
                FT_locally = [0] * task_graph.task_number
                FT_wr = [0] * task_graph.task_number
                plan = []

                for i in task_graph.prioritize_sequence:
                    task = task_graph.task_list[i]

                    # local candidate
                    if len(task_graph.pre_task_sets[i]) != 0:
                        start_time = max(local_avaliable_time,
                                         max([max(FT_locally[j], FT_wr[j]) for j in task_graph.pre_task_sets[i]]))
                    else:
                        start_time = local_avaliable_time
                    local_running_time = self.resource_cluster.locally_execution_cost(task.processing_data_size)
                    FT_locally[i] = start_time + local_running_time

                    # remote candidate
                    if len(task_graph.pre_task_sets[i]) != 0:
                        ws_start_time = max(ws_avaliable_time,
                                            max([max(FT_locally[j], FT_ws[j]) for j in task_graph.pre_task_sets[i]]))
                        FT_ws[i] = ws_start_time + self.resource_cluster.up_transmission_cost(task.processing_data_size)
                        cloud_start_time = max(cloud_avaliable_time,
                                               max([max(FT_ws[i], FT_cloud[j]) for j in task_graph.pre_task_sets[i]]))
                        cloud_finish_time = cloud_start_time + self.resource_cluster.mec_execution_cost(
                            task.processing_data_size)
                        FT_cloud[i] = cloud_finish_time
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
                        FT_wr[i] = 0.0; FT_cloud[i] = 0.0; FT_ws[i] = 0.0
                    else:
                        action = 1
                        FT_locally[i] = 0.0
                        cloud_avaliable_time = FT_cloud[i]
                        ws_avaliable_time = FT_ws[i]
                    plan.append((i, action))

                finish_time = max(max(FT_wr), max(FT_locally))
                plan_batchs.append(plan)
                finish_time_plan.append(finish_time)

            finish_time_batchs.append(finish_time_plan)
            result_plan.append(plan_batchs)

        return result_plan, finish_time_batchs

    def calculate_optimal_solution(self):
        # Exhaustive (warning: exponential)
        def exhaustion_plans(n):
            plan_batch = []
            for i in range(2 ** n):
                plan_str = bin(i)
                plan = [int(x) for x in plan_str[2:]]
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

                    _, task_finish_time = self.get_scheduling_cost_step_by_step(plan_sequence, task_graph)
                    plans_costs.append(task_finish_time)
                    prioritize_plan.append(plan_sequence)

                optimal_plan.append(prioritize_plan[np.argmin(plans_costs)])
                task_graph_batch_cost.append(min(plans_costs))

            print("task_graph_batch cost shape is {}".format(np.array(task_graph_batch_cost).shape))
            task_graph_optimal_costs.append(np.mean(task_graph_batch_cost))

        self.optimal_solution = task_graph_optimal_costs
        return task_graph_optimal_costs, optimal_plan

    # ---------- misc ----------
    def get_running_cost(self, action_sequence_batch, task_graph_batch):
        cost_batch = []
        for action_sequence, task_graph in zip(action_sequence_batch, task_graph_batch):
            plan_sequence = []
            for action, task_id in zip(action_sequence, task_graph.prioritize_sequence):
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
