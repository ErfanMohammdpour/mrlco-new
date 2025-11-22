from samplers.base import Sampler
from samplers.vectorized_env_executor import MetaParallelEnvExecutor, MetaIterativeEnvExecutor
from utils import utils, logger
from collections import OrderedDict

from pyprind import ProgBar
import numpy as np
import time
import itertools
import tensorflow as tf

class Seq2SeqSampler(Sampler):
    """
    Sampler for MRLCO

    Args:
        env (meta_policy_search.envs.base.MetaEnv) : environment object
        policy (meta_policy_search.policies.base.Policy) : policy object
        batch_size (int) : number of trajectories per task
        meta_batch_size (int) : number of meta tasks
        max_path_length (int) : max number of steps per trajectory
        envs_per_task (int) : number of envs to run vectorized for each task (influences the memory usage)
    """

    def __init__(self,
                env,
                policy,
                rollouts_per_meta_task,
                max_path_length,
                envs_per_task=None,
                parallel=False
                ):
        super(Seq2SeqSampler, self).__init__(env, policy, rollouts_per_meta_task, max_path_length)
        assert hasattr(env, 'set_task')

        self.envs_per_task = rollouts_per_meta_task if envs_per_task is None else envs_per_task
        self.total_samples = rollouts_per_meta_task * max_path_length
        self.parallel = parallel
        self.total_timesteps_sampled = 0
        self.env = env

    def obtain_samples(self, log=False, log_prefix=''):
        """
        Collect batch_size trajectories from each task

        Args:
            log (boolean): whether to log sampling times
            log_prefix (str) : prefix for logger

        Returns:
            (dict) : A dict of paths of size [meta_batch_size] x (batch_size) x [5] x (max_path_length)
        """

        # initial setup / preparation
        paths = []

        n_samples = 0
        running_paths = dict()

        pbar = ProgBar(self.total_samples)
        policy_time, env_time = 0, 0

        policy = self.policy

        # initial reset of envs
        obses = self.env.reset()

        while n_samples < self.total_samples:
            # execute policy
            t = time.time()
            obs_per_task = np.array(obses)

            actions, logits, values = policy.get_actions(obs_per_task)
            policy_time += time.time() - t

            # step environments
            t = time.time()
            next_obses, rewards, dones, env_infos = self.env.step(actions)

            env_time += time.time() - t

            #  stack agent_infos and if no infos were provided (--> None) create empty dicts
            new_samples = 0
            # Handle energy if enabled (env_infos is a tuple (task_finish_time, energy_batch) when energy enabled)
            if isinstance(env_infos, tuple) and len(env_infos) == 2:
                task_finish_times_batch, energy_batch = env_infos
            else:
                task_finish_times_batch = env_infos
                energy_batch = None
            
            # Convert to list if needed (ensure it's iterable)
            if not isinstance(task_finish_times_batch, (list, np.ndarray)):
                task_finish_times_batch = [task_finish_times_batch]
            elif isinstance(task_finish_times_batch, np.ndarray) and task_finish_times_batch.ndim == 0:
                # Handle scalar numpy array
                task_finish_times_batch = [float(task_finish_times_batch)]
            elif isinstance(task_finish_times_batch, np.ndarray):
                # Convert numpy array to list for easier indexing
                task_finish_times_batch = task_finish_times_batch.tolist()
            
            if energy_batch is not None:
                if not isinstance(energy_batch, (list, np.ndarray)):
                    energy_batch = [energy_batch]
                elif isinstance(energy_batch, np.ndarray) and energy_batch.ndim == 0:
                    energy_batch = [energy_batch]
                elif isinstance(energy_batch, np.ndarray):
                    energy_batch = energy_batch.tolist()
            
            # Iterate over individual trajectories in the batch
            for i, (observation, action, logit, reward, value) in enumerate(zip(obses, actions, logits,
                                                                       rewards, values)):
                running_paths["observations"] = observation
                running_paths["actions"] = action
                running_paths["logits"] = logit
                running_paths["rewards"] = reward
                running_paths["values"] = value
                
                # Extract individual finish_time and energy from batch
                if i < len(task_finish_times_batch):
                    finish_time = task_finish_times_batch[i]
                    # Ensure finish_time is a scalar (not array)
                    if isinstance(finish_time, (list, np.ndarray)):
                        finish_time = finish_time[0] if len(finish_time) == 1 else float(finish_time[-1])
                    running_paths["finish_time"] = finish_time
                else:
                    running_paths["finish_time"] = 0.0
                
                if energy_batch is not None and i < len(energy_batch):
                    energy = energy_batch[i]
                    running_paths["energy"] = energy if isinstance(energy, (list, np.ndarray)) else [energy]
                else:
                    running_paths["energy"] = None
                
                # handling
                path_dict = dict(
                    observations=np.squeeze(np.asarray(running_paths["observations"])),
                    actions=np.squeeze(np.asarray(running_paths["actions"])),
                    logits=np.squeeze(np.asarray(running_paths["logits"])),
                    rewards=np.squeeze(np.asarray(running_paths["rewards"])),
                    values=np.squeeze(np.asarray(running_paths["values"])),
                    finish_time=np.squeeze(np.asarray(running_paths["finish_time"]))
                )
                
                # Add energy to path if available
                if running_paths["energy"] is not None:
                    path_dict["energy"] = np.asarray(running_paths["energy"])
                
                paths.append(path_dict)

                # if running path is done, add it to paths and empty the running path
                new_samples += len(running_paths["rewards"])
                running_paths = _get_empty_running_paths_dict()

            pbar.update(new_samples)
            n_samples += new_samples
            obses = next_obses
        pbar.stop()

        self.total_timesteps_sampled += self.total_samples
        if log:
            logger.logkv(log_prefix + "PolicyExecTime", policy_time)
            logger.logkv(log_prefix + "EnvExecTime", env_time)
        return paths


def _get_empty_running_paths_dict():
    return dict()



