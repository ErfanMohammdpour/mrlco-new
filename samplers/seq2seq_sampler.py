from samplers.base import Sampler
from samplers.vectorized_env_executor import ParallelEnvExecutor, IterativeEnvExecutor
from utils import utils, logger
from collections import OrderedDict

from pyprind import ProgBar
import numpy as np
import time
import itertools

class Seq2SeqSampler(Sampler):
    """
    Sampler for single-policy RL (non-meta)

    Args:
        env: environment object
        policy: policy object
        rollouts_per_task (int): number of trajectories per task
        max_path_length (int): max number of steps per trajectory
        parallel (bool): whether to use parallel execution
    """

    def __init__(self,
                env,
                policy,
                rollouts_per_task,
                max_path_length,
                parallel=False
                ):
        super(Seq2SeqSampler, self).__init__(env, policy, rollouts_per_task, max_path_length)
        
        self.rollouts_per_task = rollouts_per_task
        self.total_samples = rollouts_per_task * max_path_length
        self.parallel = parallel
        self.total_timesteps_sampled = 0

        # setup vectorized environment
        if self.parallel:
            self.vec_env = ParallelEnvExecutor(env, self.rollouts_per_task, self.max_path_length)
        else:
            self.vec_env = IterativeEnvExecutor(env, self.rollouts_per_task, self.max_path_length)

    def obtain_samples(self, log=False, log_prefix=''):
        """
        Collect rollouts_per_task trajectories from the environment

        Args:
            log (boolean): whether to log sampling times
            log_prefix (str): prefix for logger

        Returns:
            (list): A list of paths
        """

        # initial setup / preparation
        paths = []

        n_samples = 0
        running_paths = [_get_empty_running_paths_dict() for _ in range(self.vec_env.num_envs)]

        pbar = ProgBar(self.total_samples)
        policy_time, env_time = 0, 0

        policy = self.policy

        # initial reset of envs
        obses = self.vec_env.reset()

        while n_samples < self.total_samples:
            # execute policy
            t = time.time()
            # Convert list of observations to numpy array for policy
            if isinstance(obses, list):
                obses_array = np.array(obses)
            else:
                obses_array = obses
            actions, logits, values = policy.get_actions(obses_array)
            policy_time += time.time() - t

            # Convert actions to list format for vectorized environment
            if isinstance(actions, np.ndarray):
                actions_list = [actions[i] for i in range(len(actions))]
            else:
                actions_list = actions

            # step environments
            t = time.time()
            next_obses, rewards, dones, env_infos = self.vec_env.step(actions_list)
            env_time += time.time() - t

            # stack agent_infos and if no infos were provided (--> None) create empty dicts
            new_samples = 0
            
            # Convert actions, logits, values to lists if they're numpy arrays
            if isinstance(actions, np.ndarray):
                actions_list = [actions[i] for i in range(len(actions))]
            else:
                actions_list = actions
                
            if isinstance(logits, np.ndarray):
                logits_list = [logits[i] for i in range(len(logits))]
            else:
                logits_list = logits
                
            if isinstance(values, np.ndarray):
                values_list = [values[i] for i in range(len(values))]
            else:
                values_list = values
            
            for idx, observation, action, logit, reward, value, done, task_finish_times in zip(itertools.count(), obses, actions_list, logits_list,
                                                                                    rewards, values_list, dones, env_infos):
                # append new samples to running paths
                for single_ob, single_ac, single_logit, single_reward, single_value, single_task_finish_time \
                        in zip(observation, action, logit, reward, value, task_finish_times):
                    running_paths[idx]["observations"]= single_ob
                    running_paths[idx]["actions"] = single_ac
                    running_paths[idx]["logits"] = single_logit
                    running_paths[idx]["rewards"] = single_reward
                    running_paths[idx]["finish_time"] = single_task_finish_time
                    running_paths[idx]["values"] = single_value

                    paths.append(dict(
                        observations=np.squeeze(np.asarray(running_paths[idx]["observations"])),
                        actions=np.squeeze(np.asarray(running_paths[idx]["actions"])),
                        logits = np.squeeze(np.asarray(running_paths[idx]["logits"])),
                        rewards=np.squeeze(np.asarray(running_paths[idx]["rewards"])),
                        finish_time = np.squeeze(np.asarray(running_paths[idx]["finish_time"])),
                        values  = np.squeeze(np.asarray(running_paths[idx]["values"]))
                    ))

                # if running path is done, add it to paths and empty the running path
                new_samples += len(running_paths[idx]["rewards"])
                running_paths[idx] = _get_empty_running_paths_dict()

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
    return dict(observations=[], actions=[], logits=[], rewards=[])