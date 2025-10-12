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

    def update_tasks(self):
        """
        Sample a new task for single-policy RL (similar to meta-RL)
        """
        # For single-policy RL, we randomly select one task
        task_id = self.env.sample_tasks(n_tasks=1)[0]
        self.env.set_task(task_id)
        return [task_id]

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
            
            # Policy expects (batch_size, sequence_length, features)
            # We have (num_envs, batch_size, sequence_length, features)
            if len(obses_array.shape) == 4:
                # Shape is (num_envs, batch_size, sequence_length, features)
                # Reshape to (num_envs * batch_size, sequence_length, features)
                original_shape = obses_array.shape
                obses_array = obses_array.reshape(-1, original_shape[2], original_shape[3])
            elif len(obses_array.shape) == 3:
                # Shape is (num_envs, sequence_length, features) - this is correct
                pass
            else:
                raise ValueError(f"Unexpected observation shape: {obses_array.shape}")
            
            actions, logits, values = policy.get_actions(obses_array)
            policy_time += time.time() - t

            # Convert actions to list format for vectorized environment
            if isinstance(actions, np.ndarray):
                # Reshape actions back to match the original environment structure
                if len(obses) > 0 and hasattr(obses[0], 'shape') and len(obses[0].shape) == 3:
                    # Original shape was (num_envs, batch_size, sequence_length, features)
                    # Actions should be reshaped to (num_envs, batch_size, sequence_length)
                    batch_size_per_env = obses[0].shape[0]  # batch_size per environment
                    num_envs = len(obses)
                    actions_reshaped = actions.reshape(num_envs, batch_size_per_env, -1)
                    actions_list = [actions_reshaped[i] for i in range(num_envs)]
                else:
                    # Fallback to simple list conversion
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
            # Note: actions_list was already processed above for environment stepping
            if isinstance(logits, np.ndarray):
                # Reshape logits back to match the original environment structure
                if len(obses) > 0 and hasattr(obses[0], 'shape') and len(obses[0].shape) == 3:
                    batch_size_per_env = obses[0].shape[0]
                    num_envs = len(obses)
                    logits_reshaped = logits.reshape(num_envs, batch_size_per_env, -1)
                    logits_list = [logits_reshaped[i] for i in range(num_envs)]
                else:
                    logits_list = [logits[i] for i in range(len(logits))]
            else:
                logits_list = logits
                
            if isinstance(values, np.ndarray):
                # Reshape values back to match the original environment structure
                if len(obses) > 0 and hasattr(obses[0], 'shape') and len(obses[0].shape) == 3:
                    batch_size_per_env = obses[0].shape[0]
                    num_envs = len(obses)
                    values_reshaped = values.reshape(num_envs, batch_size_per_env, -1)
                    values_list = [values_reshaped[i] for i in range(num_envs)]
                else:
                    values_list = [values[i] for i in range(len(values))]
            else:
                values_list = values
            
            for idx, observation, action, logit, reward, value, done, task_finish_times in zip(itertools.count(), obses, actions_list, logits_list,
                                                                                    rewards, values_list, dones, env_infos):
                # Store the complete trajectory for this environment
                running_paths[idx]["observations"] = observation
                running_paths[idx]["actions"] = action
                running_paths[idx]["logits"] = logit
                running_paths[idx]["rewards"] = reward
                running_paths[idx]["finish_time"] = task_finish_times
                running_paths[idx]["values"] = value

                # If episode is done, add the complete path
                if done:
                    # Ensure data is in the correct format
                    obs = np.asarray(running_paths[idx]["observations"])
                    act = np.asarray(running_paths[idx]["actions"])
                    log = np.asarray(running_paths[idx]["logits"])
                    rew = np.asarray(running_paths[idx]["rewards"])
                    val = np.asarray(running_paths[idx]["values"])
                    fin = np.asarray(running_paths[idx]["finish_time"])
                    
                    paths.append(dict(
                        observations=obs,
                        actions=act,
                        logits=log,
                        rewards=rew,
                        finish_time=fin,
                        values=val
                    ))

                    # Reset for next episode
                    new_samples += len(rew)
                    running_paths[idx] = _get_empty_running_paths_dict()

            pbar.update(new_samples)
            n_samples += new_samples
            obses = next_obses
        pbar.stop()

        self.total_timesteps_sampled += self.total_samples
        if log is True:
            logger.logkv(log_prefix + "PolicyExecTime", policy_time)
            logger.logkv(log_prefix + "EnvExecTime", env_time)
        return paths

def _get_empty_running_paths_dict():
    return dict(observations=[], actions=[], logits=[], rewards=[])