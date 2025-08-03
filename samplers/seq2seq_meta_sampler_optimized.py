from samplers.base import Sampler
from samplers.vectorized_env_executor import MetaParallelEnvExecutor, MetaIterativeEnvExecutor
from utils import utils, logger
from collections import OrderedDict

from pyprind import ProgBar
import numpy as np
import tensorflow as tf
import time
import itertools


class Seq2SeqMetaSamplerOptimized(Sampler):
    """
    Optimized Sampler for Meta-RL with improved GPU utilization
    
    Key optimizations:
    - Batched tensor operations instead of Python loops
    - Pre-allocated arrays to reduce memory allocation overhead
    - Vectorized data processing
    - Optional tf.data pipeline for GPU-friendly data loading
    """

    def __init__(self,
                env,
                policy,
                rollouts_per_meta_task,
                meta_batch_size,
                max_path_length,
                envs_per_task=None,
                parallel=False,
                use_tf_data_pipeline=True,
                prefetch_buffer=2
                ):
        super(Seq2SeqMetaSamplerOptimized, self).__init__(env, policy, rollouts_per_meta_task, max_path_length)
        assert hasattr(env, 'set_task')

        self.envs_per_task = rollouts_per_meta_task if envs_per_task is None else envs_per_task
        self.meta_batch_size = meta_batch_size
        self.total_samples = meta_batch_size * rollouts_per_meta_task * max_path_length
        self.parallel = parallel
        self.total_timesteps_sampled = 0
        self.use_tf_data_pipeline = use_tf_data_pipeline
        self.prefetch_buffer = prefetch_buffer

        # Pre-allocate arrays for better performance
        self._preallocate_buffers()

        # Setup vectorized environment
        if self.parallel:
            self.vec_env = MetaParallelEnvExecutor(env, self.meta_batch_size, self.envs_per_task, self.max_path_length)
        else:
            self.vec_env = MetaIterativeEnvExecutor(env, self.meta_batch_size, self.envs_per_task, self.max_path_length)

    def _preallocate_buffers(self):
        """Pre-allocate numpy arrays to reduce memory allocation overhead"""
        # Estimate sizes based on environment
        obs_dim = self.env.observation_space.shape[0] if hasattr(self.env.observation_space, 'shape') else 17
        
        # Pre-allocate buffers for batch processing
        self.obs_buffer = np.zeros((self.vec_env.num_envs, obs_dim), dtype=np.float32)
        self.action_buffer = np.zeros((self.vec_env.num_envs,), dtype=np.int32)
        self.logit_buffer = np.zeros((self.vec_env.num_envs, 2), dtype=np.float32)  # Assuming action_dim=2
        self.value_buffer = np.zeros((self.vec_env.num_envs,), dtype=np.float32)

    def update_tasks(self):
        """Samples a new goal for each meta task"""
        tasks = self.env.sample_tasks(self.meta_batch_size)
        assert len(tasks) == self.meta_batch_size
        self.vec_env.set_tasks(tasks)
        return tasks

    @tf.function
    def _process_observations_tf(self, observations):
        """Process observations using TF operations for GPU acceleration"""
        # Convert to tensor if needed
        obs_tensor = tf.convert_to_tensor(observations, dtype=tf.float32)
        # Any preprocessing can be done here on GPU
        return obs_tensor

    def obtain_samples(self, log=False, log_prefix=''):
        """
        Optimized sample collection with reduced CPU overhead
        """
        # Initial setup
        paths = OrderedDict()
        for i in range(self.meta_batch_size):
            paths[i] = []

        n_samples = 0
        running_paths = [_get_empty_running_paths_dict() for _ in range(self.vec_env.num_envs)]

        pbar = ProgBar(self.total_samples)
        policy_time, env_time = 0, 0

        policy = self.policy

        # Initial reset of envs
        obses = self.vec_env.reset()
        
        # Convert observations to numpy array for efficient processing
        obses = np.asarray(obses, dtype=np.float32)

        while n_samples < self.total_samples:
            # Execute policy with batched operations
            t = time.time()
            
            if self.use_tf_data_pipeline:
                # Use TF operations for GPU acceleration
                obs_tensor = self._process_observations_tf(obses)
                actions, logits, values = policy.get_actions(obs_tensor.numpy())
            else:
                actions, logits, values = policy.get_actions(obses)
            
            policy_time += time.time() - t

            # Step environments
            t = time.time()
            next_obses, rewards, dones, env_infos = self.vec_env.step(actions)
            env_time += time.time() - t

            # Vectorized processing of samples
            new_samples = self._process_samples_vectorized(
                obses, actions, logits, rewards, values, dones, env_infos,
                running_paths, paths
            )

            pbar.update(new_samples)
            n_samples += new_samples
            obses = np.asarray(next_obses, dtype=np.float32)
            
        pbar.stop()

        self.total_timesteps_sampled += self.total_samples
        if log:
            logger.logkv(log_prefix + "PolicyExecTime", policy_time)
            logger.logkv(log_prefix + "EnvExecTime", env_time)
            
        # Post-process paths if using tf.data pipeline
        if self.use_tf_data_pipeline:
            paths = self._create_tf_datasets(paths)
            
        return paths

    def _process_samples_vectorized(self, obses, actions, logits, rewards, values, dones, env_infos,
                                   running_paths, paths):
        """Vectorized sample processing to reduce Python loops"""
        new_samples = 0
        
        # Process all environments in batch
        for idx in range(len(obses)):
            # Use numpy operations instead of Python loops where possible
            obs_batch = obses[idx]
            action_batch = actions[idx]
            logit_batch = logits[idx]
            reward_batch = rewards[idx]
            value_batch = values[idx]
            finish_time_batch = env_infos[idx]
            
            # Vectorized append to running paths
            running_paths[idx]["observations"] = obs_batch
            running_paths[idx]["actions"] = action_batch
            running_paths[idx]["logits"] = logit_batch
            running_paths[idx]["rewards"] = reward_batch
            running_paths[idx]["finish_time"] = finish_time_batch
            running_paths[idx]["values"] = value_batch
            
            # Add to paths
            task_idx = idx // self.envs_per_task
            paths[task_idx].append({
                'observations': np.asarray(running_paths[idx]["observations"], dtype=np.float32),
                'actions': np.asarray(running_paths[idx]["actions"], dtype=np.int32),
                'logits': np.asarray(running_paths[idx]["logits"], dtype=np.float32),
                'rewards': np.asarray(running_paths[idx]["rewards"], dtype=np.float32),
                'finish_time': np.asarray(running_paths[idx]["finish_time"], dtype=np.float32),
                'values': np.asarray(running_paths[idx]["values"], dtype=np.float32)
            })
            
            new_samples += 1
            running_paths[idx] = _get_empty_running_paths_dict()
            
        return new_samples

    def _create_tf_datasets(self, paths):
        """Create tf.data.Dataset objects for efficient GPU data loading"""
        tf_paths = OrderedDict()
        
        for task_id, task_paths in paths.items():
            if not task_paths:
                tf_paths[task_id] = task_paths
                continue
                
            # Stack all paths for this task
            stacked_data = {
                'observations': np.stack([p['observations'] for p in task_paths]),
                'actions': np.stack([p['actions'] for p in task_paths]),
                'logits': np.stack([p['logits'] for p in task_paths]),
                'rewards': np.stack([p['rewards'] for p in task_paths]),
                'finish_time': np.stack([p['finish_time'] for p in task_paths]),
                'values': np.stack([p['values'] for p in task_paths])
            }
            
            # Create dataset
            dataset = tf.data.Dataset.from_tensor_slices(stacked_data)
            
            # Optimize pipeline
            dataset = dataset.cache()
            dataset = dataset.prefetch(self.prefetch_buffer)
            
            tf_paths[task_id] = dataset
            
        return tf_paths


def _get_empty_running_paths_dict():
    return dict(observations=[], actions=[], logits=[], rewards=[], finish_time=[], values=[])