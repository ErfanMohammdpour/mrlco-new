from env.mec_offloaing_envs.offloading_env import OffloadingEnvironment, Resources
import numpy as np

class SinglePolicyOffloadingEnvironment(OffloadingEnvironment):
    """
    Single-policy RL wrapper for the offloading environment.
    Removes meta-learning interface and provides standard RL interface.
    """
    
    def __init__(self, resource_cluster, batch_size, graph_number, graph_file_paths, time_major):
        super().__init__(resource_cluster, batch_size, graph_number, graph_file_paths, time_major)
        
        # For single-policy RL, we'll randomly sample tasks during training
        self.current_task_id = 0
        
    def reset(self):
        """
        Resets the environment and returns an initial observation.
        For single-policy RL, we randomly select a task.
        
        Returns:
            observation (object): the initial observation.
        """
        # Randomly select a task for single-policy RL
        self.current_task_id = np.random.randint(0, self.total_task)
        self.set_task(self.current_task_id)
        
        # Reset the resource environment
        self.resource_cluster.reset()
        
        return np.array(self.encoder_batchs[self.current_task_id])
    
    def step(self, action):
        """
        Run one timestep of the environment's dynamics.
        
        Args:
            action (object): an action provided by the agent
            
        Returns:
            observation (object): agent's observation of the current environment
            reward (float): amount of reward returned after previous action
            done (bool): whether the episode has ended
            info (dict): contains auxiliary diagnostic information
        """
        plan_batch = []
        task_graph_batch = self.task_graphs_batchs[self.current_task_id]
        max_running_time_batch = self.max_running_time_batchs[self.current_task_id]
        min_running_time_batch = self.min_running_time_batchs[self.current_task_id]

        for action_sequence, task_graph in zip(action, task_graph_batch):
            plan_sequence = []
            for action, task_id in zip(action_sequence, task_graph.prioritize_sequence):
                plan_sequence.append((task_id, action))
            plan_batch.append(plan_sequence)

        reward_batch, task_finish_time = self.get_reward_batch_step_by_step(
            plan_batch, task_graph_batch, max_running_time_batch, min_running_time_batch)

        done = True
        observation = np.array(self.encoder_batchs[self.current_task_id])
        info = task_finish_time

        return observation, reward_batch, done, info
    
    def sample_tasks(self, n_tasks=1):
        """
        For single-policy RL, we just return a single random task.
        This maintains compatibility with the existing interface.
        
        Args:
            n_tasks (int): number of tasks (ignored for single-policy)
            
        Returns:
            tasks (list): a list containing one random task
        """
        return [np.random.randint(0, self.total_task)]
    
    def set_task(self, task):
        """
        Sets the specified task to the current environment
        
        Args:
            task: task ID of the environment
        """
        self.current_task_id = task
        super().set_task(task)
    
    def get_task(self):
        """
        Gets the current task ID
        
        Returns:
            task: current task ID
        """
        return self.current_task_id
