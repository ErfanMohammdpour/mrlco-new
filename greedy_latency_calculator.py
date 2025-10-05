#!/usr/bin/env python3
"""
Script to calculate greedy latency for a specific offloading task using the exact same process as meta_trainer.py.

Usage:
    python greedy_latency_calculator.py <task_file_path>

Example:
    python greedy_latency_calculator.py "./env/mec_offloaing_envs/data/dags/offloading_random_7/offloading_random_7.20."
"""

import sys
import os
import numpy as np
from env.mec_offloaing_envs.offloading_env import Resources, OffloadingEnvironment


def calculate_greedy_latency_using_env(task_file_path):
    """
    Calculate greedy latency using the exact same process as meta_evaluator.py.
    
    Args:
        task_file_path (str): Path pattern to the .gv task files
        
    Returns:
        float: Average greedy latency
    """
    
    # Create resource cluster exactly like in meta_evaluator.py
    resource_cluster = Resources(
        mec_process_capable=(10.0 * 1024 * 1024),
        mobile_process_capable=(1.0 * 1024 * 1024),
        bandwidth_up=7.0, 
        bandwidth_dl=7.0
    )
    
    # Create environment exactly like in meta_evaluator.py
    env = OffloadingEnvironment(
        resource_cluster=resource_cluster,
        batch_size=100,
        graph_number=100,
        graph_file_paths=[task_file_path],
        time_major=False
    )
    
    print("calculate baseline solution======")
    
    # Set task and get greedy solution exactly like meta_evaluator.py
    env.set_task(0)
    action, finish_time = env.greedy_solution()
    
    # Get reward batch step by step like meta_evaluator.py
    target_batch, task_finish_time_batch = env.get_reward_batch_step_by_step(
        action[env.task_id],
        env.task_graphs_batchs[env.task_id],
        env.max_running_time_batchs[env.task_id],
        env.min_running_time_batchs[env.task_id]
    )
    
    # Calculate average like meta_evaluator.py (line 114)
    avg_greedy_latency = np.mean(task_finish_time_batch)
    
    print("avg greedy solution (task_finish_time_batch): ", avg_greedy_latency)
    print("avg greedy solution (finish_time): ", np.mean(finish_time))
    
    return avg_greedy_latency


def main():
    if len(sys.argv) != 2:
        print("Usage: python greedy_latency_calculator.py <task_file_path>")
        print("Example:")
        print("  python greedy_latency_calculator.py './env/mec_offloaing_envs/data/dags/offloading_random_7/offloading_random_7.20.'")
        sys.exit(1)
    
    task_file_path = sys.argv[1]
    
    try:
        # Use the exact same process as meta_trainer.py
        avg_greedy_latency = calculate_greedy_latency_using_env(task_file_path)
        print(f"Average Greedy Latency: {avg_greedy_latency:.6f} seconds")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
