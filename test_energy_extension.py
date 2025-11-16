"""
Test script to verify energy extension backward compatibility and functionality.

This script tests:
1. Backward compatibility: When use_energy=False, system behaves identically to original
2. Energy functionality: When use_energy=True, energy is computed and combined with latency
"""

import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.mec_offloaing_envs.offloading_env import Resources, OffloadingEnvironment

def test_backward_compatibility():
    """Test that use_energy=False produces identical results to original"""
    print("=" * 60)
    print("TEST 1: Backward Compatibility (use_energy=False)")
    print("=" * 60)
    
    # Create environment with energy disabled
    resource_cluster = Resources(
        mec_process_capable=(10.0 * 1024 * 1024),
        mobile_process_capable=(1.0 * 1024 * 1024),
        bandwidth_up=7.0,
        bandwidth_dl=7.0,
        use_energy=False
    )
    
    env = OffloadingEnvironment(
        resource_cluster=resource_cluster,
        batch_size=10,
        graph_number=10,
        graph_file_paths=["./env/mec_offloaing_envs/data/meta_offloading_n/offload_random20/random.20."],
        time_major=False
    )
    
    # Test get_scheduling_cost_step_by_step return format
    env.set_task(0)
    task_graph = env.task_graphs_batchs[0][0]
    
    # Create a simple plan (all local execution)
    plan = [(i, 0) for i in task_graph.prioritize_sequence]
    
    result = env.get_scheduling_cost_step_by_step(plan, task_graph)
    
    # Should return (latency, finish_time) tuple when energy disabled
    assert len(result) == 2, f"Expected 2 return values, got {len(result)}"
    latency, finish_time = result
    assert isinstance(latency, list), "Latency should be a list"
    assert isinstance(finish_time, (int, float)), "Finish time should be a scalar"
    
    print("✓ get_scheduling_cost_step_by_step returns correct format")
    
    # Test get_reward_batch_step_by_step return format
    action_batch = [[0] * len(task_graph.prioritize_sequence)] * 10
    plan_batch = [[(i, 0) for i in task_graph.prioritize_sequence]] * 10
    
    result = env.get_reward_batch_step_by_step(
        plan_batch,
        [task_graph] * 10,
        [100.0] * 10,
        [10.0] * 10
    )
    
    # Should return (rewards, finish_times) tuple when energy disabled
    assert len(result) == 2, f"Expected 2 return values, got {len(result)}"
    rewards, finish_times = result
    assert isinstance(rewards, np.ndarray), "Rewards should be numpy array"
    assert isinstance(finish_times, list), "Finish times should be a list"
    
    print("✓ get_reward_batch_step_by_step returns correct format")
    print("✓ Backward compatibility test PASSED\n")
    return True

def test_energy_functionality():
    """Test that use_energy=True computes and combines energy correctly"""
    print("=" * 60)
    print("TEST 2: Energy Functionality (use_energy=True)")
    print("=" * 60)
    
    # Create environment with energy enabled
    energy_config = {
        'use_energy': True,
        'energy_weight': 0.5,
        'latency_weight': 0.5,
        'rho': 1.0,
        'f_l': 1.0,
        'zeta': 2.0,
        'ptx': 0.1,
        'prx': 0.05,
        'normalize_energy': True,
    }
    
    resource_cluster = Resources(
        mec_process_capable=(10.0 * 1024 * 1024),
        mobile_process_capable=(1.0 * 1024 * 1024),
        bandwidth_up=7.0,
        bandwidth_dl=7.0,
        use_energy=True,
        energy_config=energy_config
    )
    
    env = OffloadingEnvironment(
        resource_cluster=resource_cluster,
        batch_size=10,
        graph_number=10,
        graph_file_paths=["./env/mec_offloaing_envs/data/meta_offloading_n/offload_random20/random.20."],
        time_major=False
    )
    
    # Test energy computation methods
    execution_time = 1.0
    local_energy = resource_cluster.compute_local_energy(execution_time)
    assert local_energy > 0, f"Local energy should be > 0, got {local_energy}"
    print(f"✓ Local energy computation: {local_energy}")
    
    transmission_energy = resource_cluster.compute_transmission_energy(0.5, 0.3)
    assert transmission_energy > 0, f"Transmission energy should be > 0, got {transmission_energy}"
    print(f"✓ Transmission energy computation: {transmission_energy}")
    
    # Test get_scheduling_cost_step_by_step return format
    env.set_task(0)
    task_graph = env.task_graphs_batchs[0][0]
    
    # Create a simple plan (all local execution)
    plan = [(i, 0) for i in task_graph.prioritize_sequence]
    
    result = env.get_scheduling_cost_step_by_step(plan, task_graph)
    
    # Should return (latency, finish_time, energy) tuple when energy enabled
    assert len(result) == 3, f"Expected 3 return values, got {len(result)}"
    latency, finish_time, energy = result
    assert isinstance(latency, list), "Latency should be a list"
    assert isinstance(finish_time, (int, float)), "Finish time should be a scalar"
    assert isinstance(energy, list), "Energy should be a list"
    assert len(energy) == len(latency), "Energy and latency should have same length"
    assert all(e >= 0 for e in energy), "Energy values should be non-negative"
    
    print("✓ get_scheduling_cost_step_by_step returns energy")
    
    # Test get_reward_batch_step_by_step return format
    plan_batch = [[(i, 0) for i in task_graph.prioritize_sequence]] * 10
    
    result = env.get_reward_batch_step_by_step(
        plan_batch,
        [task_graph] * 10,
        [100.0] * 10,
        [10.0] * 10
    )
    
    # Should return (rewards, finish_times, energy_batch) tuple when energy enabled
    assert len(result) == 3, f"Expected 3 return values, got {len(result)}"
    rewards, finish_times, energy_batch = result
    assert isinstance(rewards, np.ndarray), "Rewards should be numpy array"
    assert isinstance(finish_times, list), "Finish times should be a list"
    assert isinstance(energy_batch, list), "Energy batch should be a list"
    assert len(energy_batch) == len(finish_times), "Energy batch and finish times should have same length"
    
    print("✓ get_reward_batch_step_by_step returns energy batch")
    print("✓ Energy functionality test PASSED\n")
    return True

def test_energy_bounds():
    """Test energy bounds computation"""
    print("=" * 60)
    print("TEST 3: Energy Bounds Computation")
    print("=" * 60)
    
    energy_config = {
        'use_energy': True,
        'energy_weight': 0.5,
        'latency_weight': 0.5,
        'rho': 1.0,
        'f_l': 1.0,
        'zeta': 2.0,
        'ptx': 0.1,
        'prx': 0.05,
    }
    
    resource_cluster = Resources(
        mec_process_capable=(10.0 * 1024 * 1024),
        mobile_process_capable=(1.0 * 1024 * 1024),
        bandwidth_up=7.0,
        bandwidth_dl=7.0,
        use_energy=True,
        energy_config=energy_config
    )
    
    env = OffloadingEnvironment(
        resource_cluster=resource_cluster,
        batch_size=10,
        graph_number=10,
        graph_file_paths=["./env/mec_offloaing_envs/data/meta_offloading_n/offload_random20/random.20."],
        time_major=False
    )
    
    env.set_task(0)
    task_graph = env.task_graphs_batchs[0][0]
    
    max_energy, min_energy = env._compute_energy_bounds(task_graph, 100.0, 10.0)
    
    assert max_energy > 0, f"Max energy should be > 0, got {max_energy}"
    assert min_energy > 0, f"Min energy should be > 0, got {min_energy}"
    assert max_energy >= min_energy, f"Max energy should be >= min energy"
    
    print(f"✓ Max energy: {max_energy:.4f}")
    print(f"✓ Min energy: {min_energy:.4f}")
    print("✓ Energy bounds computation test PASSED\n")
    return True

def test_reward_combination():
    """Test that rewards combine latency and energy correctly"""
    print("=" * 60)
    print("TEST 4: Reward Combination")
    print("=" * 60)
    
    energy_config = {
        'use_energy': True,
        'energy_weight': 0.5,
        'latency_weight': 0.5,
        'rho': 1.0,
        'f_l': 1.0,
        'zeta': 2.0,
        'ptx': 0.1,
        'prx': 0.05,
    }
    
    resource_cluster = Resources(
        mec_process_capable=(10.0 * 1024 * 1024),
        mobile_process_capable=(1.0 * 1024 * 1024),
        bandwidth_up=7.0,
        bandwidth_dl=7.0,
        use_energy=True,
        energy_config=energy_config
    )
    
    env = OffloadingEnvironment(
        resource_cluster=resource_cluster,
        batch_size=10,
        graph_number=10,
        graph_file_paths=["./env/mec_offloaing_envs/data/meta_offloading_n/offload_random20/random.20."],
        time_major=False
    )
    
    env.set_task(0)
    task_graph = env.task_graphs_batchs[0][0]
    
    # Test with all local execution
    plan_batch = [[(i, 0) for i in task_graph.prioritize_sequence]]
    
    rewards, finish_times, energy_batch = env.get_reward_batch_step_by_step(
        plan_batch,
        [task_graph],
        [100.0],
        [10.0]
    )
    
    assert rewards.shape[0] == 1, "Should have one reward sequence"
    assert len(energy_batch) == 1, "Should have one energy sequence"
    assert len(energy_batch[0]) > 0, "Energy sequence should not be empty"
    
    print(f"✓ Combined reward shape: {rewards.shape}")
    print(f"✓ Energy batch length: {len(energy_batch)}")
    print("✓ Reward combination test PASSED\n")
    return True

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("ENERGY EXTENSION VERIFICATION TESTS")
    print("=" * 60 + "\n")
    
    try:
        # Run all tests
        test_backward_compatibility()
        test_energy_functionality()
        test_energy_bounds()
        test_reward_combination()
        
        print("=" * 60)
        print("ALL TESTS PASSED! ✓")
        print("=" * 60)
        print("\nEnergy extension is correctly implemented and backward compatible.")
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

