import numpy as np
import os
import sys
from itertools import product
from env.mec_offloaing_envs.offloading_env import Resources, OffloadingEnvironment
from env.mec_offloaing_envs.offloading_task_graph import OffloadingTaskGraph


class ExhaustiveSearchOracle:
    def __init__(self, resource_cluster):
        self.resource_cluster = resource_cluster
        
    def calculate_task_latency(self, task_graph, allocation):
        """
        محاسبه latency برای یک allocation مشخص
        allocation: لیست 0 و 1 که 0 یعنی local و 1 یعنی MEC server
        """
        cloud_available_time = 0.0
        ws_available_time = 0.0
        local_available_time = 0.0
        
        # finish time on cloud for each task
        FT_cloud = [0] * task_graph.task_number
        # finish time on sending channel for each task
        FT_ws = [0] * task_graph.task_number
        # finish time locally for each task
        FT_locally = [0] * task_graph.task_number
        # finish time receiving channel for each task
        FT_wr = [0] * task_graph.task_number
        
        # Process tasks in prioritized order
        for i in task_graph.prioritize_sequence:
            task = task_graph.task_list[i]
            action = allocation[i]
            
            # Local execution (action = 0)
            if action == 0:
                if len(task_graph.pre_task_sets[i]) != 0:
                    start_time = max(local_available_time,
                                   max([max(FT_locally[j], FT_wr[j]) for j in task_graph.pre_task_sets[i]]))
                else:
                    start_time = local_available_time
                
                T_l = self.resource_cluster.locally_execution_cost(task.processing_data_size)
                FT_locally[i] = start_time + T_l
                local_available_time = FT_locally[i]
                
            # MEC execution (action = 1)
            else:
                if len(task_graph.pre_task_sets[i]) != 0:
                    # Upload start time
                    ws_start_time = max(ws_available_time,
                                      max([max(FT_locally[j], FT_ws[j]) for j in task_graph.pre_task_sets[i]]))
                    
                    T_ul = self.resource_cluster.up_transmission_cost(task.processing_data_size)
                    ws_finish_time = ws_start_time + T_ul
                    FT_ws[i] = ws_finish_time
                    ws_available_time = ws_finish_time
                    
                    # Cloud execution
                    cloud_start_time = max(cloud_available_time,
                                          max([max(FT_ws[i], FT_cloud[j]) for j in task_graph.pre_task_sets[i]]))
                    cloud_finish_time = cloud_start_time + self.resource_cluster.mec_execution_cost(task.processing_data_size)
                    FT_cloud[i] = cloud_finish_time
                    cloud_available_time = cloud_finish_time
                    
                    # Download
                    wr_start_time = FT_cloud[i]
                    T_dl = self.resource_cluster.dl_transmission_cost(task.transmission_data_size)
                    wr_finish_time = wr_start_time + T_dl
                    FT_wr[i] = wr_finish_time
                    
                else:
                    # No predecessors
                    ws_start_time = ws_available_time
                    T_ul = self.resource_cluster.up_transmission_cost(task.processing_data_size)
                    ws_finish_time = ws_start_time + T_ul
                    FT_ws[i] = ws_finish_time
                    ws_available_time = ws_finish_time
                    
                    cloud_start_time = max(cloud_available_time, FT_ws[i])
                    cloud_finish_time = cloud_start_time + self.resource_cluster.mec_execution_cost(task.processing_data_size)
                    FT_cloud[i] = cloud_finish_time
                    cloud_available_time = cloud_finish_time
                    
                    wr_start_time = FT_cloud[i]
                    T_dl = self.resource_cluster.dl_transmission_cost(task.transmission_data_size)
                    wr_finish_time = wr_start_time + T_dl
                    FT_wr[i] = wr_finish_time
        
        # Total latency is the maximum finish time
        total_latency = max(max(FT_locally), max(FT_wr))
        return total_latency
    
    def find_optimal_allocation(self, task_graph):
        """
        پیدا کردن بهترین allocation با جستجوی exhaustive
        """
        n = task_graph.task_number
        best_allocation = None
        best_latency = float('inf')
        all_results = []
        
        # Generate all possible allocations (2^n possibilities)
        total_allocations = 2 ** n
        print(f"  Testing {total_allocations:,} possible allocations for {n} tasks...")
        
        # Progress tracking
        progress_interval = max(1, total_allocations // 100)  # Report every 1%
        
        for i in range(total_allocations):
            # Convert integer to binary allocation
            allocation = []
            temp = i
            for _ in range(n):
                allocation.append(temp & 1)
                temp >>= 1
            allocation = allocation[::-1]  # Reverse to get correct order
            
            # Reset resources for each evaluation
            self.resource_cluster.reset()
            
            # Calculate latency for this allocation
            latency = self.calculate_task_latency(task_graph, allocation)
            
            # Store result (only store samples to save memory for large searches)
            if n <= 15 or i % 1000 == 0:  # Store all for small problems, sample for large
                all_results.append({
                    'allocation': allocation.copy(),
                    'latency': latency
                })
            
            # Update best if found better
            if latency < best_latency:
                best_latency = latency
                best_allocation = allocation.copy()
            
            # Progress reporting
            if i > 0 and i % progress_interval == 0:
                progress = (i / total_allocations) * 100
                print(f"    Progress: {progress:.1f}% ({i:,}/{total_allocations:,}) - Current best: {best_latency:.6f}")
        
        # Sort results by latency (if we have them)
        if all_results:
            all_results.sort(key=lambda x: x['latency'])
        else:
            # If no results stored, create minimal set
            all_results = [
                {'allocation': best_allocation, 'latency': best_latency},
                {'allocation': [1]*n, 'latency': best_latency}  # Dummy worst case
            ]
        
        return best_allocation, best_latency, all_results
    
    def analyze_task_graphs(self, task_prefix, graph_count=100):
        """
        تحلیل همه گراف‌های یک task
        """
        results = []
        
        for i in range(graph_count):
            # Check different possible path patterns
            possible_paths = [
                f"env/mec_offloaing_envs/data/{task_prefix}/{task_prefix}_{i}.gv",
                f"env/mec_offloaing_envs/data/{task_prefix}/random.20.{i}.gv",
                f"env/mec_offloaing_envs/data/meta_offloading_20/{task_prefix}/random.20.{i}.gv"
            ]
            
            # Special handling for offload_random20_X tasks
            if task_prefix.startswith("offload_random20_"):
                possible_paths.insert(0, f"env/mec_offloaing_envs/data/meta_offloading_20/{task_prefix}/random.20.{i}.gv")
            
            graph_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    graph_path = path
                    break
            
            if not graph_path:
                print(f"Graph file not found for index {i}")
                continue
                
            print(f"\nAnalyzing graph {i+1}/{graph_count}: {graph_path}")
            
            # Load task graph
            task_graph = OffloadingTaskGraph(graph_path)
            
            # Prioritize tasks (required for execution order)
            task_graph.prioritize_tasks(self.resource_cluster)
            
            # Find optimal allocation
            best_allocation, best_latency, all_allocations = self.find_optimal_allocation(task_graph)
            
            # Count local vs server tasks
            local_count = best_allocation.count(0)
            server_count = best_allocation.count(1)
            
            result = {
                'graph_id': i,
                'task_count': task_graph.task_number,
                'best_latency': best_latency,
                'best_allocation': best_allocation,
                'local_tasks': local_count,
                'server_tasks': server_count,
                'worst_latency': all_allocations[-1]['latency'] if len(all_allocations) > 1 else best_latency,
                'average_latency': np.mean([a['latency'] for a in all_allocations]) if len(all_allocations) > 1 else best_latency
            }
            
            results.append(result)
            
            print(f"  Best latency: {best_latency:.6f}")
            print(f"  Allocation: Local={local_count}, Server={server_count}")
            print(f"  Total allocations tested: {2**task_graph.task_number:,}")
        
        return results
    
    def print_summary(self, results):
        """
        چاپ خلاصه نتایج
        """
        if not results:
            print("No results to summarize")
            return
            
        print("\n" + "="*60)
        print("SUMMARY OF EXHAUSTIVE SEARCH RESULTS")
        print("="*60)
        
        # Overall statistics
        best_latencies = [r['best_latency'] for r in results]
        worst_latencies = [r['worst_latency'] for r in results]
        avg_latencies = [r['average_latency'] for r in results]
        
        print(f"\nTotal graphs analyzed: {len(results)}")
        print(f"\nBest latency statistics:")
        print(f"  Minimum: {min(best_latencies):.6f}")
        print(f"  Maximum: {max(best_latencies):.6f}")
        print(f"  Average: {np.mean(best_latencies):.6f}")
        print(f"  Std Dev: {np.std(best_latencies):.6f}")
        
        print(f"\nWorst latency statistics:")
        print(f"  Minimum: {min(worst_latencies):.6f}")
        print(f"  Maximum: {max(worst_latencies):.6f}")
        print(f"  Average: {np.mean(worst_latencies):.6f}")
        
        print(f"\nAverage allocation pattern:")
        avg_local = np.mean([r['local_tasks'] for r in results])
        avg_server = np.mean([r['server_tasks'] for r in results])
        print(f"  Average local tasks: {avg_local:.2f}")
        print(f"  Average server tasks: {avg_server:.2f}")
        
        # Find best and worst graphs
        best_graph = min(results, key=lambda x: x['best_latency'])
        worst_graph = max(results, key=lambda x: x['best_latency'])
        
        print(f"\nBest performing graph:")
        print(f"  Graph ID: {best_graph['graph_id']}")
        print(f"  Latency: {best_graph['best_latency']:.6f}")
        print(f"  Allocation: Local={best_graph['local_tasks']}, Server={best_graph['server_tasks']}")
        
        print(f"\nWorst performing graph:")
        print(f"  Graph ID: {worst_graph['graph_id']}")
        print(f"  Latency: {worst_graph['best_latency']:.6f}")
        print(f"  Allocation: Local={worst_graph['local_tasks']}, Server={worst_graph['server_tasks']}")
        
        # Final summary
        print(f"\n" + "="*60)
        print(f"FINAL RESULT: Average best latency across all {len(results)} graphs = {np.mean(best_latencies):.6f}")
        print("="*60)


def main():
    # Initialize resources with given parameters
    resource_cluster = Resources(
        mec_process_capable=(10.0 * 1024 * 1024),  # 10 MB/s
        mobile_process_capable=(1.0 * 1024 * 1024),  # 1 MB/s
        bandwidth_up=7.0,  # 7 Mbps
        bandwidth_dl=7.0   # 7 Mbps
    )
    
    # Create oracle searcher
    oracle = ExhaustiveSearchOracle(resource_cluster)
    
    # Get task name from command line or use default
    if len(sys.argv) > 1:
        task_name = sys.argv[1]
    else:
        task_name = "offload_random20_12"
        print(f"No task name provided, using default: {task_name}")
    
    # Number of graphs to analyze
    graph_count = 100
    if len(sys.argv) > 2:
        graph_count = int(sys.argv[2])
    
    print(f"\nStarting exhaustive search for task: {task_name}")
    print(f"Number of graphs to analyze: {graph_count}")
    print(f"\nResource configuration:")
    print(f"  MEC processing: {resource_cluster.mec_process_capble / (1024*1024):.1f} MB/s")
    print(f"  Mobile processing: {resource_cluster.mobile_process_capable / (1024*1024):.1f} MB/s")
    print(f"  Bandwidth (up/down): {resource_cluster.bandwidth_up}/{resource_cluster.bandwidth_dl} Mbps")
    
    # Analyze all task graphs
    results = oracle.analyze_task_graphs(task_name, graph_count)
    
    # Print summary
    oracle.print_summary(results)
    
    # Save results to file
    output_file = f"exhaustive_search_results_{task_name}.txt"
    with open(output_file, 'w') as f:
        f.write(f"Exhaustive Search Results for {task_name}\n")
        f.write("="*60 + "\n\n")
        
        for result in results:
            f.write(f"Graph {result['graph_id']}:\n")
            f.write(f"  Best latency: {result['best_latency']:.6f}\n")
            f.write(f"  Allocation: {result['best_allocation']}\n")
            f.write(f"  Local/Server: {result['local_tasks']}/{result['server_tasks']}\n")
            f.write(f"  Worst latency: {result['worst_latency']:.6f}\n")
            f.write(f"  Average latency: {result['average_latency']:.6f}\n\n")
    
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()