"""
Simple integration module for adding visualization to meta_evaluator.py
"""

import numpy as np
import json
import os
from pathlib import Path

# Import visualization pipeline
from evaluate_viz.io_utils import write_jsonl, ensure_output_dirs
from evaluate_viz.schema import EpisodeRecord, NodeSpec, Decision
from evaluate_viz.dag_figure import save_dag_figure
from evaluate_viz.gantt_figure import save_gantt_figure
from evaluate_viz.cdf_figure import save_cdf_figure
from evaluate_viz.adapt_figure import save_adaptation_figure
from evaluate_viz.frontier_figure import save_frontier_figure
from evaluate_viz.animate_episode import create_episode_animation


class VisualizationCollector:
    """Simple class to collect evaluation data and generate visualizations."""
    
    def __init__(self, env, output_dir='evaluation_results', animate_episode=10):
        self.env = env
        self.output_dir = output_dir
        self.animate_episode = animate_episode
        self.evaluation_data = []
        self.episode_counter = 0
        
        # Ensure output directories exist
        ensure_output_dirs(output_dir)
    
    def collect_episode_data(self, samples_data, task_id, iteration):
        """Collect data for one episode."""
        try:
            # Get the task graph for this episode
            if hasattr(self.env, 'task_graphs_batchs') and task_id < len(self.env.task_graphs_batchs):
                task_graph = self.env.task_graphs_batchs[task_id]
            else:
                return None
            
            # Extract node information
            nodes = []
            for i, task in enumerate(task_graph.task_list):
                if task is not None:
                    nodes.append(NodeSpec(
                        id=task.id_name,
                        cpu_cycles=task.processing_data_size,
                        up_size=task.transmission_data_size,
                        down_size=task.transmission_data_size
                    ))
            
            # Extract edge information
            edges = []
            if hasattr(task_graph, 'dependencies'):
                for dep in task_graph.dependencies:
                    if len(dep) >= 2:
                        edges.append([dep[0] + 1, dep[1] + 1])
            
            # Extract decisions from actions
            decisions = []
            actions = samples_data.get('actions', [])
            finish_times = samples_data.get('finish_time', [])
            
            # Handle different data structures
            if isinstance(actions, np.ndarray) and len(actions.shape) > 1:
                task_actions = actions[task_id] if task_id < len(actions) else []
            else:
                task_actions = actions if isinstance(actions, list) else []
            
            if isinstance(finish_times, np.ndarray) and len(finish_times.shape) > 1:
                task_finish_times = finish_times[task_id] if task_id < len(finish_times) else []
            else:
                task_finish_times = finish_times if isinstance(finish_times, list) else []
            
            # Create decisions for each node
            for i, node in enumerate(nodes):
                if i < len(task_actions) and i < len(task_finish_times):
                    action = task_actions[i]
                    finish_time = task_finish_times[i]
                    action_str = "LOCAL" if action == 0 else "EDGE"
                    
                    # Calculate durations
                    if action_str == "LOCAL":
                        t_local = self.env.resource_cluster.locally_execution_cost(node.cpu_cycles)
                        decision = Decision(
                            node=node.id,
                            action=action_str,
                            t_local=t_local,
                            finish_times={"ue": finish_time}
                        )
                    else:  # EDGE
                        t_net_up = self.env.resource_cluster.up_transmission_cost(node.up_size)
                        t_edge = self.env.resource_cluster.mec_execution_cost(node.cpu_cycles)
                        t_net_down = self.env.resource_cluster.dl_transmission_cost(node.down_size)
                        
                        uplink_finish = finish_time - t_net_down - t_edge
                        edge_finish = finish_time - t_net_down
                        
                        decision = Decision(
                            node=node.id,
                            action=action_str,
                            t_net_up=t_net_up,
                            t_edge=t_edge,
                            t_net_down=t_net_down,
                            finish_times={
                                "uplink": uplink_finish,
                                "edge": edge_finish,
                                "downlink": finish_time
                            }
                        )
                    
                    decisions.append(decision)
            
            # Calculate total latency
            total_latency = max(task_finish_times) if task_finish_times else 0.0
            
            # Get rates
            rates = {
                "uplink": self.env.resource_cluster.bandwidth_up,
                "downlink": self.env.resource_cluster.bandwidth_dl
            }
            
            # Calculate costs
            energy_ue = sum(node.cpu_cycles for node in nodes 
                          if any(d.node == node.id and d.action == "LOCAL" for d in decisions))
            comm_cost = sum(node.up_size + node.down_size for node in nodes 
                          if any(d.node == node.id and d.action == "EDGE" for d in decisions))
            
            # Calculate baselines
            baselines = self._calculate_baselines(task_id)
            
            # Create episode record
            episode_record = EpisodeRecord(
                episode_id=self.episode_counter,
                method="ours",
                dag={"nodes": [node.model_dump() for node in nodes], "edges": edges},
                decisions=decisions,
                latency_total=total_latency,
                rates=rates,
                adapt_step=iteration,
                energy_ue=energy_ue if energy_ue > 0 else None,
                comm_cost=comm_cost if comm_cost > 0 else None,
                baselines=baselines
            )
            
            self.evaluation_data.append(episode_record)
            self.episode_counter += 1
            
            return episode_record
            
        except Exception as e:
            print(f"Error collecting episode data: {e}")
            return None
    
    def _calculate_baselines(self, task_id):
        """Calculate baseline solutions."""
        baselines = {}
        
        try:
            self.env.set_task(task_id)
            
            # Greedy baseline
            action, finish_time = self.env.greedy_solution()
            baselines["greedy"] = np.mean(finish_time)
            
            # All local baseline
            local_finish_time = self.env.get_all_locally_execute_time()
            baselines["all_local"] = np.mean(local_finish_time)
            
            # All edge baseline
            edge_finish_time = self.env.get_all_mec_execute_time()
            baselines["all_edge"] = np.mean(edge_finish_time)
            
        except Exception as e:
            print(f"Error calculating baselines: {e}")
        
        return baselines
    
    def generate_visualizations(self):
        """Generate all visualizations."""
        if not self.evaluation_data:
            print("No evaluation data available for visualization")
            return
        
        try:
            figures_dir = f"{self.output_dir}/figures"
            videos_dir = f"{self.output_dir}/videos"
            
            print(f"Generating visualizations for {len(self.evaluation_data)} episodes...")
            
            # Generate individual episode figures (limit to first 20)
            max_episodes = min(20, len(self.evaluation_data))
            for i, record in enumerate(self.evaluation_data[:max_episodes]):
                print(f"Generating figures for Episode {record.episode_id} ({i+1}/{max_episodes})")
                
                try:
                    # DAG figure
                    dag_path = f"{figures_dir}/dag_ep{record.episode_id}_{record.method}"
                    save_dag_figure(record, dag_path, formats=['png', 'svg'])
                    
                    # Gantt figure
                    gantt_path = f"{figures_dir}/gantt_ep{record.episode_id}_{record.method}"
                    save_gantt_figure(record, gantt_path, formats=['png', 'svg'])
                    
                except Exception as e:
                    print(f"Error generating figures for episode {record.episode_id}: {e}")
            
            # Generate aggregate figures
            print("Generating aggregate figures...")
            
            try:
                # CDF figure
                cdf_path = f"{figures_dir}/cdf_latency"
                save_cdf_figure(self.evaluation_data, cdf_path, formats=['png', 'svg'])
                
                # Adaptation figure
                adapt_path = f"{figures_dir}/adaptation"
                save_adaptation_figure(self.evaluation_data, adapt_path, formats=['png', 'svg'])
                
                # Frontier figure
                has_energy = any(hasattr(r, 'energy_ue') and getattr(r, 'energy_ue') is not None for r in self.evaluation_data)
                has_comm = any(hasattr(r, 'comm_cost') and getattr(r, 'comm_cost') is not None for r in self.evaluation_data)
                
                if has_energy or has_comm:
                    frontier_path = f"{figures_dir}/frontier"
                    save_frontier_figure(self.evaluation_data, frontier_path, formats=['png', 'svg'])
                
            except Exception as e:
                print(f"Error generating aggregate figures: {e}")
            
            # Generate animation for specified episode
            if self.animate_episode is not None:
                episode_records = [r for r in self.evaluation_data if r.episode_id == self.animate_episode]
                
                if episode_records:
                    print(f"Generating animation for Episode {self.animate_episode}")
                    try:
                        for record in episode_records:
                            output_path = f"{videos_dir}/episode{record.episode_id}_{record.method}_cinematic"
                            create_episode_animation(
                                record, 
                                output_path, 
                                fps=30, 
                                speed=1.0,
                                formats=['mp4', 'gif']
                            )
                    except Exception as e:
                        print(f"Error generating animation: {e}")
                else:
                    print(f"No data found for episode {self.animate_episode}")
            
            # Save evaluation data
            jsonl_path = f"{self.output_dir}/evaluation_data.jsonl"
            write_jsonl(self.evaluation_data, jsonl_path)
            print(f"Evaluation data saved to {jsonl_path}")
            
            print("✅ All visualizations generated successfully!")
            
        except Exception as e:
            print(f"❌ Error generating visualizations: {e}")


def add_visualization_to_trainer(trainer_class, env, output_dir='evaluation_results', animate_episode=10):
    """Add visualization capabilities to an existing trainer class."""
    
    # Create visualization collector
    viz_collector = VisualizationCollector(env, output_dir, animate_episode)
    
    # Store original train method
    original_train = trainer_class.train
    
    def train_with_viz(self):
        """Enhanced train method with visualization."""
        # Call original train method
        result = original_train(self)
        
        # Generate visualizations
        print("\n🎨 Generating visualizations...")
        viz_collector.generate_visualizations()
        
        return result
    
    # Replace train method
    trainer_class.train = train_with_viz
    
    # Add method to collect data during training
    def collect_data(self, samples_data, iteration):
        """Collect data during training."""
        batch_size = len(samples_data['finish_time']) if 'finish_time' in samples_data else 0
        
        for task_id in range(batch_size):
            viz_collector.collect_episode_data(samples_data, task_id, iteration)
    
    trainer_class.collect_data = collect_data
    
    return trainer_class
