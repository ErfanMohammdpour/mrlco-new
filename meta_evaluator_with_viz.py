import tensorflow as tf
import numpy as np
import time
import json
import os
from pathlib import Path
from utils import logger

# Import visualization pipeline
from evaluate_viz.io_utils import write_jsonl, ensure_output_dirs
from evaluate_viz.schema import EpisodeRecord, NodeSpec, Decision
from evaluate_viz.dag_figure import save_dag_figure
from evaluate_viz.gantt_figure import save_gantt_figure
from evaluate_viz.cdf_figure import save_cdf_figure
from evaluate_viz.adapt_figure import save_adaptation_figure
from evaluate_viz.frontier_figure import save_frontier_figure
from evaluate_viz.animate_episode import create_episode_animation


class TrainerWithViz():
    def __init__(self, algo,
                env,
                sampler,
                sample_processor,
                policy,
                n_itr,
                batch_size=500,
                start_itr=0,
                num_inner_grad_steps=3,
                viz_config=None):
        self.algo = algo
        self.env = env
        self.sampler = sampler
        self.sampler_processor = sample_processor
        self.policy = policy
        self.n_itr = n_itr
        self.start_itr = start_itr
        self.num_inner_grad_steps = num_inner_grad_steps
        self.batch_size = batch_size
        
        # Visualization configuration
        self.viz_config = viz_config or {
            'enable_viz': True,
            'output_dir': 'evaluation_results',
            'animate_episode': 10,  # Animate episode 10 as requested
            'formats': ['png', 'svg', 'mp4', 'gif'],
            'fps': 30,
            'speed': 1.0
        }
        
        # Store evaluation data
        self.evaluation_data = []
        self.episode_counter = 0

    def extract_episode_data(self, samples_data, task_id, iteration):
        """Extract episode data from samples for visualization."""
        try:
            # Get the task graph for this episode
            task_graph = self.env.task_graphs_batchs[task_id]
            
            # Extract node information
            nodes = []
            for i, node in enumerate(task_graph.nodes):
                nodes.append(NodeSpec(
                    id=node.id,
                    cpu_cycles=node.cpu_cycles,
                    up_size=node.up_size,
                    down_size=node.down_size
                ))
            
            # Extract edge information
            edges = []
            for edge in task_graph.edges:
                edges.append([edge[0], edge[1]])
            
            # Extract decisions from actions
            decisions = []
            actions = samples_data['actions'][task_id] if 'actions' in samples_data else []
            finish_times = samples_data['finish_time'][task_id] if 'finish_time' in samples_data else []
            
            for i, (node, action, finish_time) in enumerate(zip(nodes, actions, finish_times)):
                action_str = "LOCAL" if action == 0 else "EDGE"
                
                # Calculate durations based on action and node properties
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
                    
                    # Calculate finish times for each phase
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
            total_latency = max(finish_times) if finish_times else 0.0
            
            # Get rates from environment
            rates = {
                "uplink": self.env.resource_cluster.bandwidth_up,
                "downlink": self.env.resource_cluster.bandwidth_dl
            }
            
            # Calculate energy and communication costs
            energy_ue = sum(node.cpu_cycles for node in nodes if any(d.node == node.id and d.action == "LOCAL" for d in decisions))
            comm_cost = sum(node.up_size + node.down_size for node in nodes if any(d.node == node.id and d.action == "EDGE" for d in decisions))
            
            # Create episode record
            episode_record = EpisodeRecord(
                episode_id=self.episode_counter,
                method="ours",
                dag={"nodes": [node.model_dump() for node in nodes], "edges": edges},
                decisions=decisions,
                latency_total=total_latency,
                rates=rates,
                adapt_step=iteration,
                energy_ue=energy_ue,
                comm_cost=comm_cost,
                baselines={}  # Will be filled later
            )
            
            return episode_record
            
        except Exception as e:
            logger.log(f"Error extracting episode data: {e}")
            return None

    def calculate_baselines(self, task_id):
        """Calculate baseline solutions for comparison."""
        baselines = {}
        
        try:
            # Set the task
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
            logger.log(f"Error calculating baselines: {e}")
        
        return baselines

    def generate_visualizations(self):
        """Generate all visualizations from collected evaluation data."""
        if not self.evaluation_data:
            logger.log("No evaluation data available for visualization")
            return
        
        try:
            # Ensure output directories exist
            output_dirs = ensure_output_dirs(self.viz_config['output_dir'])
            figures_dir = output_dirs['figures']
            videos_dir = output_dirs['videos']
            
            logger.log(f"Generating visualizations in {self.viz_config['output_dir']}")
            
            # Generate individual episode figures
            for record in self.evaluation_data:
                logger.log(f"Generating figures for Episode {record.episode_id}")
                
                # DAG figure
                dag_path = f"{figures_dir}/dag_ep{record.episode_id}_{record.method}"
                save_dag_figure(record, dag_path, formats=self.viz_config['formats'][:2])  # PNG, SVG
                
                # Gantt figure
                gantt_path = f"{figures_dir}/gantt_ep{record.episode_id}_{record.method}"
                save_gantt_figure(record, gantt_path, formats=self.viz_config['formats'][:2])  # PNG, SVG
            
            # Generate aggregate figures
            logger.log("Generating aggregate figures...")
            
            # CDF figure
            cdf_path = f"{figures_dir}/cdf_latency"
            save_cdf_figure(self.evaluation_data, cdf_path, formats=self.viz_config['formats'][:2])
            
            # Adaptation figure
            adapt_path = f"{figures_dir}/adaptation"
            save_adaptation_figure(self.evaluation_data, adapt_path, formats=self.viz_config['formats'][:2])
            
            # Frontier figure (if energy/comm data available)
            has_energy = any(hasattr(r, 'energy_ue') and getattr(r, 'energy_ue') is not None for r in self.evaluation_data)
            has_comm = any(hasattr(r, 'comm_cost') and getattr(r, 'comm_cost') is not None for r in self.evaluation_data)
            
            if has_energy or has_comm:
                frontier_path = f"{figures_dir}/frontier"
                save_frontier_figure(self.evaluation_data, frontier_path, formats=self.viz_config['formats'][:2])
            
            # Generate animation for specified episode
            if self.viz_config['animate_episode'] is not None:
                animate_episode_id = self.viz_config['animate_episode']
                episode_records = [r for r in self.evaluation_data if r.episode_id == animate_episode_id]
                
                if episode_records:
                    logger.log(f"Generating animation for Episode {animate_episode_id}")
                    for record in episode_records:
                        output_path = f"{videos_dir}/episode{record.episode_id}_{record.method}_cinematic"
                        create_episode_animation(
                            record, 
                            output_path, 
                            fps=self.viz_config['fps'], 
                            speed=self.viz_config['speed'],
                            formats=self.viz_config['formats'][2:]  # MP4, GIF
                        )
                else:
                    logger.log(f"No data found for episode {animate_episode_id}")
            
            # Save evaluation data as JSONL
            jsonl_path = f"{self.viz_config['output_dir']}/evaluation_data.jsonl"
            write_jsonl(self.evaluation_data, jsonl_path)
            logger.log(f"Evaluation data saved to {jsonl_path}")
            
            logger.log("✅ All visualizations generated successfully!")
            
        except Exception as e:
            logger.log(f"❌ Error generating visualizations: {e}")

    def train(self):
        """
        Implement the repilte algorithm for ppo reinforcement learning with visualization
        """
        start_time = time.time()
        avg_ret = []
        avg_pg_loss = []
        avg_vf_loss = []
        avg_latencies = []

        for itr in range(self.start_itr, self.n_itr):
            itr_start_time = time.time()
            logger.log("\n ---------------- Iteration %d ----------------" % itr)
            logger.log("Sampling set of tasks/goals for this meta-batch...")

            paths = self.sampler.obtain_samples(log=True, log_prefix='')

            """ ----------------- Processing Samples ---------------------"""
            logger.log("Processing samples...")
            samples_data = self.sampler_processor.process_samples(paths, log='all', log_prefix='')

            """ ------------------- Inner Policy Update --------------------"""
            policy_losses, value_losses = self.algo.UpdatePPOTarget(samples_data, batch_size=self.batch_size)

            print("average policy losses: ", np.mean(policy_losses))
            avg_pg_loss.append(np.mean(policy_losses))

            print("average value losses: ", np.mean(value_losses))
            avg_vf_loss.append(np.mean(value_losses))

            """ ------------------- Logging Stuff --------------------------"""
            ret = np.sum(samples_data['rewards'], axis=-1)
            avg_reward = np.mean(ret)

            latency = samples_data['finish_time']
            avg_latency = np.mean(latency)
            avg_latencies.append(avg_latency)

            logger.logkv('Itr', itr)
            logger.logkv('Average reward, ', avg_reward)
            logger.logkv('Average latency,', avg_latency)
            logger.dumpkvs()
            avg_ret.append(avg_reward)

            """ ------------------- Collect Data for Visualization --------------------"""
            if self.viz_config['enable_viz']:
                # Extract episode data for each task in the batch
                batch_size = len(samples_data['finish_time']) if 'finish_time' in samples_data else 0
                
                for task_id in range(batch_size):
                    try:
                        # Extract episode data
                        episode_record = self.extract_episode_data(samples_data, task_id, itr)
                        
                        if episode_record:
                            # Calculate baselines for this task
                            baselines = self.calculate_baselines(task_id)
                            episode_record.baselines = baselines
                            
                            # Add to evaluation data
                            self.evaluation_data.append(episode_record)
                            self.episode_counter += 1
                            
                    except Exception as e:
                        logger.log(f"Error processing task {task_id} for visualization: {e}")

        # Generate visualizations at the end
        if self.viz_config['enable_viz'] and self.evaluation_data:
            logger.log("\n🎨 Generating visualizations...")
            self.generate_visualizations()

        return avg_ret, avg_pg_loss, avg_vf_loss, avg_latencies


if __name__ == "__main__":
    from env.mec_offloaing_envs.offloading_env import Resources
    from env.mec_offloaing_envs.offloading_env import OffloadingEnvironment
    from policies.meta_seq2seq_policy import Seq2SeqPolicy
    from samplers.seq2seq_sampler import Seq2SeqSampler
    from samplers.seq2seq_sampler_process import Seq2SeSamplerProcessor
    from baselines.vf_baseline import ValueFunctionBaseline
    from meta_algos.ppo_offloading import PPO
    from utils import utils, logger

    logger.configure(dir="./meta_evaluate_ppo_log/task_offloading", format_strs=['stdout', 'log', 'csv'])

    resource_cluster = Resources(mec_process_capable=(10.0 * 1024 * 1024),
                                 mobile_process_capable=(1.0 * 1024 * 1024),
                                 bandwidth_up=7.0, bandwidth_dl=7.0)

    env = OffloadingEnvironment(resource_cluster=resource_cluster,
                                batch_size=100,
                                graph_number=100,
                                graph_file_paths=[
                                    "./env/mec_offloaing_envs/data/offloading_random_1/offloading_random_1.40."
                                    ],
                                time_major=False)

    print("calculate baseline solution======")

    env.set_task(0)
    action, finish_time = env.greedy_solution()
    target_batch, task_finish_time_batch = env.get_reward_batch_step_by_step(action[env.task_id],
                                          env.task_graphs_batchs[env.task_id],
                                          env.max_running_time_batchs[env.task_id],
                                          env.min_running_time_batchs[env.task_id])
    discounted_reward = []
    for reward_path in target_batch:
        discounted_reward.append(utils.discount_cumsum(reward_path, 1.0)[0])

    print("avg greedy solution: ", np.mean(discounted_reward))
    print("avg greedy solution: ", np.mean(task_finish_time_batch))
    print("avg greedy solution: ", np.mean(finish_time))

    print()
    finish_time = env.get_all_mec_execute_time()
    print("avg all remote solution: ", np.mean(finish_time))
    print()
    finish_time = env.get_all_locally_execute_time()
    print("avg all local solution: ", np.mean(finish_time))

    policy = Seq2SeqPolicy(obs_dim=17,
                           encoder_units=128,
                           decoder_units=128,
                           vocab_size=2,
                           name="core_policy")

    sampler = Seq2SeqSampler(env,
                             policy,
                             rollouts_per_meta_task=1,
                             max_path_length=40000,
                             envs_per_task=None,
                             parallel=False)

    baseline = ValueFunctionBaseline()

    sample_processor = Seq2SeSamplerProcessor(baseline=baseline,
                                              discount=0.99,
                                              gae_lambda=0.95,
                                              normalize_adv=True,
                                              positive_adv=False)
    algo = PPO(policy=policy,
               meta_sampler=sampler,
               meta_sampler_process=sample_processor,
               lr=1e-4,
               num_inner_grad_steps=3,
               clip_value=0.2,
               max_grad_norm=None)

    # Visualization configuration
    viz_config = {
        'enable_viz': True,
        'output_dir': 'evaluation_results',
        'animate_episode': 10,  # Animate episode 10 as requested
        'formats': ['png', 'svg', 'mp4', 'gif'],
        'fps': 30,
        'speed': 1.0
    }

    # define the trainer of ppo to evaluate the performance of the trained meta policy for new tasks.
    trainer = TrainerWithViz(algo=algo,
                      env=env,
                      sampler=sampler,
                      sample_processor=sample_processor,
                      policy=policy,
                      n_itr=21,
                      start_itr=0,
                      batch_size=500,
                      num_inner_grad_steps=3,
                      viz_config=viz_config)

    with tf.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        policy.load_variables(load_path="./meta_model_4400.ckpt")
        avg_ret, avg_pg_loss, avg_vf_loss, avg_latencies = trainer.train()
