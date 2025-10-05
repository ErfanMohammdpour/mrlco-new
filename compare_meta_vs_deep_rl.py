"""
Comparison between Meta-RL and Deep RL approaches for task offloading
"""

import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt
from utils import logger
from env.mec_offloaing_envs.offloading_env import Resources, OffloadingEnvironment
from deep_rl_offloading import DeepRLOffloadingAgent
from policies.meta_seq2seq_policy import MetaSeq2SeqPolicy
from meta_algos.MRLCO import MRLCO
from samplers.seq2seq_meta_sampler import Seq2SeqMetaSampler
from samplers.seq2seq_meta_sampler_process import Seq2SeqMetaSamplerProcessor
from baselines.vf_baseline import ValueFunctionBaseline


class MetaRLvsDeepRLComparison:
    """Compare Meta-RL and Deep RL approaches"""
    
    def __init__(self, env, test_tasks=20):
        self.env = env
        self.test_tasks = test_tasks
        self.results = {
            'meta_rl': {'rewards': [], 'latencies': [], 'convergence': []},
            'deep_rl': {'rewards': [], 'latencies': [], 'convergence': []},
            'greedy': {'rewards': [], 'latencies': []}
        }
    
    def evaluate_greedy_baseline(self):
        """Evaluate greedy baseline performance"""
        logger.log("Evaluating Greedy Baseline...")
        
        greedy_rewards = []
        greedy_latencies = []
        
        for _ in range(self.test_tasks):
            task_id = self.env.sample_tasks(1)[0]
            self.env.set_task(task_id)
            
            # Get greedy solution
            action, finish_time = self.env.greedy_solution_for_current_task()
            
            # Evaluate greedy solution
            obs = self.env.reset()
            _, rewards, _, info = self.env.step(action)
            
            greedy_rewards.append(np.sum(rewards))
            greedy_latencies.append(finish_time)
        
        self.results['greedy']['rewards'] = greedy_rewards
        self.results['greedy']['latencies'] = greedy_latencies
        
        logger.log(f"Greedy - Avg Reward: {np.mean(greedy_rewards):.4f}, Avg Latency: {np.mean(greedy_latencies):.4f}")
    
    def evaluate_meta_rl(self, meta_policy_path=None):
        """Evaluate Meta-RL approach"""
        logger.log("Evaluating Meta-RL...")
        
        # Load or create Meta-RL policy
        if meta_policy_path and os.path.exists(meta_policy_path):
            # Load pre-trained meta policy
            meta_policy = MetaSeq2SeqPolicy(
                meta_batch_size=1,
                obs_dim=17,
                encoder_units=128,
                decoder_units=128,
                vocab_size=2
            )
            
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                meta_policy.core_policy.load_variables(meta_policy_path)
                
                meta_rewards, meta_latencies = self._evaluate_meta_rl_policy(meta_policy, sess)
        else:
            # Train Meta-RL from scratch (simplified)
            meta_rewards, meta_latencies = self._train_and_evaluate_meta_rl()
        
        self.results['meta_rl']['rewards'] = meta_rewards
        self.results['meta_rl']['latencies'] = meta_latencies
        
        logger.log(f"Meta-RL - Avg Reward: {np.mean(meta_rewards):.4f}, Avg Latency: {np.mean(meta_latencies):.4f}")
    
    def evaluate_deep_rl(self, deep_rl_path=None):
        """Evaluate Deep RL approach"""
        logger.log("Evaluating Deep-RL...")
        
        # Load or create Deep RL agent
        if deep_rl_path and os.path.exists(deep_rl_path):
            # Load pre-trained Deep RL agent
            agent = DeepRLOffloadingAgent(
                obs_dim=17,
                action_dim=2,
                encoder_units=128,
                decoder_units=128
            )
            
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                agent.load_model(deep_rl_path)
                
                deep_rl_rewards, deep_rl_latencies = self._evaluate_deep_rl_agent(agent, sess)
        else:
            # Train Deep RL from scratch (simplified)
            deep_rl_rewards, deep_rl_latencies = self._train_and_evaluate_deep_rl()
        
        self.results['deep_rl']['rewards'] = deep_rl_rewards
        self.results['deep_rl']['latencies'] = deep_rl_latencies
        
        logger.log(f"Deep-RL - Avg Reward: {np.mean(deep_rl_rewards):.4f}, Avg Latency: {np.mean(deep_rl_latencies):.4f}")
    
    def _evaluate_meta_rl_policy(self, meta_policy, sess):
        """Evaluate Meta-RL policy on test tasks"""
        rewards = []
        latencies = []
        
        for _ in range(self.test_tasks):
            task_id = self.env.sample_tasks(1)[0]
            self.env.set_task(task_id)
            
            # Get action from meta policy
            obs = self.env.reset()
            actions, _, _ = meta_policy.get_actions([obs])
            
            # Execute action
            _, reward, _, info = self.env.step(actions[0])
            
            rewards.append(np.sum(reward))
            latencies.append(info[0] if isinstance(info, list) else info)
        
        return rewards, latencies
    
    def _evaluate_deep_rl_agent(self, agent, sess):
        """Evaluate Deep RL agent on test tasks"""
        rewards = []
        latencies = []
        
        for _ in range(self.test_tasks):
            task_id = self.env.sample_tasks(1)[0]
            self.env.set_task(task_id)
            
            obs = self.env.reset()
            sequence_length = np.array([obs.shape[1]] * obs.shape[0], dtype=np.int32)
            
            episode_reward = 0
            episode_latency = 0
            
            for step in range(50):  # Max episode length
                actions = agent.get_action(obs, sequence_length, training=False)
                next_obs, reward, done, info = self.env.step(actions)
                
                episode_reward += np.sum(reward)
                episode_latency = info[0] if isinstance(info, list) else info
                
                obs = next_obs
                
                if done:
                    break
            
            rewards.append(episode_reward)
            latencies.append(episode_latency)
        
        return rewards, latencies
    
    def _train_and_evaluate_meta_rl(self):
        """Simplified Meta-RL training and evaluation"""
        # This is a placeholder - in practice, you'd load a pre-trained model
        logger.log("Meta-RL training not implemented in this comparison")
        return [0] * self.test_tasks, [0] * self.test_tasks
    
    def _train_and_evaluate_deep_rl(self):
        """Simplified Deep RL training and evaluation"""
        # This is a placeholder - in practice, you'd load a pre-trained model
        logger.log("Deep-RL training not implemented in this comparison")
        return [0] * self.test_tasks, [0] * self.test_tasks
    
    def generate_comparison_report(self):
        """Generate comprehensive comparison report"""
        
        logger.log("Generating comparison report...")
        
        # Calculate statistics
        stats = {}
        for method in ['greedy', 'meta_rl', 'deep_rl']:
            if self.results[method]['rewards']:
                stats[method] = {
                    'reward_mean': np.mean(self.results[method]['rewards']),
                    'reward_std': np.std(self.results[method]['rewards']),
                    'latency_mean': np.mean(self.results[method]['latencies']),
                    'latency_std': np.std(self.results[method]['latencies'])
                }
        
        # Print comparison table
        logger.log("\n" + "="*60)
        logger.log("COMPARISON RESULTS")
        logger.log("="*60)
        logger.log(f"{'Method':<15} {'Reward Mean':<12} {'Reward Std':<12} {'Latency Mean':<12} {'Latency Std':<12}")
        logger.log("-"*60)
        
        for method, stat in stats.items():
            logger.log(f"{method.upper():<15} {stat['reward_mean']:<12.4f} {stat['reward_std']:<12.4f} "
                      f"{stat['latency_mean']:<12.4f} {stat['latency_std']:<12.4f}")
        
        logger.log("="*60)
        
        # Calculate improvements
        if 'greedy' in stats and 'deep_rl' in stats:
            reward_improvement = ((stats['deep_rl']['reward_mean'] - stats['greedy']['reward_mean']) / 
                                abs(stats['greedy']['reward_mean']) * 100)
            latency_improvement = ((stats['greedy']['latency_mean'] - stats['deep_rl']['latency_mean']) / 
                                 stats['greedy']['latency_mean'] * 100)
            
            logger.log(f"\nDeep-RL vs Greedy:")
            logger.log(f"  Reward Improvement: {reward_improvement:.2f}%")
            logger.log(f"  Latency Improvement: {latency_improvement:.2f}%")
        
        if 'meta_rl' in stats and 'deep_rl' in stats:
            reward_improvement = ((stats['deep_rl']['reward_mean'] - stats['meta_rl']['reward_mean']) / 
                                abs(stats['meta_rl']['reward_mean']) * 100)
            latency_improvement = ((stats['meta_rl']['latency_mean'] - stats['deep_rl']['latency_mean']) / 
                                 stats['meta_rl']['latency_mean'] * 100)
            
            logger.log(f"\nDeep-RL vs Meta-RL:")
            logger.log(f"  Reward Improvement: {reward_improvement:.2f}%")
            logger.log(f"  Latency Improvement: {latency_improvement:.2f}%")
        
        # Generate plots
        self._generate_comparison_plots()
        
        return stats
    
    def _generate_comparison_plots(self):
        """Generate comparison plots"""
        
        try:
            import matplotlib.pyplot as plt
            
            # Create figure with subplots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # Plot 1: Reward comparison
            methods = []
            rewards = []
            errors = []
            
            for method in ['greedy', 'meta_rl', 'deep_rl']:
                if self.results[method]['rewards']:
                    methods.append(method.upper())
                    rewards.append(np.mean(self.results[method]['rewards']))
                    errors.append(np.std(self.results[method]['rewards']))
            
            ax1.bar(methods, rewards, yerr=errors, capsize=5, alpha=0.7)
            ax1.set_title('Average Reward Comparison')
            ax1.set_ylabel('Reward')
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Latency comparison
            methods = []
            latencies = []
            errors = []
            
            for method in ['greedy', 'meta_rl', 'deep_rl']:
                if self.results[method]['latencies']:
                    methods.append(method.upper())
                    latencies.append(np.mean(self.results[method]['latencies']))
                    errors.append(np.std(self.results[method]['latencies']))
            
            ax2.bar(methods, latencies, yerr=errors, capsize=5, alpha=0.7, color='orange')
            ax2.set_title('Average Latency Comparison')
            ax2.set_ylabel('Latency')
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Reward distribution
            for method in ['greedy', 'meta_rl', 'deep_rl']:
                if self.results[method]['rewards']:
                    ax3.hist(self.results[method]['rewards'], alpha=0.6, label=method.upper(), bins=20)
            
            ax3.set_title('Reward Distribution')
            ax3.set_xlabel('Reward')
            ax3.set_ylabel('Frequency')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Plot 4: Latency distribution
            for method in ['greedy', 'meta_rl', 'deep_rl']:
                if self.results[method]['latencies']:
                    ax4.hist(self.results[method]['latencies'], alpha=0.6, label=method.upper(), bins=20)
            
            ax4.set_title('Latency Distribution')
            ax4.set_xlabel('Latency')
            ax4.set_ylabel('Frequency')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('./comparison_results.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            logger.log("Comparison plots saved to comparison_results.png")
            
        except ImportError:
            logger.log("Matplotlib not available, skipping plots")
        except Exception as e:
            logger.log(f"Error generating plots: {str(e)}")


def main():
    """Main comparison function"""
    
    # Set up logging
    tf.logging.set_verbosity(tf.logging.ERROR)
    logger.configure(dir="./comparison_log/", format_strs=['stdout', 'log', 'csv'])
    
    # Environment setup
    resource_cluster = Resources(
        mec_process_capable=(10.0 * 1024 * 1024),
        mobile_process_capable=(1.0 * 1024 * 1024),
        bandwidth_up=7.0, 
        bandwidth_dl=7.0
    )
    
    env = OffloadingEnvironment(
        resource_cluster=resource_cluster,
        batch_size=1,
        graph_number=100,
        graph_file_paths=[
            "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_1/random.20.",
            "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_2/random.20.",
            "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_3/random.20.",
        ],
        time_major=False
    )
    
    # Create comparison object
    comparison = MetaRLvsDeepRLComparison(env, test_tasks=20)
    
    # Run evaluations
    comparison.evaluate_greedy_baseline()
    comparison.evaluate_meta_rl()  # You can provide a path to pre-trained model
    comparison.evaluate_deep_rl()  # You can provide a path to pre-trained model
    
    # Generate report
    stats = comparison.generate_comparison_report()
    
    return stats


if __name__ == "__main__":
    main()
