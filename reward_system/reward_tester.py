"""
Reward Tester - Single Testing File (TensorFlow 1.x Compatible)

This is the ONLY testing file. It handles all reward testing and comparison
during training when test_rewards=True flag is set.

Adapted for TensorFlow 1.x Session-based training.
"""

import numpy as np
import copy
import time
import os
import matplotlib.pyplot as plt
import tempfile
import joblib
from datetime import datetime
from reward_system.reward_registry import RewardRegistry


class RewardTester:
    """
    Integrated reward testing framework for TF 1.x.
    
    Tests all reward formulas during training and compares results.
    This is the SINGLE testing file that handles everything.
    """
    
    def __init__(self, 
                 base_env,
                 base_algo,
                 base_sampler,
                 base_sample_processor,
                 base_policy,
                 greedy_finish_time,
                 n_itr_per_reward=50,
                 reward_names=None,
                 inner_batch_size=500,
                 save_interval=100,
                 checkpoint_dir="./checkpoints",
                 session=None):
        """
        Initialize reward tester.
        
        Args:
            base_env: Base environment instance (will be cloned for each reward)
            base_algo: Base algorithm instance
            base_sampler: Base sampler instance
            base_sample_processor: Base sample processor instance
            base_policy: Base policy instance (will be cloned for each reward)
            greedy_finish_time: Greedy baseline completion times
            n_itr_per_reward: Number of training iterations per reward test
            reward_names: List of reward names to test (None = all registered)
            inner_batch_size: Batch size for inner loop updates
            save_interval: Interval for saving checkpoints
            checkpoint_dir: Directory for checkpoints
            session: TensorFlow 1.x Session (optional, will use default if None)
        """
        import tensorflow as tf
        
        self.base_env = base_env
        self.base_algo = base_algo
        self.base_sampler = base_sampler
        self.base_sample_processor = base_sample_processor
        self.base_policy = base_policy
        self.greedy_finish_time = greedy_finish_time
        self.n_itr_per_reward = n_itr_per_reward
        self.inner_batch_size = inner_batch_size
        self.save_interval = save_interval
        self.checkpoint_dir = checkpoint_dir
        self.session = session
        
        # Get reward names to test
        if reward_names is None:
            self.reward_names = RewardRegistry.list_all()
        else:
            # Validate reward names
            available_rewards = RewardRegistry.list_all()
            self.reward_names = [r for r in reward_names if r in available_rewards]
            if len(self.reward_names) != len(reward_names):
                missing = set(reward_names) - set(self.reward_names)
                print(f"Warning: Some rewards not found: {missing}")
        
        if not self.reward_names:
            raise ValueError("No valid reward names provided")
        
        # Results storage
        self.results = {}
        
        # Create results directory
        self.results_dir = os.path.join(checkpoint_dir, "reward_comparison")
        os.makedirs(self.results_dir, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        # Temporary directory for policy cloning
        self.temp_dir = tempfile.mkdtemp(prefix="reward_tester_")
    
    def run_comparison_training(self):
        """
        Run training with all rewards and compare results.
        
        This is the main method called when test_rewards=True.
        Must be called within a TF 1.x Session context.
        
        Returns:
            Dictionary with comparison results
        """
        import tensorflow as tf
        
        print("\n" + "="*80)
        print("REWARD TESTING MODE - Testing All Reward Functions")
        print("="*80)
        print(f"Rewards to test: {self.reward_names}")
        print(f"Iterations per reward: {self.n_itr_per_reward}")
        print("="*80 + "\n")
        
        # Get or use default session
        sess = self.session or tf.compat.v1.get_default_session()
        if sess is None:
            raise RuntimeError("No TensorFlow session available. Please run within a session context.")
        
        # Test each reward
        for idx, reward_name in enumerate(self.reward_names, 1):
            print(f"\n{'='*80}")
            print(f"Testing Reward {idx}/{len(self.reward_names)}: {reward_name.upper()}")
            print(f"{'='*80}")
            
            try:
                # Create environment with this reward
                env = self._create_env_with_reward(reward_name)
                
                # Clone policy for this reward (TF 1.x compatible)
                policy = self._clone_policy_tf1(sess, reward_name)
                
                # Create sampler with new environment
                sampler = self._create_sampler_with_env(env, policy)
                
                # Create trainer
                from meta_trainer import Trainer
                trainer = Trainer(
                    algo=self.base_algo,
                    env=env,
                    sampler=sampler,
                    sample_processor=self.base_sample_processor,
                    policy=policy,
                    n_itr=self.n_itr_per_reward,
                    greedy_finish_time=self.greedy_finish_time,
                    start_itr=0,
                    inner_batch_size=self.inner_batch_size,
                    save_interval=self.save_interval
                )
                
                # Run training (within session context)
                start_time = time.time()
                training_results = trainer.train()
                elapsed_time = time.time() - start_time
                
                # Extract metrics
                metrics = self._extract_metrics(training_results, elapsed_time)
                
                # Store results
                self.results[reward_name] = {
                    'metrics': metrics,
                    'training_results': training_results,
                    'elapsed_time': elapsed_time
                }
                
                print(f"\n✓ Completed {reward_name} in {elapsed_time:.2f}s")
                print(f"  Final latency: {metrics['final_latency']:.4f}")
                print(f"  Best latency: {metrics['best_latency']:.4f}")
                print(f"  Convergence: {metrics['convergence_rate']:.4f}")
                
            except Exception as e:
                print(f"\n✗ Error testing {reward_name}: {str(e)}")
                import traceback
                traceback.print_exc()
                self.results[reward_name] = {
                    'error': str(e),
                    'metrics': None
                }
        
        # Compare results
        comparison = self._compare_results()
        
        # Generate report
        self._generate_report(comparison)
        
        # Cleanup temp files
        self._cleanup()
        
        return comparison
    
    def _create_env_with_reward(self, reward_name):
        """
        Create environment configured with specific reward.
        
        Args:
            reward_name: Name of reward function
        
        Returns:
            Environment instance with reward configured
        """
        # Get reward function
        reward_function = RewardRegistry.get(reward_name)
        if reward_function is None:
            raise ValueError(f"Reward '{reward_name}' not found in registry")
        
        # Get default parameters
        reward_params = reward_function.get_params()
        
        # Clone environment
        env = copy.deepcopy(self.base_env)
        
        # Set reward function
        env.reward_type = reward_name
        env.reward_params = reward_params
        env.reward_function = reward_function
        
        # Modify score_func method
        def score_func_with_reward(cost, max_time, min_time, **kwargs):
            params = {**reward_params, **kwargs}
            return reward_function.compute(cost, max_time, min_time, **params)
        
        env.score_func = score_func_with_reward
        
        return env
    
    def _clone_policy_tf1(self, sess, reward_name):
        """
        Clone policy for TF 1.x (using save_variables/load_variables).
        
        Args:
            sess: TensorFlow session
            reward_name: Name of reward (for unique temp file)
        
        Returns:
            Cloned policy instance
        """
        import tensorflow as tf
        
        # Create a new policy with same structure
        from policies.meta_seq2seq_policy import MetaSeq2SeqPolicy
        
        # Get policy parameters
        meta_batch_size = self.base_policy.meta_batch_size
        obs_dim = self.base_policy.obs_dim
        encoder_units = self.base_policy.core_policy.network.encoder_hidden_unit
        decoder_units = self.base_policy.core_policy.network.decoder_hidden_unit
        vocab_size = self.base_policy.action_dim
        
        # Create new policy
        cloned_policy = MetaSeq2SeqPolicy(
            meta_batch_size=meta_batch_size,
            obs_dim=obs_dim,
            encoder_units=encoder_units,
            decoder_units=decoder_units,
            vocab_size=vocab_size
        )
        
        # Save base policy weights to temp file
        temp_save_path = os.path.join(self.temp_dir, f"base_policy_{reward_name}.pkl")
        self.base_policy.core_policy.save_variables(temp_save_path, sess=sess)
        
        # Load weights into cloned policy
        cloned_policy.core_policy.load_variables(temp_save_path, sess=sess)
        
        # Copy core to meta policies
        sess.run(cloned_policy.assign_old_eq_new_tasks)
        
        return cloned_policy
    
    def _create_sampler_with_env(self, env, policy):
        """
        Create sampler with new environment and policy.
        
        Args:
            env: Environment instance
            policy: Policy instance
        
        Returns:
            Sampler instance
        """
        # Clone sampler and update environment/policy
        sampler = copy.deepcopy(self.base_sampler)
        
        # Update environment in sampler
        if hasattr(sampler, 'vec_env') and hasattr(sampler.vec_env, 'envs'):
            sampler.vec_env.envs = [env] * len(sampler.vec_env.envs)
        elif hasattr(sampler, 'env'):
            sampler.env = env
        
        # Update policy
        sampler.policy = policy
        
        return sampler
    
    def _extract_metrics(self, training_results, elapsed_time):
        """
        Extract metrics from training results.
        
        Args:
            training_results: Results from trainer.train() (tuple: avg_ret, avg_loss, avg_latencies)
            elapsed_time: Time taken for training
        
        Returns:
            Dictionary of metrics
        """
        # Extract data from training results
        # Format: (avg_returns, avg_losses, avg_latencies)
        if isinstance(training_results, tuple) and len(training_results) >= 3:
            avg_ret, avg_loss, avg_latencies = training_results[0], training_results[1], training_results[2]
        else:
            # Fallback if format is different
            avg_ret = []
            avg_loss = []
            avg_latencies = []
        
        # Convert to numpy arrays
        avg_ret = np.array(avg_ret) if len(avg_ret) > 0 else np.array([])
        avg_loss = np.array(avg_loss) if len(avg_loss) > 0 else np.array([])
        avg_latencies = np.array(avg_latencies) if len(avg_latencies) > 0 else np.array([])
        
        # Create iterations array
        iterations = np.arange(len(avg_latencies)) if len(avg_latencies) > 0 else np.array([])
        
        # Compute metrics
        metrics = {
            # Summary metrics
            'final_latency': float(np.mean(avg_latencies[-10:])) if len(avg_latencies) > 0 else float('inf'),
            'best_latency': float(np.min(avg_latencies)) if len(avg_latencies) > 0 else float('inf'),
            'average_latency': float(np.mean(avg_latencies)) if len(avg_latencies) > 0 else float('inf'),
            'final_loss': float(np.mean(avg_loss[-10:])) if len(avg_loss) > 0 else float('inf'),
            'average_loss': float(np.mean(avg_loss)) if len(avg_loss) > 0 else float('inf'),
            'convergence_rate': self._compute_convergence_rate(avg_latencies),
            'stability': self._compute_stability(avg_latencies),
            'elapsed_time': elapsed_time,
            # Full curves for plotting
            'iteration': iterations.tolist(),
            'avg_reward': avg_ret.tolist(),
            'avg_policy_loss': avg_loss.tolist(),
            'avg_value_loss': [],
            'avg_latency': avg_latencies.tolist(),
            'greedy_latency': [],
            'elapsed_time_curve': np.linspace(0, elapsed_time, len(iterations)).tolist() if len(iterations) > 0 else []
        }
        
        return metrics
    
    def _compute_convergence_rate(self, latencies):
        """Compute convergence rate metric."""
        if len(latencies) < 10:
            return float('inf')
        
        first_quarter = np.mean(latencies[:len(latencies)//4])
        last_quarter = np.mean(latencies[-len(latencies)//4:])
        
        if first_quarter == 0:
            return float('inf')
        
        improvement = (first_quarter - last_quarter) / first_quarter
        return float(improvement)
    
    def _compute_stability(self, latencies):
        """Compute stability metric (coefficient of variation)."""
        if len(latencies) < 10:
            return float('inf')
        
        recent_latencies = latencies[len(latencies)//2:]
        
        if np.mean(recent_latencies) == 0:
            return float('inf')
        
        cv = np.std(recent_latencies) / np.mean(recent_latencies)
        return float(cv)
    
    def _compare_results(self):
        """Compare results across all rewards."""
        comparison = {
            'best_latency': {},
            'fastest_convergence': {},
            'most_stable': {},
            'lowest_loss': {},
            'overall_ranking': [],
            'detailed_comparison': {}
        }
        
        # Filter out failed tests
        valid_results = {k: v for k, v in self.results.items() 
                        if v.get('metrics') is not None and 'error' not in v}
        
        if not valid_results:
            print("Warning: No valid results to compare")
            return comparison
        
        # Find best in each category
        if valid_results:
            best_latency_reward = min(valid_results.items(), 
                                     key=lambda x: x[1]['metrics']['best_latency'])
            comparison['best_latency'] = {
                'reward': best_latency_reward[0],
                'value': best_latency_reward[1]['metrics']['best_latency']
            }
            
            fastest_convergence_reward = max(valid_results.items(),
                                            key=lambda x: x[1]['metrics']['convergence_rate'])
            comparison['fastest_convergence'] = {
                'reward': fastest_convergence_reward[0],
                'value': fastest_convergence_reward[1]['metrics']['convergence_rate']
            }
            
            most_stable_reward = min(valid_results.items(),
                                    key=lambda x: x[1]['metrics']['stability'])
            comparison['most_stable'] = {
                'reward': most_stable_reward[0],
                'value': most_stable_reward[1]['metrics']['stability']
            }
            
            lowest_loss_reward = min(valid_results.items(),
                                    key=lambda x: x[1]['metrics']['final_loss'])
            comparison['lowest_loss'] = {
                'reward': lowest_loss_reward[0],
                'value': lowest_loss_reward[1]['metrics']['final_loss']
            }
            
            # Overall ranking (by final latency)
            ranked = sorted(valid_results.items(),
                           key=lambda x: x[1]['metrics']['final_latency'])
            comparison['overall_ranking'] = [r[0] for r in ranked]
            
            # Detailed comparison
            for reward_name, result in valid_results.items():
                comparison['detailed_comparison'][reward_name] = result['metrics']
        
        return comparison
    
    def _generate_report(self, comparison):
        """Generate and print comparison report with plots."""
        print("\n" + "="*80)
        print("REWARD COMPARISON RESULTS")
        print("="*80)
        
        if not comparison.get('overall_ranking'):
            print("No valid results to compare.")
            return
        
        # Best performers
        print("\n🏆 BEST PERFORMERS:")
        print(f"  Best Latency:      {comparison['best_latency']['reward']:20s} ({comparison['best_latency']['value']:.4f})")
        print(f"  Fastest Convergence: {comparison['fastest_convergence']['reward']:20s} ({comparison['fastest_convergence']['value']:.4f})")
        print(f"  Most Stable:       {comparison['most_stable']['reward']:20s} ({comparison['most_stable']['value']:.4f})")
        print(f"  Lowest Loss:       {comparison['lowest_loss']['reward']:20s} ({comparison['lowest_loss']['value']:.4f})")
        
        # Overall ranking
        print("\n📊 OVERALL RANKING (by final latency):")
        for i, reward in enumerate(comparison['overall_ranking'], 1):
            metrics = comparison['detailed_comparison'][reward]
            print(f"  {i}. {reward:20s} - Latency: {metrics['final_latency']:.4f}, "
                  f"Best: {metrics['best_latency']:.4f}, "
                  f"Convergence: {metrics['convergence_rate']:.4f}")
        
        print("\n" + "="*80)
        
        # Generate comparison plots (using automated_reporting style)
        print("\n📊 Generating comparison plots...")
        self._generate_comparison_plots(comparison)
        
        # Save results to file
        self._save_results(comparison)
    
    def _generate_comparison_plots(self, comparison):
        """Generate comparison plots (adapted from automated_reporting style)."""
        # Get valid results
        valid_results = {k: v for k, v in self.results.items() 
                        if v.get('metrics') is not None and 'error' not in v}
        
        if not valid_results:
            print("Warning: No valid results to plot")
            return
        
        # Use matplotlib style similar to automated_reporting
        try:
            plt.style.use('seaborn-darkgrid')
        except:
            plt.style.use('default')
        
        # Get colors
        colors = plt.cm.tab10(np.linspace(0, 1, len(valid_results)))
        color_map = {reward: colors[i] for i, reward in enumerate(valid_results.keys())}
        
        # Create comprehensive comparison figure
        fig = plt.figure(figsize=(20, 15))
        fig.suptitle(f'Reward System Comparison Report\nSession: {self.timestamp}', 
                    fontsize=16, fontweight='bold')
        
        # Plot 1: Loss curves comparison
        ax1 = plt.subplot(3, 3, 1)
        for reward_name, result in valid_results.items():
            metrics = result['metrics']
            if len(metrics['avg_policy_loss']) > 0:
                plt.plot(metrics['iteration'], metrics['avg_policy_loss'],
                        label=f'{reward_name}', 
                        color=color_map[reward_name], 
                        linewidth=2, marker='o', markersize=3, alpha=0.7)
        plt.xlabel('Iteration')
        plt.ylabel('Policy Loss')
        plt.title('Policy Loss Comparison')
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Reward curves comparison
        ax2 = plt.subplot(3, 3, 2)
        for reward_name, result in valid_results.items():
            metrics = result['metrics']
            if len(metrics['avg_reward']) > 0:
                plt.plot(metrics['iteration'], metrics['avg_reward'],
                        label=reward_name,
                        color=color_map[reward_name],
                        linewidth=2, marker='o', markersize=3)
        plt.xlabel('Iteration')
        plt.ylabel('Average Reward')
        plt.title('Reward Comparison')
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Latency comparison
        ax3 = plt.subplot(3, 3, 3)
        for reward_name, result in valid_results.items():
            metrics = result['metrics']
            if len(metrics['avg_latency']) > 0:
                plt.plot(metrics['iteration'], metrics['avg_latency'],
                        label=reward_name,
                        color=color_map[reward_name],
                        linewidth=2, marker='o', markersize=3)
        plt.xlabel('Iteration')
        plt.ylabel('Average Latency')
        plt.title('Latency Comparison')
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)
        
        # Plot 4: Learning progress (smoothed rewards)
        ax4 = plt.subplot(3, 3, 4)
        for reward_name, result in valid_results.items():
            metrics = result['metrics']
            if len(metrics['avg_reward']) > 10:
                rewards = np.array(metrics['avg_reward'])
                window = min(10, len(rewards) // 4)
                if window > 1:
                    moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
                    iterations = metrics['iteration'][window-1:]
                    plt.plot(iterations, moving_avg,
                            label=f'{reward_name} (smoothed)',
                            color=color_map[reward_name],
                            linewidth=3, alpha=0.8)
        plt.xlabel('Iteration')
        plt.ylabel('Reward (Smoothed)')
        plt.title('Learning Progress Comparison')
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)
        
        # Plot 5-8: Bar charts for metrics
        reward_names = comparison['overall_ranking']
        colors_list = [color_map[r] for r in reward_names]
        
        # Final latency
        ax5 = plt.subplot(3, 3, 5)
        final_latencies = [comparison['detailed_comparison'][r]['final_latency'] for r in reward_names]
        bars = plt.bar(range(len(reward_names)), final_latencies, color=colors_list, alpha=0.7)
        plt.xticks(range(len(reward_names)), reward_names, rotation=45, ha='right')
        plt.ylabel('Final Latency')
        plt.title('Final Latency Comparison')
        plt.grid(True, alpha=0.3, axis='y')
        for bar, val in zip(bars, final_latencies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{val:.4f}', ha='center', va='bottom', fontsize=8)
        
        # Best latency
        ax6 = plt.subplot(3, 3, 6)
        best_latencies = [comparison['detailed_comparison'][r]['best_latency'] for r in reward_names]
        bars = plt.bar(range(len(reward_names)), best_latencies, color=colors_list, alpha=0.7)
        plt.xticks(range(len(reward_names)), reward_names, rotation=45, ha='right')
        plt.ylabel('Best Latency')
        plt.title('Best Latency Comparison')
        plt.grid(True, alpha=0.3, axis='y')
        for bar, val in zip(bars, best_latencies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{val:.4f}', ha='center', va='bottom', fontsize=8)
        
        # Convergence rate
        ax7 = plt.subplot(3, 3, 7)
        convergence_rates = [comparison['detailed_comparison'][r]['convergence_rate'] for r in reward_names]
        bars = plt.bar(range(len(reward_names)), convergence_rates, color=colors_list, alpha=0.7)
        plt.xticks(range(len(reward_names)), reward_names, rotation=45, ha='right')
        plt.ylabel('Convergence Rate')
        plt.title('Convergence Rate Comparison')
        plt.grid(True, alpha=0.3, axis='y')
        for bar, val in zip(bars, convergence_rates):
            if val != float('inf'):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f'{val:.4f}', ha='center', va='bottom', fontsize=8)
        
        # Stability
        ax8 = plt.subplot(3, 3, 8)
        stabilities = [comparison['detailed_comparison'][r]['stability'] for r in reward_names]
        bars = plt.bar(range(len(reward_names)), stabilities, color=colors_list, alpha=0.7)
        plt.xticks(range(len(reward_names)), reward_names, rotation=45, ha='right')
        plt.ylabel('Stability (CV)')
        plt.title('Stability Comparison')
        plt.grid(True, alpha=0.3, axis='y')
        for bar, val in zip(bars, stabilities):
            if val != float('inf'):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f'{val:.4f}', ha='center', va='bottom', fontsize=8)
        
        # Summary statistics
        ax9 = plt.subplot(3, 3, 9)
        ax9.axis('off')
        stats_text = "Comparison Summary:\n\n"
        stats_text += f"Best Latency: {comparison['best_latency']['reward']}\n"
        stats_text += f"  Value: {comparison['best_latency']['value']:.4f}\n\n"
        stats_text += f"Fastest Convergence: {comparison['fastest_convergence']['reward']}\n"
        stats_text += f"  Value: {comparison['fastest_convergence']['value']:.4f}\n\n"
        stats_text += f"Most Stable: {comparison['most_stable']['reward']}\n"
        stats_text += f"  Value: {comparison['most_stable']['value']:.4f}\n\n"
        stats_text += f"Total Rewards Tested: {len(valid_results)}\n"
        stats_text += f"Session: {self.timestamp}"
        ax9.text(0.05, 0.95, stats_text, transform=ax9.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
        
        plt.tight_layout()
        
        # Save comprehensive comparison plot
        comparison_plot_file = os.path.join(self.results_dir, "reward_comparison_report.png")
        plt.savefig(comparison_plot_file, dpi=300, bbox_inches='tight')
        print(f"✓ Comparison plot saved: {comparison_plot_file}")
        plt.close(fig)
        
        # Generate individual plots
        self._generate_individual_reward_plots(valid_results, color_map)
    
    def _generate_individual_reward_plots(self, valid_results, color_map):
        """Generate individual plots for each reward."""
        plots_dir = os.path.join(self.results_dir, "individual_reward_plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        for reward_name, result in valid_results.items():
            metrics = result['metrics']
            reward_dir = os.path.join(plots_dir, reward_name)
            os.makedirs(reward_dir, exist_ok=True)
            
            # Loss plot
            if len(metrics['avg_policy_loss']) > 0:
                plt.figure(figsize=(10, 6))
                plt.plot(metrics['iteration'], metrics['avg_policy_loss'],
                        label='Policy Loss', linewidth=2, marker='o', markersize=4,
                        color=color_map[reward_name])
                plt.xlabel('Iteration')
                plt.ylabel('Loss')
                plt.title(f'{reward_name.replace("_", " ").title()} - Training Losses')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.savefig(os.path.join(reward_dir, "losses.png"), dpi=300, bbox_inches='tight')
                plt.close()
            
            # Reward plot
            if len(metrics['avg_reward']) > 0:
                plt.figure(figsize=(10, 6))
                plt.plot(metrics['iteration'], metrics['avg_reward'],
                        color=color_map[reward_name], linewidth=2, marker='o', markersize=4)
                plt.xlabel('Iteration')
                plt.ylabel('Average Reward')
                plt.title(f'{reward_name.replace("_", " ").title()} - Reward Progress')
                plt.grid(True, alpha=0.3)
                plt.savefig(os.path.join(reward_dir, "rewards.png"), dpi=300, bbox_inches='tight')
                plt.close()
            
            # Latency plot
            if len(metrics['avg_latency']) > 0:
                plt.figure(figsize=(10, 6))
                plt.plot(metrics['iteration'], metrics['avg_latency'],
                        color=color_map[reward_name], linewidth=2, marker='o', markersize=4)
                plt.xlabel('Iteration')
                plt.ylabel('Average Latency')
                plt.title(f'{reward_name.replace("_", " ").title()} - Latency Progress')
                plt.grid(True, alpha=0.3)
                plt.savefig(os.path.join(reward_dir, "latency.png"), dpi=300, bbox_inches='tight')
                plt.close()
        
        print(f"✓ Individual plots saved to: {plots_dir}")
    
    def _save_results(self, comparison):
        """Save comparison results to file."""
        import json
        
        # Save as JSON
        results_file = os.path.join(self.results_dir, "comparison_results.json")
        with open(results_file, 'w') as f:
            json_comparison = self._convert_to_json_serializable(comparison)
            json.dump(json_comparison, f, indent=2)
        
        print(f"✓ Results saved to: {results_file}")
    
    def _convert_to_json_serializable(self, obj):
        """Convert numpy types to Python types for JSON serialization."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(item) for item in obj]
        else:
            return obj
    
    def _cleanup(self):
        """Clean up temporary files."""
        import shutil
        try:
            shutil.rmtree(self.temp_dir)
        except:
            pass

