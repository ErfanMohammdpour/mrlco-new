"""
Hyperparameter Tuning for Deep RL Offloading System
Uses grid search and random search for optimization
"""

import tensorflow as tf
import numpy as np
import itertools
import random
import json
import os
import time
from deep_rl_config import DeepRLConfig
from deep_rl_offloading import DeepRLOffloadingAgent
from deep_rl_trainer import DeepRLTrainer
from env.mec_offloaing_envs.offloading_env import Resources, OffloadingEnvironment
from utils import logger


class HyperparameterTuner:
    """Hyperparameter tuning for Deep RL offloading system"""
    
    def __init__(self, search_type='grid', n_trials=20, n_episodes=500):
        self.search_type = search_type
        self.n_trials = n_trials
        self.n_episodes = n_episodes
        self.results = []
        self.best_config = None
        self.best_score = float('-inf')
        
        # Create environment
        self.env = self._create_environment()
        
    def _create_environment(self):
        """Create offloading environment"""
        resource_cluster = Resources(
            mec_process_capable=DeepRLConfig.ENV_CONFIG['mec_process_capable'],
            mobile_process_capable=DeepRLConfig.ENV_CONFIG['mobile_process_capable'],
            bandwidth_up=DeepRLConfig.ENV_CONFIG['bandwidth_up'],
            bandwidth_dl=DeepRLConfig.ENV_CONFIG['bandwidth_dl']
        )
        
        env = OffloadingEnvironment(
            resource_cluster=resource_cluster,
            batch_size=DeepRLConfig.ENV_CONFIG['batch_size'],
            graph_number=DeepRLConfig.ENV_CONFIG['graph_number'],
            graph_file_paths=DeepRLConfig.TASK_PATHS[:5],  # Use subset for faster tuning
            time_major=DeepRLConfig.ENV_CONFIG['time_major']
        )
        
        return env
    
    def grid_search(self):
        """Perform grid search over hyperparameters"""
        logger.log("Starting Grid Search...")
        
        # Define search space
        search_space = {
            'learning_rate': [1e-4, 3e-4, 1e-3],
            'gamma': [0.95, 0.99],
            'epsilon_decay': [0.99, 0.995],
            'batch_size': [32, 64],
            'encoder_units': [64, 128],
        }
        
        # Generate all combinations
        param_names = list(search_space.keys())
        param_values = list(search_space.values())
        combinations = list(itertools.product(*param_values))
        
        logger.log(f"Testing {len(combinations)} combinations...")
        
        for i, combination in enumerate(combinations):
            if i >= self.n_trials:
                break
                
            # Create parameter dictionary
            params = dict(zip(param_names, combination))
            
            # Run trial
            score = self._run_trial(params, trial_id=i)
            
            # Store results
            result = {
                'trial_id': i,
                'params': params,
                'score': score,
                'timestamp': time.time()
            }
            self.results.append(result)
            
            # Update best
            if score > self.best_score:
                self.best_score = score
                self.best_config = params.copy()
            
            logger.log(f"Trial {i+1}/{min(len(combinations), self.n_trials)}: Score = {score:.4f}")
    
    def random_search(self):
        """Perform random search over hyperparameters"""
        logger.log("Starting Random Search...")
        
        # Define search space
        search_space = {
            'learning_rate': (1e-5, 1e-2),
            'gamma': (0.9, 0.999),
            'tau': (0.001, 0.01),
            'epsilon_decay': (0.99, 0.999),
            'batch_size': (16, 128),
            'encoder_units': (32, 256),
            'decoder_units': (32, 256),
            'buffer_size': (10000, 200000),
        }
        
        for i in range(self.n_trials):
            # Sample random parameters
            params = {}
            for param_name, (min_val, max_val) in search_space.items():
                if param_name in ['batch_size', 'encoder_units', 'decoder_units', 'buffer_size']:
                    # Integer parameters
                    params[param_name] = random.randint(int(min_val), int(max_val))
                else:
                    # Float parameters
                    params[param_name] = random.uniform(min_val, max_val)
            
            # Run trial
            score = self._run_trial(params, trial_id=i)
            
            # Store results
            result = {
                'trial_id': i,
                'params': params,
                'score': score,
                'timestamp': time.time()
            }
            self.results.append(result)
            
            # Update best
            if score > self.best_score:
                self.best_score = score
                self.best_config = params.copy()
            
            logger.log(f"Trial {i+1}/{self.n_trials}: Score = {score:.4f}")
    
    def _run_trial(self, params, trial_id):
        """Run a single trial with given parameters"""
        
        try:
            # Create agent with given parameters
            agent = DeepRLOffloadingAgent(
                obs_dim=17,
                action_dim=2,
                encoder_units=params.get('encoder_units', 128),
                decoder_units=params.get('decoder_units', 128),
                learning_rate=params.get('learning_rate', 3e-4),
                gamma=params.get('gamma', 0.99),
                tau=params.get('tau', 0.005),
                epsilon_start=1.0,
                epsilon_end=0.01,
                epsilon_decay=params.get('epsilon_decay', 0.995),
                buffer_size=params.get('buffer_size', 100000),
                batch_size=params.get('batch_size', 64),
                update_frequency=4,
                target_update_frequency=100
            )
            
            # Create trainer
            trainer = DeepRLTrainer(
                agent=agent,
                env=self.env,
                sampler=None,
                sample_processor=None,
                n_episodes=self.n_episodes,
                max_episode_length=30,  # Shorter for faster tuning
                save_interval=1000,  # Don't save during tuning
                eval_interval=1000,  # Don't evaluate during tuning
                log_interval=1000,  # Minimal logging
            )
            
            # Run training
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                
                # Train for specified episodes
                episode_rewards = []
                
                for episode in range(self.n_episodes):
                    # Sample task
                    task_id = self.env.sample_tasks(1)[0]
                    self.env.set_task(task_id)
                    
                    # Run episode
                    episode_reward, _, _ = trainer._run_episode()
                    episode_rewards.append(episode_reward)
                    
                    # Update agent
                    if len(agent.replay_buffer) >= agent.batch_size:
                        agent.update()
                
                # Calculate score (average of last 100 episodes)
                if len(episode_rewards) >= 100:
                    score = np.mean(episode_rewards[-100:])
                else:
                    score = np.mean(episode_rewards)
                
                return score
                
        except Exception as e:
            logger.log(f"Trial {trial_id} failed: {str(e)}")
            return float('-inf')
    
    def bayesian_optimization(self):
        """Perform Bayesian optimization (placeholder for future implementation)"""
        logger.log("Bayesian optimization not implemented yet")
        logger.log("Falling back to random search...")
        self.random_search()
    
    def run_tuning(self):
        """Run hyperparameter tuning"""
        
        logger.log(f"Starting {self.search_type} search with {self.n_trials} trials...")
        
        if self.search_type == 'grid':
            self.grid_search()
        elif self.search_type == 'random':
            self.random_search()
        elif self.search_type == 'bayesian':
            self.bayesian_optimization()
        else:
            raise ValueError(f"Unknown search type: {self.search_type}")
        
        # Generate report
        self._generate_tuning_report()
        
        return self.best_config, self.best_score
    
    def _generate_tuning_report(self):
        """Generate hyperparameter tuning report"""
        
        logger.log("\n" + "="*60)
        logger.log("HYPERPARAMETER TUNING RESULTS")
        logger.log("="*60)
        
        # Sort results by score
        sorted_results = sorted(self.results, key=lambda x: x['score'], reverse=True)
        
        # Print top 5 results
        logger.log("Top 5 Results:")
        logger.log("-"*60)
        logger.log(f"{'Rank':<5} {'Score':<10} {'Learning Rate':<15} {'Gamma':<10} {'Batch Size':<12} {'Encoder Units':<15}")
        logger.log("-"*60)
        
        for i, result in enumerate(sorted_results[:5]):
            params = result['params']
            logger.log(f"{i+1:<5} {result['score']:<10.4f} {params.get('learning_rate', 'N/A'):<15.6f} "
                      f"{params.get('gamma', 'N/A'):<10.3f} {params.get('batch_size', 'N/A'):<12} "
                      f"{params.get('encoder_units', 'N/A'):<15}")
        
        # Print best configuration
        logger.log("\nBest Configuration:")
        logger.log("-"*30)
        for param, value in self.best_config.items():
            logger.log(f"{param}: {value}")
        
        logger.log(f"\nBest Score: {self.best_score:.4f}")
        
        # Save results to file
        self._save_results()
    
    def _save_results(self):
        """Save tuning results to file"""
        
        results_data = {
            'search_type': self.search_type,
            'n_trials': self.n_trials,
            'n_episodes': self.n_episodes,
            'best_config': self.best_config,
            'best_score': self.best_score,
            'all_results': self.results,
            'timestamp': time.time()
        }
        
        os.makedirs('./tuning_results/', exist_ok=True)
        
        filename = f'./tuning_results/tuning_results_{self.search_type}_{int(time.time())}.json'
        
        with open(filename, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        logger.log(f"Results saved to {filename}")


def main():
    """Main function for hyperparameter tuning"""
    
    # Set up logging
    tf.logging.set_verbosity(tf.logging.ERROR)
    logger.configure(dir="./hyperparameter_tuning_log/", format_strs=['stdout', 'log', 'csv'])
    
    # Create tuner
    tuner = HyperparameterTuner(
        search_type='random',  # 'grid', 'random', or 'bayesian'
        n_trials=20,
        n_episodes=300
    )
    
    # Run tuning
    best_config, best_score = tuner.run_tuning()
    
    print(f"\nBest configuration found:")
    for param, value in best_config.items():
        print(f"  {param}: {value}")
    print(f"Best score: {best_score:.4f}")


if __name__ == "__main__":
    main()
