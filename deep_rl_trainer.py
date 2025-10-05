"""
Deep RL Trainer for Task Offloading
Replaces meta_trainer.py with single-policy Deep RL approach
"""

import tensorflow as tf
import numpy as np
import time
import os
from utils import logger
from automated_reporting import create_training_report
from deep_rl_offloading import DeepRLOffloadingAgent
from env.mec_offloaing_envs.offloading_env import Resources, OffloadingEnvironment
from samplers.seq2seq_sampler import Seq2SeqSampler
from samplers.seq2seq_sampler_process import Seq2SeSamplerProcessor
from baselines.vf_baseline import ValueFunctionBaseline


class DeepRLTrainer:
    """Deep RL Trainer for task offloading"""
    
    def __init__(self, 
                 agent,
                 env,
                 sampler,
                 sample_processor,
                 n_episodes=1000,
                 max_episode_length=100,
                 save_interval=100,
                 eval_interval=50,
                 log_interval=10):
        
        self.agent = agent
        self.env = env
        self.sampler = sampler
        self.sample_processor = sample_processor
        self.n_episodes = n_episodes
        self.max_episode_length = max_episode_length
        self.save_interval = save_interval
        self.eval_interval = eval_interval
        self.log_interval = log_interval
        
        # Training metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_latencies = []
        self.actor_losses = []
        self.critic_losses = []
        self.epsilon_history = []
        
        # Evaluation metrics
        self.eval_rewards = []
        self.eval_latencies = []
        
    def train(self):
        """Main training loop for Deep RL"""
        
        logger.log("Starting Deep RL Training...")
        start_time = time.time()
        
        for episode in range(self.n_episodes):
            episode_start_time = time.time()
            
            # Sample a random task
            task_id = self.env.sample_tasks(1)[0]
            self.env.set_task(task_id)
            
            # Run episode
            episode_reward, episode_length, episode_latency = self._run_episode()
            
            # Store metrics
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            self.episode_latencies.append(episode_latency)
            self.epsilon_history.append(self.agent.epsilon)
            
            # Update agent
            if len(self.agent.replay_buffer) >= self.agent.batch_size:
                actor_loss, critic_loss = self.agent.update()
                self.actor_losses.append(actor_loss)
                self.critic_losses.append(critic_loss)
            
            # Logging
            if episode % self.log_interval == 0:
                self._log_episode_stats(episode, episode_start_time)
            
            # Evaluation
            if episode % self.eval_interval == 0:
                self._evaluate_agent()
            
            # Save model
            if episode % self.save_interval == 0:
                self._save_model(episode)
        
        # Final evaluation and reporting
        self._final_evaluation()
        self._generate_training_report()
        
        logger.log("Training completed!")
        return self.episode_rewards, self.episode_latencies
    
    def _run_episode(self):
        """Run a single episode"""
        
        obs = self.env.reset()
        episode_reward = 0
        episode_length = 0
        episode_latency = 0
        
        # Get sequence length
        sequence_length = np.array([obs.shape[1]] * obs.shape[0], dtype=np.int32)
        
        # Store initial observation
        prev_obs = obs.copy()
        
        # Run episode steps
        for step in range(self.max_episode_length):
            # Get action from agent
            actions = self.agent.get_action(obs, sequence_length, training=True)
            
            # Take step in environment
            next_obs, rewards, done, info = self.env.step(actions)
            
            # Store experience
            dones = np.full((obs.shape[0], obs.shape[1]), done, dtype=bool)
            self.agent.store_experience(
                prev_obs, actions, rewards, next_obs, dones, sequence_length
            )
            
            # Update metrics
            episode_reward += np.sum(rewards)
            episode_length += 1
            episode_latency = info[0] if isinstance(info, list) else info
            
            # Update observation
            prev_obs = next_obs.copy()
            
            if done:
                break
        
        return episode_reward, episode_length, episode_latency
    
    def _evaluate_agent(self):
        """Evaluate agent on multiple tasks"""
        
        logger.log("Evaluating agent...")
        
        eval_rewards = []
        eval_latencies = []
        
        # Evaluate on multiple random tasks
        for _ in range(10):
            task_id = self.env.sample_tasks(1)[0]
            self.env.set_task(task_id)
            
            obs = self.env.reset()
            sequence_length = np.array([obs.shape[1]] * obs.shape[0], dtype=np.int32)
            
            episode_reward = 0
            episode_latency = 0
            
            for step in range(self.max_episode_length):
                # Get greedy action (no exploration)
                actions = self.agent.get_action(obs, sequence_length, training=False)
                
                # Take step
                next_obs, rewards, done, info = self.env.step(actions)
                
                episode_reward += np.sum(rewards)
                episode_latency = info[0] if isinstance(info, list) else info
                
                obs = next_obs
                
                if done:
                    break
            
            eval_rewards.append(episode_reward)
            eval_latencies.append(episode_latency)
        
        # Store evaluation metrics
        self.eval_rewards.append(np.mean(eval_rewards))
        self.eval_latencies.append(np.mean(eval_latencies))
        
        logger.logkv('Eval_Reward', np.mean(eval_rewards))
        logger.logkv('Eval_Latency', np.mean(eval_latencies))
    
    def _log_episode_stats(self, episode, episode_start_time):
        """Log episode statistics"""
        
        # Recent performance
        recent_rewards = self.episode_rewards[-100:] if len(self.episode_rewards) >= 100 else self.episode_rewards
        recent_latencies = self.episode_latencies[-100:] if len(self.episode_latencies) >= 100 else self.episode_latencies
        
        avg_reward = np.mean(recent_rewards)
        avg_latency = np.mean(recent_latencies)
        avg_length = np.mean(self.episode_lengths[-100:]) if len(self.episode_lengths) >= 100 else np.mean(self.episode_lengths)
        
        # Losses
        avg_actor_loss = np.mean(self.actor_losses[-100:]) if len(self.actor_losses) >= 100 else 0
        avg_critic_loss = np.mean(self.critic_losses[-100:]) if len(self.critic_losses) >= 100 else 0
        
        # Log metrics
        logger.logkv('Episode', episode)
        logger.logkv('Avg_Reward_100', avg_reward)
        logger.logkv('Avg_Latency_100', avg_latency)
        logger.logkv('Avg_Length_100', avg_length)
        logger.logkv('Epsilon', self.agent.epsilon)
        logger.logkv('Actor_Loss', avg_actor_loss)
        logger.logkv('Critic_Loss', avg_critic_loss)
        logger.logkv('Buffer_Size', len(self.agent.replay_buffer))
        logger.logkv('Episode_Time', time.time() - episode_start_time)
        
        logger.dumpkvs()
    
    def _save_model(self, episode):
        """Save model checkpoint"""
        save_path = f"./deep_rl_model/checkpoint_episode_{episode}.ckpt"
        self.agent.save_model(save_path)
        logger.log(f"Model saved to {save_path}")
    
    def _final_evaluation(self):
        """Final evaluation on test tasks"""
        
        logger.log("Running final evaluation...")
        
        # Test on multiple tasks
        test_rewards = []
        test_latencies = []
        
        for _ in range(50):
            task_id = self.env.sample_tasks(1)[0]
            self.env.set_task(task_id)
            
            obs = self.env.reset()
            sequence_length = np.array([obs.shape[1]] * obs.shape[0], dtype=np.int32)
            
            episode_reward = 0
            episode_latency = 0
            
            for step in range(self.max_episode_length):
                actions = self.agent.get_action(obs, sequence_length, training=False)
                next_obs, rewards, done, info = self.env.step(actions)
                
                episode_reward += np.sum(rewards)
                episode_latency = info[0] if isinstance(info, list) else info
                
                obs = next_obs
                
                if done:
                    break
            
            test_rewards.append(episode_reward)
            test_latencies.append(episode_latency)
        
        # Log final results
        logger.logkv('Final_Test_Reward_Mean', np.mean(test_rewards))
        logger.logkv('Final_Test_Reward_Std', np.std(test_rewards))
        logger.logkv('Final_Test_Latency_Mean', np.mean(test_latencies))
        logger.logkv('Final_Test_Latency_Std', np.std(test_latencies))
        
        logger.dumpkvs()
    
    def _generate_training_report(self):
        """Generate comprehensive training report"""
        
        try:
            logger.log("Generating training report...")
            
            additional_metrics = {
                'episode_lengths': self.episode_lengths,
                'actor_losses': self.actor_losses,
                'critic_losses': self.critic_losses,
                'epsilon_history': self.epsilon_history,
                'eval_rewards': self.eval_rewards,
                'eval_latencies': self.eval_latencies
            }
            
            report_dir = create_training_report(
                avg_ret=self.episode_rewards,
                avg_loss=self.actor_losses,  # Using actor loss as main loss
                avg_latencies=self.episode_latencies,
                additional_metrics=additional_metrics
            )
            
            logger.log(f"Training report generated at: {report_dir}")
            
        except Exception as e:
            logger.log(f"Warning: Failed to generate training report: {str(e)}")


def main():
    """Main training function"""
    
    # Set up logging
    tf.logging.set_verbosity(tf.logging.ERROR)
    logger.configure(dir="./deep_rl_offloading_log/", format_strs=['stdout', 'log', 'csv'])
    
    # Environment setup
    resource_cluster = Resources(
        mec_process_capable=(10.0 * 1024 * 1024),
        mobile_process_capable=(1.0 * 1024 * 1024),
        bandwidth_up=7.0, 
        bandwidth_dl=7.0
    )
    
    env = OffloadingEnvironment(
        resource_cluster=resource_cluster,
        batch_size=1,  # Single task per episode
        graph_number=100,
        graph_file_paths=[
            "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_1/random.20.",
            "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_2/random.20.",
            "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_3/random.20.",
            "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_5/random.20.",
            "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_6/random.20.",
            "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_7/random.20.",
            "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_9/random.20.",
            "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_10/random.20.",
            "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_11/random.20.",
            "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_13/random.20.",
        ],
        time_major=False
    )
    
    # Create Deep RL agent
    agent = DeepRLOffloadingAgent(
        obs_dim=17,
        action_dim=2,
        encoder_units=128,
        decoder_units=128,
        learning_rate=3e-4,
        gamma=0.99,
        tau=0.005,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        buffer_size=100000,
        batch_size=64,
        update_frequency=4,
        target_update_frequency=100
    )
    
    # Create trainer
    trainer = DeepRLTrainer(
        agent=agent,
        env=env,
        sampler=None,  # Not needed for Deep RL
        sample_processor=None,  # Not needed for Deep RL
        n_episodes=2000,
        max_episode_length=50,
        save_interval=200,
        eval_interval=100,
        log_interval=10
    )
    
    # Train the agent
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        trainer.train()


if __name__ == "__main__":
    main()
