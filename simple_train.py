"""
Simple Deep RL Training Script
Trains model for 4500 episodes and saves weights every 200 episodes
"""

import tensorflow as tf
import numpy as np
import os
import time
from utils import logger
from deep_rl_offloading import DeepRLOffloadingAgent
from env.mec_offloaing_envs.offloading_env import Resources, OffloadingEnvironment


def main():
    """Simple training function"""
    
    # Set up logging
    tf.logging.set_verbosity(tf.logging.ERROR)
    logger.configure(dir="./simple_training_log/", format_strs=['stdout', 'log', 'csv'])
    
    print("="*60)
    print("SIMPLE DEEP RL TRAINING")
    print("="*60)
    
    # Configuration
    n_episodes = 4500
    save_interval = 200
    max_episode_length = 50
    
    # Create environment
    print("Creating environment...")
    resource_cluster = Resources(
        mec_process_capable=(10.0 * 1024 * 1024),
        mobile_process_capable=(1.0 * 1024 * 1024),
        bandwidth_up=7.0,
        bandwidth_dl=7.0
    )
    
    env = OffloadingEnvironment(resource_cluster=resource_cluster,
                                batch_size=100,
                                graph_number=100,
                                graph_file_paths=[
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_1/random.20.",
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_2/random.20.",
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_3/random.20.",
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_4/random.20.",
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_5/random.20.",
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_6/random.20.",
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_7/random.20.",
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_8/random.20.",
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_9/random.20.",
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_11/random.20.",
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_12/random.20.",
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_13/random.20.",
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_14/random.20.",
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_15/random.20.",
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_16/random.20.",
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_17/random.20.",
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_18/random.20.",
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_19/random.20.",
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_20/random.20.",
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_22/random.20.",
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_23/random.20.",
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_24/random.20.",
                                ],
                                time_major=False)
    
    # Create agent
    print("Creating Deep RL agent...")
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
    
    # Training metrics
    episode_rewards = []
    episode_latencies = []
    episode_lengths = []
    actor_losses = []
    critic_losses = []
    epsilon_history = []
    
    # Create model directory
    os.makedirs("./simple_model/", exist_ok=True)
    
    print(f"Starting training for {n_episodes} episodes...")
    print(f"Saving model every {save_interval} episodes...")
    
    start_time = time.time()
    
    # Training loop
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for episode in range(n_episodes):
            episode_start_time = time.time()
            
            # Sample a random task
            task_id = env.sample_tasks(1)[0]
            env.set_task(task_id)
            
            # Run episode
            obs = env.reset()
            sequence_length = np.array([obs.shape[1]] * obs.shape[0], dtype=np.int32)
            
            episode_reward = 0
            episode_length = 0
            episode_latency = 0
            
            # Store initial observation
            prev_obs = obs.copy()
            
            # Run episode steps
            for step in range(max_episode_length):
                # Get action from agent
                actions = agent.get_action(obs, sequence_length, training=True)
                
                # Take step in environment
                next_obs, rewards, done, info = env.step(actions)
                
                # Store experience
                dones = np.full((obs.shape[0], obs.shape[1]), done, dtype=bool)
                agent.store_experience(
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
            
            # Update agent if buffer has enough samples
            if len(agent.replay_buffer) >= agent.batch_size:
                actor_loss, critic_loss = agent.update()
                actor_losses.append(actor_loss)
                critic_losses.append(critic_loss)
            
            # Store episode metrics
            episode_rewards.append(episode_reward)
            episode_latencies.append(episode_latency)
            episode_lengths.append(episode_length)
            epsilon_history.append(agent.epsilon)
            
            # Print progress
            if episode % 50 == 0:
                recent_rewards = episode_rewards[-100:] if len(episode_rewards) >= 100 else episode_rewards
                recent_latencies = episode_latencies[-100:] if len(episode_latencies) >= 100 else episode_latencies
                
                avg_reward = np.mean(recent_rewards)
                avg_latency = np.mean(recent_latencies)
                avg_length = np.mean(episode_lengths[-100:]) if len(episode_lengths) >= 100 else np.mean(episode_lengths)
                
                progress = (episode / n_episodes) * 100
                print(f"Episode {episode:4d}/{n_episodes} ({progress:5.1f}%): Reward={avg_reward:8.4f}, Latency={avg_latency:8.4f}, "
                      f"Length={avg_length:6.2f}, Epsilon={agent.epsilon:.4f}, "
                      f"Buffer={len(agent.replay_buffer):5d}")
            
            # Save model every 200 episodes
            if episode % save_interval == 0 and episode > 0:
                save_path = f"./simple_model/checkpoint_episode_{episode}.ckpt"
                agent.save_model(save_path)
                print(f"Model saved to: {save_path}")
        
        # Save final model
        final_save_path = "./simple_model/checkpoint_final.ckpt"
        agent.save_model(final_save_path)
        print(f"Final model saved to: {final_save_path}")
    
    training_time = time.time() - start_time
    
    # Print final results
    print("\n" + "="*60)
    print("TRAINING COMPLETED")
    print("="*60)
    print(f"Total episodes: {n_episodes}")
    print(f"Training time: {training_time:.2f} seconds")
    print(f"Final average reward: {np.mean(episode_rewards[-100:]):.4f}")
    print(f"Final average latency: {np.mean(episode_latencies[-100:]):.4f}")
    print(f"Final average length: {np.mean(episode_lengths[-100:]):.4f}")
    print(f"Final epsilon: {agent.epsilon:.4f}")
    print(f"Models saved in: ./simple_model/")
    print("="*60)
    
    # Save training results
    training_results = {
        'episode_rewards': episode_rewards,
        'episode_latencies': episode_latencies,
        'episode_lengths': episode_lengths,
        'actor_losses': actor_losses,
        'critic_losses': critic_losses,
        'epsilon_history': epsilon_history,
        'training_time': training_time,
        'n_episodes': n_episodes
    }
    
    import joblib
    joblib.dump(training_results, "./simple_model/training_results.pkl")
    print("Training results saved to: ./simple_model/training_results.pkl")


if __name__ == "__main__":
    main()
