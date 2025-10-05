"""
Test script to verify Deep RL implementation works correctly
"""

import tensorflow as tf
import numpy as np
import sys
import os

def test_imports():
    """Test that all required modules can be imported"""
    print("Testing imports...")
    
    try:
        from deep_rl_offloading import DeepRLOffloadingAgent, ReplayBuffer
        print("✅ Deep RL agent imports successful")
    except ImportError as e:
        print(f"❌ Deep RL agent import failed: {e}")
        return False
    
    try:
        from deep_rl_trainer import DeepRLTrainer
        print("✅ Deep RL trainer imports successful")
    except ImportError as e:
        print(f"❌ Deep RL trainer import failed: {e}")
        return False
    
    try:
        from deep_rl_config import DeepRLConfig
        print("✅ Deep RL config imports successful")
    except ImportError as e:
        print(f"❌ Deep RL config import failed: {e}")
        return False
    
    try:
        from env.mec_offloaing_envs.offloading_env import Resources, OffloadingEnvironment
        print("✅ Environment imports successful")
    except ImportError as e:
        print(f"❌ Environment import failed: {e}")
        return False
    
    return True

def test_agent_creation():
    """Test that the Deep RL agent can be created"""
    print("\nTesting agent creation...")
    
    try:
        from deep_rl_offloading import DeepRLOffloadingAgent
        
        # Create agent
        agent = DeepRLOffloadingAgent(
            obs_dim=17,
            action_dim=2,
            encoder_units=64,  # Smaller for testing
            decoder_units=64,
            learning_rate=1e-3,
            buffer_size=1000,  # Smaller for testing
            batch_size=16
        )
        
        print("✅ Agent creation successful")
        print(f"   - Observation dim: {agent.obs_dim}")
        print(f"   - Action dim: {agent.action_dim}")
        print(f"   - Buffer size: {len(agent.replay_buffer)}")
        print(f"   - Epsilon: {agent.epsilon}")
        
        return True
        
    except Exception as e:
        print(f"❌ Agent creation failed: {e}")
        return False

def test_environment_creation():
    """Test that the environment can be created"""
    print("\nTesting environment creation...")
    
    try:
        from env.mec_offloaing_envs.offloading_env import Resources, OffloadingEnvironment
        
        # Create resource cluster
        resource_cluster = Resources(
            mec_process_capable=(10.0 * 1024 * 1024),
            mobile_process_capable=(1.0 * 1024 * 1024),
            bandwidth_up=7.0,
            bandwidth_dl=7.0
        )
        
        # Create environment with minimal setup
        env = OffloadingEnvironment(
            resource_cluster=resource_cluster,
            batch_size=1,
            graph_number=5,  # Small number for testing
            graph_file_paths=[
                "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_1/random.20."
            ],
            time_major=False
        )
        
        print("✅ Environment creation successful")
        print(f"   - Total tasks: {env.total_task}")
        print(f"   - Input dim: {env.input_dim}")
        
        return True
        
    except Exception as e:
        print(f"❌ Environment creation failed: {e}")
        return False

def test_agent_environment_interaction():
    """Test that agent can interact with environment"""
    print("\nTesting agent-environment interaction...")
    
    try:
        from deep_rl_offloading import DeepRLOffloadingAgent
        from env.mec_offloaing_envs.offloading_env import Resources, OffloadingEnvironment
        
        # Create minimal setup
        resource_cluster = Resources(
            mec_process_capable=(10.0 * 1024 * 1024),
            mobile_process_capable=(1.0 * 1024 * 1024),
            bandwidth_up=7.0,
            bandwidth_dl=7.0
        )
        
        env = OffloadingEnvironment(
            resource_cluster=resource_cluster,
            batch_size=1,
            graph_number=3,
            graph_file_paths=[
                "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_1/random.20."
            ],
            time_major=False
        )
        
        agent = DeepRLOffloadingAgent(
            obs_dim=17,
            action_dim=2,
            encoder_units=32,  # Very small for testing
            decoder_units=32,
            buffer_size=100,
            batch_size=8
        )
        
        # Test interaction
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            
            # Sample a task
            task_id = env.sample_tasks(1)[0]
            env.set_task(task_id)
            
            # Get observation
            obs = env.reset()
            sequence_length = np.array([obs.shape[1]] * obs.shape[0], dtype=np.int32)
            
            # Get action from agent
            actions = agent.get_action(obs, sequence_length, training=True)
            
            # Take step in environment
            next_obs, rewards, done, info = env.step(actions)
            
            # Store experience
            dones = np.full((obs.shape[0], obs.shape[1]), done, dtype=bool)
            agent.store_experience(obs, actions, rewards, next_obs, dones, sequence_length)
            
            print("✅ Agent-environment interaction successful")
            print(f"   - Observation shape: {obs.shape}")
            print(f"   - Action shape: {actions.shape}")
            print(f"   - Reward shape: {rewards.shape}")
            print(f"   - Buffer size after interaction: {len(agent.replay_buffer)}")
            
            return True
            
    except Exception as e:
        print(f"❌ Agent-environment interaction failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_configuration():
    """Test configuration system"""
    print("\nTesting configuration system...")
    
    try:
        from deep_rl_config import DeepRLConfig
        
        # Test default config
        config = DeepRLConfig.get_config('default')
        print("✅ Default configuration loaded")
        
        # Test fast training config
        fast_config = DeepRLConfig.get_config('fast_training')
        print("✅ Fast training configuration loaded")
        
        # Test configuration validation
        DeepRLConfig.validate_config(config)
        print("✅ Configuration validation successful")
        
        # Test directory creation
        DeepRLConfig.create_directories()
        print("✅ Directory creation successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def run_quick_training_test():
    """Run a very quick training test"""
    print("\nRunning quick training test...")
    
    try:
        from deep_rl_offloading import DeepRLOffloadingAgent
        from env.mec_offloaing_envs.offloading_env import Resources, OffloadingEnvironment
        
        # Create minimal setup
        resource_cluster = Resources(
            mec_process_capable=(10.0 * 1024 * 1024),
            mobile_process_capable=(1.0 * 1024 * 1024),
            bandwidth_up=7.0,
            bandwidth_dl=7.0
        )
        
        env = OffloadingEnvironment(
            resource_cluster=resource_cluster,
            batch_size=1,
            graph_number=3,
            graph_file_paths=[
                "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_1/random.20."
            ],
            time_major=False
        )
        
        agent = DeepRLOffloadingAgent(
            obs_dim=17,
            action_dim=2,
            encoder_units=32,
            decoder_units=32,
            buffer_size=100,
            batch_size=8,
            learning_rate=1e-3
        )
        
        # Run a few episodes
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            
            episode_rewards = []
            
            for episode in range(5):  # Very short test
                # Sample task
                task_id = env.sample_tasks(1)[0]
                env.set_task(task_id)
                
                # Run episode
                obs = env.reset()
                sequence_length = np.array([obs.shape[1]] * obs.shape[0], dtype=np.int32)
                episode_reward = 0
                
                for step in range(10):  # Short episode
                    actions = agent.get_action(obs, sequence_length, training=True)
                    next_obs, rewards, done, info = env.step(actions)
                    
                    dones = np.full((obs.shape[0], obs.shape[1]), done, dtype=bool)
                    agent.store_experience(obs, actions, rewards, next_obs, dones, sequence_length)
                    
                    episode_reward += np.sum(rewards)
                    obs = next_obs
                    
                    if done:
                        break
                
                episode_rewards.append(episode_reward)
                
                # Update agent if buffer has enough samples
                if len(agent.replay_buffer) >= agent.batch_size:
                    actor_loss, critic_loss = agent.update()
                    print(f"   Episode {episode+1}: Reward = {episode_reward:.4f}, Actor Loss = {actor_loss:.4f}, Critic Loss = {critic_loss:.4f}")
                else:
                    print(f"   Episode {episode+1}: Reward = {episode_reward:.4f} (no update yet)")
            
            print("✅ Quick training test successful")
            print(f"   - Average reward: {np.mean(episode_rewards):.4f}")
            print(f"   - Final buffer size: {len(agent.replay_buffer)}")
            print(f"   - Final epsilon: {agent.epsilon:.4f}")
            
            return True
            
    except Exception as e:
        print(f"❌ Quick training test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("="*60)
    print("DEEP RL IMPLEMENTATION TEST")
    print("="*60)
    
    tests = [
        ("Import Test", test_imports),
        ("Agent Creation", test_agent_creation),
        ("Environment Creation", test_environment_creation),
        ("Agent-Environment Interaction", test_agent_environment_interaction),
        ("Configuration System", test_configuration),
        ("Quick Training Test", run_quick_training_test),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
    
    print("\n" + "="*60)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    print("="*60)
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! The Deep RL implementation is ready to run.")
        print("\nTo start training, run:")
        print("  python deep_rl_trainer.py")
        print("\nTo run hyperparameter tuning, run:")
        print("  python hyperparameter_tuning.py")
        print("\nTo compare with Meta-RL, run:")
        print("  python compare_meta_vs_deep_rl.py")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
