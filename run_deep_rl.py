"""
Simple script to run Deep RL training with different configurations
"""

import sys
import os
import argparse
from deep_rl_config import DeepRLConfig

def main():
    parser = argparse.ArgumentParser(description='Run Deep RL Offloading Training')
    parser.add_argument('--mode', type=str, default='train', 
                       choices=['test', 'train', 'tune', 'compare'],
                       help='Mode to run: test, train, tune, or compare')
    parser.add_argument('--config', type=str, default='default',
                       choices=['default', 'fast_training', 'production'],
                       help='Configuration preset to use')
    parser.add_argument('--episodes', type=int, default=None,
                       help='Number of training episodes (overrides config)')
    parser.add_argument('--gpu', action='store_true',
                       help='Use GPU if available')
    
    args = parser.parse_args()
    
    # Set up TensorFlow
    if not args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU usage
    
    print("="*60)
    print("DEEP RL OFFLOADING SYSTEM")
    print("="*60)
    print(f"Mode: {args.mode}")
    print(f"Config: {args.config}")
    print(f"GPU: {'Yes' if args.gpu else 'No'}")
    print("="*60)
    
    if args.mode == 'test':
        print("Running implementation tests...")
        from test_deep_rl import main as test_main
        success = test_main()
        if not success:
            print("❌ Tests failed. Please fix issues before training.")
            sys.exit(1)
        print("✅ All tests passed!")
        
    elif args.mode == 'train':
        print("Starting Deep RL training...")
        
        # Load configuration
        config = DeepRLConfig.get_config(args.config)
        
        # Override episodes if specified
        if args.episodes:
            config['training']['n_episodes'] = args.episodes
            print(f"Overriding episodes to: {args.episodes}")
        
        # Create directories
        DeepRLConfig.create_directories()
        
        # Import and run trainer
        from deep_rl_trainer import main as train_main
        train_main()
        
    elif args.mode == 'tune':
        print("Starting hyperparameter tuning...")
        
        from hyperparameter_tuning import HyperparameterTuner
        
        tuner = HyperparameterTuner(
            search_type='random',  # or 'grid'
            n_trials=20,
            n_episodes=300
        )
        
        best_config, best_score = tuner.run_tuning()
        print(f"\n🎉 Best configuration found with score: {best_score:.4f}")
        
    elif args.mode == 'compare':
        print("Running comparison with Meta-RL...")
        
        from compare_meta_vs_deep_rl import main as compare_main
        stats = compare_main()
        
        print("\n🎉 Comparison completed!")
        print("Check the generated plots and logs for detailed results.")
    
    print("\n" + "="*60)
    print("DONE!")
    print("="*60)

if __name__ == "__main__":
    main()
