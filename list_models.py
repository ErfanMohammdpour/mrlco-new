"""
List all available trained models
"""

import os
import glob


def main():
    """List all available models"""
    
    print("="*60)
    print("AVAILABLE TRAINED MODELS")
    print("="*60)
    
    model_dir = "./simple_model/"
    
    if not os.path.exists(model_dir):
        print("❌ Model directory not found: ./simple_model/")
        print("Run simple_train.py first to train models.")
        return
    
    # Find all checkpoint files
    checkpoint_files = glob.glob(os.path.join(model_dir, "*.ckpt"))
    
    if not checkpoint_files:
        print("❌ No model checkpoints found in ./simple_model/")
        print("Run simple_train.py first to train models.")
        return
    
    # Sort by episode number
    episode_models = []
    other_models = []
    
    for model_file in checkpoint_files:
        filename = os.path.basename(model_file)
        if 'episode' in filename:
            try:
                episode_num = int(filename.split('_')[-1].split('.')[0])
                episode_models.append((episode_num, model_file))
            except:
                other_models.append(model_file)
        else:
            other_models.append(model_file)
    
    # Sort episode models by episode number
    episode_models.sort(key=lambda x: x[0])
    
    print(f"Found {len(checkpoint_files)} model(s):")
    print()
    
    if episode_models:
        print("EPISODE CHECKPOINTS:")
        print("-" * 40)
        for episode_num, model_file in episode_models:
            file_size = os.path.getsize(model_file) / (1024 * 1024)  # Size in MB
            print(f"  Episode {episode_num:4d}: {model_file} ({file_size:.1f} MB)")
    
    if other_models:
        print("\nOTHER MODELS:")
        print("-" * 40)
        for model_file in other_models:
            filename = os.path.basename(model_file)
            file_size = os.path.getsize(model_file) / (1024 * 1024)  # Size in MB
            print(f"  {filename:30s} ({file_size:.1f} MB)")
    
    print()
    print("="*60)
    print("USAGE:")
    print("="*60)
    print("To evaluate with latest model:")
    print("  python simple_evaluate.py")
    print()
    print("To evaluate with specific model:")
    print("  python simple_evaluate.py ./simple_model/checkpoint_episode_500.ckpt")
    print("  python simple_evaluate.py ./simple_model/checkpoint_final.ckpt")
    print("="*60)


if __name__ == "__main__":
    main()
