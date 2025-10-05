# Simple Deep RL Offloading System

## 🎯 Overview
A simplified Deep Reinforcement Learning system for mobile edge computing task offloading, based on the original Meta-RL project.

## 📁 Essential Files

### Core Deep RL System
- `simple_train.py` - Main training script (saves weights every 100 episodes)
- `simple_evaluate.py` - Main evaluation script (loads latest model, prints per-step results)
- `deep_rl_offloading.py` - Deep RL agent implementation
- `list_models.py` - List all available trained models

### Supporting Files
- `run_simple.bat` - Windows batch file to run training + evaluation
- `env/` - Environment and task graph files
- `policies/` - Graph2Seq encoder and policy files
- `baselines/` - Baseline implementations
- `utils/` - Utility functions

### Original Meta-RL (Reference)
- `meta_algos/` - Original MRLCO algorithm
- `meta_trainer.py` - Original meta training
- `meta_evaluator.py` - Original meta evaluation
- `samplers/` - Sampling utilities

## 🚀 Quick Start

### 1. Train Model
```bash
python simple_train.py
```
- Trains for 1000 episodes
- Saves weights every 100 episodes to `./simple_model/`
- Saves final model as `checkpoint_final.ckpt`

### 2. Evaluate Model
```bash
python simple_evaluate.py
```
- Automatically loads latest model
- Evaluates on 50 tasks
- Prints results for each task
- Saves evaluation results

### 3. Use Specific Model
```bash
python simple_evaluate.py ./simple_model/checkpoint_episode_500.ckpt
```

### 4. List Available Models
```bash
python list_models.py
```

### 5. Run Both (Training + Evaluation)
```bash
run_simple.bat
```

## 📊 Model Weights

**Saved every 100 episodes:**
```
./simple_model/
├── checkpoint_episode_100.ckpt
├── checkpoint_episode_200.ckpt
├── checkpoint_episode_300.ckpt
├── ...
├── checkpoint_episode_1000.ckpt
└── checkpoint_final.ckpt
```

## 🎛️ Configuration

**Training Parameters:**
- Episodes: 1000
- Save interval: 100 episodes
- Log interval: 10 episodes
- Max episode length: 50 steps

**Evaluation Parameters:**
- Tasks: 50
- Max episode length: 50 steps

## 📈 Output

**Training:**
- Progress every 10 episodes
- Model saves every 100 episodes
- Final training results saved to `training_results.pkl`

**Evaluation:**
- Per-task results printed to console
- Final evaluation summary
- Results saved to `evaluation_results.pkl` and `evaluation_results.txt`

## 🔧 Requirements

- Python 3.6+
- TensorFlow 1.15
- NumPy
- Joblib
- NetworkX (for graph processing)

## 📝 Notes

- This is a simplified version focused on easy training and evaluation
- Original Meta-RL files are kept for reference
- All complex comparison and testing files have been removed
- System is designed for straightforward use without flags or complex configuration
