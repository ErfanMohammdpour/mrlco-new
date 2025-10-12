# DRL Fine-tuning System Guide

This guide explains how to use the complete fine-tuning system for Deep Reinforcement Learning (DRL) on task offloading in Mobile Edge Computing (MEC) environments.

## 🎯 Overview

The fine-tuning system provides a complete workflow for:
1. **Pre-training** on 22 maps (general learning)
2. **Saving weights** after pre-training
3. **Loading weights** for evaluation
4. **Fine-tuning** on specific maps for 20 steps
5. **Evaluating** performance on specific maps

## 📁 System Components

### Core Files
- `weight_manager.py` - Manages saving/loading of policy weights
- `pretrain_on_maps.py` - Pre-training script for multiple maps
- `finetune_on_map.py` - Fine-tuning script for specific maps
- `evaluate_policy.py` - Evaluation script with loaded weights
- `finetuning_workflow.py` - Complete workflow automation

### Supporting Files
- `single_policy_trainer.py` - Single-policy trainer
- `single_policy_ppo.py` - PPO algorithm for single-policy RL
- `env/single_policy_offloading_env.py` - Environment wrapper
- `samplers/seq2seq_sampler.py` - Single-policy sampler
- `samplers/seq2seq_sampler_process.py` - Sample processor

## 🚀 Quick Start

### 1. Pre-training on 22 Maps
```bash
# Pre-train on 22 maps for 1000 iterations
python pretrain_on_maps.py --maps 22 --iterations 1000 --batch_size 100 --learning_rate 5e-4

# Or use the workflow script
python finetuning_workflow.py --mode pretrain --maps 22 --iterations 1000
```

### 2. Fine-tuning on Specific Map
```bash
# Fine-tune on map 1 for 20 steps
python finetune_on_map.py --map_id 1 --weights path/to/pretrained.ckpt --steps 20

# Or use the workflow script
python finetuning_workflow.py --mode finetune --map_id 1 --steps 20
```

### 3. Evaluation
```bash
# Evaluate on map 1 with 10 episodes
python evaluate_policy.py --weights path/to/weights.ckpt --map_id 1 --episodes 10

# Or use the workflow script
python finetuning_workflow.py --mode evaluate --map_id 1 --episodes 10
```

### 4. Complete Workflow
```bash
# Run complete workflow: pre-train + fine-tune + evaluate
python finetuning_workflow.py --mode workflow --maps 22 --workflow_maps 1 2 3 --workflow_steps 20
```

## 📊 Detailed Usage

### Pre-training Phase

**Purpose**: Train the policy on multiple maps to learn general offloading strategies.

**Command**:
```bash
python pretrain_on_maps.py --maps 22 --iterations 1000 --batch_size 100 --learning_rate 5e-4 --save_interval 100
```

**Parameters**:
- `--maps`: Number of maps to use for pre-training (default: 22)
- `--iterations`: Number of training iterations (default: 1000)
- `--batch_size`: Batch size for training (default: 100)
- `--learning_rate`: Learning rate for PPO (default: 5e-4)
- `--save_interval`: Save weights every N iterations (default: 100)

**Output**:
- Pre-trained weights saved to `./weights/pretrained/`
- Training logs saved to `./pretraining_logs/`
- Metadata including training statistics

### Fine-tuning Phase

**Purpose**: Fine-tune the pre-trained policy on a specific map for better performance.

**Command**:
```bash
python finetune_on_map.py --map_id 1 --weights path/to/pretrained.ckpt --steps 20 --learning_rate 1e-4 --batch_size 50
```

**Parameters**:
- `--map_id`: ID of the specific map (1-25)
- `--weights`: Path to pre-trained weights
- `--steps`: Number of fine-tuning steps (default: 20)
- `--learning_rate`: Learning rate for fine-tuning (default: 1e-4, usually lower than pre-training)
- `--batch_size`: Batch size for fine-tuning (default: 50)

**Output**:
- Fine-tuned weights saved to `./weights/finetuned/`
- Fine-tuning logs saved to `./finetuning_logs/`
- Metadata including fine-tuning statistics

### Evaluation Phase

**Purpose**: Evaluate the policy performance on specific maps.

**Command**:
```bash
python evaluate_policy.py --weights path/to/weights.ckpt --map_id 1 --episodes 10 --render
```

**Parameters**:
- `--weights`: Path to trained weights (pre-trained or fine-tuned)
- `--map_id`: ID of the specific map (1-25)
- `--episodes`: Number of episodes to evaluate (default: 10)
- `--render`: Whether to render the environment (optional)

**Output**:
- Performance metrics and statistics
- Comparison with baseline solutions (greedy, all-remote, all-local)
- Improvement percentages

## 🔧 Advanced Usage

### Weight Management

The `WeightManager` class provides comprehensive weight management:

```python
from weight_manager import WeightManager

wm = WeightManager()

# Save pre-trained weights
weight_path, metadata_path = wm.save_pretrained_weights(
    policy=policy,
    iteration=1000,
    map_count=22,
    additional_info={'final_reward': 0.5}
)

# Load weights
success = wm.load_weights(policy, weight_path)

# Get latest pre-trained weights
weight_path, metadata_path = wm.get_latest_pretrained_weights()

# Get fine-tuned weights for specific map
weight_path, metadata_path = wm.get_finetuned_weights_for_map(map_id=1)

# List all available weights
weights = wm.list_available_weights()
```

### Custom Map Configuration

To use your own maps, modify the `create_map_list()` function in `pretrain_on_maps.py`:

```python
def create_map_list(num_maps=22):
    base_maps = [
        "path/to/your/map1/",
        "path/to/your/map2/",
        # ... add your map paths
    ]
    return base_maps[:num_maps]
```

### Custom Fine-tuning Parameters

For different fine-tuning strategies, adjust the parameters in `finetune_on_map.py`:

```python
# Conservative fine-tuning (small changes)
learning_rate = 1e-5
num_grad_steps = 1
clip_value = 0.05

# Aggressive fine-tuning (larger changes)
learning_rate = 1e-3
num_grad_steps = 5
clip_value = 0.2
```

## 📈 Performance Monitoring

### Training Metrics
- **Average Reward**: Policy performance
- **Average Loss**: Training stability
- **Average Latency**: Task completion time
- **Greedy Baseline**: Comparison with greedy solution

### Evaluation Metrics
- **Reward Improvement**: Percentage improvement over baseline
- **Latency Improvement**: Percentage improvement over greedy solution
- **Consistency**: Standard deviation of performance across episodes

### Logging
All training and evaluation logs are saved with timestamps:
- `./pretraining_logs/maps_22_iter_1000/`
- `./finetuning_logs/map_1_steps_20/`

## 🎯 Best Practices

### 1. Pre-training
- Use diverse maps for better generalization
- Train for sufficient iterations (1000+)
- Use higher learning rate (5e-4)
- Save weights at regular intervals

### 2. Fine-tuning
- Use lower learning rate (1e-4 or lower)
- Fine-tune for limited steps (20-50)
- Use smaller batch sizes (50)
- Monitor for overfitting

### 3. Evaluation
- Evaluate on multiple episodes (10+)
- Compare with multiple baselines
- Test on different maps
- Monitor consistency

### 4. Weight Management
- Keep multiple weight versions
- Clean up old weights regularly
- Save metadata with weights
- Use descriptive filenames

## 🔍 Troubleshooting

### Common Issues

1. **Weight Loading Failed**
   - Check file path and permissions
   - Verify weight file exists
   - Check TensorFlow version compatibility

2. **Environment Creation Failed**
   - Verify map file paths
   - Check map file format
   - Ensure sufficient memory

3. **Training Diverged**
   - Reduce learning rate
   - Increase batch size
   - Check gradient clipping

4. **Poor Fine-tuning Performance**
   - Reduce learning rate
   - Increase fine-tuning steps
   - Check pre-trained weights quality

### Debug Tips

1. **Enable Verbose Logging**
   ```python
   tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
   ```

2. **Check Weight Metadata**
   ```python
   metadata = wm.load_metadata(metadata_path)
   print(metadata)
   ```

3. **Monitor Training Progress**
   - Check log files regularly
   - Plot training curves
   - Compare with baselines

## 📋 Example Workflows

### Workflow 1: Quick Test
```bash
# Pre-train on 5 maps for 100 iterations
python finetuning_workflow.py --mode pretrain --maps 5 --iterations 100

# Fine-tune on map 1 for 10 steps
python finetuning_workflow.py --mode finetune --map_id 1 --steps 10

# Evaluate on map 1
python finetuning_workflow.py --mode evaluate --map_id 1 --episodes 5
```

### Workflow 2: Production Training
```bash
# Pre-train on 22 maps for 2000 iterations
python finetuning_workflow.py --mode pretrain --maps 22 --iterations 2000

# Fine-tune on multiple maps
python finetuning_workflow.py --mode workflow --maps 22 --workflow_maps 1 2 3 4 5 --workflow_steps 50
```

### Workflow 3: Research Experiment
```bash
# Pre-train on all available maps
python finetuning_workflow.py --mode pretrain --maps 25 --iterations 5000

# Fine-tune on specific maps with different parameters
python finetune_on_map.py --map_id 1 --weights latest_pretrained.ckpt --steps 20 --learning_rate 5e-5
python finetune_on_map.py --map_id 2 --weights latest_pretrained.ckpt --steps 50 --learning_rate 1e-4
```

## 🎉 Success Metrics

A successful fine-tuning system should achieve:
- **Pre-training**: Stable learning curve with improving rewards
- **Fine-tuning**: Quick adaptation to specific maps (5-20 steps)
- **Evaluation**: Better performance than baselines
- **Consistency**: Low variance across multiple runs

The system is designed to be flexible, scalable, and easy to use for various DRL fine-tuning scenarios in MEC environments.
