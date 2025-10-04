# Visualization Integration for Meta Evaluator

This guide shows how to integrate the visualization pipeline with your existing `meta_evaluator.py` to automatically generate videos and figures at the end of evaluation.

## 🚀 Quick Start

### Option 1: Use the Enhanced Version (Recommended)

Simply replace your `meta_evaluator.py` with `meta_evaluator_with_viz_integration.py`:

```bash
cp meta_evaluator_with_viz_integration.py meta_evaluator.py
python meta_evaluator.py
```

### Option 2: Apply Patch to Existing File

```bash
python patch_meta_evaluator.py
cp meta_evaluator_patched.py meta_evaluator.py
python meta_evaluator.py
```

### Option 3: Manual Integration

Add these lines to your existing `meta_evaluator.py`:

1. **Add import at the top:**
```python
from viz_integration import VisualizationCollector
```

2. **Modify Trainer.__init__:**
```python
def __init__(self, algo, env, sampler, sample_processor, policy, n_itr,
             batch_size=500, start_itr=0, num_inner_grad_steps=3,
             enable_viz=True, viz_output_dir='evaluation_results', animate_episode=10):
    # ... existing code ...
    
    # Visualization setup
    self.enable_viz = enable_viz
    if self.enable_viz:
        self.viz_collector = VisualizationCollector(env, viz_output_dir, animate_episode)
        print(f"🎨 Visualization enabled - output directory: {viz_output_dir}")
        print(f"🎬 Animation will be generated for episode: {animate_episode}")
```

3. **Add data collection in train method:**
```python
# After the logging section in train()
if self.enable_viz:
    batch_size = len(samples_data['finish_time']) if 'finish_time' in samples_data else 0
    for task_id in range(batch_size):
        self.viz_collector.collect_episode_data(samples_data, task_id, itr)
```

4. **Add visualization generation at the end:**
```python
# Before return statement in train()
if self.enable_viz and self.viz_collector.evaluation_data:
    print(f"\n🎨 Generating visualizations for {len(self.viz_collector.evaluation_data)} episodes...")
    self.viz_collector.generate_visualizations()
```

## 📊 What Gets Generated

When you run the evaluation with 100 graphs, the system will automatically generate:

### Static Figures (PNG + SVG)
- **DAG Visualizations**: Per-episode DAG with decisions (LOCAL=green, EDGE=blue)
- **Timeline Charts**: 4-lane execution timeline (uplink, edge, downlink, UE)
- **CDF Analysis**: Latency distribution across all episodes
- **Adaptation Curves**: Performance improvement over iterations
- **Pareto Frontiers**: Multi-objective optimization plots

### Cinematic Animation (MP4 + GIF)
- **Episode 10 Animation**: As requested, episode 10 will be animated
- **Animated DAG**: Topological execution with glowing nodes
- **Resource Utilization**: Real-time CPU and link usage
- **Timeline Animation**: Moving clock cursor with task execution
- **Live KPIs**: Elapsed time, decision counts, rates
- **Summary Card**: Performance vs baselines

## 🎛️ Configuration Options

You can customize the visualization behavior:

```python
trainer = Trainer(
    # ... existing parameters ...
    enable_viz=True,                    # Enable/disable visualization
    viz_output_dir='evaluation_results', # Output directory
    animate_episode=10                  # Which episode to animate
)
```

### Advanced Configuration

For more control, you can modify the `VisualizationCollector` directly:

```python
# In your meta_evaluator.py
viz_collector = VisualizationCollector(
    env=env,
    output_dir='custom_output',
    animate_episode=25  # Animate episode 25 instead
)

# Customize formats
viz_collector.formats = ['png', 'svg', 'mp4']  # Skip GIF
viz_collector.fps = 60  # Higher FPS for animation
```

## 📁 Output Structure

```
evaluation_results/
├── figures/
│   ├── dag_ep0_ours.png
│   ├── dag_ep0_ours.svg
│   ├── gantt_ep0_ours.png
│   ├── gantt_ep0_ours.svg
│   ├── cdf_latency.png
│   ├── cdf_latency.svg
│   ├── adaptation.png
│   ├── adaptation.svg
│   ├── frontier.png
│   └── frontier.svg
├── videos/
│   ├── episode10_ours_cinematic.mp4
│   └── episode10_ours_cinematic.gif
└── evaluation_data.jsonl
```

## 🔧 Troubleshooting

### Common Issues

1. **Import Error**: Make sure all visualization dependencies are installed:
   ```bash
   pip install -r requirements.txt
   ```

2. **No Data Collected**: Check that your environment provides the expected data structure:
   - `samples_data['actions']` - Decision actions
   - `samples_data['finish_time']` - Task completion times
   - `env.task_graphs_batchs` - Task graph information

3. **Animation Fails**: Ensure you have ffmpeg installed:
   ```bash
   # Ubuntu/Debian
   sudo apt install ffmpeg
   
   # macOS
   brew install ffmpeg
   
   # Windows
   # Download from https://ffmpeg.org/download.html
   ```

### Debug Mode

Enable debug logging to see what's happening:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🎯 Performance Considerations

- **Data Collection**: Minimal overhead during training
- **Figure Generation**: Only at the end of evaluation
- **Memory Usage**: Episodes are processed in batches
- **Disk Space**: Each episode generates ~2-5MB of figures

## 🎨 Customization

### Custom Colors and Themes

Edit `evaluate_viz/theme.py` to customize:
- Node colors (LOCAL vs EDGE)
- Timeline lane colors
- Animation effects
- Font sizes and styles

### Custom Animation

Modify `evaluate_viz/animate_episode.py` to:
- Change animation speed
- Add custom effects
- Modify layout
- Add new KPI displays

## 📈 Example Usage

```python
# Run evaluation with visualization
python meta_evaluator.py

# Output:
# 🎨 Visualization enabled - output directory: evaluation_results
# 🎬 Animation will be generated for episode: 10
# 
# ---------------- Iteration 0 ----------------
# 📊 Collected data for 100 episodes so far
# 
# ---------------- Iteration 1 ----------------
# 📊 Collected data for 200 episodes so far
# 
# ...
# 
# 🎨 Generating visualizations for 2100 episodes...
# Generating figures for Episode 0 (1/20)
# Generating figures for Episode 1 (2/20)
# ...
# Generating animation for Episode 10
# ✅ All visualizations generated successfully!
```

## 🎉 Results

After running the evaluation, you'll have:
- **Professional static figures** for all episodes
- **Cinematic animation** for episode 10
- **Comprehensive analysis** with CDF, adaptation curves, and Pareto frontiers
- **Baseline comparisons** showing improvement percentages
- **Raw data** in JSONL format for further analysis

The visualization pipeline seamlessly integrates with your existing evaluation workflow, providing rich visual insights into your offloading scheduler's performance!
