# Offloading Visualization Pipeline

A comprehensive visual evaluation pipeline for offloading scheduler systems (UE ↔ MEC). Generates static figures and cinematic animations from episode data.

## Features

### Static Figures
- **DAG Visualization**: Per-episode DAG with decisions (LOCAL vs EDGE)
- **Timeline/Gantt Charts**: 4-lane timeline (uplink, edge, downlink, UE)
- **CDF Analysis**: Latency cumulative distribution function across methods
- **Adaptation Curves**: Latency vs adapt_step with confidence intervals
- **Pareto Frontiers**: Multi-objective optimization visualization

### Cinematic Animations
- **Animated DAG**: Topological execution order with glowing nodes
- **Resource Utilization**: Real-time UE/MEC CPU and link usage
- **Timeline Animation**: Moving clock cursor with task execution
- **Live KPIs**: Elapsed time, remaining time, decision counts, rates
- **Summary Overlay**: Performance vs baselines with improvement percentages

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Generate Sample Data
```bash
python cli/offload_viz.py sample-data --episodes 5 --methods ours,heft,greedy
```

### Generate All Figures
```bash
python cli/offload_viz.py figures --input data/sample_eval.jsonl --outdir reports/figures
```

### Create Animation
```bash
python cli/offload_viz.py animate --input data/sample_eval.jsonl --episode 1 --outdir reports/videos
```

### Complete Pipeline
```bash
python cli/offload_viz.py all --input data/sample_eval.jsonl --animate-episode 1
```

## Data Format

The pipeline expects JSONL input with the following schema:

```json
{
  "episode_id": 1,
  "method": "ours",
  "dag": {
    "nodes": [
      {"id": 1, "cpu_cycles": 100.0, "up_size": 50.0, "down_size": 25.0}
    ],
    "edges": [[1, 2]]
  },
  "decisions": [
    {
      "node": 1,
      "action": "LOCAL",
      "t_local": 2.0,
      "finish_times": {"ue": 5.0}
    }
  ],
  "latency_total": 10.0,
  "rates": {"uplink": 50.0, "downlink": 30.0},
  "adapt_step": 5,
  "energy_ue": 200.0,
  "comm_cost": 100.0,
  "baselines": {"heft": 12.0, "greedy": 8.0}
}
```

## CLI Commands

### `figures`
Generate static figures for episodes.

**Options:**
- `--input, -i`: Input JSONL file path (required)
- `--outdir, -o`: Output directory (default: reports/figures)
- `--methods, -m`: Comma-separated methods (default: ours,heft,greedy)
- `--episodes, -e`: Episode IDs or "all" (default: all)
- `--formats, -f`: Output formats (default: png,svg)
- `--title, -t`: Custom title for figures

### `animate`
Create cinematic animation for specific episode.

**Options:**
- `--input, -i`: Input JSONL file path (required)
- `--episode, -e`: Episode ID to animate (required)
- `--outdir, -o`: Output directory (default: reports/videos)
- `--fps, -f`: Frames per second (default: 30)
- `--speed, -s`: Animation speed multiplier (default: 1.0)
- `--formats`: Output formats (default: mp4,gif)
- `--method, -m`: Specific method to animate

### `all`
Generate all figures and optionally animate.

**Options:**
- `--input, -i`: Input JSONL file path (required)
- `--outdir, -o`: Output directory (default: reports)
- `--methods, -m`: Comma-separated methods
- `--episodes, -e`: Episode IDs or "all"
- `--animate-episode, -a`: Specific episode ID to animate
- `--fps, -f`: Frames per second for animation
- `--speed, -s`: Animation speed multiplier
- `--formats`: Output formats

### `sample-data`
Generate sample evaluation data for testing.

**Options:**
- `--output, -o`: Output file path (default: data/sample_eval.jsonl)
- `--episodes, -e`: Number of episodes (default: 3)
- `--methods, -m`: Comma-separated methods

## Output Structure

```
reports/
├── figures/
│   ├── dag_ep1_ours.png
│   ├── dag_ep1_ours.svg
│   ├── gantt_ep1_ours.png
│   ├── gantt_ep1_ours.svg
│   ├── cdf_latency.png
│   ├── cdf_latency.svg
│   ├── adaptation.png
│   ├── adaptation.svg
│   ├── frontier.png
│   └── frontier.svg
└── videos/
    ├── episode1_ours_cinematic.mp4
    └── episode1_ours_cinematic.gif
```

## Architecture

### Core Modules
- **`schema.py`**: Pydantic models for data validation
- **`io_utils.py`**: Data I/O and processing utilities
- **`metrics.py`**: Statistical analysis and calculations
- **`theme.py`**: Visualization styling and colors

### Figure Generators
- **`dag_figure.py`**: DAG visualization with Plotly
- **`gantt_figure.py`**: Timeline/Gantt charts
- **`cdf_figure.py`**: CDF and distribution analysis
- **`adapt_figure.py`**: Adaptation curve visualization
- **`frontier_figure.py`**: Pareto frontier analysis

### Animation System
- **`animate_episode.py`**: Cinematic animation with matplotlib

### CLI Interface
- **`cli/offload_viz.py`**: Command-line interface with Click

## Testing

```bash
pytest tests/
```

## Requirements

- Python 3.10+
- pydantic>=2
- pandas, numpy, networkx
- matplotlib, plotly, kaleido
- imageio[ffmpeg]
- click, scipy

## License

MIT License