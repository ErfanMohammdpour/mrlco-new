"""
CLI interface for offloading visualization pipeline.
"""

import click
import json
from pathlib import Path
from typing import List, Optional
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluate_viz.io_utils import read_jsonl, filter_episodes, ensure_output_dirs
from evaluate_viz.dag_figure import save_dag_figure
from evaluate_viz.gantt_figure import save_gantt_figure
from evaluate_viz.cdf_figure import save_cdf_figure
from evaluate_viz.adapt_figure import save_adaptation_figure
from evaluate_viz.frontier_figure import save_frontier_figure
from evaluate_viz.animate_episode import create_episode_animation


@click.group()
def cli():
    """Offloading visualization pipeline for MRLCO project."""
    pass


@cli.command()
@click.option('--input', '-i', required=True, help='Input JSONL file path')
@click.option('--outdir', '-o', default='reports/figures', help='Output directory for figures')
@click.option('--methods', '-m', default='ours,heft,greedy', help='Comma-separated list of methods to include')
@click.option('--episodes', '-e', default='all', help='Episode IDs to include (comma-separated or "all")')
@click.option('--formats', '-f', default='png,svg', help='Output formats (comma-separated: png,svg,html)')
@click.option('--title', '-t', help='Custom title for figures')
def figures(input, outdir, methods, episodes, formats, title):
    """Generate static figures (DAG, Gantt, CDF, Adaptation, Frontier)."""
    try:
        # Parse inputs
        method_list = [m.strip() for m in methods.split(',')]
        format_list = [f.strip() for f in formats.split(',')]
        
        if episodes == 'all':
            episode_list = None
        else:
            episode_list = [int(e.strip()) for e in episodes.split(',')]
        
        # Ensure output directory exists
        ensure_output_dirs(outdir)
        
        # Read data
        click.echo(f"Reading data from {input}...")
        records = read_jsonl(input)
        click.echo(f"Loaded {len(records)} episode records")
        
        # Filter records
        filtered_records = filter_episodes(records, episode_list, method_list)
        click.echo(f"Filtered to {len(filtered_records)} records")
        
        if not filtered_records:
            click.echo("No records found after filtering. Exiting.")
            return
        
        # Generate figures for each episode
        episode_ids = set(r.episode_id for r in filtered_records)
        click.echo(f"Generating figures for {len(episode_ids)} episodes...")
        
        for episode_id in episode_ids:
            episode_records = [r for r in filtered_records if r.episode_id == episode_id]
            
            for record in episode_records:
                click.echo(f"  Processing Episode {record.episode_id} - {record.method}")
                
                # Generate DAG figure
                dag_path = f"{outdir}/dag_ep{record.episode_id}_{record.method}"
                save_dag_figure(record, dag_path, title, format_list)
                
                # Generate Gantt figure
                gantt_path = f"{outdir}/gantt_ep{record.episode_id}_{record.method}"
                save_gantt_figure(record, gantt_path, title, format_list)
        
        # Generate aggregate figures
        click.echo("Generating aggregate figures...")
        
        # CDF figure
        cdf_path = f"{outdir}/cdf_latency"
        save_cdf_figure(filtered_records, cdf_path, title, method_list, format_list)
        
        # Adaptation figure
        adapt_path = f"{outdir}/adaptation"
        save_adaptation_figure(filtered_records, adapt_path, title, method_list, format_list)
        
        # Frontier figure (if energy/comm data available)
        has_energy = any(hasattr(r, 'energy_ue') and getattr(r, 'energy_ue') is not None for r in filtered_records)
        has_comm = any(hasattr(r, 'comm_cost') and getattr(r, 'comm_cost') is not None for r in filtered_records)
        
        if has_energy or has_comm:
            frontier_path = f"{outdir}/frontier"
            save_frontier_figure(filtered_records, frontier_path, title, method_list, format_list)
        
        click.echo(f"✅ All figures generated successfully in {outdir}")
        
    except Exception as e:
        click.echo(f"❌ Error generating figures: {e}", err=True)
        raise click.Abort()


@cli.command()
@click.option('--input', '-i', required=True, help='Input JSONL file path')
@click.option('--episode', '-e', required=True, type=int, help='Episode ID to animate')
@click.option('--outdir', '-o', default='reports/videos', help='Output directory for videos')
@click.option('--fps', '-f', default=30, help='Frames per second')
@click.option('--speed', '-s', default=1.0, help='Animation speed multiplier')
@click.option('--formats', default='mp4,gif', help='Output formats (comma-separated: mp4,gif)')
@click.option('--method', '-m', help='Specific method to animate (if multiple methods for episode)')
def animate(input, episode, outdir, fps, speed, formats, method):
    """Create cinematic animation for a specific episode."""
    try:
        # Parse inputs
        format_list = [f.strip() for f in formats.split(',')]
        
        # Ensure output directory exists
        ensure_output_dirs(outdir)
        
        # Read data
        click.echo(f"Reading data from {input}...")
        records = read_jsonl(input)
        
        # Filter to specific episode
        episode_records = [r for r in records if r.episode_id == episode]
        
        if not episode_records:
            click.echo(f"❌ No records found for episode {episode}", err=True)
            raise click.Abort()
        
        # Filter by method if specified
        if method:
            episode_records = [r for r in episode_records if r.method == method]
            if not episode_records:
                click.echo(f"❌ No records found for episode {episode} with method {method}", err=True)
                raise click.Abort()
        
        # Create animation for each method
        for record in episode_records:
            click.echo(f"Creating animation for Episode {record.episode_id} - {record.method}")
            
            output_path = f"{outdir}/episode{record.episode_id}_{record.method}_cinematic"
            create_episode_animation(record, output_path, fps, speed, format_list)
        
        click.echo(f"✅ Animations generated successfully in {outdir}")
        
    except Exception as e:
        click.echo(f"❌ Error creating animation: {e}", err=True)
        raise click.Abort()


@cli.command()
@click.option('--input', '-i', required=True, help='Input JSONL file path')
@click.option('--outdir', '-o', default='reports', help='Output directory for all outputs')
@click.option('--methods', '-m', default='ours,heft,greedy', help='Comma-separated list of methods to include')
@click.option('--episodes', '-e', default='all', help='Episode IDs to include (comma-separated or "all")')
@click.option('--animate-episode', '-a', type=int, help='Specific episode ID to animate')
@click.option('--fps', '-f', default=30, help='Frames per second for animation')
@click.option('--speed', '-s', default=1.0, help='Animation speed multiplier')
@click.option('--formats', default='png,svg,mp4,gif', help='Output formats (comma-separated)')
def all(input, outdir, methods, episodes, animate_episode, fps, speed, formats):
    """Generate all figures and optionally animate a specific episode."""
    try:
        # Parse inputs
        method_list = [m.strip() for m in methods.split(',')]
        format_list = [f.strip() for f in formats.split(',')]
        
        if episodes == 'all':
            episode_list = None
        else:
            episode_list = [int(e.strip()) for e in episodes.split(',')]
        
        # Ensure output directories exist
        ensure_output_dirs(outdir)
        
        # Read data
        click.echo(f"Reading data from {input}...")
        records = read_jsonl(input)
        click.echo(f"Loaded {len(records)} episode records")
        
        # Filter records
        filtered_records = filter_episodes(records, episode_list, method_list)
        click.echo(f"Filtered to {len(filtered_records)} records")
        
        if not filtered_records:
            click.echo("No records found after filtering. Exiting.")
            return
        
        # Generate static figures
        click.echo("🎨 Generating static figures...")
        figures_ctx = click.Context(figures)
        figures_ctx.invoke(figures, 
                          input=input, 
                          outdir=f"{outdir}/figures", 
                          methods=methods, 
                          episodes=episodes, 
                          formats=','.join([f for f in format_list if f in ['png', 'svg', 'html']]))
        
        # Generate animation if requested
        if animate_episode:
            click.echo(f"🎬 Creating animation for episode {animate_episode}...")
            animate_ctx = click.Context(animate)
            animate_ctx.invoke(animate,
                              input=input,
                              episode=animate_episode,
                              outdir=f"{outdir}/videos",
                              fps=fps,
                              speed=speed,
                              formats=','.join([f for f in format_list if f in ['mp4', 'gif']]))
        
        click.echo("✅ All visualizations generated successfully!")
        
    except Exception as e:
        click.echo(f"❌ Error in pipeline: {e}", err=True)
        raise click.Abort()


@cli.command()
@click.option('--output', '-o', default='data/sample_eval.jsonl', help='Output file path')
@click.option('--episodes', '-e', default=3, help='Number of episodes to generate')
@click.option('--methods', '-m', default='ours,heft,greedy', help='Comma-separated list of methods')
def sample_data(output, episodes, methods):
    """Generate sample evaluation data for testing."""
    try:
        import random
        import json
        from pathlib import Path
        
        # Parse methods
        method_list = [m.strip() for m in methods.split(',')]
        
        # Ensure output directory exists
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        
        click.echo(f"Generating sample data with {episodes} episodes and {len(method_list)} methods...")
        
        # Generate sample data
        sample_records = []
        
        for episode_id in range(1, episodes + 1):
            for method in method_list:
                # Generate random DAG
                num_nodes = random.randint(5, 10)
                nodes = []
                edges = []
                
                for i in range(num_nodes):
                    nodes.append({
                        "id": i + 1,
                        "cpu_cycles": random.uniform(100, 1000),
                        "up_size": random.uniform(10, 100),
                        "down_size": random.uniform(10, 100)
                    })
                
                # Add some random edges
                for _ in range(random.randint(3, 8)):
                    source = random.randint(1, num_nodes)
                    target = random.randint(1, num_nodes)
                    if source != target and [source, target] not in edges:
                        edges.append([source, target])
                
                # Generate decisions
                decisions = []
                for node in nodes:
                    action = random.choice(["LOCAL", "EDGE"])
                    decision = {
                        "node": node["id"],
                        "action": action,
                        "finish_times": {}
                    }
                    
                    if action == "LOCAL":
                        decision["t_local"] = random.uniform(0.1, 2.0)
                        decision["finish_times"]["ue"] = random.uniform(1, 10)
                    else:
                        decision["t_net_up"] = random.uniform(0.1, 1.0)
                        decision["t_edge"] = random.uniform(0.1, 1.5)
                        decision["t_net_down"] = random.uniform(0.1, 1.0)
                        decision["finish_times"]["uplink"] = random.uniform(1, 5)
                        decision["finish_times"]["edge"] = random.uniform(2, 7)
                        decision["finish_times"]["downlink"] = random.uniform(3, 10)
                    
                    decisions.append(decision)
                
                # Generate record
                record = {
                    "episode_id": episode_id,
                    "method": method,
                    "dag": {"nodes": nodes, "edges": edges},
                    "decisions": decisions,
                    "latency_total": random.uniform(5, 15),
                    "rates": {
                        "uplink": random.uniform(10, 100),
                        "downlink": random.uniform(10, 100)
                    },
                    "adapt_step": random.randint(0, 100) if method == "ours" else None,
                    "energy_ue": random.uniform(100, 500),
                    "comm_cost": random.uniform(50, 200),
                    "baselines": {
                        "heft": random.uniform(8, 18),
                        "greedy": random.uniform(6, 16)
                    }
                }
                
                sample_records.append(record)
        
        # Write to file
        with open(output, 'w') as f:
            for record in sample_records:
                f.write(json.dumps(record) + '\n')
        
        click.echo(f"✅ Sample data generated: {output}")
        click.echo(f"   Episodes: {episodes}")
        click.echo(f"   Methods: {', '.join(method_list)}")
        click.echo(f"   Total records: {len(sample_records)}")
        
    except Exception as e:
        click.echo(f"❌ Error generating sample data: {e}", err=True)
        raise click.Abort()


if __name__ == '__main__':
    cli()

