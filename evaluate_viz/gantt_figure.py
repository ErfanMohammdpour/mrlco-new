"""
Gantt chart visualization for 4-lane timeline (uplink, edge, downlink, UE).
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import List, Dict, Tuple, Optional
import pandas as pd
import numpy as np

from .schema import EpisodeRecord, Decision
from .theme import COLORS, PLOTLY_THEME


def create_gantt_figure(
    record: EpisodeRecord,
    title: Optional[str] = None,
    show_utilization: bool = True
) -> go.Figure:
    """Create a 4-lane Gantt chart figure."""
    lanes = ["uplink", "edge", "downlink", "ue"]
    lane_labels = ["Uplink", "Edge Processing", "Downlink", "UE Processing"]
    
    # Create subplots for each lane
    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=lane_labels,
        vertical_spacing=0.05,
        specs=[[{"type": "bar"}], [{"type": "bar"}], [{"type": "bar"}], [{"type": "bar"}]]
    )
    
    # Process each lane
    for lane_idx, lane in enumerate(lanes):
        tasks = []
        
        for decision in record.decisions:
            if not decision.finish_times or lane not in decision.finish_times:
                continue
            
            finish_time = decision.finish_times[lane]
            
            # Calculate duration for this lane
            duration = 0
            if lane == "ue" and decision.action == "LOCAL" and decision.t_local:
                duration = decision.t_local
            elif lane == "uplink" and decision.action == "EDGE" and decision.t_net_up:
                duration = decision.t_net_up
            elif lane == "edge" and decision.action == "EDGE" and decision.t_edge:
                duration = decision.t_edge
            elif lane == "downlink" and decision.action == "EDGE" and decision.t_net_down:
                duration = decision.t_net_down
            
            if duration > 0:
                start_time = finish_time - duration
                tasks.append({
                    'node_id': decision.node,
                    'start': start_time,
                    'end': finish_time,
                    'duration': duration,
                    'action': decision.action
                })
        
        # Sort tasks by start time
        tasks.sort(key=lambda x: x['start'])
        
        # Create Gantt bars for this lane
        if tasks:
            y_positions = list(range(len(tasks)))
            starts = [task['start'] for task in tasks]
            durations = [task['duration'] for task in tasks]
            colors = [COLORS.get(task['action'], COLORS["IDLE"]) for task in tasks]
            
            # Add hover text
            hover_text = [
                f"Node {task['node_id']}<br>"
                f"Action: {task['action']}<br>"
                f"Start: {task['start']:.2f}s<br>"
                f"Duration: {task['duration']:.2f}s<br>"
                f"End: {task['end']:.2f}s"
                for task in tasks
            ]
            
            fig.add_trace(
                go.Bar(
                    x=starts,
                    y=y_positions,
                    width=durations,
                    orientation='h',
                    marker_color=colors,
                    marker_line=dict(width=1, color='white'),
                    hovertemplate="%{customdata}<extra></extra>",
                    customdata=hover_text,
                    name=f"Tasks ({lane})",
                    showlegend=False
                ),
                row=lane_idx + 1, col=1
            )
        
        # Update lane subplot
        fig.update_xaxes(
            title_text="Time (s)" if lane_idx == 3 else "",
            showgrid=True,
            gridcolor=COLORS["GRID"],
            row=lane_idx + 1, col=1
        )
        fig.update_yaxes(
            title_text=lane_labels[lane_idx],
            showgrid=False,
            showticklabels=False,
            row=lane_idx + 1, col=1
        )
    
    # Update layout
    fig.update_layout(
        title=title or f"Timeline - Episode {record.episode_id} ({record.method})",
        titlefont=PLOTLY_THEME["layout"]["font"],
        height=600,
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    return fig


def create_utilization_figure(
    record: EpisodeRecord,
    title: Optional[str] = None
) -> go.Figure:
    """Create resource utilization over time figure."""
    from .metrics import calculate_resource_utilization
    
    utilization = calculate_resource_utilization(record)
    
    # Create time series data
    max_time = record.latency_total
    time_points = np.linspace(0, max_time, 100)
    
    # Calculate utilization at each time point
    utilization_data = {}
    for resource, intervals in utilization.items():
        util_values = []
        for t in time_points:
            # Count how many tasks are active at time t
            active_tasks = sum(1 for start, end in intervals if start <= t <= end)
            util_values.append(active_tasks)
        utilization_data[resource] = util_values
    
    # Create figure
    fig = go.Figure()
    
    colors = {
        'ue': COLORS["LOCAL"],
        'edge': COLORS["EDGE"],
        'uplink': COLORS["UPLINK"],
        'downlink': COLORS["DOWNLINK"]
    }
    
    labels = {
        'ue': 'UE CPU',
        'edge': 'Edge CPU',
        'uplink': 'Uplink',
        'downlink': 'Downlink'
    }
    
    for resource, values in utilization_data.items():
        fig.add_trace(
            go.Scatter(
                x=time_points,
                y=values,
                mode='lines',
                name=labels[resource],
                line=dict(color=colors[resource], width=2),
                fill='tonexty' if resource != 'ue' else 'tozeroy'
            )
        )
    
    fig.update_layout(
        title=title or f"Resource Utilization - Episode {record.episode_id}",
        xaxis_title="Time (s)",
        yaxis_title="Active Tasks",
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=PLOTLY_THEME["layout"]["font"]
    )
    
    return fig


def create_comparison_gantt_figure(
    records: List[EpisodeRecord],
    title: Optional[str] = None
) -> go.Figure:
    """Create comparison Gantt chart for multiple methods."""
    if not records:
        return go.Figure()
    
    # Create subplots for each method
    n_methods = len(records)
    fig = make_subplots(
        rows=n_methods, cols=1,
        subplot_titles=[f"{record.method} (Episode {record.episode_id})" for record in records],
        vertical_spacing=0.05,
        specs=[[{"type": "bar"} for _ in range(1)] for _ in range(n_methods)]
    )
    
    for i, record in enumerate(records):
        # Create Gantt for this method
        gantt_fig = create_gantt_figure(record)
        
        # Add traces to subplot
        for trace in gantt_fig.data:
            fig.add_trace(trace, row=i + 1, col=1)
        
        # Update axes
        fig.update_xaxes(
            title_text="Time (s)" if i == n_methods - 1 else "",
            showgrid=True,
            gridcolor=COLORS["GRID"],
            row=i + 1, col=1
        )
        fig.update_yaxes(
            showgrid=False,
            showticklabels=False,
            row=i + 1, col=1
        )
    
    fig.update_layout(
        title=title or "Timeline Comparison",
        height=200 * n_methods,
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    return fig


def save_gantt_figure(
    record: EpisodeRecord,
    output_path: str,
    title: Optional[str] = None,
    formats: List[str] = ["png", "svg"],
    include_utilization: bool = True
) -> None:
    """Save Gantt figure in specified formats."""
    # Save main Gantt chart
    gantt_fig = create_gantt_figure(record, title)
    
    for fmt in formats:
        if fmt == "png":
            gantt_fig.write_image(f"{output_path}.png", width=1000, height=600, scale=2)
        elif fmt == "svg":
            gantt_fig.write_image(f"{output_path}.svg", width=1000, height=600)
        elif fmt == "html":
            gantt_fig.write_html(f"{output_path}.html")
    
    # Save utilization chart if requested
    if include_utilization:
        util_fig = create_utilization_figure(record, f"{title} - Utilization" if title else None)
        
        for fmt in formats:
            if fmt == "png":
                util_fig.write_image(f"{output_path}_utilization.png", width=1000, height=400, scale=2)
            elif fmt == "svg":
                util_fig.write_image(f"{output_path}_utilization.svg", width=1000, height=400)
            elif fmt == "html":
                util_fig.write_html(f"{output_path}_utilization.html")
    
    print(f"Gantt figure saved: {output_path}.{{{','.join(formats)}}}")


def create_timeline_summary_figure(
    records: List[EpisodeRecord],
    title: Optional[str] = None
) -> go.Figure:
    """Create a summary timeline showing all methods side by side."""
    if not records:
        return go.Figure()
    
    # Create a single timeline with all methods
    fig = go.Figure()
    
    colors = [COLORS["LOCAL"], COLORS["EDGE"], COLORS["UPLINK"], COLORS["DOWNLINK"]]
    
    for i, record in enumerate(records):
        # Calculate total latency for this method
        total_latency = record.latency_total
        
        # Add a horizontal bar for this method
        fig.add_trace(
            go.Bar(
                x=[total_latency],
                y=[record.method],
                orientation='h',
                marker_color=colors[i % len(colors)],
                name=record.method,
                hovertemplate=f"Method: {record.method}<br>Latency: {total_latency:.2f}s<extra></extra>"
            )
        )
    
    fig.update_layout(
        title=title or "Method Comparison - Total Latency",
        xaxis_title="Total Latency (s)",
        yaxis_title="Method",
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=PLOTLY_THEME["layout"]["font"],
        height=400
    )
    
    return fig


