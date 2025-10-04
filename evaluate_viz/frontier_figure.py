"""
Pareto frontier visualization for multi-objective optimization.
"""

import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple

from .schema import EpisodeRecord
from .theme import COLORS, PLOTLY_THEME
from .metrics import calculate_pareto_frontier


def create_pareto_figure(
    records: List[EpisodeRecord],
    x_metric: str = "latency_total",
    y_metric: str = "energy_ue",
    title: Optional[str] = None,
    methods: Optional[List[str]] = None
) -> go.Figure:
    """Create Pareto frontier figure for two metrics."""
    if not records:
        return go.Figure()
    
    # Filter by methods if specified
    if methods:
        records = [r for r in records if r.method in methods]
    
    # Calculate Pareto frontier
    pareto_data = calculate_pareto_frontier(records, x_metric, y_metric)
    
    if not pareto_data:
        return go.Figure()
    
    # Create figure
    fig = go.Figure()
    
    # Color palette for methods
    colors = [COLORS["LOCAL"], COLORS["EDGE"], COLORS["UPLINK"], COLORS["DOWNLINK"]]
    
    for i, (method, pareto_points) in enumerate(pareto_data.items()):
        if not pareto_points:
            continue
        
        # Extract coordinates
        x_coords = [p[0] for p in pareto_points]
        y_coords = [p[1] for p in pareto_points]
        
        # Add Pareto frontier line
        fig.add_trace(
            go.Scatter(
                x=x_coords,
                y=y_coords,
                mode='lines+markers',
                name=f"{method} (Pareto)",
                line=dict(color=colors[i % len(colors)], width=3),
                marker=dict(size=8),
                hovertemplate=f"Method: {method}<br>{x_metric}: %{{x:.3f}}<br>{y_metric}: %{{y:.3f}}<extra></extra>"
            )
        )
    
    # Update layout
    fig.update_layout(
        title=title or f"Pareto Frontier: {x_metric} vs {y_metric}",
        xaxis_title=x_metric.replace('_', ' ').title(),
        yaxis_title=y_metric.replace('_', ' ').title(),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=PLOTLY_THEME["layout"]["font"],
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor=COLORS["GRID"],
            zeroline=True,
            zerolinecolor=COLORS["GRID"]
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=COLORS["GRID"],
            zeroline=True,
            zerolinecolor=COLORS["GRID"]
        )
    )
    
    return fig


def create_scatter_figure(
    records: List[EpisodeRecord],
    x_metric: str = "latency_total",
    y_metric: str = "energy_ue",
    title: Optional[str] = None,
    methods: Optional[List[str]] = None,
    show_pareto: bool = True
) -> go.Figure:
    """Create scatter plot figure with optional Pareto frontier."""
    if not records:
        return go.Figure()
    
    # Filter by methods if specified
    if methods:
        records = [r for r in records if r.method in methods]
    
    # Prepare data
    data = []
    for record in records:
        x_val = getattr(record, x_metric, None)
        y_val = getattr(record, y_metric, None)
        
        if x_val is not None and y_val is not None:
            data.append({
                'method': record.method,
                'x': x_val,
                'y': y_val,
                'episode_id': record.episode_id
            })
    
    if not data:
        return go.Figure()
    
    df = pd.DataFrame(data)
    
    # Create scatter plot
    fig = px.scatter(
        df,
        x='x',
        y='y',
        color='method',
        title=title or f"Scatter Plot: {x_metric} vs {y_metric}",
        color_discrete_sequence=[COLORS["LOCAL"], COLORS["EDGE"], COLORS["UPLINK"], COLORS["DOWNLINK"]],
        hover_data=['episode_id']
    )
    
    # Add Pareto frontier if requested
    if show_pareto:
        pareto_data = calculate_pareto_frontier(records, x_metric, y_metric)
        
        for i, (method, pareto_points) in enumerate(pareto_data.items()):
            if not pareto_points:
                continue
            
            x_coords = [p[0] for p in pareto_points]
            y_coords = [p[1] for p in pareto_points]
            
            fig.add_trace(
                go.Scatter(
                    x=x_coords,
                    y=y_coords,
                    mode='lines',
                    name=f"{method} (Pareto)",
                    line=dict(
                        color=COLORS["TEXT"],
                        width=2,
                        dash='dash'
                    ),
                    showlegend=True,
                    hovertemplate=f"Method: {method}<br>{x_metric}: %{{x:.3f}}<br>{y_metric}: %{{y:.3f}}<extra></extra>"
                )
            )
    
    # Update layout
    fig.update_layout(
        xaxis_title=x_metric.replace('_', ' ').title(),
        yaxis_title=y_metric.replace('_', ' ').title(),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=PLOTLY_THEME["layout"]["font"]
    )
    
    return fig


def create_3d_pareto_figure(
    records: List[EpisodeRecord],
    x_metric: str = "latency_total",
    y_metric: str = "energy_ue",
    z_metric: str = "comm_cost",
    title: Optional[str] = None,
    methods: Optional[List[str]] = None
) -> go.Figure:
    """Create 3D Pareto frontier figure for three metrics."""
    if not records:
        return go.Figure()
    
    # Filter by methods if specified
    if methods:
        records = [r for r in records if r.method in methods]
    
    # Prepare data
    data = []
    for record in records:
        x_val = getattr(record, x_metric, None)
        y_val = getattr(record, y_metric, None)
        z_val = getattr(record, z_metric, None)
        
        if x_val is not None and y_val is not None and z_val is not None:
            data.append({
                'method': record.method,
                'x': x_val,
                'y': y_val,
                'z': z_val,
                'episode_id': record.episode_id
            })
    
    if not data:
        return go.Figure()
    
    df = pd.DataFrame(data)
    
    # Create 3D scatter plot
    fig = go.Figure()
    
    # Color palette for methods
    colors = [COLORS["LOCAL"], COLORS["EDGE"], COLORS["UPLINK"], COLORS["DOWNLINK"]]
    
    for i, method in enumerate(df['method'].unique()):
        method_data = df[df['method'] == method]
        
        fig.add_trace(
            go.Scatter3d(
                x=method_data['x'],
                y=method_data['y'],
                z=method_data['z'],
                mode='markers',
                name=method,
                marker=dict(
                    size=8,
                    color=colors[i % len(colors)],
                    opacity=0.7
                ),
                hovertemplate=f"Method: {method}<br>{x_metric}: %{{x:.3f}}<br>{y_metric}: %{{y:.3f}}<br>{z_metric}: %{{z:.3f}}<extra></extra>"
            )
        )
    
    # Update layout
    fig.update_layout(
        title=title or f"3D Pareto: {x_metric} vs {y_metric} vs {z_metric}",
        scene=dict(
            xaxis_title=x_metric.replace('_', ' ').title(),
            yaxis_title=y_metric.replace('_', ' ').title(),
            zaxis_title=z_metric.replace('_', ' ').title(),
            bgcolor='white',
            gridcolor=COLORS["GRID"]
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=PLOTLY_THEME["layout"]["font"]
    )
    
    return fig


def create_hypervolume_figure(
    records: List[EpisodeRecord],
    metrics: List[str] = ["latency_total", "energy_ue", "comm_cost"],
    title: Optional[str] = None,
    methods: Optional[List[str]] = None
) -> go.Figure:
    """Create hypervolume comparison figure."""
    if not records:
        return go.Figure()
    
    # Filter by methods if specified
    if methods:
        records = [r for r in records if r.method in methods]
    
    # Calculate hypervolume for each method
    hypervolume_data = {}
    
    for method in set(r.method for r in records):
        method_records = [r for r in records if r.method == method]
        
        # Filter records that have all metrics
        valid_records = []
        for record in method_records:
            if all(getattr(record, metric, None) is not None for metric in metrics):
                valid_records.append(record)
        
        if not valid_records:
            continue
        
        # Calculate hypervolume (simplified - in practice, use proper hypervolume calculation)
        hypervolume = 0
        for record in valid_records:
            # Simple product of normalized metrics (inverse for minimization)
            values = [getattr(record, metric) for metric in metrics]
            if all(v > 0 for v in values):
                hypervolume += 1 / np.prod(values)
        
        hypervolume_data[method] = hypervolume
    
    if not hypervolume_data:
        return go.Figure()
    
    # Create bar chart
    methods_list = list(hypervolume_data.keys())
    hypervolumes = list(hypervolume_data.values())
    
    fig = go.Figure(data=[
        go.Bar(
            x=methods_list,
            y=hypervolumes,
            marker_color=[COLORS["LOCAL"], COLORS["EDGE"], COLORS["UPLINK"], COLORS["DOWNLINK"]][:len(methods_list)],
            hovertemplate="Method: %{x}<br>Hypervolume: %{y:.3f}<extra></extra>"
        )
    ])
    
    fig.update_layout(
        title=title or f"Hypervolume Comparison ({', '.join(metrics)})",
        xaxis_title="Method",
        yaxis_title="Hypervolume",
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=PLOTLY_THEME["layout"]["font"]
    )
    
    return fig


def create_radar_figure(
    records: List[EpisodeRecord],
    metrics: List[str] = ["latency_total", "energy_ue", "comm_cost"],
    title: Optional[str] = None,
    methods: Optional[List[str]] = None
) -> go.Figure:
    """Create radar chart for multi-metric comparison."""
    if not records:
        return go.Figure()
    
    # Filter by methods if specified
    if methods:
        records = [r for r in records if r.method in methods]
    
    # Calculate normalized metrics for each method
    method_data = {}
    
    for method in set(r.method for r in records):
        method_records = [r for r in records if r.method == method]
        
        # Filter records that have all metrics
        valid_records = []
        for record in method_records:
            if all(getattr(record, metric, None) is not None for metric in metrics):
                valid_records.append(record)
        
        if not valid_records:
            continue
        
        # Calculate mean values for each metric
        mean_values = []
        for metric in metrics:
            values = [getattr(record, metric) for record in valid_records]
            mean_values.append(np.mean(values))
        
        method_data[method] = mean_values
    
    if not method_data:
        return go.Figure()
    
    # Create radar chart
    fig = go.Figure()
    
    # Color palette for methods
    colors = [COLORS["LOCAL"], COLORS["EDGE"], COLORS["UPLINK"], COLORS["DOWNLINK"]]
    
    for i, (method, values) in enumerate(method_data.items()):
        # Normalize values (0-1 scale)
        normalized_values = []
        for j, metric in enumerate(metrics):
            all_values = [getattr(record, metric) for record in records if getattr(record, metric) is not None]
            if all_values:
                min_val = min(all_values)
                max_val = max(all_values)
                if max_val > min_val:
                    normalized_values.append((values[j] - min_val) / (max_val - min_val))
                else:
                    normalized_values.append(0.5)
            else:
                normalized_values.append(0.5)
        
        # Close the radar chart
        normalized_values.append(normalized_values[0])
        
        fig.add_trace(
            go.Scatterpolar(
                r=normalized_values,
                theta=metrics + [metrics[0]],  # Close the circle
                fill='toself',
                name=method,
                line_color=colors[i % len(colors)],
                fillcolor=colors[i % len(colors)],
                opacity=0.3
            )
        )
    
    fig.update_layout(
        title=title or f"Radar Chart: {', '.join(metrics)}",
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=PLOTLY_THEME["layout"]["font"]
    )
    
    return fig


def save_frontier_figure(
    records: List[EpisodeRecord],
    output_path: str,
    x_metric: str = "latency_total",
    y_metric: str = "energy_ue",
    title: Optional[str] = None,
    methods: Optional[List[str]] = None,
    formats: List[str] = ["png", "svg"],
    include_other_plots: bool = True
) -> None:
    """Save Pareto frontier figure and related plots in specified formats."""
    # Save main Pareto plot
    pareto_fig = create_pareto_figure(records, x_metric, y_metric, title, methods)
    
    for fmt in formats:
        if fmt == "png":
            pareto_fig.write_image(f"{output_path}.png", width=800, height=600, scale=2)
        elif fmt == "svg":
            pareto_fig.write_image(f"{output_path}.svg", width=800, height=600)
        elif fmt == "html":
            pareto_fig.write_html(f"{output_path}.html")
    
    if include_other_plots:
        # Save scatter plot
        scatter_fig = create_scatter_figure(records, x_metric, y_metric, f"{title} - Scatter" if title else "Scatter Plot", methods)
        
        for fmt in formats:
            if fmt == "png":
                scatter_fig.write_image(f"{output_path}_scatter.png", width=800, height=600, scale=2)
            elif fmt == "svg":
                scatter_fig.write_image(f"{output_path}_scatter.svg", width=800, height=600)
            elif fmt == "html":
                scatter_fig.write_html(f"{output_path}_scatter.html")
        
        # Save 3D plot if we have 3 metrics
        if len([m for m in ["latency_total", "energy_ue", "comm_cost"] if any(hasattr(r, m) and getattr(r, m) is not None for r in records)]) >= 3:
            metrics_3d = ["latency_total", "energy_ue", "comm_cost"]
            available_metrics = [m for m in metrics_3d if any(hasattr(r, m) and getattr(r, m) is not None for r in records)]
            
            if len(available_metrics) >= 3:
                x_3d, y_3d, z_3d = available_metrics[:3]
                fig_3d = create_3d_pareto_figure(records, x_3d, y_3d, z_3d, f"{title} - 3D" if title else "3D Pareto", methods)
                
                for fmt in formats:
                    if fmt == "png":
                        fig_3d.write_image(f"{output_path}_3d.png", width=800, height=600, scale=2)
                    elif fmt == "svg":
                        fig_3d.write_image(f"{output_path}_3d.svg", width=800, height=600)
                    elif fmt == "html":
                        fig_3d.write_html(f"{output_path}_3d.html")
        
        # Save radar chart
        available_metrics = [m for m in ["latency_total", "energy_ue", "comm_cost"] if any(hasattr(r, m) and getattr(r, m) is not None for r in records)]
        if len(available_metrics) >= 2:
            radar_fig = create_radar_figure(records, available_metrics, f"{title} - Radar" if title else "Radar Chart", methods)
            
            for fmt in formats:
                if fmt == "png":
                    radar_fig.write_image(f"{output_path}_radar.png", width=800, height=600, scale=2)
                elif fmt == "svg":
                    radar_fig.write_image(f"{output_path}_radar.svg", width=800, height=600)
                elif fmt == "html":
                    radar_fig.write_html(f"{output_path}_radar.html")
    
    print(f"Frontier figures saved: {output_path}.{{{','.join(formats)}}}")


def create_frontier_summary_figure(
    records: List[EpisodeRecord],
    title: Optional[str] = None,
    methods: Optional[List[str]] = None
) -> go.Figure:
    """Create frontier summary figure with multiple subplots."""
    if not records:
        return go.Figure()
    
    # Filter by methods if specified
    if methods:
        records = [r for r in records if r.method in methods]
    
    # Create subplots
    from plotly.subplots import make_subplots
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Latency vs Energy",
            "Latency vs Comm Cost",
            "Energy vs Comm Cost",
            "Hypervolume Comparison"
        ],
        specs=[
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "scatter"}, {"type": "bar"}]
        ]
    )
    
    # Add latency vs energy
    if any(hasattr(r, 'energy_ue') and getattr(r, 'energy_ue') is not None for r in records):
        pareto_fig1 = create_pareto_figure(records, "latency_total", "energy_ue", methods=methods)
        for trace in pareto_fig1.data:
            fig.add_trace(trace, row=1, col=1)
    
    # Add latency vs comm cost
    if any(hasattr(r, 'comm_cost') and getattr(r, 'comm_cost') is not None for r in records):
        pareto_fig2 = create_pareto_figure(records, "latency_total", "comm_cost", methods=methods)
        for trace in pareto_fig2.data:
            fig.add_trace(trace, row=1, col=2)
    
    # Add energy vs comm cost
    if (any(hasattr(r, 'energy_ue') and getattr(r, 'energy_ue') is not None for r in records) and
        any(hasattr(r, 'comm_cost') and getattr(r, 'comm_cost') is not None for r in records)):
        pareto_fig3 = create_pareto_figure(records, "energy_ue", "comm_cost", methods=methods)
        for trace in pareto_fig3.data:
            fig.add_trace(trace, row=2, col=1)
    
    # Add hypervolume comparison
    available_metrics = [m for m in ["latency_total", "energy_ue", "comm_cost"] if any(hasattr(r, m) and getattr(r, m) is not None for r in records)]
    if len(available_metrics) >= 2:
        hypervol_fig = create_hypervolume_figure(records, available_metrics, methods=methods)
        for trace in hypervol_fig.data:
            fig.add_trace(trace, row=2, col=2)
    
    # Update layout
    fig.update_layout(
        title=title or "Pareto Frontier Analysis Summary",
        height=800,
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=PLOTLY_THEME["layout"]["font"]
    )
    
    return fig


