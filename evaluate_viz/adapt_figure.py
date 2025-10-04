"""
Adaptation curve visualization showing latency vs adapt_step.
"""

import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple

from .schema import EpisodeRecord
from .theme import COLORS, PLOTLY_THEME
from .metrics import calculate_adaptation_curve, calculate_confidence_intervals


def create_adaptation_figure(
    records: List[EpisodeRecord],
    title: Optional[str] = None,
    methods: Optional[List[str]] = None,
    show_confidence: bool = True
) -> go.Figure:
    """Create adaptation curve figure showing latency vs adapt_step."""
    if not records:
        return go.Figure()
    
    # Filter by methods if specified
    if methods:
        records = [r for r in records if r.method in methods]
    
    # Calculate adaptation curve data
    adaptation_data = calculate_adaptation_curve(records)
    
    if not adaptation_data:
        return go.Figure()
    
    # Create figure
    fig = go.Figure()
    
    # Color palette for methods
    colors = [COLORS["LOCAL"], COLORS["EDGE"], COLORS["UPLINK"], COLORS["DOWNLINK"]]
    
    for i, (method, data) in enumerate(adaptation_data.items()):
        adapt_steps = data['adapt_steps']
        mean_latency = data['mean_latency']
        std_latency = data['std_latency']
        ci_lower = data['ci_lower']
        ci_upper = data['ci_upper']
        count = data['count']
        
        # Add main line
        fig.add_trace(
            go.Scatter(
                x=adapt_steps,
                y=mean_latency,
                mode='lines+markers',
                name=method,
                line=dict(color=colors[i % len(colors)], width=3),
                marker=dict(size=8),
                hovertemplate=f"Method: {method}<br>Adapt Step: %{{x}}<br>Mean Latency: %{{y:.3f}}s<br>Count: {count[0] if count else 0}<extra></extra>"
            )
        )
        
        # Add confidence interval if requested
        if show_confidence and len(ci_lower) > 0:
            # Upper bound
            fig.add_trace(
                go.Scatter(
                    x=adapt_steps + adapt_steps[::-1],
                    y=ci_upper + ci_lower[::-1],
                    fill='tonexty',
                    fillcolor=colors[i % len(colors)],
                    opacity=0.2,
                    line=dict(color='rgba(255,255,255,0)'),
                    showlegend=False,
                    hoverinfo='skip'
                )
            )
        
        # Add error bars
        if std_latency and any(std > 0 for std in std_latency):
            fig.add_trace(
                go.Scatter(
                    x=adapt_steps,
                    y=mean_latency,
                    mode='markers',
                    error_y=dict(
                        type='data',
                        array=std_latency,
                        visible=True,
                        color=colors[i % len(colors)]
                    ),
                    showlegend=False,
                    hoverinfo='skip'
                )
            )
    
    # Update layout
    fig.update_layout(
        title=title or "Adaptation Curve - Latency vs Adapt Step",
        xaxis_title="Adapt Step",
        yaxis_title="Mean Latency (s)",
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


def create_learning_curve_figure(
    records: List[EpisodeRecord],
    title: Optional[str] = None,
    methods: Optional[List[str]] = None,
    window_size: int = 10
) -> go.Figure:
    """Create learning curve figure with moving average."""
    if not records:
        return go.Figure()
    
    # Filter by methods if specified
    if methods:
        records = [r for r in records if r.method in methods]
    
    # Group by method
    method_data = {}
    for record in records:
        if record.method not in method_data:
            method_data[record.method] = []
        method_data[record.method].append({
            'adapt_step': record.adapt_step or 0,
            'latency': record.latency_total
        })
    
    # Create figure
    fig = go.Figure()
    
    # Color palette for methods
    colors = [COLORS["LOCAL"], COLORS["EDGE"], COLORS["UPLINK"], COLORS["DOWNLINK"]]
    
    for i, (method, data) in enumerate(method_data.items()):
        if not data:
            continue
        
        # Sort by adapt_step
        data.sort(key=lambda x: x['adapt_step'])
        adapt_steps = [d['adapt_step'] for d in data]
        latencies = [d['latency'] for d in data]
        
        # Calculate moving average
        if len(latencies) >= window_size:
            moving_avg = []
            for j in range(len(latencies)):
                start_idx = max(0, j - window_size + 1)
                end_idx = j + 1
                window_latencies = latencies[start_idx:end_idx]
                moving_avg.append(np.mean(window_latencies))
        else:
            moving_avg = latencies
        
        # Add raw data points
        fig.add_trace(
            go.Scatter(
                x=adapt_steps,
                y=latencies,
                mode='markers',
                name=f"{method} (raw)",
                marker=dict(
                    color=colors[i % len(colors)],
                    size=4,
                    opacity=0.3
                ),
                showlegend=False,
                hovertemplate=f"Method: {method}<br>Adapt Step: %{{x}}<br>Latency: %{{y:.3f}}s<extra></extra>"
            )
        )
        
        # Add moving average line
        fig.add_trace(
            go.Scatter(
                x=adapt_steps,
                y=moving_avg,
                mode='lines',
                name=method,
                line=dict(color=colors[i % len(colors)], width=3),
                hovertemplate=f"Method: {method}<br>Adapt Step: %{{x}}<br>Moving Avg: %{{y:.3f}}s<extra></extra>"
            )
        )
    
    # Update layout
    fig.update_layout(
        title=title or f"Learning Curve (Window Size: {window_size})",
        xaxis_title="Adapt Step",
        yaxis_title="Latency (s)",
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=PLOTLY_THEME["layout"]["font"],
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig


def create_convergence_figure(
    records: List[EpisodeRecord],
    title: Optional[str] = None,
    methods: Optional[List[str]] = None,
    convergence_threshold: float = 0.01
) -> go.Figure:
    """Create convergence analysis figure."""
    if not records:
        return go.Figure()
    
    # Filter by methods if specified
    if methods:
        records = [r for r in records if r.method in methods]
    
    # Group by method
    method_data = {}
    for record in records:
        if record.method not in method_data:
            method_data[record.method] = []
        method_data[record.method].append({
            'adapt_step': record.adapt_step or 0,
            'latency': record.latency_total
        })
    
    # Create figure
    fig = go.Figure()
    
    # Color palette for methods
    colors = [COLORS["LOCAL"], COLORS["EDGE"], COLORS["UPLINK"], COLORS["DOWNLINK"]]
    
    for i, (method, data) in enumerate(method_data.items()):
        if not data:
            continue
        
        # Sort by adapt_step
        data.sort(key=lambda x: x['adapt_step'])
        adapt_steps = [d['adapt_step'] for d in data]
        latencies = [d['latency'] for d in data]
        
        # Calculate convergence metrics
        if len(latencies) > 1:
            # Calculate rolling standard deviation
            window_size = min(10, len(latencies) // 2)
            rolling_std = []
            for j in range(len(latencies)):
                start_idx = max(0, j - window_size + 1)
                end_idx = j + 1
                window_latencies = latencies[start_idx:end_idx]
                rolling_std.append(np.std(window_latencies))
            
            # Find convergence point
            convergence_step = None
            for j, std in enumerate(rolling_std):
                if std < convergence_threshold:
                    convergence_step = adapt_steps[j]
                    break
            
            # Add convergence line
            if convergence_step is not None:
                fig.add_vline(
                    x=convergence_step,
                    line_dash="dash",
                    line_color=colors[i % len(colors)],
                    opacity=0.7,
                    annotation_text=f"{method} converges at step {convergence_step}",
                    annotation_position="top right"
                )
        
        # Add main line
        fig.add_trace(
            go.Scatter(
                x=adapt_steps,
                y=latencies,
                mode='lines+markers',
                name=method,
                line=dict(color=colors[i % len(colors)], width=3),
                marker=dict(size=6),
                hovertemplate=f"Method: {method}<br>Adapt Step: %{{x}}<br>Latency: %{{y:.3f}}s<extra></extra>"
            )
        )
    
    # Update layout
    fig.update_layout(
        title=title or f"Convergence Analysis (Threshold: {convergence_threshold})",
        xaxis_title="Adapt Step",
        yaxis_title="Latency (s)",
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=PLOTLY_THEME["layout"]["font"],
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig


def create_improvement_figure(
    records: List[EpisodeRecord],
    baseline_method: str = "heft",
    title: Optional[str] = None,
    methods: Optional[List[str]] = None
) -> go.Figure:
    """Create improvement over baseline figure."""
    if not records:
        return go.Figure()
    
    # Filter by methods if specified
    if methods:
        records = [r for r in records if r.method in methods]
    
    # Group by episode and method
    episodes = {}
    for record in records:
        if record.episode_id not in episodes:
            episodes[record.episode_id] = {}
        episodes[record.episode_id][record.method] = {
            'latency': record.latency_total,
            'adapt_step': record.adapt_step or 0
        }
    
    # Calculate improvements
    improvement_data = {}
    for episode_id, method_latencies in episodes.items():
        if baseline_method in method_latencies:
            baseline_latency = method_latencies[baseline_method]['latency']
            baseline_adapt_step = method_latencies[baseline_method]['adapt_step']
            
            for method, data in method_latencies.items():
                if method != baseline_method:
                    if method not in improvement_data:
                        improvement_data[method] = []
                    
                    improvement = (baseline_latency - data['latency']) / baseline_latency * 100
                    improvement_data[method].append({
                        'adapt_step': data['adapt_step'],
                        'improvement': improvement
                    })
    
    # Create figure
    fig = go.Figure()
    
    # Color palette for methods
    colors = [COLORS["LOCAL"], COLORS["EDGE"], COLORS["UPLINK"], COLORS["DOWNLINK"]]
    
    for i, (method, data) in enumerate(improvement_data.items()):
        if not data:
            continue
        
        # Sort by adapt_step
        data.sort(key=lambda x: x['adapt_step'])
        adapt_steps = [d['adapt_step'] for d in data]
        improvements = [d['improvement'] for d in data]
        
        # Add line
        fig.add_trace(
            go.Scatter(
                x=adapt_steps,
                y=improvements,
                mode='lines+markers',
                name=method,
                line=dict(color=colors[i % len(colors)], width=3),
                marker=dict(size=6),
                hovertemplate=f"Method: {method}<br>Adapt Step: %{{x}}<br>Improvement: %{{y:.2f}}%<extra></extra>"
            )
        )
    
    # Add zero line
    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color=COLORS["TEXT"],
        opacity=0.5,
        annotation_text="Baseline performance",
        annotation_position="right"
    )
    
    # Update layout
    fig.update_layout(
        title=title or f"Improvement over {baseline_method.title()} Baseline",
        xaxis_title="Adapt Step",
        yaxis_title="Improvement (%)",
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=PLOTLY_THEME["layout"]["font"],
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig


def save_adaptation_figure(
    records: List[EpisodeRecord],
    output_path: str,
    title: Optional[str] = None,
    methods: Optional[List[str]] = None,
    formats: List[str] = ["png", "svg"],
    include_other_plots: bool = True
) -> None:
    """Save adaptation figure and related plots in specified formats."""
    # Save main adaptation plot
    adapt_fig = create_adaptation_figure(records, title, methods)
    
    for fmt in formats:
        if fmt == "png":
            adapt_fig.write_image(f"{output_path}.png", width=800, height=600, scale=2)
        elif fmt == "svg":
            adapt_fig.write_image(f"{output_path}.svg", width=800, height=600)
        elif fmt == "html":
            adapt_fig.write_html(f"{output_path}.html")
    
    if include_other_plots:
        # Save learning curve
        learning_fig = create_learning_curve_figure(records, f"{title} - Learning Curve" if title else "Learning Curve", methods)
        
        for fmt in formats:
            if fmt == "png":
                learning_fig.write_image(f"{output_path}_learning.png", width=800, height=600, scale=2)
            elif fmt == "svg":
                learning_fig.write_image(f"{output_path}_learning.svg", width=800, height=600)
            elif fmt == "html":
                learning_fig.write_html(f"{output_path}_learning.html")
        
        # Save convergence analysis
        conv_fig = create_convergence_figure(records, f"{title} - Convergence" if title else "Convergence Analysis", methods)
        
        for fmt in formats:
            if fmt == "png":
                conv_fig.write_image(f"{output_path}_convergence.png", width=800, height=600, scale=2)
            elif fmt == "svg":
                conv_fig.write_image(f"{output_path}_convergence.svg", width=800, height=600)
            elif fmt == "html":
                conv_fig.write_html(f"{output_path}_convergence.html")
        
        # Save improvement plot
        imp_fig = create_improvement_figure(records, title=f"{title} - Improvement" if title else "Improvement Analysis", methods=methods)
        
        for fmt in formats:
            if fmt == "png":
                imp_fig.write_image(f"{output_path}_improvement.png", width=800, height=600, scale=2)
            elif fmt == "svg":
                imp_fig.write_image(f"{output_path}_improvement.svg", width=800, height=600)
            elif fmt == "html":
                imp_fig.write_html(f"{output_path}_improvement.html")
    
    print(f"Adaptation figures saved: {output_path}.{{{','.join(formats)}}}")


def create_adaptation_summary_figure(
    records: List[EpisodeRecord],
    title: Optional[str] = None,
    methods: Optional[List[str]] = None
) -> go.Figure:
    """Create adaptation summary figure with multiple subplots."""
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
            "Adaptation Curve",
            "Learning Curve",
            "Convergence Analysis",
            "Improvement over Baseline"
        ],
        specs=[
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "scatter"}, {"type": "scatter"}]
        ]
    )
    
    # Add adaptation curve
    adapt_fig = create_adaptation_figure(records, methods=methods)
    for trace in adapt_fig.data:
        fig.add_trace(trace, row=1, col=1)
    
    # Add learning curve
    learning_fig = create_learning_curve_figure(records, methods=methods)
    for trace in learning_fig.data:
        fig.add_trace(trace, row=1, col=2)
    
    # Add convergence analysis
    conv_fig = create_convergence_figure(records, methods=methods)
    for trace in conv_fig.data:
        fig.add_trace(trace, row=2, col=1)
    
    # Add improvement plot
    imp_fig = create_improvement_figure(records, methods=methods)
    for trace in imp_fig.data:
        fig.add_trace(trace, row=2, col=2)
    
    # Update layout
    fig.update_layout(
        title=title or "Adaptation Analysis Summary",
        height=800,
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=PLOTLY_THEME["layout"]["font"]
    )
    
    return fig


