"""
CDF (Cumulative Distribution Function) visualization for latency comparison.
"""

import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from scipy import stats

from .schema import EpisodeRecord
from .theme import COLORS, PLOTLY_THEME


def create_cdf_figure(
    records: List[EpisodeRecord],
    title: Optional[str] = None,
    methods: Optional[List[str]] = None
) -> go.Figure:
    """Create CDF figure for latency comparison across methods."""
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
        method_data[record.method].append(record.latency_total)
    
    # Create figure
    fig = go.Figure()
    
    # Color palette for methods
    colors = [COLORS["LOCAL"], COLORS["EDGE"], COLORS["UPLINK"], COLORS["DOWNLINK"]]
    
    for i, (method, latencies) in enumerate(method_data.items()):
        if not latencies:
            continue
        
        # Sort latencies for CDF
        sorted_latencies = np.sort(latencies)
        n = len(sorted_latencies)
        
        # Calculate CDF values
        cdf_values = np.arange(1, n + 1) / n
        
        # Add CDF trace
        fig.add_trace(
            go.Scatter(
                x=sorted_latencies,
                y=cdf_values,
                mode='lines',
                name=method,
                line=dict(
                    color=colors[i % len(colors)],
                    width=3
                ),
                hovertemplate=f"Method: {method}<br>Latency: %{{x:.2f}}s<br>CDF: %{{y:.3f}}<extra></extra>"
            )
        )
        
        # Add median line
        median_latency = np.median(sorted_latencies)
        fig.add_vline(
            x=median_latency,
            line_dash="dash",
            line_color=colors[i % len(colors)],
            opacity=0.7,
            annotation_text=f"{method} median: {median_latency:.2f}s",
            annotation_position="top right"
        )
    
    # Update layout
    fig.update_layout(
        title=title or "Latency CDF Comparison",
        xaxis_title="Latency (s)",
        yaxis_title="Cumulative Probability",
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
            zerolinecolor=COLORS["GRID"],
            range=[0, 1.05]
        )
    )
    
    return fig


def create_box_plot_figure(
    records: List[EpisodeRecord],
    title: Optional[str] = None,
    methods: Optional[List[str]] = None
) -> go.Figure:
    """Create box plot figure for latency distribution comparison."""
    if not records:
        return go.Figure()
    
    # Filter by methods if specified
    if methods:
        records = [r for r in records if r.method in methods]
    
    # Prepare data for box plot
    data = []
    for record in records:
        data.append({
            'method': record.method,
            'latency': record.latency_total
        })
    
    df = pd.DataFrame(data)
    
    # Create box plot
    fig = px.box(
        df,
        x='method',
        y='latency',
        title=title or "Latency Distribution by Method",
        color='method',
        color_discrete_sequence=[COLORS["LOCAL"], COLORS["EDGE"], COLORS["UPLINK"], COLORS["DOWNLINK"]]
    )
    
    # Update layout
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=PLOTLY_THEME["layout"]["font"],
        xaxis_title="Method",
        yaxis_title="Latency (s)",
        showlegend=False
    )
    
    return fig


def create_histogram_figure(
    records: List[EpisodeRecord],
    title: Optional[str] = None,
    methods: Optional[List[str]] = None,
    bins: int = 20
) -> go.Figure:
    """Create histogram figure for latency distribution."""
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
        method_data[record.method].append(record.latency_total)
    
    # Create figure
    fig = go.Figure()
    
    # Color palette for methods
    colors = [COLORS["LOCAL"], COLORS["EDGE"], COLORS["UPLINK"], COLORS["DOWNLINK"]]
    
    for i, (method, latencies) in enumerate(method_data.items()):
        if not latencies:
            continue
        
        # Create histogram
        fig.add_trace(
            go.Histogram(
                x=latencies,
                name=method,
                opacity=0.7,
                marker_color=colors[i % len(colors)],
                nbinsx=bins,
                hovertemplate=f"Method: {method}<br>Latency: %{{x:.2f}}s<br>Count: %{{y}}<extra></extra>"
            )
        )
    
    # Update layout
    fig.update_layout(
        title=title or "Latency Distribution Histogram",
        xaxis_title="Latency (s)",
        yaxis_title="Count",
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=PLOTLY_THEME["layout"]["font"],
        barmode='overlay',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig


def create_statistics_table_figure(
    records: List[EpisodeRecord],
    title: Optional[str] = None,
    methods: Optional[List[str]] = None
) -> go.Figure:
    """Create statistics table figure."""
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
        method_data[record.method].append(record.latency_total)
    
    # Calculate statistics
    stats_data = []
    for method, latencies in method_data.items():
        if not latencies:
            continue
        
        latencies = np.array(latencies)
        stats_data.append({
            'Method': method,
            'Count': len(latencies),
            'Mean': f"{np.mean(latencies):.3f}",
            'Median': f"{np.median(latencies):.3f}",
            'Std': f"{np.std(latencies):.3f}",
            'Min': f"{np.min(latencies):.3f}",
            'Max': f"{np.max(latencies):.3f}",
            'Q25': f"{np.percentile(latencies, 25):.3f}",
            'Q75': f"{np.percentile(latencies, 75):.3f}"
        })
    
    # Create table
    df = pd.DataFrame(stats_data)
    
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=list(df.columns),
            fill_color=COLORS["BACKGROUND"],
            font=dict(color=COLORS["TEXT"], size=12),
            align="left"
        ),
        cells=dict(
            values=[df[col] for col in df.columns],
            fill_color='white',
            font=dict(color=COLORS["TEXT"], size=11),
            align="left"
        )
    )])
    
    fig.update_layout(
        title=title or "Latency Statistics by Method",
        font=PLOTLY_THEME["layout"]["font"],
        height=400
    )
    
    return fig


def save_cdf_figure(
    records: List[EpisodeRecord],
    output_path: str,
    title: Optional[str] = None,
    methods: Optional[List[str]] = None,
    formats: List[str] = ["png", "svg"],
    include_other_plots: bool = True
) -> None:
    """Save CDF figure and related plots in specified formats."""
    # Save CDF plot
    cdf_fig = create_cdf_figure(records, title, methods)
    
    for fmt in formats:
        if fmt == "png":
            cdf_fig.write_image(f"{output_path}.png", width=800, height=600, scale=2)
        elif fmt == "svg":
            cdf_fig.write_image(f"{output_path}.svg", width=800, height=600)
        elif fmt == "html":
            cdf_fig.write_html(f"{output_path}.html")
    
    if include_other_plots:
        # Save box plot
        box_fig = create_box_plot_figure(records, f"{title} - Box Plot" if title else "Latency Box Plot", methods)
        
        for fmt in formats:
            if fmt == "png":
                box_fig.write_image(f"{output_path}_box.png", width=800, height=600, scale=2)
            elif fmt == "svg":
                box_fig.write_image(f"{output_path}_box.svg", width=800, height=600)
            elif fmt == "html":
                box_fig.write_html(f"{output_path}_box.html")
        
        # Save histogram
        hist_fig = create_histogram_figure(records, f"{title} - Histogram" if title else "Latency Histogram", methods)
        
        for fmt in formats:
            if fmt == "png":
                hist_fig.write_image(f"{output_path}_hist.png", width=800, height=600, scale=2)
            elif fmt == "svg":
                hist_fig.write_image(f"{output_path}_hist.svg", width=800, height=600)
            elif fmt == "html":
                hist_fig.write_html(f"{output_path}_hist.html")
        
        # Save statistics table
        stats_fig = create_statistics_table_figure(records, f"{title} - Statistics" if title else "Latency Statistics", methods)
        
        for fmt in formats:
            if fmt == "png":
                stats_fig.write_image(f"{output_path}_stats.png", width=1000, height=400, scale=2)
            elif fmt == "svg":
                stats_fig.write_image(f"{output_path}_stats.svg", width=1000, height=400)
            elif fmt == "html":
                stats_fig.write_html(f"{output_path}_stats.html")
    
    print(f"CDF figures saved: {output_path}.{{{','.join(formats)}}}")


def create_percentile_comparison_figure(
    records: List[EpisodeRecord],
    title: Optional[str] = None,
    methods: Optional[List[str]] = None,
    percentiles: List[float] = [50, 75, 90, 95, 99]
) -> go.Figure:
    """Create percentile comparison figure."""
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
        method_data[record.method].append(record.latency_total)
    
    # Calculate percentiles
    percentile_data = []
    for method, latencies in method_data.items():
        if not latencies:
            continue
        
        latencies = np.array(latencies)
        method_percentiles = []
        for p in percentiles:
            method_percentiles.append(np.percentile(latencies, p))
        
        percentile_data.append({
            'method': method,
            'percentiles': method_percentiles
        })
    
    # Create figure
    fig = go.Figure()
    
    colors = [COLORS["LOCAL"], COLORS["EDGE"], COLORS["UPLINK"], COLORS["DOWNLINK"]]
    
    for i, data in enumerate(percentile_data):
        fig.add_trace(
            go.Scatter(
                x=percentiles,
                y=data['percentiles'],
                mode='lines+markers',
                name=data['method'],
                line=dict(color=colors[i % len(colors)], width=3),
                marker=dict(size=8),
                hovertemplate=f"Method: {data['method']}<br>Percentile: %{{x}}%<br>Latency: %{{y:.3f}}s<extra></extra>"
            )
        )
    
    fig.update_layout(
        title=title or "Latency Percentile Comparison",
        xaxis_title="Percentile",
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


