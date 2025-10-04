"""
DAG visualization using Plotly.
"""

import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
from typing import List, Dict, Tuple, Optional
import numpy as np

from .schema import EpisodeRecord, Decision
from .theme import COLORS, PLOTLY_THEME


def create_dag_figure(
    record: EpisodeRecord,
    title: Optional[str] = None,
    show_edge_weights: bool = True
) -> go.Figure:
    """Create a DAG visualization figure."""
    nodes = record.get_nodes()
    edges = record.get_edges()
    
    # Create NetworkX graph
    G = nx.DiGraph()
    
    # Add nodes
    for node in nodes:
        G.add_node(node.id, **node.model_dump())
    
    # Add edges
    for edge in edges:
        if len(edge) >= 2:
            G.add_edge(edge[0], edge[1])
    
    # Use spring layout for positioning
    pos = nx.spring_layout(G, k=3, iterations=50)
    
    # Extract node positions
    node_x = [pos[node.id][0] for node in nodes]
    node_y = [pos[node.id][1] for node in nodes]
    
    # Prepare node data
    node_text = []
    node_colors = []
    node_sizes = []
    
    for node in nodes:
        decision = record.get_decision_for_node(node.id)
        action = decision.action if decision else "UNKNOWN"
        
        # Node text with hover information
        text = f"Node {node.id}<br>"
        text += f"CPU: {node.cpu_cycles:.1f}<br>"
        text += f"Up: {node.up_size:.1f}<br>"
        text += f"Down: {node.down_size:.1f}<br>"
        text += f"Action: {action}"
        
        if decision and decision.t_local is not None:
            text += f"<br>Local: {decision.t_local:.2f}s"
        if decision and decision.t_net_up is not None:
            text += f"<br>Up: {decision.t_net_up:.2f}s"
        if decision and decision.t_edge is not None:
            text += f"<br>Edge: {decision.t_edge:.2f}s"
        if decision and decision.t_net_down is not None:
            text += f"<br>Down: {decision.t_net_down:.2f}s"
        
        node_text.append(text)
        node_colors.append(COLORS.get(action, COLORS["IDLE"]))
        
        # Size based on CPU cycles
        size = max(20, min(50, node.cpu_cycles / 100))
        node_sizes.append(size)
    
    # Create node trace
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(width=2, color='white'),
            opacity=0.8
        ),
        text=[f"Node {node.id}" for node in nodes],
        textposition="middle center",
        textfont=dict(size=10, color="white"),
        hovertemplate="%{customdata}<extra></extra>",
        customdata=node_text,
        name="Nodes"
    )
    
    # Prepare edge data
    edge_x = []
    edge_y = []
    edge_info = []
    
    for edge in edges:
        if len(edge) >= 2:
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
            # Calculate edge weight based on data transfer
            source_node = next((n for n in nodes if n.id == edge[0]), None)
            if source_node:
                weight = source_node.down_size
                edge_info.append(f"Data: {weight:.1f}")
            else:
                edge_info.append("Data: Unknown")
    
    # Create edge trace
    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode='lines',
        line=dict(width=2, color=COLORS["IDLE"]),
        hoverinfo='none',
        showlegend=False
    )
    
    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace])
    
    # Update layout
    fig.update_layout(
        title=title or f"DAG - Episode {record.episode_id} ({record.method})",
        titlefont=PLOTLY_THEME["layout"]["font"],
        showlegend=True,
        hovermode='closest',
        margin=dict(b=20,l=5,r=5,t=40),
        annotations=[
            dict(
                text="Green: LOCAL, Blue: EDGE",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.005, y=-0.002,
                xanchor="left", yanchor="bottom",
                font=dict(size=12, color=COLORS["TEXT"])
            )
        ],
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    return fig


def save_dag_figure(
    record: EpisodeRecord,
    output_path: str,
    title: Optional[str] = None,
    formats: List[str] = ["png", "svg"]
) -> None:
    """Save DAG figure in specified formats."""
    fig = create_dag_figure(record, title)
    
    for fmt in formats:
        if fmt == "png":
            fig.write_image(f"{output_path}.png", width=800, height=600, scale=2)
        elif fmt == "svg":
            fig.write_image(f"{output_path}.svg", width=800, height=600)
        elif fmt == "html":
            fig.write_html(f"{output_path}.html")
    
    print(f"DAG figure saved: {output_path}.{{{','.join(formats)}}}")


def create_comparison_dag_figure(
    records: List[EpisodeRecord],
    title: Optional[str] = None
) -> go.Figure:
    """Create a comparison DAG figure showing multiple methods."""
    if not records:
        return go.Figure()
    
    # Use the first record's DAG structure
    base_record = records[0]
    nodes = base_record.get_nodes()
    edges = base_record.get_edges()
    
    # Create subplots
    from plotly.subplots import make_subplots
    
    n_methods = len(records)
    cols = min(3, n_methods)
    rows = (n_methods + cols - 1) // cols
    
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=[f"{record.method} (Episode {record.episode_id})" for record in records],
        specs=[[{"type": "scatter"} for _ in range(cols)] for _ in range(rows)]
    )
    
    # Create NetworkX graph for layout
    G = nx.DiGraph()
    for node in nodes:
        G.add_node(node.id)
    for edge in edges:
        if len(edge) >= 2:
            G.add_edge(edge[0], edge[1])
    
    pos = nx.spring_layout(G, k=3, iterations=50)
    
    for i, record in enumerate(records):
        row = i // cols + 1
        col = i % cols + 1
        
        # Create DAG for this method
        dag_fig = create_dag_figure(record)
        
        # Add traces to subplot
        for trace in dag_fig.data:
            fig.add_trace(trace, row=row, col=col)
    
    # Update layout
    fig.update_layout(
        title=title or "DAG Comparison",
        titlefont=PLOTLY_THEME["layout"]["font"],
        showlegend=False,
        height=400 * rows,
        width=600 * cols
    )
    
    # Remove axis labels
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False, row=i, col=j)
            fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False, row=i, col=j)
    
    return fig


