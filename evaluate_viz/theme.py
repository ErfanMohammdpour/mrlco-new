"""
Theme configuration for visualizations.
"""

from typing import Dict, Any
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# Color palette
COLORS = {
    "LOCAL": "#22c55e",      # Green
    "EDGE": "#3b82f6",       # Blue
    "UPLINK": "#f59e0b",     # Orange
    "DOWNLINK": "#a855f7",   # Purple
    "IDLE": "#9ca3af",       # Gray
    "ACTIVE": "#ef4444",     # Red
    "COMPLETED": "#10b981",  # Emerald
    "PENDING": "#6b7280",    # Slate
    "BACKGROUND": "#f8fafc", # Light gray
    "TEXT": "#1f2937",       # Dark gray
    "GRID": "#e5e7eb",       # Light gray
}

# Font configuration
FONTS = {
    "title": {"size": 16, "weight": "bold"},
    "subtitle": {"size": 14, "weight": "semibold"},
    "label": {"size": 12, "weight": "normal"},
    "tick": {"size": 10, "weight": "normal"},
    "annotation": {"size": 11, "weight": "normal"},
}

# Layout configuration
LAYOUT = {
    "figure_size": (12, 8),
    "dpi": 100,
    "padding": 0.1,
    "margin": 0.05,
    "animation_fps": 30,
    "animation_duration": 20,  # max seconds
}

# Animation configuration
ANIMATION = {
    "node_glow_alpha": 0.8,
    "edge_pulse_alpha": 0.6,
    "transition_duration": 0.3,  # seconds
    "intro_duration": 0.5,       # seconds
    "outro_duration": 0.5,       # seconds
}

# Plotly theme
PLOTLY_THEME = {
    "layout": {
        "font": {"family": "Arial, sans-serif", "size": 12, "color": COLORS["TEXT"]},
        "plot_bgcolor": "white",
        "paper_bgcolor": "white",
        "margin": {"l": 50, "r": 50, "t": 50, "b": 50},
        "showlegend": True,
        "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    },
    "colorscale": "Viridis",
}

# Matplotlib style configuration
def configure_matplotlib_style():
    """Configure matplotlib global style."""
    plt.style.use('default')
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.titlesize': 14,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.color': COLORS["GRID"],
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.edgecolor': COLORS["GRID"],
        'axes.linewidth': 0.8,
    })


def get_node_color(action: str, state: str = "normal") -> str:
    """Get color for a node based on action and state."""
    if state == "completed":
        return COLORS["COMPLETED"]
    elif state == "active":
        return COLORS["ACTIVE"]
    elif state == "pending":
        return COLORS["PENDING"]
    else:
        return COLORS.get(action.upper(), COLORS["IDLE"])


def get_edge_color(active: bool = False) -> str:
    """Get color for an edge based on activity."""
    if active:
        return COLORS["ACTIVE"]
    return COLORS["IDLE"]


def get_lane_color(lane: str) -> str:
    """Get color for a timeline lane."""
    return COLORS.get(lane.upper(), COLORS["IDLE"])


# Initialize matplotlib style
configure_matplotlib_style()

