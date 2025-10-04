"""
Cinematic animation system for episode visualization.
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch
import numpy as np
import networkx as nx
from typing import List, Dict, Tuple, Optional
import imageio
from pathlib import Path

from .schema import EpisodeRecord, Decision
from .theme import COLORS, LAYOUT, ANIMATION, configure_matplotlib_style
from .metrics import find_execution_order, calculate_resource_utilization


class EpisodeAnimator:
    """Cinematic animator for episode visualization."""
    
    def __init__(self, record: EpisodeRecord, fps: int = 30, speed: float = 1.0):
        self.record = record
        self.fps = fps
        self.speed = speed
        self.total_duration = record.latency_total / speed
        
        # Calculate frame parameters
        self.total_frames = int(self.total_duration * fps)
        self.frame_duration = 1.0 / fps
        
        # Initialize matplotlib
        configure_matplotlib_style()
        
        # Create figure and subplots
        self.fig, self.axes = self._create_layout()
        
        # Initialize animation data
        self._init_animation_data()
        
        # Create NetworkX graph for DAG layout
        self._create_dag_layout()
    
    def _create_layout(self):
        """Create the 4-panel layout for the animation."""
        fig = plt.figure(figsize=(16, 10))
        
        # Create subplots
        # Left 60%: DAG panel
        ax_dag = plt.subplot2grid((3, 5), (0, 0), rowspan=3, colspan=3)
        
        # Right top: Utilization bars
        ax_util = plt.subplot2grid((3, 5), (0, 3), colspan=2)
        
        # Right middle: Timeline strip
        ax_timeline = plt.subplot2grid((3, 5), (1, 3), colspan=2)
        
        # Right bottom: KPI overlay
        ax_kpi = plt.subplot2grid((3, 5), (2, 3), colspan=2)
        
        return fig, {
            'dag': ax_dag,
            'util': ax_util,
            'timeline': ax_timeline,
            'kpi': ax_kpi
        }
    
    def _init_animation_data(self):
        """Initialize data structures for animation."""
        self.nodes = self.record.get_nodes()
        self.edges = self.record.get_edges()
        self.decisions = {d.node: d for d in self.record.decisions}
        
        # Find execution order
        self.execution_order = find_execution_order(self.record)
        
        # Calculate resource utilization
        self.utilization = calculate_resource_utilization(self.record)
        
        # Animation state
        self.current_time = 0.0
        self.current_node_idx = 0
        self.completed_nodes = set()
        self.active_tasks = {}  # {lane: [(start, end, node_id), ...]}
        
        # Initialize active tasks
        for lane in ['ue', 'edge', 'uplink', 'downlink']:
            self.active_tasks[lane] = []
    
    def _create_dag_layout(self):
        """Create NetworkX layout for DAG visualization."""
        self.G = nx.DiGraph()
        
        # Add nodes
        for node in self.nodes:
            self.G.add_node(node.id, **node.model_dump())
        
        # Add edges
        for edge in self.edges:
            if len(edge) >= 2:
                self.G.add_edge(edge[0], edge[1])
        
        # Use spring layout
        self.pos = nx.spring_layout(self.G, k=3, iterations=50)
    
    def _draw_dag(self, frame):
        """Draw the DAG with current state."""
        ax = self.axes['dag']
        ax.clear()
        
        # Draw edges
        for edge in self.edges:
            if len(edge) >= 2:
                x0, y0 = self.pos[edge[0]]
                x1, y1 = self.pos[edge[1]]
                
                # Check if edge is active (data transfer)
                is_active = self._is_edge_active(edge, frame)
                color = COLORS["ACTIVE"] if is_active else COLORS["IDLE"]
                width = 3 if is_active else 1
                
                ax.plot([x0, x1], [y0, y1], color=color, linewidth=width, alpha=0.7)
        
        # Draw nodes
        for node in self.nodes:
            x, y = self.pos[node.id]
            
            # Determine node state
            if node.id in self.completed_nodes:
                state = "completed"
            elif node.id == self.execution_order[self.current_node_idx] if self.current_node_idx < len(self.execution_order) else None:
                state = "active"
            else:
                state = "pending"
            
            # Get decision
            decision = self.decisions.get(node.id)
            action = decision.action if decision else "UNKNOWN"
            
            # Draw node
            color = self._get_node_color(action, state)
            size = 300 if state == "active" else 200
            
            circle = Circle((x, y), 0.1, color=color, alpha=0.8)
            ax.add_patch(circle)
            
            # Add node ID
            ax.text(x, y, str(node.id), ha='center', va='center', 
                   fontsize=10, color='white', weight='bold')
            
            # Add action label
            if decision:
                ax.text(x, y-0.15, action, ha='center', va='center',
                       fontsize=8, color=COLORS["TEXT"], weight='bold')
        
        # Set title and formatting
        ax.set_title(f"Episode {self.record.episode_id} - {self.record.method}", 
                    fontsize=14, weight='bold')
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect('equal')
        ax.axis('off')
    
    def _draw_utilization(self, frame):
        """Draw resource utilization bars."""
        ax = self.axes['util']
        ax.clear()
        
        # Calculate current utilization
        current_time = frame * self.frame_duration * self.speed
        
        util_data = {}
        for lane in ['ue', 'edge', 'uplink', 'downlink']:
            active_count = sum(1 for start, end in self.active_tasks[lane] 
                             if start <= current_time <= end)
            util_data[lane] = min(active_count, 1)  # Cap at 1 for simplicity
        
        # Draw utilization bars
        lanes = ['UE CPU', 'Edge CPU', 'Uplink', 'Downlink']
        colors = [COLORS["LOCAL"], COLORS["EDGE"], COLORS["UPLINK"], COLORS["DOWNLINK"]]
        
        y_pos = np.arange(len(lanes))
        bars = ax.barh(y_pos, [util_data[lane] for lane in ['ue', 'edge', 'uplink', 'downlink']], 
                      color=colors, alpha=0.7)
        
        # Add labels
        ax.set_yticks(y_pos)
        ax.set_yticklabels(lanes)
        ax.set_xlabel('Utilization')
        ax.set_title('Resource Utilization')
        ax.set_xlim(0, 1)
        
        # Add current time
        ax.text(0.5, -0.5, f"Time: {current_time:.2f}s", 
               ha='center', va='top', fontsize=10, weight='bold')
    
    def _draw_timeline(self, frame):
        """Draw the 4-lane timeline with moving cursor."""
        ax = self.axes['timeline']
        ax.clear()
        
        current_time = frame * self.frame_duration * self.speed
        
        # Draw timeline lanes
        lanes = ['Uplink', 'Edge', 'Downlink', 'UE']
        lane_colors = [COLORS["UPLINK"], COLORS["EDGE"], COLORS["DOWNLINK"], COLORS["LOCAL"]]
        
        for i, (lane, color) in enumerate(zip(lanes, lane_colors)):
            y_pos = i * 0.2
            
            # Draw lane background
            ax.barh(y_pos, self.total_duration, height=0.15, 
                   color=color, alpha=0.2, edgecolor=color)
            
            # Draw completed tasks
            for start, end, node_id in self.active_tasks[lane.lower()]:
                if end <= current_time:
                    ax.barh(y_pos, end - start, height=0.15, left=start,
                           color=color, alpha=0.8, edgecolor='white')
                    ax.text(start + (end - start) / 2, y_pos, str(node_id),
                           ha='center', va='center', fontsize=8, color='white')
            
            # Draw active tasks
            for start, end, node_id in self.active_tasks[lane.lower()]:
                if start <= current_time <= end:
                    ax.barh(y_pos, current_time - start, height=0.15, left=start,
                           color=color, alpha=1.0, edgecolor='white')
                    ax.text(start + (current_time - start) / 2, y_pos, str(node_id),
                           ha='center', va='center', fontsize=8, color='white')
        
        # Draw time cursor
        ax.axvline(x=current_time, color=COLORS["ACTIVE"], linewidth=2, alpha=0.8)
        
        # Formatting
        ax.set_yticks([i * 0.2 for i in range(len(lanes))])
        ax.set_yticklabels(lanes)
        ax.set_xlabel('Time (s)')
        ax.set_title('Execution Timeline')
        ax.set_xlim(0, self.total_duration)
        ax.set_ylim(-0.1, 0.8)
    
    def _draw_kpi(self, frame):
        """Draw KPI overlay."""
        ax = self.axes['kpi']
        ax.clear()
        
        current_time = frame * self.frame_duration * self.speed
        
        # Calculate KPIs
        local_count = sum(1 for d in self.record.decisions if d.action == "LOCAL")
        edge_count = sum(1 for d in self.record.decisions if d.action == "EDGE")
        remaining_time = max(0, self.total_duration - current_time)
        
        # Get current rates
        uplink_rate = self.record.rates.get('uplink', 0)
        downlink_rate = self.record.rates.get('downlink', 0)
        
        # Display KPIs
        kpi_text = f"""
        Elapsed: {current_time:.2f}s
        Remaining: {remaining_time:.2f}s
        LOCAL: {local_count}
        EDGE: {edge_count}
        Uplink Rate: {uplink_rate:.1f} Mbps
        Downlink Rate: {downlink_rate:.1f} Mbps
        """
        
        ax.text(0.1, 0.8, kpi_text, transform=ax.transAxes, 
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle="round,pad=0.3", facecolor=COLORS["BACKGROUND"], alpha=0.8))
        
        # Add progress bar
        progress = min(current_time / self.total_duration, 1.0)
        ax.barh(0.1, progress, height=0.05, color=COLORS["ACTIVE"], alpha=0.7)
        ax.barh(0.1, 1.0, height=0.05, color=COLORS["GRID"], alpha=0.3)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('Live KPIs', fontsize=12, weight='bold')
    
    def _is_edge_active(self, edge, frame):
        """Check if an edge is currently active (data transfer)."""
        if len(edge) < 2:
            return False
        
        source, target = edge[0], edge[1]
        current_time = frame * self.frame_duration * self.speed
        
        # Check if source is completed and target is active
        if source in self.completed_nodes:
            decision = self.decisions.get(source)
            if decision and decision.action == "EDGE":
                # Check uplink and downlink phases
                for lane in ['uplink', 'downlink']:
                    if lane in decision.finish_times:
                        start_time = decision.finish_times[lane] - (getattr(decision, f't_net_{lane.split("link")[0]}', 0) or 0)
                        end_time = decision.finish_times[lane]
                        if start_time <= current_time <= end_time:
                            return True
        
        return False
    
    def _get_node_color(self, action, state):
        """Get color for a node based on action and state."""
        if state == "completed":
            return COLORS["COMPLETED"]
        elif state == "active":
            return COLORS["ACTIVE"]
        elif state == "pending":
            return COLORS["PENDING"]
        else:
            return COLORS.get(action, COLORS["IDLE"])
    
    def _update_animation_state(self, frame):
        """Update animation state for current frame."""
        current_time = frame * self.frame_duration * self.speed
        
        # Update completed nodes
        for decision in self.record.decisions:
            if not decision.finish_times:
                continue
            
            # Check if all phases are complete
            max_finish_time = max(decision.finish_times.values())
            if max_finish_time <= current_time:
                self.completed_nodes.add(decision.node)
        
        # Update active tasks
        for lane in ['ue', 'edge', 'uplink', 'downlink']:
            self.active_tasks[lane] = []
            
            for decision in self.record.decisions:
                if lane in decision.finish_times:
                    finish_time = decision.finish_times[lane]
                    duration = getattr(decision, f't_{lane}', 0) or 0
                    if duration > 0:
                        start_time = finish_time - duration
                        self.active_tasks[lane].append((start_time, finish_time, decision.node))
        
        # Update current node
        if self.current_node_idx < len(self.execution_order):
            current_node = self.execution_order[self.current_node_idx]
            if current_node in self.completed_nodes:
                self.current_node_idx += 1
    
    def animate(self, frame):
        """Main animation function."""
        # Update state
        self._update_animation_state(frame)
        
        # Draw all panels
        self._draw_dag(frame)
        self._draw_utilization(frame)
        self._draw_timeline(frame)
        self._draw_kpi(frame)
        
        # Add intro/outro effects
        if frame < self.fps * ANIMATION["intro_duration"]:
            # Intro effect
            self.fig.suptitle(f"Episode {self.record.episode_id} - {self.record.method}", 
                            fontsize=16, weight='bold', alpha=frame / (self.fps * ANIMATION["intro_duration"]))
        elif frame >= self.total_frames - self.fps * ANIMATION["outro_duration"]:
            # Outro effect - show summary
            self._draw_summary(frame)
    
    def _draw_summary(self, frame):
        """Draw summary overlay for outro."""
        # Calculate improvements vs baselines
        improvements = {}
        for baseline_name, baseline_latency in self.record.baselines.items():
            improvement = (baseline_latency - self.record.latency_total) / baseline_latency * 100
            improvements[baseline_name] = improvement
        
        # Create summary text
        summary_text = f"""
        Episode {self.record.episode_id} Complete!
        
        Total Latency: {self.record.latency_total:.2f}s
        
        Improvements vs Baselines:
        """
        
        for baseline, improvement in improvements.items():
            summary_text += f"  {baseline}: {improvement:+.1f}%\n"
        
        # Add summary overlay
        ax = self.axes['kpi']
        ax.text(0.5, 0.5, summary_text, transform=ax.transAxes, 
               ha='center', va='center', fontsize=12, weight='bold',
               bbox=dict(boxstyle="round,pad=0.5", facecolor=COLORS["BACKGROUND"], alpha=0.9))
    
    def create_animation(self, output_path: str, format: str = "mp4"):
        """Create and save the animation."""
        # Create animation
        anim = animation.FuncAnimation(
            self.fig, self.animate, frames=self.total_frames,
            interval=1000/self.fps, blit=False, repeat=False
        )
        
        # Save animation
        if format == "mp4":
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=self.fps, metadata=dict(artist='MRLCO'), bitrate=1800)
            anim.save(f"{output_path}.mp4", writer=writer)
        elif format == "gif":
            # Save as GIF using imageio
            frames = []
            for frame in range(0, self.total_frames, max(1, self.total_frames // 100)):  # Limit to 100 frames for GIF
                self.animate(frame)
                self.fig.canvas.draw()
                frame_data = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
                frame_data = frame_data.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
                frames.append(frame_data)
            
            imageio.mimsave(f"{output_path}.gif", frames, fps=self.fps//2)  # Half FPS for GIF
        
        plt.close(self.fig)
        print(f"Animation saved: {output_path}.{format}")


def create_episode_animation(
    record: EpisodeRecord,
    output_path: str,
    fps: int = 30,
    speed: float = 1.0,
    formats: List[str] = ["mp4", "gif"]
) -> None:
    """Create cinematic animation for an episode."""
    animator = EpisodeAnimator(record, fps, speed)
    
    for format in formats:
        animator.create_animation(output_path, format)
    
    print(f"Episode animation saved: {output_path}.{{{','.join(formats)}}}")


def create_comparison_animation(
    records: List[EpisodeRecord],
    output_path: str,
    fps: int = 30,
    speed: float = 1.0,
    formats: List[str] = ["mp4", "gif"]
) -> None:
    """Create comparison animation for multiple episodes."""
    # This would create a side-by-side comparison
    # For now, create separate animations
    for i, record in enumerate(records):
        episode_output = f"{output_path}_ep{record.episode_id}_{record.method}"
        create_episode_animation(record, episode_output, fps, speed, formats)
    
    print(f"Comparison animations saved: {output_path}_ep*")


