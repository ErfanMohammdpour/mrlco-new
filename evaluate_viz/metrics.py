"""
Metrics calculation and statistical analysis utilities.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from scipy import stats
import warnings

from .schema import EpisodeRecord


def calculate_latency_stats(records: List[EpisodeRecord]) -> Dict[str, Dict[str, float]]:
    """Calculate latency statistics by method."""
    df = pd.DataFrame([
        {
            'method': record.method,
            'latency': record.latency_total,
            'episode_id': record.episode_id
        }
        for record in records
    ])
    
    stats_by_method = {}
    for method in df['method'].unique():
        method_data = df[df['method'] == method]['latency']
        stats_by_method[method] = {
            'mean': method_data.mean(),
            'median': method_data.median(),
            'std': method_data.std(),
            'min': method_data.min(),
            'max': method_data.max(),
            'count': len(method_data)
        }
    
    return stats_by_method


def calculate_improvement_vs_baseline(
    records: List[EpisodeRecord],
    baseline_method: str = "heft"
) -> Dict[str, Dict[str, float]]:
    """Calculate improvement percentages vs baseline method."""
    # Group by episode_id to compare methods
    episodes = {}
    for record in records:
        if record.episode_id not in episodes:
            episodes[record.episode_id] = {}
        episodes[record.episode_id][record.method] = record.latency_total
    
    improvements = {}
    for episode_id, method_latencies in episodes.items():
        if baseline_method in method_latencies:
            baseline_latency = method_latencies[baseline_method]
            for method, latency in method_latencies.items():
                if method != baseline_method:
                    improvement = (baseline_latency - latency) / baseline_latency * 100
                    if method not in improvements:
                        improvements[method] = []
                    improvements[method].append(improvement)
    
    # Calculate statistics for each method
    result = {}
    for method, improvements_list in improvements.items():
        if improvements_list:
            result[method] = {
                'mean_improvement': np.mean(improvements_list),
                'median_improvement': np.median(improvements_list),
                'std_improvement': np.std(improvements_list),
                'min_improvement': np.min(improvements_list),
                'max_improvement': np.max(improvements_list),
                'count': len(improvements_list)
            }
    
    return result


def calculate_confidence_intervals(
    data: List[float],
    confidence: float = 0.95,
    n_bootstrap: int = 1000
) -> Tuple[float, float]:
    """Calculate confidence intervals using bootstrap."""
    if len(data) < 2:
        return (data[0] if data else 0, data[0] if data else 0)
    
    # Bootstrap sampling
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_means.append(np.mean(sample))
    
    # Calculate percentiles
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_means, 100 * alpha / 2)
    upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
    
    return lower, upper


def calculate_adaptation_curve(records: List[EpisodeRecord]) -> Dict[str, Dict]:
    """Calculate adaptation curve data (latency vs adapt_step)."""
    # Filter records with adapt_step
    adaptation_records = [r for r in records if r.adapt_step is not None]
    
    if not adaptation_records:
        return {}
    
    df = pd.DataFrame([
        {
            'method': record.method,
            'adapt_step': record.adapt_step,
            'latency': record.latency_total,
            'episode_id': record.episode_id
        }
        for record in adaptation_records
    ])
    
    adaptation_data = {}
    for method in df['method'].unique():
        method_data = df[df['method'] == method]
        
        # Group by adapt_step and calculate statistics
        grouped = method_data.groupby('adapt_step')['latency'].agg([
            'mean', 'std', 'count'
        ]).reset_index()
        
        # Calculate confidence intervals
        ci_lower = []
        ci_upper = []
        
        for _, row in grouped.iterrows():
            if row['count'] > 1:
                # Get all latencies for this adapt_step
                step_latencies = method_data[method_data['adapt_step'] == row['adapt_step']]['latency'].tolist()
                lower, upper = calculate_confidence_intervals(step_latencies)
                ci_lower.append(lower)
                ci_upper.append(upper)
            else:
                ci_lower.append(row['mean'])
                ci_upper.append(row['mean'])
        
        adaptation_data[method] = {
            'adapt_steps': grouped['adapt_step'].tolist(),
            'mean_latency': grouped['mean'].tolist(),
            'std_latency': grouped['std'].tolist(),
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'count': grouped['count'].tolist()
        }
    
    return adaptation_data


def calculate_pareto_frontier(
    records: List[EpisodeRecord],
    x_metric: str = "latency_total",
    y_metric: str = "energy_ue"
) -> Dict[str, List[Tuple[float, float]]]:
    """Calculate Pareto frontier for given metrics."""
    # Filter records that have both metrics
    valid_records = [
        r for r in records 
        if hasattr(r, x_metric) and hasattr(r, y_metric) 
        and getattr(r, x_metric) is not None and getattr(r, y_metric) is not None
    ]
    
    if not valid_records:
        return {}
    
    pareto_data = {}
    for method in set(r.method for r in valid_records):
        method_records = [r for r in valid_records if r.method == method]
        points = [(getattr(r, x_metric), getattr(r, y_metric)) for r in method_records]
        
        # Calculate Pareto frontier (minimize both metrics)
        pareto_points = []
        for i, (x1, y1) in enumerate(points):
            is_pareto = True
            for j, (x2, y2) in enumerate(points):
                if i != j and x2 <= x1 and y2 <= y1 and (x2 < x1 or y2 < y1):
                    is_pareto = False
                    break
            if is_pareto:
                pareto_points.append((x1, y1))
        
        # Sort by x-axis
        pareto_points.sort(key=lambda p: p[0])
        pareto_data[method] = pareto_points
    
    return pareto_data


def calculate_resource_utilization(record: EpisodeRecord) -> Dict[str, List[Tuple[float, float]]]:
    """Calculate resource utilization over time for an episode."""
    # This is a simplified implementation
    # In practice, you'd need to consider resource capacities and scheduling
    
    utilization = {
        'ue': [],
        'edge': [],
        'uplink': [],
        'downlink': []
    }
    
    # Sort decisions by finish times to get chronological order
    sorted_decisions = sorted(record.decisions, key=lambda d: min(d.finish_times.values()) if d.finish_times else 0)
    
    for decision in sorted_decisions:
        if not decision.finish_times:
            continue
        
        # Calculate utilization for each resource
        for resource, finish_time in decision.finish_times.items():
            if resource in utilization:
                # Simplified: assume 100% utilization during task execution
                start_time = finish_time - (getattr(decision, f't_{resource}', 0) or 0)
                if start_time >= 0:
                    utilization[resource].append((start_time, finish_time))
    
    return utilization


def calculate_task_dependencies(record: EpisodeRecord) -> Dict[int, List[int]]:
    """Calculate task dependencies from DAG structure."""
    edges = record.get_edges()
    dependencies = {}
    
    # Initialize with empty lists
    for node in record.get_nodes():
        dependencies[node.id] = []
    
    # Add dependencies from edges
    for edge in edges:
        if len(edge) >= 2:
            parent, child = edge[0], edge[1]
            if child in dependencies:
                dependencies[child].append(parent)
    
    return dependencies


def find_execution_order(record: EpisodeRecord) -> List[int]:
    """Find valid execution order respecting dependencies."""
    dependencies = calculate_task_dependencies(record)
    visited = set()
    execution_order = []
    
    def visit(node_id):
        if node_id in visited:
            return
        visited.add(node_id)
        
        # Visit dependencies first
        for dep in dependencies.get(node_id, []):
            visit(dep)
        
        execution_order.append(node_id)
    
    # Visit all nodes
    for node in record.get_nodes():
        visit(node.id)
    
    return execution_order

