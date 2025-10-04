"""
I/O utilities for reading and processing episode data.
"""

import json
from typing import List, Dict, Any, Optional
import pandas as pd
from pathlib import Path

from .schema import EpisodeRecord, NodeSpec, Decision


def read_jsonl(file_path: str) -> List[EpisodeRecord]:
    """Read episode records from JSONL file."""
    records = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                # Convert dag nodes to NodeSpec objects
                if 'dag' in data and 'nodes' in data['dag']:
                    nodes = []
                    for node_data in data['dag']['nodes']:
                        if isinstance(node_data, dict):
                            nodes.append(NodeSpec(**node_data))
                        else:
                            nodes.append(node_data)
                    data['dag']['nodes'] = nodes
                
                # Convert decisions to Decision objects
                if 'decisions' in data:
                    decisions = []
                    for decision_data in data['decisions']:
                        if isinstance(decision_data, dict):
                            decisions.append(Decision(**decision_data))
                        else:
                            decisions.append(decision_data)
                    data['decisions'] = decisions
                
                records.append(EpisodeRecord(**data))
            except Exception as e:
                print(f"Warning: Failed to parse line {line_num}: {e}")
                continue
    return records


def write_jsonl(records: List[EpisodeRecord], file_path: str) -> None:
    """Write episode records to JSONL file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        for record in records:
            # Convert to dict and handle Pydantic models
            data = record.model_dump()
            f.write(json.dumps(data) + '\n')


def to_dataframe(records: List[EpisodeRecord]) -> pd.DataFrame:
    """Convert episode records to flattened DataFrame."""
    data = []
    
    for record in records:
        row = {
            'episode_id': record.episode_id,
            'method': record.method,
            'latency_total': record.latency_total,
            'adapt_step': record.adapt_step,
            'energy_ue': record.energy_ue,
            'comm_cost': record.comm_cost,
            'oracle_latency': record.oracle_latency,
        }
        
        # Add rates
        for key, value in record.rates.items():
            row[f'rate_{key}'] = value
        
        # Add baselines
        for key, value in record.baselines.items():
            row[f'baseline_{key}'] = value
        
        # Add DAG metrics
        nodes = record.get_nodes()
        edges = record.get_edges()
        row['num_nodes'] = len(nodes)
        row['num_edges'] = len(edges)
        
        # Count decisions by type
        local_count = sum(1 for d in record.decisions if d.action == "LOCAL")
        edge_count = sum(1 for d in record.decisions if d.action == "EDGE")
        row['num_local'] = local_count
        row['num_edge'] = edge_count
        row['local_ratio'] = local_count / len(record.decisions) if record.decisions else 0
        
        # Add total CPU cycles and data sizes
        total_cpu = sum(node.cpu_cycles for node in nodes)
        total_up_size = sum(node.up_size for node in nodes)
        total_down_size = sum(node.down_size for node in nodes)
        row['total_cpu_cycles'] = total_cpu
        row['total_up_size'] = total_up_size
        row['total_down_size'] = total_down_size
        
        data.append(row)
    
    return pd.DataFrame(data)


def compute_finish_times_from_durations(record: EpisodeRecord) -> EpisodeRecord:
    """Compute finish_times from durations if missing."""
    # This is a simplified implementation - in practice, you'd need
    # to consider dependencies and resource constraints
    for decision in record.decisions:
        if not decision.finish_times:
            current_time = 0.0
            finish_times = {}
            
            # Compute finish times based on action
            if decision.action == "LOCAL":
                if decision.t_local is not None:
                    finish_times["ue"] = current_time + decision.t_local
            else:  # EDGE
                if decision.t_net_up is not None:
                    finish_times["uplink"] = current_time + decision.t_net_up
                    current_time += decision.t_net_up
                
                if decision.t_edge is not None:
                    finish_times["edge"] = current_time + decision.t_edge
                    current_time += decision.t_edge
                
                if decision.t_net_down is not None:
                    finish_times["downlink"] = current_time + decision.t_net_down
            
            decision.finish_times = finish_times
    
    return record


def filter_episodes(
    records: List[EpisodeRecord],
    episode_ids: Optional[List[int]] = None,
    methods: Optional[List[str]] = None
) -> List[EpisodeRecord]:
    """Filter episodes by IDs and methods."""
    filtered = records
    
    if episode_ids is not None:
        filtered = [r for r in filtered if r.episode_id in episode_ids]
    
    if methods is not None:
        filtered = [r for r in filtered if r.method in methods]
    
    return filtered


def ensure_output_dirs(base_path: str) -> Dict[str, Path]:
    """Ensure output directories exist."""
    base = Path(base_path)
    dirs = {
        'figures': base / 'figures',
        'videos': base / 'videos',
    }
    
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return dirs

