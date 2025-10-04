"""
Pydantic models for episode data and evaluation metrics.
"""

from typing import List, Dict, Literal, Optional, Any
from pydantic import BaseModel, Field


Action = Literal["LOCAL", "EDGE"]


class NodeSpec(BaseModel):
    """Specification for a DAG node."""
    id: int
    cpu_cycles: float
    up_size: float
    down_size: float


class Decision(BaseModel):
    """Decision made for a specific node."""
    node: int
    action: Action
    # Durations for chosen path (some may be None for unused path)
    t_local: Optional[float] = None
    t_net_up: Optional[float] = None
    t_edge: Optional[float] = None
    t_net_down: Optional[float] = None
    # Absolute finish times per resource lane
    finish_times: Dict[str, float] = Field(
        default_factory=dict,
        description="Keys: uplink, edge, downlink, ue with float timestamps"
    )


class EpisodeRecord(BaseModel):
    """Complete episode record with all evaluation data."""
    episode_id: int
    method: str = "ours"
    dag: Dict[str, List[Any]] = Field(
        description="DAG structure: {nodes: [NodeSpec...], edges: [[u,v],...]}"
    )
    decisions: List[Decision]
    latency_total: float
    rates: Dict[str, float] = Field(
        default_factory=dict,
        description="Data rates: {uplink: ..., downlink: ...}"
    )
    adapt_step: Optional[int] = None
    energy_ue: Optional[float] = None
    comm_cost: Optional[float] = None
    baselines: Dict[str, float] = Field(
        default_factory=dict,
        description="Baseline latencies: {heft: ..., greedy: ..., drl_ft: ...}"
    )
    oracle_latency: Optional[float] = None

    def get_nodes(self) -> List[NodeSpec]:
        """Extract NodeSpec objects from dag structure."""
        nodes_data = self.dag.get("nodes", [])
        return [NodeSpec(**node) if isinstance(node, dict) else node for node in nodes_data]

    def get_edges(self) -> List[List[int]]:
        """Extract edge list from dag structure."""
        return self.dag.get("edges", [])

    def get_decision_for_node(self, node_id: int) -> Optional[Decision]:
        """Get decision for a specific node."""
        for decision in self.decisions:
            if decision.node == node_id:
                return decision
        return None

