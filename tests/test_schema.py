"""
Tests for schema validation and data models.
"""

import pytest
import json
from evaluate_viz.schema import EpisodeRecord, NodeSpec, Decision, Action


def test_node_spec_creation():
    """Test NodeSpec model creation."""
    node = NodeSpec(
        id=1,
        cpu_cycles=100.0,
        up_size=50.0,
        down_size=25.0
    )
    
    assert node.id == 1
    assert node.cpu_cycles == 100.0
    assert node.up_size == 50.0
    assert node.down_size == 25.0


def test_decision_creation():
    """Test Decision model creation."""
    decision = Decision(
        node=1,
        action="LOCAL",
        t_local=2.0,
        finish_times={"ue": 5.0}
    )
    
    assert decision.node == 1
    assert decision.action == "LOCAL"
    assert decision.t_local == 2.0
    assert decision.finish_times == {"ue": 5.0}


def test_episode_record_creation():
    """Test EpisodeRecord model creation."""
    nodes = [
        NodeSpec(id=1, cpu_cycles=100.0, up_size=50.0, down_size=25.0),
        NodeSpec(id=2, cpu_cycles=200.0, up_size=75.0, down_size=30.0)
    ]
    
    decisions = [
        Decision(node=1, action="LOCAL", t_local=2.0, finish_times={"ue": 5.0}),
        Decision(node=2, action="EDGE", t_net_up=1.0, t_edge=1.5, t_net_down=0.5,
                finish_times={"uplink": 2.0, "edge": 3.5, "downlink": 4.0})
    ]
    
    record = EpisodeRecord(
        episode_id=1,
        method="ours",
        dag={"nodes": [node.model_dump() for node in nodes], "edges": [[1, 2]]},
        decisions=decisions,
        latency_total=10.0,
        rates={"uplink": 50.0, "downlink": 30.0},
        adapt_step=5,
        energy_ue=200.0,
        comm_cost=100.0,
        baselines={"heft": 12.0, "greedy": 8.0}
    )
    
    assert record.episode_id == 1
    assert record.method == "ours"
    assert len(record.get_nodes()) == 2
    assert len(record.get_edges()) == 1
    assert len(record.decisions) == 2
    assert record.latency_total == 10.0


def test_episode_record_serialization():
    """Test EpisodeRecord serialization to/from JSON."""
    nodes = [
        NodeSpec(id=1, cpu_cycles=100.0, up_size=50.0, down_size=25.0)
    ]
    
    decisions = [
        Decision(node=1, action="LOCAL", t_local=2.0, finish_times={"ue": 5.0})
    ]
    
    record = EpisodeRecord(
        episode_id=1,
        method="ours",
        dag={"nodes": [node.model_dump() for node in nodes], "edges": []},
        decisions=decisions,
        latency_total=10.0,
        rates={"uplink": 50.0, "downlink": 30.0}
    )
    
    # Serialize to dict
    data = record.model_dump()
    assert isinstance(data, dict)
    assert data["episode_id"] == 1
    
    # Deserialize from dict
    record2 = EpisodeRecord(**data)
    assert record2.episode_id == record.episode_id
    assert record2.method == record.method
    assert record2.latency_total == record.latency_total


def test_action_validation():
    """Test Action literal validation."""
    # Valid actions
    decision1 = Decision(node=1, action="LOCAL", finish_times={})
    decision2 = Decision(node=2, action="EDGE", finish_times={})
    
    assert decision1.action == "LOCAL"
    assert decision2.action == "EDGE"
    
    # Invalid action should raise validation error
    with pytest.raises(ValueError):
        Decision(node=1, action="INVALID", finish_times={})


def test_optional_fields():
    """Test optional fields handling."""
    # Minimal record
    record = EpisodeRecord(
        episode_id=1,
        method="ours",
        dag={"nodes": [], "edges": []},
        decisions=[],
        latency_total=10.0
    )
    
    assert record.adapt_step is None
    assert record.energy_ue is None
    assert record.comm_cost is None
    assert record.baselines == {}
    assert record.oracle_latency is None


def test_get_decision_for_node():
    """Test get_decision_for_node method."""
    decisions = [
        Decision(node=1, action="LOCAL", finish_times={}),
        Decision(node=2, action="EDGE", finish_times={})
    ]
    
    record = EpisodeRecord(
        episode_id=1,
        method="ours",
        dag={"nodes": [], "edges": []},
        decisions=decisions,
        latency_total=10.0
    )
    
    decision1 = record.get_decision_for_node(1)
    decision2 = record.get_decision_for_node(2)
    decision3 = record.get_decision_for_node(3)
    
    assert decision1 is not None
    assert decision1.action == "LOCAL"
    assert decision2 is not None
    assert decision2.action == "EDGE"
    assert decision3 is None


def test_dag_structure_access():
    """Test DAG structure access methods."""
    nodes = [
        NodeSpec(id=1, cpu_cycles=100.0, up_size=50.0, down_size=25.0),
        NodeSpec(id=2, cpu_cycles=200.0, up_size=75.0, down_size=30.0)
    ]
    
    record = EpisodeRecord(
        episode_id=1,
        method="ours",
        dag={"nodes": [node.model_dump() for node in nodes], "edges": [[1, 2]]},
        decisions=[],
        latency_total=10.0
    )
    
    retrieved_nodes = record.get_nodes()
    retrieved_edges = record.get_edges()
    
    assert len(retrieved_nodes) == 2
    assert len(retrieved_edges) == 1
    assert retrieved_edges[0] == [1, 2]
    
    # Check that retrieved nodes are NodeSpec objects
    assert all(isinstance(node, NodeSpec) for node in retrieved_nodes)

