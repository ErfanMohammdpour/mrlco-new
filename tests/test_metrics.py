"""
Tests for metrics calculation functions.
"""

import pytest
import numpy as np
from evaluate_viz.schema import EpisodeRecord, NodeSpec, Decision
from evaluate_viz.metrics import (
    calculate_latency_stats,
    calculate_improvement_vs_baseline,
    calculate_confidence_intervals,
    calculate_adaptation_curve,
    calculate_pareto_frontier,
    find_execution_order
)


def create_sample_record(episode_id, method, latency, adapt_step=None):
    """Create a sample episode record for testing."""
    nodes = [
        NodeSpec(id=1, cpu_cycles=100.0, up_size=50.0, down_size=25.0),
        NodeSpec(id=2, cpu_cycles=200.0, up_size=75.0, down_size=30.0)
    ]
    
    decisions = [
        Decision(node=1, action="LOCAL", t_local=2.0, finish_times={"ue": 5.0}),
        Decision(node=2, action="EDGE", t_net_up=1.0, t_edge=1.5, t_net_down=0.5,
                finish_times={"uplink": 2.0, "edge": 3.5, "downlink": 4.0})
    ]
    
    return EpisodeRecord(
        episode_id=episode_id,
        method=method,
        dag={"nodes": [node.model_dump() for node in nodes], "edges": [[1, 2]]},
        decisions=decisions,
        latency_total=latency,
        rates={"uplink": 50.0, "downlink": 30.0},
        adapt_step=adapt_step,
        energy_ue=200.0,
        comm_cost=100.0,
        baselines={"heft": 12.0, "greedy": 8.0}
    )


def test_calculate_latency_stats():
    """Test latency statistics calculation."""
    records = [
        create_sample_record(1, "ours", 10.0),
        create_sample_record(2, "ours", 12.0),
        create_sample_record(1, "heft", 15.0),
        create_sample_record(2, "heft", 18.0)
    ]
    
    stats = calculate_latency_stats(records)
    
    assert "ours" in stats
    assert "heft" in stats
    
    # Check ours stats
    ours_stats = stats["ours"]
    assert ours_stats["count"] == 2
    assert ours_stats["mean"] == 11.0
    assert ours_stats["min"] == 10.0
    assert ours_stats["max"] == 12.0
    
    # Check heft stats
    heft_stats = stats["heft"]
    assert heft_stats["count"] == 2
    assert heft_stats["mean"] == 16.5


def test_calculate_improvement_vs_baseline():
    """Test improvement calculation vs baseline."""
    records = [
        create_sample_record(1, "ours", 10.0),
        create_sample_record(1, "heft", 15.0),
        create_sample_record(2, "ours", 8.0),
        create_sample_record(2, "heft", 12.0)
    ]
    
    improvements = calculate_improvement_vs_baseline(records, "heft")
    
    assert "ours" in improvements
    ours_improvements = improvements["ours"]
    
    # Episode 1: (15-10)/15 = 33.33%
    # Episode 2: (12-8)/12 = 33.33%
    assert len(ours_improvements["mean_improvement"]) > 0
    assert abs(ours_improvements["mean_improvement"] - 33.33) < 1.0


def test_calculate_confidence_intervals():
    """Test confidence interval calculation."""
    data = [1, 2, 3, 4, 5]
    lower, upper = calculate_confidence_intervals(data, confidence=0.95)
    
    assert lower <= upper
    assert lower <= np.mean(data)
    assert upper >= np.mean(data)
    
    # Test with single value
    single_data = [5.0]
    lower_single, upper_single = calculate_confidence_intervals(single_data)
    assert lower_single == upper_single == 5.0


def test_calculate_adaptation_curve():
    """Test adaptation curve calculation."""
    records = [
        create_sample_record(1, "ours", 10.0, adapt_step=0),
        create_sample_record(2, "ours", 8.0, adapt_step=10),
        create_sample_record(3, "ours", 6.0, adapt_step=20),
        create_sample_record(1, "heft", 15.0, adapt_step=0),
        create_sample_record(2, "heft", 15.0, adapt_step=10),
        create_sample_record(3, "heft", 15.0, adapt_step=20)
    ]
    
    adaptation_data = calculate_adaptation_curve(records)
    
    assert "ours" in adaptation_data
    assert "heft" in adaptation_data
    
    ours_data = adaptation_data["ours"]
    assert len(ours_data["adapt_steps"]) == 3
    assert 0 in ours_data["adapt_steps"]
    assert 10 in ours_data["adapt_steps"]
    assert 20 in ours_data["adapt_steps"]


def test_calculate_pareto_frontier():
    """Test Pareto frontier calculation."""
    records = [
        create_sample_record(1, "ours", 10.0),
        create_sample_record(2, "ours", 8.0),
        create_sample_record(1, "heft", 15.0),
        create_sample_record(2, "heft", 12.0)
    ]
    
    # Test latency vs energy
    pareto_data = calculate_pareto_frontier(records, "latency_total", "energy_ue")
    
    assert "ours" in pareto_data
    assert "heft" in pareto_data
    
    # Check that Pareto points are non-dominated
    ours_points = pareto_data["ours"]
    for i, (x1, y1) in enumerate(ours_points):
        for j, (x2, y2) in enumerate(ours_points):
            if i != j:
                # No point should dominate another
                assert not (x2 <= x1 and y2 <= y1 and (x2 < x1 or y2 < y1))


def test_find_execution_order():
    """Test execution order finding."""
    # Create a simple DAG: 1 -> 2 -> 3
    nodes = [
        NodeSpec(id=1, cpu_cycles=100.0, up_size=50.0, down_size=25.0),
        NodeSpec(id=2, cpu_cycles=200.0, up_size=75.0, down_size=30.0),
        NodeSpec(id=3, cpu_cycles=300.0, up_size=100.0, down_size=40.0)
    ]
    
    record = EpisodeRecord(
        episode_id=1,
        method="ours",
        dag={"nodes": [node.model_dump() for node in nodes], "edges": [[1, 2], [2, 3]]},
        decisions=[],
        latency_total=10.0
    )
    
    execution_order = find_execution_order(record)
    
    # Should respect dependencies: 1 before 2, 2 before 3
    assert execution_order.index(1) < execution_order.index(2)
    assert execution_order.index(2) < execution_order.index(3)
    assert len(execution_order) == 3


def test_empty_records():
    """Test functions with empty record lists."""
    empty_records = []
    
    stats = calculate_latency_stats(empty_records)
    assert stats == {}
    
    improvements = calculate_improvement_vs_baseline(empty_records)
    assert improvements == {}
    
    adaptation_data = calculate_adaptation_curve(empty_records)
    assert adaptation_data == {}
    
    pareto_data = calculate_pareto_frontier(empty_records)
    assert pareto_data == {}


def test_single_record():
    """Test functions with single record."""
    records = [create_sample_record(1, "ours", 10.0)]
    
    stats = calculate_latency_stats(records)
    assert "ours" in stats
    assert stats["ours"]["count"] == 1
    assert stats["ours"]["mean"] == 10.0
    
    # Single record should not have improvements
    improvements = calculate_improvement_vs_baseline(records)
    assert improvements == {}


def test_missing_metrics():
    """Test functions with missing metrics."""
    # Create record without energy_ue
    nodes = [NodeSpec(id=1, cpu_cycles=100.0, up_size=50.0, down_size=25.0)]
    record = EpisodeRecord(
        episode_id=1,
        method="ours",
        dag={"nodes": [node.model_dump() for node in nodes], "edges": []},
        decisions=[],
        latency_total=10.0
        # No energy_ue field
    )
    
    records = [record]
    
    # Should handle missing metrics gracefully
    pareto_data = calculate_pareto_frontier(records, "latency_total", "energy_ue")
    assert pareto_data == {}

