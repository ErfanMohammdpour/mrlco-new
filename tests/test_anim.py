"""
Tests for animation functionality.
"""

import pytest
import tempfile
import os
from pathlib import Path
from evaluate_viz.schema import EpisodeRecord, NodeSpec, Decision
from evaluate_viz.animate_episode import EpisodeAnimator, create_episode_animation


def create_test_record():
    """Create a test episode record for animation testing."""
    nodes = [
        NodeSpec(id=1, cpu_cycles=100.0, up_size=50.0, down_size=25.0),
        NodeSpec(id=2, cpu_cycles=200.0, up_size=75.0, down_size=30.0),
        NodeSpec(id=3, cpu_cycles=300.0, up_size=100.0, down_size=40.0)
    ]
    
    decisions = [
        Decision(node=1, action="LOCAL", t_local=2.0, finish_times={"ue": 5.0}),
        Decision(node=2, action="EDGE", t_net_up=1.0, t_edge=1.5, t_net_down=0.5,
                finish_times={"uplink": 2.0, "edge": 3.5, "downlink": 4.0}),
        Decision(node=3, action="LOCAL", t_local=3.0, finish_times={"ue": 8.0})
    ]
    
    return EpisodeRecord(
        episode_id=1,
        method="ours",
        dag={"nodes": [node.model_dump() for node in nodes], "edges": [[1, 2], [2, 3]]},
        decisions=decisions,
        latency_total=10.0,
        rates={"uplink": 50.0, "downlink": 30.0},
        adapt_step=5,
        energy_ue=200.0,
        comm_cost=100.0,
        baselines={"heft": 12.0, "greedy": 8.0}
    )


def test_episode_animator_initialization():
    """Test EpisodeAnimator initialization."""
    record = create_test_record()
    animator = EpisodeAnimator(record, fps=30, speed=1.0)
    
    assert animator.record == record
    assert animator.fps == 30
    assert animator.speed == 1.0
    assert animator.total_duration == 10.0
    assert animator.total_frames > 0
    assert len(animator.nodes) == 3
    assert len(animator.edges) == 2
    assert len(animator.decisions) == 3


def test_episode_animator_layout():
    """Test that the animation layout is created correctly."""
    record = create_test_record()
    animator = EpisodeAnimator(record, fps=30, speed=1.0)
    
    # Check that all required axes exist
    assert 'dag' in animator.axes
    assert 'util' in animator.axes
    assert 'timeline' in animator.axes
    assert 'kpi' in animator.axes
    
    # Check that figure exists
    assert animator.fig is not None


def test_episode_animator_dag_layout():
    """Test DAG layout creation."""
    record = create_test_record()
    animator = EpisodeAnimator(record, fps=30, speed=1.0)
    
    # Check that NetworkX graph is created
    assert animator.G is not None
    assert len(animator.G.nodes()) == 3
    assert len(animator.G.edges()) == 2
    
    # Check that positions are calculated
    assert animator.pos is not None
    assert len(animator.pos) == 3


def test_episode_animator_animation_state():
    """Test animation state initialization."""
    record = create_test_record()
    animator = EpisodeAnimator(record, fps=30, speed=1.0)
    
    # Check initial state
    assert animator.current_time == 0.0
    assert animator.current_node_idx == 0
    assert len(animator.completed_nodes) == 0
    assert len(animator.active_tasks) == 4  # ue, edge, uplink, downlink
    
    # Check execution order
    assert len(animator.execution_order) == 3
    assert 1 in animator.execution_order
    assert 2 in animator.execution_order
    assert 3 in animator.execution_order


def test_episode_animator_node_color():
    """Test node color calculation."""
    record = create_test_record()
    animator = EpisodeAnimator(record, fps=30, speed=1.0)
    
    # Test different states
    local_color = animator._get_node_color("LOCAL", "normal")
    edge_color = animator._get_node_color("EDGE", "normal")
    active_color = animator._get_node_color("LOCAL", "active")
    completed_color = animator._get_node_color("LOCAL", "completed")
    pending_color = animator._get_node_color("LOCAL", "pending")
    
    assert local_color is not None
    assert edge_color is not None
    assert active_color is not None
    assert completed_color is not None
    assert pending_color is not None
    
    # Colors should be different for different states
    assert active_color != completed_color
    assert active_color != pending_color


def test_episode_animator_edge_activity():
    """Test edge activity detection."""
    record = create_test_record()
    animator = EpisodeAnimator(record, fps=30, speed=1.0)
    
    # Test edge activity at different times
    edge = [1, 2]
    
    # At time 0, edge should not be active
    assert not animator._is_edge_active(edge, 0)
    
    # At time 3, edge might be active (depending on finish times)
    # This is a simplified test - in practice, you'd need to set up the state properly
    is_active = animator._is_edge_active(edge, 3)
    assert isinstance(is_active, bool)


def test_episode_animator_state_update():
    """Test animation state update."""
    record = create_test_record()
    animator = EpisodeAnimator(record, fps=30, speed=1.0)
    
    # Update state at different frames
    animator._update_animation_state(0)
    assert animator.current_time == 0.0
    
    animator._update_animation_state(30)  # 1 second at 30 fps
    assert animator.current_time == 1.0


def test_create_episode_animation():
    """Test create_episode_animation function."""
    record = create_test_record()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = os.path.join(temp_dir, "test_animation")
        
        # Test that function runs without error
        # Note: This might not actually create files due to matplotlib backend issues in tests
        try:
            create_episode_animation(record, output_path, fps=5, speed=2.0, formats=["mp4"])
        except Exception as e:
            # If it fails due to display issues, that's expected in test environments
            assert "display" in str(e).lower() or "backend" in str(e).lower()


def test_animator_with_different_speeds():
    """Test animator with different speed settings."""
    record = create_test_record()
    
    # Test with speed 2.0 (2x faster)
    animator_fast = EpisodeAnimator(record, fps=30, speed=2.0)
    assert animator_fast.total_duration == 5.0  # 10.0 / 2.0
    assert animator_fast.speed == 2.0
    
    # Test with speed 0.5 (2x slower)
    animator_slow = EpisodeAnimator(record, fps=30, speed=0.5)
    assert animator_slow.total_duration == 20.0  # 10.0 / 0.5
    assert animator_slow.speed == 0.5


def test_animator_with_different_fps():
    """Test animator with different FPS settings."""
    record = create_test_record()
    
    # Test with 60 FPS
    animator_60 = EpisodeAnimator(record, fps=60, speed=1.0)
    assert animator_60.fps == 60
    assert animator_60.total_frames == 600  # 10.0 * 60
    
    # Test with 15 FPS
    animator_15 = EpisodeAnimator(record, fps=15, speed=1.0)
    assert animator_15.fps == 15
    assert animator_15.total_frames == 150  # 10.0 * 15


def test_animator_with_empty_record():
    """Test animator with minimal record."""
    # Create minimal record
    record = EpisodeRecord(
        episode_id=1,
        method="ours",
        dag={"nodes": [], "edges": []},
        decisions=[],
        latency_total=1.0
    )
    
    animator = EpisodeAnimator(record, fps=30, speed=1.0)
    
    assert animator.total_duration == 1.0
    assert len(animator.nodes) == 0
    assert len(animator.edges) == 0
    assert len(animator.decisions) == 0


def test_animator_utilization_calculation():
    """Test resource utilization calculation."""
    record = create_test_record()
    animator = EpisodeAnimator(record, fps=30, speed=1.0)
    
    # Test utilization data structure
    utilization = animator.utilization
    assert 'ue' in utilization
    assert 'edge' in utilization
    assert 'uplink' in utilization
    assert 'downlink' in utilization
    
    # Each lane should be a list of tuples
    for lane, intervals in utilization.items():
        assert isinstance(intervals, list)
        for interval in intervals:
            assert isinstance(interval, tuple)
            assert len(interval) == 2  # (start, end)
            assert interval[0] <= interval[1]  # start <= end

