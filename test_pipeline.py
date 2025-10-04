#!/usr/bin/env python3
"""
Test script to verify the visualization pipeline works correctly.
"""

import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluate_viz.io_utils import read_jsonl, ensure_output_dirs
from evaluate_viz.dag_figure import create_dag_figure
from evaluate_viz.gantt_figure import create_gantt_figure
from evaluate_viz.cdf_figure import create_cdf_figure
from evaluate_viz.adapt_figure import create_adaptation_figure
from evaluate_viz.frontier_figure import create_pareto_figure


def test_pipeline():
    """Test the complete visualization pipeline."""
    print("🧪 Testing visualization pipeline...")
    
    # Test data reading
    print("📖 Testing data reading...")
    try:
        records = read_jsonl("data/sample_eval.jsonl")
        print(f"✅ Loaded {len(records)} records")
    except Exception as e:
        print(f"❌ Error reading data: {e}")
        return False
    
    # Test output directory creation
    print("📁 Testing output directory creation...")
    try:
        ensure_output_dirs("test_reports")
        print("✅ Output directories created")
    except Exception as e:
        print(f"❌ Error creating directories: {e}")
        return False
    
    # Test figure generation
    print("🎨 Testing figure generation...")
    
    try:
        # Test DAG figure
        if records:
            dag_fig = create_dag_figure(records[0])
            print("✅ DAG figure created")
            
            # Test Gantt figure
            gantt_fig = create_gantt_figure(records[0])
            print("✅ Gantt figure created")
            
            # Test CDF figure
            cdf_fig = create_cdf_figure(records)
            print("✅ CDF figure created")
            
            # Test adaptation figure
            adapt_fig = create_adaptation_figure(records)
            print("✅ Adaptation figure created")
            
            # Test Pareto figure
            pareto_fig = create_pareto_figure(records)
            print("✅ Pareto figure created")
        
    except Exception as e:
        print(f"❌ Error creating figures: {e}")
        return False
    
    # Test metrics calculation
    print("📊 Testing metrics calculation...")
    try:
        from evaluate_viz.metrics import calculate_latency_stats, calculate_improvement_vs_baseline
        
        stats = calculate_latency_stats(records)
        print(f"✅ Latency stats calculated for {len(stats)} methods")
        
        improvements = calculate_improvement_vs_baseline(records)
        print(f"✅ Improvements calculated for {len(improvements)} methods")
        
    except Exception as e:
        print(f"❌ Error calculating metrics: {e}")
        return False
    
    print("🎉 All tests passed! Pipeline is working correctly.")
    return True


if __name__ == "__main__":
    success = test_pipeline()
    sys.exit(0 if success else 1)

