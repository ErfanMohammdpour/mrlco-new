#!/usr/bin/env python3
"""
Test script to verify the visualization integration works correctly.
"""

import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all required modules can be imported."""
    print("🧪 Testing imports...")
    
    try:
        from viz_integration import VisualizationCollector
        print("✅ viz_integration imported successfully")
    except Exception as e:
        print(f"❌ Error importing viz_integration: {e}")
        return False
    
    try:
        from evaluate_viz.schema import EpisodeRecord, NodeSpec, Decision
        print("✅ evaluate_viz.schema imported successfully")
    except Exception as e:
        print(f"❌ Error importing evaluate_viz.schema: {e}")
        return False
    
    try:
        from evaluate_viz.io_utils import write_jsonl, ensure_output_dirs
        print("✅ evaluate_viz.io_utils imported successfully")
    except Exception as e:
        print(f"❌ Error importing evaluate_viz.io_utils: {e}")
        return False
    
    return True

def test_visualization_collector():
    """Test VisualizationCollector functionality."""
    print("\n🧪 Testing VisualizationCollector...")
    
    try:
        from viz_integration import VisualizationCollector
        from evaluate_viz.schema import EpisodeRecord, NodeSpec, Decision
        
        # Create a mock environment
        class MockEnv:
            def __init__(self):
                self.resource_cluster = MockResourceCluster()
                self.task_graphs_batchs = [MockTaskGraph()]
        
        class MockResourceCluster:
            def __init__(self):
                self.bandwidth_up = 7.0
                self.bandwidth_dl = 7.0
            
            def locally_execution_cost(self, cpu_cycles):
                return cpu_cycles / 1000.0
            
            def mec_execution_cost(self, cpu_cycles):
                return cpu_cycles / 2000.0
            
            def up_transmission_cost(self, data_size):
                return data_size / 100.0
            
            def dl_transmission_cost(self, data_size):
                return data_size / 100.0
        
        class MockTask:
            def __init__(self, id_name, processing_data_size, transmission_data_size):
                self.id_name = id_name
                self.processing_data_size = processing_data_size
                self.transmission_data_size = transmission_data_size
        
        class MockTaskGraph:
            def __init__(self):
                self.task_list = [
                    MockTask("1", 100.0, 50.0),
                    MockTask("2", 200.0, 75.0),
                    MockTask("3", 300.0, 100.0)
                ]
                self.dependencies = [[0, 1, 25.0], [1, 2, 30.0]]
        
        # Create collector
        env = MockEnv()
        collector = VisualizationCollector(env, 'test_output', animate_episode=1)
        
        # Test data collection
        samples_data = {
            'actions': [[0, 1, 0]],  # LOCAL, EDGE, LOCAL
            'finish_time': [[2.0, 5.0, 8.0]]  # Finish times
        }
        
        episode_record = collector.collect_episode_data(samples_data, 0, 0)
        
        if episode_record:
            print("✅ Episode data collected successfully")
            print(f"   Episode ID: {episode_record.episode_id}")
            print(f"   Method: {episode_record.method}")
            print(f"   Latency: {episode_record.latency_total}")
            print(f"   Nodes: {len(episode_record.get_nodes())}")
            print(f"   Decisions: {len(episode_record.decisions)}")
        else:
            print("❌ Failed to collect episode data")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing VisualizationCollector: {e}")
        return False

def test_file_creation():
    """Test that the integration files exist."""
    print("\n🧪 Testing file creation...")
    
    required_files = [
        'viz_integration.py',
        'meta_evaluator_with_viz_integration.py',
        'meta_evaluator_enhanced.py',
        'patch_meta_evaluator.py',
        'VISUALIZATION_INTEGRATION.md'
    ]
    
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file} exists")
        else:
            print(f"❌ {file} missing")
            return False
    
    return True

def test_dependencies():
    """Test that required dependencies are available."""
    print("\n🧪 Testing dependencies...")
    
    required_packages = [
        'numpy',
        'pandas',
        'matplotlib',
        'plotly',
        'networkx',
        'click',
        'imageio',
        'scipy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} available")
        except ImportError:
            print(f"❌ {package} missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    return True

def main():
    """Run all tests."""
    print("🚀 Testing Visualization Integration")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_visualization_collector,
        test_file_creation,
        test_dependencies
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Integration is ready to use.")
        print("\n📖 Next steps:")
        print("1. Run: python meta_evaluator_with_viz_integration.py")
        print("2. Or apply patch: python patch_meta_evaluator.py")
        print("3. Check output in 'evaluation_results/' directory")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
