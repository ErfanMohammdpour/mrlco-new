"""
Test script to demonstrate automated reporting functionality
"""

import numpy as np
from automated_reporting import AutomatedReporter, create_training_report


def test_automated_reporting():
    """Test the automated reporting system with simulated training data."""
    
    print("Testing Automated Reporting System")
    print("="*60)
    
    # Simulate training data (20 iterations)
    num_iterations = 20
    
    # Generate synthetic metrics that show typical training patterns
    avg_ret = []
    avg_loss = []
    avg_latencies = []
    policy_losses = []
    value_losses = []
    greedy_latencies = []
    
    for i in range(num_iterations):
        # Rewards increase over time with some noise
        reward = -50 + 2*i + np.random.normal(0, 5)
        avg_ret.append(reward)
        
        # Losses decrease over time
        loss = 10 * np.exp(-0.1 * i) + np.random.normal(0, 0.5)
        avg_loss.append(loss)
        
        # Latencies improve (decrease) over time
        latency = 100 - 2*i + np.random.normal(0, 3)
        avg_latencies.append(latency)
        
        # Policy losses
        policy_loss = 5 * np.exp(-0.15 * i) + np.random.normal(0, 0.2)
        policy_losses.append(policy_loss)
        
        # Value losses
        value_loss = 3 * np.exp(-0.1 * i) + np.random.normal(0, 0.1)
        value_losses.append(value_loss)
        
        # Greedy latencies (baseline)
        greedy_latency = 80 + np.random.normal(0, 2)
        greedy_latencies.append(greedy_latency)
    
    # Convert to numpy arrays
    avg_ret = np.array(avg_ret)
    avg_loss = np.array(avg_loss)
    avg_latencies = np.array(avg_latencies)
    
    # Test 1: Basic reporting
    print("\nTest 1: Basic Reporting")
    print("-"*40)
    try:
        report_dir = create_training_report(
            avg_ret=avg_ret,
            avg_loss=avg_loss,
            avg_latencies=avg_latencies
        )
        print(f"✓ Basic report created successfully at: {report_dir}")
    except Exception as e:
        print(f"✗ Basic report failed: {e}")
    
    # Test 2: Reporting with additional metrics
    print("\nTest 2: Reporting with Additional Metrics")
    print("-"*40)
    try:
        additional_metrics = {
            'policy_losses': np.array(policy_losses),
            'value_losses': np.array(value_losses),
            'greedy_latencies': np.array(greedy_latencies)
        }
        
        report_dir = create_training_report(
            avg_ret=avg_ret,
            avg_loss=avg_loss,
            avg_latencies=avg_latencies,
            additional_metrics=additional_metrics
        )
        print(f"✓ Extended report created successfully at: {report_dir}")
    except Exception as e:
        print(f"✗ Extended report failed: {e}")
    
    # Test 3: Direct reporter usage with custom iteration data
    print("\nTest 3: Custom Iteration Data")
    print("-"*40)
    try:
        reporter = AutomatedReporter(base_dir="custom_reports")
        
        # Create custom iteration data
        iteration_data = []
        for i in range(num_iterations):
            iteration_data.append({
                'iteration': i,
                'reward': avg_ret[i],
                'loss': avg_loss[i],
                'latency': avg_latencies[i],
                'policy_loss': policy_losses[i],
                'value_loss': value_losses[i],
                'greedy_latency': greedy_latencies[i],
                'improvement_over_greedy': greedy_latencies[i] - avg_latencies[i]
            })
        
        metrics_data = {
            'reward': avg_ret,
            'loss': avg_loss,
            'latency': avg_latencies,
            'improvement_over_greedy': np.array(greedy_latencies) - avg_latencies
        }
        
        report_dir = reporter.create_report(metrics_data, iteration_data)
        print(f"✓ Custom report created successfully at: {report_dir}")
    except Exception as e:
        print(f"✗ Custom report failed: {e}")
    
    # Test 4: Error handling - empty data
    print("\nTest 4: Error Handling - Empty Data")
    print("-"*40)
    try:
        report_dir = create_training_report(
            avg_ret=np.array([]),
            avg_loss=np.array([]),
            avg_latencies=np.array([])
        )
        print(f"✓ Empty data handled gracefully")
    except Exception as e:
        print(f"✓ Empty data correctly raised error: {type(e).__name__}")
    
    print("\n" + "="*60)
    print("Automated Reporting Test Complete!")
    print("Check the generated folders for reports with:")
    print("  - data.csv: Raw iteration data")
    print("  - data.json: Complete metrics and iteration data")
    print("  - summary.txt: Statistical summaries")
    print("  - *.png: Plots for each metric")
    

if __name__ == "__main__":
    test_automated_reporting()