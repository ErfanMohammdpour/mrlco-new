"""
Automated Reporting Module for Meta-Trainer Runs
This module creates comprehensive reports after each meta_trainer.run() call.
"""

import os
import json
import csv
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import traceback


class AutomatedReporter:
    """
    Automated reporting system for meta-training runs.
    Creates timestamped folders with data exports, statistics, and plots.
    """
    
    def __init__(self, base_dir="training_reports"):
        """
        Initialize the reporter.
        
        Args:
            base_dir: Base directory for all reports
        """
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
        
    def create_report(self, metrics_data, iteration_data=None):
        """
        Create a comprehensive report for a training run.
        
        Args:
            metrics_data: Dictionary containing metric arrays (e.g., {'rewards': [...], 'losses': [...], ...})
            iteration_data: Optional detailed per-iteration data
            
        Returns:
            report_dir: Path to the created report directory
        """
        try:
            # Step 1: Create timestamped folder
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_dir = os.path.join(self.base_dir, timestamp)
            os.makedirs(report_dir)
            print(f"Created report directory: {report_dir}")
            
            # Step 2: Export data
            self._export_data(metrics_data, iteration_data, report_dir)
            
            # Step 3: Compute and save summary statistics
            self._save_summary_statistics(metrics_data, report_dir)
            
            # Step 4: Generate plots
            self._generate_plots(metrics_data, report_dir)
            
            # Step 5: Validation
            self._validate_report(report_dir)
            
            print(f"Report successfully created in: {report_dir}")
            return report_dir
            
        except Exception as e:
            error_msg = f"Failed to create report: {str(e)}\n{traceback.format_exc()}"
            print(f"ERROR: {error_msg}")
            
            # Try to save error log
            try:
                error_file = os.path.join(report_dir if 'report_dir' in locals() else self.base_dir, 
                                        f"error_{timestamp}.txt")
                with open(error_file, 'w') as f:
                    f.write(error_msg)
            except:
                pass
                
            raise
    
    def _export_data(self, metrics_data, iteration_data, report_dir):
        """Export data as CSV and JSON files."""
        # Prepare data for export
        if iteration_data is None:
            # Create iteration data from metrics
            iteration_data = []
            num_iterations = len(next(iter(metrics_data.values())))
            
            for i in range(num_iterations):
                row = {'iteration': i}
                for metric_name, values in metrics_data.items():
                    if i < len(values):
                        # Convert numpy types to Python native types
                        value = values[i]
                        if hasattr(value, 'item'):  # numpy scalar
                            value = value.item()
                        row[metric_name] = value
                iteration_data.append(row)
        
        # Save as CSV
        csv_path = os.path.join(report_dir, "data.csv")
        if iteration_data:
            keys = iteration_data[0].keys()
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(iteration_data)
            print(f"Saved data.csv with {len(iteration_data)} rows")
        
        # Save as JSON
        json_path = os.path.join(report_dir, "data.json")
        with open(json_path, 'w') as f:
            # Convert numpy types in iteration_data
            converted_iteration_data = []
            for row in iteration_data:
                converted_row = {}
                for k, v in row.items():
                    if hasattr(v, 'item'):  # numpy scalar
                        converted_row[k] = v.item()
                    elif isinstance(v, np.ndarray):
                        converted_row[k] = v.tolist()
                    else:
                        converted_row[k] = v
                converted_iteration_data.append(converted_row)
            
            # Convert metrics_data with proper numpy handling
            converted_metrics = {}
            for k, v in metrics_data.items():
                if hasattr(v, 'item'):  # numpy scalar
                    converted_metrics[k] = v.item()
                elif isinstance(v, np.ndarray):
                    converted_metrics[k] = v.tolist()
                elif isinstance(v, list):
                    # Handle lists that might contain numpy types
                    converted_metrics[k] = [
                        val.item() if hasattr(val, 'item') else val 
                        for val in v
                    ]
                else:
                    converted_metrics[k] = v
            
            json.dump({
                'metrics_data': converted_metrics,
                'iteration_data': converted_iteration_data
            }, f, indent=2)
        print(f"Saved data.json")
    
    def _save_summary_statistics(self, metrics_data, report_dir):
        """Compute and save summary statistics."""
        summary_path = os.path.join(report_dir, "summary.txt")
        
        with open(summary_path, 'w') as f:
            f.write("TRAINING SUMMARY STATISTICS\n")
            f.write("=" * 50 + "\n\n")
            
            for metric_name, values in metrics_data.items():
                if len(values) > 0:
                    values_array = np.array(values)
                    f.write(f"{metric_name.upper()}:\n")
                    f.write(f"  Mean:     {np.mean(values_array):.6f}\n")
                    f.write(f"  Min:      {np.min(values_array):.6f}\n")
                    f.write(f"  Max:      {np.max(values_array):.6f}\n")
                    f.write(f"  Std Dev:  {np.std(values_array):.6f}\n")
                    f.write(f"  Count:    {len(values_array)}\n")
                    f.write("\n")
        
        print(f"Saved summary.txt")
    
    def _generate_plots(self, metrics_data, report_dir):
        """Generate plots for all metrics."""
        # Use a matplotlib style that's available in older versions
        try:
            plt.style.use('seaborn-darkgrid')
        except:
            # Fallback to default if seaborn style not available
            plt.style.use('default')
        
        for metric_name, values in metrics_data.items():
            if len(values) > 0:
                plt.figure(figsize=(10, 6))
                iterations = range(len(values))
                plt.plot(iterations, values, linewidth=2, marker='o', markersize=4, 
                        markevery=max(1, len(values)//20))
                
                plt.title(f'{metric_name.replace("_", " ").title()} Over Iterations', 
                         fontsize=16, fontweight='bold')
                plt.xlabel('Iteration', fontsize=14)
                plt.ylabel(metric_name.replace("_", " ").title(), fontsize=14)
                plt.grid(True, alpha=0.3)
                
                # Add trend line
                if len(values) > 1:
                    z = np.polyfit(iterations, values, 1)
                    p = np.poly1d(z)
                    plt.plot(iterations, p(iterations), "r--", alpha=0.8, 
                            label=f'Trend: {z[0]:.2e}x + {z[1]:.2f}')
                    plt.legend()
                
                # Save plot
                plot_path = os.path.join(report_dir, f"{metric_name}.png")
                plt.tight_layout()
                plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                print(f"Saved {metric_name}.png")
    
    def _validate_report(self, report_dir):
        """Validate that all expected files were created."""
        expected_files = ['data.csv', 'data.json', 'summary.txt']
        
        for filename in expected_files:
            filepath = os.path.join(report_dir, filename)
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Expected file not found: {filename}")
            
            # Check file is not empty
            if os.path.getsize(filepath) == 0:
                raise ValueError(f"File is empty: {filename}")
        
        # Check for at least one plot
        png_files = [f for f in os.listdir(report_dir) if f.endswith('.png')]
        if not png_files:
            raise FileNotFoundError("No plot files (.png) were created")
        
        print(f"Validation passed: All required files present")


def create_training_report(avg_ret, avg_loss, avg_latencies, additional_metrics=None):
    """
    Convenience function to create a report from training metrics.
    
    Args:
        avg_ret: Average returns per iteration
        avg_loss: Average losses per iteration  
        avg_latencies: Average latencies per iteration
        additional_metrics: Optional dictionary of additional metrics
        
    Returns:
        report_dir: Path to created report directory
    """
    reporter = AutomatedReporter()
    
    metrics_data = {
        'average_reward': avg_ret,
        'average_loss': avg_loss,
        'average_latency': avg_latencies
    }
    
    if additional_metrics:
        metrics_data.update(additional_metrics)
    
    return reporter.create_report(metrics_data)