#!/usr/bin/env python3
"""
Weight Manager for DRL Fine-tuning

This module handles saving and loading of policy weights for fine-tuning.
Supports both pre-trained weights and fine-tuned weights.
"""

import os
import joblib
import tensorflow as tf
import numpy as np
from datetime import datetime

class WeightManager:
    """
    Manages saving and loading of policy weights for fine-tuning
    """
    
    def __init__(self, base_dir="./weights"):
        self.base_dir = base_dir
        self.pretrained_dir = os.path.join(base_dir, "pretrained")
        self.finetuned_dir = os.path.join(base_dir, "finetuned")
        
        # Create directories if they don't exist
        os.makedirs(self.pretrained_dir, exist_ok=True)
        os.makedirs(self.finetuned_dir, exist_ok=True)
    
    def save_pretrained_weights(self, policy, iteration, map_count, additional_info=None):
        """
        Save pre-trained weights after training on multiple maps
        
        Args:
            policy: The trained policy
            iteration: Training iteration number
            map_count: Number of maps used for pre-training
            additional_info: Additional information to save
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"pretrained_policy_iter{iteration}_maps{map_count}_{timestamp}.ckpt"
        filepath = os.path.join(self.pretrained_dir, filename)
        
        # Save policy weights
        policy.save_variables(save_path=filepath)
        
        # Save metadata
        metadata = {
            'iteration': iteration,
            'map_count': map_count,
            'timestamp': timestamp,
            'policy_type': 'pretrained',
            'additional_info': additional_info or {}
        }
        
        metadata_path = filepath.replace('.ckpt', '_metadata.pkl')
        joblib.dump(metadata, metadata_path)
        
        print(f"✓ Pre-trained weights saved to: {filepath}")
        print(f"✓ Metadata saved to: {metadata_path}")
        
        return filepath, metadata_path
    
    def save_finetuned_weights(self, policy, map_id, iteration, additional_info=None):
        """
        Save fine-tuned weights after training on specific map
        
        Args:
            policy: The fine-tuned policy
            map_id: ID of the specific map used for fine-tuning
            iteration: Fine-tuning iteration number
            additional_info: Additional information to save
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"finetuned_policy_map{map_id}_iter{iteration}_{timestamp}.ckpt"
        filepath = os.path.join(self.finetuned_dir, filename)
        
        # Save policy weights
        policy.save_variables(save_path=filepath)
        
        # Save metadata
        metadata = {
            'map_id': map_id,
            'iteration': iteration,
            'timestamp': timestamp,
            'policy_type': 'finetuned',
            'additional_info': additional_info or {}
        }
        
        metadata_path = filepath.replace('.ckpt', '_metadata.pkl')
        joblib.dump(metadata, metadata_path)
        
        print(f"✓ Fine-tuned weights saved to: {filepath}")
        print(f"✓ Metadata saved to: {metadata_path}")
        
        return filepath, metadata_path
    
    def load_weights(self, policy, weight_path):
        """
        Load weights into policy
        
        Args:
            policy: The policy to load weights into
            weight_path: Path to the weight file
        """
        try:
            policy.load_variables(load_path=weight_path)
            print(f"✓ Weights loaded successfully from: {weight_path}")
            return True
        except Exception as e:
            print(f"✗ Failed to load weights from {weight_path}: {e}")
            return False
    
    def get_latest_pretrained_weights(self):
        """
        Get the path to the latest pre-trained weights
        
        Returns:
            tuple: (weight_path, metadata_path) or (None, None) if not found
        """
        if not os.path.exists(self.pretrained_dir):
            return None, None
        
        # Find all pre-trained weight files
        weight_files = [f for f in os.listdir(self.pretrained_dir) if f.endswith('.ckpt')]
        
        if not weight_files:
            return None, None
        
        # Sort by modification time (newest first)
        weight_files.sort(key=lambda x: os.path.getmtime(os.path.join(self.pretrained_dir, x)), reverse=True)
        
        latest_weight = os.path.join(self.pretrained_dir, weight_files[0])
        latest_metadata = latest_weight.replace('.ckpt', '_metadata.pkl')
        
        return latest_weight, latest_metadata
    
    def get_finetuned_weights_for_map(self, map_id):
        """
        Get the latest fine-tuned weights for a specific map
        
        Args:
            map_id: ID of the map
            
        Returns:
            tuple: (weight_path, metadata_path) or (None, None) if not found
        """
        if not os.path.exists(self.finetuned_dir):
            return None, None
        
        # Find fine-tuned weight files for this map
        weight_files = [f for f in os.listdir(self.finetuned_dir) 
                       if f.endswith('.ckpt') and f'map{map_id}' in f]
        
        if not weight_files:
            return None, None
        
        # Sort by modification time (newest first)
        weight_files.sort(key=lambda x: os.path.getmtime(os.path.join(self.finetuned_dir, x)), reverse=True)
        
        latest_weight = os.path.join(self.finetuned_dir, weight_files[0])
        latest_metadata = latest_weight.replace('.ckpt', '_metadata.pkl')
        
        return latest_weight, latest_metadata
    
    def list_available_weights(self):
        """
        List all available weight files
        
        Returns:
            dict: Dictionary with 'pretrained' and 'finetuned' lists
        """
        weights = {'pretrained': [], 'finetuned': []}
        
        # List pre-trained weights
        if os.path.exists(self.pretrained_dir):
            pretrained_files = [f for f in os.listdir(self.pretrained_dir) if f.endswith('.ckpt')]
            weights['pretrained'] = [os.path.join(self.pretrained_dir, f) for f in pretrained_files]
        
        # List fine-tuned weights
        if os.path.exists(self.finetuned_dir):
            finetuned_files = [f for f in os.listdir(self.finetuned_dir) if f.endswith('.ckpt')]
            weights['finetuned'] = [os.path.join(self.finetuned_dir, f) for f in finetuned_files]
        
        return weights
    
    def load_metadata(self, metadata_path):
        """
        Load metadata from file
        
        Args:
            metadata_path: Path to metadata file
            
        Returns:
            dict: Metadata dictionary or None if failed
        """
        try:
            metadata = joblib.load(metadata_path)
            return metadata
        except Exception as e:
            print(f"✗ Failed to load metadata from {metadata_path}: {e}")
            return None
    
    def cleanup_old_weights(self, keep_latest=5):
        """
        Clean up old weight files, keeping only the latest N files
        
        Args:
            keep_latest: Number of latest files to keep
        """
        for weight_type in ['pretrained', 'finetuned']:
            weight_dir = getattr(self, f'{weight_type}_dir')
            
            if not os.path.exists(weight_dir):
                continue
            
            # Get all weight files
            weight_files = [f for f in os.listdir(weight_dir) if f.endswith('.ckpt')]
            
            if len(weight_files) <= keep_latest:
                continue
            
            # Sort by modification time (oldest first)
            weight_files.sort(key=lambda x: os.path.getmtime(os.path.join(weight_dir, x)))
            
            # Remove old files
            files_to_remove = weight_files[:-keep_latest]
            for file in files_to_remove:
                file_path = os.path.join(weight_dir, file)
                metadata_path = file_path.replace('.ckpt', '_metadata.pkl')
                
                try:
                    os.remove(file_path)
                    if os.path.exists(metadata_path):
                        os.remove(metadata_path)
                    print(f"✓ Removed old weight file: {file}")
                except Exception as e:
                    print(f"✗ Failed to remove {file}: {e}")

if __name__ == "__main__":
    # Test the weight manager
    wm = WeightManager()
    
    print("Available weights:")
    weights = wm.list_available_weights()
    for weight_type, files in weights.items():
        print(f"\n{weight_type.upper()}:")
        for file in files:
            print(f"  - {file}")
