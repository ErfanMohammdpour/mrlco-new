#!/usr/bin/env python3
"""
Complete Fine-tuning Workflow for DRL

This script provides a complete workflow for:
1. Pre-training on 22 maps
2. Fine-tuning on specific maps
3. Evaluating performance

Usage:
    python finetuning_workflow.py --mode pretrain --maps 22 --iterations 1000
    python finetuning_workflow.py --mode finetune --map_id 1 --steps 20
    python finetuning_workflow.py --mode evaluate --map_id 1 --episodes 10
"""

import tensorflow as tf
import numpy as np
import os
import sys
import argparse
from datetime import datetime

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from weight_manager import WeightManager
from pretrain_on_maps import pretrain_policy
from finetune_on_map import finetune_policy
from evaluate_policy import evaluate_policy

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Complete DRL fine-tuning workflow')
    parser.add_argument('--mode', type=str, required=True, 
                       choices=['pretrain', 'finetune', 'evaluate', 'workflow'],
                       help='Mode to run: pretrain, finetune, evaluate, or workflow')
    
    # Pre-training arguments
    parser.add_argument('--maps', type=int, default=22, help='Number of maps for pre-training')
    parser.add_argument('--iterations', type=int, default=1000, help='Number of pre-training iterations')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size for pre-training')
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='Learning rate for pre-training')
    
    # Fine-tuning arguments
    parser.add_argument('--map_id', type=int, help='ID of the specific map for fine-tuning/evaluation')
    parser.add_argument('--steps', type=int, default=20, help='Number of fine-tuning steps')
    parser.add_argument('--finetune_lr', type=float, default=1e-4, help='Learning rate for fine-tuning')
    parser.add_argument('--finetune_batch_size', type=int, default=50, help='Batch size for fine-tuning')
    
    # Evaluation arguments
    parser.add_argument('--episodes', type=int, default=10, help='Number of episodes for evaluation')
    parser.add_argument('--render', action='store_true', help='Whether to render during evaluation')
    
    # Workflow arguments
    parser.add_argument('--workflow_maps', type=int, nargs='+', default=[1, 2, 3], 
                       help='List of map IDs for workflow mode')
    parser.add_argument('--workflow_steps', type=int, default=20, 
                       help='Number of fine-tuning steps for workflow mode')
    
    args = parser.parse_args()
    
    # Initialize weight manager
    weight_manager = WeightManager()
    
    if args.mode == 'pretrain':
        print("🚀 Starting pre-training mode...")
        
        # Pre-train the policy
        policy, weight_path, metadata_path = pretrain_policy(
            num_maps=args.maps,
            iterations=args.iterations,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            save_interval=100
        )
        
        if policy is not None:
            print("\n✅ Pre-training completed successfully!")
            print(f"Pre-trained weights saved to: {weight_path}")
            return 0
        else:
            print("\n❌ Pre-training failed!")
            return 1
    
    elif args.mode == 'finetune':
        if args.map_id is None:
            print("❌ Map ID is required for fine-tuning mode!")
            return 1
        
        print(f"🔧 Starting fine-tuning mode for map {args.map_id}...")
        
        # Get latest pre-trained weights
        weight_path, metadata_path = weight_manager.get_latest_pretrained_weights()
        if weight_path is None:
            print("❌ No pre-trained weights found! Please run pre-training first.")
            return 1
        
        print(f"Using pre-trained weights: {weight_path}")
        
        # Fine-tune the policy
        policy, finetuned_weight_path, finetuned_metadata_path = finetune_policy(
            map_id=args.map_id,
            weight_path=weight_path,
            steps=args.steps,
            learning_rate=args.finetune_lr,
            batch_size=args.finetune_batch_size
        )
        
        if policy is not None:
            print("\n✅ Fine-tuning completed successfully!")
            print(f"Fine-tuned weights saved to: {finetuned_weight_path}")
            return 0
        else:
            print("\n❌ Fine-tuning failed!")
            return 1
    
    elif args.mode == 'evaluate':
        if args.map_id is None:
            print("❌ Map ID is required for evaluation mode!")
            return 1
        
        print(f"📊 Starting evaluation mode for map {args.map_id}...")
        
        # Get weights (pre-trained or fine-tuned)
        weight_path = None
        
        # First try to find fine-tuned weights for this map
        finetuned_weight_path, finetuned_metadata_path = weight_manager.get_finetuned_weights_for_map(args.map_id)
        if finetuned_weight_path is not None:
            weight_path = finetuned_weight_path
            print(f"Using fine-tuned weights: {weight_path}")
        else:
            # Fall back to pre-trained weights
            pretrained_weight_path, pretrained_metadata_path = weight_manager.get_latest_pretrained_weights()
            if pretrained_weight_path is not None:
                weight_path = pretrained_weight_path
                print(f"Using pre-trained weights: {weight_path}")
            else:
                print("❌ No weights found! Please run pre-training first.")
                return 1
        
        # Evaluate the policy
        results = evaluate_policy(
            weight_path=weight_path,
            map_id=args.map_id,
            episodes=args.episodes,
            render=args.render
        )
        
        if results is not None:
            print("\n✅ Evaluation completed successfully!")
            return 0
        else:
            print("\n❌ Evaluation failed!")
            return 1
    
    elif args.mode == 'workflow':
        print("🔄 Starting complete workflow...")
        print(f"Pre-training on {args.maps} maps, then fine-tuning on maps {args.workflow_maps}")
        
        # Step 1: Pre-training
        print("\n" + "="*60)
        print("STEP 1: PRE-TRAINING")
        print("="*60)
        
        policy, weight_path, metadata_path = pretrain_policy(
            num_maps=args.maps,
            iterations=args.iterations,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            save_interval=100
        )
        
        if policy is None:
            print("❌ Pre-training failed! Stopping workflow.")
            return 1
        
        print("✅ Pre-training completed!")
        
        # Step 2: Fine-tuning on each map
        print("\n" + "="*60)
        print("STEP 2: FINE-TUNING")
        print("="*60)
        
        finetuned_weights = {}
        
        for map_id in args.workflow_maps:
            print(f"\nFine-tuning on map {map_id}...")
            
            policy, finetuned_weight_path, finetuned_metadata_path = finetune_policy(
                map_id=map_id,
                weight_path=weight_path,
                steps=args.workflow_steps,
                learning_rate=args.finetune_lr,
                batch_size=args.finetune_batch_size
            )
            
            if policy is not None:
                finetuned_weights[map_id] = finetuned_weight_path
                print(f"✅ Fine-tuning on map {map_id} completed!")
            else:
                print(f"❌ Fine-tuning on map {map_id} failed!")
                finetuned_weights[map_id] = None
        
        # Step 3: Evaluation
        print("\n" + "="*60)
        print("STEP 3: EVALUATION")
        print("="*60)
        
        evaluation_results = {}
        
        for map_id in args.workflow_maps:
            print(f"\nEvaluating on map {map_id}...")
            
            # Use fine-tuned weights if available, otherwise pre-trained
            eval_weight_path = finetuned_weights[map_id] if finetuned_weights[map_id] else weight_path
            
            results = evaluate_policy(
                weight_path=eval_weight_path,
                map_id=map_id,
                episodes=args.episodes,
                render=False
            )
            
            if results is not None:
                evaluation_results[map_id] = results
                print(f"✅ Evaluation on map {map_id} completed!")
            else:
                print(f"❌ Evaluation on map {map_id} failed!")
                evaluation_results[map_id] = None
        
        # Step 4: Summary
        print("\n" + "="*60)
        print("WORKFLOW SUMMARY")
        print("="*60)
        
        print(f"Pre-training: ✅ Completed on {args.maps} maps")
        print(f"Fine-tuning: ✅ Completed on {len([w for w in finetuned_weights.values() if w is not None])} maps")
        print(f"Evaluation: ✅ Completed on {len([r for r in evaluation_results.values() if r is not None])} maps")
        
        print("\nDetailed results:")
        for map_id in args.workflow_maps:
            if evaluation_results[map_id] is not None:
                summary = evaluation_results[map_id]['summary']
                print(f"  Map {map_id}:")
                print(f"    Average reward: {summary['avg_reward']:.4f}")
                print(f"    Average latency: {summary['avg_latency']:.4f}")
                print(f"    Latency improvement: {summary['latency_improvement']:.2f}%")
            else:
                print(f"  Map {map_id}: ❌ Failed")
        
        print("\n🎉 Complete workflow finished!")
        return 0
    
    else:
        print(f"❌ Unknown mode: {args.mode}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
