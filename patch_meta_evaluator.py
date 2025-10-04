"""
Patch script to add visualization to existing meta_evaluator.py
"""

def patch_meta_evaluator():
    """Apply visualization patch to meta_evaluator.py"""
    
    # Read the original file
    with open('meta_evaluator.py', 'r') as f:
        content = f.read()
    
    # Add import at the top
    if 'from viz_integration import VisualizationCollector' not in content:
        content = content.replace(
            'from utils import logger',
            'from utils import logger\nfrom viz_integration import VisualizationCollector'
        )
    
    # Modify the Trainer class __init__ method
    if 'enable_viz=True' not in content:
        # Find the __init__ method and add visualization parameters
        init_pattern = 'def __init__(self,algo,\n                env,\n                sampler,\n                sample_processor,\n                policy,\n                n_itr,\n                batch_size=500,\n                start_itr=0,\n                num_inner_grad_steps=3):'
        
        new_init = '''def __init__(self,algo,
                env,
                sampler,
                sample_processor,
                policy,
                n_itr,
                batch_size=500,
                start_itr=0,
                num_inner_grad_steps=3,
                enable_viz=True,
                viz_output_dir='evaluation_results',
                animate_episode=10):'''
        
        content = content.replace(init_pattern, new_init)
        
        # Add visualization setup in __init__
        init_body_pattern = '        self.batch_size = batch_size'
        new_init_body = '''        self.batch_size = batch_size
        
        # Visualization setup
        self.enable_viz = enable_viz
        if self.enable_viz:
            self.viz_collector = VisualizationCollector(env, viz_output_dir, animate_episode)
            print(f"🎨 Visualization enabled - output directory: {viz_output_dir}")
            print(f"🎬 Animation will be generated for episode: {animate_episode}")'''
        
        content = content.replace(init_body_pattern, new_init_body)
    
    # Add data collection in the train method
    if 'Collect Data for Visualization' not in content:
        # Find the logging section and add data collection after it
        logging_section = '            avg_ret.append(avg_reward)'
        
        new_logging_section = '''            avg_ret.append(avg_reward)

            """ ------------------- Collect Data for Visualization --------------------"""
            if self.enable_viz:
                # Collect data for visualization
                batch_size = len(samples_data['finish_time']) if 'finish_time' in samples_data else 0
                
                for task_id in range(batch_size):
                    self.viz_collector.collect_episode_data(samples_data, task_id, itr)
                
                if itr % 5 == 0:  # Log every 5 iterations
                    print(f"📊 Collected data for {self.viz_collector.episode_counter} episodes so far")'''
        
        content = content.replace(logging_section, new_logging_section)
    
    # Add visualization generation at the end of train method
    if 'Generate visualizations at the end' not in content:
        return_section = '        return avg_ret, avg_pg_loss,avg_vf_loss, avg_latencies'
        
        new_return_section = '''        # Generate visualizations at the end
        if self.enable_viz and self.viz_collector.evaluation_data:
            print(f"\\n🎨 Generating visualizations for {len(self.viz_collector.evaluation_data)} episodes...")
            self.viz_collector.generate_visualizations()

        return avg_ret, avg_pg_loss,avg_vf_loss, avg_latencies'''
        
        content = content.replace(return_section, new_return_section)
    
    # Update the trainer instantiation
    if 'enable_viz=True' not in content:
        trainer_pattern = '    trainer = Trainer(algo=algo,\n                      env=env,\n                      sampler=sampler,\n                      sample_processor=sample_processor,\n                      policy=policy,\n                      n_itr=21,\n                      start_itr=0,\n                      batch_size=500,\n                      num_inner_grad_steps=3)'
        
        new_trainer = '''    trainer = Trainer(algo=algo,
                      env=env,
                      sampler=sampler,
                      sample_processor=sample_processor,
                      policy=policy,
                      n_itr=21,
                      start_itr=0,
                      batch_size=500,
                      num_inner_grad_steps=3,
                      enable_viz=True,  # Enable visualization
                      viz_output_dir='evaluation_results',  # Output directory
                      animate_episode=10)  # Animate episode 10 as requested'''
        
        content = content.replace(trainer_pattern, new_trainer)
    
    # Write the patched file
    with open('meta_evaluator_patched.py', 'w') as f:
        f.write(content)
    
    print("✅ Patch applied successfully!")
    print("📁 Patched file saved as 'meta_evaluator_patched.py'")
    print("🎨 Visualization will be enabled with the following settings:")
    print("   - Output directory: evaluation_results")
    print("   - Animation episode: 10")
    print("   - Formats: PNG, SVG, MP4, GIF")


if __name__ == "__main__":
    patch_meta_evaluator()
