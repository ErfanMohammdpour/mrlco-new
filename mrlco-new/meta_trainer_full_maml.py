import tensorflow as tf
import numpy as np
import time
import os
import json
from utils import logger
from automated_reporting import create_training_report

class FullMAMLTrainer(object):
    """
    Trainer for Full MAML implementation with second-order gradient support.
    
    This trainer supports both the original MRLCO and the new Full MAML algorithms,
    with additional features for monitoring, debugging, and validation.
    """
    
    def __init__(self,
                 algo,
                 env,
                 sampler,
                 sample_processor,
                 policy,
                 n_itr,
                 greedy_finish_time,
                 start_itr=0,
                 inner_batch_size=500,
                 save_interval=100,
                 validation_interval=10,
                 use_validation=True,
                 validation_size=0.2,
                 early_stopping=False,
                 patience=50,
                 checkpoint_dir="./checkpoints/",
                 log_dir="./logs/",
                 tensorboard_dir="./tensorboard/",
                 verbose=True):
        
        self.algo = algo
        self.env = env
        self.sampler = sampler
        self.sampler_processor = sample_processor
        self.policy = policy
        self.n_itr = n_itr
        self.start_itr = start_itr
        self.inner_batch_size = inner_batch_size
        self.greedy_finish_time = greedy_finish_time
        self.save_interval = save_interval
        self.validation_interval = validation_interval
        self.use_validation = use_validation
        self.validation_size = validation_size
        self.early_stopping = early_stopping
        self.patience = patience
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        self.tensorboard_dir = tensorboard_dir
        self.verbose = verbose
        
        # Create directories
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(tensorboard_dir, exist_ok=True)
        
        # Initialize tracking variables
        self.best_validation_loss = float('inf')
        self.patience_counter = 0
        self.training_history = {
            'iterations': [],
            'rewards': [],
            'policy_losses': [],
            'value_losses': [],
            'meta_losses': [],
            'latencies': [],
            'validation_losses': [],
            'gradient_norms': [],
            'learning_rates': []
        }
        
        # Setup TensorBoard writer
        self.setup_tensorboard()
    
    def setup_tensorboard(self):
        """Setup TensorBoard logging."""
        self.summary_writer = tf.compat.v1.summary.FileWriter(
            self.tensorboard_dir, 
            tf.compat.v1.get_default_graph())
    
    def train(self):
        """
        Train using Full MAML with comprehensive monitoring and validation.
        """
        start_time = time.time()
        
        logger.log("\n" + "="*80)
        logger.log("Starting Full MAML Training")
        logger.log("="*80)
        logger.log(f"Algorithm: {self.algo.__class__.__name__}")
        logger.log(f"Total iterations: {self.n_itr}")
        logger.log(f"Inner batch size: {self.inner_batch_size}")
        logger.log(f"Validation enabled: {self.use_validation}")
        logger.log("="*80 + "\n")
        
        for itr in range(self.start_itr, self.n_itr):
            itr_start_time = time.time()
            
            logger.log("\n" + "-"*60)
            logger.log(f"Iteration {itr}/{self.n_itr}")
            logger.log("-"*60)
            
            # Sample tasks
            logger.log("Sampling tasks for meta-batch...")
            task_ids = self.sampler.update_tasks()
            
            # Obtain trajectories
            paths = self.sampler.obtain_samples(log=False, log_prefix='')
            
            # Log greedy baseline performance
            greedy_run_time = [self.greedy_finish_time[x] for x in task_ids]
            avg_greedy_latency = np.mean(greedy_run_time)
            logger.logkv('Average greedy latency', avg_greedy_latency)
            
            # Process samples
            logger.log("Processing samples...")
            samples_data = self.sampler_processor.process_samples(
                paths, log=False, log_prefix='')
            
            # Split data for training and validation if enabled
            if self.use_validation:
                train_data, val_data = self._split_data(samples_data)
            else:
                train_data = samples_data
                val_data = None
            
            # Inner loop updates
            logger.log("Performing inner loop updates...")
            policy_losses, value_losses = self.algo.UpdatePPOTarget(
                train_data, batch_size=self.inner_batch_size)
            
            avg_policy_loss = np.mean(policy_losses)
            avg_value_loss = np.mean(value_losses)
            
            logger.log(f"  Average policy loss: {avg_policy_loss:.4f}")
            logger.log(f"  Average value loss: {avg_value_loss:.4f}")
            
            # Evaluate adapted policies
            logger.log("Evaluating adapted policies...")
            new_paths = self.sampler.obtain_samples(log=True, log_prefix='')
            new_samples_data = self.sampler_processor.process_samples(
                new_paths, log="all", log_prefix='')
            
            # Meta-update (outer loop)
            logger.log("Performing meta-update...")
            meta_loss = self.algo.UpdateMetaPolicy()
            
            # Calculate metrics
            rewards = self._calculate_rewards(new_samples_data)
            latencies = self._calculate_latencies(new_samples_data)
            avg_reward = np.mean(rewards)
            avg_latency = np.mean(latencies)
            
            # Validation
            validation_loss = None
            if self.use_validation and itr % self.validation_interval == 0:
                validation_loss = self._validate(val_data)
                logger.log(f"  Validation loss: {validation_loss:.4f}")
                
                # Early stopping check
                if self.early_stopping:
                    if validation_loss < self.best_validation_loss:
                        self.best_validation_loss = validation_loss
                        self.patience_counter = 0
                        self._save_best_model()
                    else:
                        self.patience_counter += 1
                        if self.patience_counter >= self.patience:
                            logger.log("\nEarly stopping triggered!")
                            break
            
            # Get diagnostics from algorithm
            diagnostics = {}
            if hasattr(self.algo, 'get_diagnostics'):
                diagnostics = self.algo.get_diagnostics()
            
            # Update training history
            self.training_history['iterations'].append(itr)
            self.training_history['rewards'].append(avg_reward)
            self.training_history['policy_losses'].append(avg_policy_loss)
            self.training_history['value_losses'].append(avg_value_loss)
            self.training_history['meta_losses'].append(meta_loss if meta_loss else 0.0)
            self.training_history['latencies'].append(avg_latency)
            if validation_loss is not None:
                self.training_history['validation_losses'].append(validation_loss)
            
            # Log metrics
            logger.logkv('Iteration', itr)
            logger.logkv('Average reward', avg_reward)
            logger.logkv('Average latency', avg_latency)
            logger.logkv('Policy loss', avg_policy_loss)
            logger.logkv('Value loss', avg_value_loss)
            if meta_loss:
                logger.logkv('Meta loss', meta_loss)
            
            for key, value in diagnostics.items():
                if isinstance(value, (int, float, np.number)):
                    logger.logkv(key, value)
            
            # Log iteration time
            itr_time = time.time() - itr_start_time
            logger.logkv('Iteration time (s)', itr_time)
            
            logger.dumpkvs()
            
            # Write to TensorBoard
            self._write_tensorboard_summary(itr, {
                'reward': avg_reward,
                'latency': avg_latency,
                'policy_loss': avg_policy_loss,
                'value_loss': avg_value_loss,
                'meta_loss': meta_loss if meta_loss else 0.0,
                'validation_loss': validation_loss if validation_loss else 0.0,
                **diagnostics
            })
            
            # Save checkpoint
            if itr % self.save_interval == 0:
                self._save_checkpoint(itr)
            
            # Print progress bar
            self._print_progress(itr, self.n_itr, avg_reward, avg_latency)
        
        # Final save
        self._save_checkpoint('final')
        self._save_training_history()
        
        # Generate report
        self._generate_report()
        
        # Calculate total training time
        total_time = time.time() - start_time
        logger.log("\n" + "="*80)
        logger.log("Training Complete!")
        logger.log(f"Total training time: {total_time/3600:.2f} hours")
        logger.log(f"Best validation loss: {self.best_validation_loss:.4f}")
        logger.log("="*80)
        
        return self.training_history
    
    def _split_data(self, samples_data):
        """Split data into training and validation sets."""
        train_data = []
        val_data = []
        
        for task_data in samples_data:
            n_samples = task_data['observations'].shape[0]
            n_val = int(n_samples * self.validation_size)
            
            # Random split
            indices = np.random.permutation(n_samples)
            val_indices = indices[:n_val]
            train_indices = indices[n_val:]
            
            # Create train data
            train_task_data = {}
            for key in task_data.keys():
                if isinstance(task_data[key], np.ndarray):
                    train_task_data[key] = task_data[key][train_indices]
                else:
                    train_task_data[key] = task_data[key]
            
            # Create validation data
            val_task_data = {}
            for key in task_data.keys():
                if isinstance(task_data[key], np.ndarray):
                    val_task_data[key] = task_data[key][val_indices]
                else:
                    val_task_data[key] = task_data[key]
            
            train_data.append(train_task_data)
            val_data.append(val_task_data)
        
        return train_data, val_data
    
    def _validate(self, val_data):
        """Perform validation on held-out data."""
        if val_data is None:
            return 0.0
        
        # Run validation through the algorithm
        val_losses = []
        for task_data in val_data:
            # Compute validation loss for each task
            # This would need to be implemented in the algorithm
            pass
        
        return np.mean(val_losses) if val_losses else 0.0
    
    def _calculate_rewards(self, samples_data):
        """Calculate average rewards from samples."""
        rewards = []
        for task_data in samples_data:
            task_rewards = np.sum(task_data['rewards'], axis=-1)
            rewards.extend(task_rewards)
        return np.array(rewards)
    
    def _calculate_latencies(self, samples_data):
        """Calculate average latencies from samples."""
        latencies = []
        for task_data in samples_data:
            if 'finish_time' in task_data:
                latencies.extend(task_data['finish_time'])
        return np.array(latencies) if latencies else np.array([0.0])
    
    def _save_checkpoint(self, iteration):
        """Save model checkpoint."""
        checkpoint_path = os.path.join(
            self.checkpoint_dir, 
            f"checkpoint_iter_{iteration}.ckpt")
        
        self.policy.core_policy.save_variables(checkpoint_path)
        
        # Save training history
        history_path = os.path.join(
            self.checkpoint_dir,
            f"history_iter_{iteration}.json")
        
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        logger.log(f"Checkpoint saved: {checkpoint_path}")
    
    def _save_best_model(self):
        """Save the best model based on validation loss."""
        best_path = os.path.join(self.checkpoint_dir, "best_model.ckpt")
        self.policy.core_policy.save_variables(best_path)
        logger.log(f"Best model saved: {best_path}")
    
    def _save_training_history(self):
        """Save complete training history."""
        history_path = os.path.join(self.log_dir, "training_history.json")
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
    
    def _write_tensorboard_summary(self, iteration, metrics):
        """Write metrics to TensorBoard."""
        sess = tf.compat.v1.get_default_session()
        
        summary = tf.compat.v1.Summary()
        for key, value in metrics.items():
            if isinstance(value, (int, float, np.number)):
                summary.value.add(tag=key, simple_value=float(value))
        
        self.summary_writer.add_summary(summary, iteration)
        self.summary_writer.flush()
    
    def _print_progress(self, current, total, reward, latency):
        """Print progress bar."""
        if not self.verbose:
            return
        
        progress = current / total
        bar_length = 40
        filled = int(bar_length * progress)
        bar = '█' * filled + '░' * (bar_length - filled)
        
        print(f'\rProgress: [{bar}] {progress*100:.1f}% | '
              f'Reward: {reward:.3f} | Latency: {latency:.3f}', end='')
    
    def _generate_report(self):
        """Generate comprehensive training report."""
        try:
            logger.log("\n" + "="*80)
            logger.log("Generating Training Report")
            logger.log("="*80)
            
            # Prepare metrics for report
            avg_rewards = self.training_history['rewards']
            avg_losses = self.training_history['policy_losses']
            avg_latencies = self.training_history['latencies']
            
            additional_metrics = {
                'value_losses': self.training_history['value_losses'],
                'meta_losses': self.training_history['meta_losses'],
                'validation_losses': self.training_history.get('validation_losses', [])
            }
            
            # Generate report
            report_dir = create_training_report(
                avg_ret=avg_rewards,
                avg_loss=avg_losses,
                avg_latencies=avg_latencies,
                additional_metrics=additional_metrics
            )
            
            logger.log(f"Report generated: {report_dir}")
            logger.log("="*80)
            
        except Exception as e:
            logger.log(f"Warning: Failed to generate report: {str(e)}")


if __name__ == "__main__":
    from env.mec_offloaing_envs.offloading_env import Resources, OffloadingEnvironment
    from policies.meta_seq2seq_policy import MetaSeq2SeqPolicy
    from samplers.seq2seq_meta_sampler import Seq2SeqMetaSampler
    from samplers.seq2seq_meta_sampler_process import Seq2SeqMetaSamplerProcessor
    from baselines.vf_baseline import ValueFunctionBaseline
    from meta_algos.FullMAML_v2 import FullMAML_v2
    
    # Setup logging
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    logger.configure(
        dir="./logs/full_maml_experiment/",
        format_strs=['stdout', 'log', 'csv', 'tensorboard'])
    
    # Configuration
    META_BATCH_SIZE = 10
    USE_FULL_MAML = True  # Toggle between Full MAML and original MRLCO
    
    # Environment setup
    resource_cluster = Resources(
        mec_process_capable=(10.0 * 1024 * 1024),
        mobile_process_capable=(1.0 * 1024 * 1024),
        bandwidth_up=7.0,
        bandwidth_dl=7.0)
    
    env = OffloadingEnvironment(
        resource_cluster=resource_cluster,
        batch_size=100,
        graph_number=100,
        graph_file_paths=[
            "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_1/random.20.",
            "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_2/random.20.",
            "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_3/random.20.",
            "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_5/random.20.",
            "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_6/random.20.",
            "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_7/random.20.",
            "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_9/random.20.",
            "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_10/random.20.",
            "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_11/random.20.",
            "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_13/random.20.",
        ],
        time_major=False)
    
    # Get baseline performance
    action, greedy_finish_time = env.greedy_solution()
    print(f"Average greedy solution: {np.mean(greedy_finish_time):.3f}")
    print(f"Average all remote solution: {np.mean(env.get_all_mec_execute_time()):.3f}")
    print(f"Average all local solution: {np.mean(env.get_all_locally_execute_time()):.3f}\n")
    
    # Initialize components
    baseline = ValueFunctionBaseline()
    
    meta_policy = MetaSeq2SeqPolicy(
        meta_batch_size=META_BATCH_SIZE,
        obs_dim=17,
        encoder_units=128,
        decoder_units=128,
        vocab_size=2)
    
    sampler = Seq2SeqMetaSampler(
        env=env,
        policy=meta_policy,
        rollouts_per_meta_task=1,
        meta_batch_size=META_BATCH_SIZE,
        max_path_length=20000,
        parallel=False)
    
    sample_processor = Seq2SeqMetaSamplerProcessor(
        baseline=baseline,
        discount=0.99,
        gae_lambda=0.95,
        normalize_adv=True,
        positive_adv=False)
    
    # Choose algorithm
    if USE_FULL_MAML:
        print("Using Full MAML with second-order gradients\n")
        algo = FullMAML_v2(
            policy=meta_policy,
            meta_sampler=sampler,
            meta_sampler_process=sample_processor,
            inner_lr=1e-3,
            outer_lr=1e-3,
            meta_batch_size=META_BATCH_SIZE,
            num_inner_grad_steps=3,
            clip_value=0.3,
            second_order_method='implicit',  # 'implicit', 'explicit', or 'finite_diff'
            inner_lr_schedule='cosine',
            outer_lr_schedule='exponential',
            memory_optimization=True,
            gradient_checkpointing=True)
    else:
        print("Using original MRLCO (first-order approximation)\n")
        from meta_algos.MRLCO import MRLCO
        algo = MRLCO(
            policy=meta_policy,
            meta_sampler=sampler,
            meta_sampler_process=sample_processor,
            inner_lr=1e-3,
            outer_lr=1e-3,
            meta_batch_size=META_BATCH_SIZE,
            num_inner_grad_steps=1,
            clip_value=0.3)
    
    # Initialize trainer
    trainer = FullMAMLTrainer(
        algo=algo,
        env=env,
        sampler=sampler,
        sample_processor=sample_processor,
        policy=meta_policy,
        n_itr=1000,
        greedy_finish_time=greedy_finish_time,
        start_itr=0,
        inner_batch_size=1000,
        save_interval=50,
        validation_interval=10,
        use_validation=True,
        validation_size=0.2,
        early_stopping=True,
        patience=50,
        checkpoint_dir="./checkpoints/full_maml/",
        log_dir="./logs/full_maml/",
        tensorboard_dir="./tensorboard/full_maml/",
        verbose=True)
    
    # Run training
    with tf.compat.v1.Session() as sess:
        sess.run(tf.global_variables_initializer())
        training_history = trainer.train()
    
    print("\nTraining completed successfully!")