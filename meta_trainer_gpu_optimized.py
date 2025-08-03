import tensorflow as tf
# Enable mixed precision for better GPU utilization
from tensorflow.keras import mixed_precision

import numpy as np
import time
from utils import logger
from utils.gpu import setup_gpu_and_strategy, log_tensor_device, ensure_tensor_conversion, run_device_diagnostics
from utils.distributed_tf1 import DistributedTF1Trainer
from automated_reporting import create_training_report

# Set memory growth before any GPU operations
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Enable mixed precision for better GPU utilization
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
print(f'Compute dtype: {policy.compute_dtype}')
print(f'Variable dtype: {policy.variable_dtype}')

class GPUOptimizedTrainer(object):
    def __init__(self, algo, env, sampler, sample_processor, policy, n_itr, greedy_finish_time,
                 start_itr=0, inner_batch_size=500, save_interval=100, 
                 gradient_accumulation_steps=4):
        self.algo = algo
        self.env = env
        self.sampler = sampler
        self.sampler_processor = sample_processor
        self.policy = policy
        self.n_itr = n_itr
        self.start_itr = start_itr
        self.inner_batch_size = inner_batch_size * gradient_accumulation_steps  # Increase effective batch size
        self.greedy_finish_time = greedy_finish_time
        self.save_interval = save_interval
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        # Set up distribution strategy
        self.strategy = self._setup_strategy()
        
        # Pre-compile tf.functions for faster execution
        self._compile_functions()
    
    def _setup_strategy(self):
        """Set up distribution strategy based on available GPUs"""
        gpus = tf.config.experimental.list_physical_devices('GPU')
        
        if len(gpus) >= 2:
            strategy = tf.distribute.MirroredStrategy()
            print(f"Using MirroredStrategy with {len(gpus)} GPUs")
        elif len(gpus) == 1:
            strategy = tf.distribute.OneDeviceStrategy("/GPU:0")
            print(f"Using OneDeviceStrategy with GPU:0")
        else:
            strategy = tf.distribute.OneDeviceStrategy("/CPU:0")
            print(f"WARNING: No GPUs found, falling back to CPU")
        
        return strategy
    
    def _compile_functions(self):
        """Pre-compile tf.functions with proper input signatures"""
        # This will be done when we create the optimized train step
        pass
    
    def _create_tf_dataset(self, samples_data):
        """Create optimized tf.data pipeline"""
        # Convert samples to tensors
        dataset = tf.data.Dataset.from_tensor_slices({
            'observations': samples_data['observations'],
            'actions': samples_data['actions'],
            'decoder_inputs': samples_data['decoder_inputs'],
            'decoder_full_length': samples_data['decoder_full_length'],
            'logits': samples_data['logits'],
            'values': samples_data['values'],
            'advantages': samples_data['advantages'],
            'returns': samples_data['returns']
        })
        
        # Apply optimizations
        dataset = dataset.cache()  # Cache in memory
        dataset = dataset.shuffle(buffer_size=1000)
        dataset = dataset.batch(self.inner_batch_size, drop_remainder=True)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    @tf.function
    def distributed_train_step(self, batch, task_id):
        """Distributed training step executed via strategy.run"""
        def step_fn(inputs):
            # Ensure all operations stay on GPU
            observations = tf.cast(inputs['observations'], tf.float32)
            actions = inputs['actions']
            decoder_inputs = inputs['decoder_inputs']
            decoder_full_length = inputs['decoder_full_length']
            old_logits = tf.cast(inputs['logits'], tf.float32)
            old_v = tf.cast(inputs['values'], tf.float32)
            advs = tf.cast(inputs['advantages'], tf.float32)
            r = tf.cast(inputs['returns'], tf.float32)
            
            with tf.GradientTape() as tape:
                # Forward pass - ensure model is on GPU
                new_logits = self.policy.meta_policies[task_id].network.decoder_logits
                vpred = self.policy.meta_policies[task_id].vf
                
                # PPO loss computation
                likelihood_ratio = self.policy.distribution.likelihood_ratio_sym(
                    actions, old_logits, new_logits
                )
                
                clipped_obj = tf.minimum(
                    likelihood_ratio * advs,
                    tf.clip_by_value(likelihood_ratio, 0.8, 1.2) * advs
                )
                surr_obj = -tf.reduce_mean(clipped_obj)
                
                # Value function loss
                vpredclipped = old_v + tf.clip_by_value(vpred - old_v, -0.2, 0.2)
                vf_losses1 = tf.square(vpred - r)
                vf_losses2 = tf.square(vpredclipped - r)
                vf_loss = 0.5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))
                
                total_loss = surr_obj + 0.5 * vf_loss
                
                # Scale loss for mixed precision
                scaled_loss = self.policy.meta_policies[task_id].optimizer.get_scaled_loss(total_loss)
            
            # Compute gradients
            scaled_gradients = tape.gradient(scaled_loss, 
                                            self.policy.meta_policies[task_id].network.get_trainable_variables())
            gradients = self.policy.meta_policies[task_id].optimizer.get_unscaled_gradients(scaled_gradients)
            
            # Clip gradients
            gradients, _ = tf.clip_by_global_norm(gradients, 0.5)
            
            # Apply gradients
            self.policy.meta_policies[task_id].optimizer.apply_gradients(
                zip(gradients, self.policy.meta_policies[task_id].network.get_trainable_variables())
            )
            
            return surr_obj, vf_loss
        
        # Execute on strategy
        per_replica_losses = self.strategy.run(step_fn, args=(batch,))
        
        # Reduce across replicas
        surr_loss = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses[0], axis=None)
        vf_loss = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses[1], axis=None)
        
        return surr_loss, vf_loss
    
    def warmup_gpu(self):
        """Perform GPU warmup with sample operations"""
        print("\n========== GPU Warmup ==========")
        with self.strategy.scope():
            # Create sample tensors on GPU
            warmup_size = 1000
            a = tf.random.normal([warmup_size, warmup_size], dtype=tf.float32)
            b = tf.random.normal([warmup_size, warmup_size], dtype=tf.float32)
            
            # Perform matmul operations
            @tf.function
            def warmup_op():
                for _ in range(10):
                    c = tf.matmul(a, b)
                    d = tf.nn.relu(c)
                    e = tf.reduce_sum(d)
                return e
            
            # Execute warmup
            start_time = time.time()
            result = self.strategy.run(warmup_op)
            
            # Log device placement
            if hasattr(result, 'device'):
                print(f"Warmup completed on device: {result.device}")
            else:
                print(f"Warmup completed on strategy: {self.strategy}")
            
            print(f"Warmup time: {time.time() - start_time:.3f} seconds")
        print("================================\n")
    
    def train(self, sess=None):
        """GPU-optimized training loop"""
        # Perform GPU warmup
        self.warmup_gpu()
        
        start_time = time.time()
        avg_ret = []
        avg_loss = []
        avg_latencies = []
        policy_losses_all = []
        value_losses_all = []
        greedy_latencies_all = []
        
        # Create models within strategy scope
        with self.strategy.scope():
            # Ensure optimizers use mixed precision
            for i in range(len(self.policy.meta_policies)):
                if not hasattr(self.policy.meta_policies[i], 'optimizer'):
                    self.policy.meta_policies[i].optimizer = tf.keras.optimizers.Adam(
                        learning_rate=0.1, epsilon=1e-5
                    )
                    self.policy.meta_policies[i].optimizer = mixed_precision.LossScaleOptimizer(
                        self.policy.meta_policies[i].optimizer
                    )
        
        for itr in range(self.start_itr, self.n_itr):
            itr_start_time = time.time()
            logger.log("\n ---------------- Iteration %d ----------------" % itr)
            logger.log("Sampling set of tasks/goals for this meta-batch...")
            
            # Log device placement on first iteration
            if itr == self.start_itr:
                with self.strategy.scope():
                    # Create and log a test tensor
                    test_tensor = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32)
                    log_tensor_device(test_tensor, "Test tensor", step=0)
                    print(f"[Step 0] Strategy: {self.strategy}")
                    print(f"[Step 0] Number of replicas: {self.strategy.num_replicas_in_sync}")
            
            # Sample tasks
            task_ids = self.sampler.update_tasks()
            paths = self.sampler.obtain_samples(log=False, log_prefix='')
            
            # Process greedy latencies
            greedy_run_time = [self.greedy_finish_time[x] for x in task_ids]
            logger.logkv('Average greedy latency,', np.mean(greedy_run_time))
            greedy_latencies_all.append(np.mean(greedy_run_time))
            
            # Process samples
            logger.log("Processing samples...")
            samples_data = self.sampler_processor.process_samples(paths, log=False, log_prefix='')
            
            # Inner policy update with GPU optimization
            logger.log("Running GPU-optimized PPO updates...")
            policy_losses = []
            value_losses = []
            
            with self.strategy.scope():
                for task_idx, task_samples in enumerate(samples_data):
                    # Create tf.data pipeline
                    dataset = self._create_tf_dataset(task_samples)
                    
                    task_policy_losses = []
                    task_value_losses = []
                    
                    # Train on batches
                    for batch in dataset:
                        surr_loss, vf_loss = self.distributed_train_step(batch, task_idx)
                        task_policy_losses.append(surr_loss.numpy())
                        task_value_losses.append(vf_loss.numpy())
                    
                    policy_losses.append(np.mean(task_policy_losses))
                    value_losses.append(np.mean(task_value_losses))
            
            print("average task losses: ", np.mean(policy_losses))
            avg_loss.append(np.mean(policy_losses))
            policy_losses_all.append(np.mean(policy_losses))
            
            print("average value losses: ", np.mean(value_losses))
            value_losses_all.append(np.mean(value_losses))
            
            # Resample and evaluate
            print("Evaluate the one-step update for sub-task policy")
            new_paths = self.sampler.obtain_samples(log=True, log_prefix='')
            new_samples_data = self.sampler_processor.process_samples(new_paths, log="all", log_prefix='')
            
            # Outer policy update
            logger.log("Optimizing meta policy...")
            with self.strategy.scope():
                self.algo.UpdateMetaPolicy(sess=sess)
            
            # Compute metrics
            ret = np.array([])
            for i in range(len(new_samples_data)):
                ret = np.concatenate((ret, np.sum(new_samples_data[i]['rewards'], axis=-1)), axis=-1)
            avg_reward = np.mean(ret)
            
            latency = np.array([])
            for i in range(len(new_samples_data)):
                latency = np.concatenate((latency, new_samples_data[i]['finish_time']), axis=-1)
            avg_latency = np.mean(latency)
            avg_latencies.append(avg_latency)
            
            # Log metrics
            logger.logkv('Itr', itr)
            logger.logkv('Average reward, ', avg_reward)
            logger.logkv('Average latency,', avg_latency)
            logger.logkv('Iteration time', time.time() - itr_start_time)
            logger.dumpkvs()
            avg_ret.append(avg_reward)
            
            # Save model periodically
            if itr % self.save_interval == 0:
                self.policy.core_policy.save_variables(
                    save_path=f"./meta_model_gpu_optimized/meta_model_{itr}.ckpt"
                )
        
        # Save final model
        self.policy.core_policy.save_variables(
            save_path="./meta_model_gpu_optimized/meta_model_final.ckpt"
        )
        
        # Generate report
        try:
            print("\n==================== GENERATING AUTOMATED REPORT ====================")
            additional_metrics = {
                'policy_losses': policy_losses_all,
                'value_losses': value_losses_all,
                'greedy_latencies': greedy_latencies_all
            }
            
            report_dir = create_training_report(
                avg_ret=avg_ret,
                avg_loss=avg_loss,
                avg_latencies=avg_latencies,
                additional_metrics=additional_metrics
            )
            print(f"Report generated successfully at: {report_dir}")
            print("=====================================================================\n")
        except Exception as e:
            print(f"WARNING: Failed to generate automated report: {str(e)}")
        
        return avg_ret, avg_loss, avg_latencies


if __name__ == "__main__":
    print("Starting GPU-optimized meta_trainer...")
    
    # Import required modules
    from env.mec_offloaing_envs.offloading_env import Resources, OffloadingEnvironment
    from policies.meta_seq2seq_policy import MetaSeq2SeqPolicy
    from samplers.seq2seq_meta_sampler import Seq2SeqMetaSampler
    from samplers.seq2seq_meta_sampler_process import Seq2SeqMetaSamplerProcessor
    from baselines.vf_baseline import ValueFunctionBaseline
    from meta_algos.MRLCO_distributed import MRLCODistributed
    
    # Configure logging
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    
    # Training parameters
    META_BATCH_SIZE = 4  # Increased for better GPU utilization
    INNER_BATCH_SIZE = 1000  # Increased batch size
    GRADIENT_ACCUMULATION_STEPS = 4
    
    # Environment setup
    RESOURCES_CONFIG = Resources(
        mec_process_capable=(10.0 * 1024 * 1024,),
        mobile_process_capable=(1.0 * 1024 * 1024,),
        bandwidth_up=(7.0 * 1024 * 1024,),
        bandwidth_dl=(7.0 * 1024 * 1024,)
    )
    
    env = OffloadingEnvironment(
        resource_cluster=RESOURCES_CONFIG,
        batch_size=100,
        graph_number=100,
        graph_file_paths=[
            "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_1/random.20.",
            "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_2/random.20.",
            "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_3/random.20."
        ],
        time_major=False
    )
    
    # Initialize greedy baselines
    greedy_finish_times = {}
    for env_id in range(3):
        greedy_policy = env.greedy_solution(env_id)
        greedy_finish_times[env_id] = (
            np.array(greedy_policy[1]) / RESOURCES_CONFIG.mec_process_capable[0]
        ).sum()
    
    # Create strategy and models within scope
    strategy = tf.distribute.MirroredStrategy() if len(tf.config.list_physical_devices('GPU')) >= 2 else \
               tf.distribute.OneDeviceStrategy("/GPU:0") if len(tf.config.list_physical_devices('GPU')) == 1 else \
               tf.distribute.OneDeviceStrategy("/CPU:0")
    
    with strategy.scope():
        # Create policy
        policy = MetaSeq2SeqPolicy(
            obs_dim=env.graph_embedding_dim,
            max_path_length=env.max_task_num,
            action_dim=3,
            meta_batch_size=META_BATCH_SIZE
        )
        
        # Create baseline
        baseline = ValueFunctionBaseline()
        
        # Create sampler
        sampler = Seq2SeqMetaSampler(
            env=env,
            policy=policy,
            rollouts_per_meta_task=100,
            max_path_length=20,
            envs_per_task=1,
            parallel=True
        )
        
        # Create sample processor
        sample_processor = Seq2SeqMetaSamplerProcessor(
            baseline=baseline,
            discount=0.99,
            gae_lambda=0.95,
            normalize_adv=True,
            positive_adv=False
        )
        
        # Get device list for distributed training
        gpus = tf.config.list_physical_devices('GPU')
        device_list = [f'/GPU:{i}' for i in range(len(gpus))] if gpus else ['/CPU:0']
        
        # Create distributed algorithm
        algo = MRLCODistributed(
            policy=policy,
            meta_batch_size=META_BATCH_SIZE,
            meta_sampler=sampler,
            meta_sampler_process=sample_processor,
            num_inner_grad_steps=10,
            inner_lr=0.1,
            outer_lr=1e-4,
            device_list=device_list
        )
        
        # Create trainer
        trainer = GPUOptimizedTrainer(
            algo=algo,
            env=env,
            sampler=sampler,
            sample_processor=sample_processor,
            policy=policy,
            n_itr=5,  # Short test run
            start_itr=0,
            inner_batch_size=INNER_BATCH_SIZE,
            greedy_finish_time=greedy_finish_times,
            save_interval=10,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS
        )
    
    # Run training
    print("\n========== Starting GPU-Optimized Training ==========")
    start_time = time.time()
    avg_ret, avg_loss, avg_latencies = trainer.train()
    total_time = time.time() - start_time
    
    print(f"\n========== Training Complete ==========")
    print(f"Total training time: {total_time:.2f} seconds")
    print(f"Final average reward: {avg_ret[-1] if avg_ret else 'N/A'}")
    print(f"Final average latency: {avg_latencies[-1] if avg_latencies else 'N/A'}")
    print("=====================================")