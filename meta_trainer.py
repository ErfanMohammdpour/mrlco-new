import tensorflow as tf
# MIGRATION: Disable eager execution for TF1 compatibility with placeholders and sessions
tf.compat.v1.disable_eager_execution()
import numpy as np
import time
from utils import logger
from utils.gpu import setup_gpu_and_strategy, log_tensor_device, ensure_tensor_conversion, run_device_diagnostics
from utils.distributed_tf1 import DistributedTF1Trainer
from automated_reporting import create_training_report

class Trainer(object):
    def __init__(self,algo,
                env,
                sampler,
                sample_processor,
                policy,
                n_itr,
                greedy_finish_time,
                start_itr=0,
                inner_batch_size = 500,
                save_interval = 100,
                distributed_trainer=None):
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
        self.distributed_trainer = distributed_trainer

    def train(self, sess=None):
        """
        Implement the MRLCO training process for task offloading problem
        """

        start_time = time.time()
        avg_ret = []
        avg_loss = []
        avg_latencies = []
        
        # Additional metrics for comprehensive reporting
        policy_losses_all = []
        value_losses_all = []
        greedy_latencies_all = []
        
        for itr in range(self.start_itr, self.n_itr):
            itr_start_time = time.time()
            logger.log("\n ---------------- Iteration %d ----------------" % itr)
            logger.log("Sampling set of tasks/goals for this meta-batch...")
            
            # Log device placement on first iteration
            if itr == self.start_itr and sess is not None:
                # Get a sample tensor from the model to check device placement
                sample_var = tf.compat.v1.global_variables()[0] if tf.compat.v1.global_variables() else None
                if sample_var is not None:
                    var_device = sess.run(sample_var).dtype  # Just to trigger computation
                    print(f"[Step 0] Model variables are initialized and ready for training")
                    if self.distributed_trainer and self.distributed_trainer.num_replicas > 1:
                        print(f"[Step 0] Using distributed training across {self.distributed_trainer.num_replicas} devices")

            task_ids = self.sampler.update_tasks()
            
            # Handle dynamic variable initialization on each iteration
            if sess is not None:
                try:
                    paths = self.sampler.obtain_samples(log=False, log_prefix='')
                except tf.errors.FailedPreconditionError as e:
                    # Initialize any new variables that were created during model build
                    logger.log("FailedPreconditionError caught - initializing new variables...")
                    uninitialized_vars = sess.run(tf.compat.v1.report_uninitialized_variables())
                    if len(uninitialized_vars) > 0:
                        logger.log(f"Initializing {len(uninitialized_vars)} new variables...")
                        new_vars = [v for v in tf.compat.v1.global_variables() 
                                   if sess.run(tf.compat.v1.is_variable_initialized(v)) == False]
                        if new_vars:
                            sess.run(tf.compat.v1.variables_initializer(new_vars))
                    # Try sampling again
                    paths = self.sampler.obtain_samples(log=False, log_prefix='')
            else:
                paths = self.sampler.obtain_samples(log=False, log_prefix='')

            #print("sampled path length is: ", len(paths[0]))

            greedy_run_time = [self.greedy_finish_time[x] for x in task_ids]
            logger.logkv('Average greedy latency,', np.mean(greedy_run_time))
            greedy_latencies_all.append(np.mean(greedy_run_time))

            """ ----------------- Processing Samples ---------------------"""
            logger.log("Processing samples...")
            samples_data = self.sampler_processor.process_samples(paths, log=False, log_prefix='')

            """ ------------------- Inner Policy Update --------------------"""
            policy_losses, value_losses = self.algo.UpdatePPOTarget(samples_data, batch_size=self.inner_batch_size, sess=sess)

            #print("task losses: ", losses)
            print("average task losses: ", np.mean(policy_losses))
            avg_loss.append(np.mean(policy_losses))
            policy_losses_all.append(np.mean(policy_losses))

            print("average value losses: ", np.mean(value_losses))
            value_losses_all.append(np.mean(value_losses))

            """ ------------------ Resample from updated sub-task policy ------------"""
            print("Evaluate the one-step update for sub-task policy")
            new_paths = self.sampler.obtain_samples(log=True, log_prefix='')
            new_samples_data = self.sampler_processor.process_samples(new_paths, log="all", log_prefix='')

            """ ------------------ Outer Policy Update ---------------------"""
            logger.log("Optimizing policy...")
            self.algo.UpdateMetaPolicy(sess=sess)

            """ ------------------- Logging Stuff --------------------------"""

            ret = np.array([])
            for i in range(len(new_samples_data)):
                ret = np.concatenate((ret, np.sum(new_samples_data[i]['rewards'], axis=-1)), axis=-1)

            avg_reward = np.mean(ret)

            latency = np.array([])
            for i in range(len(new_samples_data)):
                latency = np.concatenate((latency, new_samples_data[i]['finish_time']), axis=-1)

            avg_latency = np.mean(latency)
            avg_latencies.append(avg_latency)


            logger.logkv('Itr', itr)
            logger.logkv('Average reward, ', avg_reward)
            logger.logkv('Average latency,', avg_latency)

            logger.dumpkvs()
            avg_ret.append(avg_reward)

            if itr % self.save_interval == 0:
                self.policy.core_policy.save_variables(save_path="./meta_model_inner_step1/meta_model_"+str(itr)+".ckpt")

        self.policy.core_policy.save_variables(save_path="./meta_model_inner_step1/meta_model_final.ckpt")

        # Generate automated report
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
            print("Training completed successfully but report generation failed.")

        return avg_ret, avg_loss, avg_latencies


if __name__ == "__main__":
    print("Starting meta_trainer.py...")
    from env.mec_offloaing_envs.offloading_env import Resources
    print("Imported Resources")
    from env.mec_offloaing_envs.offloading_env import OffloadingEnvironment
    print("Imported OffloadingEnvironment")
    print("Importing MetaSeq2SeqPolicy...")
    from policies.meta_seq2seq_policy import MetaSeq2SeqPolicy
    print("Imported MetaSeq2SeqPolicy")
    from samplers.seq2seq_meta_sampler import Seq2SeqMetaSampler
    print("Imported Seq2SeqMetaSampler")
    from samplers.seq2seq_meta_sampler_process import Seq2SeqMetaSamplerProcessor
    print("Imported Seq2SeqMetaSamplerProcessor")
    from baselines.vf_baseline import ValueFunctionBaseline
    print("Imported ValueFunctionBaseline")
    from meta_algos.MRLCO_distributed import MRLCODistributed
    print("Imported MRLCODistributed")

    print("Setting TF logging verbosity...")
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    print("Configuring logger...")
    # TODO: MPI logger hangs - temporarily disabled
    # logger.configure(dir="./meta_offloading20_log-inner_step1/", format_strs=['stdout', 'log', 'csv'])
    print("Logger configured (skipped for now)")

    META_BATCH_SIZE = 2
    
    # Set up GPU strategy before creating any models
    # Use TF1 compatibility mode since MRLCO uses TF1-style graph building
    print("\n========== Setting up GPU/CPU strategy ==========\n")
    strategy, device_info = setup_gpu_and_strategy(tf1_compatibility_mode=True)
    
    print("Creating resource cluster...")
    resource_cluster = Resources(mec_process_capable=(10.0 * 1024 * 1024),
                                 mobile_process_capable=(1.0 * 1024 * 1024),
                                 bandwidth_up=7.0, bandwidth_dl=7.0)
    print("Resource cluster created")

    print("Creating OffloadingEnvironment...")
    env = OffloadingEnvironment(resource_cluster=resource_cluster,
                                batch_size=100,
                                graph_number=100,
                                graph_file_paths=[
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_1/random.20.",
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_2/random.20.",
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_3/random.20.",
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_5/random.20.",
                                    #"./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_6/random.20.",
                                    #"./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_7/random.20.",
                                    #"./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_9/random.20.",
                                    #"./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_10/random.20.",
                                    #"./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_11/random.20.",
                                    #"./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_13/random.20.",
                                    #"./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_14/random.20.",
                                    #"./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_15/random.20.",
                                    #"./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_17/random.20.",
                                    #"./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_18/random.20.",
                                    #"./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_19/random.20.",
                                    #"./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_21/random.20.",
                                    #"./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_22/random.20.",
                                    #"./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_23/random.20.",
                                    #"./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_25/random.20.",
                                ],
                                time_major=False)

    action, greedy_finish_time = env.greedy_solution()
    print("avg greedy solution: ", np.mean(greedy_finish_time))
    print()
    finish_time = env.get_all_mec_execute_time()
    print("avg all remote solution: ", np.mean(finish_time))
    print()
    finish_time = env.get_all_locally_execute_time()
    print("avg all local solution: ", np.mean(finish_time))
    print()

    # Build all models within the strategy scope
    with strategy.scope():
        print("Creating ValueFunctionBaseline...")
        baseline = ValueFunctionBaseline()
        print("ValueFunctionBaseline created successfully")

        print("Creating MetaSeq2SeqPolicy...")
        meta_policy = MetaSeq2SeqPolicy(meta_batch_size=META_BATCH_SIZE, obs_dim=17, encoder_units=128, decoder_units=128,
                                        vocab_size=2)
        print("MetaSeq2SeqPolicy created successfully")

        sampler = Seq2SeqMetaSampler(
            env=env,
            policy=meta_policy,
            rollouts_per_meta_task=1,  # This batch_size is confusing
            meta_batch_size=META_BATCH_SIZE,
            max_path_length=20000,
            parallel=False,
        )

        sample_processor = Seq2SeqMetaSamplerProcessor(baseline=baseline,
                                                       discount=0.99,
                                                       gae_lambda=0.95,
                                                       normalize_adv=True,
                                                       positive_adv=False)
        # Note: MRLCO uses TF1-style graph building which conflicts with MirroredStrategy
        # For compatibility, we'll build outside strategy scope
        pass
    
    # Build MRLCO algorithm with distributed support
    algo = MRLCODistributed(policy=meta_policy,
                         meta_sampler=sampler,
                         meta_sampler_process=sample_processor,
                         inner_lr=1e-3,  # Increased from 5e-4 to improve learning
                         outer_lr=1e-3,  # Increased from 5e-4 to improve learning
                         meta_batch_size=META_BATCH_SIZE,
                         num_inner_grad_steps=1,
                         clip_value = 0.3,
                         device_list=device_info['selected_devices'])

    # Create distributed trainer wrapper
    distributed_trainer = DistributedTF1Trainer(strategy, device_info)
    
    trainer = Trainer(algo = algo,
                        env=env,
                        sampler=sampler,
                        sample_processor=sample_processor,
                        policy=meta_policy,
                        n_itr=3,
                        greedy_finish_time= greedy_finish_time,
                        start_itr=0,
                        inner_batch_size=1000,
                        distributed_trainer=distributed_trainer)

    # Restore TF1-style session management for exact compatibility
    # Configure session to use the selected device strategy
    config = tf.compat.v1.ConfigProto()
    config.allow_soft_placement = True
    config.log_device_placement = False  # Set to True if you want verbose device logs
    
    # For GPU memory growth and multi-GPU support
    if device_info['num_gpus'] > 0:
        config.gpu_options.allow_growth = True
        # Allow TF to see all GPUs
        config.gpu_options.visible_device_list = ','.join(str(i) for i in range(device_info['num_gpus']))
    
    with tf.compat.v1.Session(config=config) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        
        # Run device diagnostics
        print("\n========== Running Device Diagnostics ==========\n")
        diagnostic_results = run_device_diagnostics(strategy, device_info, detailed=True)
        
        # Log initial tensor device placement after initialization
        print("\n========== Verifying GPU placement ==========\n")
        # Create a test computation to verify GPU is being used
        with tf.device(device_info['selected_devices'][0]):
            test_computation = tf.constant([[1.0, 2.0], [3.0, 4.0]])
            test_result = sess.run(test_computation)
            print(f"Test computation executed successfully on selected device")
        
        # Pass session to trainer for dynamic variable initialization
        avg_ret, avg_loss, avg_latencies = trainer.train(sess=sess)


