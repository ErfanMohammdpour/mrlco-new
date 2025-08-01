import tensorflow as tf
import numpy as np
import time
import argparse
import warnings
from utils import logger
from automated_reporting import create_training_report

# Suppress specific TensorFlow warnings about IndexedSlices
warnings.filterwarnings('ignore', message='Converting sparse IndexedSlices.*')

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
                save_interval = 100):
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

    def _check_model_health(self, sess, step=0):
        """
        Check for NaN values and shape consistency in model parameters.
        """
        print(f"\nPerforming model health check at step {step}...")
        
        # Get all trainable variables
        trainable_vars = tf.trainable_variables()
        
        # Check for NaN values
        nan_checks = []
        for var in trainable_vars:
            nan_checks.append(tf.reduce_any(tf.is_nan(var)))
            
        has_nan = sess.run(nan_checks)
        
        for i, (var, has_nan_val) in enumerate(zip(trainable_vars, has_nan)):
            if has_nan_val:
                print(f"WARNING: NaN detected in variable {var.name}")
                var_val = sess.run(var)
                print(f"Variable shape: {var_val.shape}")
                print(f"Number of NaN values: {np.sum(np.isnan(var_val))}")
                
        # Check shapes match expected dimensions
        if self.policy.feature_mode == 'core5':
            expected_input_dim = 13
        else:
            expected_input_dim = 17
            
        # Verify encoder input dimensions
        for var in trainable_vars:
            if 'encoder_embeddings' in var.name and 'weights' in var.name:
                shape = var.get_shape().as_list()
                if shape[0] != expected_input_dim:
                    print(f"ERROR: Expected encoder input dimension {expected_input_dim}, but got {shape[0]}")
                    
        print("Model health check completed.\n")
        
        return not any(has_nan)
    
    def train(self):
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
        
        # Smoke test: run forward and backward pass before starting training
        if self.start_itr == 0:
            logger.log("\nPerforming smoke test for forward+backward pass...")
            try:
                # Get a small batch of data for testing
                test_task_ids = self.sampler.update_tasks()[:1]  # Use just one task
                test_paths = self.sampler.obtain_samples(log=False, log_prefix='')
                test_samples = self.sampler_processor.process_samples(test_paths[:1], log=False, log_prefix='')
                
                # Run a single update to test forward+backward pass
                test_losses = self.algo.UpdatePPOTarget(test_samples, batch_size=10)
                
                logger.log(f"Smoke test passed! Test policy loss: {np.mean(test_losses[0]):.4f}")
                logger.log(f"Test value loss: {np.mean(test_losses[1]):.4f}")
                
                # Check for NaN in test losses
                if np.isnan(np.mean(test_losses[0])) or np.isnan(np.mean(test_losses[1])):
                    raise ValueError("NaN detected in smoke test losses!")
                    
            except Exception as e:
                logger.log(f"ERROR in smoke test: {str(e)}")
                raise
            
            logger.log("Smoke test completed successfully.\n")
        
        for itr in range(self.start_itr, self.n_itr):
            itr_start_time = time.time()
            logger.log("\n ---------------- Iteration %d ----------------" % itr)
            logger.log("Sampling set of tasks/goals for this meta-batch...")

            task_ids = self.sampler.update_tasks()
            paths = self.sampler.obtain_samples(log=False, log_prefix='')

            #print("sampled path length is: ", len(paths[0]))

            greedy_run_time = [self.greedy_finish_time[x] for x in task_ids]
            logger.logkv('Average greedy latency,', np.mean(greedy_run_time))
            greedy_latencies_all.append(np.mean(greedy_run_time))

            """ ----------------- Processing Samples ---------------------"""
            logger.log("Processing samples...")
            samples_data = self.sampler_processor.process_samples(paths, log=False, log_prefix='')

            """ ------------------- Inner Policy Update --------------------"""
            policy_losses, value_losses = self.algo.UpdatePPOTarget(samples_data, batch_size=self.inner_batch_size )

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
            self.algo.UpdateMetaPolicy()

            """ ------------------- Logging Stuff --------------------------"""

            ret = np.array([])
            for i in range(5):
                ret = np.concatenate((ret, np.sum(new_samples_data[i]['rewards'], axis=-1)), axis=-1)

            avg_reward = np.mean(ret)

            latency = np.array([])
            for i in range(5):
                latency = np.concatenate((latency, new_samples_data[i]['finish_time']), axis=-1)

            avg_latency = np.mean(latency)
            avg_latencies.append(avg_latency)


            logger.logkv('Itr', itr)
            logger.logkv('Average reward, ', avg_reward)
            logger.logkv('Average latency,', avg_latency)

            logger.dumpkvs()
            avg_ret.append(avg_reward)
            
            # Perform health check every 500 steps
            if itr > 0 and itr % 500 == 0:
                sess = tf.compat.v1.get_default_session()
                self._check_model_health(sess, step=itr)

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
    from env.mec_offloaing_envs.offloading_env import Resources
    from env.mec_offloaing_envs.offloading_env import OffloadingEnvironment
    from policies.meta_seq2seq_policy import MetaSeq2SeqPolicy
    from samplers.seq2seq_meta_sampler import Seq2SeqMetaSampler
    from samplers.seq2seq_meta_sampler_process import Seq2SeqMetaSamplerProcessor
    from baselines.vf_baseline import ValueFunctionBaseline
    from meta_algos.MRLCO import MRLCO

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='MRLCO Meta Training')
    parser.add_argument('--feature_mode', type=str, choices=['full17', 'core5'], default='full17',
                        help='Feature mode: full17 (17-dim) or core5 (5 scalars + 8-dim embedding)')
    args = parser.parse_args()

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    logger.configure(dir="./meta_offloading20_log-inner_step1/", format_strs=['stdout', 'log', 'csv'])

    META_BATCH_SIZE = 10

    resource_cluster = Resources(mec_process_capable=(10.0 * 1024 * 1024),
                                 mobile_process_capable=(1.0 * 1024 * 1024),
                                 bandwidth_up=7.0, bandwidth_dl=7.0)

    # Configure environment based on feature mode
    use_72dim_features = (args.feature_mode == 'core5')
    
    env = OffloadingEnvironment(resource_cluster=resource_cluster,
                                batch_size=100,
                                graph_number=100,
                                use_72dim_features=use_72dim_features,
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
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_14/random.20.",
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_15/random.20.",
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_17/random.20.",
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_18/random.20.",
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_19/random.20.",
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_21/random.20.",
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_22/random.20.",
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_23/random.20.",
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_25/random.20.",
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

    baseline = ValueFunctionBaseline()

    # Set observation dimension based on feature mode
    obs_dim = 17 if args.feature_mode == 'full17' else 13
    meta_policy = MetaSeq2SeqPolicy(meta_batch_size=META_BATCH_SIZE, obs_dim=obs_dim, 
                                    encoder_units=128, decoder_units=128,
                                    vocab_size=2, feature_mode=args.feature_mode)

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
    algo = MRLCO(policy=meta_policy,
                         meta_sampler=sampler,
                         meta_sampler_process=sample_processor,
                         inner_lr=1e-3,  # Increased from 5e-4 to improve learning
                         outer_lr=1e-3,  # Increased from 5e-4 to improve learning
                         meta_batch_size=META_BATCH_SIZE,
                         num_inner_grad_steps=1,
                         clip_value = 0.3)

    trainer = Trainer(algo = algo,
                        env=env,
                        sampler=sampler,
                        sample_processor=sample_processor,
                        policy=meta_policy,
                        n_itr=1000,
                        greedy_finish_time= greedy_finish_time,
                        start_itr=0,
                        inner_batch_size=1000)

    with tf.compat.v1.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        # Perform initial model health check
        trainer._check_model_health(sess, step=0)
        
        avg_ret, avg_loss, avg_latencies = trainer.train()


