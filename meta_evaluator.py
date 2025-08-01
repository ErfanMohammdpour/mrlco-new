import tensorflow as tf
import numpy as np
import time
import argparse
from utils import logger

class Trainer():
    def __init__(self,algo,
                env,
                sampler,
                sample_processor,
                policy,
                n_itr,
                batch_size=500,
                start_itr=0,
                num_inner_grad_steps=3):
        self.algo = algo
        self.env = env
        self.sampler = sampler
        self.sampler_processor = sample_processor
        self.policy = policy
        self.n_itr = n_itr
        self.start_itr = start_itr
        self.num_inner_grad_steps = num_inner_grad_steps
        self.batch_size = batch_size

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
                
        print("Model health check completed.\n")
        
        return not any(has_nan)
    
    def train(self):
        """
        Implement the repilte algorithm for ppo reinforcement learning
        """
        start_time = time.time()
        avg_ret = []
        avg_pg_loss = []
        avg_vf_loss = []

        avg_latencies = []
        
        # Smoke test: run forward pass before starting evaluation
        if self.start_itr == 0:
            logger.log("\nPerforming smoke test for forward pass...")
            try:
                # Get a small batch of data for testing
                test_paths = self.sampler.obtain_samples(log=False, log_prefix='')
                test_samples = self.sampler_processor.process_samples(test_paths, log=False, log_prefix='')
                
                # Run a single forward pass
                test_reward = np.mean(np.sum(test_samples['rewards'], axis=-1))
                logger.log(f"Smoke test passed! Test reward: {test_reward:.4f}")
                
                if np.isnan(test_reward):
                    raise ValueError("NaN detected in smoke test reward!")
                    
            except Exception as e:
                logger.log(f"ERROR in smoke test: {str(e)}")
                raise
            
            logger.log("Smoke test completed successfully.\n")
        
        for itr in range(self.start_itr, self.n_itr):
            itr_start_time = time.time()
            logger.log("\n ---------------- Iteration %d ----------------" % itr)
            logger.log("Sampling set of tasks/goals for this meta-batch...")

            paths = self.sampler.obtain_samples(log=True, log_prefix='')

            """ ----------------- Processing Samples ---------------------"""
            logger.log("Processing samples...")
            samples_data = self.sampler_processor.process_samples(paths, log='all', log_prefix='')

            """ ------------------- Inner Policy Update --------------------"""
            policy_losses, value_losses = self.algo.UpdatePPOTarget(samples_data, batch_size=self.batch_size)

            #print("task losses: ", losses)
            print("average policy losses: ", np.mean(policy_losses))
            avg_pg_loss.append(np.mean(policy_losses))

            print("average value losses: ", np.mean(value_losses))
            avg_vf_loss.append(np.mean(value_losses))

            """ ------------------- Logging Stuff --------------------------"""

            ret = np.sum(samples_data['rewards'], axis=-1)
            avg_reward = np.mean(ret)

            latency = samples_data['finish_time']
            avg_latency = np.mean(latency)

            avg_latencies.append(avg_latency)


            logger.logkv('Itr', itr)
            logger.logkv('Average reward, ', avg_reward)
            logger.logkv('Average latency,', avg_latency)
            logger.dumpkvs()
            avg_ret.append(avg_reward)

        return avg_ret, avg_pg_loss,avg_vf_loss, avg_latencies

if __name__ == "__main__":
    from env.mec_offloaing_envs.offloading_env import Resources
    from env.mec_offloaing_envs.offloading_env import OffloadingEnvironment
    from policies.meta_seq2seq_policy import  Seq2SeqPolicy
    from samplers.seq2seq_sampler import Seq2SeqSampler
    from samplers.seq2seq_sampler_process import Seq2SeSamplerProcessor
    from baselines.vf_baseline import ValueFunctionBaseline
    from meta_algos.ppo_offloading import PPO
    from utils import utils, logger

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='MRLCO Meta Evaluation')
    parser.add_argument('--feature_mode', type=str, choices=['full17', 'core5'], default='full17',
                        help='Feature mode: full17 (17-dim) or core5 (5 scalars + 8-dim embedding)')
    args = parser.parse_args()

    logger.configure(dir="./meta_evaluate_ppo_log/task_offloading", format_strs=['stdout', 'log', 'csv'])

    resource_cluster = Resources(mec_process_capable=(10.0 * 1024 * 1024),
                                 mobile_process_capable=(1.0 * 1024 * 1024),
                                 bandwidth_up=7.0, bandwidth_dl=7.0)

    env = OffloadingEnvironment(resource_cluster=resource_cluster,
                                batch_size=100,
                                graph_number=100,
                                graph_file_paths=[
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_12/random.20."
                                    ],
                                time_major=False)

    print("calculate baseline solution======")

    env.set_task(0)
    action, finish_time = env.greedy_solution()
    target_batch, task_finish_time_batch = env.get_reward_batch_step_by_step(action[env.task_id],
                                          env.task_graphs_batchs[env.task_id],
                                          env.max_running_time_batchs[env.task_id],
                                          env.min_running_time_batchs[env.task_id])
    discounted_reward = []
    for reward_path in target_batch:
        discounted_reward.append(utils.discount_cumsum(reward_path, 1.0)[0])

    print("avg greedy solution: ", np.mean(discounted_reward))
    print("avg greedy solution: ", np.mean(task_finish_time_batch))
    print("avg greedy solution: ", np.mean(finish_time))

    print()
    finish_time = env.get_all_mec_execute_time()
    print("avg all remote solution: ", np.mean(finish_time))
    print()
    finish_time = env.get_all_locally_execute_time()
    print("avg all local solution: ", np.mean(finish_time))

    # Set observation dimension based on feature mode
    obs_dim = 17 if args.feature_mode == 'full17' else 13
    policy = Seq2SeqPolicy(obs_dim=obs_dim,
                           encoder_units=128,
                           decoder_units=128,
                           vocab_size=2,
                           feature_mode=args.feature_mode,
                           name="core_policy")

    sampler = Seq2SeqSampler(env,
                             policy,
                             rollouts_per_meta_task=1,
                             max_path_length=40000,
                             envs_per_task=None,
                             parallel=False)

    baseline = ValueFunctionBaseline()

    sample_processor = Seq2SeSamplerProcessor(baseline=baseline,
                                              discount=0.99,
                                              gae_lambda=0.95,
                                              normalize_adv=True,
                                              positive_adv=False)
    algo = PPO(policy=policy,
               meta_sampler=sampler,
               meta_sampler_process=sample_processor,
               lr=1e-4,
               num_inner_grad_steps=3,
               clip_value=0.2,
               max_grad_norm=None)

    # define the trainer of ppo to evaluate the performance of the trained meta policy for new tasks.
    trainer = Trainer(algo=algo,
                      env=env,
                      sampler=sampler,
                      sample_processor=sample_processor,
                      policy=policy,
                      n_itr=21,
                      start_itr=0,
                      batch_size=500,
                      num_inner_grad_steps=3)

    with tf.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        policy.load_variables(load_path="./meta_model_inner_step1/meta_model_final.ckpt")
        
        # Perform initial model health check
        trainer._check_model_health(sess, step=0)
        
        avg_ret, avg_pg_loss, avg_vf_loss, avg_latencies = trainer.train()


